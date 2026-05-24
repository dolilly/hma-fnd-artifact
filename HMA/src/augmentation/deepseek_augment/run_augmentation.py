#!/usr/bin/env python
# coding: utf-8
# ------------------------------------------------------------
# HMA - Section 3.3: Fine-grained Multi-style Augmentation
#
# For samples selected by the Stage-2 gate, generate augmented
# text via DeepSeek-R1 along three styles (Rewrite / Expand /
# Disguise) x three strengths (Light / Medium / Strong).
#
# This single script unifies the five per-dataset versions.
# Dataset-specific prompt sets / cleaning rules live in JSON,
# everything else (paths, threads, model) lives in YAML.
#
# SECURITY: the API key is NEVER hardcoded. It is read from the
# DEEPSEEK_API_KEY environment variable (or a local .env file).
#
# Usage:
#   export DEEPSEEK_API_KEY=sk-xxxx
#   python run_augmentation.py --config configs/augmentation/pheme.yaml
# ------------------------------------------------------------

import os
import re
import json
import time
import random
import logging
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed

import yaml
import pandas as pd
from tqdm import tqdm
from openai import OpenAI

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)


# ==========================================
# 配置 / 密钥
# ==========================================
def load_yaml(path):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_dotenv_if_present():
    """若存在 .env 文件，加载其中的 KEY=VALUE 到环境变量（不覆盖已有值）。"""
    env_path = os.path.join(os.getcwd(), ".env")
    if not os.path.exists(env_path):
        return
    with open(env_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


def get_api_key():
    load_dotenv_if_present()
    key = os.environ.get("DEEPSEEK_API_KEY")
    if not key:
        raise RuntimeError(
            "DEEPSEEK_API_KEY not found. Set it via environment variable "
            "or a local .env file (see .env.example). NEVER hardcode keys."
        )
    return key


# ==========================================
# 增强器
# ==========================================
class DeepSeekAugmenter:
    def __init__(self, api_key, base_url, model_name, prompt_set):
        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.model_name = model_name
        # 从 JSON prompt 集中读取所有规则
        self.prompts = prompt_set["prompts"]
        self.system_prompt = prompt_set["system_prompt"]
        self.clean_prefixes = prompt_set["clean_prefixes"]
        self.expand_max_tokens = prompt_set.get("expand_max_tokens", 150)
        self.min_output_len = prompt_set.get("min_output_len", 3)
        # 仅中文 Weibo 有安全熔断关键词
        self.refusal_keywords = prompt_set.get("refusal_keywords", [])

    def _clean_output(self, raw_text):
        """清洗：移除 <think>、Markdown、废话前缀；中文场景额外做安全熔断。"""
        if not raw_text:
            return ""

        # 1. 移除 DeepSeek-R1 的思维链
        text = re.sub(r"<think>.*?</think>", "", raw_text, flags=re.DOTALL)

        # 2. 移除 Markdown 标记
        text = text.replace("**", "").replace("__", "").replace("`", "")

        # 3. 移除常见废话前缀（来自 JSON 配置）
        for pattern in self.clean_prefixes:
            text = re.sub(pattern, "", text, flags=re.IGNORECASE | re.MULTILINE)

        # 4. 安全熔断：若包含 AI 说教词汇，直接丢弃（宁缺毋滥）
        for kw in self.refusal_keywords:
            if kw in text:
                return ""

        # 5. 其他清洗：去非法字符、引号、转单行
        text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f]", "", text)
        text = text.strip().strip('"').strip("'").strip()
        text = re.sub(r"\n+", " ", text)
        return text

    def generate_with_retry(self, original_text, style, strength, retry_limit):
        key = f"{style}_{strength}"
        if key not in self.prompts:
            return None

        prompt_content = self.prompts[key].format(text=original_text)

        # 动态 max_tokens：扩写任务严格限长，防止生成小作文
        max_gen_tokens = self.expand_max_tokens if "expand" in style else 2048

        for attempt in range(retry_limit):
            try:
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=[
                        {"role": "system", "content": self.system_prompt},
                        {"role": "user", "content": prompt_content},
                    ],
                    temperature=0.6,
                    max_tokens=max_gen_tokens,
                )
                raw_content = response.choices[0].message.content
                final_content = self._clean_output(raw_content)

                if final_content and len(final_content) >= self.min_output_len:
                    return final_content
                raise ValueError("Generated text is empty, too short, or filtered.")

            except Exception as e:
                wait_time = (2 ** attempt) + random.random()
                if attempt < retry_limit - 1:
                    time.sleep(wait_time)
                else:
                    logging.warning(
                        f"Failed {key} after {retry_limit} attempts: {str(e)[:50]}..."
                    )
                    return None
        return None


# ==========================================
# 单行处理
# ==========================================
def process_single_row(row, augmenter, level_map, retry_limit):
    results = []
    if pd.isna(row["text"]):
        return results
    original_text = str(row["text"])

    try:
        level_code = int(row["stage2_level_pred"])
    except (ValueError, TypeError):
        return results

    if level_code not in level_map:
        return results

    strength = level_map[level_code]
    label = row["label"]

    styles = ["rewrite", "expand", "disguise"]
    for style in styles:
        # HMA++ 策略：Strong Disguise 生成 2 个样本
        num_generations = 2 if (style == "disguise" and strength == "strong") else 1
        for _ in range(num_generations):
            aug_text = augmenter.generate_with_retry(
                original_text, style, strength, retry_limit
            )
            if aug_text:
                results.append(
                    {
                        "original_id": row["id"],
                        "text": aug_text,
                        "label": label,
                        "aug_type": style,
                        "aug_strength": strength,
                        "is_augmented": 1,
                    }
                )
    return results


# ==========================================
# 主流程
# ==========================================
def main():
    parser = argparse.ArgumentParser("HMA Section 3.3 - DeepSeek Augmentation")
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()

    cfg = load_yaml(args.config)

    api_key = get_api_key()
    base_url = cfg["api"]["base_url"]
    model_name = cfg["api"]["model_name"]
    max_workers = cfg["run"]["max_workers"]
    retry_limit = cfg["run"]["retry_limit"]

    input_file = cfg["io"]["input_file"]
    output_file = cfg["io"]["output_file"]

    # prompt 集路径相对于配置文件目录解析
    cfg_dir = os.path.dirname(os.path.abspath(args.config))
    prompt_path = os.path.join(cfg_dir, cfg["prompts"]["file"])
    prompt_set = load_json(prompt_path)

    # 1. 读取数据（兼容 xlsx / csv）
    input_path = input_file
    if not os.path.exists(input_path):
        csv_variant = input_file.replace(".xlsx", ".csv")
        if os.path.exists(csv_variant):
            input_path = csv_variant
        else:
            print(f"找不到输入文件: {input_file}")
            return

    print(f"正在读取: {input_path} ...")
    try:
        if input_path.endswith((".xlsx", ".xls")):
            df = pd.read_excel(input_path, engine="openpyxl")
        else:
            df = pd.read_csv(input_path)
    except Exception as e:
        print(f"读取失败: {e}")
        return

    required_cols = ["id", "text", "label", "m_hard", "stage2_level_pred"]
    for col in required_cols:
        if col not in df.columns:
            print(f"数据缺少必要列: '{col}'")
            return

    # 2. 筛选难例（通过第一阶段门控且分配了强度）
    target_df = df[(df["m_hard"] == 1) & (df["stage2_level_pred"] != -1)].copy()
    print(f"总样本: {len(df)} | 待增强难例: {len(target_df)}")
    if len(target_df) == 0:
        print("没有符合条件的难例。")
        return

    # 3. 初始化增强器并多线程生成
    augmenter = DeepSeekAugmenter(api_key, base_url, model_name, prompt_set)
    level_map = {0: "light", 1: "medium", 2: "strong"}
    augmented_rows = []

    print(f"开始多线程增强 (Threads={max_workers})...")
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_id = {
            executor.submit(process_single_row, row, augmenter, level_map, retry_limit): row["id"]
            for _, row in target_df.iterrows()
        }
        for future in tqdm(as_completed(future_to_id), total=len(target_df), desc="Generating"):
            try:
                augmented_rows.extend(future.result())
            except Exception as e:
                logging.error(f"Thread Error: {e}")

    # 4. 合并原始数据与增强数据并保存
    if not augmented_rows:
        print("未生成数据。")
        return

    aug_df = pd.DataFrame(augmented_rows)

    df_copy = df.copy()
    df_copy["aug_type"] = "original"
    df_copy["aug_strength"] = "none"
    df_copy["is_augmented"] = 0
    df_copy["original_id"] = df_copy["id"]

    aug_df["id"] = aug_df["original_id"].astype(str) + "_aug_" + aug_df.index.astype(str)
    common_cols = ["id", "original_id", "text", "label", "aug_type", "aug_strength", "is_augmented"]
    for col in common_cols:
        if col not in df_copy.columns:
            df_copy[col] = None

    final_df = pd.concat([df_copy[common_cols], aug_df[common_cols]], ignore_index=True)

    os.makedirs(os.path.dirname(os.path.abspath(output_file)), exist_ok=True)
    print(f"正在保存至: {output_file} ...")
    try:
        final_df.to_excel(output_file, index=False, engine="openpyxl")
        print(f"成功! 增强样本数: {len(aug_df)}")
    except Exception as e:
        print(f"保存 Excel 失败，改存 CSV: {e}")
        final_df.to_csv(output_file.replace(".xlsx", ".csv"), index=False)


if __name__ == "__main__":
    main()
