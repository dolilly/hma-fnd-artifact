#!/usr/bin/env python
# coding: utf-8
# ------------------------------------------------------------
# HMA - Section 3.1.2: LLM-based Semantic Hardness (h^LLM)
#
# Soft Prompt Tuning on top of a frozen LLM (e.g. Llama-3-8B,
# Qwen2.5-7B/1.5B) to estimate sample-level semantic hardness.
#
# This single script unifies the five per-dataset versions
# (Constraint / PHEME / Twitter15 / Twitter16 / Weibo). All
# dataset-specific differences are pushed into a YAML config:
#   - Soft Prompt initialization text (4 dimensions + prefix)
#   - dropout / pos_weight / lr_prompt
#   - padding side (left / right)
#   - pooling (last-token / dynamic-position)
#   - optional LayerNorm in the scoring head
#   - confusion-matrix label names
#
# Usage:
#   python train_llm_hardness.py --config configs/llm_hardness/pheme.yaml
# ------------------------------------------------------------

import os
import argparse
import logging

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import yaml
import matplotlib

# 服务器无显示器模式（必须在 import pyplot 之前设置）
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    get_linear_schedule_with_warmup,
)
from peft import get_peft_model, PromptTuningConfig, TaskType, PromptTuningInit
from sklearn.metrics import accuracy_score, confusion_matrix
from tqdm import tqdm


# ==========================================
# 配置加载
# ==========================================
def load_config(path):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def build_init_text(soft_prompt_cfg):
    """根据配置拼接 Soft Prompt 初始化文本。

    join_with: 维度描述之间的连接符。英文数据集用 " "（空格），
    中文（Weibo）用 ""（无分隔）。prefix 末尾的空格已显式写入配置。
    """
    prefix = soft_prompt_cfg["prefix"]
    dims = soft_prompt_cfg["dimensions"]
    join_with = soft_prompt_cfg.get("join_with", " ")
    return prefix + join_with.join(dims)


# ==========================================
# 数据集类（兼容 real/fake 文本标签与 0/1 数字标签）
# ==========================================
class HardnessDataset(Dataset):
    def __init__(self, filepath, tokenizer, max_len, label_map=None, logger=None):
        if not os.path.exists(filepath):
            if logger:
                logger.warning(f"File not found {filepath}")
            self.df = pd.DataFrame(columns=["text", "label", "id"])
        else:
            self.df = self._read_table(filepath)

        self.tokenizer = tokenizer
        self.max_len = max_len
        self.df["text"] = self.df["text"].astype(str).fillna("")

        # 标签处理：若为字符串（real/fake）则映射，否则直接转 int
        if "label" in self.df.columns:
            if self.df["label"].dtype == object and label_map:
                self.df["label"] = (
                    self.df["label"].map(label_map).fillna(0).astype(int)
                )
            else:
                self.df["label"] = self.df["label"].astype(int)
        else:
            self.df["label"] = 0

    @staticmethod
    def _read_table(path):
        # 同时支持 xlsx（Llama 版本）与 csv（Qwen 版本）
        if str(path).endswith(".csv"):
            return pd.read_csv(path)
        return pd.read_excel(path)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        row = self.df.iloc[index]
        encoding = self.tokenizer(
            row["text"],
            truncation=True,
            padding="max_length",
            max_length=self.max_len,
            return_tensors="pt",
        )
        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "label": torch.tensor(int(row["label"]), dtype=torch.float),
            "id": row["id"] if "id" in row else index,
        }


# ==========================================
# 模型定义（Soft Prompt + 冻结 LLM + 评分头）
# ==========================================
class HMALLMHardnessModule(nn.Module):
    def __init__(self, cfg, init_text, device, model_dtype, logger):
        super().__init__()
        model_name = cfg["model"]["name"]
        self.cfg = cfg
        self.device = device
        self.num_virtual_tokens = cfg["soft_prompt"]["num_virtual_tokens"]
        # 是否使用动态池化（PHEME 的抢救修改）
        self.use_dynamic_pooling = cfg["model"].get("use_dynamic_pooling", False)

        logger.info(f"Loading Tokenizer from: {model_name}...")
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name, trust_remote_code=True
        )
        # padding side：PHEME 用 right，其余用 left
        self.tokenizer.padding_side = cfg["model"].get("padding_side", "left")

        # Llama-3 / Qwen 默认无 pad_token，映射到 eos_token
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
            logger.info("Set pad_token to eos_token")

        logger.info(f"Loading base LLM to {device}...")
        self.base_model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=model_dtype,
            device_map="auto",
            trust_remote_code=True,
            attn_implementation="sdpa",
        )
        self.base_model.gradient_checkpointing_enable()

        for param in self.base_model.parameters():
            param.requires_grad = False
            if param.ndim == 1:
                param.data = param.data.to(torch.float32)
        self.base_model.enable_input_require_grads()

        peft_config = PromptTuningConfig(
            task_type=TaskType.CAUSAL_LM,
            prompt_tuning_init=PromptTuningInit.TEXT,
            prompt_tuning_init_text=init_text,
            num_virtual_tokens=self.num_virtual_tokens,
            tokenizer_name_or_path=model_name,
        )
        self.model = get_peft_model(self.base_model, peft_config)
        self.model.to(model_dtype)

        hidden_size = self.base_model.config.hidden_size
        dropout = cfg["model"]["dropout"]
        use_layernorm = cfg["model"].get("use_layernorm", False)

        # 评分头：PHEME 额外加 LayerNorm 控制方差爆炸
        layers = []
        if use_layernorm:
            layers.append(nn.LayerNorm(hidden_size))
        layers += [
            nn.Linear(hidden_size, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 1),
        ]
        self.score_head = nn.Sequential(*layers).to(device).float()

    def forward(self, input_ids, attention_mask):
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
        )
        hidden_states = outputs.hidden_states[-1]

        if self.use_dynamic_pooling:
            # 动态定位真正的最后一个有效 token（PHEME 右填充场景）
            # 加 NUM_VIRTUAL_TOKENS 补偿软提示词向前插入的长度位移
            batch_size = input_ids.shape[0]
            seq_lengths = (
                attention_mask.sum(dim=1) - 1 + self.num_virtual_tokens
            )
            last_hidden_state = hidden_states[
                torch.arange(batch_size, device=hidden_states.device),
                seq_lengths,
            ]
        else:
            # 默认取序列最后一个 token（左填充场景）
            last_hidden_state = hidden_states[:, -1, :]

        logits = self.score_head(last_hidden_state.to(torch.float32))
        return logits


# ==========================================
# 评估与保存
# ==========================================
def evaluate_and_save(model_module, dataset, result_dir, output_filename,
                      device, logger, desc="Evaluating"):
    if len(dataset) == 0:
        return pd.DataFrame()
    dataloader = DataLoader(dataset, batch_size=16, shuffle=False)
    model_module.eval()
    results = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc=desc):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)

            logits = model_module(input_ids, attention_mask)
            probs = torch.sigmoid(logits)
            prob_values = probs.cpu().numpy().flatten()
            preds = (prob_values > 0.5).astype(int)
            # 难度分数 h^LLM = 1 - 2*|p - 0.5|（越接近 0.5 越难）
            hardness = 1.0 - 2.0 * np.abs(prob_values - 0.5)

            for j in range(len(prob_values)):
                results.append(
                    {
                        "y_prob": prob_values[j],
                        "y_pred": preds[j],
                        "h_LLM": hardness[j],
                    }
                )

    df_res = dataset.df.iloc[: len(results)].copy()
    df_res["y_prob"] = [r["y_prob"] for r in results]
    df_res["y_pred"] = [r["y_pred"] for r in results]
    df_res["h_LLM"] = [r["h_LLM"] for r in results]

    save_path = os.path.join(result_dir, output_filename)
    df_res.to_excel(save_path, index=False)

    acc = accuracy_score(df_res["label"], df_res["y_pred"])
    logger.info(f"Saved {desc} (Acc: {acc:.4f})")
    return df_res


def plot_analysis(train_df, test_df, cfg, result_dir, logger):
    if train_df.empty or test_df.empty:
        return
    logger.info("Generating Analysis Plots...")
    sns.set_style("whitegrid")
    plt.rcParams["axes.unicode_minus"] = False

    dataset_name = cfg["dataset"]["name"]
    label_names = cfg["plot"]["label_names"]

    plt.figure(figsize=(10, 6))
    sns.histplot(train_df["h_LLM"], color="skyblue", label="Train Set",
                 kde=True, alpha=0.5, bins=30)
    sns.histplot(test_df["h_LLM"], color="salmon", label="Test Set",
                 kde=True, alpha=0.5, bins=30)
    plt.title(f"Distribution of Hardness ({dataset_name})", fontsize=14)
    plt.xlabel("Hardness Score", fontsize=12)
    plt.legend()
    plt.savefig(os.path.join(result_dir, "hardness_distribution.png"), dpi=300)
    plt.close()

    cm = confusion_matrix(test_df["label"], test_df["y_pred"])
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False,
                xticklabels=label_names, yticklabels=label_names)
    plt.title(f"Confusion Matrix ({dataset_name} Test)", fontsize=14)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.savefig(os.path.join(result_dir, "confusion_matrix.png"), dpi=300)
    plt.close()


# ==========================================
# 主训练流程
# ==========================================
def train(cfg):
    # ---- 路径 ----
    data_dir = cfg["dataset"]["data_dir"]
    result_dir = cfg["dataset"]["result_dir"]
    os.makedirs(result_dir, exist_ok=True)

    # ---- 日志 ----
    log_file = os.path.join(result_dir, "train_log.txt")
    if os.path.exists(log_file):
        os.remove(log_file)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(message)s",
        handlers=[logging.FileHandler(log_file), logging.StreamHandler()],
    )
    logger = logging.getLogger(__name__)

    # ---- Soft Prompt 初始化文本 ----
    init_text = build_init_text(cfg["soft_prompt"])
    logger.info(f">>> Soft Prompt Initialized: {init_text}")

    # ---- 设备与精度 ----
    if torch.cuda.is_available():
        device = torch.device("cuda")
        model_dtype = torch.bfloat16  # 8B 模型必须用 BF16
        logger.info(">>> Using CUDA")
    else:
        device = torch.device("cpu")
        model_dtype = torch.float32

    # ---- 训练超参数 ----
    tr = cfg["training"]
    max_len = tr["max_len"]
    batch_size = tr["batch_size"]
    grad_accum = tr["grad_accum_steps"]
    epochs = tr["epochs"]
    lr_prompt = tr["lr_prompt"]
    lr_head = tr["lr_head"]
    pos_weight_val = tr["pos_weight"]

    # ---- 模型 ----
    logger.info(f"Initializing LLM Hardness Model ({cfg['dataset']['name']})...")
    model_module = HMALLMHardnessModule(cfg, init_text, device, model_dtype, logger)

    # ---- 数据 ----
    logger.info(f"Loading datasets from {data_dir}...")
    label_map = cfg["dataset"].get("label_map", None)

    def pick(name):
        # 兼容 xlsx 与 csv
        for ext in [".xlsx", ".csv", ".xls"]:
            p = os.path.join(data_dir, f"{name}{ext}")
            if os.path.exists(p):
                return p
        return os.path.join(data_dir, f"{name}.xlsx")

    train_dataset = HardnessDataset(pick("train"), model_module.tokenizer,
                                    max_len, label_map, logger)
    val_dataset = HardnessDataset(pick("val"), model_module.tokenizer,
                                  max_len, label_map, logger)
    test_dataset = HardnessDataset(pick("test"), model_module.tokenizer,
                                   max_len, label_map, logger)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)

    # ---- 优化器与调度器 ----
    optimizer = torch.optim.AdamW(
        [
            {
                "params": [
                    p for n, p in model_module.model.named_parameters()
                    if p.requires_grad
                ],
                "lr": lr_prompt,
            },
            {"params": model_module.score_head.parameters(), "lr": lr_head},
        ],
        eps=1e-4,
    )
    total_steps = len(train_loader) * epochs // grad_accum
    scheduler = get_linear_schedule_with_warmup(
        optimizer, int(total_steps * 0.1), total_steps
    )

    pos_weight = torch.tensor([pos_weight_val]).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    best_val_acc = 0
    save_path = os.path.join(result_dir, "best_model.pth")

    logger.info(f"Start Training on {device}...")
    for epoch in range(epochs):
        model_module.train()
        total_loss = 0
        current_accum_loss = 0
        optimizer.zero_grad()

        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{epochs}")
        for i, batch in enumerate(progress_bar):
            input_ids = batch["input_ids"].to(device)
            mask = batch["attention_mask"].to(device)
            labels = batch["label"].to(device).float().unsqueeze(1)

            logits = model_module(input_ids, mask)
            loss = criterion(logits, labels)
            loss = loss / grad_accum
            loss.backward()
            current_accum_loss += loss.item()

            if (i + 1) % grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(model_module.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

                progress_bar.set_postfix({"loss": current_accum_loss * grad_accum})
                total_loss += current_accum_loss * grad_accum
                current_accum_loss = 0

        # 验证
        model_module.eval()
        val_preds, val_labels = [], []
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch["input_ids"].to(device)
                mask = batch["attention_mask"].to(device)
                labels = batch["label"].to(device).unsqueeze(1)
                logits = model_module(input_ids, mask)
                preds = (torch.sigmoid(logits) > 0.5).long().cpu().numpy()
                val_preds.extend(preds)
                val_labels.extend(labels.cpu().numpy())

        val_acc = accuracy_score(val_labels, val_preds)
        denom = (len(train_loader) / grad_accum) if len(train_loader) > 0 else 1
        avg_loss = total_loss / denom
        logger.info(f"Epoch {epoch + 1}: Loss = {avg_loss:.4f}, Val Acc = {val_acc:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            # 轻量化保存：只存可训练参数（Soft Prompt + 评分头）
            state_dict_to_save = {
                k: v for k, v in model_module.named_parameters() if v.requires_grad
            }
            torch.save(state_dict_to_save, save_path)
            logger.info(f" -> New Best Model Saved (Acc: {best_val_acc:.4f}) [Lightweight]")

    # 评估并生成结果
    logger.info("Training Finished. Generating results...")
    if os.path.exists(save_path):
        saved_dict = torch.load(save_path, map_location=device)
        model_module.load_state_dict(saved_dict, strict=False)
        logger.info("Loaded best lightweight checkpoint.")

    df_train = evaluate_and_save(model_module, train_dataset, result_dir,
                                 "train_result.xlsx", device, logger, "Eval Train")
    evaluate_and_save(model_module, val_dataset, result_dir,
                      "val_result.xlsx", device, logger, "Eval Val")
    df_test = evaluate_and_save(model_module, test_dataset, result_dir,
                                "test_result.xlsx", device, logger, "Eval Test")

    plot_analysis(df_train, df_test, cfg, result_dir, logger)
    logger.info(f"Done! Results saved to {result_dir}")


def main():
    parser = argparse.ArgumentParser("HMA Section 3.1.2 - LLM Hardness")
    parser.add_argument("--config", type=str, required=True,
                        help="Path to dataset YAML config")
    args = parser.parse_args()
    cfg = load_config(args.config)
    train(cfg)


if __name__ == "__main__":
    main()
