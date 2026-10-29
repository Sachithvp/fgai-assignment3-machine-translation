"""End-to-end runner for ICT4403 Assignment 3.

Run from the repository root:
    python run_experiment.py --config config.json

The script trains a basic Transformer and mT5-small, evaluates them together with
prompted Llama and Gemma models on one held-out test subset, and writes all tables,
plots, checkpoints, and translation samples used by the report.
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import os
import random
import time
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any, Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sacrebleu
import torch
from datasets import Dataset, load_dataset
from tokenizers import Tokenizer
from tokenizers.decoders import ByteLevel as ByteLevelDecoder
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import ByteLevel
from tokenizers.trainers import BpeTrainer
from torch import Tensor, nn
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset as TorchDataset
from tqdm.auto import tqdm
from transformers import (
    AutoModelForCausalLM,
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    set_seed,
)


SPECIAL_TOKENS = ["<pad>", "<bos>", "<eos>", "<unk>"]


@dataclass
class Config:
    seed: int = 42
    dataset_name: str = "Helsinki-NLP/opus_books"
    dataset_config: str = "en-fr"
    source_lang: str = "en"
    target_lang: str = "fr"
    total_examples: int = 10_000
    train_examples: int = 8_000
    validation_examples: int = 1_000
    test_examples: int = 1_000
    evaluation_examples: int = 200
    max_source_length: int = 96
    max_target_length: int = 96
    basic_vocab_size: int = 8_000
    basic_d_model: int = 256
    basic_heads: int = 8
    basic_encoder_layers: int = 3
    basic_decoder_layers: int = 3
    basic_ffn: int = 512
    basic_dropout: float = 0.1
    basic_epochs: int = 8
    basic_batch_size: int = 64
    basic_learning_rate: float = 3e-4
    mt5_model: str = "google/mt5-small"
    mt5_epochs: int = 1
    mt5_batch_size: int = 4
    mt5_gradient_accumulation: int = 4
    mt5_learning_rate: float = 3e-4
    llama_model: str = "unsloth/Llama-3.2-1B-Instruct"
    gemma_model: str = "unsloth/gemma-3-270m-it"
    decoder_batch_size: int = 8
    few_shot_examples: int = 3
    temperature_sample_size: int = 25
    temperatures: tuple[float, ...] = (0.0, 0.3, 0.7, 1.0)

    @classmethod
    def from_json(cls, path: Path) -> "Config":
        raw = json.loads(path.read_text(encoding="utf-8"))
        valid = {f.name for f in fields(cls)}
        unknown = set(raw) - valid
        if unknown:
            raise ValueError(f"Unknown config keys: {sorted(unknown)}")
        if "temperatures" in raw:
            raw["temperatures"] = tuple(raw["temperatures"])
        cfg = cls(**raw)
        if cfg.train_examples + cfg.validation_examples + cfg.test_examples != cfg.total_examples:
            raise ValueError("train + validation + test examples must equal total_examples")
        if cfg.evaluation_examples > cfg.test_examples:
            raise ValueError("evaluation_examples cannot exceed test_examples")
        return cfg


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    set_seed(seed)


def clean_pair(pair: dict[str, str], cfg: Config) -> tuple[str, str] | None:
    source = " ".join(pair[cfg.source_lang].strip().split())
    target = " ".join(pair[cfg.target_lang].strip().split())
    if not source or not target:
        return None
    if len(source) < 2 or len(target) < 2:
        return None
    # Remove extreme length mismatches and very long examples before tokenization.
    ratio = len(source) / max(len(target), 1)
    if not 0.25 <= ratio <= 4.0 or len(source) > 500 or len(target) > 500:
        return None
    return source, target


def prepare_data(cfg: Config, output_dir: Path) -> dict[str, Dataset]:
    raw = load_dataset(cfg.dataset_name, cfg.dataset_config, split="train")
    rows: list[dict[str, str]] = []
    for item in raw.shuffle(seed=cfg.seed):
        cleaned = clean_pair(item["translation"], cfg)
        if cleaned is not None:
            rows.append({"source": cleaned[0], "target": cleaned[1]})
        if len(rows) == cfg.total_examples:
            break
    if len(rows) < cfg.total_examples:
        raise RuntimeError(f"Only {len(rows)} clean pairs found; requested {cfg.total_examples}")

    train_end = cfg.train_examples
    validation_end = train_end + cfg.validation_examples
    splits = {
        "train": Dataset.from_list(rows[:train_end]),
        "validation": Dataset.from_list(rows[train_end:validation_end]),
        "test": Dataset.from_list(rows[validation_end:]),
    }
    data_dir = output_dir / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    for name, split in splits.items():
        split.to_json(data_dir / f"{name}.jsonl", force_ascii=False)
    stats = {
        name: {
            "examples": len(split),
            "mean_source_chars": round(float(np.mean([len(x) for x in split["source"]])), 2),
            "mean_target_chars": round(float(np.mean([len(x) for x in split["target"]])), 2),
        }
        for name, split in splits.items()
    }
    (output_dir / "dataset_statistics.json").write_text(
        json.dumps(stats, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return splits


def train_basic_tokenizer(train_split: Dataset, cfg: Config, checkpoint_dir: Path) -> Tokenizer:
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    tokenizer_path = checkpoint_dir / "tokenizer.json"
    if tokenizer_path.exists():
        return Tokenizer.from_file(str(tokenizer_path))
    tokenizer = Tokenizer(BPE(unk_token="<unk>"))
    tokenizer.pre_tokenizer = ByteLevel(add_prefix_space=False)
    tokenizer.decoder = ByteLevelDecoder()
    trainer = BpeTrainer(vocab_size=cfg.basic_vocab_size, special_tokens=SPECIAL_TOKENS)

    def text_iterator() -> Iterable[str]:
        for row in train_split:
            yield row["source"]
            yield row["target"]

    tokenizer.train_from_iterator(text_iterator(), trainer=trainer)
    tokenizer.save(str(tokenizer_path))
    return tokenizer


class ParallelTokenDataset(TorchDataset):
    def __init__(self, split: Dataset, tokenizer: Tokenizer, cfg: Config):
        self.rows = split
        self.tokenizer = tokenizer
        self.cfg = cfg
        self.bos = tokenizer.token_to_id("<bos>")
        self.eos = tokenizer.token_to_id("<eos>")

    def __len__(self) -> int:
        return len(self.rows)

    def _encode(self, text: str, max_len: int) -> Tensor:
        ids = self.tokenizer.encode(text).ids[: max_len - 2]
        return torch.tensor([self.bos, *ids, self.eos], dtype=torch.long)

    def __getitem__(self, index: int) -> tuple[Tensor, Tensor]:
        row = self.rows[index]
        return (
            self._encode(row["source"], self.cfg.max_source_length),
            self._encode(row["target"], self.cfg.max_target_length),
        )


def make_collator(pad_id: int):
    def collate(batch: list[tuple[Tensor, Tensor]]) -> tuple[Tensor, Tensor]:
        source, target = zip(*batch)
        return (
            pad_sequence(source, batch_first=True, padding_value=pad_id),
            pad_sequence(target, batch_first=True, padding_value=pad_id),
        )

    return collate


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float, max_len: int = 512):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10_000.0) / d_model))
        pe = torch.zeros(1, max_len, d_model)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)

    def forward(self, x: Tensor) -> Tensor:
        return self.dropout(x + self.pe[:, : x.size(1)])


class BasicTransformer(nn.Module):
    def __init__(self, vocab_size: int, pad_id: int, cfg: Config):
        super().__init__()
        self.d_model = cfg.basic_d_model
        self.pad_id = pad_id
        self.embedding = nn.Embedding(vocab_size, self.d_model, padding_idx=pad_id)
        self.position = PositionalEncoding(self.d_model, cfg.basic_dropout)
        self.transformer = nn.Transformer(
            d_model=self.d_model,
            nhead=cfg.basic_heads,
            num_encoder_layers=cfg.basic_encoder_layers,
            num_decoder_layers=cfg.basic_decoder_layers,
            dim_feedforward=cfg.basic_ffn,
            dropout=cfg.basic_dropout,
            batch_first=True,
            norm_first=True,
        )
        self.output = nn.Linear(self.d_model, vocab_size)

    def forward(self, source: Tensor, target_input: Tensor) -> Tensor:
        source_pad = source.eq(self.pad_id)
        target_pad = target_input.eq(self.pad_id)
        target_mask = torch.triu(
            torch.ones(
                target_input.size(1),
                target_input.size(1),
                dtype=torch.bool,
                device=target_input.device,
            ),
            diagonal=1,
        )
        source_emb = self.position(self.embedding(source) * math.sqrt(self.d_model))
        target_emb = self.position(self.embedding(target_input) * math.sqrt(self.d_model))
        hidden = self.transformer(
            source_emb,
            target_emb,
            tgt_mask=target_mask,
            src_key_padding_mask=source_pad,
            tgt_key_padding_mask=target_pad,
            memory_key_padding_mask=source_pad,
        )
        return self.output(hidden)


def parameter_count(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def train_basic_transformer(
    splits: dict[str, Dataset], cfg: Config, checkpoint_dir: Path, device: torch.device
) -> tuple[BasicTransformer, Tokenizer, dict[str, Any]]:
    tokenizer = train_basic_tokenizer(splits["train"], cfg, checkpoint_dir)
    pad_id = tokenizer.token_to_id("<pad>")
    model = BasicTransformer(tokenizer.get_vocab_size(), pad_id, cfg).to(device)
    checkpoint_path = checkpoint_dir / "model.pt"
    history_path = checkpoint_dir / "history.json"
    if checkpoint_path.exists() and history_path.exists():
        model.load_state_dict(torch.load(checkpoint_path, map_location=device, weights_only=True))
        history = json.loads(history_path.read_text(encoding="utf-8"))
        return model, tokenizer, history

    collate = make_collator(pad_id)
    train_loader = DataLoader(
        ParallelTokenDataset(splits["train"], tokenizer, cfg),
        batch_size=cfg.basic_batch_size,
        shuffle=True,
        collate_fn=collate,
        num_workers=0,
    )
    validation_loader = DataLoader(
        ParallelTokenDataset(splits["validation"], tokenizer, cfg),
        batch_size=cfg.basic_batch_size,
        shuffle=False,
        collate_fn=collate,
        num_workers=0,
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.basic_learning_rate)
    criterion = nn.CrossEntropyLoss(ignore_index=pad_id, label_smoothing=0.1)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    history: dict[str, Any] = {"train_loss": [], "validation_loss": [], "training_seconds": 0.0}
    started = time.perf_counter()
    best_loss = float("inf")
    best_state: dict[str, Tensor] | None = None

    for epoch in range(cfg.basic_epochs):
        model.train()
        running = 0.0
        for source, target in tqdm(train_loader, desc=f"Basic Transformer epoch {epoch + 1}"):
            source, target = source.to(device), target.to(device)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"):
                logits = model(source, target[:, :-1])
                loss = criterion(logits.reshape(-1, logits.size(-1)), target[:, 1:].reshape(-1))
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            running += loss.item()
        history["train_loss"].append(running / len(train_loader))

        model.eval()
        running = 0.0
        with torch.inference_mode():
            for source, target in validation_loader:
                source, target = source.to(device), target.to(device)
                logits = model(source, target[:, :-1])
                running += criterion(
                    logits.reshape(-1, logits.size(-1)), target[:, 1:].reshape(-1)
                ).item()
        validation_loss = running / len(validation_loader)
        history["validation_loss"].append(validation_loss)
        if validation_loss < best_loss:
            best_loss = validation_loss
            best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}

    history["training_seconds"] = time.perf_counter() - started
    assert best_state is not None
    model.load_state_dict(best_state)
    torch.save(best_state, checkpoint_path)
    history_path.write_text(json.dumps(history, indent=2), encoding="utf-8")
    return model, tokenizer, history


@torch.inference_mode()
def basic_translate_batch(
    model: BasicTransformer,
    tokenizer: Tokenizer,
    texts: list[str],
    cfg: Config,
    device: torch.device,
) -> list[str]:
    pad_id = tokenizer.token_to_id("<pad>")
    bos_id = tokenizer.token_to_id("<bos>")
    eos_id = tokenizer.token_to_id("<eos>")
    encoded = []
    for text in texts:
        ids = tokenizer.encode(text).ids[: cfg.max_source_length - 2]
        encoded.append(torch.tensor([bos_id, *ids, eos_id], dtype=torch.long))
    source = pad_sequence(encoded, batch_first=True, padding_value=pad_id).to(device)
    generated = torch.full((len(texts), 1), bos_id, dtype=torch.long, device=device)
    finished = torch.zeros(len(texts), dtype=torch.bool, device=device)
    model.eval()
    for _ in range(cfg.max_target_length - 1):
        logits = model(source, generated)
        next_ids = logits[:, -1].argmax(dim=-1)
        next_ids = torch.where(finished, torch.full_like(next_ids, pad_id), next_ids)
        generated = torch.cat([generated, next_ids[:, None]], dim=1)
        finished |= next_ids.eq(eos_id)
        if finished.all():
            break
    results = []
    for row in generated[:, 1:].tolist():
        trimmed = [token for token in row if token not in {pad_id, bos_id, eos_id}]
        results.append(tokenizer.decode(trimmed).strip())
    return results


def preprocess_mt5(split: Dataset, tokenizer: Any, cfg: Config) -> Dataset:
    prefix = f"translate {cfg.source_lang} to {cfg.target_lang}: "

    def tokenize(batch: dict[str, list[str]]) -> dict[str, Any]:
        model_inputs = tokenizer(
            [prefix + text for text in batch["source"]],
            max_length=cfg.max_source_length,
            truncation=True,
        )
        labels = tokenizer(
            text_target=batch["target"], max_length=cfg.max_target_length, truncation=True
        )
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    return split.map(tokenize, batched=True, remove_columns=split.column_names)


def train_mt5(
    splits: dict[str, Dataset], cfg: Config, checkpoint_dir: Path, device: torch.device
) -> tuple[Any, Any, dict[str, Any]]:
    tokenizer = AutoTokenizer.from_pretrained(cfg.mt5_model)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    final_dir = checkpoint_dir / "final"
    metrics_path = checkpoint_dir / "training_metrics.json"
    if (final_dir / "config.json").exists() and metrics_path.exists():
        model = AutoModelForSeq2SeqLM.from_pretrained(final_dir).to(device)
        return model, tokenizer, json.loads(metrics_path.read_text(encoding="utf-8"))

    model = AutoModelForSeq2SeqLM.from_pretrained(cfg.mt5_model)
    tokenized_train = preprocess_mt5(splits["train"], tokenizer, cfg)
    tokenized_validation = preprocess_mt5(splits["validation"], tokenizer, cfg)
    use_bf16 = device.type == "cuda" and torch.cuda.is_bf16_supported()
    use_fp16 = device.type == "cuda" and not use_bf16
    args = Seq2SeqTrainingArguments(
        output_dir=str(checkpoint_dir / "trainer"),
        num_train_epochs=cfg.mt5_epochs,
        per_device_train_batch_size=cfg.mt5_batch_size,
        per_device_eval_batch_size=cfg.mt5_batch_size * 2,
        gradient_accumulation_steps=cfg.mt5_gradient_accumulation,
        learning_rate=cfg.mt5_learning_rate,
        warmup_ratio=0.05,
        weight_decay=0.01,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=1,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        fp16=use_fp16,
        bf16=use_bf16,
        logging_steps=50,
        report_to="none",
        optim="adamw_torch",
    )
    trainer = Seq2SeqTrainer(
        model=model,
        args=args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_validation,
        data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model),
        processing_class=tokenizer,
    )
    started = time.perf_counter()
    result = trainer.train()
    metrics = dict(result.metrics)
    metrics["training_seconds"] = time.perf_counter() - started
    trainer.save_model(final_dir)
    tokenizer.save_pretrained(final_dir)
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    return trainer.model.to(device), tokenizer, metrics


@torch.inference_mode()
def mt5_translate_batch(
    model: Any, tokenizer: Any, texts: list[str], cfg: Config, device: torch.device
) -> list[str]:
    prefix = f"translate {cfg.source_lang} to {cfg.target_lang}: "
    batch = tokenizer(
        [prefix + text for text in texts],
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=cfg.max_source_length,
    ).to(device)
    outputs = model.generate(
        **batch,
        max_new_tokens=cfg.max_target_length,
        num_beams=4,
        early_stopping=True,
    )
    return tokenizer.batch_decode(outputs, skip_special_tokens=True)


def build_translation_prompt(source: str, few_shot: list[dict[str, str]]) -> str:
    demonstrations = "\n".join(
        f"English: {row['source']}\nFrench: {row['target']}" for row in few_shot
    )
    return (
        "Translate the final English sentence into natural French. Return only the French "
        "translation, with no explanation.\n\n"
        f"{demonstrations}\n\nEnglish: {source}\nFrench:"
    )


@torch.inference_mode()
def causal_translate_batch(
    model: Any,
    tokenizer: Any,
    texts: list[str],
    few_shot: list[dict[str, str]],
    cfg: Config,
    device: torch.device,
    temperature: float = 0.0,
) -> list[str]:
    prompts = [build_translation_prompt(text, few_shot) for text in texts]
    chats = [[{"role": "user", "content": prompt}] for prompt in prompts]
    rendered = [
        tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)
        for chat in chats
    ]
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    inputs = tokenizer(
        rendered,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=1024,
    ).to(device)
    generation_args: dict[str, Any] = {
        "max_new_tokens": cfg.max_target_length,
        "pad_token_id": tokenizer.pad_token_id,
        "eos_token_id": tokenizer.eos_token_id,
        "do_sample": temperature > 0,
    }
    if temperature > 0:
        generation_args.update(temperature=temperature, top_p=0.9)
    outputs = model.generate(**inputs, **generation_args)
    prompt_length = inputs["input_ids"].shape[1]
    return [
        tokenizer.decode(row[prompt_length:], skip_special_tokens=True).strip().split("\n")[0]
        for row in outputs
    ]


def load_causal_model(model_name: str, device: torch.device) -> tuple[Any, Any]:
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    # Gemma 3's logit soft-capping is numerically unstable in fp16 on this GPU;
    # bfloat16 prevents the all-padding output observed in the smoke test.
    if device.type == "cuda":
        dtype = torch.bfloat16 if "gemma" in model_name.lower() else torch.float16
    else:
        dtype = torch.float32
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        dtype=dtype,
        low_cpu_mem_usage=True,
    ).to(device)
    model.eval()
    return model, tokenizer


def timed_predictions(
    fn: Any, texts: list[str], batch_size: int
) -> tuple[list[str], float]:
    predictions: list[str] = []
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    started = time.perf_counter()
    for start in tqdm(range(0, len(texts), batch_size), desc="Translating"):
        predictions.extend(fn(texts[start : start + batch_size]))
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - started
    return predictions, elapsed


def score_predictions(references: list[str], predictions: list[str]) -> dict[str, float]:
    bleu = sacrebleu.corpus_bleu(predictions, [references])
    chrf = sacrebleu.corpus_chrf(predictions, [references])
    return {"bleu": float(bleu.score), "chrf": float(chrf.score)}


def cleanup_device() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def save_plots(metrics: pd.DataFrame, temperature: pd.DataFrame, output_dir: Path) -> None:
    plot_dir = output_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    plt.style.use("seaborn-v0_8-whitegrid")

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    metrics.plot.bar(x="model", y=["bleu", "chrf"], ax=axes[0], color=["#2f6bff", "#ff8c42"])
    axes[0].set_title("Translation quality")
    axes[0].set_ylabel("Score")
    axes[0].tick_params(axis="x", rotation=20)
    metrics.plot.bar(x="model", y="inference_seconds_per_example", ax=axes[1], color="#6f42c1")
    axes[1].set_title("Inference latency")
    axes[1].set_ylabel("Seconds per example")
    axes[1].tick_params(axis="x", rotation=20)
    fig.tight_layout()
    fig.savefig(plot_dir / "model_comparison.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    if not temperature.empty:
        fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
        for model_name, group in temperature.groupby("model"):
            axes[0].plot(group["temperature"], group["bleu"], marker="o", label=model_name)
            axes[1].plot(group["temperature"], group["chrf"], marker="o", label=model_name)
        axes[0].set(title="Temperature vs BLEU", xlabel="Temperature", ylabel="BLEU")
        axes[1].set(title="Temperature vs chrF", xlabel="Temperature", ylabel="chrF")
        axes[0].legend()
        axes[1].legend()
        fig.tight_layout()
        fig.savefig(plot_dir / "temperature_analysis.png", dpi=200, bbox_inches="tight")
        plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("config.json"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"))
    parser.add_argument("--checkpoint-dir", type=Path, default=Path("checkpoints"))
    args = parser.parse_args()
    cfg = Config.from_json(args.config)
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    seed_everything(cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    splits = prepare_data(cfg, output_dir)
    evaluation = splits["test"].select(range(cfg.evaluation_examples))
    sources = list(evaluation["source"])
    references = list(evaluation["target"])
    predictions = pd.DataFrame({"source": sources, "reference": references})
    metrics_rows: list[dict[str, Any]] = []

    basic_dir = args.checkpoint_dir / "basic_transformer"
    basic, basic_tokenizer, basic_history = train_basic_transformer(
        splits, cfg, basic_dir, device
    )
    basic_predictions, basic_time = timed_predictions(
        lambda batch: basic_translate_batch(basic, basic_tokenizer, batch, cfg, device),
        sources,
        cfg.basic_batch_size,
    )
    predictions["basic_transformer"] = basic_predictions
    metrics_rows.append(
        {
            "model": "Basic Transformer",
            **score_predictions(references, basic_predictions),
            "training_seconds": basic_history["training_seconds"],
            "inference_seconds": basic_time,
            "inference_seconds_per_example": basic_time / len(sources),
            "parameters": parameter_count(basic),
            "adaptation": "trained from scratch",
        }
    )
    del basic
    cleanup_device()

    mt5_dir = args.checkpoint_dir / "mt5_small"
    mt5, mt5_tokenizer, mt5_training = train_mt5(splits, cfg, mt5_dir, device)
    mt5_predictions, mt5_time = timed_predictions(
        lambda batch: mt5_translate_batch(mt5, mt5_tokenizer, batch, cfg, device),
        sources,
        cfg.mt5_batch_size * 2,
    )
    predictions["mt5_small"] = mt5_predictions
    metrics_rows.append(
        {
            "model": "mT5-small",
            **score_predictions(references, mt5_predictions),
            "training_seconds": mt5_training["training_seconds"],
            "inference_seconds": mt5_time,
            "inference_seconds_per_example": mt5_time / len(sources),
            "parameters": parameter_count(mt5),
            "adaptation": "full fine-tuning",
        }
    )
    del mt5
    cleanup_device()

    few_shot = [dict(row) for row in splits["train"].select(range(cfg.few_shot_examples))]
    temperature_rows: list[dict[str, Any]] = []
    decoder_specs = [("Llama 3.2 1B", cfg.llama_model), ("Gemma 3 270M", cfg.gemma_model)]
    for display_name, model_name in decoder_specs:
        model, tokenizer = load_causal_model(model_name, device)
        model_predictions, model_time = timed_predictions(
            lambda batch: causal_translate_batch(
                model, tokenizer, batch, few_shot, cfg, device, temperature=0.0
            ),
            sources,
            cfg.decoder_batch_size,
        )
        column = "llama_3_2_1b" if display_name.startswith("Llama") else "gemma_3_270m"
        predictions[column] = model_predictions
        metrics_rows.append(
            {
                "model": display_name,
                **score_predictions(references, model_predictions),
                "training_seconds": 0.0,
                "inference_seconds": model_time,
                "inference_seconds_per_example": model_time / len(sources),
                "parameters": parameter_count(model),
                "adaptation": f"{cfg.few_shot_examples}-shot prompting",
            }
        )

        temp_sources = sources[: cfg.temperature_sample_size]
        temp_references = references[: cfg.temperature_sample_size]
        for temperature in cfg.temperatures:
            temp_predictions, elapsed = timed_predictions(
                lambda batch, temp=temperature: causal_translate_batch(
                    model, tokenizer, batch, few_shot, cfg, device, temperature=temp
                ),
                temp_sources,
                cfg.decoder_batch_size,
            )
            temperature_rows.append(
                {
                    "model": display_name,
                    "temperature": temperature,
                    **score_predictions(temp_references, temp_predictions),
                    "inference_seconds": elapsed,
                    "mean_output_words": float(np.mean([len(x.split()) for x in temp_predictions])),
                }
            )
        del model
        cleanup_device()

    metrics = pd.DataFrame(metrics_rows)
    temperature = pd.DataFrame(temperature_rows)
    predictions.to_csv(output_dir / "translations.csv", index=False, encoding="utf-8-sig")
    metrics.to_csv(output_dir / "model_metrics.csv", index=False)
    temperature.to_csv(output_dir / "temperature_analysis.csv", index=False)
    (output_dir / "experiment_metadata.json").write_text(
        json.dumps(
            {
                "config": cfg.__dict__,
                "device": str(device),
                "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
                "torch_version": torch.__version__,
                "sacrebleu_version": sacrebleu.__version__,
            },
            indent=2,
            default=list,
        ),
        encoding="utf-8",
    )
    save_plots(metrics, temperature, output_dir)
    print("\nModel metrics\n", metrics.to_string(index=False))
    print(f"\nArtifacts written to {output_dir.resolve()}")


if __name__ == "__main__":
    main()

