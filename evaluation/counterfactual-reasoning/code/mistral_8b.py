import os
import re
import math
import argparse
from typing import Tuple, Dict, Any, List

import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_csv",
        type=str,
        default="dataset/large-scale.csv",
        help="Path to input csv file"
    )
    parser.add_argument(
        "--output_csv",
        type=str,
        default="results/mistral/output_mistral_8b_metrics.csv",
        help="Path to save per-example results"
    )
    parser.add_argument(
        "--summary_csv",
        type=str,
        default="results/mistral/output_mistral_8b_summary.csv",
        help="Path to save summary statistics"
    )
    parser.add_argument(
        "--pairwise_csv",
        type=str,
        default="results/mistral/output_mistral_8b_pairwise.csv",
        help="Path to save pairwise comparison results"
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="mistralai/Mistral-7B-Instruct-v0.3",
        help="HF model name"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="cpu / cuda"
    )
    parser.add_argument(
        "--max_rows",
        type=int,
        default=None,
        help="Only evaluate first N rows for debugging"
    )
    return parser.parse_args()


def load_model(model_name: str, device: str):
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    dtype = torch.float16 if device.startswith("cuda") else torch.float32

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=dtype,
        device_map="auto" if device.startswith("cuda") else None,
    )

    if not device.startswith("cuda"):
        model.to(device)

    model.eval()
    return tokenizer, model


def split_prefix_and_suffix(sentence: str) -> Tuple[str, str]:

    sentence = sentence.strip()

    match = re.match(r"^(.*\s)(\S+)$", sentence)
    if match is None:
        return "", sentence

    prefix = match.group(1)
    suffix = match.group(2)
    return prefix, suffix


def make_context_key(sentence: str, condition: str = "") -> str:

    prefix, _ = split_prefix_and_suffix(sentence)
    return f"{condition} || {prefix.strip().lower()}"


@torch.no_grad()
def score_full_sentence_nll(
    sentence: str,
    tokenizer,
    model,
    device: str,
) -> Dict[str, Any]:

    enc = tokenizer(sentence, return_tensors="pt", add_special_tokens=True)
    input_ids = enc["input_ids"].to(model.device if hasattr(model, "device") else device)
    attention_mask = enc["attention_mask"].to(model.device if hasattr(model, "device") else device)

    outputs = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        labels=input_ids
    )

    avg_nll = outputs.loss.item()


    token_count = max(input_ids.shape[1] - 1, 1)
    total_nll = avg_nll * token_count
    ppl = math.exp(avg_nll) if avg_nll < 50 else float("inf")

    return {
        "sentence_total_nll": total_nll,
        "sentence_avg_nll": avg_nll,
        "sentence_token_count": token_count,
        "sentence_ppl": ppl,
    }


@torch.no_grad()
def score_suffix_conditional_nll(
    prefix: str,
    suffix: str,
    tokenizer,
    model,
    device: str,
) -> Dict[str, Any]:

    if prefix is None:
        prefix = ""
    if suffix is None:
        suffix = ""


    prefix_ids = tokenizer(prefix, add_special_tokens=False)["input_ids"]
    suffix_ids = tokenizer(suffix, add_special_tokens=False)["input_ids"]


    full_ids = prefix_ids + suffix_ids
    if len(full_ids) == 0:
        return {
            "suffix_total_nll": float("nan"),
            "suffix_avg_nll": float("nan"),
            "suffix_token_count": 0,
            "suffix_ppl": float("nan"),
        }


    input_ids = torch.tensor([full_ids], dtype=torch.long).to(
        model.device if hasattr(model, "device") else device
    )
    attention_mask = torch.ones_like(input_ids)

    labels = input_ids.clone()
    if len(prefix_ids) > 0:
        labels[:, :len(prefix_ids)] = -100

    outputs = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        labels=labels
    )

    avg_nll = outputs.loss.item()
    suffix_token_count = max(len(suffix_ids), 1)
    total_nll = avg_nll * suffix_token_count
    ppl = math.exp(avg_nll) if avg_nll < 50 else float("inf")

    return {
        "suffix_total_nll": total_nll,
        "suffix_avg_nll": avg_nll,
        "suffix_token_count": suffix_token_count,
        "suffix_ppl": ppl,
    }


def infer_columns(df: pd.DataFrame) -> Dict[str, str]:

    df.columns = [c.strip() for c in df.columns]

    sentence_col = None
    condition_col = None
    congruent_col = None

    for c in df.columns:
        lc = c.lower()
        if lc == "sentence":
            sentence_col = c
        elif lc == "condition":
            condition_col = c
        elif "congruent" in lc:
            congruent_col = c

    if sentence_col is None:
        raise ValueError(f"Cannot find sentence column. Available columns: {df.columns.tolist()}")

    if condition_col is None:
        print("Warning: no 'condition' column found. Will fill empty condition.")
    if congruent_col is None:
        print("Warning: no congruent column found. Group metrics will be limited.")

    return {
        "sentence_col": sentence_col,
        "condition_col": condition_col,
        "congruent_col": congruent_col,
    }


def summarize_by_label(df: pd.DataFrame, congruent_col: str) -> pd.DataFrame:
    metric_cols = [
        "suffix_total_nll",
        "suffix_avg_nll",
        "suffix_ppl",
        "sentence_total_nll",
        "sentence_avg_nll",
        "sentence_ppl",
    ]
    available = [c for c in metric_cols if c in df.columns]

    if congruent_col is None or congruent_col not in df.columns:
        return pd.DataFrame()

    summary = df.groupby(congruent_col)[available].agg(["mean", "std", "count"])
    summary.columns = ["__".join(col).strip() for col in summary.columns.values]
    summary = summary.reset_index()
    return summary


def compute_pairwise_metrics(df: pd.DataFrame, congruent_col: str) -> pd.DataFrame:

    if congruent_col is None or congruent_col not in df.columns:
        return pd.DataFrame()

    rows = []

    grouped = df.groupby("context_key")
    for context_key, group in grouped:
        labels = set(group[congruent_col].astype(str).tolist())
        if "Y" not in labels or "N" not in labels:
            continue

        y_group = group[group[congruent_col].astype(str) == "Y"]
        n_group = group[group[congruent_col].astype(str) == "N"]

        if len(y_group) == 0 or len(n_group) == 0:
            continue

        y_suffix = y_group["suffix_avg_nll"].mean()
        n_suffix = n_group["suffix_avg_nll"].mean()

        y_sent = y_group["sentence_avg_nll"].mean()
        n_sent = n_group["sentence_avg_nll"].mean()

        rows.append({
            "context_key": context_key,
            "y_suffix_avg_nll": y_suffix,
            "n_suffix_avg_nll": n_suffix,
            "suffix_prefers_Y": int(y_suffix < n_suffix),

            "y_sentence_avg_nll": y_sent,
            "n_sentence_avg_nll": n_sent,
            "sentence_prefers_Y": int(y_sent < n_sent),
        })

    pairwise_df = pd.DataFrame(rows)
    return pairwise_df


def main():
    args = parse_args()

    print(f"Loading model: {args.model_name}")
    tokenizer, model = load_model(args.model_name, args.device)

    print(f"Reading csv: {args.input_csv}")
    df = pd.read_csv(args.input_csv)
    df.columns = [c.strip() for c in df.columns]

    if args.max_rows is not None:
        df = df.iloc[:args.max_rows].copy()

    col_info = infer_columns(df)
    sentence_col = col_info["sentence_col"]
    condition_col = col_info["condition_col"]
    congruent_col = col_info["congruent_col"]

    if condition_col is None:
        df["condition_filled"] = ""
        condition_col = "condition_filled"

    results: List[Dict[str, Any]] = []

    for idx, row in df.iterrows():
        sentence = str(row[sentence_col]).strip()
        condition = str(row[condition_col]).strip() if condition_col in row else ""

        prefix, suffix = split_prefix_and_suffix(sentence)


        suffix_scores = score_suffix_conditional_nll(
            prefix=prefix,
            suffix=suffix,
            tokenizer=tokenizer,
            model=model,
            device=args.device,
        )


        sentence_scores = score_full_sentence_nll(
            sentence=sentence,
            tokenizer=tokenizer,
            model=model,
            device=args.device,
        )

        item = {
            "row_id": idx,
            "sentence": sentence,
            "condition": condition,
            "prefix": prefix,
            "suffix": suffix,
            "context_key": make_context_key(sentence, condition),
        }

        if congruent_col is not None and congruent_col in df.columns:
            item[congruent_col] = row[congruent_col]

        item.update(suffix_scores)
        item.update(sentence_scores)
        results.append(item)

        if (len(results) % 20) == 0:
            print(f"Processed {len(results)} examples...")

    result_df = pd.DataFrame(results)


    result_df.to_csv(args.output_csv, index=False)
    print(f"Saved per-example results to: {args.output_csv}")


    summary_df = summarize_by_label(result_df, congruent_col)
    if len(summary_df) > 0:
        summary_df.to_csv(args.summary_csv, index=False)
        print(f"Saved summary statistics to: {args.summary_csv}")
    else:
        print("No valid congruent column found, skipped summary by label.")

    # paired preference
    pairwise_df = compute_pairwise_metrics(result_df, congruent_col)
    if len(pairwise_df) > 0:
        pairwise_df.to_csv(args.pairwise_csv, index=False)
        print(f"Saved pairwise comparison results to: {args.pairwise_csv}")

        suffix_acc = pairwise_df["suffix_prefers_Y"].mean()
        sent_acc = pairwise_df["sentence_prefers_Y"].mean()

        print("\n===== Pairwise Preference Accuracy =====")
        print(f"Suffix metric prefers Y:   {suffix_acc:.4f}")
        print(f"Sentence metric prefers Y: {sent_acc:.4f}")
    else:
        print("No Y/N paired groups found, skipped pairwise metrics.")


    print("\n===== Quick Summary =====")
    show_cols = [
        "suffix_avg_nll",
        "sentence_avg_nll",
        "sentence_ppl",
    ]
    for c in show_cols:
        if c in result_df.columns:
            print(f"{c}: mean={result_df[c].mean():.4f}, std={result_df[c].std():.4f}")

    if congruent_col is not None and congruent_col in result_df.columns:
        print("\n===== Mean by congruent label =====")
        grouped = result_df.groupby(congruent_col)[show_cols].mean()
        print(grouped)


if __name__ == "__main__":
    main()