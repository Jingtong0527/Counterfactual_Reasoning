import math
import argparse
from typing import Dict, Any, List, Optional

import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--input_csv", type=str, required=True)
    parser.add_argument("--output_csv", type=str, default="output_prompt_metrics.csv")
    parser.add_argument("--summary_csv", type=str, default="output_prompt_summary.csv")
    parser.add_argument("--pairwise_csv", type=str, default="output_prompt_pairwise.csv")

    parser.add_argument(
        "--model_name",
        type=str,
        default="meta-llama/Llama-3.1-8B-Instruct"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument("--max_rows", type=int, default=None)

    parser.add_argument(
        "--prompt_style",
        type=str,
        default="baseline",
        choices=["baseline", "structured_cot"]
    )
    parser.add_argument("--save_prompt", action="store_true")

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


def safe_str(x: Any) -> str:
    if pd.isna(x):
        return ""
    return str(x).strip()


def build_prompt(premise: str, qcc: str, prompt_style: str) -> str:
    if prompt_style == "baseline":
        return f"""# Role: Counterfactual Reasoning Analyst

Your task is to identify causal variables and construct factual and counterfactual causal graphs.
All outputs must follow strict JSON formatting for automated saving.

## Context
Premise:
{premise}

Questionized Counterfactual Conditional (QCC):
{qcc}

## Task

Identify the following variables:
- X: Exposure (the manipulated action or intervention)
- Y: Outcome (the final consequence)
- Z: Covariate (background condition influencing X, M, and Y)
- M: Mediator (mechanism connecting X and Y)

Then construct:

### Factual graph
It must include:
- Z -> X
- Z -> M
- Z -> Y
- X -> M
- M -> Y
- X -> Y

### Counterfactual graph
It must include:
- Z -> M'
- Z -> Y'
- X' -> M'
- M' -> Y'
- X' -> Y'

Use prime notation for counterfactual variables.

## Output JSON
{{
  "variables": {{
    "X": "...",
    "Y": "...",
    "Z": "...",
    "M": "..."
  }},
  "factual_graph": [
    "Z -> X",
    "Z -> M",
    "Z -> Y",
    "X -> M",
    "M -> Y",
    "X -> Y"
  ],
  "counterfactual_graph": [
    "Z -> M'",
    "Z -> Y'",
    "X' -> M'",
    "M' -> Y'",
    "X' -> Y'"
  ]
}}
"""

    if prompt_style == "structured_cot":
        return f"""# Role: Counterfactual Reasoning Analyst

Your task is to identify causal variables and construct factual and counterfactual causal graphs.
You must reason using the structured causal schema below and return strict JSON.

## Context
Premise:
{premise}

Questionized Counterfactual Conditional (QCC):
{qcc}

## Structured Reasoning Procedure

Step 1. Identify the factual event in the premise.

Step 2. Identify the counterfactual intervention in the QCC and the target consequence whose value may change.

Step 3. Identify the causal variables:
- X: the manipulated action or intervention
- Y: the final consequence
- Z: the background condition or context that can influence X, M, and Y
- M: the intermediate mechanism linking X and Y

Step 4. Explain the mechanism briefly:
- why X changes M,
- why M changes Y,
- how Z supports or constrains the process,
- how the intervention changes the counterfactual outcome.

Step 5. Construct the causal graphs.

### Required factual edges
- Z -> X
- Z -> M
- Z -> Y
- X -> M
- M -> Y
- X -> Y

### Required counterfactual edges
- Z -> M'
- Z -> Y'
- X' -> M'
- M' -> Y'
- X' -> Y'

Use prime notation for counterfactual variables.

## Output JSON
{{
  "reasoning": {{
    "step_1_factual_event": "...",
    "step_2_counterfactual_intervention": "...",
    "step_2_target_consequence": "...",
    "step_4_mechanism": {{
      "x_to_m": "...",
      "m_to_y": "...",
      "z_role": "...",
      "counterfactual_change": "..."
    }}
  }},
  "variables": {{
    "X": "...",
    "Y": "...",
    "Z": "...",
    "M": "..."
  }},
  "factual_graph": [
    "Z -> X",
    "Z -> M",
    "Z -> Y",
    "X -> M",
    "M -> Y",
    "X -> Y"
  ],
  "counterfactual_graph": [
    "Z -> M'",
    "Z -> Y'",
    "X' -> M'",
    "M' -> Y'",
    "X' -> Y'"
  ]
}}
"""
    raise ValueError(f"Unsupported prompt_style: {prompt_style}")


@torch.no_grad()
def score_answer_given_prompt_nll(prompt: str, answer: str, tokenizer, model) -> Dict[str, Any]:
    prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
    answer_ids = tokenizer(answer, add_special_tokens=False)["input_ids"]
    full_ids = prompt_ids + answer_ids

    if len(full_ids) == 0:
        return {
            "answer_total_nll": float("nan"),
            "answer_avg_nll": float("nan"),
            "answer_token_count": 0,
            "answer_ppl": float("nan"),
        }

    model_device = next(model.parameters()).device
    input_ids = torch.tensor([full_ids], dtype=torch.long, device=model_device)
    attention_mask = torch.ones_like(input_ids)

    labels = input_ids.clone()
    labels[:, :len(prompt_ids)] = -100

    outputs = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        labels=labels
    )

    avg_nll = outputs.loss.item()
    token_count = max(len(answer_ids), 1)
    total_nll = avg_nll * token_count
    ppl = math.exp(avg_nll) if avg_nll < 50 else float("inf")

    return {
        "answer_total_nll": total_nll,
        "answer_avg_nll": avg_nll,
        "answer_token_count": token_count,
        "answer_ppl": ppl,
    }


@torch.no_grad()
def score_full_text_nll(text: str, tokenizer, model) -> Dict[str, Any]:
    enc = tokenizer(text, return_tensors="pt", add_special_tokens=True)
    model_device = next(model.parameters()).device

    input_ids = enc["input_ids"].to(model_device)
    attention_mask = enc["attention_mask"].to(model_device)

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
        "full_total_nll": total_nll,
        "full_avg_nll": avg_nll,
        "full_token_count": token_count,
        "full_ppl": ppl,
    }


def infer_columns(df: pd.DataFrame) -> Dict[str, Optional[str]]:
    df.columns = [c.strip() for c in df.columns]

    premise_col = None
    qcc_col = None
    answer_col = None
    congruent_col = None

    for c in df.columns:
        lc = c.lower().strip()
        if lc == "premise":
            premise_col = c
        elif lc in ["qcc", "questionized counterfactual conditional"]:
            qcc_col = c
        elif lc in ["answer", "gold_answer", "gold answer", "target"]:
            answer_col = c
        elif "congruent" in lc:
            congruent_col = c

    if premise_col is None or qcc_col is None or answer_col is None:
        raise ValueError(
            f"Need columns premise, qcc, answer. Available columns: {df.columns.tolist()}"
        )

    return {
        "premise_col": premise_col,
        "qcc_col": qcc_col,
        "answer_col": answer_col,
        "congruent_col": congruent_col,
    }


def summarize_by_label(df: pd.DataFrame, congruent_col: Optional[str]) -> pd.DataFrame:
    metric_cols = [
        "answer_total_nll",
        "answer_avg_nll",
        "answer_ppl",
        "full_total_nll",
        "full_avg_nll",
        "full_ppl",
    ]
    available = [c for c in metric_cols if c in df.columns]

    if congruent_col is None or congruent_col not in df.columns or len(available) == 0:
        return pd.DataFrame()

    summary = df.groupby(congruent_col)[available].agg(["mean", "std", "count"])
    summary.columns = ["__".join(col).strip() for col in summary.columns.values]
    summary = summary.reset_index()
    return summary


def compute_pairwise_metrics(df: pd.DataFrame, congruent_col: Optional[str]) -> pd.DataFrame:
    if congruent_col is None or congruent_col not in df.columns:
        return pd.DataFrame()
    if "context_key" not in df.columns:
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

        row = {"context_key": context_key}

        y_ans = y_group["answer_avg_nll"].mean()
        n_ans = n_group["answer_avg_nll"].mean()
        row["y_answer_avg_nll"] = y_ans
        row["n_answer_avg_nll"] = n_ans
        row["answer_prefers_Y"] = int(y_ans < n_ans)

        y_full = y_group["full_avg_nll"].mean()
        n_full = n_group["full_avg_nll"].mean()
        row["y_full_avg_nll"] = y_full
        row["n_full_avg_nll"] = n_full
        row["full_prefers_Y"] = int(y_full < n_full)

        rows.append(row)

    return pd.DataFrame(rows)


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
    premise_col = col_info["premise_col"]
    qcc_col = col_info["qcc_col"]
    answer_col = col_info["answer_col"]
    congruent_col = col_info["congruent_col"]

    results: List[Dict[str, Any]] = []

    for idx, row in df.iterrows():
        premise = safe_str(row[premise_col])
        qcc = safe_str(row[qcc_col])
        answer = safe_str(row[answer_col])

        prompt = build_prompt(premise=premise, qcc=qcc, prompt_style=args.prompt_style)

        answer_scores = score_answer_given_prompt_nll(
            prompt=prompt,
            answer=answer,
            tokenizer=tokenizer,
            model=model,
        )

        full_scores = score_full_text_nll(
            text=prompt + answer,
            tokenizer=tokenizer,
            model=model,
        )

        item = {
            "row_id": idx,
            "premise": premise,
            "qcc": qcc,
            "answer": answer,
            "prompt_style": args.prompt_style,
            "context_key": f"{premise} || {qcc}",
        }

        if args.save_prompt:
            item["prompt"] = prompt

        if congruent_col is not None and congruent_col in df.columns:
            item[congruent_col] = row[congruent_col]

        item.update(answer_scores)
        item.update(full_scores)
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
        print("Skipped summary by label.")

    pairwise_df = compute_pairwise_metrics(result_df, congruent_col)
    if len(pairwise_df) > 0:
        pairwise_df.to_csv(args.pairwise_csv, index=False)
        print(f"Saved pairwise comparison results to: {args.pairwise_csv}")
        print("\n===== Pairwise Preference Accuracy =====")
        print(f"answer_prefers_Y: {pairwise_df['answer_prefers_Y'].mean():.4f}")
        print(f"full_prefers_Y:   {pairwise_df['full_prefers_Y'].mean():.4f}")
    else:
        print("No Y/N paired groups found, skipped pairwise metrics.")

    print("\n===== Quick Summary =====")
    for c in ["answer_avg_nll", "answer_ppl", "full_avg_nll", "full_ppl"]:
        if c in result_df.columns:
            print(f"{c}: mean={result_df[c].mean():.4f}, std={result_df[c].std():.4f}")


if __name__ == "__main__":
    main()