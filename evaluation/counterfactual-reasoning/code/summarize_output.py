import pandas as pd
import argparse


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_csv",
        type=str,
        default="results/mistral/output_mistral_8b_metrics.csv",
        help="Path to row-level Mistral csv file"
    )
    parser.add_argument(
        "--parsed_csv",
        type=str,
        default="results/mistral/mistral_parsed.csv",
        help="Path to save normalized row-level csv"
    )
    parser.add_argument(
        "--summary_csv",
        type=str,
        default="results/mistral/output_summary.csv",
        help="Path to save summary csv"
    )
    parser.add_argument(
        "--pairwise_csv",
        type=str,
        default="results/mistral/output_pairwise.csv",
        help="Path to save pairwise csv"
    )
    return parser.parse_args()


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [c.strip() for c in df.columns]

    required_cols = [
        "row_id",
        "sentence",
        "condition",
        "prefix",
        "suffix",
        "context_key",
        "CW- or CWC-congruent",
        "suffix_total_nll",
        "suffix_avg_nll",
        "suffix_token_count",
        "suffix_ppl",
        "sentence_total_nll",
        "sentence_avg_nll",
        "sentence_token_count",
        "sentence_ppl",
    ]

    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    return df


def compute_pairwise_preferences(df: pd.DataFrame, label_col: str) -> pd.DataFrame:
    rows = []

    for context_key, group in df.groupby("context_key"):
        labels = set(group[label_col].astype(str))
        if "Y" not in labels or "N" not in labels:
            continue

        y_group = group[group[label_col].astype(str) == "Y"]
        n_group = group[group[label_col].astype(str) == "N"]

        if len(y_group) == 0 or len(n_group) == 0:
            continue

        y_sentence_avg_nll = y_group["sentence_avg_nll"].mean()
        n_sentence_avg_nll = n_group["sentence_avg_nll"].mean()

        y_suffix_avg_nll = y_group["suffix_avg_nll"].mean()
        n_suffix_avg_nll = n_group["suffix_avg_nll"].mean()

        rows.append({
            "context_key": context_key,
            "condition": y_group["condition"].iloc[0],
            "y_sentence_avg_nll": y_sentence_avg_nll,
            "n_sentence_avg_nll": n_sentence_avg_nll,
            "sentence_prefers_Y": int(y_sentence_avg_nll < n_sentence_avg_nll),
            "y_suffix_avg_nll": y_suffix_avg_nll,
            "n_suffix_avg_nll": n_suffix_avg_nll,
            "suffix_prefers_Y": int(y_suffix_avg_nll < n_suffix_avg_nll),
        })

    return pd.DataFrame(rows)


def build_summary(df: pd.DataFrame, pairwise_df: pd.DataFrame) -> pd.DataFrame:
    summary_rows = []

    for condition, group in df.groupby("condition"):
        row = {
            "condition": condition,
            "sentence_avg_nll": group["sentence_avg_nll"].mean(),
            "suffix_avg_nll": group["suffix_avg_nll"].mean(),
            "sentence_ppl_mean": group["sentence_ppl"].mean(),
            "sentence_ppl_std": group["sentence_ppl"].std(),
        }

        pair_group = pairwise_df[pairwise_df["condition"] == condition] if len(pairwise_df) > 0 else pd.DataFrame()

        if len(pair_group) > 0:
            row["sentence_prefers_Y"] = pair_group["sentence_prefers_Y"].mean()
            row["suffix_prefers_Y"] = pair_group["suffix_prefers_Y"].mean()
            row["num_pairs"] = len(pair_group)
        else:
            row["sentence_prefers_Y"] = float("nan")
            row["suffix_prefers_Y"] = float("nan")
            row["num_pairs"] = 0

        summary_rows.append(row)

    summary_df = pd.DataFrame(summary_rows)

    desired_order = [
        "condition",
        "sentence_avg_nll",
        "suffix_avg_nll",
        "sentence_prefers_Y",
        "suffix_prefers_Y",
        "sentence_ppl_mean",
        "sentence_ppl_std",
        "num_pairs",
    ]
    summary_df = summary_df[desired_order]
    return summary_df.sort_values("condition").reset_index(drop=True)


def main():
    args = parse_args()

    df = pd.read_csv(args.input_csv)
    df = normalize_columns(df)


    df.to_csv(args.parsed_csv, index=False)

    label_col = "CW- or CWC-congruent"


    pairwise_df = compute_pairwise_preferences(df, label_col)
    pairwise_df.to_csv(args.pairwise_csv, index=False)

  
    summary_df = build_summary(df, pairwise_df)
    summary_df.to_csv(args.summary_csv, index=False)

    print(f"Parsed rows saved to: {args.parsed_csv}")
    print(f"Pairwise results saved to: {args.pairwise_csv}")
    print(f"Summary saved to: {args.summary_csv}")
    print("\nSummary:")
    print(summary_df.to_string(index=False))


if __name__ == "__main__":
    main()