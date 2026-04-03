import re
import json
import argparse
from typing import Dict, Any, Tuple

import pandas as pd


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_csv", type=str, required=True)
    parser.add_argument("--output_inference_csv", type=str, default="converted_inference.csv")
    parser.add_argument("--output_weak_gold_csv", type=str, default="converted_weak_gold.csv")
    return parser.parse_args()


def safe_str(x):
    if pd.isna(x):
        return ""
    return str(x).strip()


def split_if_clause(sentence: str) -> Tuple[str, str]:
    """
    Heuristic split:
    'If A, B.' -> antecedent=A, consequent=B
    """
    sentence = sentence.strip().strip('"').strip()
    m = re.match(r"^If\s+(.+?),\s*(.+)$", sentence, flags=re.IGNORECASE)
    if m:
        antecedent = m.group(1).strip()
        consequent = m.group(2).strip()
        return antecedent, consequent
    return "", sentence


def normalize_consequent(consequent: str) -> str:
    return consequent.rstrip(" .")


def make_qcc_from_sentence(sentence: str) -> str:
    antecedent, consequent = split_if_clause(sentence)
    consequent_clean = normalize_consequent(consequent)

    if antecedent:
        return f"If it were not the case that {antecedent}, would it still be the case that {consequent_clean}?"
    return f"If the premise were false, would it still be the case that {consequent_clean}?"


def make_premise_from_sentence(sentence: str) -> str:
    """
    Heuristic factualized premise.
    This is weak and surface-based.
    """
    antecedent, consequent = split_if_clause(sentence)
    consequent_clean = normalize_consequent(consequent)

    if antecedent:
        return f"Suppose that {antecedent}. Under that condition, {consequent_clean}."
    return sentence.strip().strip('"')


def extract_simple_variables(sentence: str) -> Dict[str, str]:
    """
    Very rough heuristic variables:
    X <- antecedent
    Y <- consequent
    Z <- general background context
    M <- generic mechanism from antecedent to consequent

    This is weak supervision only.
    """
    antecedent, consequent = split_if_clause(sentence)
    consequent_clean = normalize_consequent(consequent)

    if antecedent:
        x = antecedent
        y = consequent_clean
        z = f"context in which {antecedent} is relevant"
        m = f"mechanism by which {antecedent} leads to {consequent_clean}"
    else:
        x = "implied intervention in the sentence"
        y = consequent_clean if consequent_clean else sentence
        z = "background contextual factors"
        m = f"mechanism leading to {y}"

    return {"X": x, "Y": y, "Z": z, "M": m}


def build_baseline_answer(sentence: str) -> str:
    vars_ = extract_simple_variables(sentence)
    answer = {
        "variables": vars_,
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
    }
    return json.dumps(answer, ensure_ascii=False)


def build_structured_cot_answer(sentence: str) -> str:
    vars_ = extract_simple_variables(sentence)
    antecedent, consequent = split_if_clause(sentence)
    consequent_clean = normalize_consequent(consequent)

    answer = {
        "reasoning": {
            "step_1_factual_event": sentence.strip().strip('"'),
            "step_2_counterfactual_intervention": (
                f"negating or altering the antecedent: {antecedent}" if antecedent else "altering the implied premise"
            ),
            "step_2_target_consequence": consequent_clean if consequent_clean else sentence.strip().strip('"'),
            "step_4_mechanism": {
                "x_to_m": f"{vars_['X']} changes the intermediate process {vars_['M']}",
                "m_to_y": f"{vars_['M']} affects whether {vars_['Y']} holds",
                "z_role": f"{vars_['Z']} provides the background conditions",
                "counterfactual_change": f"changing X alters M and therefore changes Y'"
            }
        },
        "variables": vars_,
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
    }
    return json.dumps(answer, ensure_ascii=False)


def infer_congruent_col(df: pd.DataFrame):
    for c in df.columns:
        if "congruent" in c.lower():
            return c
    return None


def main():
    args = parse_args()

    df = pd.read_csv(args.input_csv)
    df.columns = [c.strip() for c in df.columns]

    if "sentence" not in df.columns:
        raise ValueError(f"Need a 'sentence' column. Available columns: {df.columns.tolist()}")

    congruent_col = infer_congruent_col(df)

    inference_rows = []
    weak_gold_rows = []

    for idx, row in df.iterrows():
        sentence = safe_str(row["sentence"])
        condition = safe_str(row["condition"]) if "condition" in df.columns else ""
        congruent = safe_str(row[congruent_col]) if congruent_col is not None else ""

        premise = make_premise_from_sentence(sentence)
        qcc = make_qcc_from_sentence(sentence)

        inference_item = {
            "row_id": idx,
            "sentence": sentence,
            "condition": condition,
            "premise": premise,
            "qcc": qcc,
        }
        if congruent_col is not None:
            inference_item["congruent"] = congruent

        weak_gold_item = {
            "row_id": idx,
            "sentence": sentence,
            "condition": condition,
            "premise": premise,
            "qcc": qcc,
            "answer_baseline": build_baseline_answer(sentence),
            "answer_structured_cot": build_structured_cot_answer(sentence),
        }
        if congruent_col is not None:
            weak_gold_item["congruent"] = congruent

        inference_rows.append(inference_item)
        weak_gold_rows.append(weak_gold_item)

    inference_df = pd.DataFrame(inference_rows)
    weak_gold_df = pd.DataFrame(weak_gold_rows)

    inference_df.to_csv(args.output_inference_csv, index=False)
    weak_gold_df.to_csv(args.output_weak_gold_csv, index=False)

    print(f"Saved inference dataset to: {args.output_inference_csv}")
    print(f"Saved weak-gold dataset to: {args.output_weak_gold_csv}")
    print("\nImportant: the generated answers are weak heuristic annotations, not human gold.")
    print("You should manually inspect a subset before using them as evaluation targets.")


if __name__ == "__main__":
    main()