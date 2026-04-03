import pandas as pd
import torch
from transformers import BertTokenizer, BertForMaskedLM

tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
model = BertForMaskedLM.from_pretrained("bert-base-uncased")
model.eval()


def prediction(sentence, target_sentence, mask_token='[MASK]'):
    labels = tokenizer(target_sentence, return_tensors="pt")["input_ids"]
    inputs = tokenizer(sentence, return_tensors="pt")

    target_length = len(labels[0])
    input_length = len(inputs["input_ids"][0])

    if target_length != input_length:
        mask = mask_token
        for _ in range(target_length - input_length):
            mask = mask + " " + mask_token
        sentence = sentence.replace(mask_token, mask)
        inputs = tokenizer(sentence, return_tensors="pt")

    outputs = model(**inputs, labels=labels)
    return outputs.loss.item()


def main(file):
    df = pd.read_csv(file)
    df.columns = df.columns.str.strip()

    target_sentence = list(df["sentence"])
    masked_sentences = []

    for item in target_sentence:
        words = item.strip().split()
        words[-1] = "[MASK]"
        masked_sentences.append(" ".join(words))

    df["masked_sentence"] = masked_sentences
    losses = []

    with open("output.txt", "w") as f:
        for i in range(len(df)):
            loss_item = prediction(masked_sentences[i], target_sentence[i])
            losses.append(loss_item)

            f.write(
                f"{masked_sentences[i]} | "
                f"{target_sentence[i]} | "
                f"{loss_item} | "
                f"{df['condition'][i]} | "
                f"{df['CW- or CWC-congruent'][i]}\n"
            )

    df["loss"] = losses
    df.to_csv("output.csv", index=False)
    return df


main("dataset/large-scale.csv")