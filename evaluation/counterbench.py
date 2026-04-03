import json
from collections import defaultdict
import torch
import gc
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from tqdm import tqdm

# DATASET
print("Loading Dataset...")
file_path = "data_balanced_alpha_V1.json"
with open(file_path, "r", encoding="utf-8") as f:
    data = json.load(f)


# MODEL
torch.cuda.empty_cache()
gc.collect()
hf_token = "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"


model_id = "meta-llama/Meta-Llama-3.1-8B-Instruct"
# model_id = "mistralai/Mistral-7B-Instruct-v0.3"
# model_id = "Qwen/Qwen3-8B"

print("Configuring 4-bit quantization parameters...")
quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
)

print(f"Downloading/Loading model: {model_id} (this may take a few minutes)...")

tokenizer = AutoTokenizer.from_pretrained(model_id, token=hf_token)

model = AutoModelForCausalLM.from_pretrained(
    model_id,
    quantization_config=quantization_config,
    device_map={"": 0},
    token=hf_token
)

print("Model loaded successfully!")

memory_footprint_bytes = model.get_memory_footprint()
memory_footprint_gb = memory_footprint_bytes / (1024 ** 3)
print(f"Current model weight memory footprint: {memory_footprint_gb:.2f} GB")

def generate_answer(model, item):
    system_prompt = ""
    question = item["question"]
    given_info = item["given_info"]
    prompt = (
        f"Imagine a self-contained, hypothetical world with only the following conditions, and without any unmentioned factors or causal relationships: {given_info} {question}.\n"
        "Start your answer with \"Yes\" or \"No\", followed by additional reasoning or evidence to support your explanation."
    )

    messages = [
        {"role": "system", "content": system_prompt},
        {
            "role": "user",
            "content": prompt
        }
    ]
    prompt_text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False
    )
    model_inputs = tokenizer([prompt_text], return_tensors="pt").to(model.device)
    with torch.no_grad():
        generated_ids = model.generate(
            **model_inputs,
            max_new_tokens=150,
            temperature=0.1,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )
    input_length = model_inputs.input_ids.shape[1]
    response_ids = generated_ids[0][input_length:]
    response = tokenizer.decode(response_ids, skip_special_tokens=True)
    return response.strip()

print("Generating answers...")
for item in tqdm(data):
    answer = generate_answer(model, item)
    item["pred"] = answer

with open(f"{model_id.split('/')[-1]}_output.json", "w") as f:
    json.dump(data, f, indent=4)

import re
def parse_answer(text):
    if not text:
        return None
    first_word = text.strip().split()[0].lower()
    first_word = re.sub(r'[^\w]', '', first_word)

    if first_word == 'yes':
        return 'yes'
    elif first_word == 'no':
        return 'no'
    else:
        return None

def evaluate(output):
    results = {'overall': {'correct': 0, 'total': 0},}
    for item in output:
        pred = item["pred"]
        pred = parse_answer(pred)
        gt = item["answer"]
        item_type = item["type"]
        if item_type not in results:
            results[item_type] = {'correct': 0, 'total': 0}
        results['overall']['total'] += 1
        results[item_type]['total'] += 1
        if pred == gt:
            results['overall']['correct'] += 1
            results[item_type]['correct'] += 1
    return results


with open(f"{model_id.split('/')[-1]}_output.json", "r") as f:
    output = json.load(f)
results = evaluate(output)
print(f"Evaluation results on CounterBench ({model_id}):")
for key in results:
    correct, total = results[key]['correct'], results[key]['total']
    accuracy = correct / total if total > 0 else 0
    print(f"{key.capitalize()} - Acc {accuracy} | Total {total}")