#%%
import torch
import json
from transformers import AutoTokenizer
from src.models import calculate_prompt_tokens

def load_model_and_tokenizer():

    if torch.cuda.is_available():
        DEVICE = "cuda"
        print(f"GPU name: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        print(f"GPU Memory Available: {torch.cuda.memory_reserved(0) / 1e9:.1f} GB")
    else:
        print("Usando CPU")
        DEVICE = "cpu"

    # === CARICAMENTO MODELLO E TOKENIZER ===
    MODEL_NAME = "openai/gpt-oss-20b"

    try:
        # Caricamento tokenizer
        # trust_remote_code=True: Necessario per modelli che usano codice custom
        # use_fast=True: Usa il tokenizer veloce (implementazione Rust) se disponibile
        tokenizer = AutoTokenizer.from_pretrained(
            MODEL_NAME, 
            trust_remote_code=True,
            use_fast=True
        )

        # Set pad token per gpt-oss
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        return tokenizer, DEVICE

    except torch.cuda.OutOfMemoryError as e:
        print(f" Out of memory durante il caricamento: {e}")
        raise

tok, dev = load_model_and_tokenizer()

dataset_types = ["Alzheimer", "BPD", "Schizophrenia", "Depression", "Bipolar"]

for dataset_type in dataset_types:
    prompt_tkns = calculate_prompt_tokens(tok, dataset_type)
    print(f"{dataset_type}: {prompt_tkns} => {8000 - prompt_tkns}")

'''
Alzheimer con tutti gli shots: 10205 => -2205

Alzheimer: 6281 => 1719
BPD: 5731 => 2269
Schizophrenia: 7874 => 126
Depression: 4181 => 3819
Bipolar: 8625 => -625

Con solo outcome_measurement_description:
Alzheimer: 8625 => -625
BPD: 7210 => 790
Schizophrenia: 5971 => 2029
Depression: 3896 => 4104
Bipolar: 7454 => 546
'''


# %%
import json

dataset_types = ["Alzheimer", "BPD"]

for dataset_type in dataset_types:

    with open(f"/home/cdemaria/Documents/biomarker-extractor/results/{dataset_type}/biomarkers.json", "r", encoding="utf-8") as f:
        parsed_biomarkers = json.load(f)

    final_biomarkers_sorted = sorted(parsed_biomarkers, key=lambda x: len(x['occurrences']), reverse=True)

    with open(f"/home/cdemaria/Documents/biomarker-extractor/results/{dataset_type}/biomarkers.json", "w", encoding="utf-8") as f:
        json.dump(final_biomarkers_sorted, f, ensure_ascii=False, indent=2)

