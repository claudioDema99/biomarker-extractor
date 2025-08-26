import json
import re
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

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
print(f"Caricamento modello {MODEL_NAME}..")

# Caricamento tokenizer
# trust_remote_code=True: Necessario per modelli che usano codice custom
# use_fast=True: Usa il tokenizer veloce (implementazione Rust) se disponibile
tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME, 
    trust_remote_code=True,
    use_fast=True
)

# Caricamento modello
# torch_dtype: Tipo di dato per i pesi del modello (float16 per GPU, float32 per CPU)
# device_map="auto": Distribuisce automaticamente il modello tra GPU/CPU disponibili
# trust_remote_code=True: Permette l'esecuzione di codice personalizzato dal modello
# low_cpu_mem_usage=True: Riduce l'uso della memoria CPU durante il caricamento
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME, 
    torch_dtype=torch.bfloat16 if DEVICE=="cuda" else torch.float32,
    device_map="auto",
    trust_remote_code=True,
    low_cpu_mem_usage=True
)

def read_json_arrays_to_list(filename):
    """
    Read a file containing multiple JSON arrays and combine them into one big list.
    """
    combined_list = []
    
    with open(filename, 'r') as file:
        content = file.read()
        
        # Find all JSON arrays in the content using regex
        # This looks for patterns that start with [ and end with ]
        json_arrays = re.findall(r'\[.*?\]', content, re.DOTALL)
        
        for json_str in json_arrays:
            try:
                array_data = json.loads(json_str)
                if isinstance(array_data, list):
                    combined_list.extend(array_data)
                else:
                    print(f"Warning: Found non-array JSON: {json_str[:100]}...")
            except json.JSONDecodeError as e:
                print(f"Error parsing JSON: {e}")
                continue
    
    return combined_list

def call_model(biomarkers):
    system_prompt = f"""You are an expert biomedical data curator with deep knowledge of Alzheimer's disease pathophysiology and biomarkers. Your primary function is to process lists of biomarker terms, and group them by their underlying biological significance.
Rules for Operation:
Core Task: You will receive lists of raw biomarker strings. You must aggregate synonymous terms, even if they have different spellings, abbreviations, or syntactic structures (e.g., "P-Tau 181", "pTau181", "phosphorylated tau at threonine 181").
Canonical Naming: For each group of synonyms, you must assign a single, scientifically precise, and human-readable canonical name (e.g., "P-Tau 181").
Output Format: You will always output a valid JSON object. No other text, explanations, or apologies should precede or follow the JSON code block.
Handling Uncertainty: If you encounter a highly ambiguous term, group it cautiously and use the "Other" category. It is more important to be accurate than to force a categorization.
Focus: Do not provide commentary, summaries, or analysis beyond the requested JSON structure. Your entire output must be the JSON object.
"""

    user_prompt = f"""Please process the following list of raw Alzheimer's disease biomarker terms. Aggregate synonymous terms, and provide a list of all occurrences and their count.
Output a JSON object where each key is an object with these exact keys:
"canonical_biomarker": (string) The standardized name.
"occurrences": (array of strings) All raw terms grouped under this canonical name.

Example Input:
["P-Tau 181", "pTau181", "Amyloid Beta 42/40 ratio"]

Expected Output Structure:
[
    {{
        "canonical_biomarker": "P-Tau 181",
        "occurrences": ["P-Tau 181", "pTau181"]
    }},
    {{
        "canonical_biomarker": "Aβ42/40 Ratio",
        "occurrences": ["Amyloid Beta 42/40 ratio, "Aβ42/40 Ratio", "Abeta42/40"]
    }}
]

Here is the full list of biomarkers to process:

{biomarkers}
"""
    #print(system_prompt)
    #print("\n_____________________\n")
    #print(user_prompt)
    #input()

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]
    
    # apply_chat_template: Converte i messaggi nel formato richiesto dal modello
    # tokenize=False: Restituisce stringa invece di token IDs
    # add_generation_prompt=True: Aggiunge il prompt per iniziare la generazione
    full_prompt = tokenizer.apply_chat_template(
        messages, 
        tokenize=False, 
        add_generation_prompt=True
    )

    # Tokenizzazione dell'input completo
    # return_tensors="pt": Restituisce tensori PyTorch
    # padding=True: Aggiunge padding se necessario
    # truncation=True: Tronca se supera la lunghezza massima
    inputs = tokenizer(
        full_prompt, 
        return_tensors="pt", 
        padding=True, 
        truncation=True,
        max_length=8192  # Context window di gpt-oss-20b
    ).to(DEVICE)
    # Conteggio token effettivi
    token_count = inputs.input_ids.shape[1]
    #print(f"Token count effettivi: {token_count}")
    print(f"--- Waiting for the {MODEL_NAME} response ---")
    # Generazione della risposta
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=4096,          # Massimo numero di nuovi token da generare
            do_sample=False,              # Usa greedy decoding (deterministic!!!)
            temperature=0.7,              # Controllo randomness (se do_sample=True) quindi inutile
            top_p=0.9,                    # Nucleus sampling (se do_sample=True) quindi inutile
            pad_token_id=tokenizer.eos_token_id,  # Token di padding
            eos_token_id=tokenizer.eos_token_id,  # Token di fine sequenza
            use_cache=True,               ### Usala STA CACHE!!!!!
            repetition_penalty=1.1,       # Penalità per ripetizioni
            length_penalty=1.0            # Penalità per lunghezza
        )

    # Estrazione solo della parte generata (esclude l'input)
    input_length = inputs.input_ids.shape[1]
    generated_tokens = outputs[0][input_length:]
    
    # Decodifica dei token generati
    # skip_special_tokens=True: Rimuove token speciali (<eos>, <pad>, ecc.)
    # clean_up_tokenization_spaces=True: Pulisce spazi extra dalla tokenizzazione
    output_text = tokenizer.decode(
        generated_tokens, 
        skip_special_tokens=False,         # Mi servono per filtrare il ragionamento
        clean_up_tokenization_spaces=True
    )

    final_start = '<|end|><|start|>assistant<|channel|>final<|message|>'
    if final_start in output_text:
        parts = output_text.split(final_start)
        cot = final_start.join(parts[:-1])  # Tutto prima del tag finale
        response = parts[-1].strip()              # Solo la risposta finale
        cot = (
            cot
            .replace('<|channel|>analysis<|message|>', '')
            .replace('<|end|>', '')
            .strip()
        )
        response = (
            response
            .replace('<|return|>', '')
            .strip()
        )
    else:
        cot = ""
        response = output_text.strip()
        # ritorna l'output completo se non trova i tag
        print(f"No filtering tags found, returning full output.")

    #print(f"Output from {MODEL_NAME}: {output_text}")

    try:
        # Isola la sezione JSON anche se c'è testo extra
        start = response.find('[')
        end   = response.rfind(']') + 1   # rfind => ultima graffa

        if start == -1 or end == 0:
            return [],

        json_str = response[start:end]

        # Carica il JSON
        data = [json.loads(json_str)]

        # Garantisci che sia effettivamente una lista, altrimenti torna lista vuota
        if isinstance(data, list) and len(data) > 0:
            return data
        else:
            return []

    except (json.JSONDecodeError, TypeError) as e:
        print(f"Errore nell'estrazione dei biomarkers: {e}")
        return []


filename = 'risultati_acronyms.txt'
acronyms = read_json_arrays_to_list(filename)

print(f"Total dictionaries: {len(acronyms)}")
print(f"First few items: {acronyms[:5]}")

validated_biomarkers = []
for acronym in acronyms:
    if acronym["valid"] == True:
        validated_biomarkers.append(acronym["acronym"])

print(f"Total valid biomarkers: {len(validated_biomarkers)} on {len(acronyms)} total biomarkers")

input()

# try to add biomarkers until a max token count of 7000 is reached
batch = []
current_token_count = 0
max_token_count = 6000

while validated_biomarkers:
    biomarker = validated_biomarkers[0]
    tokenized = tokenizer(biomarker, return_tensors="pt", padding=True, truncation=True)
    token_count = tokenized.input_ids.shape[1]
    
    if current_token_count + token_count <= max_token_count:
        batch.append(biomarker)
        validated_biomarkers.pop(0)
        current_token_count += token_count
    else:
        call_model(batch)
        batch = [biomarker]  # Start new batch with current biomarker
        validated_biomarkers.pop(0)
        current_token_count = token_count

# Process any remaining batch
if batch:
    call_model(batch)

# extract the canonical biomarkers from the grouped biomarkers and ask to an LLM to compare them and find synonyms