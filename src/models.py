import json
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from src.prompts import get_prompt

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

# Set pad token per gpt-oss
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

def get_token_count(records_json):
    return len(tokenizer.encode(json.dumps(records_json)))

def call_model(records, dataset_type):
    # QUI MI SCELGO I PROMPT IN BASE AL DATASET_TYPE
    name, examples, shots = get_prompt(dataset_type)
    system_prompt = f"""You are an expert clinical data analyst specialized in identifying biomarkers in {name}'s disease clinical trials. Your task: analyze the provided records and extract ONLY biomarkers explicitly present in the text. Do NOT invent biomarkers not present in the records.

BIOMARKER DEFINITION: A biomarker is a quantifiable characteristic of the body that serves as an objective indicator of biological processes or pathological conditions in {name}'s disease.

MANDATORY OUTPUT RULES (must be followed exactly):
1. Output **exactly one** JSON object and nothing else (no surrounding text, no code fences). The JSON must have two keys:
   {{"analysis": "<4-5 concise sentences>", "biomarkers": [list of biomarker with required syntax]}}
2. "analysis" must be a single string of 4–5 sentences that reference the evidence in the records.
3. "biomarkers" must be a JSON array. Each element in the array MUST follow this exact syntax:
   "ACRONYM: expanded form of the acronym (or brief description if no acronym exists)"
   **CRITICAL**: ALWAYS use the ACRONYM (in UPPERCASE) before the colon when available. Extract acronyms from text even if they appear in parentheses after full names. If no acronym exists, create a logical abbreviation or use the shortest recognizable form.
   Examples:{examples}
4. Collapse duplicates (each biomarker appears once).
5. If you cannot follow these rules, output exactly:
   {{"analysis":"", "biomarkers":[]}}
Always reason with clinical rigor and refer only to evidence in the records.
"""

    user_prompt = f"""You are an expert clinical data analyst specialized in identifying biomarkers in {name}'s disease clinical trials. Your task: analyze the provided records and extract ONLY biomarkers explicitly present in the text. Do NOT invent biomarkers not present in the records.
Follow the structure and formatting style shown in the examples exactly, and apply it only to the new input records.

Examples:
{shots}   

Task: Identify all biomarkers explicitly present in the Records and produce output **only** the exact JSON object required in the prompt.

Input records:
{json.dumps(records)}

Output:
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

    # se vuoi impostare il reasoning su low invece che medium
    if full_prompt.count(" medium ") > 0:
        full_prompt = full_prompt.replace("medium", "low", 1)
        print("\n\nChanged reasoning level to low for better performance.\n")

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
        start = response.find('{')
        end   = response.rfind('}') + 1   # rfind => ultima graffa

        if start == -1 or end == 0:
            return [], "", ""                        # nessuna struttura JSON trovata

        json_str = response[start:end]

        # Carica il JSON
        data = json.loads(json_str)

        # Estrai la lista biomarkers, se esiste
        biomarkers = data.get("biomarkers", [])
        # Garantisci che sia effettivamente una lista, altrimenti torna lista vuota
        if isinstance(biomarkers, list) and cot != "" and response != "":
            return biomarkers, cot, response 
        else:
            return [], "", ""

    except (json.JSONDecodeError, TypeError) as e:
        print(f"Errore nell'estrazione dei biomarkers: {e}")
        return [], "", ""
