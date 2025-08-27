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

def call_model(biomarkers, task):
    # SYSTEM + USER PROMPTS TOKEN COUNT = 788
    if task == "grouping_biomarkers":
        system_prompt = """You are an expert biomedical data curator with deep knowledge of Alzheimer's disease pathophysiology and biomarkers. Your primary function is to process lists of biomarker terms, and group them by their underlying biological significance.

    CRITICAL ACCURACY REQUIREMENTS:
    - You MUST NOT invent, create, or add any biomarkers that are not explicitly provided in the input list
    - You MUST NOT omit, skip, or forget any biomarkers from the input list
    - Every single biomarker term from the input MUST appear exactly once in the "occurrences" arrays
    - If you are uncertain about a biomarker's identity or grouping, you MUST still include it (use "Other" category if needed)
    - DOUBLE-CHECK: Before finalizing your response, verify that every input term appears exactly once in your output

    Rules for Operation:
    Core Task: You will receive lists of raw biomarker strings. You must aggregate synonymous terms, even if they have different spellings, abbreviations, or syntactic structures (e.g., "P-Tau 181", "pTau181", "phosphorylated tau at threonine 181").
    Canonical Naming: For each group of synonyms, you must assign a single, scientifically precise, and human-readable canonical name (e.g., "P-Tau 181").
    Output Format: You will always output a valid JSON object. No other text, explanations, or apologies should precede or follow the JSON code block.
    Handling Uncertainty: If you encounter a highly ambiguous term, group it cautiously and use the "Other" category. It is more important to be accurate than to force a categorization. NEVER omit uncertain terms - include them somewhere.
    Focus: Do not provide commentary, summaries, or analysis beyond the requested JSON structure. Your entire output must be the JSON object.

    MANDATORY VERIFICATION STEPS:
    1. Count the total number of input biomarkers
    2. Count the total number of biomarkers in all "occurrences" arrays combined
    3. These numbers MUST be identical
    4. If they don't match, you have made an error and must correct it
    """
        user_prompt = f"""Please process the following list of raw Alzheimer's disease biomarker terms. Aggregate synonymous terms, and provide a list of all occurrences and their count.

    CRITICAL INSTRUCTIONS - READ CAREFULLY:
    - You MUST include EVERY SINGLE biomarker from the input list in your output
    - You MUST NOT add any biomarkers that are not in the input list
    - If you are unsure about a biomarker, include it anyway (use "Other" category if needed)
    - MANDATORY: Before providing your final answer, mentally count that all input biomarkers appear exactly once in your "occurrences" arrays

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
            "occurrences": ["Amyloid Beta 42/40 ratio"]
        }}
    ]

    INPUT COUNT: {len(biomarkers) if isinstance(biomarkers, list) else "Please count the biomarkers below"}

    Here is the full list of biomarkers to process:

    {biomarkers}
    """
    elif task == "merging_groups":
        canonical_biomarkers = [marker["canonical_biomarker"] for marker in biomarkers]
        system_prompt = """You are an expert biomedical data curator specialized in Alzheimer's disease biomarkers. Your task is to identify groups of canonical biomarker names that represent the same biological entity and should be merged together.

CRITICAL ACCURACY REQUIREMENTS:
- You MUST analyze ONLY the canonical biomarker names provided in the input list
- You MUST NOT invent, add, or reference any biomarker names not in the input
- You MUST NOT skip or omit any biomarker from your analysis
- If you are uncertain whether two biomarkers should be merged, err on the side of caution and keep them separate
- MANDATORY: Every input biomarker must be accounted for in your output (either as a standalone group or part of a merge group)

CORE PRINCIPLES:
- Merge groups only when biomarkers represent the EXACT SAME biological measurement
- Consider different naming conventions, abbreviations, and synonyms (e.g., "P-Tau 181" and "Phosphorylated Tau 181")
- Do NOT merge biomarkers that are related but measure different things (e.g., "Aβ40" vs "Aβ42" vs "Aβ42/40 ratio")
- When in doubt, keep separate rather than incorrectly merging

OUTPUT FORMAT:
- Return only a valid JSON array
- Each element represents a group of biomarkers to be merged
- Use 0-based indices corresponding to the input list positions
- Single biomarkers that don't need merging should appear as single-element arrays
- No explanations, comments, or additional text outside the JSON

VERIFICATION REQUIREMENTS:
- Every index from 0 to (input_length-1) must appear exactly once across all merge groups
- No index should be duplicated
- No index should be omitted"""

        user_prompt = f"""Analyze the following list of canonical biomarker names and identify which groups should be merged because they represent the same biological measurement.

INSTRUCTIONS:
1. Examine each biomarker name for synonyms, abbreviations, or alternative naming conventions
2. Group biomarkers that measure the EXACT SAME biological entity
3. Return the 0-based indices of biomarkers that should be merged together
4. Each biomarker must appear in exactly one group (even if it's a group of one)

INPUT VALIDATION:
- Total biomarkers to analyze: {len(canonical_biomarkers)}
- You must account for ALL biomarkers from index 0 to {len(canonical_biomarkers)-1}

EXPECTED OUTPUT FORMAT:
[
  [0, 1, 3],     // Indices of biomarkers to merge together
  [2],           // Single biomarker (no merging needed)
  [4, 7],        // Another group to merge
  [5],           // Another standalone biomarker
  [6]            // And so on...
]

CANONICAL BIOMARKERS TO ANALYZE:
{canonical_biomarkers}

Remember: Return ONLY the JSON array of index groups. No additional text.
"""
    else:
        return []

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
    print(f"Token count effettivi: {token_count}")
    print(f"--- Waiting for the {MODEL_NAME} response ---")
    # Generazione della risposta
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=4096,          # Massimo numero di nuovi token da generare
            do_sample=False,              # Usa greedy decoding (deterministic!!!)
            #temperature=0.7,              # Controllo randomness (se do_sample=True) quindi inutile
            #top_p=0.9,                    # Nucleus sampling (se do_sample=True) quindi inutile
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
    print(response)

    try:
        # Isola la sezione JSON anche se c'è testo extra
        start = response.find('[')
        end   = response.rfind(']') + 1   # rfind => ultima graffa

        if start == -1 or end == 0:
            return [],

        json_str = response[start:end]

        # Carica il JSON
        data = json.loads(json_str)

        # Garantisci che sia effettivamente una lista, altrimenti torna lista vuota
        if isinstance(data, list) and len(data) > 0:
            return data
        else:
            return []

    except (json.JSONDecodeError, TypeError) as e:
        print(f"Errore nell'estrazione dei biomarkers: {e}")
        return []

def merge_biomarker_groups(biomarkers, merging_indexes):
    """
    Merge biomarker groups based on the indices provided by the LLM.
    When merging, choose the best canonical name from the group.
    
    Args:
        biomarkers: List of dictionaries, each containing 'canonical_biomarker' and 'occurrences'
        merging_indexes: List of lists, where each inner list contains indices to merge
    
    Returns:
        List of merged biomarker dictionaries
    """
    merged_biomarkers = []
    
    for index_group in merging_indexes:
        if len(index_group) == 1:
            # Single biomarker, no merging needed
            merged_biomarkers.append(biomarkers[index_group[0]])
        else:
            # Multiple biomarkers to merge
            # Choose the best canonical name (you can customize this logic)
            canonical_names = [biomarkers[idx]["canonical_biomarker"] for idx in index_group]
            
            # Strategy 1: Choose the shortest name
            best_canonical = min(canonical_names, key=len)

            # Combine all occurrences from the groups
            combined_occurrences = []
            for idx in index_group:
                combined_occurrences.extend(biomarkers[idx]["occurrences"])
            
            # Create the merged group
            merged_group = {
                "canonical_biomarker": best_canonical,
                "occurrences": combined_occurrences
            }
            merged_biomarkers.append(merged_group)
    
    return merged_biomarkers

filename = "risultati_acronyms.txt" #"./results/acronym_results.txt"
acronyms = read_json_arrays_to_list(filename)

print(f"Total dictionaries: {len(acronyms)}")
print(f"First few items: {acronyms[:5]}")

validated_biomarkers = []
for acronym in acronyms:
    if acronym["valid"] == True:
        validated_biomarkers.append(acronym["acronym"])

print(f"Total valid biomarkers: {len(validated_biomarkers)} on {len(acronyms)} total biomarkers")

# try to add biomarkers until a max token count of 7000 is reached
batch = []
batch_storage = []
current_token_count = 0
max_token_count = 300 #6000
grouped_biomarkers = []

while validated_biomarkers:
    batch = validated_biomarkers[:20]      # prendi fino a 20 biomarcatori
    del validated_biomarkers[:20]

    tokenized = tokenizer(
        batch,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=8192
    ).to(DEVICE)

    batch_size, seq_len = tokenized.input_ids.shape
    token_count = batch_size * seq_len
    print(f"Current token count: {current_token_count + token_count}")

    if current_token_count + token_count <= max_token_count:
        print("Adding")
        batch_storage.append(batch)
        current_token_count += token_count
    else:
        print(f"Processing batch of {len(batch_storage)} biomarkers with total token count {current_token_count}...")
        #input("Press Enter to continue...")
        grouped_biomarkers.extend(call_model(batch_storage, "grouping_biomarkers"))
        batch_storage = [batch]  # inizia nuovo batch
        current_token_count = token_count


# Process any remaining batch
if batch:
    print(f"Processing last batch of {len(batch_storage)} biomarkers with total token count {current_token_count}...")
    grouped_biomarkers.extend(call_model(batch_storage, "grouping_biomarkers"))

print(grouped_biomarkers)

print(f"\n\nNumber of groups (not merged): {len(grouped_biomarkers)}")
merging_indexes = call_model(grouped_biomarkers, "merging_groups")
print(merging_indexes)

final_grouped_biomarkers = merge_biomarker_groups(grouped_biomarkers, merging_indexes)
print(final_grouped_biomarkers)

# se c'è errore nel model response, riprova (dopo 5 tentativi skippa ma tieni traccia)
