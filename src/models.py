import json
import torch
import gc
from transformers import AutoTokenizer, AutoModelForCausalLM
from src.prompts import get_prompt, get_system_user_prompt
import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

def clear_gpu_memory():
    """Clear GPU memory and force garbage collection"""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()

def load_model_and_tokenizer():

    clear_gpu_memory()

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

    try:
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
            dtype=torch.bfloat16 if DEVICE=="cuda" else torch.float32,
            device_map="auto",
            trust_remote_code=True,
            low_cpu_mem_usage=True
        )

        # Set pad token per gpt-oss
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        return model, tokenizer, DEVICE

    except torch.cuda.OutOfMemoryError as e:
        print(f" Out of memory durante il caricamento: {e}")
        clear_gpu_memory()
        raise

def get_token_count(records_json, tokenizer):
    return len(tokenizer.encode(json.dumps(records_json)))

def call_model(task: str, dataset_type: str, model: AutoModelForCausalLM, tokenizer: AutoTokenizer, device: str, records: str="", biomarkers: str="", low_reasoning: bool=False, half_shots: bool=False):
    max_retries=5
    for attempt in range(max_retries):
        try:
            # Clear memory before each attempt
            if attempt > 0:
                clear_gpu_memory()
                print(f" Memory cleared for retry {attempt + 1}")
            # Build the system and user prompt
            name, full_name, examples, shots = get_prompt(dataset_type)
            system_prompt, user_prompt = get_system_user_prompt(task=task, name=name, full_name=full_name, examples=examples, shots=shots, records=records, biomarkers=biomarkers)
            
            if half_shots:
                split_prompt = "3. Input record:"
                half_shots = shots.split(split_prompt)
                shots = half_shots[0]

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
            if low_reasoning:
                if full_prompt.count(" medium") > 0:
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
            ).to(device)

            print(f"--- Waiting for the model response ---")
            # Generazione della risposta
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=4096,          # Massimo numero di nuovi token da generare
                    do_sample=False,              # Usa greedy decoding (deterministic!!!)
                    #temperature=0.7,             # Controllo randomness (se do_sample=True) quindi inutile
                    #top_p=0.9,                   # Nucleus sampling (se do_sample=True) quindi inutile
                    pad_token_id=tokenizer.eos_token_id,  # Token di padding
                    eos_token_id=tokenizer.eos_token_id,  # Token di fine sequenza
                    use_cache=True,               # usiamola
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

            # chat template di gpt-oss: dividiamo CoT da risposta
            final_start = '<|end|><|start|>assistant<|channel|>' #final<|message|>'
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
            
            if task == "groups":
                # Isola la sezione JSON anche se c'è testo extra
                start = response.find('[')
                end   = response.rfind(']') + 1   # rfind => ultima quadra
            else:
                # Isola la sezione JSON anche se c'è testo extra
                start = response.find('{')
                end   = response.rfind('}') + 1   # rfind => ultima graffa

            if start == -1 or end == 0:
                raise ValueError("JSON structure not found in response")

            json_str = response[start:end]
            data = json.loads(json_str)
            
            if not cot and not response:
                raise ValueError("Missing CoT and response content")
            
            # Se arriviamo qui, tutto è andato bene
            if attempt > 0:
                print(f" Successo al tentativo {attempt + 1}")
            
            return data, cot, response

        except torch.cuda.OutOfMemoryError as e:
            print(f" OOM Error - Tentativo {attempt + 1}/{max_retries}: {e}")
            # reduce the shots because of not enough memory 
            print("Because of the input too long, we reduce the shots from 5 to 2.")
            half_shots = True
            clear_gpu_memory()
            
            if attempt == max_retries - 1:
                print(f" Tutti i {max_retries} tentativi falliti per OOM. Ritorno liste vuote.")
                return None, "", ""
            else:
                print(" Cleaning memory and retrying...")
                
        except Exception as e:
            print(f" Generic Error - Tentativo {attempt + 1}/{max_retries}: {e}")
            clear_gpu_memory()
            
            if attempt == max_retries - 1:
                print(f" Tutti i {max_retries} tentativi falliti. Ritorno liste vuote.")
                return None, "", response
            else:
                print(" Riprovo...")

    return [], "", ""

def call_model_for_unprocessed_lines(analysis, dataset_type, model, tokenizer, device, max_retries=5, low_reasoning=False, half_shots = False):
    for attempt in range(max_retries):
        try:
            # Clear memory before each attempt
            if attempt > 0:
                clear_gpu_memory()
                print(f" Memory cleared for retry {attempt + 1}")
            # Adatto i prompt in base al database
            name, full_name, examples, shots = get_prompt(dataset_type)
            
            if half_shots:
                split_prompt = "3. Input record:"
                half_shots = shots.split(split_prompt)
                shots = half_shots[0]

            system_prompt = f"""You are an expert clinical data analyst specialized in identifying markers of {full_name}'s disease in clinical trials. 
            
MARKERS DEFINITION: The markers are the quantitative criteria and features (biological, molecular, clinical, imaging, histological, genetical) that are used to diagnose {name}'s disease or to evaluate pathological symptoms and conditions related to {name}'s disease.            
            
Your task: extract ONLY markers explicitly present in the analysis I give you. Do NOT invent markers not present in the analysis. You have to include techniques or modalities, but only if a marker extracted from these techniques is explicitly present in the text. Example: Magnetic resonance imaging (MRI) is a technique, include it if hippocampal volume (or another quantity extracted from MRI) is present as a marker; Electromiography (EMG) is a technique, include it if affect-modulated startle (AMS) or startle eye-blink is present in the text as a marker. 

MANDATORY OUTPUT RULES (must be followed exactly):
1. Output **exactly one** JSON object and nothing else (no surrounding text, no code fences). The JSON must have one key:
{{"markers": [list of marker with required syntax]}}
2. "markers" must be a JSON array. Each element in the array MUST follow this exact syntax:
"ACRONYM: expanded form of the acronym (or brief description if no acronym exists)"
**CRITICAL**: ALWAYS use the ACRONYM (in UPPERCASE) before the colon when available. Extract acronyms from text even if they appear in parentheses after full names. If no acronym exists, create a logical abbreviation or use the shortest recognizable form.
Examples:{examples}
4. Collapse duplicates (each marker appears once).
5. If you cannot follow these rules, output exactly:
{{"markers":[]}}
Always reason with clinical rigor and refer only to evidence in the records.
"""

            user_prompt = f"""You are an expert clinical data analyst specialized in identifying markers in {full_name}'s disease analysis. Your task: extract ONLY markers explicitly present in this analysis and produce output **only** the exact JSON object required in the prompt.
. Do NOT invent markers not present in the analysis.
Follow the structure and formatting style shown in the examples exactly, and apply it only to the new input records.

Input analysis:
{json.dumps(analysis)}

Output:
"""

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
            if low_reasoning:
                if full_prompt.count(" medium") > 0:
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
            ).to(device)

            print(f"--- Waiting for the model response ---")
            # Generazione della risposta
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=4096,          # Massimo numero di nuovi token da generare
                    do_sample=False,              # Usa greedy decoding (deterministic!!!)
                    #temperature=0.7,             # Controllo randomness (se do_sample=True) quindi inutile
                    #top_p=0.9,                   # Nucleus sampling (se do_sample=True) quindi inutile
                    pad_token_id=tokenizer.eos_token_id,  # Token di padding
                    eos_token_id=tokenizer.eos_token_id,  # Token di fine sequenza
                    use_cache=True,               # usiamola
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

            # chat template di gpt-oss: dividiamo CoT da risposta
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
            elif final_start[:36] in output_text:
                parts = output_text.split(final_start[:36])
                cot = final_start[:36].join(parts[:-1])  # Tutto prima del tag finale
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

            # Isola la sezione JSON anche se c'è testo extra
            start = response.find('{')
            end   = response.rfind('}') + 1   # rfind => ultima graffa

            if start == -1 or end == 0:
                raise ValueError("JSON structure not found in response")

            json_str = response[start:end]
            data = json.loads(json_str)

            # Estrai la lista biomarkers, se esiste
            biomarkers = data.get("markers", None)

            if not isinstance(biomarkers, list):
                raise ValueError("Biomarkers is not a list")
            
            if not cot and not response:
                raise ValueError("Missing CoT and response content")
            
            # Se arriviamo qui, tutto è andato bene
            if attempt > 0:
                print(f" Successo al tentativo {attempt + 1}")
            
            return biomarkers, cot, response

        except torch.cuda.OutOfMemoryError as e:
            print(f" OOM Error - Tentativo {attempt + 1}/{max_retries}: {e}")
            # reduce the shots because of not enough memory 
            print("Because of the input too long, we reduce the shots from 5 to 2.")
            half_shots = True
            clear_gpu_memory()
            
            if attempt == max_retries - 1:
                print(f" Tutti i {max_retries} tentativi falliti per OOM. Ritorno liste vuote.")
                return None, "", ""
            else:
                print(" Cleaning memory and retrying...")
                
        except Exception as e:
            print(f" Generic Error - Tentativo {attempt + 1}/{max_retries}: {e}")
            clear_gpu_memory()
            
            if attempt == max_retries - 1:
                print(f" Tutti i {max_retries} tentativi falliti. Ritorno liste vuote.")
                return None, "", response
            else:
                print(" Riprovo...")

    return [], "", ""

def calculate_prompt_tokens(tokenizer, dataset_type: str="Alzheimer"):
    name, full_name, examples, shots = get_prompt(dataset_type)

    system_prompt = f"""You are an expert clinical data analyst specialized in identifying markers of {full_name}'s disease in clinical trials.

MARKERS DEFINITION: The markers are the quantitative criteria and features (biological, molecular, clinical, imaging, histological) that are used to diagnose {name}'s disease or to evaluate pathological symptoms and conditions related to {name}'s disease.

Your task: analyze the provided records and extract ONLY markers explicitly present in the text. Do NOT invent markers not present in the records. Do NOT include techniques or modalities, but you can include quantities extracted from them (example: MRI is a technique, then do not include it; hippocampal volume derived from MRI is a quantity, then include it). 

MANDATORY OUTPUT RULES (must be followed exactly):
1. Output **exactly one** JSON object and nothing else (no surrounding text, no code fences). The JSON must have two keys:
{{"analysis": "<4-5 concise sentences>", "markers": [list of marker with required syntax]}}
2. "analysis" must be a single string of 4–5 sentences that reference the evidence in the records.
3. "markers" must be a JSON array. Each element in the array MUST follow this exact syntax:
"ACRONYM: expanded form of the acronym (or brief description if no acronym exists)"
**CRITICAL**: ALWAYS use the ACRONYM (in UPPERCASE) before the colon when available. Extract acronyms from text even if they appear in parentheses after full names. If no acronym exists, create a logical abbreviation or use the shortest recognizable form.
Examples:{examples}
4. Collapse duplicates (each marker appears once).
5. If you cannot follow these rules, output exactly:
{{"analysis":"", "markers":[]}}
Always reason with clinical rigor and refer only to evidence in the records.
"""

    user_prompt = f"""You are an expert clinical data analyst specialized in identifying markers in {full_name}'s disease clinical trials. Your task: analyze the provided records and extract ONLY markers explicitly present in the text. Do NOT invent markers not present in the records.
Follow the structure and formatting style shown in the examples exactly, and apply it only to the new input records.

Examples:
{shots}   

Task: Identify all markers explicitly present in the Records and produce output **only** the exact JSON object required in the prompt.

Input records:

Output:
"""
    return get_token_count(system_prompt + user_prompt, tokenizer)
