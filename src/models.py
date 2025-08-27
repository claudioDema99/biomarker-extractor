import json
import torch
import gc
from transformers import AutoTokenizer, AutoModelForCausalLM
from src.prompts import get_prompt
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
            torch_dtype=torch.bfloat16 if DEVICE=="cuda" else torch.float32,
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

def call_model(records, dataset_type, model, tokenizer, device, max_retries=5, max_full_retries=5, cooldown_minutes=5):
    """
    Call model with automatic chunking fallback on OOM errors
    
    Args:
        records: Lista dei record da processare
        dataset_type: Tipo di dataset per i prompt
        model, tokenizer, device: Componenti del modello
        max_retries: Tentativi per ogni batch/chunk (default: 5)
        max_full_retries: Tentativi completi con cooldown (default: 3)
        cooldown_minutes: Minuti di pausa tra tentativi completi (default: 5)
    """
    
    def _process_single_batch(record_batch, attempt_info=""):
        """Process a single batch of records"""
        for attempt in range(max_retries):
            try:
                # Clear memory before each attempt
                if attempt > 0:
                    clear_gpu_memory()
                    print(f"Memory cleared for retry {attempt + 1}{attempt_info}")
                
                # Adatto i prompt in base al database
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

                messages = [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ]
                
                full_prompt = tokenizer.apply_chat_template(
                    messages, 
                    tokenize=False, 
                    add_generation_prompt=True
                )
                
                inputs = tokenizer(
                    full_prompt, 
                    return_tensors="pt", 
                    padding=True, 
                    truncation=True,
                    max_length=8192
                ).to(device)
                
                # Check input size
                input_tokens = inputs.input_ids.shape[1]
                print(f"Processing {len(record_batch)} records, {input_tokens} input tokens{attempt_info}")
                
                with torch.no_grad():
                    outputs = model.generate(
                        **inputs,
                        max_new_tokens=4096,
                        do_sample=False,
                        pad_token_id=tokenizer.eos_token_id,
                        eos_token_id=tokenizer.eos_token_id,
                        use_cache=False,
                        repetition_penalty=1.1,
                        length_penalty=1.0
                    )

                # Cleanup immediato
                input_length = inputs.input_ids.shape[1]
                generated_tokens = outputs[0][input_length:]
                del inputs, outputs
                clear_gpu_memory()
                
                output_text = tokenizer.decode(
                    generated_tokens, 
                    skip_special_tokens=False,
                    clean_up_tokenization_spaces=True
                )

                # Parsing del template gpt-oss
                final_start = '<|end|><|start|>assistant<|channel|>final<|message|>'
                if final_start in output_text:
                    parts = output_text.split(final_start)
                    cot = final_start.join(parts[:-1])
                    response = parts[-1].strip()
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
                    print(f"No filtering tags found, returning full output.")

                # JSON parsing
                start = response.find('{')
                end = response.rfind('}') + 1

                if start == -1 or end == 0:
                    raise ValueError("JSON structure not found in response")

                json_str = response[start:end]
                data = json.loads(json_str)

                biomarkers = data.get("biomarkers", [])

                if not isinstance(biomarkers, list):
                    raise ValueError("Biomarkers is not a list")
                
                if not (cot and response):
                    raise ValueError("Missing CoT or response content")
                
                # Successo
                if attempt > 0:
                    print(f"Successo al tentativo {attempt + 1}{attempt_info}")
                
                return biomarkers, cot, response

            except torch.cuda.OutOfMemoryError as e:
                print(f"OOM Error - Tentativo {attempt + 1}/{max_retries}{attempt_info}: {e}")
                clear_gpu_memory()
                
                if attempt == max_retries - 1:
                    # Se è l'ultimo tentativo di questo batch, rilanciamo OOM per chunking
                    raise torch.cuda.OutOfMemoryError(f"OOM after {max_retries} attempts")
                else:
                    print("Cleaning memory and retrying...")
                    
            except Exception as e:
                print(f"Generic Error - Tentativo {attempt + 1}/{max_retries}{attempt_info}: {e}")
                
                if attempt == max_retries - 1:
                    print(f"Tutti i {max_retries} tentativi falliti per questo batch. Ritorno liste vuote.")
                    return [], "", ""
                else:
                    print("Riprovo...")

        return [], "", ""

    # === LOGICA PRINCIPALE CON RETRY GLOBALE ===
    
    import time
    
    for full_attempt in range(max_full_retries):
        full_attempt_info = f" (tentativo globale {full_attempt + 1}/{max_full_retries})"
        
        if full_attempt > 0:
            print(f"\nPausa di {cooldown_minutes} minuti per recovery del sistema...")
            print(f"Inizio pausa: {time.strftime('%H:%M:%S')}")
            time.sleep(cooldown_minutes * 60)  # Converti minuti in secondi
            print(f"Fine pausa: {time.strftime('%H:%M:%S')}")
            print(f"Riprovo dall'inizio{full_attempt_info}")
            
            # Clear aggressivo dopo la pausa
            clear_gpu_memory()
        
        try:
            # Prima prova con tutti i records
            print(f"Tentativo completo con {len(records)} records{full_attempt_info}")
            return _process_single_batch(records, full_attempt_info)
            
        except torch.cuda.OutOfMemoryError:
            print(f"\nOOM con {len(records)} records{full_attempt_info}. Provo chunking automatico...")
            clear_gpu_memory()
            
            # Strategia di chunking: prova dimensioni progressivamente più piccole
            chunk_sizes = [
                max(1, len(records) // 2),  # Metà
                max(1, len(records) // 4),  # Un quarto  
                max(1, len(records) // 8),  # Un ottavo
                1                           # Singolo record (last resort)
            ]
            
            chunking_success = False
            
            for chunk_size in chunk_sizes:
                if chunk_size >= len(records):
                    continue  # Skip se chunk_size è troppo grande
                    
                print(f"Provo chunking con dimensione: {chunk_size} records{full_attempt_info}")
                
                try:
                    # Dividi records in chunk
                    chunks = [records[i:i + chunk_size] for i in range(0, len(records), chunk_size)]
                    
                    all_biomarkers = []
                    all_cots = []
                    all_responses = []
                    
                    for idx, chunk in enumerate(chunks):
                        chunk_info = f" (chunk {idx + 1}/{len(chunks)}{full_attempt_info.rstrip(')')})"
                        print(f"Processing chunk {idx + 1}/{len(chunks)} with {len(chunk)} records{full_attempt_info}")
                        
                        try:
                            biomarkers, cot, response = _process_single_batch(chunk, chunk_info)
                            
                            if biomarkers:  # Solo se ha trovato biomarkers
                                all_biomarkers.extend(biomarkers)
                                all_cots.append(cot)
                                all_responses.append(response)
                                
                        except torch.cuda.OutOfMemoryError:
                            print(f"OOM anche con chunk size {chunk_size}{full_attempt_info}. Provo dimensione più piccola...")
                            raise  # Rilancia per provare chunk size più piccolo
                    
                    # Combina risultati
                    if all_biomarkers:
                        # Deduplica biomarkers mantenendo l'ordine
                        unique_biomarkers = []
                        seen = set()
                        for bio in all_biomarkers:
                            if bio not in seen:
                                unique_biomarkers.append(bio)
                                seen.add(bio)
                        
                        combined_cot = "\n\n--- CHUNK SEPARATOR ---\n\n".join(all_cots)
                        combined_response = "\n\n--- CHUNK SEPARATOR ---\n\n".join(all_responses)
                        
                        print(f"Chunking completato{full_attempt_info}! Trovati {len(unique_biomarkers)} biomarkers unici da {len(chunks)} chunks")
                        return unique_biomarkers, combined_cot, combined_response
                    else:
                        print(f"Nessun biomarker trovato con chunk size {chunk_size}{full_attempt_info}")
                        # Continua con chunk size più piccolo
                        raise torch.cuda.OutOfMemoryError("No biomarkers found")
                        
                except torch.cuda.OutOfMemoryError:
                    if chunk_size == 1:
                        print(f"OOM anche con singoli records{full_attempt_info}.")
                        break  # Esci dal loop dei chunk sizes
                    continue  # Prova chunk size più piccolo
            
            # Se arriviamo qui, tutti i chunk sizes hanno fallito per questo tentativo globale
            if full_attempt == max_full_retries - 1:
                print(f"Tutti i {max_full_retries} tentativi globali falliti. Impossibile processare.")
                return [], "", ""
            else:
                print(f"Tentativo globale {full_attempt + 1} fallito. Proverò di nuovo dopo la pausa...")
                continue  # Vai al prossimo tentativo globale
    
    # Backup finale (non dovrebbe mai essere raggiunto)
    print(f"Errore imprevisto: tutti i tentativi esauriti.")
    return [], "", ""