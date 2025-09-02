import json
import torch
import re
from collections import defaultdict

def read_json_arrays_to_list(filename):
    """
    Read a JSON file containing an array and return it as a list of dictionaries.
    """
    try:
        with open(filename, 'r', encoding='utf-8') as file:
            data = json.load(file)
            
            if isinstance(data, list):
                return data
            else:
                print(f"Warning: File does not contain a JSON array")
                return []
                
    except json.JSONDecodeError as e:
        print(f"Error parsing JSON: {e}")
        return []
    except FileNotFoundError as e:
        print(f"File not found: {e}")
        return []

def process_name(name):
    """
    Process a biomarker name for comparison:
    - Remove leading/trailing spaces
    - Remove special characters (_, -, +, etc.)
    - Convert to uppercase
    - Remove all numbers
    """
    # Remove leading/trailing spaces
    processed = name.strip()
    # Remove special characters and replace with empty string
    processed = re.sub(r'[_\-+\.\(\)\[\]\/\\:;,<>?|`~!@#$%^&*={}]', '', processed)
    # Remove all spaces (not just leading/trailing)
    processed = re.sub(r'\s+', '', processed)
    # Convert to uppercase
    processed = processed.upper()
    # Remove all numbers
    processed = re.sub(r'\d+', '', processed)
    return processed

def choose_canonical_name(group):
    """
    Choose the canonical name for a group.
    Priority: shortest name, then alphabetical
    """
    # Sort by length first, then alphabetically
    sorted_group = sorted(group, key=lambda x: (len(x), x))
    return sorted_group[0]

def call_model(biomarkers, task, model, tokenizer, device, biomarker_groups=[]):
    max_retries = 5
    for attempt in range(max_retries):
        try:
            if task == "grouping_biomarkers":
                system_prompt = """You are a biomarker term matcher. Your job is simple: group identical biomarkers that use different names, abbreviations, or spellings.

TASK: Look for synonym groups and output JSON immediately.
RULES:
- Include ALL input terms exactly once
- Group obvious synonyms (e.g., "P-Tau 181" = "pTau181" = "phospho-tau-181")  
- Use clear canonical names
- When unsure, keep terms separate
- Output ONLY valid JSON - no explanations

JSON FORMAT:
[
    {
        "canonical_biomarker": "Standard Name",
        "occurrences": ["variant1", "variant2"]
    }
]"""

                user_prompt = f"""Group these {len(biomarkers) if isinstance(biomarkers, list) else "N"} biomarker terms by synonyms. Output JSON only:

{biomarkers}"""
            elif task == "merging_groups":
                #canonical_biomarkers = [marker["canonical_biomarker"] for marker in biomarkers]
                system_prompt = """You are an expert in biomarkers and biomedical nomenclature. Your task is to assign ungrouped biomarkers to existing groups based exclusively on biological and biomedical synonymy.

FUNDAMENTAL RULES:
1. Assign a biomarker to a group ONLY if the biomarker is a biological synonym of the group name
2. Consider synonyms: alternative names of the same protein/molecule, standard abbreviations, different forms of the same compound
3. When in doubt, always choose NA rather than an uncertain assignment
4. Each biomarker must be assigned to ONE group or to NA
5. Maintain maximum scientific precision in assignments

OUTPUT FORMAT: Return a list of lists in the format:
[["biomarker_name", "assigned_group_name"]] or [["biomarker_name", "NA"]]

Be extremely conservative: it is better to leave a biomarker unassigned than to assign it incorrectly.
"""
                user_prompt = f"""I have 10 biomarkers to classify and 118 existing groups. I need to assign each biomarker to the appropriate group only if it is a true synonym of the group name.

**BIOMARKERS TO CLASSIFY:**
{biomarkers}

**AVAILABLE GROUPS:**
{biomarker_groups}

**INSTRUCTIONS:**
- Analyze each biomarker and verify if it is a synonym of one of the existing groups
- Assign the biomarker to a group only if it represents the same molecule/protein/compound
- If you don't find synonymous correspondence, assign "NA"
- Consider standard abbreviations, alternative names, ionic/derivative forms of the same molecule

**REQUIRED OUTPUT:**
List of lists in the format: [["biomarker1", "assigned_group"], ["biomarker2", "NA"], ...]

Proceed with the analysis and classification.
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
            
            if full_prompt.count(" medium") > 0:
                full_prompt = full_prompt.replace("medium", "low", 1)
                print("\n\nChanged reasoning level to low for better performance.\n")
            

            #print(full_prompt)
            #input()

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
            # Conteggio token effettivi
            token_count = inputs.input_ids.shape[1]
            print(f"Token count effettivi: {token_count}")
            print(f"--- Waiting for the model response ---")
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

            # Isola la sezione JSON anche se c'è testo extra
            start = response.find('[')
            end   = response.rfind(']') + 1   # rfind => ultima graffa

            if start == -1 or end == 0:
                raise ValueError("JSON array not found in response")

            json_str = response[start:end]
            data = json.loads(json_str)

            if not isinstance(data, list) or len(data) == 0:
                raise ValueError("Response is not a valid non-empty list")
            
            if attempt > 0:
                print(f"Successo al tentativo {attempt + 1}")
            return data

        except Exception as e:
            print(f"Tentativo {attempt + 1}/{max_retries} fallito: {e}")
            
            if attempt == max_retries - 1:
                print(f"Tutti i {max_retries} tentativi falliti. Ritorno lista vuota.")
                return []
            else:
                print("Riprovo...")

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

def aggregation(model, tokenizer, device, evaluated_biomarkers):
    
    if evaluated_biomarkers == [] or evaluated_biomarkers == None:
        filename = "./results/evaluated_biomarkers.json"
        evaluated_biomarkers = read_json_arrays_to_list(filename)

    validated_biomarkers = []
    for acronym in evaluated_biomarkers:
        if acronym["valid"] == True:
            validated_biomarkers.append(acronym["acronym"])

    print(f"Valid biomarkers: {len(validated_biomarkers)} on {len(evaluated_biomarkers)} total biomarkers extracted.")

    # PRIMA FASE AGGREGAZIONE: exact matching degli acronimi (togliendo caratteri speciali e numeri)
    # Dictionary to group items by their processed name
    groups = defaultdict(list)

    # Process each item and group by processed name
    for biomarker in validated_biomarkers:
        clean_biomarker = process_name(biomarker)
        if clean_biomarker:  # Only add if processed name is not empty
            # se esiste già 'processed' aggiungo alla lista già esistente, così raggruppo i duplicati
            # se non esiste creo una nuova lista con il primo elemento
            groups[clean_biomarker].append(biomarker)

    # Separate groups (duplicates) from single items
    duplicate_groups = {}
    remaining_items = []
    for processed_name, original_names in groups.items():
        if len(original_names) > 1:  # Group has duplicates
            duplicate_groups[processed_name] = original_names
        else:  # Single item
            remaining_items.extend(original_names)

    # FOR DEBUGGING
    #print("\n DETAILED GROUPING:")
    total_grouped_items = 0
    for processed_name, original_names in duplicate_groups.items():
        #print(f"Processed name: '{processed_name}' → {original_names}")
        total_grouped_items += len(original_names)
    #print(f"\nRemaining items: {remaining_items}")
    # Verify conservation
    print(f"\nExact matching finito:")
    print(f"Initial items: {len(validated_biomarkers)}")
    print(f"Grouped items: {total_grouped_items}")
    print(f"Remaining items: {len(remaining_items)}")
    #print(f"Total: {total_grouped_items + len(remaining_items)}")
    #print(f"Conservation check: {'✅ OK' if total_grouped_items + len(remaining_items) == len(validated_biomarkers) else '❌ ERROR'}")

    parsed_biomarkers = []
    for processed_name, original_names in duplicate_groups.items():
        canonical = choose_canonical_name(original_names)
        group_entry = {
            "canonical_biomarker": canonical,
            "occurrences": sorted(original_names)  # Sort members alphabetically
        }
        parsed_biomarkers.append(group_entry)

    # SECONDA FASE AGGREGAZIONE: processiamo i restanti (non aggregati) biomarkers con LLM provando a farli assegnare a qualche gruppo creato
    groups = []
    for parsed in parsed_biomarkers:
        groups.append(parsed["canonical_biomarker"])

    # chiediamo all'LLM se i biomarkers che non sono stati assegnati a nessun gruppo durante il exact match parsing, appartengono a uno dei gruppi definiti 
    # in quanto sinonimi, varianti di acronimi e/o nomenclatura
    couple_matches = []
    for i in range(0, len(remaining_items), 10):
        # assegnamo un gruppo agli elementi singoli oppure "NA" se non esiste un gruppo corrispondente
        couple_matches.extend(call_model(remaining_items[i:i+10], "merging_groups", model, tokenizer, device, biomarker_groups=groups))

    # aggiungiamo ai gruppi i biomarkers assegnati dall'LLM
    for biomarker, group in couple_matches:
        # Skip if group is "NA"
        if group == "NA":
            continue
        # Find matching biomarker in parsed_biomarkers and add group to occurrences
        for biomarker_dict in parsed_biomarkers:
            if biomarker_dict.get("canonical_biomarker") == group:                
                # Add the group to occurrences
                biomarker_dict["occurrences"].append(biomarker)
                break  # Found the match, no need to continue searching

    # Aggiungi count e ordina in modo decrescente
    for group in parsed_biomarkers:
        group["count"] = len(group["occurrences"])
    final_biomarkers_sorted = sorted(parsed_biomarkers, key=lambda x: x['count'], reverse=True)

    with open("./results/biomarkers.json", "w", encoding="utf-8") as f:
        json.dump(final_biomarkers_sorted, f, ensure_ascii=False, indent=2)
    return final_biomarkers_sorted