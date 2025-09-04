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
    - Remove specific sequences: "CSF", "PET", "MRI"
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
    # Remove specific sequences: CSF, PET, MRI
    processed = re.sub(r'CSF|PET|MRI', '', processed)
    return processed

def choose_canonical_name(group):
    """
    Choose the canonical name for a group.
    Priority: shortest name, then alphabetical
    """
    # Sort by length first, then alphabetically
    sorted_group = sorted(group, key=lambda x: (len(x), x))
    return sorted_group[0]

def merge_groups(parsed_biomarkers, actual_correlations):
    # Crea una copia della lista originale per non modificarla
    result = parsed_biomarkers.copy()
    
    # Ordina gli indici da eliminare in ordine decrescente
    # così eliminiamo prima quelli con indici più alti
    indices_to_remove = []
    
    for correlation_group in actual_correlations:
        if len(correlation_group) <= 1:
            continue  # Skip se c'è solo un elemento o nessuno
            
        # Il primo indice sarà quello che manteniamo (target)
        target_idx = correlation_group[0]
        
        # Unisci tutte le occurrences nel dizionario target
        for idx in correlation_group[1:]:
            result[target_idx]["occurrences"].extend(result[idx]["occurrences"])
            indices_to_remove.append(idx)
    
    # Rimuovi i duplicati e ordina in ordine decrescente
    indices_to_remove = sorted(set(indices_to_remove), reverse=True)
    
    # Elimina gli elementi partendo dagli indici più alti
    for idx in indices_to_remove:
        result.pop(idx)
    
    return result

def call_model(biomarkers, task, model, tokenizer, device, biomarker_groups=[]):
    max_retries = 5
    for attempt in range(max_retries):
        try:
            if task == "defining_variants":
                system_prompt = """You are a marker variant grouper. Task: group occurrences by variant type.

DEFINITION
- Variant
  - Molecules: same base marker with a numeric/letter form (e.g., Aβ1-37/1-40/1-42, p-tau181/217, ApoE2/3/4, GFAP-δ).
  - Tests/scales: same base test with a version/subscale tag (e.g., ADAS-Cog-11/13/14, v2, SF/LF, CDR-SB/SoB)

RULES:
1. Find the base marker name in each occurrence
2. Group by variant indicators (numbers, letters, versions)
3. Use first occurrence as group key
4. Output JSON only - no explanations

EXAMPLE OF VARIANT INDICATORS:
- Numbers: 1-40, 1-42, 181, 217
- Letters: δ, α, β
- Versions: v2, -11, -13, SF, LF

OUTPUT FORMAT:
{"variants": {"key1": ["item1", "item2"], "key2": ["item3"]}}

No text before or after JSON. No long reasoning. Just JSON."""
                user_prompt = f"""Group these marker occurrences by variant. Output JSON only.
Note: It is not certain or mandatory that all groups actually have variants. If all occurrences represent the same variant or no clear variants exist, return them all grouped under a single variant key.

Example of the task:
Input:
{{
  "canonical_biomarker": "ABETA",
  "occurrences": [
    "ABETA",
    "ABETA1-40",
    "ABeta 1-38",
    "Abeta 1-42",
    "Abeta-40",
    "Abeta40",
    "Abeta42",
    "Abeta42",
    "Abeta42"
  ]
}}
Output:
{{
  "variants": {{
    "ABETA": ["ABETA"],
    "ABETA1-40": ["ABETA1-40", "Abeta-40", "Abeta40"],
    "ABeta 1-38": ["ABeta 1-38"],
    "Abeta 1-42": ["Abeta 1-42", "Abeta42", "Abeta42", "Abeta42"]
  }}
}}

Here is the input you have to process:
{biomarkers}

Output:"""
            elif task == "merging_groups":
                #canonical_biomarkers = [marker["canonical_biomarker"] for marker in biomarkers]
                system_prompt = """You are a data analysis assistant specialized in identifying potential correlations between biological/scientific markers. Your task is to analyze canonical marker names and identify groups that could be semantically related.

**Guidelines:**
- Use your knowledge of scientific terminology, biological pathways, and semantic relationships
- Consider synonyms, abbreviations, different naming conventions and nomenclature
- Avoid random or speculative groupings - only group when confident of the relationship
- Maximum 5 markers per group (groups with >5 members are highly unlikely)
- No too long reasoning

**Output Format:**
- Return only a list of lists containing marker indices
- Each inner list represents a group of potentially related markers
- Use zero-based indexing (0, 1, 2, ...)
- No explanations, comments, or additional text
"""
                user_prompt = f"""The following canonical marker names were identified from grouped datasets using exact matching:

{biomarkers}

Analyze these markers and identify any groups that could be correlated or linked to each other. Return the indices of potentially related marker groups.
**Important:** Only group markers when you are confident they are related.

**Example:**
If your marker list is:
['CDR', 'ADAS-Cog', 'Aβ', 'NPI', 'ABETA', 'Aβ1‑40', ...]

Then indices 2, 4, and 5 should be grouped together as they all refer to amyloid-beta variants:
- Index 2: 'Aβ' (amyloid-beta)
- Index 4: 'ABETA' (amyloid-beta, full name)
- Index 5: 'Aβ1‑40' (amyloid-beta fragment)

Output: [[2, 4, 5]]

**Response:**
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

            '''
            if full_prompt.count(" medium") > 0:
                full_prompt = full_prompt.replace("medium", "low", 1)
                print("\n\nChanged reasoning level to low for better performance.\n")
            '''
            

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
                    max_new_tokens=8192,          # Massimo numero di nuovi token da generare
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

            if task == "defining_variants":
                print(response)
                #input()
                #output: {"variants":{"CDR":["CDR","CDR","CDR"],"CDR‑SB":["CDR‑SB","CDR‑SB","CDR‑SB","CDR‑SB","CDR‑SB","CDR‑SB","CDR‑SB","CDR‑SB","CDR‑SB","CDR‑SB","CDR‑SB","CDR‑SB","CDR-SB","CDR-SB","CDR-SB"]}}
                # Isola la sezione JSON anche se c'è testo extra
                start = response.find('{')
                end   = response.rfind('}') + 1   # rfind => ultima graffa
            elif task == "merging_groups":
                # Isola la sezione JSON anche se c'è testo extra
                start = response.find('[')
                end   = response.rfind(']') + 1   # rfind => ultima quadra

            if start == -1 or end == 0:
                raise ValueError("JSON array not found in response")

            json_str = response[start:end]
            data = json.loads(json_str)

            if (not isinstance(data, list) or len(data) == 0) and task == "merging_groups":
                raise ValueError("Response is not a valid non-empty list")
            elif not isinstance(data, dict) and task == "defining_variants":
                raise ValueError("Response is not a valid dict")
            
            if attempt > 0:
                print(f"Successo al tentativo {attempt + 1}")
            return data

        except Exception as e:
            print(f"Tentativo {attempt + 1}/{max_retries} fallito: {e}")
            # COMMENTA POI NEXT LINE
            print(response)
            
            if attempt == max_retries - 1:
                print(f"Tutti i {max_retries} tentativi falliti. Ritorno lista vuota.")
                return []
            else:
                print("Riprovo...")

    return []

def merge_biomarker_groups(biomarkers, merging_indexes):
    """
    Merge specified biomarker groups and keep unmerged ones.
    """
    # Track which indices are being merged
    indices_to_merge = set()
    for group in merging_indexes:
        indices_to_merge.update(group)
    
    merged_biomarkers = []
    
    # Add merged groups
    for index_group in merging_indexes:
        if len(index_group) == 1:
            # Single biomarker, no merging needed
            merged_biomarkers.append(biomarkers[index_group[0]])
        else:
            # Multiple biomarkers to merge
            canonical_names = [biomarkers[idx]["canonical_biomarker"] for idx in index_group]
            best_canonical = min(canonical_names, key=len)
            
            combined_occurrences = []
            for idx in index_group:
                combined_occurrences.extend(biomarkers[idx]["occurrences"])
            
            merged_group = {
                "canonical_biomarker": best_canonical,
                "occurrences": combined_occurrences
            }
            merged_biomarkers.append(merged_group)
    
    # Add unmerged groups (the ones not involved in any correlation)
    for idx, biomarker in enumerate(biomarkers):
        if idx not in indices_to_merge:
            merged_biomarkers.append(biomarker)
    
    return merged_biomarkers

def has_duplicates(correlations):
    all_numbers = [num for sublist in correlations for num in sublist]
    return len(all_numbers) != len(set(all_numbers))

def aggregation(model, tokenizer, device, evaluated_biomarkers):
    
    if evaluated_biomarkers == [] or evaluated_biomarkers == None:
        #filename = "./results/evaluated_biomarkers.json"
        filename = "./results/acronyms_logs.json"
        acronyms = read_json_arrays_to_list(filename)

    biomarkers = []
    #biomarkers_w_rows = []
    for acronym in acronyms:
        if acronym["acronym"] != "":
            biomarkers.append(acronym["acronym"])
            #biomarkers_w_rows.append(())

    # PRIMA FASE AGGREGAZIONE: exact matching degli acronimi (togliendo caratteri speciali, numeri e sigle "CSF", "PET", "MRI")
    # Dictionary to group items by their processed name
    groups = defaultdict(list)

    # Process each item and group by processed name
    for biomarker in biomarkers:
        clean_biomarker = process_name(biomarker)
        if clean_biomarker and len(clean_biomarker) > 1:  # Only add if processed name is not empty and of at least two characters
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
    print(f"Initial items: {len(biomarkers)}")
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
    
    #DEBUG
    with open("./parsed_biomarkers.json", "w", encoding="utf-8") as f:
        json.dump(parsed_biomarkers, f, ensure_ascii=False, indent=2)

    # FASE IN CUI PROVO AD AGGREGARE I GRUPPI GIÀ AGGREGATI (QUALCHE GRUPPONE L'HO DIVISO)
    # Chiedo ad LLM: 
    # sono stati identificati ed estratti tanti biomarkers da un dataset enorme, dopodichè questi biomarkers sono stati raggruppati
    # il raggruppamento è stato fatto tramite exact matching e questi sono i nomi canonici identificati per ognuno di questi gruppi
    # secondo te, ci sono all'interno di questa lista, dei nomi canonici i quali rispettivi gruppi potrebbero essere correlati e legati tra di loro?
    # Se si, indica gli la coppia (o più) di indici di biomarkers che sono correlati tra loro all'interno della seguente lista.
    # (considera che il tutto verrà poi visionato da un esperto, dunque se hai dubbi, prova comunque a restituire gli indici di eventuali possibili e 
    # dubbi gruppi correlati)
    # per esempio: 
    #
    # output è una lista di liste di indici correlati: [[1,5,7],[4,6],[8,18]]
    #
    # Faccio loop che finchè tutti i tentativi di merging sono stati skippati, mi continua a fare richiesta al modello e tentativo di merging
    try_again_merging_groups = True
    while try_again_merging_groups:
        try_again_merging_groups = False
        canonical_biomarkers = [d["canonical_biomarker"] for d in parsed_biomarkers]
        print(f"Current parsed_biomarkers length: {len(parsed_biomarkers)}")
        print(f"Current canonical_biomarkers length: {len(canonical_biomarkers)}")
        possible_correlations = call_model(biomarkers=canonical_biomarkers, task="merging_groups", model=model, tokenizer=tokenizer, device=device)
        if has_duplicates(possible_correlations):
            print("\n\n\n=== ERROR ===")
            print("Duplicate indices found!")
            input("Press Enter to go on.")
        # human check if the correlations detected by the LLM are valid
        actual_correlations = []
        print(f"The LLM has detected {len(possible_correlations)} possible related groups: let's check them!")
        for correlation in possible_correlations:
            if len(correlation) == 1:
                continue
            print("\nPossible related groups:")
            for corr in correlation:
                if corr >= (len(parsed_biomarkers) - 1):
                    print(f"{corr} >= {len(parsed_biomarkers) - 1} (corr >= len(parsed_biomarkers) - 1)")
                    input()
                else:
                    print(f" {parsed_biomarkers[int(corr)]['canonical_biomarker']} - {corr}")
            user_input = input("\nAre they related?\n <yes> for all\n <enter> for none (skip)\n <index1> <index2> ... to specify which groups to merge\n\n -> ").strip().lower()

            if user_input == "yes":
                print(f"\n All groups have been merged -> {correlation}\n")
                actual_correlations.append(correlation)
                try_again_merging_groups = True
            elif user_input and all(c.isdigit() or c.isspace() for c in user_input):
                # Parse numbers from input like "5 18 23" or "2 7"
                numbers = [int(x) for x in user_input.split()]
                print(f"\n Only these group indices have been merged -> {numbers}\n")
                actual_correlations.append(numbers)
                try_again_merging_groups = True
            else:
                print("\n Skip\n")
        # facciamo il merge dei gruppi identificati
        parsed_biomarkers = merge_groups(parsed_biomarkers, actual_correlations)
        if try_again_merging_groups:
            print("We try again to find some relations between the groups.\n")
    #DEBUG
    with open("./parsed_biomarkers_grouped.json", "w", encoding="utf-8") as f:
        json.dump(parsed_biomarkers, f, ensure_ascii=False, indent=2)



    print("\n\n\nFINISH! NOW WE GO ON!!\n\n\n")
    # chiedo ad LLM (iterando su ciascun gruppo identificato) di identificare le varianti appartenenti allo stesso gruppo:
    # varianti possono essere molecola leggermente diversa, test clinico leggermente diverso, etc.
    # esempio
    # rispondi nel seguente formato
    #
    # trip allucinante per passare solamente una volta le occurrences "duplicate" se il numero di occurrences è maggiore di 100
    for group in parsed_biomarkers:
        if len(group["occurrences"]) > 20:
            # Count occurrences of each unique string
            occurrence_counts = {}
            for occurrence in group["occurrences"]:
                occurrence_counts[occurrence] = occurrence_counts.get(occurrence, 0) + 1
            
            # Store original data for reconstruction
            group["_original_occurrences"] = group["occurrences"].copy()
            group["_occurrence_counts"] = occurrence_counts.copy()
            
            # Replace occurrences with unique strings only
            group["occurrences"] = list(occurrence_counts.keys())
            group_for_llm = {
                "canonical_biomarker": group["canonical_biomarker"],
                "occurrences": group["occurrences"]
            }
            
            response = call_model(biomarkers=group_for_llm, task="defining_variants", model=model, tokenizer=tokenizer, device=device)
            if response:
                group["variants"] = response["variants"]
            
            # After LLM responds, reconstruct the original variants
            if group.get("variants"):
                reconstructed_variants = {}
                for variant_key, unique_occurrences in group["variants"].items():
                    reconstructed_list = []
                    for unique_occurrence in unique_occurrences:
                        # Add back the original count for each unique occurrence
                        count = group["_occurrence_counts"].get(unique_occurrence, 1)
                        reconstructed_list.extend([unique_occurrence] * count)
                    reconstructed_variants[variant_key] = reconstructed_list
                
                # Replace variants with reconstructed ones
                group["variants"] = reconstructed_variants
            
            # Restore original occurrences
            group["occurrences"] = group["_original_occurrences"]
            
            # Clean up temporary keys
            del group["_original_occurrences"]
            del group["_occurrence_counts"]
            del group_for_llm
            print("variants added!")
            print(group)
        else:
            response = call_model(biomarkers=group, task="defining_variants", model=model, tokenizer=tokenizer, device=device)
            if response:
                group["variants"] = response["variants"]
                print("variants added!")
                print(group)

    
    '''
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
    '''

    for group in parsed_biomarkers:
        group["count"] = len(group["occurrences"])
        
        # Check if variants exist and is not empty
        if group.get("variants"):
            for variant_key, variant_list in group["variants"].items():
                # Calculate percentage
                pct = len(variant_list) / group["count"] * 100
                # Add percentage as new key
                group["variants"][f"{variant_key}-pct"] = f"{pct} %"

    final_biomarkers_sorted = sorted(parsed_biomarkers, key=lambda x: x['count'], reverse=True)

    with open("./results/biomarkers.json", "w", encoding="utf-8") as f:
        json.dump(final_biomarkers_sorted, f, ensure_ascii=False, indent=2)
    return final_biomarkers_sorted