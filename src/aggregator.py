import json
import torch
import re
from collections import defaultdict
from inputimeout import inputimeout, TimeoutOccurred

def input_con_timer(input: str, time: int):
    """
    Permette di eseguire un input con un timer: se il timer scade, viene ritornata la str -1

    Args:
        - input: la str da stampare per l'input
        - time: il numero di secondi a cui impostare il timer
    """
    try:
        risposta = inputimeout(prompt=input, timeout=time)
        return risposta
    except TimeoutOccurred:
        return "-1"

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
                system_prompt = f"""You are a marker variant grouper. Task: group occurrences by variant type.

DEFINITION
- Variant: Different forms of the same base biomarker, distinguished by:
  - Molecular forms: numeric/letter suffixes indicating isoforms, fragments, or modifications
  - Assessment versions: version numbers, subscales, or form indicators
  - Measurement types: different quantification methods of the same underlying marker

RULES:
1. Find the base marker name in each occurrence
2. Group by variant indicators (numbers, letters, versions)
3. Use first occurrence as group key
4. Output JSON only - no explanations

COMMON VARIANT PATTERNS:
- Numeric suffixes: -1, -2, 1-40, 1-42, 181, 217
- Greek letters: α, β, γ, δ
- Version indicators: v1, v2, -11, -13, -R, -SR
- Form types: SF (short form), LF (long form), A/B forms
- Measurement contexts: total, free, bound, phosphorylated (p-), etc.

OUTPUT FORMAT:
{{"variants": {{"key1": ["item1", "item2"], "key2": ["item3"]}}}}

No text before or after JSON. No long reasoning. Just JSON."""
                user_prompt = f"""Group these marker occurrences by variant. Output JSON only.
Note: It is not certain or mandatory that all groups actually have variants. If all occurrences represent the same variant or no clear variants exist, return them all grouped under a single variant key.

Example of the task:
Input:
{{
  "canonical_biomarker": "BDNF",
  "occurrences": [
    "BDNF",
    "BDNF Val66Met",
    "BDNF rs6265",
    "BDNF Val/Val",
    "BDNF Val/Met",
    "Bdnf-Val66Met",
    "BDNF rs6265",
    "bdnf"
  ]
}}
Output:
{{
  "variants": {{
    "BDNF": ["BDNF", "bdnf"],
    "BDNF Val66Met": ["BDNF Val66Met", "Bdnf-Val66Met"],
    "BDNF rs6265": ["BDNF rs6265", "BDNF rs6265"],
    "BDNF Val/Val": ["BDNF Val/Val"],
    "BDNF Val/Met": ["BDNF Val/Met"]
  }}
}}

Here is the input you have to process:
{biomarkers}

Output:"""
            elif task == "merging_groups":
                #canonical_biomarkers = [marker["canonical_biomarker"] for marker in biomarkers]
                system_prompt = """You are a data analysis assistant specialized in identifying potential correlations between biological/scientific markers. Your task is to analyze canonical marker names and identify groups that could be semantically related.

**Guidelines:**
- **CRITICAL: Actively look for synonyms, shared roots, and partial matches** - these are the primary indicators of related markers
- Use your knowledge of scientific terminology, biological pathways, and semantic relationships
- **Pay special attention to:**
  - Common roots/stems
  - Abbreviations vs full names
  - Different naming conventions
  - Alternative spellings/formats
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
['marker-A', 'marker-B', 'marker-C', 'marker-D', 'marker-C-1', 'marker-C-2', ...]

Then indices 2, 4, and 5 should be grouped together as they all refer to marker-C variants:

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

def find_rows_from_biomarkers(dict_list, couples):
    """
    Process a list of dictionaries by matching their 'occurrences' with markers in couples.
    Used to return the rows of the datasets where a biomarker has been identified-
    
    Args:
        dict_list: List of dictionaries, each containing an 'occurrences' key
        couples: List of tuples (marker, line) where marker is str and line is int
    
    Returns:
        The modified list of dictionaries with new 'rows' key added
    """
    # Create a lookup dictionary for faster searching
    marker_to_lines = {}
    for marker, line in couples:
        if marker not in marker_to_lines:
            marker_to_lines[marker] = []
        marker_to_lines[marker].append(line)
    
    # Process each dictionary
    for dictionary in dict_list:
        rows = []
        
        # Get occurrences from the dictionary
        occurrences = dictionary.get('occurrences', [])
        
        # Handle case where occurrences might be a single value or a list
        if not isinstance(occurrences, list):
            occurrences = [occurrences]
        
        # For each occurrence, find matching markers and collect their lines
        for occurrence in occurrences:
            if occurrence in marker_to_lines:
                rows.extend(marker_to_lines[occurrence])
        
        # Add the rows key to the dictionary
        dictionary['rows'] = sorted(set(rows))
    
    return dict_list

def aggregation(model, tokenizer, device, total_len, dataset_type: str="Alzheimer"):

    with open(f"./results/{dataset_type}/acronyms_logs.json", "r", encoding="utf-8") as f:
        data = json.load(f)
        acronyms_w_rows = []
        for d in data:
            acronyms_w_rows.append((d["acronym"], d["row_id"]))

    # PRIMA FASE: exact matching degli acronimi (togliendo caratteri speciali, numeri e sigle "CSF", "PET", "MRI")
    # Dictionary to group items by their processed name
    groups = defaultdict(list)

    # Process each item and group by processed name
    for biomarker, _ in acronyms_w_rows:
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

    # Printing the exact matching results
    total_grouped_items = 0
    for processed_name, original_names in duplicate_groups.items():
        total_grouped_items += len(original_names)
    print(f"\nExact matching finito:")
    print(f"Initial items: {len(acronyms_w_rows)}")
    print(f"Grouped items: {total_grouped_items}")
    print(f"Remaining items: {len(remaining_items)}")
    with open(f"./results/{dataset_type}/remaining_biomarkers.txt", "w") as f:
        for biomarker in remaining_items:
            f.write(f"{biomarker}\n")

    parsed_biomarkers = []
    for processed_name, original_names in duplicate_groups.items():
        canonical = choose_canonical_name(original_names)
        group_entry = {
            "canonical_biomarker": canonical,
            "occurrences": sorted(original_names)  # Sort members alphabetically
        }
        parsed_biomarkers.append(group_entry)

    # SECONDA FASE: chiediamo a LLM di trovare dei gruppi che potrebbero essere aggregati tra quelli già definiti dall'exact matching
    # Successsivamente iteriamo sulle proposte dell'LLM e possiamo confermare (quindi fare il merge), rifiutare (skip), 
    # o selezionare un sottoinsieme dei gruppi proposti da unire
    # Continuo a iterare finchè tutti i tentativi di merging sono stati skippati da parte dell'utente
    try_again_merging_groups = True
    timer_exit = False
    while try_again_merging_groups:
        try_again_merging_groups = False
        canonical_biomarkers = [d["canonical_biomarker"] for d in parsed_biomarkers]
        possible_correlations = call_model(biomarkers=canonical_biomarkers, task="merging_groups", model=model, tokenizer=tokenizer, device=device)
        if has_duplicates(possible_correlations):
            print("\n\n\n=== ERROR ===")
            print("Duplicate indices found!")
        # human check if the correlations detected by the LLM are valid
        actual_correlations = []
        print(f"The LLM has detected {len(possible_correlations)} possible related groups: let's check them!")
        for correlation in possible_correlations:
            if len(correlation) == 1:
                continue
            print("\nPossible related groups:")
            for corr in correlation:
                if corr >= len(parsed_biomarkers):
                    print("\n\n\n=== ERROR ===")
                    print(f"{corr} >= {len(parsed_biomarkers)} (corr >= len(parsed_biomarkers))")
                    #input()
                else:
                    print(f" {parsed_biomarkers[int(corr)]['canonical_biomarker']} - {corr}")
            str_user_input = "\nAre they related?\n <yes> for all\n <enter> for none (skip)\n <index1> <index2> ... to specify which groups to merge\n\n -> "
            # Imposto il timer a 5 minuti (300 secondi)
            # Se dopo 5 minuti non è ancora stato inviato alcun input, salvo attuali variabili e vado avanti
            user_input = input_con_timer(str_user_input, 300)
            if user_input == "-1":
                timer_exit = True
                break
            elif user_input == "yes":
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
        if timer_exit:
            break
        # facciamo il merge dei gruppi identificati
        parsed_biomarkers = merge_groups(parsed_biomarkers, actual_correlations)
        if try_again_merging_groups:
            print("We try again to find some relations between the groups.\n")
    if timer_exit:
        print(f"L'utente non è disponibile per la convalida dei possibili gruppi da unire.\nSi salva dunque 'parsed_biomarkers_{dataset_type}.json' e 'acronyms_w_rows_{dataset_type}.json' all'interno della cartella /checkpoints")
        with open(f"./checkpoints/parsed_biomarkers_{dataset_type}.json", "w", encoding="utf-8") as f:
            json.dump(parsed_biomarkers, f, ensure_ascii=False, indent=2)
        with open(f"./checkpoints/acronyms_w_rows_{dataset_type}.json", "w", encoding="utf-8") as f:
            json.dump(acronyms_w_rows, f, ensure_ascii=False, indent=2)
        timer_exit = False

    # TERZA FASE: chiedo a LLM di identificare varianti tra le componenti di uno stesso gruppo (non passo tutte le 'occurrences' ma solo quelle diverse tra di loro)
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
            print("Variants added!")
            print(group)
        else:
            response = call_model(biomarkers=group, task="defining_variants", model=model, tokenizer=tokenizer, device=device)
            if response:
                group["variants"] = response["variants"]
                print("Variants added!")
                print(group)

    # conteggi finali e salvataggio
    for group in parsed_biomarkers:
        group["total_count"] = f"{len(group['occurrences'])} / {total_len}"
        total_pct = len(group["occurrences"]) / total_len * 100
        group["total_percentage"] = f"{total_pct:.2f} %"
        
        # Check if variants exist and is not empty
        if group.get("variants"):
            # Initialize the dictionary of variant percentages
            group["variant_percentages"] = {}  
            # Create a copy of items to iterate over
            for variant_key, variant_list in list(group["variants"].items()):
                # Calculate percentage
                pct = len(variant_list) / len(group['occurrences']) * 100
                # Add percentage as new key
                group["variant_percentages"][variant_key] = f"{pct:.2f} %"
    
    # Find the rows of the dataset from which each biomarker was extracted (using acronyms_w_rows, which pairs each biomarker with the row it was extracted from)
    parsed_biomarkers = find_rows_from_biomarkers(parsed_biomarkers, acronyms_w_rows)

    final_biomarkers_sorted = sorted(parsed_biomarkers, key=lambda x: len(x['occurrences']), reverse=True)

    with open(f"./results/{dataset_type}/biomarkers.json", "w", encoding="utf-8") as f:
        json.dump(final_biomarkers_sorted, f, ensure_ascii=False, indent=2)
    return final_biomarkers_sorted

def aggregation_resume(model, tokenizer, device, total_len, dataset_type: str="Alzheimer"):
    print(f"Resuming the final analysis: loading '/checkpoints/parsed_biomarkers_{dataset_type}.json' and '/checkpoints/acronyms_w_rows_{dataset_type}.json'.")
    with open(f"./checkpoints/parsed_biomarkers_{dataset_type}.json", "r", encoding="utf-8") as f:
        parsed_biomarkers = json.load(f)
    with open(f"./checkpoints/acronyms_w_rows_{dataset_type}.json", "r", encoding="utf-8") as f:
        acronyms_w_rows = json.load(f)

    # SECONDA FASE: chiediamo a LLM di trovare dei gruppi che potrebbero essere aggregati tra quelli già definiti dall'exact matching
    # Successsivamente iteriamo sulle proposte dell'LLM e possiamo confermare (quindi fare il merge), rifiutare (skip), 
    # o selezionare un sottoinsieme dei gruppi proposti da unire
    # Continuo a iterare finchè tutti i tentativi di merging sono stati skippati da parte dell'utente
    try_again_merging_groups = True
    timer_exit = False
    while try_again_merging_groups:
        try_again_merging_groups = False
        canonical_biomarkers = [d["canonical_biomarker"] for d in parsed_biomarkers]
        possible_correlations = call_model(biomarkers=canonical_biomarkers, task="merging_groups", model=model, tokenizer=tokenizer, device=device)
        if has_duplicates(possible_correlations):
            print("\n\n\n=== ERROR ===")
            print("Duplicate indices found!")
        # human check if the correlations detected by the LLM are valid
        actual_correlations = []
        print(f"The LLM has detected {len(possible_correlations)} possible related groups: let's check them!")
        for correlation in possible_correlations:
            if len(correlation) == 1:
                continue
            print("\nPossible related groups:")
            for corr in correlation:
                if corr >= (len(parsed_biomarkers)):
                    print("\n\n\n=== ERROR ===")
                    print(f"{corr} >= {len(parsed_biomarkers)} (corr >= len(parsed_biomarkers))")
                    #input()
                else:
                    print(f" {parsed_biomarkers[int(corr)]['canonical_biomarker']} - {corr}")
            str_user_input = "\nAre they related?\n <yes> for all\n <enter> for none (skip)\n <index1> <index2> ... to specify which groups to merge\n\n -> "
            # Imposto il timer a 5 minuti (300 secondi)
            # Se dopo 5 minuti non è ancora stato inviato alcun input, salvo attuali variabili e vado avanti
            user_input = input_con_timer(str_user_input, 300)
            if user_input == "-1":
                timer_exit = True
                break
            elif user_input == "yes":
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
        if timer_exit:
            break
        # facciamo il merge dei gruppi identificati
        parsed_biomarkers = merge_groups(parsed_biomarkers, actual_correlations)
        if try_again_merging_groups:
            print("We try again to find some relations between the groups.\n")
    if timer_exit:
        print(f"L'utente non è disponibile per la convalida dei possibili gruppi da unire.\nSi salva dunque 'parsed_biomarkers_{dataset_type}.json' e 'acronyms_w_rows_{dataset_type}.json'.")
        with open(f"./checkpoints/parsed_biomarkers_{dataset_type}.json", "w", encoding="utf-8") as f:
            json.dump(parsed_biomarkers, f, ensure_ascii=False, indent=2)
        with open(f"./checkpoints/acronyms_w_rows_{dataset_type}.json", "w", encoding="utf-8") as f:
            json.dump(acronyms_w_rows, f, ensure_ascii=False, indent=2)
        timer_exit = False

    # TERZA FASE: chiedo a LLM di identificare varianti tra le componenti di uno stesso gruppo (non passo tutte le 'occurrences' ma solo quelle diverse tra di loro)
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
            print("Variants added!")
            print(group)
        else:
            response = call_model(biomarkers=group, task="defining_variants", model=model, tokenizer=tokenizer, device=device)
            if response:
                group["variants"] = response["variants"]
                print("Variants added!")
                print(group)

    # conteggi finali e salvataggio
    for group in parsed_biomarkers:
        group["total_count"] = f"{len(group['occurrences'])} / {total_len}"
        total_pct = len(group["occurrences"]) / total_len * 100
        group["total_percentage"] = f"{total_pct:.2f} %"
        
        # Check if variants exist and is not empty
        if group.get("variants"):
            # Initialize the dictionary of variant percentages
            group["variant_percentages"] = {}  
            # Create a copy of items to iterate over
            for variant_key, variant_list in list(group["variants"].items()):
                # Calculate percentage
                pct = len(variant_list) / len(group['occurrences']) * 100
                # Add percentage as new key
                group["variant_percentages"][variant_key] = f"{pct:.2f} %"
    
    # Find the rows of the dataset from which each biomarker was extracted (using acronyms_w_rows, which pairs each biomarker with the row it was extracted from)
    parsed_biomarkers = find_rows_from_biomarkers(parsed_biomarkers, acronyms_w_rows)

    final_biomarkers_sorted = sorted(parsed_biomarkers, key=lambda x: len(x['occurrences']), reverse=True)

    with open(f"./results/{dataset_type}/biomarkers.json", "w", encoding="utf-8") as f:
        json.dump(final_biomarkers_sorted, f, ensure_ascii=False, indent=2)
    return final_biomarkers_sorted

def aggregation_resume_part1(dataset_type: str="Alzheimer"):

    with open(f"./results/{dataset_type}/acronyms_logs.json", "r", encoding="utf-8") as f:
        data = json.load(f)
        acronyms_w_rows = []
        for d in data:
            acronyms_w_rows.append((d["acronym"], d["row_id"]))

    # PRIMA FASE: exact matching degli acronimi (togliendo caratteri speciali, numeri e sigle "CSF", "PET", "MRI")
    # Dictionary to group items by their processed name
    groups = defaultdict(list)

    # Process each item and group by processed name
    for biomarker, _ in acronyms_w_rows:
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

    # Printing the exact matching results
    total_grouped_items = 0
    for processed_name, original_names in duplicate_groups.items():
        total_grouped_items += len(original_names)
    print(f"\nExact matching finito:")
    print(f"Initial items: {len(acronyms_w_rows)}")
    print(f"Grouped items: {total_grouped_items}")
    print(f"Remaining items: {len(remaining_items)}")
    with open(f"./results/{dataset_type}/remaining_biomarkers.txt", "w") as f:
        for biomarker in remaining_items:
            f.write(f"{biomarker}\n")

    parsed_biomarkers = []
    for processed_name, original_names in duplicate_groups.items():
        canonical = choose_canonical_name(original_names)
        group_entry = {
            "canonical_biomarker": canonical,
            "occurrences": sorted(original_names)  # Sort members alphabetically
        }
        parsed_biomarkers.append(group_entry)
    with open(f"./checkpoints/parsed_biomarkers_{dataset_type}.json", "w", encoding="utf-8") as f:
        json.dump(parsed_biomarkers, f, ensure_ascii=False, indent=2)
    with open(f"./checkpoints/acronyms_w_rows_{dataset_type}.json", "w", encoding="utf-8") as f:
        json.dump(acronyms_w_rows, f, ensure_ascii=False, indent=2)
    
    return parsed_biomarkers

def aggregation_resume_part2(model, tokenizer, device, total_len, dataset_type: str="Alzheimer"):
    print(f"Resuming the final analysis: loading '/checkpoints/parsed_biomarkers_{dataset_type}.json' and '/checkpoints/acronyms_w_rows_{dataset_type}.json'.")
    with open(f"./checkpoints/parsed_biomarkers_{dataset_type}.json", "r", encoding="utf-8") as f:
        parsed_biomarkers = json.load(f)
    with open(f"./checkpoints/acronyms_w_rows_{dataset_type}.json", "r", encoding="utf-8") as f:
        acronyms_w_rows = json.load(f)

    # SECONDA FASE: chiediamo a LLM di trovare dei gruppi che potrebbero essere aggregati tra quelli già definiti dall'exact matching
    # Successsivamente iteriamo sulle proposte dell'LLM e possiamo confermare (quindi fare il merge), rifiutare (skip), 
    # o selezionare un sottoinsieme dei gruppi proposti da unire
    # Continuo a iterare finchè tutti i tentativi di merging sono stati skippati da parte dell'utente
    try_again_merging_groups = True
    timer_exit = False
    while try_again_merging_groups:
        try_again_merging_groups = False
        canonical_biomarkers = [d["canonical_biomarker"] for d in parsed_biomarkers]
        possible_correlations = call_model(biomarkers=canonical_biomarkers, task="merging_groups", model=model, tokenizer=tokenizer, device=device)
        if has_duplicates(possible_correlations):
            print("\n\n\n=== ERROR ===")
            print("Duplicate indices found!")
        # human check if the correlations detected by the LLM are valid
        actual_correlations = []
        print(f"The LLM has detected {len(possible_correlations)} possible related groups: let's check them!")
        for correlation in possible_correlations:
            if len(correlation) == 1:
                continue
            print("\nPossible related groups:")
            for corr in correlation:
                if corr >= (len(parsed_biomarkers)):
                    print("\n\n\n=== ERROR ===")
                    print(f"{corr} >= {len(parsed_biomarkers)} (corr >= len(parsed_biomarkers))")
                    #input()
                else:
                    print(f" {parsed_biomarkers[int(corr)]['canonical_biomarker']} - {corr}")
            str_user_input = "\nAre they related?\n <yes> for all\n <enter> for none (skip)\n <index1> <index2> ... to specify which groups to merge\n\n -> "
            # Imposto il timer a 5 minuti (300 secondi)
            # Se dopo 5 minuti non è ancora stato inviato alcun input, salvo attuali variabili e vado avanti
            user_input = input_con_timer(str_user_input, 300)
            if user_input == "-1":
                timer_exit = True
                break
            elif user_input == "yes":
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
        if timer_exit:
            break
        # facciamo il merge dei gruppi identificati
        parsed_biomarkers = merge_groups(parsed_biomarkers, actual_correlations)
        if try_again_merging_groups:
            print("We try again to find some relations between the groups.\n")
    if timer_exit:
        print(f"L'utente non è disponibile per la convalida dei possibili gruppi da unire.\nSi salva dunque 'parsed_biomarkers_{dataset_type}.json' e 'acronyms_w_rows_{dataset_type}.json'.")
    with open(f"./checkpoints/parsed_biomarkers_{dataset_type}.json", "w", encoding="utf-8") as f:
        json.dump(parsed_biomarkers, f, ensure_ascii=False, indent=2)
    with open(f"./checkpoints/acronyms_w_rows_{dataset_type}.json", "w", encoding="utf-8") as f:
        json.dump(acronyms_w_rows, f, ensure_ascii=False, indent=2)

    return parsed_biomarkers

def aggregation_resume_part3(model, tokenizer, device, total_len, dataset_type: str="Alzheimer"):
    print(f"Resuming the final analysis: loading '/checkpoints/parsed_biomarkers_{dataset_type}.json' and '/checkpoints/acronyms_w_rows_{dataset_type}.json'.")
    with open(f"./checkpoints/parsed_biomarkers_{dataset_type}.json", "r", encoding="utf-8") as f:
        parsed_biomarkers = json.load(f)
    with open(f"./checkpoints/acronyms_w_rows_{dataset_type}.json", "r", encoding="utf-8") as f:
        acronyms_w_rows = json.load(f)

    # TERZA FASE: chiedo a LLM di identificare varianti tra le componenti di uno stesso gruppo (non passo tutte le 'occurrences' ma solo quelle diverse tra di loro)
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
            print("Variants added!")
            print(group)
        else:
            response = call_model(biomarkers=group, task="defining_variants", model=model, tokenizer=tokenizer, device=device)
            if response:
                group["variants"] = response["variants"]
                print("Variants added!")
                print(group)

    # conteggi finali e salvataggio
    for group in parsed_biomarkers:
        group["total_count"] = f"{len(group['occurrences'])} / {total_len}"
        total_pct = len(group["occurrences"]) / total_len * 100
        group["total_percentage"] = f"{total_pct:.2f} %"
        
        # Check if variants exist and is not empty
        if group.get("variants"):
            # Initialize the dictionary of variant percentages
            group["variant_percentages"] = {}  
            # Create a copy of items to iterate over
            for variant_key, variant_list in list(group["variants"].items()):
                # Calculate percentage
                pct = len(variant_list) / len(group['occurrences']) * 100
                # Add percentage as new key
                group["variant_percentages"][variant_key] = f"{pct:.2f} %"
    
    # Find the rows of the dataset from which each biomarker was extracted (using acronyms_w_rows, which pairs each biomarker with the row it was extracted from)
    parsed_biomarkers = find_rows_from_biomarkers(parsed_biomarkers, acronyms_w_rows)

    final_biomarkers_sorted = sorted(parsed_biomarkers, key=lambda x: len(x['occurrences']), reverse=True)

    with open(f"./results/{dataset_type}/biomarkers.json", "w", encoding="utf-8") as f:
        json.dump(final_biomarkers_sorted, f, ensure_ascii=False, indent=2)
    return final_biomarkers_sorted

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