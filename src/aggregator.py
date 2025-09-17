import json
import torch
import re
import os
from collections import defaultdict
from inputimeout import inputimeout, TimeoutOccurred
from src.models import call_model

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

def has_duplicates(correlations: list):
    if len(correlations) == 0:
        return False
    all_numbers = [num for sublist in correlations for num in sublist]
    return len(all_numbers) != len(set(all_numbers))

def parse_percentage(percentage_str):
    """
    Extract numeric value from percentage string like "33.77 %"
    """
    # Remove % symbol and extra spaces, then convert to float
    numeric_str = re.sub(r'[%\s]', '', percentage_str)
    return float(numeric_str)

def process_results(biomarkers: None, output_file: str):
    """
    Process biomarkers JSON file according to the specified criteria
    """
    try:
        # Read the input JSON file
        if biomarkers is None:
            return None
        
        # First, remove unwanted keys from all dictionaries
        cleaned_biomarkers = []
        for biomarker in biomarkers:
            try:
                # Create a copy and remove unwanted keys
                cleaned_biomarker = biomarker.copy()
                cleaned_biomarker.pop('occurrences', None)
                cleaned_biomarker.pop('variants', None)
                
                # Parse the total_percentage to validate it
                parse_percentage(biomarker['total_percentage'])
                cleaned_biomarkers.append(cleaned_biomarker)
                
            except (ValueError, KeyError) as e:
                print(f"Warning: Could not process percentage for biomarker {biomarker.get('canonical_biomarker', 'unknown')}: {e}")
                continue
        
        # Count how many dictionaries have total_percentage > 10%
        count_above_10 = 0
        for biomarker in cleaned_biomarkers:
            percentage = parse_percentage(biomarker['total_percentage'])
            if percentage > 10:
                count_above_10 += 1
        
        # Apply filtering logic based on count of items > 10%
        final_biomarkers = []
        
        if 5 < count_above_10 < 20:
            # If count is between 5 and 20, keep items > 10%
            threshold = 10
        elif count_above_10 < 5:
            # If count is < 5, select items > 5%
            threshold = 5
        elif count_above_10 > 20:
            # If count is > 20, select items > 15%
            threshold = 15
        else:
            # count_above_10 == 5 or count_above_10 == 20 (edge cases)
            threshold = 10
        
        # Filter based on the determined threshold
        for biomarker in cleaned_biomarkers:
            percentage = parse_percentage(biomarker['total_percentage'])
            if percentage > threshold:
                final_biomarkers.append(biomarker)
        
        # Save the processed biomarkers
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(final_biomarkers, f, indent=2, ensure_ascii=False)
        
        return final_biomarkers

    except Exception as e:
        print(f"Unexpected error: {e}")
        return None

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

def reorganize_biomarkers(biomarkers=None, acronyms_w_rows=None):
    """Funzione principale per riorganizzare i biomarkers secondo la logica richiesta.
    
    Args:
        biomarkers: Lista di biomarkers, ognuno con le sue varianti
        acronyms_w_rows: Lista di coppie (acronimo, row_id)
    """
    
    if biomarkers is None or acronyms_w_rows is None:
        print("Impossibile procedere senza i dati necessari.")
        return None

    # Creazione di un dizionario per mappare acronimi a row_id
    acronym_to_row_ids = defaultdict(list)
    all_dashes = r'[\u002D\u2010-\u2015\u2212]'
    for acronym, row_id in acronyms_w_rows:
        acronym_to_row_ids[re.sub(all_dashes, '', acronym)].append(row_id)
        
    # Processa ogni biomarker
    for biomarker_idx, biomarker in enumerate(biomarkers):
        if 'variants' not in biomarker:
            print(f"Errore: 'variants' non trovato nel biomarker {biomarker_idx}")
            if 'canonical_biomarker' in biomarker:
                print(f"Biomarker: {biomarker['canonical_biomarker']}")
            continue
        
        # Inizializzazione del nuovo dizionario rows per questo biomarker
        new_rows = {}
        
        # Per ogni variante in questo biomarker
        for variant_name, variant_values in biomarker['variants'].items():
            
            # Passaggio 1: Prendo la lista completa di tutti i valori
            if isinstance(variant_values, list):
                all_values = variant_values.copy()
            elif variant_values is None:
                all_values = []
            else:
                all_values = [variant_values]
            
            # Passaggio 2: Tengo solamente i valori unici (elimino duplicati)
            # Filtro anche i valori None o vuoti
            unique_values = list(set(val for val in all_values if val is not None and val != ""))
            
            # Passaggio 3: Per ogni valore unico, cerco corrispondenze negli acronimi
            row_ids_for_variant = []
            
            for unique_value in unique_values:   
                unique_value = re.sub(all_dashes, '', unique_value)
                # Cerco questo valore negli acronimi
                if unique_value in acronym_to_row_ids:
                    matching_rows = acronym_to_row_ids[unique_value]
                    row_ids_for_variant.extend(matching_rows)
            
            # Passaggio 4: Tengo solamente i valori unici anche in questa lista
            unique_row_ids = list(set(row_ids_for_variant))
            new_rows[variant_name] = unique_row_ids
                    
        # Aggiorno la struttura di questo biomarker
        biomarker['rows'] = new_rows
    
    return biomarkers

def aggregate_similar_strings(string_list):
    """
    Groups strings that share at least 3 consecutive characters.
    
    Args:
        string_list: List of strings to analyze
        
    Returns:
        List of lists, where each inner list contains indices of strings
        that share at least 3 consecutive characters
    """
    def get_3char_substrings(s):
        """Extract all 3-character substrings from a string"""
        if len(s) < 3:
            return set()
        return {s[i:i+3].lower() for i in range(len(s) - 2)}
    
    # Create groups based on shared 3-character sequences
    groups = {}
    processed = set()
    
    for i, str1 in enumerate(string_list):
        if i in processed:
            continue
            
        # Get 3-char substrings for current string
        substrings1 = get_3char_substrings(str1)
        if not substrings1:
            continue
            
        # Start a new group with current string
        current_group = [i]
        processed.add(i)
        
        # Find other strings that share at least one 3-char substring
        for j, str2 in enumerate(string_list[i+1:], i+1):
            if j in processed:
                continue
                
            substrings2 = get_3char_substrings(str2)
            
            # Check if they share any 3-character substring
            if substrings1.intersection(substrings2):
                current_group.append(j)
                processed.add(j)
        
        # Only add groups with more than one string
        if len(current_group) > 1:
            groups[len(groups)] = current_group
    
    return list(groups.values())

def aggregation_unified(model=None, tokenizer=None, device=None, total_len=None, 
                        dataset_type="Alzheimer", start_phase=1, end_phase=3, 
                        resume_from_checkpoint=False, use_similarity_fallback=True):
    """
    Unified aggregation function that can run full pipeline or specific phases.
    
    Args:
        model, tokenizer, device: Required for phases 2 and 3
        total_len: Required for phase 3
        dataset_type: Dataset identifier
        start_phase: Phase to start from (1, 2, or 3)
        end_phase: Phase to end at (1, 2, or 3)
        resume_from_checkpoint: Load from checkpoint files instead of source data
        use_similarity_fallback: Use string similarity instead of LLM for phase 2 (first iteration)
    
    Returns:
        parsed_biomarkers: The biomarker groups at the end of processing
    """
    
    # Validate parameters
    if start_phase < 1 or start_phase > 3 or end_phase < 1 or end_phase > 3:
        raise ValueError("Phases must be between 1 and 3")
    if start_phase > end_phase:
        raise ValueError("start_phase cannot be greater than end_phase")
    if (start_phase >= 2 or end_phase >= 2) and any(x is None for x in [model, tokenizer, device]):
        raise ValueError("model, tokenizer, and device are required for phases 2 and 3")
    if end_phase == 3 and total_len is None:
        raise ValueError("total_len is required for phase 3")
    
    # Initialize variables
    parsed_biomarkers = None
    acronyms_w_rows = None
    
    # ============ PHASE 1: EXACT MATCHING ============
    if start_phase <= 1 <= end_phase:
        print("=== EXACT MATCHING ===")
        
        if resume_from_checkpoint and os.path.exists(f"./checkpoints/parsed_biomarkers_{dataset_type}.json"):
            print(f"Loading phase 1 checkpoint...")
            with open(f"./checkpoints/parsed_biomarkers_{dataset_type}.json", "r", encoding="utf-8") as f:
                parsed_biomarkers = json.load(f)
            with open(f"./checkpoints/acronyms_w_rows_{dataset_type}.json", "r", encoding="utf-8") as f:
                acronyms_w_rows = json.load(f)
        else:
            # Load source data
            with open(f"./logs/{dataset_type}/acronyms_logs.json", "r", encoding="utf-8") as f:
                data = json.load(f)
                acronyms_w_rows = []
                for d in data:
                    if d["acronym"] is not None:
                        acronyms_w_rows.append((d["acronym"], d["row_id"]))
            
            # Dictionary to group items by their processed name
            groups = defaultdict(list)
            
            # Process each item and group by processed name
            for biomarker, _ in acronyms_w_rows:
                clean_biomarker = process_name(biomarker)
                if clean_biomarker and len(clean_biomarker) > 1:
                    groups[clean_biomarker].append(biomarker)
            
            # Separate groups (duplicates) from single items
            duplicate_groups = {}
            remaining_items = []
            for processed_name, original_names in groups.items():
                if len(original_names) > 1:
                    duplicate_groups[processed_name] = original_names
                else:
                    remaining_items.extend(original_names)
            
            # Print results
            total_grouped_items = sum(len(original_names) for original_names in duplicate_groups.values())
            print(f"Exact matching results:")
            print(f"Initial items: {len(acronyms_w_rows)}")
            print(f"Grouped items: {total_grouped_items}")
            print(f"Remaining items: {len(remaining_items)}")
            
            # Save remaining items
            os.makedirs(f"./results/{dataset_type}", exist_ok=True)
            with open(f"./results/{dataset_type}/remaining_biomarkers.txt", "w") as f:
                for biomarker in remaining_items:
                    f.write(f"{biomarker}\n")
            
            # Create parsed biomarkers
            parsed_biomarkers = []
            for processed_name, original_names in duplicate_groups.items():
                canonical = choose_canonical_name(original_names)
                group_entry = {
                    "canonical_biomarker": canonical,
                    "occurrences": sorted(original_names)
                }
                parsed_biomarkers.append(group_entry)
            
            # Save checkpoint
            _save_checkpoint(parsed_biomarkers, acronyms_w_rows, dataset_type)
    
    # ============ PHASE 2: DETERMINISTIC and LLM-BASED GROUPING ============
    if start_phase <= 2 <= end_phase:
        print("=== DETERMINISTIC and LLM-BASED GROUPING ===")
        
        # Load from checkpoint if needed
        if parsed_biomarkers is None:
            with open(f"./checkpoints/parsed_biomarkers_{dataset_type}.json", "r", encoding="utf-8") as f:
                parsed_biomarkers = json.load(f)
        if acronyms_w_rows is None:
            with open(f"./checkpoints/acronyms_w_rows_{dataset_type}.json", "r", encoding="utf-8") as f:
                acronyms_w_rows = json.load(f)
        
        # Merging loop with optional similarity fallback
        parsed_biomarkers = _run_merging_loop(
            parsed_biomarkers, model, tokenizer, device, dataset_type, 
            use_similarity_fallback=use_similarity_fallback
        )
        
        # Save checkpoint after phase 2
        _save_checkpoint(parsed_biomarkers, acronyms_w_rows, dataset_type)
    
    # ============ PHASE 3: VARIANT IDENTIFICATION ============
    if start_phase <= 3 <= end_phase:
        print("=== VARIANT IDENTIFICATION ===")
        
        # Load from checkpoint if needed
        if parsed_biomarkers is None:
            with open(f"./checkpoints/parsed_biomarkers_{dataset_type}.json", "r", encoding="utf-8") as f:
                parsed_biomarkers = json.load(f)
        if acronyms_w_rows is None:
            with open(f"./checkpoints/acronyms_w_rows_{dataset_type}.json", "r", encoding="utf-8") as f:
                acronyms_w_rows = json.load(f)
        
        # Process variants for each group
        for group in parsed_biomarkers:
            group = _process_group_variants(group, model, tokenizer, device, dataset_type)
                
        # Find source rows
        parsed_biomarkers = find_rows_from_biomarkers(parsed_biomarkers, acronyms_w_rows)
        # divide rows into variants
        parsed_biomarkers = reorganize_biomarkers(parsed_biomarkers, acronyms_w_rows)

        # Calculate final statistics
        for group in parsed_biomarkers:
            group["total_count"] = f"{len(group['occurrences'])} / {total_len}"
            total_pct = len(group["occurrences"]) / total_len * 100
            group["total_percentage"] = f"{total_pct:.2f} %"
            
            if group.get("variants"):
                group["variant_percentages"] = {}
                for variant_key, variant_list in group["variants"].items():
                    pct = len(variant_list) / len(group['occurrences']) * 100
                    group["variant_percentages"][variant_key] = f"{pct:.2f} %"
        
        # Sort and save final results
        final_biomarkers_sorted = sorted(parsed_biomarkers, key=lambda x: len(x['occurrences']), reverse=True)
        _ = process_results(biomarkers=final_biomarkers_sorted, output_file=f"./results/{dataset_type}/filtered_biomarkers.json")
        
        os.makedirs(f"./results/{dataset_type}", exist_ok=True)
        with open(f"./results/{dataset_type}/biomarkers.json", "w", encoding="utf-8") as f:
            json.dump(final_biomarkers_sorted, f, ensure_ascii=False, indent=2)
        
        return final_biomarkers_sorted
    
    return parsed_biomarkers


def _save_checkpoint(parsed_biomarkers, acronyms_w_rows, dataset_type):
    """Save checkpoint files"""
    os.makedirs("./checkpoints", exist_ok=True)
    with open(f"./checkpoints/parsed_biomarkers_{dataset_type}.json", "w", encoding="utf-8") as f:
        json.dump(parsed_biomarkers, f, ensure_ascii=False, indent=2)
    with open(f"./checkpoints/acronyms_w_rows_{dataset_type}.json", "w", encoding="utf-8") as f:
        json.dump(acronyms_w_rows, f, ensure_ascii=False, indent=2)


def _run_merging_loop(parsed_biomarkers, model, tokenizer, device, dataset_type, use_similarity_fallback=True):
    """Run the iterative merging loop with user interaction"""
    
    # First iteration with optional similarity fallback
    if use_similarity_fallback:
        print("Using similarity-based grouping for first iteration...")
        try_again_merging_groups = True
        timer_exit = False
        
        while try_again_merging_groups:
            try_again_merging_groups = False
            canonical_biomarkers = [d["canonical_biomarker"] for d in parsed_biomarkers 
                                  if d is not None and d.get("canonical_biomarker") is not None]
            
            possible_correlations = aggregate_similar_strings(canonical_biomarkers)
            
            if has_duplicates(possible_correlations):
                print("\n=== ERROR: Duplicate indices found! ===")
            
            actual_correlations = _get_user_correlations(possible_correlations, parsed_biomarkers)
            if actual_correlations is None:  # Timer exit
                timer_exit = True
                break
                
            if actual_correlations:
                parsed_biomarkers = merge_groups(parsed_biomarkers, actual_correlations)
                try_again_merging_groups = True
                print("Trying again to find relations between groups.\n")
        
        if timer_exit:
            return parsed_biomarkers
    
    # LLM-based iterations
    try_again_merging_groups = True
    while try_again_merging_groups:
        try_again_merging_groups = False
        canonical_biomarkers = [d["canonical_biomarker"] for d in parsed_biomarkers 
                              if d is not None and d.get("canonical_biomarker") is not None]
        
        try:
            possible_correlations, _, _ = call_model(task="groups", dataset_type=dataset_type, 
                                             model=model, tokenizer=tokenizer, device=device, 
                                             biomarkers=canonical_biomarkers)
        except Exception as e:
            print(f"Model call failed: {e}")
            possible_correlations = None
        
        if possible_correlations == None or possible_correlations == []:
            return parsed_biomarkers
        
        if has_duplicates(possible_correlations):
            print("\n=== ERROR: Duplicate indices found! ===")
        
        actual_correlations = _get_user_correlations(possible_correlations, parsed_biomarkers)
        if actual_correlations is None:  # Timer exit
            break
            
        if actual_correlations:
            parsed_biomarkers = merge_groups(parsed_biomarkers, actual_correlations)
            try_again_merging_groups = True
            print("Trying again to find relations between groups.\n")
    
    return parsed_biomarkers


def _get_user_correlations(possible_correlations, parsed_biomarkers):
    """Handle user interaction for correlation validation"""
    actual_correlations = []
    print(f"Detected {len(possible_correlations)} possible related groups. Let's check them!")
    
    for correlation in possible_correlations:
        if len(correlation) == 1:
            continue
            
        print("\nPossible related groups:")
        for corr in correlation:
            if corr >= len(parsed_biomarkers):
                print(f"\n=== ERROR: {corr} >= {len(parsed_biomarkers)} ===")
            else:
                print(f" {parsed_biomarkers[int(corr)]['canonical_biomarker']} - {corr}")
        
        str_user_input = ("\nAre they related?\n"
                         " <yes> for all\n"
                         " <enter> for none (skip)\n"
                         " <index1> <index2> ... to specify which groups to merge\n\n -> ")
        
        user_input = input_con_timer(str_user_input, 300)
        
        if user_input == "-1":  # Timer exit
            return None
        elif user_input == "yes":
            print(f"\nAll groups merged -> {correlation}\n")
            actual_correlations.append(correlation)
        elif user_input and all(c.isdigit() or c.isspace() for c in user_input):
            numbers = [int(x) for x in user_input.split()]
            print(f"\nSelected group indices merged -> {numbers}\n")
            actual_correlations.append(numbers)
        else:
            print("\nSkipped\n")
    
    return actual_correlations


def _process_group_variants(group, model, tokenizer, device, dataset_type):
    """Process variants for a single group"""
    # Count occurrences
    occurrence_counts = {}
    for occurrence in group["occurrences"]:
        occurrence_counts[occurrence] = occurrence_counts.get(occurrence, 0) + 1
    
    # Store original data
    original_occurrences = group["occurrences"].copy()
    
    # Create unique list for LLM
    group["occurrences"] = list(occurrence_counts.keys())
    group_for_llm = {
        "canonical_biomarker": group["canonical_biomarker"],
        "occurrences": group["occurrences"]
    }
    
    # Call LLM
    try:
        response, _, _ = call_model(task="variants", dataset_type=dataset_type, 
                            model=model, tokenizer=tokenizer, device=device, 
                            biomarkers=group_for_llm)
        if response and response.get("variants"):
            group["variants"] = response["variants"]
            
            # Reconstruct with original counts
            reconstructed_variants = {}
            for variant_key, unique_occurrences in group["variants"].items():
                reconstructed_list = []
                for unique_occurrence in unique_occurrences:
                    count = occurrence_counts.get(unique_occurrence, 1)
                    reconstructed_list.extend([unique_occurrence] * count)
                reconstructed_variants[variant_key] = reconstructed_list
            
            group["variants"] = reconstructed_variants
            
    except Exception as e:
        print(f"Model call failed for variants: {e}")
    
    # Restore original occurrences
    group["occurrences"] = original_occurrences
    print(f"Group {group['canonical_biomarker']}: variants processed!")
    return group


# Convenience wrapper functions for backward compatibility
def aggregation(model, tokenizer, device, total_len, dataset_type="Alzheimer"):
    """Original aggregation function - runs all phases"""
    return aggregation_unified(model, tokenizer, device, total_len, dataset_type, 1, 3, False)

def aggregation_resume(model, tokenizer, device, total_len, dataset_type="Alzheimer"):
    """Resume from checkpoints - phases 2 and 3"""
    return aggregation_unified(model, tokenizer, device, total_len, dataset_type, 2, 3, True)

def aggregation_resume_part1(dataset_type="Alzheimer"):
    """Only phase 1"""
    return aggregation_unified(dataset_type=dataset_type, start_phase=1, end_phase=1)

def aggregation_resume_part2(model, tokenizer, device, total_len, dataset_type="Alzheimer"):
    """Only phase 2 with similarity fallback then LLM"""
    return aggregation_unified(model, tokenizer, device, total_len, dataset_type, 2, 2, True, True)

def aggregation_resume_part3(model, tokenizer, device, total_len, dataset_type="Alzheimer"):
    """Only phase 3"""
    return aggregation_unified(model, tokenizer, device, total_len, dataset_type, 3, 3, True)

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
