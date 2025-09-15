import pandas as pd
import re
from typing import List, Dict, Any
import json
import os
from src.models import call_model, call_model_for_unprocessed_lines, get_token_count, calculate_prompt_tokens

def remove_duplicate_lines_in_cell(cell_content: str) -> str:
    """
    Rimuove righe duplicate all'interno di una cella dopo aver rimosso 
    i prefissi numerici e le sigle OG###.
    
    Pattern da rimuovere: "numero - OG###: " all'inizio di ogni riga
    Esempio: "898604142 - OG000: Treatment Engagement" -> "Treatment Engagement"
    
    Ignora il ";" finale quando confronta le righe per duplicati.
    
    Args:
        cell_content (str): Il contenuto della cella da processare
        
    Returns:
        str: Il contenuto della cella con prefissi rimossi e righe duplicate eliminate
    """
    if pd.isna(cell_content) or not isinstance(cell_content, str):
        return cell_content
    
    lines = cell_content.strip().split('\n')
    if len(lines) <= 1:
        # Anche per una singola riga, rimuovi il prefisso se presente
        if len(lines) == 1:
            cleaned_line = _remove_prefix_from_line(lines[0])
            return cleaned_line if cleaned_line.strip() else cell_content
        return cell_content
    
    # Set per tenere traccia delle righe già viste (dopo pulizia e normalizzazione)
    seen_normalized_lines = set()
    result_lines = []
    
    for line in lines:
        line = line.strip()
        
        # Salta righe vuote
        if not line:
            continue
        
        # Rimuovi il prefisso "numero - OG###: " dalla riga
        cleaned_line = _remove_prefix_from_line(line)
        
        # Se la riga pulita è vuota, salta
        if not cleaned_line.strip():
            continue
        
        # Normalizza la riga per il confronto: rimuovi ";" finale se presente
        normalized_line = _normalize_line_for_comparison(cleaned_line)
        
        # Aggiungi solo se non già vista (confronto normalizzato)
        if normalized_line not in seen_normalized_lines:
            seen_normalized_lines.add(normalized_line)
            result_lines.append(cleaned_line)  # Mantieni la riga originale pulita
    
    return '\n'.join(result_lines)

def _normalize_line_for_comparison(line: str) -> str:
    """
    Normalizza una riga per il confronto di duplicati rimuovendo il ";" finale.
    
    Questo permette di considerare uguali righe che differiscono solo per 
    la presenza/assenza del punto e virgola finale.
    
    Args:
        line (str): La riga da normalizzare
        
    Returns:
        str: La riga normalizzata (senza ";" finale)
    """
    line = line.strip()
    
    # Rimuovi ";" finale se presente
    if line.endswith(';'):
        return line[:-1].strip()
    
    return line

def _remove_prefix_from_line(line: str) -> str:
    """
    Rimuove il prefisso "numero - OG###: " da una singola riga.
    
    Pattern: numeri seguiti da " - OG" + numeri + ": "
    Esempi:
    - "898604142 - OG000: Treatment Engagement" -> "Treatment Engagement"
    - "898604143 - OG001: Treatment Engagement" -> "Treatment Engagement"
    
    Args:
        line (str): La riga da cui rimuovere il prefisso
        
    Returns:
        str: La riga senza prefisso
    """
    line = line.strip()
    
    # Pattern per catturare: numero - OG + numeri + : + contenuto
    # Gruppo 1: cattura tutto dopo "OG###: "
    pattern = r'^\d+\s*-\s*OG\d+\s*:\s*(.+)$'
    match = re.match(pattern, line)
    
    if match:
        return match.group(1).strip()
    else:
        # Se il pattern non corrisponde, restituisci la riga originale
        return line

def process_batch_for_deduplication(batch_df: pd.DataFrame, 
                                  columns_to_process: List[str] = None) -> List[Dict[str, Any]]:
    """
    Processa un batch di righe del DataFrame rimuovendo i prefissi e le righe duplicate 
    nelle colonne specificate.
    
    Args:
        batch_df (pd.DataFrame): Il batch di righe da processare
        columns_to_process (List[str], optional): Lista delle colonne da processare. 
                                                Se None, processa tutte le colonne.
        
    Returns:
        List[Dict[str, Any]]: Lista di record con contenuti puliti e deduplicati
    """
    # Crea una copia del batch per non modificare l'originale
    processed_batch = batch_df.copy()
    
    # Se non specificate, processa tutte le colonne
    if columns_to_process is None:
        columns_to_process = processed_batch.columns.tolist()
    
    # Applica la pulizia e deduplicazione solo alle colonne specificate
    for col in columns_to_process:
        if col in processed_batch.columns:
            processed_batch[col] = processed_batch[col].apply(remove_duplicate_lines_in_cell)
    
    # Converti in dizionario di record
    return processed_batch.to_dict(orient="records")

def save_logs_as_json(log_entries, filepath):
    """Save log entries as pretty-formatted JSON"""
    try:
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(log_entries, f, ensure_ascii=False, indent=2)
    except (IOError, OSError, PermissionError) as e:
        print(f"Error saving logs to {filepath}: {e}")

def tokens_of(df_slice, tokenizer):
    """Restituisce (token_count, records_json) per un blocco di righe."""
    records_json = process_batch_for_deduplication(df_slice)
    tok_count = get_token_count(records_json, tokenizer)
    return tok_count, records_json

def extract_row_from_unprocessed_lines(dataset_type: str):
    """
    Legge un file txt e estrae i numeri dalle righe che iniziano con "Riga $NUMERO$:"
    
    Args:
        file_path (str): Percorso del file da leggere
        
    Returns:
        list: Lista di numeri estratti (come interi)
    """
    row_numbers = []
    file_path = f"./results/{dataset_type}/unprocessed_lines.txt"
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                # Rimuove spazi all'inizio e alla fine
                line = line.strip()
                
                # Controlla se la riga inizia con "Riga $" e contiene il pattern
                if line.startswith("Riga $"):
                    # Usa regex per estrarre il numero tra i dollari
                    match = re.search(r'Riga \$(\d+)\$:', line)
                    if match:
                        number = int(match.group(1))
                        row_numbers.append(number)
    
    except FileNotFoundError:
        print(f"Errore: File '{file_path}' non trovato")
        return []
    except Exception as e:
        print(f"Errore durante la lettura del file: {e}")
        return []
    
    return row_numbers

def extraction(model, tokenizer, device, rows_id: list, df_filtered: pd.DataFrame, dataset_type: str = "Alzheimer"):
    # Validate inputs
    if not isinstance(rows_id, list):
        raise ValueError("rows_id must be a list")
    if len(rows_id) != len(df_filtered):
        raise ValueError("rows_id length must match DataFrame length")
    if df_filtered.empty:
        raise ValueError("DataFrame cannot be empty")

    all_biomarkers_extended = []
    
    # logs
    log_filepath = f"./results/{dataset_type}/extraction_logs.json"
    if os.path.exists(log_filepath):
        with open(log_filepath, "r", encoding="utf-8") as f:
            log_entries = json.load(f)
    else:
        log_entries = []

    # questo TOK_MAX dipende da quanto è lungo system_prompt e user_prompt (fissi) + examples e shots (variano tra dataset types)
    # chiamo funzione in models che me lo calcola e restituisce
    prompt_tokens = calculate_prompt_tokens(task="extraction", dataset_type=dataset_type, tokenizer=tokenizer)
    TOK_MAX = 8000 - prompt_tokens
    print(f"Tokens of the prompt = {prompt_tokens}: TOK_MAX of the rows set to {TOK_MAX}")
    i = 0

    while i < len(df_filtered):
        # I need to do [i:i+1] because I want a class DataFrame (otherwise Pandas gives me a Series)
        # preprocessing row and token count
        batch_tokens, record = tokens_of(df_filtered.iloc[i:i+1], tokenizer)

        # tokens size check
        if batch_tokens > TOK_MAX + 500:
            half_shots = True
            '''
            print(f"[WARNING] Riga {rows_id[i]}: superata TOK_MAX ({batch_tokens} > {TOK_MAX}) – saltata.")
            with open(f"./results/{dataset_type}/unprocessed_lines.txt", "a") as f:
                f.write("\n\n________________________________________________________________\n")
                f.write(f"Riga ${rows_id[i]}$: superata TOK_MAX ({batch_tokens} > {TOK_MAX}) – saltata.\n\n")
                #f.write(f"outcome_measurement_title:\n{record[0]['outcome_measurement_title']}\n")
            i += 1                  # passa alla riga successiva
            continue                # ricomincia il while
            '''
        else:
            half_shots = False

        # log e chiamata modello
        print("\n_______________________________________________________________________________________")
        # TOKEN COUNT DEL SYSTEM E USER PROMPTS DA RIVEDERE SE LO CAMBIO
        print(f"Riga {rows_id[i]}: {batch_tokens} tokens => {batch_tokens + prompt_tokens} tokens totali")
        try:
            data, cot, response = call_model(task="extraction", dataset_type=dataset_type, model=model, tokenizer=tokenizer, device=device, half_shots=half_shots, records=record)
            # Estrai la lista biomarkers, se esiste
            biomarkers = data.get("markers", None)
            if not isinstance(biomarkers, list):
                raise ValueError("Biomarkers is not a list")
        except Exception as e:
            print(f"[WARNING] Model call failed for row {rows_id[i]}: {e}")
            with open(f"./results/{dataset_type}/unprocessed_lines.txt", "a") as f:
                f.write("\n\n________________________________________________________________\n")
                f.write(f"Model call failed for row ${rows_id[i]}$: {e}) – saltata.\n\n")
                f.write(f"{response}\n\n")            
            i += 1
            continue
        if biomarkers is None or biomarkers == "":
            biomarker_value = biomarkers if biomarkers is not None else "None"
            print(f"[WARNING] Riga {rows_id[i]}: nessun biomarker trovato (biomarker = '{biomarker_value}') – saltata.")
            with open(f"./results/{dataset_type}/unprocessed_lines.txt", "a") as f:
                f.write("\n\n________________________________________________________________\n")
                f.write(f"Riga ${rows_id[i]}$: nessun biomarker trovato (biomarker = '{biomarker_value}') – saltata.\n\n")
                f.write(f"{response}\n\n")
            i += 1                  # passa alla riga successiva
        elif biomarkers == []:
            print(f"Empty list of biomarkers, reason: {cot}")
        else:
            print(f"Biomarkers trovati: {biomarkers}")
            all_biomarkers_extended.extend(biomarkers)

        log_entry = {
            "biomarkers": biomarkers if isinstance(biomarkers, list) else [],
            "row_id": rows_id[i],
            "cot": cot,
            "response": response
        }
        log_entries.append(log_entry)
        # Save after each batch
        save_logs_as_json(log_entries, log_filepath)

        # avanza l’indice
        i += 1

        print(f"Righe processate: {i} di {len(df_filtered)} ({i / len(df_filtered) * 100:.2f}%)\n")
        
    with open(f"./results/{dataset_type}/biomarkers_list.txt", "w") as f:
        for biomarker in all_biomarkers_extended:
            f.write(f"{biomarker}\n")
    return all_biomarkers_extended

def extraction_unprocessed_lines(model, tokenizer, device, rows_id, df_filtered: pd.DataFrame, dataset_type: str = "Alzheimer"):
    """
    Funzione che prova ri-processare (estrarre) le righe che non sono state processate la prima volta con la funzione 'extraction'
    """
    all_biomarkers_extended = []

    # logs
    log_filepath = f"./results/{dataset_type}/extraction_logs.json"
    if os.path.exists(log_filepath):
        with open(log_filepath, "r", encoding="utf-8") as f:
            log_entries = json.load(f)
    else:
        log_entries = []

    # Estrai le righe saltate dal file
    unprocessed_rows = extract_row_from_unprocessed_lines("Alzheimer")
    unprocessed_rows_before = len(unprocessed_rows)
    unprocessed_rows_after = 0
    unprocessed_indices = []
    # Risultato: [6, 9, 13, 14, 23, 37]

    # Trova gli indici corrispondenti
    for row in unprocessed_rows:
        if row in rows_id:
            unprocessed_indices.append(rows_id.index(row))

    # questo TOK_MAX dipende da quanto è lungo system_prompt e user_prompt (fissi) + examples e shots (variano tra dataset types)
    # chiamo funzione in models che me lo calcola e restituisce
    prompt_tokens = calculate_prompt_tokens(task="second_extraction", dataset_type=dataset_type, tokenizer=tokenizer)
    TOK_MAX = 8000 - prompt_tokens
    print(f"Tokens of the prompt = {prompt_tokens}: TOK_MAX of the rows set to {TOK_MAX}")

    for index, row in zip(unprocessed_indices, unprocessed_rows): 
        # I need to do [i:i+1] because I want a class DataFrame (otherwise Pandas gives me a Series)
        # preprocessing row and token count
        batch_tokens, record = tokens_of(df_filtered.iloc[index:index+1], tokenizer)

        # tokens size check
        if batch_tokens > TOK_MAX + 500:
            half_shots = True
        else:
            half_shots = True

        # log e chiamata modello
        print("\n_______________________________________________________________________________________")
        # TOKEN COUNT DEL SYSTEM E USER PROMPTS DA RIVEDERE SE LO CAMBIO
        print(f"Riga {row}: {batch_tokens} tokens => {batch_tokens + prompt_tokens} tokens totali")
        try:
            data, cot, response = call_model(task="extraction", dataset_type=dataset_type, model=model, tokenizer=tokenizer, device=device, records=record, low_reasoning=True, half_shots=half_shots)
            # Estrai la lista biomarkers, se esiste
            biomarkers = data.get("markers", None)
            if not isinstance(biomarkers, list):
                raise ValueError("Biomarkers is not a list")
        except Exception as e:
            print(f"[WARNING] Model call failed for row {rows_id[i]}: {e}")
            with open(f"./results/{dataset_type}/unprocessed_lines.txt", "a") as f:
                f.write("\n\n________________________________________________________________\n")
                f.write(f"Model call failed for row ${rows_id[i]}$: {e}) – saltata.\n\n")
                f.write(f"{response}\n\n")            
            i += 1
            continue
        if biomarkers is None or biomarkers == "":
            # Definisci le stringhe di delimitazione
            start_string = "<|channel|>analysis<|message|>"
            end_string = "<|return|>"

            # Trova la posizione della stringa iniziale
            start_pos = response.find(start_string)
            if start_pos != -1:
                # Calcola la posizione dopo la stringa iniziale
                content_start = start_pos + len(start_string)
                
                # Trova la posizione della stringa finale partendo dalla posizione dopo quella iniziale
                end_pos = response.find(end_string, content_start)
                if end_pos != -1:
                    # Estrae il contenuto tra le due stringhe
                    analysis = response[content_start:end_pos]
                    
                else:
                    print("Stringa finale '<|return|>' non trovata")
                    analysis = None
                    
            else:
                print("Stringa iniziale '<|channel|>analysis<|message|>' non trovata")
                analysis = None

            # Se vuoi utilizzare il contenuto estratto più avanti nello script
            if analysis is not None:
                try:
                    biomarkers, cot, response = call_model_for_unprocessed_lines(analysis=analysis, dataset_type=dataset_type, model=model, tokenizer=tokenizer, device=device, low_reasoning=True, half_shots=half_shots)
                    #data, cot, response = call_model(task="second_extraction", dataset_type=dataset_type, model=model, tokenizer=tokenizer, device=device, records=analysis, low_reasoning=True, half_shots=half_shots)
                    # Estrai la lista biomarkers, se esiste
                    #biomarkers = data.get("markers", None)
                    #if not isinstance(biomarkers, list):
                        #raise ValueError("Biomarkers is not a list")
                except Exception as e:
                    print(f"[WARNING] Model call failed for row {rows_id[i]}: {e}")
                    with open(f"./results/{dataset_type}/unprocessed_lines.txt", "a") as f:
                        f.write("\n\n________________________________________________________________\n")
                        f.write(f"Model call failed for row ${rows_id[i]}$: {e}) – saltata.\n\n")
                        f.write(f"{response}\n\n")            
                    i += 1
                    continue         
            if biomarkers is None or biomarkers == "":
                biomarker_value = biomarkers if biomarkers is not None else "None"
                print(f"[WARNING] Riga {row}: nessun biomarker trovato (biomarker = '{biomarker_value}') – saltata.")
                with open(f"./results/{dataset_type}/unprocessed_lines_2.txt", "a") as f:
                    f.write("\n\n________________________________________________________________\n")
                    f.write(f"Riga ${row}$: nessun biomarker trovato (biomarker = '{biomarker_value}') – saltata.\n\n")
                    #f.write(f"outcome_measurement_title:\n{record[0]['outcome_measurement_title']}\n")
                unprocessed_rows_after += 1
            else:
                print(f"Biomarkers trovati: {biomarkers}")
                all_biomarkers_extended.extend(biomarkers)
        elif biomarkers == []:
            print(f"Empty list of biomarkers, reason: {cot}")
        else:
            print(f"Biomarkers trovati: {biomarkers}")
            all_biomarkers_extended.extend(biomarkers)
            

        log_entry = {
            "biomarkers": biomarkers if isinstance(biomarkers, list) else [],
            "row_id": row,
            "cot": cot,
            "response": response
        }
        log_entries.append(log_entry)
        # Save after each batch
        save_logs_as_json(log_entries, log_filepath)

    with open(f"./results/{dataset_type}/biomarkers_list.txt", "a") as f:
        for biomarker in all_biomarkers_extended:
            f.write(f"{biomarker}\n")
    
    print(f"Numero di unprocessed_rows prima del secondo tentativo: {unprocessed_rows_before}\nNumero di unprocessed_rows dopo il secondo tentativo: {unprocessed_rows_after}")

    return
