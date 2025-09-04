import pandas as pd
import re
from typing import List, Dict, Any
import json
import os
from src.models import call_model, get_token_count

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
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(log_entries, f, ensure_ascii=False, indent=2)

def tokens_of(df_slice, tokenizer):
    """Restituisce (token_count, records_json) per un blocco di righe."""
    records_json = process_batch_for_deduplication(df_slice)
    tok_count = get_token_count(records_json, tokenizer)
    return tok_count, records_json

def extraction(model, tokenizer, device, df_filtered: pd.DataFrame, dataset_type: str = "Alzheimer"):
    all_biomarkers_extended = []
    
    # logs
    log_filepath = "./results/extraction_logs.json"
    if os.path.exists(log_filepath):
        with open(log_filepath, "r", encoding="utf-8") as f:
            log_entries = json.load(f)
    else:
        log_entries = []

    TOK_MAX = 4000
    i = 0

    while i < len(df_filtered):
        # I need to do [i:i+1] because I want a class DataFrame (otherwise Pandas gives me a Series)
        # preprocessing row and token count
        batch_tokens, record = tokens_of(df_filtered.iloc[i:i+1], tokenizer)

        # DA TESTARE record[0]['outcome_measurement_title] MA NON CREDO SIA ERRORE
        # tokens size check
        if batch_tokens > TOK_MAX + 500:
            print(f"[WARNING] Riga {i} supera TOK_MAX ({batch_tokens} > {TOK_MAX}) – saltata.")
            with open("./results/unprocessed_lines.txt", "a") as f:
                f.write("\n\n________________________________________________________________\n")
                f.write(f"Riga {i} supera TOK_MAX ({batch_tokens} > {TOK_MAX}) – saltata.\n\n")
                f.write(f"outcome_measurement_title:\n{record[0]['outcome_measurement_title']}\n")
            i += 1                  # passa alla riga successiva
            continue                # ricomincia il while

        # log e chiamata modello
        print("\n_______________________________________________________________________________________")
        # TOKEN COUNT DEL SYSTEM E USER PROMPTS DA RIVEDERE SE LO CAMBIO
        print(f"Riga {i}: {batch_tokens} tokens => {batch_tokens + 3492} tokens totali") # 3492 è il numero di tokens del system + user prompts senza righe del dataset
        biomarkers, cot, response = call_model(record, dataset_type, model, tokenizer, device)

        if biomarkers is None or biomarkers == "":
            with open("./results/unprocessed_lines.txt", "a") as f:
                f.write("\n\n________________________________________________________________\n")
                f.write(f"Nessun biomarker trovato per la riga {i} – saltata.\n\n")
                f.write(f"outcome_measurement_title:\n{record[0]['outcome_measurement_title']}\n")
            i += 1                  # passa alla riga successiva
        else:
            print(f"Biomarkers trovati: {biomarkers}")
            all_biomarkers_extended.extend(biomarkers)

        log_entry = {
            "biomarkers": biomarkers if isinstance(biomarkers, list) else [],
            "row_id": i,
            "cot": cot,
            "response": response
        }
        log_entries.append(log_entry)
        # Save after each batch
        save_logs_as_json(log_entries, log_filepath)

        # avanza l’indice
        i += 1

        print(f"Righe processate: {i} di {len(df_filtered)} ({i / len(df_filtered) * 100:.2f}%)\n")
        
    with open("./results/biomarkers_list.txt", "w") as f:
        for biomarker in all_biomarkers_extended:
            f.write(f"{biomarker}\n")
    return all_biomarkers_extended