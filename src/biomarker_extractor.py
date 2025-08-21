import pandas as pd
import re
from typing import List, Dict, Any
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

def tokens_of(df_slice):
    """Restituisce (token_count, records_json) per un blocco di righe."""
    records_json = process_batch_for_deduplication(df_slice)
    tok_count = get_token_count(records_json)
    return tok_count, records_json

def extraction(df_filtered: pd.DataFrame, dataset_type: str = "Alzheimer"):
    all_biomarkers = []
    all_biomarkers_extended = []
    j = 0  # Indice per debug

    # === BATCH SIZE ADATTIVO: Stabilisci threshold di token number (circa la metà della context window del modello), 
    # Vogliamo passare al modello un batch di righe che abbia un numero di token compreso tra TOK_MIN e TOK_MAX.
    # dunque aumentiamo il numero di righe da processare in un batch fino a che non raggiungiamo TOK_MIN, ma non superiamo TOK_MAX.
    TOK_MIN = 2000          # soglia minima
    TOK_MAX = 3500          # soglia massima

    i = 0
    batch_id = 1

    while i < len(df_filtered):
        # inizia con una sola riga
        rows_in_batch = 1
        batch_df      = df_filtered.iloc[i : i + rows_in_batch]
        batch_tokens, records = tokens_of(batch_df)

        # --- controllo iniziale --- (c'è una cella con più di 6000 token filtrati.. per ora le scartiamo)
        if batch_tokens > TOK_MAX + 500:
            print("\n\n\n_______________________________________________________________________________________\n")
            print(f"[WARNING] Riga {i} supera TOK_MAX ({batch_tokens} > {TOK_MAX}) – saltata.")
            print("_______________________________________________________________________________________\n\n\n")
            with open("RIGHE_NON_PROCESSATE.txt", "a") as f:
                f.write("\n\n________________________________________________________________\n")
                f.write(f"[WARNING] Riga {i} supera TOK_MAX ({batch_tokens} > {TOK_MAX}) – saltata.\n\n")
                f.write(f"{df_filtered.iloc[i]['outcome_measurement_title']}\n")
            i += 1                  # passa alla riga successiva
            continue                # ricomincia il while

        # espandi finché restiamo < TOK_MIN ma non superiamo TOK_MAX
        while batch_tokens < TOK_MIN and (i + rows_in_batch) < len(df_filtered):
            next_row_df = df_filtered.iloc[i + rows_in_batch : i + rows_in_batch + 1]

            tentative_df      = pd.concat([batch_df, next_row_df], ignore_index=True)
            tentative_tokens, tentative_records = tokens_of(tentative_df)

            if tentative_tokens <= TOK_MAX:
                batch_df      = tentative_df
                batch_tokens  = tentative_tokens
                records       = tentative_records
                rows_in_batch += 1
            else:
                break  # aggiungere la riga supererebbe TOK_MAX

        # log e chiamata modello
        print("\n_______________________________________________________________________________________")
        print(f"Batch {batch_id}: {rows_in_batch} righe – {batch_tokens} token")
        # scrivi anche che righe sono attualmente processate per debug
        if rows_in_batch == 1:
            print(f"Indice riga processata: {i}")
        else:
            print(f"Indice righe processate: {i} - {i + rows_in_batch - 1}")
        biomarkers = call_model(records, dataset_type)

        if biomarkers is None or biomarkers == "":
            print(f"\nNESSUN BIOMARKER TROVATO PER IL BATCH {batch_id}\n")
            with open("RIGHE_NON_PROCESSATE.txt", "a") as f:
                f.write("\n\n________________________________________________________________\n")
                f.write(f"[WARNING] Riga {i} Nessun biomarker trovato per il batch {batch_id} – saltata.\n\n")
                f.write(f"{df_filtered.iloc[i]['outcome_measurement_title']}\n")
            i += 1                  # passa alla riga successiva
        else:
            print(f"\nINSERITO BIOMARKERS PER IL BATCH {batch_id}\n")
            print(f"Biomarkers trovati: {biomarkers}")
            all_biomarkers.append(biomarkers)
            all_biomarkers_extended.extend(biomarkers)

        # avanza l’indice; così eviti di ripetere le righe già processate
        i += rows_in_batch
        batch_id += 1
        
        # stampa a che riga siamo arrivati rispetto al totale
        print(f"Righe processate: {i} di {len(df_filtered)} ({i / len(df_filtered) * 100:.2f}%)\n")

        if i == len(df_filtered) // 4:
            with open("./results/liste_biomarkers.txt", "w") as f:
                f.write(f"Righe processate: {i} di {len(df_filtered)} ({i / len(df_filtered) * 100:.2f}%)\n")
                for biomarker in all_biomarkers:
                    f.write(f"\n{biomarker}\n")
        if i == len(df_filtered) // 2:
            with open("./results/liste_biomarkers.txt", "w") as f:
                f.write(f"Righe processate: {i} di {len(df_filtered)} ({i / len(df_filtered) * 100:.2f}%)\n")
                for biomarker in all_biomarkers:
                    f.write(f"\n{biomarker}\n")
        if i == len(df_filtered) * 3 // 4:
            with open("./results/liste_biomarkers.txt", "w") as f:
                f.write(f"Righe processate: {i} di {len(df_filtered)} ({i / len(df_filtered) * 100:.2f}%)\n")
                for biomarker in all_biomarkers:
                    f.write(f"\n{biomarker}\n")
    with open("./results/liste_biomarkers.txt", "w") as f:
        f.write("Tutto il dataset è stato processato.\n")
        for biomarker in all_biomarkers:
            f.write(f"\n{biomarker}\n")
    print("Tutto il dataset è stato processato. Biomarkers salvati in ./results/liste_biomarkers.txt")
    return all_biomarkers, all_biomarkers_extended