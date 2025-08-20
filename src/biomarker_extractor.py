import pandas as pd
import re
from typing import List, Dict, Any
from src.models import call_model, get_token_count

def remove_duplicate_lines_in_cell(cell_content: str) -> str:
    """
    Rimuove righe duplicate o quasi-duplicate all'interno di una cella.
    
    Considera duplicate:
    - Righe completamente identiche
    - Righe identiche eccetto per un numero iniziale seguito da " - "
    
    Args:
        cell_content (str): Il contenuto della cella da processare
        
    Returns:
        str: Il contenuto della cella con le righe duplicate rimosse
    """
    if pd.isna(cell_content) or not isinstance(cell_content, str):
        return cell_content
    
    lines = cell_content.strip().split('\n')
    if len(lines) <= 1:
        return cell_content
    
    # Dizionario per raggruppare righe simili
    # Chiave: riga normalizzata (senza numero iniziale)
    # Valore: riga originale da mantenere
    normalized_lines = {}
    
    for line in lines:
        line = line.strip()
        if not line:  # Mantieni righe vuote
            if "" not in normalized_lines:
                normalized_lines[""] = line
            continue
        
        # Pattern per numero iniziale seguito da " - "
        # Cerca numero di una o più cifre all'inizio, seguito da " - "
        match = re.match(r'^\d+\s*-\s*(.+)$', line)
        
        if match:
            # Riga con pattern "numero - testo"
            normalized_key = match.group(1).strip()
        else:
            # Riga normale senza pattern
            normalized_key = line
        
        # Mantieni solo la prima occorrenza di ogni riga normalizzata
        if normalized_key not in normalized_lines:
            normalized_lines[normalized_key] = line
    
    # Ricostruisci il contenuto mantenendo l'ordine originale
    result_lines = []
    seen_normalized = set()
    
    for line in lines:
        line = line.strip()
        
        if not line:
            if "" not in seen_normalized:
                result_lines.append(line)
                seen_normalized.add("")
            continue
        
        # Determina la chiave normalizzata
        match = re.match(r'^\d+\s*-\s*(.+)$', line)
        normalized_key = match.group(1).strip() if match else line
        
        # Aggiungi solo se non già visto
        if normalized_key not in seen_normalized:
            result_lines.append(line)
            seen_normalized.add(normalized_key)
    
    return '\n'.join(result_lines)

def process_batch_for_deduplication(batch_df: pd.DataFrame) -> List[Dict[str, Any]]:
    """
    Processa un batch di righe del DataFrame rimuovendo le righe duplicate nelle colonne specificate.
    
    Args:
        batch_df (pd.DataFrame): Il batch di righe da processare
        columns_to_process (List[str]): Lista delle colonne da processare per la deduplicazione
        
    Returns:
        List[Dict[str, Any]]: Lista di record con contenuti deduplicati
    """
    # Crea una copia del batch per non modificare l'originale
    processed_batch = batch_df.copy()
    
    # Applica la deduplicazione solo alle colonne specificate
    for col in processed_batch.columns:
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
    j = 0  # Indice per debug

    # === BATCH SIZE ADATTIVO: Stabilisci threshold di token number (circa la metà della context window del modello), 
    # Vogliamo passare al modello un batch di righe che abbia un numero di token compreso tra TOK_MIN e TOK_MAX.
    # dunque aumentiamo il numero di righe da processare in un batch fino a che non raggiungiamo TOK_MIN, ma non superiamo TOK_MAX.
    TOK_MIN = 2500          # soglia minima
    TOK_MAX = 4000          # soglia massima

    i = 0
    batch_id = 1

    while i < len(df_filtered):
        # inizia con una sola riga
        rows_in_batch = 1
        batch_df      = df_filtered.iloc[i : i + rows_in_batch]
        batch_tokens, records = tokens_of(batch_df)

        # --- controllo iniziale --- (c'è una cella con più di 6000 token filtrati.. per ora le scartiamo)
        if batch_tokens > TOK_MAX:
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
        else:
            print(f"\nINSERITO BIOMARKERS PER IL BATCH {batch_id}\n")
            print(f"Biomarkers trovati: {biomarkers}")
            all_biomarkers.append(biomarkers)

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
    return all_biomarkers