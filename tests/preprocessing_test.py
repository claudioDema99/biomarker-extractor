#%%
import pandas as pd
import re
from typing import List, Dict, Any

import sys
import os

# Aggiungi la cartella parent al path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Ora puoi importare normalmente
from src.models import get_token_count

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

def write_debug():
    with open('degub_preprocessing.txt', 'a', encoding='utf-8') as file:
        # Scrivi l'header
        file.write(str(df.columns.tolist()) + '\n\n')
        # Scrivi ogni riga
        for index, row in df.iterrows():
            file.write(f"Riga {index}:\n")
            for col in df.columns:
                file.write(f"  {col}: {row[col]}\n")
            file.write('\n___________________________________________________\n')
    return


#%%
import random

df = pd.read_csv("../data/Alzheimer_1row_Puri.csv")
cols_to_keep = [
    #"study_type-intervention_type",
    "outcome_measurement_title",
    "outcome_measurement_description"
]
df = df[cols_to_keep].dropna(how="all")

#%%

n = random.randint(1, 350)
df = df.sample(n=1, random_state=n)
token_before = get_token_count(df.to_dict(orient="records"))
write_debug()
df = pd.DataFrame(process_batch_for_deduplication(df))
print(f"From {token_before} to {get_token_count(df.to_dict(orient='records'))} tokens after deduplication")
write_debug()

# %%
