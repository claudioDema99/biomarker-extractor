from src.biomarker_extractor import extraction, extraction_unprocessed_lines
from src.models import load_model_and_tokenizer
from src.rag_chroma import validation
from src.aggregator import aggregation, aggregation_resume, aggregation_resume_part1, aggregation_resume_part2, aggregation_resume_part3
import pandas as pd
import sys
import os
import shutil

ALL_DATASETS_LIST = ["Alzheimer", "Bipolar", "BPD", "CN", "Depression", "Dermatitis", "Diabete", "HT", "Hypertension", "KT", "LT", "MS", "Partial_Seizure", "PS00", "PSO01", "PSO02", "Schizophrenia", "Sclerosis"]
DATABASES = ["Alzheimer", "Bipolar", "BPD", "Depression", "Schizophrenia"]

def folder_check(folder_name: str, database: str, clean: bool):
    # Creo o pulisco la cartella dei risultati del seguente database prima di iniziare
    path = f"./{folder_name}/{database}"
    if not os.path.exists(path):
        os.makedirs(path)
        print(f"Cartella '{path}' creata.")
    else:
        if clean:
            # Se passo "clean" come argomento: rimuovo tutti i file e sottocartelle dentro
            for filename in os.listdir(path):
                file_path = os.path.join(path, filename)
                try:
                    if os.path.isfile(file_path) or os.path.islink(file_path):
                        os.unlink(file_path)  # elimina file o link
                    elif os.path.isdir(file_path):
                        shutil.rmtree(file_path)  # elimina cartella ricorsivamente
                except Exception as e:
                    print(f"Errore eliminando {file_path}: {e}")
            print(f"Cartella '{path}' pulita.")
        else:
            print(f"Cartella '{path}' già esistente (non pulita).")
    return

def load_dataset(database: str):
    if database == "BPD":
        df = pd.read_excel(f"./data/{database}_puro.xlsx")
    else:
        df = pd.read_csv(f"./data/{database}_1row_Puri.csv")
    # Filter out rows where both param_value columns are empty/invalid and keep only end_dates before 2025
    # keeps the row indices of the dataset so that we can then identify the rows relating to the biomarkers
    mask = (df['param_value'].notna() | df['param_value_decimal'].notna()) & (pd.to_datetime(df['end_date'], errors='coerce') < '2025-01-01')
    df_filtered = df[mask].copy()
    # Simply store the original indices
    rows_id = df_filtered.index.tolist()
    # add 2 to each rows_id because the csv starts from index 2 (and rows_id from 0)
    rows_id = [x+2 for x in rows_id]

    # keep only the cols selected and removes rows where all columns in the DataFrame are NaN/null
    cols_to_keep = [
        "outcome_measurement_description"
    ]
    df_filtered = df_filtered[cols_to_keep].dropna(how="all")
    return df_filtered, rows_id

def database_selection():
    databases_selected = []
    print("Inserisci il nome del database che si vuole analizzare:\n(se si vogliono analizzare più di un database, inserirli sulla stessa riga separati da spazio o virgola)")
    for database in DATABASES:
        print(f" - '{database}'")
    _database = input("\n ->  ")
    if _database and all(c.isascii() or c.isspace() or c == ',' for c in _database):
        # Sostituisce le virgole con spazi e poi divide
        choice = [x.lower().strip() for x in _database.replace(',', ' ').split()]
    databases_selected.extend(choice)
    # raccolgo gli indici e non rimuovo subito per non causare problemi con gli indici nell'iterazione
    indices_to_remove = []
    for i, database in enumerate(databases_selected):
        if database == "alzheimer":
            databases_selected[i] = "Alzheimer"
        elif database == "bipolar":
            databases_selected[i] = "Bipolar"
        elif database == "bpd":
            databases_selected[i] = "BPD"
        elif database == "depression":
            databases_selected[i] = "Depression"
        elif database == "schizophrenia":
            databases_selected[i] = "Schizophrenia"
        try:
            if databases_selected[i] not in DATABASES:
                raise ValueError(f"Error: {databases_selected[i]} non trovato all'interno della lista dei database disponibili: {DATABASES}")
        except Exception as e:
            print(f"{e}\n-Non verrà dunque processato-")
            indices_to_remove.append(i)
    # Rimuovi dal fondo verso l'inizio
    for i in reversed(indices_to_remove):
        databases_selected.pop(i)
    print(f"Verranno processati in ordine i seguenti database: {databases_selected}")
    return databases_selected

def main():

    clean = False
    if "clean" in sys.argv:
        sys.argv.remove("clean")
        clean = True

    databases_selected = database_selection()

    model, tokenizer, device = load_model_and_tokenizer()
        
    for database in databases_selected:
                
        print(f"\nSelezionato database: {database}")

        # verifica esistenza cartelle /results e /logs e se necessario le pulisce
        folder_check(folder_name="results", database=database, clean=clean)
        folder_check(folder_name="logs", database=database, clean=clean)

        df_filtered, rows_id = load_dataset(database=database)

        # A seconda dell'argomento passato, vengono eseguite diverse sotto parti della pipeline:
        # - 'extraction': solamente la parte di estrazione dei biomarkers
        # - 'validation': solamente la parte di assegnazione di acronimi con RAG (se seguito da un secondo argomento, esegue anche la parte successiva di aggregazione)
        # - 'aggregation': solamente la parte aggregazione dei biomarkers
        # - 'resume': riprende dalla fase di aggregation nella quale vi è la supervisione umana sul merging di gruppi identificati dall'LLM
        # - se nessun dei precedenti argomenti viene passato, si esegue l'intera pipeline
        # - 'clean': cancella tutti i file presenti all'interno della cartella /results per ogni dataset analizzato
        if len(sys.argv) > 1:
            go_on = False
            arg = sys.argv[1].strip()
            if len(sys.argv) > 2:
                go_on = True

            if arg == "extraction":
                # Prima parte: biomarkers extraction
                _ = extraction(model=model, tokenizer=tokenizer, device=device, rows_id=rows_id, df_filtered=df_filtered, dataset_type=database)
                if os.path.exists(f"./results/{database}/unprocessed_lines.txt"):
                    extraction_unprocessed_lines(model=model, tokenizer=tokenizer, device=device, rows_id=rows_id, df_filtered=df_filtered, dataset_type=database)
                    print(f"""\n\nTutto il dataset è stato processato con successo.
La lista dei biomarkers estratti si trova in 'logs/{database}/biomarkers_list.txt'.
Le righe non processate sono state salvate in 'logs/{database}/unprocessed_lines_2.txt' (se il file non esiste, tutte le righe son state processate).
I logs dell'analisi e estrazione dei biomarkers (con biomarkers estratti, row_id, CoT e response dell'LLM) sono stati salvati in 'logs/{database}/extraction_logs.json'.\n""")
                else:
                    print(f"""\n\nTutto il dataset è stato processato con successo.
La lista dei biomarkers estratti si trova in 'logs/{database}/biomarkers_list.txt'.
I logs dell'analisi e estrazione dei biomarkers (con biomarkers estratti, row_id, CoT e response dell'LLM) sono stati salvati in 'logs/{database}/extraction_logs.json'.\n""")

            elif arg == "validation":
                # Seconda parte: validazione dei biomarkers estratti tramite RAG
                _ = validation(model=model, tokenizer=tokenizer, device=device, create_chroma_db=True, dataset_type=database)
                print(f"""\n\nTutti i biomarkers estratti sono stati processati.
I risultati completi (con nome originale, acronimo identificato, row_id, e relativa CoT dell'LLM) si trovano in 'logs/{database}/acronyms_logs.json'.\n""")
                if go_on:
                    # Terza parte: raggruppamento dei biomarkers ripetuti tenendo conto di sinonimi, differenze di nomenclatura e acronimi
                    # per ora faccio solo exact matching
                    _ = aggregation_resume_part1(dataset_type=database)
                    print(f"""\n\nTutti i biomarkers hanno subito la prima fase di analisi e raggruppamento.
I risultati parziali sono stati salvati in 'checkpoints/{database}/'.\n""")
            
            elif arg == "aggregation":
                # Terza parte: raggruppamento dei biomarkers ripetuti tenendo conto di sinonimi, differenze di nomenclatura e acronimi
                _ = aggregation(model=model, tokenizer=tokenizer, device=device, total_len=len(rows_id), dataset_type=database)
                print(f"""\n\nTutti i biomarkers sono stati analizzati e raggruppati.
I risultati completi si trovano in 'results/{database}/biomarkers.json',
i risultati filtrati per frequenza si trovano in 'results/{database}/filtered_biomarkers.json'\n,
mentre la lista di biomarkers non raggruppati si trovano in 'results/{database}/remaining_biomarkers.txt'""")
                
            elif arg == "resume":
                if os.path.exists(f"./checkpoints/parsed_biomarkers_{database}.json") and os.path.exists(f"./checkpoints/acronyms_w_rows_{database}.json"):
                    _ = aggregation_resume_part2(model=model, tokenizer=tokenizer, device=device, total_len=len(rows_id), dataset_type=database)
                    print(f"""\n\nTutti i biomarkers hanno subito la fase di raggruppamento.
I risultati parziali sono stati salvati in 'checkpoints/{database}/'.\n""")
                else:
                    print(f"File parsed_biomarkers_{database}.json or acronyms_w_rows_{database}.json not present in the /checkpoints folder. Skip.")
            
            else:
                print("L'argomento passato non è corretto.\nGli argomenti accettati sono: 'extraction', 'validation', 'aggregation' e 'resume'. Skip.")

        else:
            # Prima parte: biomarkers extraction
            biomarker_list = extraction(model=model, tokenizer=tokenizer, device=device, rows_id=rows_id, df_filtered=df_filtered, dataset_type=database)
            if os.path.exists(f"./results/{database}/unprocessed_lines.txt"):
                extraction_unprocessed_lines(model=model, tokenizer=tokenizer, device=device, rows_id=rows_id, df_filtered=df_filtered, dataset_type=database)
                print(f"""\n\nTutto il dataset è stato processato con successo.
La lista dei biomarkers estratti si trova in 'logs/{database}/biomarkers_list.txt'.
Le righe non processate sono state salvate in 'logs/{database}/unprocessed_lines_2.txt' (se il file non esiste, tutte le righe son state processate).
I logs dell'analisi e estrazione dei biomarkers (con biomarkers estratti, row_id, CoT e response dell'LLM) sono stati salvati in 'logs/{database}/extraction_logs.json'.\n""")
            else:
                print(f"""\n\nTutto il dataset è stato processato con successo.
La lista dei biomarkers estratti si trova in 'logs/{database}/biomarkers_list.txt'.
I logs dell'analisi e estrazione dei biomarkers (con biomarkers estratti, row_id, CoT e response dell'LLM) sono stati salvati in 'logs/{database}/extraction_logs.json'.\n""")
            
            # Seconda parte: validazione dei biomarkers estratti tramite RAG
            evaluated_biomarkers = validation(model=model, tokenizer=tokenizer, device=device, create_chroma_db=True, dataset_type=database)
            print(f"""\n\nTutti i biomarkers estratti sono stati processati.
I risultati completi (con nome originale, acronimo identificato, row_id, e relativa CoT dell'LLM) si trovano in 'logs/{database}/acronyms_logs.json'.\n""")
           
            # Terza parte: raggruppamento dei biomarkers ripetuti tenendo conto di sinonimi, differenze di nomenclatura e acronimi
            biomarkers = aggregation(model=model, tokenizer=tokenizer, device=device, total_len=len(rows_id), dataset_type=database)
            print(f"""\n\nTutti i biomarkers sono stati analizzati e raggruppati.
I risultati completi si trovano in 'results/{database}/biomarkers.json',
i risultati filtrati per frequenza si trovano in 'results/{database}/filtered_biomarkers.json'\n,
mentre la lista di biomarkers non raggruppati si trovano in 'results/{database}/remaining_biomarkers.txt'""")
    
    # Ho diviso la terza fase in due per questioni di praticità, così l'utente esperto può aggregare i gruppi di tutti i dataset uno dietro l'altro
    # altrimenti tra un dataset e l'altro avrebbe dovuto aspettare che l'LLM trovasse le varianti di ciascun gruppo, per poi tronare a raggruppare il dataset successivo
    # Invece prima faccio a raggruppamenti "manuali" di tutti i dataset, poi faccio fare all'LLM l'ultima fase (identificazione delle varianti) di tutti i dataset che son giaà stati raggruppati
    if len(sys.argv) > 1:
        if arg == "resume":
            for database in databases_selected:
                
                if database == "alzheimer":
                    database = "Alzheimer"
                elif database == "bipolar":
                    database = "Bipolar"
                elif database == "bpd":
                    database = "BPD"
                elif database == "depression":
                    database = "Depression"
                elif database == "schizophrenia":
                    database = "Schizophrenia"

                if os.path.exists(f"./checkpoints/parsed_biomarkers_{database}.json") and os.path.exists(f"./checkpoints/acronyms_w_rows_{database}.json"):
                    _ = aggregation_resume_part3(model=model, tokenizer=tokenizer, device=device, total_len=len(rows_id), dataset_type=database)
                    print(f"""\n\nTutti i biomarkers sono stati analizzati e raggruppati.
I risultati completi si trovano in 'results/{database}/biomarkers.json',
i risultati filtrati per frequenza si trovano in 'results/{database}/filtered_biomarkers.json'\n,
mentre la lista di biomarkers non raggruppati si trovano in 'results/{database}/remaining_biomarkers.txt'""")

if __name__ == "__main__":
    main()