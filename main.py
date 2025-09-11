from src.biomarker_extractor import extraction, extraction_unprocessed_lines
from src.models import load_model_and_tokenizer
from src.rag_chroma import validation
from src.aggregator import aggregation, aggregation_resume, aggregation_resume_part1, aggregation_resume_part2, aggregation_resume_part3
import pandas as pd
import sys
import os
import shutil

def main():

    all_databases_list = ["Alzheimer", "Bipolar", "BPD", "CN", "Depression", "Dermatitis", "Diabete", "HT", "Hypertension", "KT", "LT", "MS", "Partial_Seizure", "PS00", "PSO01", "PSO02", "Schizophrenia", "Sclerosis"]
    #databases = ["Alzheimer", "Bipolar", "BPD", "Depression", "Schizophrenia"]
    databases = ["Alzheimer", "BPD", "Schizophrenia", "Depression", "Bipolar"]

    model, tokenizer, device = load_model_and_tokenizer()

    for database in databases:

        # Creo o pulisco la cartella dei risultati del seguente database prima di iniziare
        path_cartella = f"./results/{database}"
        if not os.path.exists(path_cartella):
            os.makedirs(path_cartella)
            print(f"Cartella '{path_cartella}' creata.")
        else:
            if "clean" in sys.argv:
                # Se esiste → rimuovo tutti i file e sottocartelle dentro
                for filename in os.listdir(path_cartella):
                    file_path = os.path.join(path_cartella, filename)
                    try:
                        if os.path.isfile(file_path) or os.path.islink(file_path):
                            os.unlink(file_path)  # elimina file o link
                        elif os.path.isdir(file_path):
                            shutil.rmtree(file_path)  # elimina cartella ricorsivamente
                    except Exception as e:
                        print(f"Errore eliminando {file_path}: {e}")
                print(f"Cartella '{path_cartella}' pulita.")
            else:
                print(f"Cartella '{path_cartella}' già esistente (non pulita).")
            
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
La lista dei biomarkers estratti si trovano in 'results/{database}/biomarkers_list.txt'.
Le righe non processate sono state salvate in 'results/{database}/unprocessed_lines_2.txt' (se il file non esiste, tutte le righe son state processate).
I logs dell'analisi e estrazione dei biomarkers (con biomarkers estratti, row_id, CoT e response dell'LLM) sono stati salvati in 'results/{database}/extraction_logs.json'.\n""")
                else:
                    print(f"""\n\nTutto il dataset è stato processato con successo.
La lista dei biomarkers estratti si trovano in 'results/{database}/biomarkers_list.txt'.
I logs dell'analisi e estrazione dei biomarkers (con biomarkers estratti, row_id, CoT e response dell'LLM) sono stati salvati in 'results/{database}/extraction_logs.json'.\n""")

            elif arg == "validation":
                # Seconda parte: validazione dei biomarkers estratti tramite RAG
                _ = validation(model=model, tokenizer=tokenizer, device=device, create_chroma_db=True, dataset_type=database)
                print(f"""\n\nTutti i biomarkers estratti sono stati processati.
I risultati completi (con nome originale, acronimo identificato, row_id, e relativa CoT dell'LLM) si trovano in 'results/{database}/acronyms_logs.json'.\n""")
                if go_on:
                    # Terza parte: raggruppamento dei biomarkers ripetuti tenendo conto di sinonimi, differenze di nomenclatura e acronimi
                    # per ora faccio solo exact matching
                    _ = aggregation_resume_part1(dataset_type=database)
                    print(f"""\n\nTutti i biomarkers sono stati analizzati e raggruppati.
I risultati finali si trovano in 'results/{database}/biomarkers.json'.\n""")
            
            elif arg == "aggregation":
                # Terza parte: raggruppamento dei biomarkers ripetuti tenendo conto di sinonimi, differenze di nomenclatura e acronimi
                _ = aggregation(model=model, tokenizer=tokenizer, device=device, total_len=len(rows_id), dataset_type=database)
                print(f"""\n\nTutti i biomarkers sono stati analizzati e raggruppati.
I risultati finali si trovano in 'results/{database}/biomarkers.json'.\n""")
                
            elif arg == "resume":
                if os.path.exists(f"./checkpoints/parsed_biomarkers_{database}.json") and os.path.exists(f"./checkpoints/acronyms_w_rows_{database}.json"):
                    _ = aggregation_resume_part2(model=model, tokenizer=tokenizer, device=device, total_len=len(rows_id), dataset_type=database)
                    print(f"""\n\nTutti i biomarkers sono stati analizzati e raggruppati.
I risultati finali si trovano in 'results/{database}/biomarkers.json'.\n""")
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
La lista dei biomarkers estratti si trovano in 'results/{database}/biomarkers_list.txt'.
Le righe non processate sono state salvate in 'results/{database}/unprocessed_lines_2.txt' (se il file non esiste, tutte le righe son state processate).
I logs dell'analisi e estrazione dei biomarkers (con biomarkers estratti, row_id, CoT e response dell'LLM) sono stati salvati in 'results/{database}/extraction_logs.json'.\n""")
            else:
                print(f"""\n\nTutto il dataset è stato processato con successo.
La lista dei biomarkers estratti si trovano in 'results/{database}/biomarkers_list.txt'.
I logs dell'analisi e estrazione dei biomarkers (con biomarkers estratti, row_id, CoT e response dell'LLM) sono stati salvati in 'results/{database}/extraction_logs.json'.\n""")
            
            # Seconda parte: validazione dei biomarkers estratti tramite RAG
            evaluated_biomarkers = validation(model=model, tokenizer=tokenizer, device=device, create_chroma_db=True, dataset_type=database)
            print(f"""\n\nTutti i biomarkers estratti sono stati processati.
I risultati completi (con nome originale, acronimo identificato, row_id, e relativa CoT dell'LLM) si trovano in 'results/{database}/acronyms_logs.json'.\n""")
           
            # Terza parte: raggruppamento dei biomarkers ripetuti tenendo conto di sinonimi, differenze di nomenclatura e acronimi
            biomarkers = aggregation(model=model, tokenizer=tokenizer, device=device, total_len=len(rows_id), dataset_type=database)
            print(f"""\n\nTutti i biomarkers sono stati analizzati e raggruppati.
I risultati finali si trovano in 'results/{database}/biomarkers.json'.\n""")
    
    # Ho diviso la terza fase in due per questioni di praticità, così l'utente esperto può aggregare i gruppi di tutti i dataset uno dietro l'altro
    # altrimenti tra un dataset e l'altro avrebbe dovuto aspettare che l'LLM trovasse le varianti di ciascun gruppo, per poi tronare a raggruppare il dataset successivo
    # Invece prima faccio a raggruppamenti "manuali" di tutti i dataset, poi faccio fare all'LLM l'ultima fase (identificazione delle varianti) di tutti i dataset che son giaà stati raggruppati
    if arg == "resume":
        if os.path.exists(f"./checkpoints/parsed_biomarkers_{database}.json") and os.path.exists(f"./checkpoints/acronyms_w_rows_{database}.json"):
            _ = aggregation_resume_part3(model=model, tokenizer=tokenizer, device=device, total_len=len(rows_id), dataset_type=database)

if __name__ == "__main__":
    main()