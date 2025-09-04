from src.biomarker_extractor import extraction
from src.models import load_model_and_tokenizer
#from src.rag_faiss import validation
from src.rag_chroma import validation
from src.aggregator import aggregation
import pandas as pd

def main():

    databases_list = ["Alzheimer", "Bipolar", "BPD", "CN", "Depression", "Dermatitis", "Diabete", "HT", "Hypertension", "KT", "LT", "MS", "Partial_Seizure", "PS00", "PSO01", "PSO02", "Schizophrenia", "Sclerosis"]

    df = pd.read_csv("./data/Alzheimer_1row_Puri.csv")

    # Filter out rows where both param_value columns are empty/invalid and keep only end_dates before 2025
    df_filtered = df[
        (df['param_value'].notna() | df['param_value_decimal'].notna()) &
        (pd.to_datetime(df['end_date'], errors='coerce') < '2025-01-01')
    ].copy()

    # keep only the cols selected and removes rows where all columns in the DataFrame are NaN/null
    cols_to_keep = [
        "outcome_measurement_title",
        "outcome_measurement_description"
    ]
    df_filtered = df_filtered[cols_to_keep].dropna(how="all")

    model, tokenizer, device = load_model_and_tokenizer()

    # Prima parte: biomarkers extraction
    biomarker_list = extraction(model=model, tokenizer=tokenizer, device=device, df_filtered=df_filtered, dataset_type="Alzheimer")
    print("""\n\nTutto il dataset è stato processato con successo.
I risultati dei biomarkers estratti si trovano in 'results/biomarkers_list.txt'.
Le righe non processate sono state salvate in 'results/unprocessed_lines.txt'.
I logs dell'analisi e estrazione dei biomarkers (con biomarkers estratti, row_id, CoT e response dell'LLM) sono stati salvati in 'results/extraction_logs.json'.\n""")
    
    # Seconda parte: validazione dei biomarkers estratti tramite RAG
    evaluated_biomarkers = validation(model=model, tokenizer=tokenizer, device=device, biomarkers=biomarker_list)
    print("""\n\nTutti i biomarkers estratti sono stati processati.
I risultati completi (con nome originale, acronimo identificato e relativa CoT dell'LLM) si trovano in 'results/acronyms_logs.json'.\n""")
    
    # Terza parte: raggruppamento dei biomarkers ripetuti tenendo conto di sinonimi, differenze di nomenclatura e acronimi
    biomarkers = aggregation(model=model, tokenizer=tokenizer, device=device, evaluated_biomarkers=evaluated_biomarkers)
    print("""\n\nTutti i biomarkers sono stati raggruppati considerando i vari sinonimi, differenze di nomenclatura e acronimi.
I risultati finali si trovano in 'results/biomarkers.json'.\n""")
    
    if input("\nVuoi stampare a video i risultati finali? (Sì/no):   ").lower() in ("sì", "si"):
        for biomarker in biomarkers:
            print(biomarker)

if __name__ == "__main__":
    main()