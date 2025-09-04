from src.biomarker_extractor import extraction
from src.models import load_model_and_tokenizer
#from src.rag_faiss import validation
from src.rag_chroma import validation
from src.aggregator import aggregation
import pandas as pd

def main():

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
I logs dell'analisi e estrazione dei biomarkers (con batch_id, rows_ids, CoT e response dell'LLM) sono stati salvati in 'results/extraction_logs.json'.\n""")
    
    # Seconda parte: validazione dei biomarkers estratti tramite RAG
    evaluated_biomarkers = validation(model=model, tokenizer=tokenizer, device=device, biomarkers=biomarker_list)
    print("""\n\nTutti i biomarkers estratti sono stati valutati.
I risultati completi della valutazione dei singoli biomarkers estratti si trovano in 'results/evaluated_biomarkers.json'.
I biomarkers scartati (insieme alle CoT e risposte del modello) si trovano in 'results/not_validated_logs.json'.\n""")
    
    # Terza parte: raggruppamento dei biomarkers ripetuti tenendo conto di sinonimi, differenze di nomenclatura e acronimi
    biomarkers = aggregation(model=model, tokenizer=tokenizer, device=device, evaluated_biomarkers=evaluated_biomarkers)
    print("""\n\nTutti i biomarkers validati sono stati raggruppati considerando i vari sinonimi, differenze di nomenclatura e acronimi.
I risultati si trovano in 'results/biomarkers.json'.\n""")
    
    if input("\nVuoi stampare a video i risultati? (Sì/no):   ").lower() in ("sì", "si"):
        for biomarker in biomarkers:
            print(biomarker)

if __name__ == "__main__":
    main()