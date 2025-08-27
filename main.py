from src.biomarker_extractor import extraction
from src.models import load_model_and_tokenizer
from src.rag_faiss import validation
from src.aggregator import aggregation
import pandas as pd

def main():
    # Caricamento dataset e selezione colonne
    df = pd.read_csv("./data/Alzheimer_1row_Puri.csv")
    cols_to_keep = [
        #"study_type-intervention_type",
        "brief_summary",
        "detailed_description",
        "outcome_measurement_title",
        "outcome_measurement_description"
    ]
    df_filtered = df[cols_to_keep].dropna(how="all")

    model, tokenizer, device = load_model_and_tokenizer()

    # Prima parte: biomarkers extraction
    _, biomarker_list = extraction(model=model, tokenizer=tokenizer, device=device, df_filtered=df_filtered, dataset_type="Alzheimer")
    with open("./results/biomarkers_list.txt", "w") as f:
        for biomarker in biomarker_list:
            f.write(f"{biomarker}\n")
    print("""\n\nTutto il dataset è stato processato con successo.\n
I risultati dei biomarkers estratti si trovano in 'results/biomarkers_list.txt'.\n
Le righe non processate sono state salvate in 'results/unprocessed_lines.txt'.\n
I logs dell'analisi e estrazione dei biomarkers (con batch_id, rows_ids, CoT e response dell'LLM) sono stati salvati in 'results/extraction_logs.jsonl'.\n\n""")
    
    # Seconda parte: validazione dei biomarkers estratti tramite RAG
    evaluated_biomarkers = validation(model=model, tokenizer=tokenizer, device=device, biomarkers=biomarker_list)
    print("""\n\nTutti i biomarkers estratti sono stati valutati.\n
I risultati completi della valutazione dei singoli biomarkers estratti si trovano in 'results/evaluated_biomarkers.jsonl'.\n
I biomarkers scartati (insieme alle CoT e risposte del modello) si trovano in 'results/not_validated_logs.jsonl'.\n\n""")
    
    # Terza parte: raggruppare i biomarkers ripetuti tenendo conto di sinonimi, differenze di nomenclatura e acronimi
    biomarkers = aggregation(model=model, tokenizer=tokenizer, device=device, evaluated_biomarkers=evaluated_biomarkers)
    print("""\n\nTutti i biomarkers validati sono stati raggruppati tenendo considerando i vari sinonimi, differenze di nomenclatura e acronimi.\n
I risultati si trovano in 'results/biomarkers.txt'.\n\n""")

if __name__ == "__main__":
    main()