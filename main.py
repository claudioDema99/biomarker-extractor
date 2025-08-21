from src.biomarker_extractor import extraction
import pandas as pd

def main():
    """
    Main function to execute the biomarker extraction process.
    """
    # === CARICAMENTO DATASET E SELEZIONE COLONNE ===
    #df = pd.read_excel("/home/cdemaria/ALL-EMBRACED/CT/Alzheimer_1row_Puri.xlsx")
    df = pd.read_csv("./data/Alzheimer_1row_Puri.csv")
    cols_to_keep = [
        #"study_type-intervention_type",
        "brief_summary",
        "detailed_description",
        "outcome_measurement_title",
        "outcome_measurement_description"
    ]
    df_filtered = df[cols_to_keep].dropna(how="all")

    # selezione del modello?
    _, biomarker_list = extraction(df_filtered, dataset_type="Alzheimer")
    with open("./results/liste_biomarkers.txt", "w") as f:
        f.write("Tutto il dataset è stato processato.\n")
        for biomarker in biomarker_list:
            f.write(f"\n{biomarker}\n")

if __name__ == "__main__":
    main()