import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
import re
import gc
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
import torch
import json
from src.prompts import get_prompt

EMBEDDING_MODEL = "sentence-transformers/all-mpnet-base-v2"
CHUNK_CHARS = 1500
CHUNK_OVERLAP = 300
MAX_BATCH_SIZE = 2

def clear_gpu_memory():
    """Clear GPU memory and force garbage collection"""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()

def save_logs_as_json(log_entries, filepath, mode="w"):
    """Save log entries as pretty-formatted JSON"""
    with open(filepath, mode, encoding="utf-8") as f:
        json.dump(log_entries, f, ensure_ascii=False, indent=2)

# Cleaning functions
def remove_after_last_occurrence(text, keyword):
    idx = text.lower().rfind(keyword.lower())
    return text[:idx] if idx != -1 else text

def clean_paper_text(text: str) -> str:
    # Step 1: Remove everything before the first occurrence of "abstract" (case insensitive)
    text = re.sub(r"^.*?(abstract)", r"\1", text, flags=re.IGNORECASE | re.DOTALL)
    # Step 2.1: Remove lines that contain only a single number or punctuation
    text = re.sub(r"(?m)^\s*[\d\W]+\s*$", "", text)
    # Step 2.2: Replace multiple consecutive newlines with a single newline
    text = re.sub(r"\n+", "\n", text)
    # Step 3: Remove "open in a new tab" (case insensitive)
    text = re.sub(r"open in a new tab", "", text, flags=re.IGNORECASE)
    text = remove_after_last_occurrence(text, "references")
    return text.strip()

def call_model(biomarker, records, model, tokenizer, device, max_retries=5):
    name, _, _ = get_prompt("Alzheimer")
    for attempt in range(max_retries):
        try:
            # Clear memory before each attempt
            if attempt > 0:
                print(f" Memory cleared for retry {attempt + 1}")
            # Adatto i prompt in base al database
            system_prompt = f"""You are an expert in {name}'s markers and their standard nomenclature. Your task is to find the correct, most used, and appropriate acronym for the marker I give you. You must:

1. **ACRONYM IDENTIFICATION**: Provide the most commonly used and standardized acronym/abbreviation for that marker in scientific literature
2. **STANDARDIZATION**: Use the most widely accepted scientific nomenclature

**Required response format:**
For each marker, use this JSON structure:
{{
  "original_name": "name as provided in the list",
  "acronym": "STANDARD_ACRONYM"
}}

If multiple acronyms exist for the same marker, provide the most commonly used one in current scientific literature. Focus on finding the appropriate acronym rather than validating the marker itself."""

            user_prompt = f"""Find the correct and most appropriate acronym for the marker I give you.

CONTEXT FROM DOCUMENTS:
{records}

MARKER:
{biomarker}

Identify the most commonly used and standardized acronym/abbreviation mentioned in the provided documents or based on standard scientific nomenclature.
Provide the answer in the JSON format specified in the system instructions. Focus solely on finding the appropriate acronym."""

            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ]

            full_prompt = tokenizer.apply_chat_template(
                messages, 
                tokenize=False, 
                add_generation_prompt=True
            )

            '''
            # se vuoi impostare il reasoning su low invece che medium
            if full_prompt.count(" medium") > 0:
                full_prompt = full_prompt.replace("medium", "low", 1)
                print("\n\nChanged reasoning level to low for better performance.\n")
            '''

            inputs = tokenizer(
                full_prompt, 
                return_tensors="pt", 
                padding=True, 
                truncation=True,
                max_length=8192  # Context window di gpt-oss-20b
            ).to(device)

            print(f"--- Waiting for the model response ---")
            # Generazione della risposta
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=4096,          # Massimo numero di nuovi token da generare
                    do_sample=False,              # Usa greedy decoding (deterministic!!!)
                    #temperature=0.7,             # Controllo randomness (se do_sample=True) quindi inutile
                    #top_p=0.9,                   # Nucleus sampling (se do_sample=True) quindi inutile
                    pad_token_id=tokenizer.eos_token_id,  # Token di padding
                    eos_token_id=tokenizer.eos_token_id,  # Token di fine sequenza
                    use_cache=True,               # usiamola
                    repetition_penalty=1.1,       # Penalità per ripetizioni
                    length_penalty=1.0            # Penalità per lunghezza
                )

            input_length = inputs.input_ids.shape[1]
            generated_tokens = outputs[0][input_length:]

            output_text = tokenizer.decode(
                generated_tokens, 
                skip_special_tokens=False,         # Mi servono per filtrare il ragionamento
                clean_up_tokenization_spaces=True
            )

            # chat template di gpt-oss: dividiamo CoT da risposta
            final_start = '<|end|><|start|>assistant<|channel|>final<|message|>'
            if final_start in output_text:
                parts = output_text.split(final_start)
                cot = final_start.join(parts[:-1])  # Tutto prima del tag finale
                response = parts[-1].strip()              # Solo la risposta finale
                cot = (
                    cot
                    .replace('<|channel|>analysis<|message|>', '')
                    .replace('<|end|>', '')
                    .strip()
                )
                response = (
                    response
                    .replace('<|return|>', '')
                    .strip()
                )
            else:
                cot = ""
                response = output_text.strip()
                # ritorna l'output completo se non trova i tag
                print(f"No filtering tags found, returning full output.")
            # QUESTO FUNZIONA SOLO SE HO UN SOLO BIOMARKER, SE NE HO PIÙ DI UNO DEVO CERCARE [] IN QUANTO ESTRAGGO LA LISTA DI DICT
            start = response.find('{')
            end   = response.rfind('}') + 1
            if start == -1 or end == 0:
                raise ValueError("JSON structure not found in response")
            response_json = response[start:end]
            data = json.loads(response_json)
            # Se arriviamo qui, tutto è andato bene
            if attempt > 0:
                print(f" Successo al tentativo {attempt + 1}")
            return data, cot, response

        except torch.cuda.OutOfMemoryError as e:
            print(f" OOM Error - Tentativo {attempt + 1}/{max_retries}: {e}")
            
            if attempt == max_retries - 1:
                print(f" Tutti i {max_retries} tentativi falliti per OOM. Ritorno liste vuote.")
                return {}, "", ""
            else:
                print(" Cleaning memory and retrying...")
                
        except Exception as e:
            print(f" Generic Error - Tentativo {attempt + 1}/{max_retries}: {e}")
            
            if attempt == max_retries - 1:
                print(f" Tutti i {max_retries} tentativi falliti. Ritorno liste vuote.")
                return {}, "", ""
            else:
                print(" Riprovo...")

    return {}, "", ""

def validation(model, tokenizer, device, biomarkers, create_chroma_db=False, dataset_type: str="Alzheimer"):
    
    # create chroma db (just first time)
    if create_chroma_db:
        # load and clean all PDFs
        pdf_folder = f"./docs/{dataset_type}"  # path to your 14 PDFs
        all_cleaned_docs = []
        for pdf_file in os.listdir(pdf_folder):
            if pdf_file.endswith(".pdf"):
                loader = PyPDFLoader(os.path.join(pdf_folder, pdf_file))
                raw_pages = loader.load()
                # Concatenate all text from PDF pages
                raw_text = "\n".join([p.page_content for p in raw_pages])
                cleaned_text = clean_paper_text(raw_text)
                all_cleaned_docs.append(cleaned_text)
        print(f"Loaded and cleaned {len(all_cleaned_docs)} papers")

        # split into chunks
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1500,
            chunk_overlap=300
        )
        docs = text_splitter.create_documents(all_cleaned_docs)
        print(f"Created {len(docs)} chunks")

        # create embeddings + Chroma collection
        embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
        vectorstore = Chroma.from_documents(
            documents=docs,
            embedding=embedding_model,
            persist_directory=f"./db/chroma_db_{dataset_type}"
        )
        vectorstore.persist()
        print(f"Chroma collection {vectorstore._persist_directory} created and persisted")
    else:
        # Load persisted collection
        vectorstore = Chroma(
            persist_directory=f"./db/chroma_db_{dataset_type}",
            embedding_function=embedding_model
        )
        print(f"Chroma collection {vectorstore._persist_directory} loaded")

    # Re-use the same embedding model you used when building the DB
    embedding_model = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)

    # Turn into a retriever
    retriever = vectorstore.as_retriever(search_kwargs={"k": 4})

    # for testing
    if biomarkers == [] or biomarkers == None:
        with open (f"./results/{dataset_type}/biomarkers_list.txt", "r") as f:
            biomarkers = [line.strip() for line in f if line.strip()]
        # creo una lista di couple: ogni couple è formata da un biomarker estratto e dalla riga del dataset dalla quale il biomarker è stato estratto
        # in questo modo, alla fine della pipeline posso risalire alla/e riga/righe nelle quali è presente il biomarkers estratto
        # DA CHEKCARE!! PER ORA NON FUNZIONA PERCHÈ extraction_logs.json È ANCORA DA AGGIORNARE
    with open(f"./results/{dataset_type}/extraction_logs.json", "r", encoding="utf-8") as f:
        data = json.load(f)
        biomarkers_w_rows = []
        for d in data:
            for i in range(len(d["biomarkers"])):
                biomarkers_w_rows.append((d["biomarkers"][i], d["row_id"]))

    LOG_PATH = f"./results/{dataset_type}/acronyms_logs.json"
    if os.path.exists(LOG_PATH):
        with open(LOG_PATH, "r", encoding="utf-8") as f:
            log_entries = json.load(f)
    else:
        log_entries = []
    
    # process each biomarkers independently, with the row_ids
    for couples in biomarkers_w_rows:
        biomarker, row_id = couples
        query = f"What's the correct, most used, and appropriate acronym for '{biomarker}'?"
        results = retriever.get_relevant_documents(query)
        cleaned_results = " - ".join([result.page_content for result in results])
        cleaned_results = re.sub(f"\n", "", cleaned_results)
        data_json, cot, response = call_model(biomarker=biomarker, records=cleaned_results, model=model, tokenizer=tokenizer, device=device)
        if data_json:
            log_entry = {
                "original_name": data_json["original_name"],
                "acronym": data_json["acronym"],
                "row_id": row_id,
                "cot": cot
            }
            log_entries.append(log_entry)
            save_logs_as_json(log_entries, LOG_PATH)
            print(f"Biomarker -> {biomarker}")
            print(f"Response:\n{response}")
        else:
            print(f"{biomarker} not processed: skipping.")
