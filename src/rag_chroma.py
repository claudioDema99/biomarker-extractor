import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
import re
import gc
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma 
import torch
import json
from src.models import call_model

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

def validation(model, tokenizer, device, create_chroma_db=False, dataset_type: str="Alzheimer"):
    
    # create chroma db (just first time)
    if create_chroma_db:
        # load and clean all PDFs
        pdf_folder = f"./docs/{dataset_type}"  # path to your 14 PDFs
        if not os.path.exists(pdf_folder):
            raise FileNotFoundError(f"PDF folder not found: {pdf_folder}")
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
        persist_dir = f"./db/chroma_db_{dataset_type}"

        # create embeddings + Chroma collection
        embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
        vectorstore = Chroma.from_documents(
            documents=docs,
            embedding=embedding_model,
            persist_directory=persist_dir
        )
        print(f"Chroma collection {persist_dir} created and persisted")
    else:
        # Load persisted collection
        vectorstore = Chroma(
            persist_directory=persist_dir,
            embedding_function=embedding_model
        )
        print(f"Chroma collection {persist_dir} loaded")

    # Re-use the same embedding model you used when building the DB
    embedding_model = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)

    # Turn into a retriever
    retriever = vectorstore.as_retriever(search_kwargs={"k": 4})

    # creo una lista di couple: ogni couple è formata da un biomarker estratto e dalla riga del dataset dalla quale il biomarker è stato estratto
    # in questo modo, alla fine della pipeline posso risalire alla/e riga/righe nelle quali è presente il biomarkers estratto
    try:
        with open(f"./logs/{dataset_type}/extraction_logs.json", "r", encoding="utf-8") as f:
            data = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError) as e:
        print(f"Error loading extraction logs: {e}")
        return 
    biomarkers_w_rows = []
    for d in data:
        for i in range(len(d["biomarkers"])):
            biomarkers_w_rows.append((d["biomarkers"][i], d["row_id"]))

    LOG_PATH = f"./logs/{dataset_type}/acronyms_logs.json"
    if os.path.exists(LOG_PATH):
        with open(LOG_PATH, "r", encoding="utf-8") as f:
            log_entries = json.load(f)
    else:
        log_entries = []
    
    # process each biomarkers independently, with the row_ids
    for couples in biomarkers_w_rows:
        biomarker, row_id = couples
        query = f"What's the correct, most used, and appropriate acronym for '{biomarker}'?"
        try:
            results = retriever.invoke(query)
            if not results:
                print(f"Warning: No relevant documents found for '{biomarker}'")
                # Could try alternative queries or skip
                continue
        except Exception as e:
            print(f"Retrieval error for '{biomarker}': {e}")
            continue
        cleaned_results = " - ".join([result.page_content for result in results])
        cleaned_results = re.sub(f"\n", "", cleaned_results)
        data_json, cot, response = call_model(task="acronyms", dataset_type=dataset_type, model=model, tokenizer=tokenizer, device=device, biomarkers=biomarker, records=cleaned_results, low_reasoning=True)
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
