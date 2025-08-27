import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
import glob
import pickle
from typing import List, Dict, Tuple
import gc
import fitz  # PyMuPDF
import numpy as np
import faiss
import torch
import json
from sentence_transformers import SentenceTransformer

# ---------- Config ----------
EMBEDDING_MODEL = "sentence-transformers/all-mpnet-base-v2"  # buona qualità
CHUNK_CHARS = 1500
CHUNK_OVERLAP = 300
FAISS_INDEX_PATH = "data/indices/faiss_index.bin"
METADATA_PATH = "data/indices/faiss_metadata.pkl"
MAX_BATCH_SIZE = 2
# ----------------------------

def clear_gpu_memory():
    """Clear GPU memory and force garbage collection"""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()

def extract_text_from_pdf(pdf_path: str) -> str:
    doc = fitz.open(pdf_path)
    pages = []
    for p in doc:
        pages.append(p.get_text("text"))
    doc.close()
    return "\n".join(pages)

def chunk_text(text: str, chunk_chars: int = CHUNK_CHARS, overlap: int = CHUNK_OVERLAP) -> List[str]:
    chunks = []
    start = 0
    text_len = len(text)
    while start < text_len:
        end = min(start + chunk_chars, text_len)
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end == text_len:
            break
        start = max(0, end - overlap)
    return chunks

def build_embeddings_and_index(pdf_folder: str,
                               embedding_model_name: str = EMBEDDING_MODEL,
                               index_path: str = FAISS_INDEX_PATH,
                               metadata_path: str = METADATA_PATH,
                               batch_size: int = 32):
    # dispositivo per sentence-transformers: preferisci GPU (ROCm) se disponibile
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device for embeddings: {device}")

    model = SentenceTransformer(embedding_model_name, device=device)

    # nostri chunks e metadati associati
    all_chunks = []
    all_meta = []
    for pdf_file in glob.glob(os.path.join(pdf_folder, "*.pdf")):
        text = extract_text_from_pdf(pdf_file)
        chunks = chunk_text(text)
        for i, c in enumerate(chunks):
            all_chunks.append(c)
            meta = {"source": os.path.basename(pdf_file), "chunk_id": i}
            all_meta.append(meta)
        
        del text, chunks
        clear_gpu_memory()

    # Process embeddings in batches
    all_embeddings = []
    for i in range(0, len(all_chunks), batch_size):
        batch_chunks = all_chunks[i:i+batch_size]
        print(f"Processing embeddings batch {i//batch_size + 1}/{(len(all_chunks)-1)//batch_size + 1}")
        
        with torch.no_grad():
            batch_embeddings = model.encode(batch_chunks, 
                                          show_progress_bar=False, 
                                          convert_to_numpy=True)
        all_embeddings.append(batch_embeddings)
        clear_gpu_memory()

    # Concatenate all embeddings
    embeddings = np.concatenate(all_embeddings, axis=0)
    del all_embeddings  # libera memoria

    # normalizza per usare coseno (IndexFlatIP)
    faiss.normalize_L2(embeddings)

    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)

    faiss.write_index(index, index_path)
    with open(metadata_path, "wb") as f:
        pickle.dump({"chunks": all_chunks, "meta": all_meta}, f)

    print(f"Index creato: {index.ntotal} vettori, dim={dim}")
    return index, all_chunks, all_meta

def load_index_and_metadata(index_path: str = FAISS_INDEX_PATH, metadata_path: str = METADATA_PATH):
    index = faiss.read_index(index_path)
    with open(metadata_path, "rb") as f:
        data = pickle.load(f)
    return index, data["chunks"], data["meta"]

def retrieve(query: str, index, chunk_texts: List[str], metas: List[Dict], model: SentenceTransformer,
             top_k: int = 5) -> List[str]:
    """Optimized retrieve function that returns only text chunks"""
    with torch.no_grad():
        q_emb = model.encode([query], convert_to_numpy=True)
        faiss.normalize_L2(q_emb)
        D, I = index.search(q_emb, top_k)
    
    # Return only the text chunks to reduce memory usage
    results = [chunk_texts[idx] for idx in I[0]]
    return results

def build_prompt(query: str, retrieved: List[Tuple[Dict, float, str]], tokenizer, max_context_chars: int = 4000) -> str:
    system_prompt = """You are an expert in Alzheimer's markers (and biomarkers). Your task is to analyze a list of potential markers extracted from datasets and validate them using the provided scientific literature.

For each element in the list, you must:

1. **VALIDATION**: Determine if the element is actually a recognized marker/biomarker for Alzheimer's
2. **ACRONYMS**: Provide the most known/used acronym/abbreviation for that marker/biomarker

**Required response format:**
For each element, use this JSON structure:
```json
{
  "original_name": "name as provided in the list",
  "valid": true/false,
  "acronym": "ACR",
}
```
If an element is NOT a valid marker, indicate:
```json
{
  "original_name": "name as provided",
  "valid": false,
  "acronym": "N/A"
}
```
The marker must be documented in scientific literature as correlated with Alzheimer's.
Be precise, concise, and base your analysis exclusively on the evidence provided in the documents.
"""

    user_prompt = f"""Analyze the following list of potential Alzheimer's markers and validate them using the information contained in the provided documents.
CONTEXT FROM DOCUMENTS:
{retrieved}
List to validate:
{query}
Based STRICTLY on the information provided in the context above, for each item in the list determine:

- If it is a recognized marker for Alzheimer's according to the documents;
- The most known/used acronym/abbreviation mentioned in the documents.

Provide the answer in the JSON format specified in the system instructions. If a marker is not mentioned in the provided documents or does not have sufficient evidence in the context, mark it as invalid. Do not use any knowledge external to the provided documents.
"""

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]
    
    # apply_chat_template: Converte i messaggi nel formato richiesto dal modello
    # tokenize=False: Restituisce stringa invece di token IDs
    # add_generation_prompt=True: Aggiunge il prompt per iniziare la generazione
    full_prompt = tokenizer.apply_chat_template(
        messages, 
        tokenize=False, 
        add_generation_prompt=True
    )

    # se vuoi impostare il reasoning su low invece che medium
    '''
    if full_prompt.count(" medium") > 0:
        full_prompt = full_prompt.replace("medium", "low", 1)
        print("\n\nChanged reasoning level to low for better performance.\n")
    '''

    return full_prompt

def generate_from_llm(prompt: str, model, tokenizer, device):
    # Tokenizzazione dell'input completo
    # return_tensors="pt": Restituisce tensori PyTorch
    # padding=True: Aggiunge padding se necessario
    # truncation=True: Tronca se supera la lunghezza massima
    inputs = tokenizer(
        prompt, 
        return_tensors="pt", 
        padding=False, 
        truncation=True,
        max_length=8192  # Context window di gpt-oss-20b
    ).to(device)
    # Conteggio token effettivi
    token_count = inputs.input_ids.shape[1]
    print(f"Token count effettivi: {token_count}")
    print(f"--- Waiting for the response ---")
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
            use_cache=True,               ### Usala STA CACHE!!!!!
            repetition_penalty=1.1,       # Penalità per ripetizioni
            length_penalty=1.0            # Penalità per lunghezza
        )

    # Estrazione solo della parte generata (esclude l'input)
    input_length = inputs.input_ids.shape[1]
    generated_tokens = outputs[0][input_length:]
    
    # Decodifica dei token generati
    # skip_special_tokens=True: Rimuove token speciali (<eos>, <pad>, ecc.)
    # clean_up_tokenization_spaces=True: Pulisce spazi extra dalla tokenizzazione
    output_text = tokenizer.decode(
        generated_tokens, 
        skip_special_tokens=False,         # Mi servono per filtrare il ragionamento
        clean_up_tokenization_spaces=True
    )

    # Clean up memory
    del inputs, outputs, generated_tokens
    clear_gpu_memory()

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
        return cot, response
    else:
        print(f"No filtering tags found, returning full output.")
        return "", output_text.strip()

def process_biomarkers_batch(batch: List[str], index, chunks, metas, emb_model, llm_model, tokenizer, device):
    """Process a batch of biomarkers with memory optimization"""
    #print(f"Processing batch: {batch}")
    
    # Retrieve context for each biomarker
    all_retrieved_texts = []
    for biomarker in batch:
        query = f"Alzheimer abbreviation acronym markers biomarkers: {biomarker}"
        
        with torch.no_grad():
            retrieved_texts = retrieve(query, index, chunks, metas, emb_model, top_k=4)
            all_retrieved_texts.extend([f"Context for {biomarker}: {text}..." for text in retrieved_texts])
        
        clear_gpu_memory()
    
    # Combine all retrieved context
    retrieved_context = "\n".join(all_retrieved_texts)
    
    # Generate prompt and get response
    prompt = build_prompt(batch, retrieved_context, tokenizer, max_context_chars=2500)

    cot, response = generate_from_llm(prompt, llm_model, tokenizer, device)
    
    return cot, response

def validation(model, tokenizer, device, biomarkers):
    pdf_folder = "./docs"

    # crea index (esegui solo la prima volta)
    # index, chunks, metas = build_embeddings_and_index(pdf_folder)

    # carica index
    index, chunks, metas = load_index_and_metadata()

    clear_gpu_memory()
    emb_model = SentenceTransformer(EMBEDDING_MODEL, device=device)

    # retrieval + generation
    '''
    with open ("./results/biomarkers_list.txt", "r") as f:
        biomarkers = [line.strip() for line in f if line.strip()]
    '''

    evaluated_biomarkers = []
    log_file = {
        "acronyms": List,
        "cot": str,
        "response": str,
    }
    
    # Process in smaller batches for memory efficiency
    for i in range(0, len(biomarkers), MAX_BATCH_SIZE):
        batch = biomarkers[i:i+MAX_BATCH_SIZE]
        print(f"\nBatch {i//MAX_BATCH_SIZE + 1}/{(len(biomarkers)-1)//MAX_BATCH_SIZE + 1}")
        
        try:
            cot, response = process_biomarkers_batch(
                batch, index, chunks, metas, emb_model, model, tokenizer, device
            )
            evaluated_biomarkers.append(response)
            
        except torch.cuda.OutOfMemoryError as e:
            print(f"OOM Error in batch {i//MAX_BATCH_SIZE + 1}: {e}")
            print("Clearing memory and retrying with smaller batch...")
            clear_gpu_memory()
            
            # Retry with batch size 1
            for single_item in batch:
                try:
                    cot, response = process_biomarkers_batch(
                        [single_item], index, chunks, metas, emb_model, model, tokenizer, device
                    )
                    response_json = json.loads(response)
                    to_log = False
                    for acronym in response_json:
                        if acronym["valid"] == False:
                            to_log = True
                    if to_log:
                        log_file["acronyms"] = response_json
                        log_file["cot"]      = cot
                        log_file["response"] = response
                    with open("./results/not_validated_logs.jsonl", "a", encoding="utf-8") as f:
                        json.dump(log_file, f, ensure_ascii=False)  # scrive il dict come JSON
                        f.write("\n") 
                    # a sto punto append response o response_json?
                    evaluated_biomarkers.append(response)

                except Exception as e2:
                    print(f"Failed to process {single_item}: {e2}")
                    evaluated_biomarkers.append(f"ERROR: Could not process {single_item}")
        
        print(f"Processed {len(evaluated_biomarkers)*MAX_BATCH_SIZE}/{len(biomarkers)} biomarkers so far.")
        
        if i % (MAX_BATCH_SIZE * 50) == 0: # salva ogni 100 biomarkers
            with open("./results/evaluated_biomarkers.txt", "w") as f:
                for item in evaluated_biomarkers:
                    f.write(item + "\n")

        # Memory cleanup between batches
        clear_gpu_memory()

    with open("./results/evaluated_biomarkers.txt", "w") as f:
        for item in evaluated_biomarkers:
            f.write(item + "\n")
    return evaluated_biomarkers