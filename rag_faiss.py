import os
import glob
import pickle
from typing import List, Dict, Tuple

import fitz  # PyMuPDF
import numpy as np
import faiss
import torch

from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForCausalLM

# ---------- Config ----------
EMBEDDING_MODEL = "sentence-transformers/all-mpnet-base-v2"  # buona qualità
CHUNK_CHARS = 1500
CHUNK_OVERLAP = 300
FAISS_INDEX_PATH = "faiss_index.bin"
METADATA_PATH = "faiss_metadata.pkl"
LLM_MODEL_ID = "openai/gpt-oss-20b"  # sostituisci con il tuo path locale o repo HF
# ----------------------------

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
                               metadata_path: str = METADATA_PATH):
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

    embeddings = model.encode(all_chunks, show_progress_bar=True, convert_to_numpy=True)

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
             top_k: int = 5) -> List[Tuple[Dict, float, str]]:
    q_emb = model.encode([query], convert_to_numpy=True)
    faiss.normalize_L2(q_emb)
    D, I = index.search(q_emb, top_k)
    results = []
    for score, idx in zip(D[0], I[0]):
        results.append((metas[idx], float(score), chunk_texts[idx]))
    return results

def build_prompt(query: str, retrieved: List[Tuple[Dict, float, str]], max_context_chars: int = 4000) -> str:
    system_prompt = """You are an expert in Alzheimer's markers and biomarkers. Your task is to analyze a list of potential markers/biomarkers extracted from datasets and validate them using the provided scientific literature.

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
If an element is NOT a valid marker/biomarker, indicate:
```json{
  "original_name": "name as provided",
  "valid": false,
  "acronym": "N/A"
}
```
The marker/biomarker must be documented in scientific literature as correlated with Alzheimer's.
Be precise, concise, and base your analysis exclusively on the evidence provided in the documents.
"""

    user_prompt = f"""Analyze the following list of potential Alzheimer's markers/biomarkers and validate them using the information contained in the provided documents.
CONTEXT FROM DOCUMENTS:
{retrieved}
List to validate:
{query}
Based STRICTLY on the information provided in the context above, for each item in the list determine:

If it is a recognized marker/biomarker for Alzheimer's according to the documents
The most known/used acronym/abbreviation mentioned in the documents

Provide the answer in the JSON format specified in the system instructions. If a biomarker is not mentioned in the provided documents or does not have sufficient evidence in the context, mark it as invalid. Do not use any knowledge external to the provided documents.
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
    if full_prompt.count(" medium ") > 0:
        full_prompt = full_prompt.replace("medium", "low", 1)
        print("\n\nChanged reasoning level to low for better performance.\n")

    return full_prompt

def load_local_llm_and_tokenizer(model_id: str = LLM_MODEL_ID, device: str = "cpu"):
    """
    Carica tokenizer e modello HF. Usa device_map='auto' e torch_dtype=float16 se possibile.
    Richiede che tu abbia installato una build di PyTorch compatibile con ROCm se vuoi usare la MI210.
    """
    print("Loading tokenizer and model..")
    tokenizer = AutoTokenizer.from_pretrained(
        model_id, 
        trust_remote_code=True,
        use_fast=True
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_id, 
        torch_dtype=torch.bfloat16 if device=="cuda" else torch.float32,
        device_map="auto",
        trust_remote_code=True,
        low_cpu_mem_usage=True
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer, model

def generate_from_llm(prompt: str, model, device):
    # Tokenizzazione dell'input completo
    # return_tensors="pt": Restituisce tensori PyTorch
    # padding=True: Aggiunge padding se necessario
    # truncation=True: Tronca se supera la lunghezza massima
    inputs = tokenizer(
        prompt, 
        return_tensors="pt", 
        padding=True, 
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
            temperature=0.7,              # Controllo randomness (se do_sample=True) quindi inutile
            top_p=0.9,                    # Nucleus sampling (se do_sample=True) quindi inutile
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
        cot = ""
        response = output_text.strip()
        # ritorna l'output completo se non trova i tag
        print(f"No filtering tags found, returning full output.")
        return cot, response

    '''
    try:
        # Isola la sezione JSON anche se c'è testo extra
        start = response.find('{')
        end   = response.rfind('}') + 1   # rfind => ultima graffa

        if start == -1 or end == 0:
            return [], "", ""                        # nessuna struttura JSON trovata

        json_str = response[start:end]

        # Carica il JSON
        data = json.loads(json_str)

        # Estrai la lista biomarkers, se esiste
        biomarkers = data.get("biomarkers", [])
        # Garantisci che sia effettivamente una lista, altrimenti torna lista vuota
        if isinstance(biomarkers, list) and cot != "" and response != "":
            return biomarkers, cot, response 
        else:
            return [], "", ""

    except (json.JSONDecodeError, TypeError) as e:
        print(f"Errore nell'estrazione dei biomarkers: {e}")
        return [], "", ""
    '''


# ---- Esempio d'uso ----
if __name__ == "__main__":
    pdf_folder = "./docs"
    # 1) crea index (esegui solo la prima volta)
    # index, chunks, metas = build_embeddings_and_index(pdf_folder)

    # 2) carica index
    index, chunks, metas = load_index_and_metadata()

    # 3) prepara modello embeddings (su GPU se disponibile)
    emb_device = "cuda" if torch.cuda.is_available() else "cpu"
    emb_model = SentenceTransformer(EMBEDDING_MODEL, device=emb_device)

    # 4) load LLM HF (gpt-oss-20b)
    tokenizer, model = load_local_llm_and_tokenizer(device=emb_device)

    # 5) retrieval + generation
    with open ("liste_biomarkers.txt", "r") as f:
        biomarkers = [line.strip() for line in f if line.strip()]
    
    # cilco for che itera su 5 elementi alla volta
    for i in range(0, len(biomarkers), 5):
        batch = biomarkers[i:i+5]
        embedding_query = "Alzheimer abbreviation acronym markers and biomarkers: " + ", ".join(batch)
        retrieved = retrieve(embedding_query, index, chunks, metas, emb_model, top_k=5)
        prompt = build_prompt(batch, retrieved, max_context_chars=3000)
        print(f"Prompt:\n{prompt}")
        input()
        # MI MANCA FARE CHECK DEI TOKENS DEL RETRIEVED, QUINDI DEL PROMPT
        cot, response = generate_from_llm(prompt, model, device=emb_device)
        print("=== COT ===")
        print(cot)
        print("\n\n====================\n\n")
        print("=== RISPOSTA ===")
        print(response)
        print("\n\n====================\n\n")
        # DEVO IMMAGAZZINARE RISULTATI
