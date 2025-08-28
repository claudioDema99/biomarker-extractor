# Biomarker Extractor

## Setup:

Se non hai un ambiente virtuale, crealo e attivalo (altrimenti attiva solo quello esistente):

```bash
python3 -m venv venv
source venv/bin/activate
```

#### Se utilizzi una GPU AMD con ROCm

```bash
venv/bin/python -m pip install -r requirements.txt
venv/bin/python -m pip install torch --index-url https://download.pytorch.org/whl/rocm5.7
```

#### Se utilizzi una GPU NVIDIA, oppure vuoi eseguire su CPU

1. Scommenta la riga `torch` nel file `requirements.txt`.
2. Installa le dipendenze:

```bash
venv/bin/python -m pip install -r requirements.txt
```

## Utilizzo:

Esegui la pipeline con:

```bash
venv/bin/python main.py
```

## Descrizione del funzionamento:

La pipeline è suddivisa in tre fasi principali: **estrazione**, **validazione** e **aggregazione** dei biomarkers.

#### 1. Estrazione

- Si processano un numero variabile di righe alla volta (dopo un preprocessing) utilizzando un **batch size adattivo**.  
  L’obiettivo è sfruttare al massimo la *context window* dell’LLM (quantità di testo misurata in token che il modello può processare contemporaneamente). Poiché la dimensione delle celle del dataset è molto variabile, il batch size adattivo conta i token di ogni riga e aggiunge righe al batch fino a saturazione prossima della context window.
- Al modello si fornisce un prompt con semplice *prompt engineering* e **4 esempi** (righe o parti di righe) con la relativa analisi e l’estrazione ideale dei biomarkers: questi esempi fungono da guida comportamentale per l’LLM.
- File generati durante l’estrazione:
  - `biomarkers_list.txt` — lista di tutti i biomarkers estratti, con acronimo e expanded form.
  - `unprocessed_lines.txt` — righe (con rispettivi indici) saltate perché troppo grandi (alcune celle superano i 6000 token).
  - `extraction_logs.json` — identificatori dei batch e delle righe processate, e le CoT (chain-of-thought / ragionamenti) del modello per tracciare le motivazioni dell’estrazione (o della non-estrazione) dei biomarkers.
- Nota sui limiti: l’LLM non è un esperto specialistico in biomarkers o nella patologia analizzata; questo può portare a identificazioni errate (es. considerare “MRI” come biomarker anziché come tecnica). Per questo motivo è prevista la fase di validazione.

#### 2. Validazione (RAG — Retrieval Augmented Generation)

- Ogni biomarker estratto viene validato dall’LLM utilizzando una base di conoscenza costruita con la tecnica **RAG**: i documenti di supporto (principalmente pubblicazioni recenti, ma anche libri o altri file rilevanti) vengono allegati al modello tramite retrieval per contestualizzare la validazione.
- Ad ogni biomarker viene assegnata una *label*:
  - `True` — considerato un effettivo biomarker dalla letteratura allegata.
  - `False` — non considerato un biomarker.
- File generati durante la validazione:
  - `evaluated_biomarkers.json` — elenco dei biomarkers con le rispettive label.
  - `not_validated_logs.json` — solo i biomarkers scartati (label = False) con le CoT del modello, per tracciare le motivazioni dello scarto.
- Nota: l’estrazione su porzioni ridotte del dataset può produrre molte occorrenze duplicate; la fase successiva si occupa di aggregare e comprimere i risultati.

#### 3. Aggregazione dei biomarkers

- L’aggregazione combina approcci LLM-based e codice deterministico:
  - L’LLM è utile per riconoscere sinonimi, varianti ortografiche, abbreviazioni e strutture diverse (es.: `P-Tau 181`, `pTau181`, `phosphorylated tau at threonine 181`).
  - Un algoritmo aggrega i gruppi creati su diversi batch minimizzando errori e duplicazioni.
- Dopo il raggruppamento si effettua il conteggio delle occorrenze e si ordina la lista in ordine decrescente per frequenza.
- File generati:
  - `biomarkers.json` — risultato finale aggregato.
- Tutti i file di output vengono salvati nella cartella `results`.

## Output / File generati:

I principali file prodotti dalla pipeline (tutti nella cartella `results`) sono:
- `biomarkers_list.txt` — elenco raw dei biomarkers estratti (acronimo + expanded form).
- `unprocessed_lines.txt` — righe saltate per eccessiva lunghezza, con indici.
- `extraction_logs.json` — log dell'estrazione: id batch, righe processate e CoT (ragionamenti).
- `evaluated_biomarkers.json` — biomarkers con le label di validazione (`True`/`False`).
- `not_validated_logs.json` — biomarkers scartati e relative CoT.
- `biomarkers.json` — risultato finale aggregato con conteggi e ordinamento.

## Cartelle **non** presenti nella repository:

Nella repo GitHub non sono incluse le seguenti cartelle:
- `data` — dataset analizzati
- `docs` — documenti usati nel processo RAG
- `shots` — esempi da fornire al modello per l’estrazione dei biomarkers
- `venv` — virtual environment
