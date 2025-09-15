# Biomarker Extractor

La seguente è una pipeline semi-automatizzata sviluppata per estrarre, convalidare ed aggregare marcatori e biomarcatori da database di Clinical Trials che contengono informazioni in formato testo libero, pur essendo generalizzabile come data-extractor ad altri dataset non strutturati o semi-strutturati.

## Setup:

Se non hai un ambiente virtuale, crealo e attivalo:

```bash
python3 -m venv venv
source venv/bin/activate
```

## Troubleshooting: Errori di Cache e Variabili d'Ambiente

### Problema: Segmentation Fault (core dumped)
Se riscontri errori di tipo Segmentation Fault (core dumped) quando esegui il progetto, il problema è probabilmente legato alle variabili d'ambiente che puntano alla cache di HuggingFace e PyTorch nella directory /home invece che nella directory corretta del progetto.

### Soluzione
Il progetto è configurato per funzionare con la cache localizzata in /media/sdb1/ENV_VARS_GPU/.cache. Prima di eseguire il codice, assicurati di settare le seguenti variabili d'ambiente:

```bash
export XDG_CACHE_HOME=/media/sdb1/ENV_VARS_GPU/.cache
export TORCHINDUCTOR_CACHE_DIR=/media/sdb1/ENV_VARS_GPU/.cache/torchinductor
export TRITON_CACHE_DIR=/media/sdb1/ENV_VARS_GPU/.cache/triton
export HF_HOME=/media/sdb1/ENV_VARS_GPU/.cache/huggingface
export TORCH_HOME=/media/sdb1/ENV_VARS_GPU/.cache/torch
export AMD_COMGR_CACHE_DIR=/media/sdb1/ENV_VARS_GPU/.cache/comgr
export TMPDIR=/media/sdb1/ENV_VARS_GPU/tmp
```

#### Se utilizzi una GPU AMD con ROCm

```bash
venv/bin/python -m pip install -r requirements
venv/bin/python -m pip install torch --index-url https://download.pytorch.org/whl/rocm6.3
```

#### Se utilizzi una GPU NVIDIA, oppure vuoi eseguire su CPU (sconsigliato)

1. Scommenta la riga `torch` nel file `requirements.txt`.
2. Installa le dipendenze:

```bash
venv/bin/python -m pip install -r requirements
```

## Utilizzo

Eseguire l'intera pipeline con:

```bash
venv/bin/python main.py
```

Sono tuttavia presenti diversi argomenti che possono essere aggiunti per eseguire sotto-parti della pipeline, in particolare:
 - `extraction`: solamente la parte di estrazione dei biomarkers
 - `validation`: solamente la parte di assegnazione di acronimi con RAG (se seguito da un secondo argomento, esegue anche la parte successiva di aggregazione)
 - `aggregation`: solamente la parte di aggregazione dei biomarkers
 - `resume`: riprende dalla fase di aggregazione nella quale vi è la supervisione umana sul merging di gruppi identificati dall'LLM
 - `clean`: cancella tutti i file presenti all'interno della cartella `/results` per ogni dataset analizzato

Esempio: per eseguire solamente la parte di estrazione, pulendo i file presenti nella cartella `/results`:
```bash
venv/bin/python main.py extraction clean
```

## Descrizione del funzionamento

La pipeline è suddivisa in tre fasi principali: **estrazione**, **validazione** e **aggregazione** dei biomarkers.

### 1. Estrazione

- Il dataset viene processato una riga alla volta, selezionando prima la colonna di interesse (`outcome_measurement_description`) e scartando le righe che non presentano valori nelle colonne `param_value` e `param_value_decimal`, così come le righe corrispondenti a Clinical Trial non ancora terminati (tramite la colonna `end_date`). L'indice della riga del dataset viene associato al marcatore estratto per tutto il corso della pipeline, in modo da ottenere alla fine, per ogni marcatore, una lista di tutti i Clinical Trials (le righe del dataset) che lo coinvolgevano.

- Al modello si fornisce un prompt con semplice *prompt engineering* e **5 esempi** con la relativa analisi e l'estrazione ideale dei biomarkers: questi esempi fungono da guida comportamentale per l'LLM.

- Dal momento che non sempre l'estrazione può andare a buon fine, specialmente se la riga è molto lunga e/o il ragionamento dell'LLM diventa troppo complesso e articolato, le righe non processate con successo vengono salvate e viene applicato un secondo tentativo di estrazione dei marcatori, questa volta impostando il livello di reasoning del modello su `low` (precedentemente era `medium`).

- Per determinate righe del dataset, si evidenzia che il modello risponde con il solo ragionamento e non con la risposta finale: per queste casistiche, viene forzato il modello con un'ulteriore chiamata ad estrarre eventuali marcatori dall'analisi fatta in precedenza. In questo modo, la totalità delle righe del dataset viene processata.

- File generati durante l'estrazione:
  - `biomarkers_list.txt` — lista di tutti i biomarkers estratti, con acronimo e expanded form
  - `unprocessed_lines.txt` — righe saltate perché troppo grandi o complesse: se il file non è presente, significa che tutte le righe del dataset sono state processate
  - `extraction_logs.json` — logs dove vengono salvate la riga processata, gli eventuali markers estratti e la risposta del modello insieme alla relativa CoT (chain-of-thought / ragionamenti) per tracciare le motivazioni dell'estrazione (o della non-estrazione) dei biomarkers

- Nota sui limiti: l'LLM non è un esperto specialistico in biomarkers o nella patologia analizzata, e ogni riga del dataset viene processata in maniera indipendente. Questo può portare a inconsistenze nella nomenclatura e causare problemi nell'automatizzare la fase successiva di aggregazione. Per questo motivo è prevista la fase di validazione, il cui scopo principale è quello di fornire a ogni marcatore estratto un acronimo standard, basandosi sulla letteratura scientifica attuale.

### 2. Validazione (RAG — Retrieval Augmented Generation)

- Ogni biomarker estratto viene analizzato dall'LLM utilizzando una base di conoscenza costruita con la tecnica **RAG**: i documenti di supporto (principalmente pubblicazioni recenti, ma anche libri o altri file rilevanti) vengono allegati al modello tramite retrieval per contestualizzare l'analisi.

- L'analisi consiste nel cercare e assegnare a ogni marcatore estratto un acronimo 'standard', coerente con la letteratura scientifica attuale. Questa fase è particolarmente importante per garantire l'efficacia della fase successiva di aggregazione.

- File generati durante la validazione:
  - `acronyms_logs.json` — elenco dei biomarkers con nome originale, acronimo identificato, row_id e relativa CoT dell'LLM per tracciare i risultati dell'analisi

- Nota: l'estrazione del dataset, processando le righe in maniera indipendente, può produrre molte occorrenze duplicate; la fase successiva si occupa di aggregare e comprimere i risultati.

### 3. Aggregazione dei biomarkers

- L'aggregazione combina approcci LLM-based e codice deterministico:
  - Inizialmente viene applicato un exact matching degli acronimi, dopo aver pulito opportunamente questi ultimi da caratteri speciali, numeri e sigle particolari (`CSF`, `PET`, `MRI`)
  - A questo punto chiediamo all'LLM di trovare dei gruppi che potrebbero essere aggregati tra quelli già definiti dall'exact matching: quindi iteriamo sulle proposte dell'LLM e l'utente (esperto) può confermare (quindi fare il merge), rifiutare (skip), o selezionare un sottoinsieme dei gruppi proposti da unire. L'iterazione continua finché tutti i tentativi di merging sono stati rifiutati da parte dell'utente
  - Infine, viene chiesto all'LLM di identificare eventuali varianti tra le componenti di uno stesso gruppo

- Dopo il raggruppamento si effettua il conteggio delle occorrenze, delle varianti e della percentuale di queste ultime rispetto alle occorrenze dell'intero gruppo, e si ordina la lista in ordine decrescente.

- File generati:
  - `biomarkers.json` — risultato finale aggregato
  - `remaining_biomarkers.txt` — lista degli eventuali biomarkers non aggregati durante l'exact matching

- Tutti i file di output vengono salvati nella cartella `results`.

## Output / File generati:

I principali file prodotti dalla pipeline (tutti nella cartella `results`) sono:

  - `biomarkers_list.txt` — lista di tutti i biomarkers estratti, con acronimo e expanded form
  - `unprocessed_lines.txt` — righe saltate perché troppo grandi o complesse: se il file non è presente, significa che tutte le righe del dataset sono state processate
  - `extraction_logs.json` — logs dove vengono salvate la riga processata, gli eventuali markers estratti e la risposta del modello insieme alla relativa CoT (chain-of-thought / ragionamenti)
  - `acronyms_logs.json` — elenco dei biomarkers con nome originale, acronimo identificato, row_id e relativa CoT dell'LLM
  - `remaining_biomarkers.txt` — lista degli eventuali biomarkers non aggregati durante l'exact matching
  - `biomarkers.json` — risultato finale aggregato

## Cartelle **non** presenti nella repository:

Nella repo GitHub non sono incluse le seguenti cartelle:
- `data` — dataset analizzati
- `docs` — documenti usati nel processo RAG
- `venv` — virtual environment