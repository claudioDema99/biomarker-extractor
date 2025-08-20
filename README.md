# Biomarker Extractor

## Configurazione Ambiente Virtuale

Se non hai un ambiente virtuale, crealo e attivalo (altrimenti attiva solo quello esistente):

```bash
python3 -m venv venv
source venv/bin/activate
```

## Setup

### GPU AMD
Se utilizzi una GPU AMD con ROCm:

```bash
venv/bin/python -m pip install -r requirements.txt
venv/bin/python -m pip install torch --index-url https://download.pytorch.org/whl/rocm5.7
```

### GPU NVIDIA
Se utilizzi una GPU NVIDIA:

1. Scommenta la riga "torch" nel file `requirements.txt`
2. Installa le dipendenze:
   ```bash
   venv/bin/python -m pip install -r requirements.txt
   ```

## Utilizzo

Per estrarre i biomarker:

```bash
venv/bin/python main.py
```