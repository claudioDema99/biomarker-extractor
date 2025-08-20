# Biomarker Extractor

(se non hai venv:
crea e attiva venv (altrimenti attiva soltanto)
    python3 -m venv venv
    source venv/bin/activate)

setup:
    se usi GPU amd-smi (se vogliamo usare la gpu ci serve la versione torch di amd, quella 'base' è per nvidia):
        venv/bin/python -m pip install -r requirements
        venv/bin/python -m pip install torch --index-url https://download.pytorch.org/whl/rocm5.7

    se usi GPU nvidia:
        scommenta "torch" in requirements
        venv/bin/python -m pip install -r requirements

estrazione biomarkers:
    venv/bin/python main.py