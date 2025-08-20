# Biomarker Extractor

(se non hai venv: 
    python3 -m venv venv
    source venv/bin/activate)

setup:

    se usi GPU amd-smi:

        venv/bin/python -m pip install -r requirements
        venv/bin/python -m pip install torch --index-url https://download.pytorch.org/whl/rocm5.7

    se usi GPU nvidia:

        scommenta "torch" in requirements
        venv/bin/python -m pip install -r requirements

estrazione:

    venv/bin/python main.py