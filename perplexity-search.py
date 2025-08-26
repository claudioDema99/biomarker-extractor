import requests
import json
import os
from dotenv import load_dotenv
load_dotenv()

# Carica la tua API key in modo sicuro
api_key = os.environ.get("PERPLEXITY_API_KEY")

with open('lista_biomarkers.txt', 'r') as f:
    all_biomarkers = [line.strip() for line in f if line.strip()]
biomarkers = all_biomarkers[:100]  # Prendi i primi 5 per il test

if not api_key:
    print("Errore: PERPLEXITY_API_KEY non impostata come variabile d'ambiente.")
else:
    url = "https://api.perplexity.ai/chat/completions"
    
    payload = {
        "model": "sonar",
        "messages": [
            {
                "role": "system",
                "content": """You are a clinical biologist: given the possible markers for the Alzheimer disease, 
                tell me if they are markers or not, and if yes, give me the acronym (if it exists)."""
            },
            {
                "role": "user",
                "content": ""
            }
        ],
        "max_tokens": 800,
        "temperature": 0.5
    }
    
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }

    for biomarker in biomarkers: 
        try:
            payload["messages"][1]["content"] = f"For each of the element of this list, tell me if they are markers for the Alzheimer disease, and if yes, give me the acronym (if it exists). {biomarkers[:10]}"
            print(f"\nRichiesta: {payload['messages'][1]['content']}")
            response = requests.post(url, headers=headers, json=payload)
            response.raise_for_status()
            
            response_data = response.json()
            #print(json.dumps(response_data, indent=2))
            
            # Estrai la risposta dell'assistente
            if response_data.get("choices"):
                assistant_message = response_data["choices"][0]["message"]["content"]
                print("\nRisposta dell'Assistente:\n", assistant_message)
            input()
                
        except requests.exceptions.RequestException as e:
            print(f"Si è verificato un errore: {e}")
            if e.response is not None:
                print("Dettagli dell'errore:", e.response.text)
