import json
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

if torch.cuda.is_available():
    DEVICE = "cuda"
    print(f"GPU name: {torch.cuda.get_device_name(0)}")
    print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    print(f"GPU Memory Available: {torch.cuda.memory_reserved(0) / 1e9:.1f} GB")
else:
    print("Usando CPU")
    DEVICE = "cpu"

# === CARICAMENTO MODELLO E TOKENIZER ===
MODEL_NAME = "openai/gpt-oss-20b"
print(f"Caricamento modello {MODEL_NAME}..")

# Caricamento tokenizer
# trust_remote_code=True: Necessario per modelli che usano codice custom
# use_fast=True: Usa il tokenizer veloce (implementazione Rust) se disponibile
tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME, 
    trust_remote_code=True,
    use_fast=True
)

# Caricamento modello
# torch_dtype: Tipo di dato per i pesi del modello (float16 per GPU, float32 per CPU)
# device_map="auto": Distribuisce automaticamente il modello tra GPU/CPU disponibili
# trust_remote_code=True: Permette l'esecuzione di codice personalizzato dal modello
# low_cpu_mem_usage=True: Riduce l'uso della memoria CPU durante il caricamento
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME, 
    torch_dtype=torch.bfloat16 if DEVICE=="cuda" else torch.float32,
    device_map="auto",
    trust_remote_code=True,
    low_cpu_mem_usage=True
)

# Set pad token per gpt-oss
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

def get_token_count(records_json):
    return len(tokenizer.encode(json.dumps(records_json)))

def call_model(records, dataset_type):
    # QUI MI SCELGO I PROMPT IN BASE AL DATASET_TYPE
    if dataset_type == "Alzheimer":
        print("Using Alzheimer dataset prompts.")
    system_prompt = """You are an expert clinical data analyst specialized in identifying biomarkers in Alzheimer's disease clinical trials. Your task: analyze the provided records and extract ONLY biomarkers explicitly present in the text. Do NOT invent biomarkers not present in the records.

BIOMARKER DEFINITION: A biomarker is a quantifiable characteristic of the body that serves as an objective indicator of biological processes or pathological conditions in Alzheimer's disease.

MAIN ALZHEIMER'S BIOMARKERS (examples): Beta-amyloid (CSF, PET), Tau/p-Tau (CSF), Brain volumes (MRI), Cognitive tests (MMSE, ADAS-Cog), Genetic markers (APOE).

MANDATORY OUTPUT RULES (must be followed exactly):
1. Output **exactly one** JSON object and nothing else (no surrounding text, no code fences). The JSON must have two keys:
   {"analysis": "<4-5 concise sentences>", "biomarkers": [list of biomarker with required syntax]}
2. "analysis" must be a single string of 4–5 sentences that reference the evidence in the records.
3. "biomarkers" must be a JSON array. Each element in the array MUST follow this exact syntax:
   "ACRONYM/NAME: expanded form of the acronym (or a brief description of the biomarker if not acronym)"
   Examples:
   - "MMSE: Mini-Mental State Examination"
   - "APOE4: Apolipoprotein E epsilon 4 allele"
   - "Beta-amyloid (CSF): Cerebrospinal fluid levels of beta-amyloid"
4. Collapse duplicates (each biomarker appears once).
5. If you cannot follow these rules, output exactly:
   {"analysis":"", "biomarkers":[]}
Always reason with clinical rigor and refer only to evidence in the records.
"""

    user_prompt = f"""You are an expert clinical data analyst specialized in identifying biomarkers in Alzheimer's disease clinical trials. Your task: analyze the provided records and extract ONLY biomarkers explicitly present in the text. Do NOT invent biomarkers not present in the records.
Follow the structure and formatting style shown in the examples exactly, and apply it only to the new input records.

Examples:
1. Input records: 
   {{'outcome_measurement_title': '799865704 - OG000: Number of Participants With Incident Dementia;\n799865705 - OG001: Number of Participants With Incident Dementia;\n799865726 - OG000: Progression of Cognitive Decline in Standardized Z-score Scale. Higher Z-scores Indicate Worse Performance.;','outcome_measurement_description': '799865704 - OG000: All cause dementia based on DSM-IV criteria as determined by an expert panel of clinicians using an adjudication process. A full neuropsychological battery was administered annually, or at 6 month visit if there was a diagnosis of dementia or initiation of medication for dementia by private physician, or change in Modified Mini Mental State Exam (3MSE), Clinical Dementia Rating (CDR), or Alzheimer Disease Assessment Scale (ADAS-Cog). Decline on tests scores based on an algorithm resulted in a neurological exam and brain imaging. These data were used in the adjudication process.;'}}
   Output:
   {{"analysis":"Although "Number of Participants With Incident Dementia" is the outcome, the 3MSE, CDR, and ADAS-Cog are explicitly named as the tools whose longitudinal decline (acting as cognitive biomarkers) is the primary indicator driving the diagnostic evaluation for that outcome. They are the measurable signals of cognitive change within this protocol.","biomarkers":["3MSE: Modified Mini Mental State Exam", "CDR: Clinical Dementia Rating", "ADAS-Cog: Alzheimer Disease Assessment Scale"]}}

2. Input records: 
   {{'outcome_measurement_title': "798273582 - OG000: Participant's Clinical Condition or Endpoint Assessed With the ADCS-Clinical Global Impression of Change (ADCS-CGIC);\n798273583 - OG001: Participant's Clinical Condition or Endpoint Assessed With the ADCS-Clinical Global Impression of Change (ADCS-CGIC);\n798273576 - OG000: Functional Performance Assessed by the Alzheimer's Disease Cooperative Study Activities of Daily Living (ADCS-ADL) Inventory;\n798273572 - OG000: Presence of Agitation and/or Psychosis Measured by the Neuropsychiatric Inventory (NPI) Combined With an Assessment of the Clinical Significance of Behavioral Change Rated by the Study Clinician;\n798273580 - OG000: Agitation Measured by the Cohen-Mansfield Agitation Inventory (CMAI), Community Version;", 'outcome_measurement_description': "798273582 - OG000: ADCS-Clinical Global Impression of Change (ADCS-CGIC) provides a means to reliably assess global change from baseline. It provides a semi-structured format to allow clinicians to gather necessary clinical information from both the participant and informant, in order to make an overall impression of clinical change. The range of this instrument is 1 to 7 with lower numbers indicating improvement and higher numbers indicating a worsened state.;\n798273576 - OG000: Alzheimer's Disease Cooperative Study Activities of Daily Living Score (ADCS-ADL) is a structured questionnaire about activities of daily living, administered to the subject's caregiver/study partner. The range of this instrument is 0 to 78 with lower numbers indicating greater impairment.;\n798273572 - OG000: NPI quantifies behavioral changes in dementia, including depression, anxiety, psychosis, agitation, and others. This is a questionnaire administered to the subject's study partner. The range of this instrument is 0 to 120 with higher numbers indicating greater impairment. To determine whether or not psychosis or agitation is present, there is no cutoff score but is based on the clinician's judgment. In the NPI, the subject responds to 'Yes' or 'No' questions. Then it is determined how often psychosis or agitation occurs and if it is mild, moderate or severe.;\n798273580 - OG000: The Cohen-Mansfield Agitation Inventory (CMAI) is a 29-item caregiver rating questionnaire for the assessment of agitation in older persons. It includes descriptions of 29 agitated behaviors, each rated on a 7-point scale of frequency. The range of this instrument is 29 to 203 with higher numbers indicating greater impairment.;"}}  
   Output: 
   {{"analysis":"The record explicitly defines four clinical assessment tools (ADCS-CGIC, ADCS-ADL, NPI, CMAI) whose standardized scores serve as quantifiable biomarkers for global clinical change, functional ability, neuropsychiatric symptoms, and agitation severity in Alzheimer's disease.", "biomarkers":["ADCS-CGIC: Alzheimer's Disease Cooperative Study Clinical Global Impression of Change","ADCS-ADL: Alzheimer's Disease Cooperative Study Activities of Daily Living Score","NPI: Neuropsychiatric Inventory","CMAI: Cohen-Mansfield Agitation Inventory"]}}

3. Input records: 
   {{'outcome_measurement_title': '798051339 - OG002: Change in Scan Interpretation Reliability After Application of Quantitation Software;\n798051331 - OG000: Change in Scan Interpretation Reliability After Application of Quantitation Software;\n798051332 - OG001: Change in Scan Interpretation Reliability After Application of Quantitation Software;\n798051330 - OG000: Change in Reader Accuracy After Application of Quantitation Software','outcome_measurement_description': 'Evaluate whether the use of quantitation software improves florbetapir (18F) scan interpretation by using the net reclassification index (NRI). The NRI is a prospective measure that quantifies the correctness of upward and downward reclassification or movement of predicted probabilities as a result of adding a new marker. NRI Values \\>0 indicate an improvement in scan interpretation accuracy and values \\<0 indicate a decline in scan interpretation accuracy after application of quantitation software.\n\nNRI = \\[P(up,event)-P(down,event)\\]-\\[P(up,nonevent)-P(down,nonevent)\\] Where P(up,event) = # events up/# events P(down,event) = # events down/# events P(up,nonevent) = # nonevents up/# nonevents P(down,nonevent) = # nonevents down/# nonevents and events: true positive case nonevents: true negative case up: scan change from negative to positive down: scan change from positive to negative\nOnly the 46 scans with autopsy from A16 are used for this outcome measure.'}} 
   Output: 
   {{"analysis":"The outcome explicitly references florbetapir (18F) PET scans, a validated imaging biomarker targeting amyloid-beta plaques. This is the core biological signal being measured.
The Net Reclassification Index (NRI) quantifies improvement in diagnostic accuracy when quantitation software is applied to the scans. Autopsy data from 46 subjects provides neuropathological confirmation. This anchors NRI not as an abstract statistic, but as a performance biomarker certifying real-world diagnostic utility. FDA biomarker qualification guidelines recognize tools that enhance interpretation reliability (e.g., software outputs) as companion biomarkers. NRI fits this role by qualifying the clinical validity of the quantitated Florbetapir-PET read.", "biomarkers":["Florbetapir (18F): tracer used in scans to assess interpretation reliability", "NRI: Net Reclassification Index"]}}

4. Input records: 
   {{'outcome_measurement_title': "797471211 - OG000: The Neuropsychiatric Inventory Questionnaire (NPI-Q);\n797471205 - OG000: The Clinical Dementia Rating Scale - Sum of Boxes (CDR-SOB);\n797471208 - OG001: The Alzheimer's Disease Cooperative Study - Activities of Daily Living (ADCS-ADL23);\n797471209 - OG000: The Alzheimer's Disease Cooperative Study - Clinical Global Impression of Change (ADCS-CGIC);\n797471204 - OG001: Alzheimer's Disease Assessment Scale-Cognitive Subscale (ADASCog/11);\n797471201 - OG000: The Mini-Mental State Examination (MMSE);", 'outcome_measurement_description': "797471211 - OG000: Change on the NPI-Q. The NPI-Q comprises 12 domains: delusions, hallucinations, dysphoria, apathy, euphoria, disinhibition, aggressivity and restlessness, irritability, anxiety aberrant motor behavior, appetite and eating disorders, and nocturnal behavior. The severity of the reported symptoms is assessed on a 3-point scale. The total severity score can range from 0 to 36 with higher scores representing worse severity.;\n797471205 - OG000: Changes in the CDR-SOB. The CDR characterizes functioning in 6 domains: memory, orientation, judgment and problem solving, community affairs, home and hobbies and personal care. The score is obtained by summing each of the domain box scores. Scores range from 0 to 18 with higher scores reflecting worse cognition.;\n797471208 - OG001: Changes in the ADCS-ADL23. The ADCS-ADL23 assesses basic and instrumental activities of daily living covering physical and mental functioning and independence in self-care. The score ranges from 0 to 78 with higher scores indicating less functional impairment.;\n797471209 - OG000: The ADCS-CGIC focuses on clinicians' observations of change in the subject's cognitive, functional, and behavioral performance since the beginning of a trial. The ADCS-CGIC is a 7-point scale with lower values (\\<4) representing an improvement, higher values (\\>4) representing a worsening, and a value of 4 indicating no change.;\n797471204 - OG001: Changes in scores on the 11-item ADASCog/11. The ADAS-Cog/11 includes 11 items assessing cognitive function. The domains include memory, language, praxis, and orientation. There are 70 possible points. Higher scores reflect greater cognitive impairment.;\n797471201 - OG000: Changes in scores on the MMSE. The MMSE consists of 5 components: orientation to time and place, registration of 3 words, attention and calculation, recall of 3 words, and language. The scores from the 5 components are summed to obtain the overall MMSE total score. The MMSE total score can range from 0 to 30, with higher scores indicating better cognition.;"}}  
   Output: 
   {{"analysis":"The outcome titles and descriptions directly name established, validated clinical assessment tools: NPI-Q, CDR-SOB, ADCS-ADL23, ADCS-CGIC, ADAS-Cog/11, and MMSE. These are the core measures being used to track disease progression. Each instrument provides a standardized quantitative measure of a key clinical domain in Alzheimer's disease. Each tool provides a defined scoring range (e.g., NPI-Q: 0-36, CDR-SOB: 0-18, MMSE: 0-30) with explicit interpretation (higher/lower scores = worse/better function). These scores are the actionable numeric biomarker values used in trial analysis. While distinct from molecular biomarkers (e.g., amyloid PET), these instruments are foundational clinical biomarkers in AD trials. They provide the quantitative, domain-specific measures mandated by regulators to track disease-modifying effects and are universally accepted as primary/secondary endpoints reflecting clinically relevant progression.", "biomarkers":["NPI-Q: Neuropsychiatric Inventory Questionnaire", "CDR-SOB: Clinical Dementia Rating Scale Sum of Boxes", "ADCS-ADL23: Alzheimer's Disease Cooperative Study Activities of Daily Living", "ADCS-CGIC: The Alzheimer's Disease Cooperative Study Clinical Global Impression of Change", "ADASCog/11: Alzheimer's Disease Assessment Scale-Cognitive Subscale", "MMSE: Mini-Mental State Examination"]}}
   
Task: Identify all biomarkers explicitly present in the Records and produce output **only** the exact JSON object required in the prompt.

Input records:
{json.dumps(records)}

Output:
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
        print("Changed reasoning level to low for better performance.")

    # Tokenizzazione dell'input completo
    # return_tensors="pt": Restituisce tensori PyTorch
    # padding=True: Aggiunge padding se necessario
    # truncation=True: Tronca se supera la lunghezza massima
    inputs = tokenizer(
        full_prompt, 
        return_tensors="pt", 
        padding=True, 
        truncation=True,
        max_length=8192  # Context window di gpt-oss-20b
    ).to(DEVICE)
    # Conteggio token effettivi
    token_count = inputs.input_ids.shape[1]
    print(f"Token count effettivi: {token_count}")
    print(f"Sending request to the {MODEL_NAME} model..")
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

    # Filtra il reasoning usando i tag speciali altrimenti mi prendo anche tutto il reasoning
    final_start = '<|end|><|start|>assistant<|channel|>final<|message|>'
    
    if final_start in output_text:
        # Estrai solo la parte finale (dopo final_start)
        output_text = output_text.split(final_start)[-1].strip()
        # Rimuovi i tag di chiusura e lo special token <|return|>
        output_text = (
            output_text
            .replace('<|end|>', '')
            .replace('<|return|>', '')
            .strip()
        )
        print(f"Filtered output..")#: \n\n{output_text}")
    else:
        # ritorna l'output completo se non trova i tag
        print(f"No filtering tags found, returning full output: {output_text}")

    #print(f"Output from {MODEL_NAME}: {output_text}")

    try:
        # Isola la sezione JSON anche se c'è testo extra
        start = output_text.find('{')
        end   = output_text.rfind('}') + 1   # rfind => ultima graffa

        if start == -1 or end == 0:
            return []                        # nessuna struttura JSON trovata

        json_str = output_text[start:end]

        # Carica il JSON
        data = json.loads(json_str)

        # Estrai la lista biomarkers, se esiste
        biomarkers = data.get("biomarkers", [])
        # Garantisci che sia effettivamente una lista, altrimenti torna lista vuota
        return biomarkers if isinstance(biomarkers, list) else []

    except (json.JSONDecodeError, TypeError) as e:
        print(f"Errore nell'estrazione dei biomarkers: {e}")
        return []
