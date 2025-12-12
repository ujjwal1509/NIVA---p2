# ============================================================
# langchain_pipeline.py — ORIGINAL logic, GROQ replacing OLLAMA
# ============================================================

import os
import re
import json
import datetime
import logging
import requests
from dotenv import load_dotenv

# Optional imports
try:
    import pandas as pd
    import numpy as np
except:
    pd = None
    np = None

load_dotenv()
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# ------------------------------------------------------------
# GROQ CONFIG (used in place of Ollama)
# ------------------------------------------------------------
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_URL = os.getenv("GROQ_URL", "https://api.groq.com/openai/v1/chat/completions")
# Recommended model
GROQ_MODEL = os.getenv("GROQ_MODEL", "groq/llama3-70b")
GROQ_TIMEOUT = int(os.getenv("GROQ_TIMEOUT", "60"))


def call_ollama(prompt, max_tokens=150, temperature=0.35, timeout=GROQ_TIMEOUT):
    """
    Backwards-compatible function name (call_ollama) but performs a Groq OpenAI-compatible request.
    Returns a text string (LLM output) or empty string on failure.
    """
    if not GROQ_API_KEY:
        logger.warning("GROQ_API_KEY not set — skipping LLM call.")
        return ""

    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json",
    }

    payload = {
        "model": GROQ_MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": temperature,
        "n": 1,
        "stream": False,
    }

    try:
        resp = requests.post(GROQ_URL, headers=headers, json=payload, timeout=timeout)
        resp.raise_for_status()
        data = resp.json()

        # Groq/OpenAI style response parsing
        if isinstance(data, dict):
            choices = data.get("choices") or []
            if choices:
                # standard OpenAI: choices[0]["message"]["content"]
                msg = choices[0].get("message", {}).get("content") or choices[0].get(
                    "text"
                )
                if msg:
                    return str(msg).strip()
        return str(data).strip()
    except Exception as e:
        logger.warning("Groq call failed: %s", e)
        return ""


# ------------------------------------------------------------
# DATASET LOADING (original)
# ------------------------------------------------------------
CSV_PATH = "data/niva_dataset1.csv"

SYMPTOM_DF = None
SYMPTOM_COLS = []
DATASET_LOADED = False

if pd is not None and os.path.exists(CSV_PATH):
    try:
        df = pd.read_csv(CSV_PATH)
        if "prognosis" in df.columns:
            SYMPTOM_COLS = [c for c in df.columns if c.lower() != "prognosis"]
        else:
            SYMPTOM_COLS = list(df.columns[:-1])
        SYMPTOM_DF = df
        DATASET_LOADED = True
    except:
        DATASET_LOADED = False


def _normalize_sym(s):
    return re.sub(r"\s+", " ", s.replace("_", " ").lower()).strip()


NORMALIZED_MAP = {}
if DATASET_LOADED:
    NORMALIZED_MAP = {col: _normalize_sym(col) for col in SYMPTOM_COLS}


def extract_symptom_keywords(text):
    if not DATASET_LOADED or not text:
        return set()
    t = text.lower()
    return {col for col, phrase in NORMALIZED_MAP.items() if phrase in t}


def suggest_next_symptom(known):
    if not DATASET_LOADED:
        return None

    df = SYMPTOM_DF
    mask = None

    for s in known:
        if s in df.columns:
            cond = df[s] == 1
            mask = cond if mask is None else (mask & cond)

    if mask is not None:
        sub = df[mask]
        if not sub.empty:
            df = sub

    freqs = df[SYMPTOM_COLS].mean(numeric_only=True)

    for s in known:
        if s in freqs:
            freqs[s] = 0

    next_col = freqs.idxmax()
    return next_col if freqs[next_col] > 0 else None


# ------------------------------------------------------------
# HYBRID QUESTION GENERATOR (original)
# ------------------------------------------------------------
def generate_conversational_reply(
    messages, user_message, k_context=3, temperature=0.35
):
    required = [
        "symptoms",
        "duration",
        "severity",
        "additional symptoms",
        "medical history",
    ]

    patient_msgs = [t for r, t in messages if r == "patient"]
    count = len(patient_msgs)

    if count >= 5:
        return "Thank you. I have collected all the required information."

    next_field = required[count]

    known = set()
    for r, txt in messages:
        if r == "patient":
            known |= extract_symptom_keywords(txt)
    known |= extract_symptom_keywords(user_message)

    dataset_hint = None
    sug = suggest_next_symptom(known)
    if sug:
        dataset_hint = _normalize_sym(sug).capitalize()

    convo = "\n".join(
        ("Patient: " + t) if r == "patient" else ("NIVA: " + t) for r, t in messages
    )

    dataset_text = f"Dataset related symptom: {dataset_hint}\n" if dataset_hint else ""

    prompt = f"""
You are NIVA, a medical intake assistant.
Ask EXACTLY ONE medically relevant question to collect this missing field:

Field: {next_field}

Rules:
- Be natural and clear.
- No diagnosis.
- Use patient's last message.
- ONE short question only.

Conversation so far:
{convo}

Patient last said: "{user_message}"

{dataset_text}

Now ask ONE question.
"""

    reply = call_ollama(prompt, max_tokens=120, temperature=temperature).strip()

    if not reply:
        fallback = {
            "symptoms": "What symptom are you experiencing?",
            "duration": "How long have you had this issue?",
            "severity": "How severe is it — mild, moderate, or severe?",
            "additional symptoms": "Do you have any additional symptoms?",
            "medical history": "Do you have any medical history?",
        }
        return fallback[next_field]

    if not reply.endswith("?"):
        reply = reply.rstrip(".") + "?"

    return reply


# ------------------------------------------------------------
# STRUCTURED EXTRACTION
# ------------------------------------------------------------
def extract_structured_from_conversation(conv_text):
    answers = [
        line.split(":", 1)[1].strip()
        for line in conv_text.split("\n")
        if line.startswith("Patient:")
    ]

    while len(answers) < 5:
        answers.append("")

    structured = {
        "symptoms": answers[0],
        "duration": answers[1],
        "severity": answers[2],
        "additional_symptoms": answers[3],
        "medical_history": answers[4],
    }

    sev = structured["severity"].lower()
    structured["provisional_urgency"] = (
        "high" if "severe" in sev else ("medium" if "moderate" in sev else "low")
    )

    return structured


# ------------------------------------------------------------
# DURATION PARSER
# ------------------------------------------------------------
def parse_duration(text):
    if not text:
        return None, None
    m = re.search(r"(\d+)\s*(hour|hours|h|day|days|d|week|weeks|w)", text.lower())
    if m:
        v = float(m.group(1))
        u = m.group(2)[0]
        if u == "h":
            return v, "hours"
        if u == "d":
            return v, "days"
        return v, "weeks"
    return None, None


# ------------------------------------------------------------
# TRIAGE PROMPT (FIXED - BRACES ESCAPED)
# ------------------------------------------------------------
TRIAGE_PROMPT = """
You are a medical triage classifier.
Return STRICT JSON ONLY.

Input:
{data}

Format:
{{
 "severity": "low|moderate|high|critical",
 "urgency": "non-urgent|soon|urgent|immediate",
 "probable_cause": "string",
 "recommended_department": "string",
 "triage_score": 1,
 "notes": "string"
}}
"""


def safe_json_load(text):
    try:
        m = re.search(r"(\{.*\})", text, flags=re.DOTALL)
        if not m:
            return None
        return json.loads(m.group(1))
    except:
        return None


def classify_triage(structured):
    dval, dunit = parse_duration(structured["duration"])
    input_obj = {
        "symptoms": structured["symptoms"],
        "duration": {"value": dval, "unit": dunit},
        "severity_note": structured["severity"],
        "additional_symptoms": structured["additional_symptoms"],
        "medical_history": structured["medical_history"],
    }

    prompt = TRIAGE_PROMPT.format(data=json.dumps(input_obj))
    raw = call_ollama(prompt, max_tokens=200, temperature=0.2)
    parsed = safe_json_load(raw)

    if parsed:
        try:
            score = float(parsed.get("triage_score", 1))
        except:
            score = 1
        parsed["priority"] = (
            "HIGH" if score >= 8 else ("MEDIUM" if score >= 5 else "LOW")
        )
        return parsed

    # fallback
    sev = structured["severity"].lower()
    score = 2
    if "severe" in sev:
        score = 7
    elif "moderate" in sev:
        score = 5

    urgency = "urgent" if score >= 5 else "non-urgent"

    return {
        "severity": structured["severity"],
        "urgency": urgency,
        "probable_cause": "Unknown (fallback)",
        "recommended_department": "Emergency" if score >= 7 else "General Medicine",
        "triage_score": score,
        "notes": "Fallback used",
        "priority": "HIGH" if score >= 8 else ("MEDIUM" if score >= 5 else "LOW"),
    }


def triage_report(structured):
    t = classify_triage(structured)
    dept = t["recommended_department"].lower()

    specialist = "General Physician"
    if "emergency" in dept:
        specialist = "Emergency Care"
    elif "neuro" in dept:
        specialist = "Neurology"
    elif "cardio" in dept:
        specialist = "Cardiology"
    elif "ent" in dept:
        specialist = "ENT"
    elif "derm" in dept:
        specialist = "Dermatology"
    elif "ortho" in dept:
        specialist = "Orthopedics"

    t["specialist"] = specialist
    return t


# ------------------------------------------------------------
# DOCTOR ASSIGNMENT
# ------------------------------------------------------------
DOCTORS_PATH = "data/doctors.csv"
DOCTORS = (
    pd.read_csv(DOCTORS_PATH)
    if (pd is not None and os.path.exists(DOCTORS_PATH))
    else None
)


def safe_val(v):
    if np is not None:
        if isinstance(v, np.integer):
            return int(v)
        if isinstance(v, np.floating):
            return float(v)
    return v if v is not None else ""


def assign_doctor(triage):
    if DOCTORS is None:
        return {
            "doctor_name": "Unavailable",
            "specialty": triage.get("specialist", ""),
            "city": "",
            "hospital": "",
            "experience_years": "",
            "contact_url": "",
        }

    spec = triage["specialist"].lower()
    df = DOCTORS[DOCTORS["specialty"].str.lower() == spec]

    if df.empty:
        df = DOCTORS

    row = df.sample(1).iloc[0]

    return {
        "doctor_name": safe_val(row.get("doctor_name")),
        "specialty": safe_val(row.get("specialty")),
        "city": safe_val(row.get("city")),
        "hospital": safe_val(row.get("hospital")),
        "experience_years": safe_val(row.get("experience_years")),
        "contact_url": safe_val(row.get("contact_url")),
    }


# ------------------------------------------------------------
# SAFE JSON SAVE
# ------------------------------------------------------------
def save_report(out, path="outputs/report.json"):
    def convert(o):
        if np is not None:
            if isinstance(o, np.integer):
                return int(o)
            if isinstance(o, np.floating):
                return float(o)
            if isinstance(o, np.ndarray):
                return o.tolist()
        if isinstance(o, dict):
            return {k: convert(v) for k, v in o.items()}
        if isinstance(o, list):
            return [convert(v) for v in o]
        return o

    safe_out = convert(out)

    os.makedirs("outputs", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(safe_out, f, indent=2, ensure_ascii=False)

    return path


# ------------------------------------------------------------
# FINAL — SAFE PDF GENERATOR
# ------------------------------------------------------------
from fpdf import FPDF


def save_report_pdf(out, path=None):
    """PDF generator with Unicode sanitization and width-safe formatting."""

    def clean(t):
        if not isinstance(t, str):
            t = str(t)

        # Remove emojis / unicode that FPDF cannot render
        t = t.encode("latin-1", "ignore").decode("latin-1")

        # Remove remaining non-printable ASCII
        t = re.sub(r"[^\x20-\x7E]", "", t)

        # Normalize whitespace
        t = t.replace("\n", " ").replace("\t", " ")
        t = re.sub(r"\s+", " ", t)

        return t.strip() or "-"

    ts = datetime.datetime.now().strftime("%Y%m%d%H%M%S")

    if path is None:
        path = f"outputs/niva_report_{ts}.pdf"

    os.makedirs("outputs", exist_ok=True)

    class PDF(FPDF):
        def header(self):
            self.set_fill_color(0, 90, 160)
            self.rect(0, 0, 210, 25, "F")
            self.set_text_color(255, 255, 255)
            self.set_font("Arial", "B", 18)
            self.cell(0, 12, "NIVA Medical Consultation Report", ln=True, align="C")
            self.ln(8)

        def footer(self):
            self.set_y(-15)
            self.set_font("Arial", "I", 8)
            self.set_text_color(150, 150, 150)
            self.cell(0, 10, "Generated by NIVA © 2025", align="C")

    pdf = PDF()
    pdf.set_auto_page_break(True, 15)
    pdf.add_page()

    structured = out["structured"]
    triage = out["triage"]
    doctor = out["doctor"]
    conv = out["conversation"]

    def section(title):
        pdf.set_fill_color(230, 245, 255)
        pdf.set_font("Arial", "B", 13)
        pdf.set_text_color(0, 0, 0)
        pdf.cell(0, 10, title, ln=True, fill=True)
        pdf.ln(2)
        pdf.set_font("Arial", "", 12)

    # Structured
    section("Structured Information")
    for label, key in [
        ("Symptoms", "symptoms"),
        ("Duration", "duration"),
        ("Severity", "severity"),
        ("Additional Symptoms", "additional_symptoms"),
        ("Medical History", "medical_history"),
        ("Provisional Urgency", "provisional_urgency"),
    ]:
        pdf.multi_cell(0, 8, f"{label}: {clean(structured.get(key, ''))}")
        pdf.ln(1)

    # Triage
    section("Triage Recommendation")
    for key in [
        "severity",
        "urgency",
        "triage_score",
        "priority",
        "probable_cause",
        "recommended_department",
        "notes",
    ]:
        pdf.multi_cell(0, 8, f"{key.title()}: {clean(str(triage.get(key, '')))}")
        pdf.ln(1)

    # Doctor
    section("Assigned Doctor")
    for k, v in doctor.items():
        pdf.multi_cell(0, 8, f"{k.replace('_', ' ').title()}: {clean(str(v))}")
        pdf.ln(1)

    # Conversation
    section("Conversation Transcript")
    for role, text in conv:
        who = "Patient" if role == "patient" else "NIVA"
        pdf.multi_cell(0, 7, f"{who}: {clean(text)}")
        pdf.ln(1)

    pdf.ln(4)
    pdf.set_font("Arial", "I", 11)
    pdf.cell(0, 8, f"Report ID: RPT-{ts}", ln=True)

    pdf.output(path)
    return path
