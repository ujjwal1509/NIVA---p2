# langchain_pipeline.py
# FULL pipeline: Ollama-based question generation + CSV-guided symptom suggestions,
# structured extraction, triage, doctor assignment (from doctors.csv),
# save JSON and safe PDF generation.
#
# Updates (2025-12-05):
# - Mandatory duration parsing
# - Ollama-driven triage classifier with strict JSON output
# - triage_score -> priority mapping (HIGH/MEDIUM/LOW)
# - Heuristic fallback when Ollama unavailable or returns invalid JSON
# - Removed duplicate urgency assignment; final urgency comes from classifier

import os
import json
import re
import datetime
import logging
from dotenv import load_dotenv
import requests

# optional heavy imports only when needed
try:
    import pandas as pd
except Exception:
    pd = None

load_dotenv()

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# --------------------- Ollama Config ---------------------
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://127.0.0.1:11434")
MODEL_NAME = os.getenv("MODEL_NAME", "llama3")


def call_ollama(prompt, max_tokens=120, temperature=0.3, timeout=60):
    """
    Simple wrapper to call Ollama /api/generate.
    Returns plain string (best-effort).
    """
    url = OLLAMA_URL.rstrip("/") + "/api/generate"
    payload = {
        "model": MODEL_NAME,
        "prompt": prompt,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "stream": False,
    }
    try:
        r = requests.post(url, json=payload, timeout=timeout)
        data = r.json()
        # sometimes Ollama returns nested fields; try common keys
        if isinstance(data, dict):
            # prefer top-level "response"
            if "response" in data:
                return data.get("response", "").strip()
            # fallback: join chunked responses
            if "choices" in data and isinstance(data["choices"], list):
                return " ".join(c.get("text", "") for c in data["choices"]).strip()
        # last fallback to str
        return str(data)
    except Exception as e:
        logger.warning("O llama call failed: %s", e)
        return ""  # return empty to trigger fallback logic


# -----------------------------------------------------------
#              CSV DATASET LOADING (SYMPTOM & DOCTORS)
# -----------------------------------------------------------

SYMPTOM_DF = None
SYMPTOM_COLS = []
DATASET_LOADED = False

CSV_PATH = os.getenv("CSV_PATH", "data/niva_dataset1.csv")

if pd is not None and os.path.exists(CSV_PATH):
    try:
        df_sym = pd.read_csv(CSV_PATH)
        if "prognosis" in df_sym.columns:
            SC = [c for c in df_sym.columns if c.lower() != "prognosis"]
        else:
            SC = list(df_sym.columns[:-1])
        SYMPTOM_DF = df_sym
        SYMPTOM_COLS = SC
        DATASET_LOADED = True
    except Exception:
        DATASET_LOADED = False

DOCTORS = None
DOCTORS_PATH = os.path.join("data", "doctors.csv")
if pd is not None and os.path.exists(DOCTORS_PATH):
    try:
        DOCTORS = pd.read_csv(DOCTORS_PATH)
    except Exception:
        DOCTORS = None


# helper to normalise names
def _normalize_symptom_name(col: str) -> str:
    col = col.replace("_", " ")
    col = re.sub(r"\s+", " ", col)
    return col.strip().lower()


# prepare normalized map for symptom detection
NORMALIZED_SYMPTOM_MAP = {}
if DATASET_LOADED:
    for c in SYMPTOM_COLS:
        NORMALIZED_SYMPTOM_MAP[c] = _normalize_symptom_name(c)


def extract_symptom_keywords_from_text(text: str):
    """
    Very-lightweight keyword detection of symptoms in free text using
    normalized symptom column names. Returns set of symptom column keys.
    """
    if not DATASET_LOADED or not text:
        return set()
    text_norm = text.lower()
    text_norm = re.sub(r"\s+", " ", text_norm)
    found = set()
    for col, phrase in NORMALIZED_SYMPTOM_MAP.items():
        if phrase and phrase in text_norm:
            found.add(col)
    return found


def _suggest_next_symptom(known_symptoms: set):
    """
    Use dataset co-occurrence to pick a next symptom column to ask about.
    Returns column name or None.
    """
    if not DATASET_LOADED or SYMPTOM_DF is None or len(SYMPTOM_COLS) == 0:
        return None

    df = SYMPTOM_DF
    sub = df

    mask = None
    for s in known_symptoms:
        if s in SYMPTOM_COLS:
            cond = df[s] == 1
            mask = cond if mask is None else (mask & cond)
    if mask is not None:
        sub = df[mask]
        if sub.empty:
            sub = df  # fallback

    if sub.empty:
        return None

    freqs = sub[SYMPTOM_COLS].mean(numeric_only=True)
    for s in known_symptoms:
        if s in freqs.index:
            freqs.loc[s] = 0.0

    if freqs.empty:
        return None

    best_sym = freqs.idxmax()
    if freqs[best_sym] <= 0:
        return None
    return best_sym


def _build_symptom_question(symptom_col: str) -> str:
    phrase = _normalize_symptom_name(symptom_col)
    phrase_readable = phrase[0].upper() + phrase[1:] if phrase else symptom_col
    return f"Are you experiencing {phrase_readable}?"


# -----------------------------------------------------------
#              ML-DRIVEN QUESTION GENERATOR (SAFE)
# -----------------------------------------------------------


def generate_conversational_reply(
    chat_messages,
    user_message,
    k_context: int = 3,
    max_turns: int = 6,
    temperature: float = 0.3,
):
    """
    Generate a single follow-up question.
    1) Try dataset-guided question (if dataset present).
    2) Otherwise fall back to Ollama LLM prompt.
    Stops after 5 patient replies.
    """
    patient_answers = sum(1 for r, _ in chat_messages if r == "patient")
    if patient_answers >= 5:
        return "Thank you. I have collected all the required information."

    dataset_question = None
    known_symptoms = set()

    # gather known symptoms from patient messages
    for role, text in chat_messages:
        if role == "patient":
            known_symptoms |= extract_symptom_keywords_from_text(text)
    known_symptoms |= extract_symptom_keywords_from_text(user_message)

    # dataset-guided next symptom
    next_symptom = _suggest_next_symptom(known_symptoms) if DATASET_LOADED else None
    if next_symptom:
        dataset_question = _build_symptom_question(next_symptom)
    if dataset_question:
        return dataset_question

    # fallback to LLM
    convo = ""
    for role, text in chat_messages:
        tag = "Patient" if role == "patient" else "Bot"
        convo += f"{tag}: {text}\n"

    known_symptom_text = ""
    if known_symptoms:
        readable = ", ".join(_normalize_symptom_name(s) for s in sorted(known_symptoms))
        known_symptom_text = f"Known symptoms from previous messages: {readable}.\n"

    prompt = (
        "You are NIVA, an intelligent medical intake assistant.\n"
        "Your job is to ask ONLY 1 medically relevant follow-up question.\n"
        "Use the patient's last answer and the known symptoms.\n"
        "Do NOT give diagnosis.\n"
        "Do NOT explain.\n"
        "Ask exactly ONE clear medical question.\n\n"
        f"{known_symptom_text}"
        "Conversation so far:\n"
        + convo
        + '\nPatient just said: "'
        + user_message
        + '"\nNow ask the next medically relevant question.'
    )

    reply = call_ollama(prompt, max_tokens=80, temperature=temperature)

    if ":" in reply:
        reply = reply.split(":", 1)[-1].strip()

    return reply


# -----------------------------------------------------------
#                     STRUCTURED EXTRACTION
# -----------------------------------------------------------


def extract_structured_from_conversation(conv_text: str):
    """
    Simple extraction: pick Patient: answers in order, map to 5 fields.
    Keeps a provisional 'urgency' inferred from the 'severity' phrase for backward compatibility,
    but final urgency and triage_score will come from the classifier.
    """
    lines = conv_text.split("\n")
    answers = [
        line.split(":", 1)[1].strip() for line in lines if line.startswith("Patient:")
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
    # provisional urgency (kept for compatibility; classifier will override)
    sev = structured.get("severity", "").lower()
    if "severe" in sev:
        structured["provisional_urgency"] = "high"
    elif "moderate" in sev:
        structured["provisional_urgency"] = "medium"
    else:
        structured["provisional_urgency"] = "low"
    return structured


# -----------------------------------------------------------
#                DURATION PARSING UTIL (MANDATORY)
# -----------------------------------------------------------


def parse_duration(duration_str: str):
    """
    Parse user-entered duration strings and return (value: float, unit: 'hours'|'days'|'weeks').
    If parsing fails or text is not numeric, return (None, None) to indicate invalid/missing duration.
    Examples supported:
      "2 hours", "3 days", "1 week", "48h", "since 2 days", "for two weeks"
    """
    if not duration_str or not isinstance(duration_str, str):
        return None, None

    s = duration_str.strip().lower()

    # common numeric words -> digits (basic)
    num_words = {
        "one": 1,
        "two": 2,
        "three": 3,
        "four": 4,
        "five": 5,
        "six": 6,
        "seven": 7,
        "eight": 8,
        "nine": 9,
        "ten": 10,
    }
    # remove commas
    s = s.replace(",", " ")
    # look for direct digits
    m = re.search(
        r"(\d+(?:\.\d+)?)\s*(h|hr|hrs|hour|hours|d|day|days|w|wk|wks|week|weeks)?", s
    )
    if m:
        val = float(m.group(1))
        unit_token = m.group(2) or ""
        if unit_token.startswith("h"):
            return val, "hours"
        if unit_token.startswith("d"):
            return val, "days"
        if unit_token.startswith("w"):
            return val, "weeks"
        # no unit -> assume days for multi-digit, hours for small numbers? choose days by default
        return val, "days"

    # try word numbers
    for word, digit in num_words.items():
        if re.search(r"\b" + re.escape(word) + r"\b", s):
            if "hour" in s or "hr" in s or "h " in s:
                return float(digit), "hours"
            if "week" in s:
                return float(digit), "weeks"
            # default to days
            return float(digit), "days"

    # last attempt: contains "since" but no numeric -> None (ask user)
    return None, None


# -----------------------------------------------------------
#                        TRIAGE (AI + heuristic fallback)
# -----------------------------------------------------------

TRIAGE_PROMPT_TEMPLATE = """
You are a medical triage classifier.

Input (JSON):
{input_json}

Return STRICT JSON exactly like this (no extra commentary):
{{
  "severity": "low | moderate | high | critical",
  "urgency": "non-urgent | soon | urgent | immediate",
  "probable_cause": "string (short)",
  "recommended_department": "ENT / Cardiology / Neurology / General Medicine / Dermatology / Ortho / Emergency",
  "triage_score": number (1-10),
  "notes": "string (brief, optional)"
}}

Rules:
- Use the provided duration (value + unit) to help decide severity and urgency.
- Dangerous red flags (chest pain, severe bleeding, loss of consciousness, severe breathlessness) should result in high/critical severity and immediate/urgent urgency.
- When unsure, err on the side of safety (choose higher urgency).
- Return ONLY the JSON object, no explanation or extra text.
"""


def build_triage_prompt(structured: dict):
    # prepare a minimal, clear input for the LLM
    duration_value, duration_unit = parse_duration(structured.get("duration", "") or "")
    input_payload = {
        "symptoms": structured.get("symptoms", ""),
        "duration": {"value": duration_value, "unit": duration_unit},
        "severity_note": structured.get("severity", ""),
        "additional_symptoms": structured.get("additional_symptoms", ""),
        "medical_history": structured.get("medical_history", ""),
    }
    # ensure JSON is compact and safe
    return TRIAGE_PROMPT_TEMPLATE.format(
        input_json=json.dumps(input_payload, ensure_ascii=False)
    )


def triage_score_to_priority(score):
    try:
        s = float(score)
    except Exception:
        return "LOW"
    if s >= 8:
        return "HIGH"
    if s >= 5:
        return "MEDIUM"
    return "LOW"


def safe_load_json(text: str):
    """
    Attempt to load JSON from text; tolerate leading/trailing text by extracting first {...} block.
    """
    if not text or not isinstance(text, str):
        return None
    text = text.strip()
    # extract first JSON object
    m = re.search(r"(\{.*\})", text, flags=re.DOTALL)
    candidate = m.group(1) if m else text
    try:
        return json.loads(candidate)
    except Exception:
        # try to fix common issues: replace single quotes, trailing commas
        try:
            candidate2 = candidate.replace("'", '"')
            candidate2 = re.sub(r",\s*}", "}", candidate2)
            candidate2 = re.sub(r",\s*]", "]", candidate2)
            return json.loads(candidate2)
        except Exception:
            return None


def classify_triage(structured: dict, temperature: float = 0.2, max_tokens: int = 200):
    """
    Use Ollama to classify triage and return stable dict with fields:
    severity, urgency, probable_cause, recommended_department, triage_score, notes, priority
    Falls back to heuristic if LLM unavailable or returns invalid JSON.
    """
    prompt = build_triage_prompt(structured)
    logger.info("Calling Ollama for triage classification.")
    raw = call_ollama(
        prompt, max_tokens=max_tokens, temperature=temperature, timeout=20
    )
    parsed = safe_load_json(raw)
    if parsed and isinstance(parsed, dict):
        # validate required fields
        required = ["severity", "urgency", "triage_score", "recommended_department"]
        if not all((k in parsed) for k in required):
            logger.warning(
                "Ollama response missing required keys; using fallback. Raw: %s", raw
            )
            parsed = None

    if parsed is None:
        # Heuristic fallback (conservative)
        logger.warning("Using heuristic fallback for triage (Ollama missing/invalid).")
        symptoms = (structured.get("symptoms") or "").lower()
        severity = "low"
        score = 1.0
        # quick red flags
        red_flags = [
            "chest pain",
            "severe bleeding",
            "faint",
            "loss of consciousness",
            "difficulty breathing",
            "severe breath",
        ]
        if any(rf in symptoms for rf in red_flags):
            severity = "critical"
            score = 9.0
        else:
            # use words in severity note
            sev_note = (structured.get("severity") or "").lower()
            if "severe" in sev_note:
                severity = "high"
                score = 7.0
            elif "moderate" in sev_note:
                severity = "moderate"
                score = 5.5
            else:
                severity = "low"
                score = 2.0
            # use duration influence
            dval, dunit = parse_duration(structured.get("duration", "") or "")
            if dval is not None:
                if dunit == "hours":
                    score += 1.0
                elif dunit == "days":
                    score += 0.5
                elif dunit == "weeks":
                    score += 0.0

        urgency = (
            "immediate" if score >= 8 else ("urgent" if score >= 5 else "non-urgent")
        )
        recommended_department = (
            "Emergency" if severity in ("critical", "high") else "General Medicine"
        )
        parsed = {
            "severity": severity,
            "urgency": urgency,
            "probable_cause": "Unknown (heuristic)",
            "recommended_department": recommended_department,
            "triage_score": score,
            "notes": "Heuristic fallback used; configure Ollama for better results.",
        }

    # ensure stable types and derive priority
    try:
        triage_score = float(parsed.get("triage_score", 1))
    except Exception:
        triage_score = 1.0
    priority = triage_score_to_priority(triage_score)

    out = {
        "severity": parsed.get("severity"),
        "urgency": parsed.get("urgency"),
        "probable_cause": parsed.get("probable_cause"),
        "recommended_department": parsed.get("recommended_department"),
        "triage_score": triage_score,
        "notes": parsed.get("notes", ""),
        "priority": priority,
    }
    return out


# old triage_report now uses classifier
def triage_report(structured: dict):
    """
    Return triage dict produced by classifier. The classifier expects
    structured to contain duration; if duration can't be parsed, still proceed
    but classifier's heuristic will note it.
    """
    # ensure duration exists
    dval, dunit = parse_duration(structured.get("duration", "") or "")
    if dval is None:
        logger.info(
            "Duration missing or unparseable in structured input: '%s'",
            structured.get("duration", ""),
        )
        # we still call classifier so it can use provisional information,
        # but callers should enforce that duration is required in the UI layer.
    triage = classify_triage(structured)
    # map specialist field for compatibility (original code relied on 'specialist')
    # recommended_department -> specialist mapping (small heuristic)
    dept = (triage.get("recommended_department") or "").lower()
    specialist = "General Physician"
    if "emergency" in dept:
        specialist = "Emergency Care"
    elif "cardio" in dept or "cardiology" in dept:
        specialist = "Cardiology"
    elif "ent" in dept:
        specialist = "ENT"
    elif "neuro" in dept:
        specialist = "Neurology"
    elif "derm" in dept:
        specialist = "Dermatology"
    elif "ortho" in dept:
        specialist = "Orthopedics"
    else:
        specialist = "General Physician"

    triage_out = {
        "specialist": specialist,
        "urgency": triage.get("urgency"),
        "severity": triage.get("severity"),
        "triage_score": triage.get("triage_score"),
        "priority": triage.get("priority"),
        "probable_cause": triage.get("probable_cause"),
        "recommended_department": triage.get("recommended_department"),
        "notes": triage.get("notes"),
    }
    return triage_out


# -----------------------------------------------------------
#           DOCTOR ASSIGNMENT USING CSV DATASET
# -----------------------------------------------------------


def assign_doctor(triage: dict):
    """
    Pick a doctor row from data/doctors.csv matching triage['specialist'].
    Returns a dict with doctor details. Fallbacks gracefully.
    """
    if DOCTORS is None:
        return {
            "doctor_name": "Not available",
            "specialty": triage.get("specialist", ""),
            "city": "",
            "hospital": "",
            "experience_years": None,
            "contact_url": "",
        }

    spec = (triage.get("specialist") or "").lower()
    # simple heuristics: exact match, otherwise contains, otherwise fallback to any
    try:
        candidates = DOCTORS[DOCTORS["specialty"].str.lower() == spec]
    except Exception:
        candidates = DOCTORS

    if candidates.empty:
        try:
            candidates = DOCTORS[
                DOCTORS["specialty"].str.lower().str.contains(spec.split()[0])
            ]
        except Exception:
            candidates = DOCTORS

    if candidates.empty:
        candidates = DOCTORS

    row = candidates.sample(1).iloc[0]
    return {
        "doctor_id": int(row.get("doctor_id", -1)) if "doctor_id" in row else -1,
        "doctor_name": row.get("doctor_name", "Unknown"),
        "specialty": row.get("specialty", ""),
        "city": row.get("city", ""),
        "hospital": row.get("hospital", ""),
        "experience_years": int(row.get("experience_years", 0))
        if pd is not None and "experience_years" in row
        else None,
        "contact_url": row.get("contact_url", "") if "contact_url" in row else "",
    }


# -----------------------------------------------------------
#                      SAVE JSON
# -----------------------------------------------------------


def save_report(out: dict, path="outputs/report.json"):
    os.makedirs("outputs", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    return path


# -----------------------------------------------------------
#                     SAFE PDF GENERATOR
# -----------------------------------------------------------


def save_report_pdf(out: dict, path: str = None):
    """
    Generate a safe PDF using fpdf. This import is local to the function to avoid
    requiring fpdf at module import time if not needed.
    """
    try:
        from fpdf import FPDF
    except Exception as e:
        raise RuntimeError(
            "fpdf is required for PDF generation. Install via `pip install fpdf`."
        ) from e

    def clean(t):
        if not isinstance(t, str):
            t = str(t)
        t = t.replace("—", "-").replace("–", "-")
        t = t.replace("“", '"').replace("”", '"')
        t = t.replace("‘", "'").replace("’", "'")
        t = re.sub(r"[\x00-\x1F\x7F]", " ", t)
        t = t.encode("latin-1", "ignore").decode("latin-1")
        return t.strip()

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
            self.cell(0, 12, "NIVA Medical Consultation Report", align="C", ln=True)
            self.ln(10)

        def footer(self):
            self.set_y(-15)
            self.set_font("Arial", "I", 8)
            self.set_text_color(100, 100, 100)
            self.cell(0, 10, "Generated by NIVA Healthcare System © 2025", align="C")

    pdf = PDF()
    pdf.set_auto_page_break(True, 15)
    pdf.add_page()

    structured = out.get("structured", {})
    triage = out.get("triage", {})
    conv = out.get("conversation", [])
    doctor = out.get("doctor", {})

    def section_title(text):
        pdf.set_fill_color(230, 245, 255)
        pdf.set_text_color(0, 0, 0)
        pdf.set_font("Arial", "B", 13)
        pdf.cell(0, 10, text, ln=True, fill=True)
        pdf.ln(2)
        pdf.set_font("Arial", "", 12)

    LEFT = 10
    WIDTH = 190

    # Structured Information
    section_title("Structured Information")
    items = [
        ("Symptoms", structured.get("symptoms", "")),
        ("Duration", structured.get("duration", "")),
        ("Severity (note)", structured.get("severity", "")),
        ("Additional Symptoms", structured.get("additional_symptoms", "")),
        ("Medical History", structured.get("medical_history", "")),
        ("Provisional Urgency", structured.get("provisional_urgency", "")),
    ]
    for label, value in items:
        pdf.set_x(LEFT)
        pdf.set_font("Arial", "B", 12)
        pdf.multi_cell(WIDTH, 7, f"{label}:")
        pdf.set_x(LEFT)
        pdf.set_font("Arial", "", 12)
        pdf.multi_cell(WIDTH, 7, clean(value) if str(value).strip() else "-")
        pdf.ln(2)

    # Triage
    section_title("Triage Recommendation")
    # Ensure we show key triage fields including derived priority
    keys_to_show = [
        "specialist",
        "severity",
        "urgency",
        "triage_score",
        "priority",
        "probable_cause",
        "recommended_department",
        "notes",
    ]
    for key in keys_to_show:
        pdf.set_x(LEFT)
        pdf.set_font("Arial", "B", 12)
        pdf.multi_cell(WIDTH, 7, f"{key.title()}:")
        pdf.set_x(LEFT)
        pdf.set_font("Arial", "", 12)
        pdf.multi_cell(WIDTH, 7, clean(str(triage.get(key, "-"))))
        pdf.ln(2)

    # Assigned Doctor
    section_title("Assigned Doctor")
    if doctor:
        for k, v in doctor.items():
            pdf.set_x(LEFT)
            pdf.set_font("Arial", "B", 12)
            pdf.multi_cell(WIDTH, 7, f"{k.replace('_', ' ').title()}:")
            pdf.set_x(LEFT)
            pdf.set_font("Arial", "", 12)
            pdf.multi_cell(WIDTH, 7, clean(str(v)))
            pdf.ln(2)
    else:
        pdf.set_x(LEFT)
        pdf.set_font("Arial", "", 12)
        pdf.multi_cell(WIDTH, 7, "-")

    # Conversation Transcript
    section_title("Conversation Transcript")
    for role, text in conv:
        who = "Patient" if role == "patient" else "NIVA"
        pdf.multi_cell(WIDTH, 7, f"{who}: {clean(text)}")
        pdf.ln(1)

    pdf.ln(10)
    pdf.set_font("Arial", "I", 11)
    pdf.cell(0, 8, f"Report ID: RPT-{ts}", ln=True)

    pdf.output(path)
    return path
