# langchain_pipeline.py
# Manual LangChain-style pipeline (lazy init). Provides:
# - call_ollama
# - get_context_for_query (retriever)
# - generate_conversational_reply (conversational GPT-like replies grounded by RAG)
# - extract_structured_from_conversation
# - triage_report
# - save_report

import os, json, re, traceback, requests
from dotenv import load_dotenv

load_dotenv()

# Config via .env (defaults)
PINECONE_INDEX = os.getenv("PINECONE_INDEX", "medical-knowledge")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_ENV = os.getenv("PINECONE_ENV", None)
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://127.0.0.1:11434")
MODEL_NAME = os.getenv("MODEL_NAME", "llama3")
EMBED_MODEL_NAME = "all-MiniLM-L6-v2"

# --------------------- Ollama caller (robust NDJSON-safe) ---------------------
OLLAMA_API = OLLAMA_URL.rstrip("/") + "/api/generate"


def call_ollama(
    prompt: str, max_tokens: int = 512, temperature: float = 0.0, timeout: int = 60
) -> str:
    payload = {
        "model": MODEL_NAME,
        "prompt": prompt,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "stream": False,
    }
    try:
        r = requests.post(OLLAMA_API, json=payload, timeout=timeout)
    except Exception as e:
        raise RuntimeError(f"Ollama request failed: {e}")
    text = (r.text or "").strip()
    # try direct JSON
    try:
        data = r.json()
        if isinstance(data, dict):
            if "response" in data:
                return data["response"]
            if "output" in data:
                out = data["output"]
                if isinstance(out, list):
                    return "".join(
                        p.get("content", "") if isinstance(p, dict) else str(p)
                        for p in out
                    )
                return str(out)
    except Exception:
        pass
    # NDJSON fallback: collect "response" fields
    out = ""
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
            if isinstance(obj, dict) and obj.get("response"):
                out += str(obj.get("response"))
        except Exception:
            continue
    if out.strip():
        return out.strip()
    return text


# --------------------- Lazy embedding & pinecone init helpers ---------------------
# We delay importing heavy libraries until needed.


def _get_embedder():
    """Return an embedding callable object with .encode() or .embed_query()"""
    # Prefer langchain_huggingface adapter if available
    try:
        from langchain_huggingface import HuggingFaceEmbeddings

        emb = HuggingFaceEmbeddings(model_name=EMBED_MODEL_NAME)
        return emb  # has embed_documents / embed_query in many adapters
    except Exception:
        pass
    # Fallback to sentence-transformers
    try:
        from sentence_transformers import SentenceTransformer

        model = SentenceTransformer(EMBED_MODEL_NAME)

        class _Simple:
            def encode(self, texts):
                # allow either single string or list
                return model.encode(texts)

            def embed_query(self, q):
                return model.encode(q).tolist()

        return _Simple()
    except Exception as e:
        raise RuntimeError(
            "No embedding model available. Install sentence-transformers or langchain_huggingface."
        ) from e


def _init_pinecone_index():
    """Initialize Pinecone index object lazily and return index instance"""
    try:
        # Try new-style client
        from pinecone import Pinecone

        pc = Pinecone(api_key=PINECONE_API_KEY)
        idx = pc.Index(PINECONE_INDEX)
        return idx
    except Exception:
        pass
    # classic client
    try:
        import pinecone

        try:
            if PINECONE_ENV:
                pinecone.init(api_key=PINECONE_API_KEY, environment=PINECONE_ENV)
            else:
                pinecone.init(api_key=PINECONE_API_KEY)
        except Exception:
            pass
        idx = pinecone.Index(PINECONE_INDEX)
        return idx
    except Exception as e:
        raise RuntimeError(f"Could not initialize Pinecone index: {e}") from e


# Cache resources in module globals after first init
_embedder = None
_pine_index = None
_lc_retriever = None
_use_lc_retriever = False


def get_context_for_query(query: str, k: int = 3) -> str:
    """Return top-k retrieved text snippets joined by double-newline."""
    global _embedder, _pine_index, _lc_retriever, _use_lc_retriever
    if _embedder is None:
        _embedder = _get_embedder()
    # try LangChain community vectorstore if available (lazy)
    if _lc_retriever is None and not _use_lc_retriever:
        try:
            from langchain_community.vectorstores import Pinecone as LC_Pinecone

            # If lc embed adapter exists, try to use it:
            try:
                # langchain_huggingface embedding instance may be different; skip strict checks
                emb = _embedder
                vectorstore = LC_Pinecone.from_existing_index(
                    index_name=PINECONE_INDEX, embedding=emb
                )
                _lc_retriever = vectorstore.as_retriever(search_kwargs={"k": k})
                _use_lc_retriever = True
            except Exception:
                _lc_retriever = None
                _use_lc_retriever = False
        except Exception:
            _use_lc_retriever = False
    if _use_lc_retriever and _lc_retriever is not None:
        try:
            docs = _lc_retriever.get_relevant_documents(query)
            ctx = "\n\n".join([d.page_content for d in docs])
            return ctx
        except Exception:
            # fallback to manual
            pass
    # manual retrieval
    if _pine_index is None:
        _pine_index = _init_pinecone_index()
    # embed query
    try:
        if hasattr(_embedder, "embed_query"):
            qvec = _embedder.embed_query(query)
        else:
            qvec = _embedder.encode(query).tolist()
    except Exception:
        # If embed returns numpy array
        try:
            qvec = _embedder.encode(query)
            if hasattr(qvec, "tolist"):
                qvec = qvec.tolist()
        except Exception:
            qvec = None
    if qvec is None:
        return ""
    # query pinecone
    try:
        res = _pine_index.query(vector=qvec, top_k=k, include_metadata=True)
        matches = res.get("matches", [])
        texts = [m.get("metadata", {}).get("text", "") for m in matches]
        return "\n\n".join(texts)
    except Exception as e:
        # If pinecone fails, return empty context
        return ""


# --------------------- Conversational prompt building ---------------------
SYSTEM_PROMPT = (
    "System: You are NIVA, a careful clinical assistant. "
    "You must NOT provide a medical diagnosis. Ask clarifying, doctor-like questions "
    "when required. If the user's content includes emergency symptoms (severe chest pain, sudden fainting, heavy bleeding, severe shortness of breath), "
    "tell them to seek emergency care immediately and advise to call emergency services."
)


def build_conversational_prompt(
    chat_messages, user_message, retrieved_context="", max_turns=6
):
    """Build a single prompt containing system, retrieved context, and recent conversation + new user message."""
    recent = chat_messages[-(max_turns * 2) :] if chat_messages else []
    convo_lines = []
    for role, text in recent:
        if role == "patient":
            convo_lines.append(f"Patient: {text}")
        else:
            convo_lines.append(f"Bot: {text}")
    convo_lines.append(f"Patient: {user_message}")
    convo_block = "\n".join(convo_lines)
    prompt = SYSTEM_PROMPT + "\n\n"
    if retrieved_context:
        prompt += "Context (retrieved):\n" + retrieved_context.strip() + "\n\n"
    prompt += "Conversation:\n" + convo_block + "\n\n"
    prompt += "Assistant: Please reply concisely and ask clarifying questions if needed. Do NOT give a diagnosis. Output plain text only."
    return prompt


# --------------------- Conversational reply generator ---------------------
def generate_conversational_reply(
    chat_messages,
    user_message,
    k_context: int = 3,
    max_turns: int = 6,
    temperature: float = 0.0,
):
    """
    Main conversational function to call from UI.
    - chat_messages: list of ("patient"/"bot", text)
    - user_message: current user input string
    """
    try:
        ctx = get_context_for_query(user_message, k=k_context)
    except Exception:
        ctx = ""
    prompt = build_conversational_prompt(
        chat_messages, user_message, retrieved_context=ctx, max_turns=max_turns
    )
    reply = call_ollama(prompt, max_tokens=300, temperature=temperature, timeout=60)
    # strip repeated prefixes
    if reply.strip().startswith("Assistant:") or reply.strip().startswith("Bot:"):
        reply = "\n".join(reply.splitlines()[1:]).strip()
    return reply


# --------------------- Structured extraction ---------------------
# Use LangChain StructuredOutputParser if present, else strong-prompt fallback
try:
    from langchain.output_parsers import StructuredOutputParser, ResponseSchema

    # define schema
    schemas = [
        ResponseSchema(name="symptoms", description="List of reported symptoms"),
        ResponseSchema(name="duration", description="Onset/duration"),
        ResponseSchema(name="severity", description="mild/moderate/severe"),
        ResponseSchema(name="current_medication", description="list or empty"),
        ResponseSchema(name="allergies", description="list or empty"),
        ResponseSchema(name="urgency", description="low/medium/high"),
        ResponseSchema(name="notes", description="free text"),
    ]
    STRUCT_PARSER = StructuredOutputParser.from_response_schemas(schemas)
    FORMAT_INSTRUCTIONS = STRUCT_PARSER.get_format_instructions()
    _use_struct_parser = True
except Exception:
    STRUCT_PARSER = None
    FORMAT_INSTRUCTIONS = (
        "Output ONLY a single valid JSON object with keys:\n"
        "- symptoms: list of strings\n"
        "- duration: string\n"
        "- severity: string (mild/moderate/severe)\n"
        "- current_medication: list\n"
        "- allergies: list\n"
        "- urgency: string (low/medium/high)\n"
        "- notes: string\n"
        "Do not output anything except the JSON object."
    )
    _use_struct_parser = False


def extract_structured_from_conversation(conversation_text: str) -> dict:
    """Return a dict parsed from model's JSON output for the conversation."""
    prompt = (
        FORMAT_INSTRUCTIONS
        + "\n\nConversation:\n"
        + conversation_text
        + "\n\nNow output ONLY the JSON."
    )
    raw = call_ollama(prompt, max_tokens=700)
    if _use_struct_parser and STRUCT_PARSER is not None:
        try:
            parsed = STRUCT_PARSER.parse(raw)
            if isinstance(parsed, dict):
                return parsed
            return dict(parsed)
        except Exception:
            # fall through to manual parse
            pass
    # manual parse: find first JSON object
    m = re.search(r"(\{.*\})", raw, re.S)
    if m:
        try:
            return json.loads(m.group(1))
        except Exception:
            return {"notes": raw}
    return {"notes": raw}


# --------------------- Simple triage rules ---------------------
def triage_report(structured: dict) -> dict:
    symptoms_text = ""
    if isinstance(structured.get("symptoms"), list):
        symptoms_text = " ".join(structured.get("symptoms", []))
    else:
        symptoms_text = str(structured.get("symptoms", ""))
    s = symptoms_text.lower()
    if any(w in s for w in ["chest", "breath", "shortness", "palpit"]):
        return {"specialist": "Cardiology/Emergency", "urgency": "high"}
    if any(w in s for w in ["rash", "itch", "lesion"]):
        return {"specialist": "Dermatology", "urgency": "medium"}
    if any(w in s for w in ["headache", "dizzy", "seizure"]):
        return {"specialist": "Neurology", "urgency": "medium"}
    return {"specialist": "General Physician", "urgency": "low"}


# --------------------- Save helper ---------------------
def save_report(out: dict, path: str = "outputs/langchain_manual_report.json") -> str:
    os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    return path


# --------------------- PDF save helper (requires fpdf2) ---------------------
def save_report_pdf(out: dict, path: str = None) -> str:
    """
    Save a simple PDF report containing:
      - Title, datetime
      - Structured JSON (pretty printed)
      - Conversation (chat transcript)
    Returns the path to the saved PDF.
    """
    try:
        from fpdf import FPDF
    except Exception as e:
        raise RuntimeError("fpdf2 not installed. Run: pip install fpdf2") from e

    import datetime

    ts = datetime.datetime.utcnow().strftime("%Y-%m-%d_%H%M%S")
    if path is None:
        path = f"outputs/niva_report_{ts}.pdf"
    os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)

    # Prepare content
    structured = out.get("structured", {})
    conv = out.get("conversation", [])

    # Convert conversation to text lines
    conv_lines = []
    for role, text in conv:
        who = "Patient" if role == "patient" else "NIVA"
        conv_lines.append(f"{who}: {text}")

    # PDF generation
    pdf = FPDF()
    pdf.set_auto_page_break(True, margin=15)
    pdf.add_page()
    pdf.set_font("Arial", size=14)
    pdf.cell(0, 8, "NIVA - Symptom Analysis Report", ln=True)
    pdf.set_font("Arial", size=10)
    pdf.cell(0, 6, f"Generated (UTC): {ts}", ln=True)
    pdf.ln(4)

    # Structured JSON (pretty)
    pdf.set_font("Arial", "B", 11)
    pdf.cell(0, 6, "Structured Report:", ln=True)
    pdf.set_font("Courier", size=9)
    json_str = json.dumps(structured, indent=2, ensure_ascii=False)
    # split to lines that fit on the page
    for line in json_str.splitlines():
        # FPDF multiline
        pdf.multi_cell(0, 5, line)
    pdf.ln(4)

    # Conversation
    pdf.set_font("Arial", "B", 11)
    pdf.cell(0, 6, "Conversation:", ln=True)
    pdf.set_font("Courier", size=9)
    for line in conv_lines:
        pdf.multi_cell(0, 5, line)

    # Optional triage block if available in out
    if "triage" in out:
        pdf.ln(4)
        pdf.set_font("Arial", "B", 11)
        pdf.cell(0, 6, "Triage:", ln=True)
        pdf.set_font("Courier", size=10)
        pdf.multi_cell(
            0, 5, json.dumps(out.get("triage", {}), indent=2, ensure_ascii=False)
        )

    # Save
    pdf.output(path)
    return path
