# stream-lit_app.py
# Updated: avoid st.stop() freeze; keep chat interactive while awaiting duration

import os
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

st.set_page_config(page_title="NIVA — Symptom Chatbot", layout="wide")
st.title("NIVA — Symptom Analysis Chatbot")


# --------- Lazy load pipeline (cached) ----------
@st.cache_resource
def load_pipeline():
    try:
        import langchain_pipeline as lp

        funcs = {
            "generate_reply": getattr(lp, "generate_conversational_reply", None),
            "extract_structured": getattr(
                lp, "extract_structured_from_conversation", None
            ),
            "triage": getattr(lp, "triage_report", None),
            "parse_duration": getattr(lp, "parse_duration", None),
            "save_report": getattr(lp, "save_report", None),
            "assign_doctor": getattr(lp, "assign_doctor", None),
            "save_report_pdf": getattr(lp, "save_report_pdf", None),
        }

        missing = [
            k
            for k, v in funcs.items()
            if v is None
            and k
            in (
                "generate_reply",
                "extract_structured",
                "triage",
                "save_report",
                "assign_doctor",
            )
        ]
        if missing:
            raise ImportError(f"langchain_pipeline missing required: {missing}")
        return funcs
    except Exception as e:
        raise ImportError(str(e)) from e


# --------- Session State ---------
STARTER_Q = (
    "What problem are you facing? Please describe your main symptom in one sentence."
)

if "messages" not in st.session_state:
    st.session_state.messages = [("bot", STARTER_Q)]
if "finished" not in st.session_state:
    st.session_state.finished = False
if "structured" not in st.session_state:
    st.session_state.structured = None
if "json_path" not in st.session_state:
    st.session_state.json_path = None
if "pdf_path" not in st.session_state:
    st.session_state.pdf_path = None
if "doctor" not in st.session_state:
    st.session_state.doctor = None
if "triage" not in st.session_state:
    st.session_state.triage = None
if "pipeline_loaded" not in st.session_state:
    st.session_state.pipeline_loaded = False
if "init_error" not in st.session_state:
    st.session_state.init_error = None

# New flags for duration flow
if "awaiting_duration" not in st.session_state:
    st.session_state.awaiting_duration = False
if "asked_duration_after_5" not in st.session_state:
    st.session_state.asked_duration_after_5 = False

# Load pipeline eagerly
try:
    pipeline = load_pipeline()
    st.session_state.pipeline_loaded = True
except Exception as e:
    pipeline = None
    st.session_state.pipeline_loaded = False
    st.session_state.init_error = str(e)


# --------- Layout ---------
col_main, col_side = st.columns([3, 1])

with col_main:
    st.subheader("Chat")

    # Chat input area (always rendered while conversation not finished)
    if not st.session_state.finished:
        # Use a simple form so Enter/Send works
        with st.form("chat_form", clear_on_submit=True):
            user_input = st.text_input("Your reply", placeholder="Type your reply...")
            submitted = st.form_submit_button("Send")

        finish_manual = st.button("Finish Now & Generate Report")

        # Handle submission
        if submitted:
            if not user_input.strip():
                st.warning("Please type something before sending.")
            else:
                # Append patient message
                st.session_state.messages.append(("patient", user_input.strip()))

                # If bot previously asked for duration and we're awaiting it -> validate
                if st.session_state.awaiting_duration and pipeline is not None:
                    parse_duration = pipeline.get("parse_duration")
                    if callable(parse_duration):
                        dval, dunit = parse_duration(user_input.strip())
                        if dval is not None:
                            # Valid duration -> proceed to finish and generate report
                            st.session_state.awaiting_duration = False
                            st.session_state.finished = True
                        else:
                            # Not parseable -> append friendly clarification and keep awaiting_duration True
                            st.session_state.messages.append(
                                (
                                    "bot",
                                    "I still couldn't parse that. Please answer like '2 days' or '5 hours'.",
                                )
                            )
                            # leave finished False so user can send again
                    else:
                        # no parse function -> fallback to finishing
                        st.session_state.awaiting_duration = False
                        st.session_state.finished = True

                else:
                    # Normal flow: call LLM to generate next question/response
                    if pipeline is None:
                        st.error("Pipeline not loaded.")
                    else:
                        generate_reply = pipeline["generate_reply"]
                        with st.spinner("Thinking..."):
                            try:
                                bot_reply = generate_reply(
                                    st.session_state.messages,
                                    user_input.strip(),
                                    k_context=3,
                                    max_turns=6,
                                )
                            except Exception as e:
                                bot_reply = (
                                    "Sorry — I couldn't generate a reply right now."
                                )
                                st.error(f"Model error: {e}")

                        st.session_state.messages.append(("bot", bot_reply))

                    # After adding reply, check patient count and if we hit 5, ask for duration explicitly
                    patient_count = sum(
                        1 for r, _ in st.session_state.messages if r == "patient"
                    )
                    if (
                        patient_count >= 5
                        and not st.session_state.asked_duration_after_5
                    ):
                        st.session_state.messages.append(
                            (
                                "bot",
                                "Before I generate the report, please tell me how long you've had the symptom (e.g., '2 days', '5 hours').",
                            )
                        )
                        st.session_state.awaiting_duration = True
                        st.session_state.asked_duration_after_5 = True
                        # do not set finished yet; wait for patient duration reply

        # Manual finish button behavior: ensure duration available or ask for it
        if finish_manual:
            if pipeline is not None:
                parse_duration = pipeline.get("parse_duration")
                extract = pipeline.get("extract_structured")
                conv_text = "\n".join(
                    [
                        f"{'Patient' if r == 'patient' else 'Bot'}: {t}"
                        for r, t in st.session_state.messages
                    ]
                )
                try:
                    structured_try = extract(conv_text) if callable(extract) else {}
                except Exception:
                    structured_try = {}

                dval, dunit = (None, None)
                if callable(parse_duration):
                    dval, dunit = parse_duration(
                        structured_try.get("duration", "") or ""
                    )

                if dval is None:
                    # Ask for duration first; do not finish
                    st.session_state.messages.append(
                        (
                            "bot",
                            "Before generating the report, please tell me how long you've had the symptom (e.g., '2 days').",
                        )
                    )
                    st.session_state.awaiting_duration = True
                    st.session_state.finished = False
                else:
                    st.session_state.finished = True
            else:
                st.session_state.finished = True

    # ---------- Report Generation ----------
    if st.session_state.finished and st.session_state.structured is None:
        conv_text = "\n".join(
            [
                f"{'Patient' if r == 'patient' else 'Bot'}: {t}"
                for r, t in st.session_state.messages
            ]
        )

        if pipeline is None:
            st.error("Pipeline not available.")
        else:
            extract = pipeline["extract_structured"]
            parse_duration = pipeline.get("parse_duration")

            # Extract fields
            try:
                structured = extract(conv_text)
            except Exception as e:
                structured = {"notes": conv_text}
                st.error(f"Extraction failed: {e}")

            st.session_state.structured = structured

            # Validate duration before calling triage
            dval, dunit = (None, None)
            if callable(parse_duration):
                dval, dunit = parse_duration(structured.get("duration", "") or "")

            if dval is None:
                # Ask for duration and do not generate report
                st.warning("I couldn't understand how long you've had the symptom.")
                st.info(
                    "Please answer: For how long have you had this symptom (e.g., '2 days', '5 hours')?"
                )
                st.session_state.messages.append(
                    (
                        "bot",
                        "Please tell me how long you've had the symptom (e.g., '2 days').",
                    )
                )
                st.session_state.awaiting_duration = True
                st.session_state.finished = False
                st.session_state.structured = None
                # allow the run to finish normally so user can reply (no st.stop())
            else:
                # ---------- If duration OK → generate report ----------
                try:
                    triage = pipeline["triage"](structured)
                    st.session_state.triage = triage

                    doctor = pipeline["assign_doctor"](triage)
                    st.session_state.doctor = doctor

                    out = {
                        "conversation": st.session_state.messages,
                        "structured": structured,
                        "triage": triage,
                        "doctor": doctor,
                    }

                    json_path = pipeline["save_report"](out)
                    st.session_state.json_path = json_path

                    # PDF
                    try:
                        pdf_path = pipeline["save_report_pdf"](out)
                    except Exception as e:
                        pdf_path = None
                        st.error(f"PDF generation failed: {e}")
                    st.session_state.pdf_path = pdf_path

                    st.success(
                        f"Report generated — JSON: {json_path}"
                        + (f", PDF: {pdf_path}" if pdf_path else "")
                    )

                except Exception as e:
                    st.error(f"Triage failed: {e}")

    # ---------- Final UI ----------
    if st.session_state.finished:
        st.markdown("### Conversation Complete")
        st.write("NIVA has generated your report.")
        if st.session_state.pdf_path and os.path.exists(st.session_state.pdf_path):
            with open(st.session_state.pdf_path, "rb") as f:
                st.download_button(
                    "Download PDF Report",
                    data=f.read(),
                    file_name=os.path.basename(st.session_state.pdf_path),
                    mime="application/pdf",
                )

    # Render conversation (always)
    for role, text in st.session_state.messages:
        st.markdown(f"**{'You' if role == 'patient' else 'NIVA'}:** {text}")


# --------- Sidebar ---------
with col_side:
    st.subheader("Status")
    if st.session_state.pipeline_loaded:
        st.success("Pipeline loaded")
    else:
        st.error("Pipeline failed to load.")
        st.code(st.session_state.init_error)

    st.markdown("---")

    if st.button("Reset Conversation"):
        # reset to initial state without reloading pipeline
        st.session_state.messages = [("bot", STARTER_Q)]
        st.session_state.finished = False
        st.session_state.structured = None
        st.session_state.json_path = None
        st.session_state.pdf_path = None
        st.session_state.doctor = None
        st.session_state.triage = None
        st.session_state.awaiting_duration = False
        st.session_state.asked_duration_after_5 = False
        st.rerun()

    st.markdown("---")
    st.subheader("Report & Doctor")

    # Show structured JSON and triage details
    if st.session_state.finished and st.session_state.structured:
        st.json(st.session_state.structured)
        if st.session_state.triage:
            st.markdown("#### Triage Details")
            st.json(st.session_state.triage)
        if st.session_state.doctor:
            st.markdown("#### Assigned Doctor")
            st.json(st.session_state.doctor)
    else:
        st.info("Report will appear here after finishing.")

st.caption("Demo: conversational CSV-guided + LLM intake (NIVA © 2025)")
