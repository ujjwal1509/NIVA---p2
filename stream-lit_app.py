# stream-lit_app.py
# Streamlit UI for NIVA symptom chatbot (conversational)
# - Reset fixed
# - Auto-report after 4 patient messages
# - Saves JSON + PDF (requires save_report_pdf in langchain_pipeline.py)

import streamlit as st
import traceback
from dotenv import load_dotenv

load_dotenv()

st.set_page_config(page_title="NIVA — Symptom Chatbot", layout="wide")
st.title("NIVA — Symptom Analysis Chatbot (Conversational)")


# --------- Lazy load pipeline (cached) ----------
@st.cache_resource
def load_pipeline():
    try:
        import langchain_pipeline as lp

        funcs = {
            # function names expected in langchain_pipeline.py
            "generate_reply": getattr(lp, "generate_conversational_reply", None),
            "extract_structured": getattr(
                lp, "extract_structured_from_conversation", None
            ),
            "triage": getattr(lp, "triage_report", None),
            "save_report": getattr(lp, "save_report", None),
            "save_report_pdf": getattr(lp, "save_report_pdf", None),
            "get_context": getattr(lp, "get_context_for_query", None),
        }
        missing = [
            k
            for k, v in funcs.items()
            if v is None
            and k in ("generate_reply", "extract_structured", "triage", "save_report")
        ]
        if missing:
            # require core functions; PDF helper is optional
            raise ImportError(
                f"langchain_pipeline missing required functions: {missing}"
            )
        return funcs
    except Exception as e:
        raise ImportError(traceback.format_exc()) from e


# --------- Session state (bot-first proactive) ----------
STARTER_Q = (
    "What problem are you facing? Please describe your main symptom in one sentence."
)

if "messages" not in st.session_state:
    st.session_state.messages = [("bot", STARTER_Q)]
if "finished" not in st.session_state:
    st.session_state.finished = False
if "structured" not in st.session_state:
    st.session_state.structured = None
if "pipeline_loaded" not in st.session_state:
    st.session_state.pipeline_loaded = False
if "init_error" not in st.session_state:
    st.session_state.init_error = None

# --------- Layout ----------
col_main, col_side = st.columns([3, 1])

with col_main:
    st.subheader("Chat")

    # input inside a form so it clears on submit (avoids widget mutation errors)
    with st.form("chat_form", clear_on_submit=True):
        user_input = st.text_input(
            "Your reply", key="user_input", placeholder="Type your reply here"
        )
        submitted = st.form_submit_button("Send")

    # finish / reset buttons (outside form)
    help_col1, help_col2 = st.columns([1, 1])
    with help_col1:
        finish_pressed = st.button("Finish & Generate Report")
    with help_col2:
        reset_pressed = st.button("Reset Conversation")

    # Reset: set conversation back to first starter message and rerun
    if reset_pressed:
        st.session_state.messages = [("bot", STARTER_Q)]
        st.session_state.finished = False
        st.session_state.structured = None
        st.session_state.pipeline_loaded = False
        st.session_state.init_error = None
        st.success("Conversation reset")
        st.rerun()

    # Lazy load pipeline on first interaction
    pipeline = None
    if (submitted or finish_pressed) and not st.session_state.pipeline_loaded:
        try:
            pipeline = load_pipeline()
            st.session_state.pipeline_loaded = True
            st.session_state.init_error = None
        except Exception as e:
            st.session_state.pipeline_loaded = False
            st.session_state.init_error = str(e)

    if st.session_state.pipeline_loaded and pipeline is None:
        try:
            pipeline = load_pipeline()
        except Exception as e:
            st.session_state.init_error = str(e)
            pipeline = None

    # Handle Send (conversational)
    if submitted:
        if not user_input or user_input.strip() == "":
            st.warning("Please type something before sending.")
        else:
            # append user message
            st.session_state.messages.append(("patient", user_input.strip()))

            if not st.session_state.pipeline_loaded or pipeline is None:
                st.error("Pipeline not loaded. See sidebar for errors.")
            else:
                generate_reply = pipeline["generate_reply"]
                with st.spinner("Thinking..."):
                    try:
                        # call function; signature is (chat_messages, user_message, ...)
                        try:
                            bot_reply = generate_reply(
                                st.session_state.messages,
                                user_input.strip(),
                                k_context=3,
                                max_turns=6,
                                temperature=0.0,
                            )
                        except TypeError:
                            # fallback for different signature
                            bot_reply = generate_reply(
                                st.session_state.messages, user_input.strip()
                            )
                    except Exception as e:
                        bot_reply = "Sorry — I couldn't generate a reply right now."
                        st.error(f"Model error: {e}")
                st.session_state.messages.append(("bot", bot_reply))

            # --- Auto-finish if user has already answered 4 times (i.e., 4 patient messages) ---
            patient_count = sum(
                1 for r, _ in st.session_state.messages if r == "patient"
            )
            if patient_count >= 4:
                # perform the finish flow automatically
                st.session_state.finished = True
                conv_text = "\n".join(
                    [
                        f"{'Patient' if r == 'patient' else 'Bot'}: {t}"
                        for r, t in st.session_state.messages
                    ]
                )

                # ensure pipeline loaded
                if not st.session_state.pipeline_loaded or pipeline is None:
                    try:
                        pipeline = load_pipeline()
                        st.session_state.pipeline_loaded = True
                    except Exception as e:
                        st.session_state.init_error = str(e)
                        pipeline = None

                if pipeline is None:
                    st.error(
                        "Pipeline could not be loaded for automatic report generation."
                    )
                else:
                    # extract structured
                    extract_fn = pipeline["extract_structured"]
                    with st.spinner("Generating structured report..."):
                        try:
                            structured = extract_fn(conv_text)
                        except Exception as e:
                            st.error(f"Automatic extraction failed: {e}")
                            structured = {"notes": conv_text}
                        st.session_state.structured = structured

                    # triage + save JSON + save PDF
                    try:
                        triage_fn = pipeline["triage"]
                        save_fn = pipeline["save_report"]
                        triage = triage_fn(st.session_state.structured)
                        out = {
                            "conversation": st.session_state.messages,
                            "structured": st.session_state.structured,
                            "triage": triage,
                        }
                        json_path = save_fn(out)
                        # try to save pdf using pipeline helper if present
                        pdf_path = None
                        try:
                            # may raise if helper not implemented
                            pdf_helper = pipeline.get("save_report_pdf", None)
                            if callable(pdf_helper):
                                pdf_path = pdf_helper(out)
                            else:
                                # fallback import directly
                                from langchain_pipeline import save_report_pdf

                                pdf_path = save_report_pdf(out)
                        except Exception:
                            pdf_path = None
                        st.success(
                            f"Auto-report saved: {json_path}"
                            + (f", PDF: {pdf_path}" if pdf_path else "")
                        )
                    except Exception as e:
                        st.error(f"Auto-triage/save failed: {e}")

    # Finish: extract structured report (manual)
    if finish_pressed:
        st.session_state.finished = True
        conv_text = "\n".join(
            [
                f"{'Patient' if r == 'patient' else 'Bot'}: {t}"
                for r, t in st.session_state.messages
            ]
        )

        if not st.session_state.pipeline_loaded:
            try:
                pipeline = load_pipeline()
                st.session_state.pipeline_loaded = True
            except Exception as e:
                st.session_state.init_error = str(e)
                pipeline = None

        if pipeline is None:
            st.error("Pipeline not available for extraction.")
        else:
            extract_fn = pipeline["extract_structured"]
            with st.spinner("Generating structured report..."):
                try:
                    structured = extract_fn(conv_text)
                except Exception as e:
                    st.error(f"Extraction failed: {e}")
                    structured = {"notes": conv_text}
                st.session_state.structured = structured

            # triage + save JSON + save PDF
            try:
                triage_fn = pipeline["triage"]
                save_fn = pipeline["save_report"]
                triage = triage_fn(st.session_state.structured)
                out = {
                    "conversation": st.session_state.messages,
                    "structured": st.session_state.structured,
                    "triage": triage,
                }
                json_path = save_fn(out)
                # save pdf using helper if available
                pdf_path = None
                try:
                    pdf_helper = pipeline.get("save_report_pdf", None)
                    if callable(pdf_helper):
                        pdf_path = pdf_helper(out)
                    else:
                        from langchain_pipeline import save_report_pdf

                        pdf_path = save_report_pdf(out)
                except Exception:
                    pdf_path = None
                st.success(
                    f"Report saved to {json_path}"
                    + (f", PDF: {pdf_path}" if pdf_path else "")
                )
            except Exception as e:
                st.error(f"Triage/save failed: {e}")

    # render chat
    for role, text in st.session_state.messages:
        if role == "patient":
            st.markdown(f"**You:** {text}")
        else:
            st.markdown(f"**NIVA:** {text}")

with col_side:
    st.subheader("Status & Controls")
    if st.session_state.pipeline_loaded:
        st.success("Pipeline loaded")
    else:
        if st.session_state.init_error:
            st.error("Pipeline initialization error")
            st.code(st.session_state.init_error)
        else:
            st.info("Pipeline not loaded. It will be imported on first interaction.")

    st.markdown("---")
    st.subheader("Report & Triage")
    if st.session_state.finished and st.session_state.structured:
        st.markdown("**Structured report**")
        st.json(st.session_state.structured)
        st.info("Report & PDF saved in outputs/ when generated.")
    else:
        st.info(
            "Click Send to chat. Click Finish & Generate Report to extract structured JSON and PDF."
        )

st.markdown("---")
st.caption("Demo: conversational RAG + Ollama (not a medical device).")
