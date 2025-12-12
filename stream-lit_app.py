# stream-lit_app.py — FINAL UI FOR NIVA (updated import)
import streamlit as st
import os
import time
from dotenv import load_dotenv

# Import from the pipeline file we just added
from langchain_pipeline import (
    generate_conversational_reply,
    extract_structured_from_conversation,
    triage_report,
    save_report,
    save_report_pdf,
    assign_doctor,
)

# load local .env if present (safe)
load_dotenv()

st.set_page_config(page_title="NIVA Medical Assistant", page_icon="💠", layout="wide")

# ---------------- SESSION INIT ----------------
if "messages" not in st.session_state:
    st.session_state.messages = [
        ("bot", "Hello, I am NIVA, your medical intake assistant. How can I help you today?")
    ]
if "finished" not in st.session_state:
    st.session_state.finished = False
if "structured" not in st.session_state:
    st.session_state.structured = None
if "pdf_path" not in st.session_state:
    st.session_state.pdf_path = None
if "typing" not in st.session_state:
    st.session_state.typing = False

# ---------------- CSS ----------------
st.markdown("""
<style>
body {background:#0F0F0F;}
.bubble {padding:14px 18px; border-radius:12px; margin-bottom:12px; max-width:80%;}
.bot {background:#1E1E1E; border-left:4px solid #4F46E5;}
.user {background:#2D2D2D; margin-left:auto; border-right:4px solid #2563EB;}
div[data-testid="stTextInput"] > div:first-child {background:transparent !important;}
div[data-testid="stTextInput"] input {background:#2A2A2A !important; color:white;}
footer {visibility:hidden;}
.top {background:#111; padding:15px; color:#fff; text-align:center; font-size:23px; font-weight:600; border-bottom:1px solid #222;}
</style>
""", unsafe_allow_html=True)

st.markdown("<div class='top'>💠 NIVA — Medical Intake Assistant</div>", unsafe_allow_html=True)

# ---------------- SIDEBAR ----------------
left, right = st.columns([1.3,5])

with left:
    st.markdown("### Settings")
    temp = st.slider("Temperature",0.0,1.0,0.35)
    kctx = st.slider("Dataset Context (K)",0,10,3)

    st.markdown("---")
    st.markdown("### Report Output")
    if st.session_state.structured:
        st.json(st.session_state.structured)

    st.markdown("---")
    st.markdown("### Reset Conversation")
    if st.button("Start New Conversation"):
        st.session_state.messages = [
            ("bot", "Hello, I am NIVA, your medical intake assistant. How can I help you today?")
        ]
        st.session_state.finished = False
        st.session_state.structured = None
        st.session_state.pdf_path = None
        st.session_state.typing = False
        st.experimental_rerun()

    st.caption("NIVA Healthcare Assistant © 2025")

# ---------------- MAIN CHAT ----------------
with right:

    # chat display
    for role, text in st.session_state.messages:
        cls = "user" if role=="patient" else "bot"
        who = "You" if role=="patient" else "NIVA"
        st.markdown(f"<div class='bubble {cls}'><b>{who}:</b><br>{text}</div>", unsafe_allow_html=True)

    if st.session_state.typing:
        st.markdown("<div class='bubble bot'>NIVA is typing…</div>", unsafe_allow_html=True)

    # ---------------- INPUT ----------------
    st.markdown("### Your Message")

    patient_replies = [t for r, t in st.session_state.messages if r=="patient"]
    required_complete = len(patient_replies) >= 5

    if not st.session_state.finished and not required_complete:
        with st.form("chat_form", clear_on_submit=True):
            user_input = st.text_input("", placeholder="Type your message here…")
            c1,c2 = st.columns([6,1])
            with c1:
                submitted = st.form_submit_button("Send")
            with c2:
                finish = st.form_submit_button("Finish")
    else:
        submitted = False
        finish = False
        st.text_input("", value="Conversation finished. Start a new one.", disabled=True)
        st.session_state.finished = True

    # ---------------- PROCESS SEND ----------------
    if submitted:
        if user_input and user_input.strip():
            st.session_state.messages.append(("patient", user_input.strip()))
            st.session_state.typing = True
            st.experimental_rerun()

    if st.session_state.typing:
        last = st.session_state.messages[-1][1]
        # call the pipeline
        bot_msg = generate_conversational_reply(st.session_state.messages, last, k_context=kctx, temperature=temp)
        st.session_state.typing = False
        st.session_state.messages.append(("bot", bot_msg))
        st.experimental_rerun()

    # ---------------- FINISH = GENERATE REPORT ----------------
    if finish and not st.session_state.finished:
        st.session_state.finished = True
        st.experimental_rerun()

    # AFTER FINISHED → generate report ONCE
    if st.session_state.finished and st.session_state.structured is None:
        conv_text = "\n".join(f"{'Patient' if r=='patient' else 'NIVA'}: {t}" for r,t in st.session_state.messages)
        st.session_state.structured = extract_structured_from_conversation(conv_text)
        tri = triage_report(st.session_state.structured)
        doc = assign_doctor(tri)

        out = {
            "conversation": st.session_state.messages,
            "structured": st.session_state.structured,
            "triage": tri,
            "doctor": doc,
        }

        json_path = save_report(out)
        pdf_path = save_report_pdf(out)
        st.session_state.pdf_path = pdf_path

        st.success("Report generated! Download below.")

    if st.session_state.pdf_path:
        try:
            with open(st.session_state.pdf_path, "rb") as f:
                st.download_button("⬇ Download PDF",
                                   f,
                                   file_name=os.path.basename(st.session_state.pdf_path),
                                   mime="application/pdf")
        except FileNotFoundError:
            st.error("PDF not found (file was removed).")
