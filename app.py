"""
Streamlit Web Frontend for Fake Job Detection System
Palette Constraints:
  - Warm Peach: #FFF1E7 (Primary Background)
  - Light Blue: #B5D2E6 (Secondary Sections)
  - Deep Blue: #326080 (Headings, Buttons, Links, Borders)
  - Warm Brown: #805232 (Secondary Accents, Highlights)
"""

import os
import warnings

# Suppress internal developer warnings from transformers and PyTorch
os.environ["TRANSFORMERS_VERBOSITY"] = "error"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
warnings.filterwarnings("ignore", category=UserWarning)

try:
    import transformers
    transformers.logging.set_verbosity_error()
except (ImportError, AttributeError):
    pass

import html

import streamlit as st
from pipeline_bridge import handle_ui_submission, handle_phone_verification
from ocr_engine import init_ocr_reader
from search_engine import EvidenceRetriever
from explainability import explain_phone_verification, summarize_with_llm

# 1. Page Configuration
st.set_page_config(
    page_title="Fake Job Detection System",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# 2. Strict Palette Custom Styles
CUSTOM_CSS = """
<style>
/* Main App Canvas */
.stApp {
    background-color: #FFF1E7;
    color: #326080;
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
}

/* Headings */
h1, h2, h3, h4, h5, h6 {
    color: #326080 !important;
    font-weight: 700;
    letter-spacing: -0.02em;
}

p, label {
    color: #326080;
}

/* Header Container */
.app-header {
    background-color: #B5D2E6;
    border: 1.5px solid #326080;
    border-radius: 4px;
    padding: 24px;
    margin-bottom: 24px;
}
.app-header h1 {
    margin: 0 0 8px 0;
    font-size: 28px;
}
.app-header p {
    margin: 0;
    font-size: 15px;
    color: #326080;
}

/* Card Containers */
.ui-card {
    background-color: #B5D2E6;
    border: 1.5px solid #326080;
    border-radius: 4px;
    padding: 20px;
    margin-bottom: 20px;
}
.ui-card-title {
    font-size: 18px;
    font-weight: 700;
    color: #326080;
    margin-bottom: 12px;
    border-bottom: 1.5px solid #326080;
    padding-bottom: 8px;
}

/* Form Inputs */
.stTextArea textarea {
    background-color: #FFF1E7 !important;
    color: #326080 !important;
    border: 1.5px solid #326080 !important;
    border-radius: 4px !important;
    font-size: 14px !important;
}
.stTextArea textarea:focus {
    border-color: #805232 !important;
    box-shadow: none !important;
}

[data-testid="stFileUploader"] {
    background-color: #FFF1E7 !important;
    border: 1.5px dashed #326080 !important;
    border-radius: 4px !important;
    padding: 12px !important;
}

/* All Buttons Base */
.stButton > button {
    border-radius: 4px !important;
    font-weight: 700 !important;
    font-size: 16px !important;
    padding: 12px 28px !important;
    width: 100% !important;
    transition: background-color 0.2s ease, border-color 0.2s ease, color 0.2s ease;
}

/* Primary Button (Analyze Job Offer): Deep Blue background, Warm Peach text */
.stButton > button[kind="primary"],
.stButton > button:not([kind="secondary"]) {
    background-color: #326080 !important;
    border: 1.5px solid #326080 !important;
    color: #FFF1E7 !important;
}
.stButton > button[kind="primary"] *,
.stButton > button:not([kind="secondary"]) * {
    color: #FFF1E7 !important;
    fill: #FFF1E7 !important;
}
.stButton > button[kind="primary"]:hover,
.stButton > button:not([kind="secondary"]):hover {
    background-color: #805232 !important;
    border-color: #805232 !important;
    color: #FFF1E7 !important;
}
.stButton > button[kind="primary"]:hover *,
.stButton > button:not([kind="secondary"]):hover * {
    color: #FFF1E7 !important;
    fill: #FFF1E7 !important;
}

/* Secondary Button (Reset Form): Light Blue background, Deep Blue text */
.stButton > button[kind="secondary"] {
    background-color: #B5D2E6 !important;
    border: 1.5px solid #326080 !important;
    color: #326080 !important;
}
.stButton > button[kind="secondary"] * {
    color: #326080 !important;
    fill: #326080 !important;
}
.stButton > button[kind="secondary"]:hover {
    background-color: #805232 !important;
    border-color: #805232 !important;
    color: #FFF1E7 !important;
}
.stButton > button[kind="secondary"]:hover * {
    color: #FFF1E7 !important;
    fill: #FFF1E7 !important;
}

/* Metrics & Score Box */
.score-box {
    background-color: #FFF1E7;
    border: 2px solid #326080;
    border-radius: 4px;
    padding: 16px;
    text-align: center;
}
.score-value {
    font-size: 38px;
    font-weight: 800;
    color: #326080;
    line-height: 1.1;
}
.score-label {
    font-size: 13px;
    font-weight: 700;
    color: #805232;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    margin-top: 4px;
}

/* Risk Badges (Palette Compliant) */
.badge-low-risk {
    background-color: #326080;
    color: #FFF1E7;
    border: 1.5px solid #326080;
    border-radius: 4px;
    padding: 12px 18px;
    text-align: center;
    font-size: 20px;
    font-weight: 800;
    letter-spacing: 0.04em;
}
.badge-med-risk {
    background-color: #805232;
    color: #FFF1E7;
    border: 1.5px solid #805232;
    border-radius: 4px;
    padding: 12px 18px;
    text-align: center;
    font-size: 20px;
    font-weight: 800;
    letter-spacing: 0.04em;
}
.badge-high-risk {
    background-color: #326080;
    color: #FFF1E7;
    border: 3px solid #805232;
    border-radius: 4px;
    padding: 12px 18px;
    text-align: center;
    font-size: 20px;
    font-weight: 800;
    letter-spacing: 0.04em;
}

/* Flags Box */
.flag-item {
    background-color: #FFF1E7;
    border-left: 4px solid #805232;
    border-top: 1px solid #B5D2E6;
    border-right: 1px solid #B5D2E6;
    border-bottom: 1px solid #B5D2E6;
    border-radius: 2px;
    padding: 10px 14px;
    margin-bottom: 8px;
    color: #326080;
    font-size: 14px;
    font-weight: 500;
}
.flag-empty {
    background-color: #FFF1E7;
    border: 1px solid #326080;
    border-radius: 2px;
    padding: 10px 14px;
    color: #326080;
    font-size: 14px;
}

/* Table / Entity Details */
.entity-table {
    width: 100%;
    border-collapse: collapse;
    background-color: #FFF1E7;
    border: 1px solid #326080;
    border-radius: 4px;
    overflow: hidden;
}
.entity-table th {
    background-color: #B5D2E6;
    color: #326080;
    font-weight: 700;
    padding: 10px 14px;
    text-align: left;
    border-bottom: 1.5px solid #326080;
    width: 28%;
}
.entity-table td {
    padding: 10px 14px;
    color: #326080;
    border-bottom: 1px solid #B5D2E6;
    font-size: 14px;
}

/* Evidence Cards */
.evidence-card {
    background-color: #FFF1E7;
    border: 1px solid #326080;
    border-radius: 4px;
    padding: 14px;
    margin-bottom: 12px;
}
.evidence-meta {
    font-size: 12px;
    color: #805232;
    font-weight: 600;
    margin-bottom: 6px;
    text-transform: uppercase;
}
.evidence-content {
    font-size: 13px;
    color: #326080;
    line-height: 1.5;
}

/* Streamlit Expander styling */
div[data-testid="stExpander"] {
    border: 1.5px solid #326080 !important;
    border-radius: 4px !important;
    background-color: #B5D2E6 !important;
    margin-bottom: 12px !important;
}
div[data-testid="stExpander"] details summary span {
    color: #326080 !important;
    font-weight: 700 !important;
}
div[data-testid="stExpander"] [data-testid="stExpanderDetails"] {
    background-color: #FFF1E7 !important;
    border-top: 1px solid #326080 !important;
    padding: 16px !important;
}

/* Error Box */
.error-box {
    background-color: #B5D2E6;
    border-left: 5px solid #805232;
    border-top: 1.5px solid #326080;
    border-right: 1.5px solid #326080;
    border-bottom: 1.5px solid #326080;
    border-radius: 4px;
    padding: 14px 18px;
    color: #326080;
    font-size: 14px;
    margin-bottom: 20px;
}
.error-box-title {
    font-weight: 800;
    color: #805232;
    margin-bottom: 4px;
}
</style>
"""

st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


# 3. Cached Resource Initializers (Executed once per process)
@st.cache_resource(show_spinner=False)
def load_ocr_reader():
    return init_ocr_reader(lang_list=["en"], gpu=False)


@st.cache_resource(show_spinner=False)
def load_retriever():
    return EvidenceRetriever(index_folder="faiss_index")


# 4. Session State Setup
if "analysis_result" not in st.session_state:
    st.session_state.analysis_result = None

if "error_message" not in st.session_state:
    st.session_state.error_message = None

if "voice_result" not in st.session_state:
    st.session_state.voice_result = None

if "voice_error" not in st.session_state:
    st.session_state.voice_error = None

if "auto_call_pending" not in st.session_state:
    st.session_state.auto_call_pending = False

# offer_id -> LLM summary (None = Ollama unavailable or summary rejected)
if "llm_summaries" not in st.session_state:
    st.session_state.llm_summaries = {}


# 5. Header Component
st.markdown(
    """
    <div class="app-header">
        <h1>🛡️ Fake Job & Internship Detection System</h1>
        <p>Upload a job offer letter image, paste the job offer text, or provide both for hybrid multi-stage verification.</p>
    </div>
    """,
    unsafe_allow_html=True,
)


# 6. Input Section
st.markdown('<div class="ui-card"><div class="ui-card-title">1. Submit Job Offer Details</div>', unsafe_allow_html=True)

col_upload, col_text = st.columns([1, 1], gap="medium")

with col_upload:
    st.markdown("<p style='font-weight: 600; margin-bottom: 6px;'>Offer Letter Document / Image</p>", unsafe_allow_html=True)
    uploaded_image = st.file_uploader(
        label="Upload Offer Letter Image",
        type=["png", "jpg", "jpeg"],
        label_visibility="collapsed",
        help="Upload an image screenshot or scanned PDF/letterhead (.png, .jpg, .jpeg)",
    )
    if uploaded_image is not None:
        st.image(uploaded_image, caption=f"Uploaded: {uploaded_image.name}", use_container_width=True)

with col_text:
    st.markdown("<p style='font-weight: 600; margin-bottom: 6px;'>Job Description / Email / Text</p>", unsafe_allow_html=True)
    user_text_input = st.text_area(
        label="Offer Text",
        height=260,
        placeholder="Paste job offer details, recruiter email message, compensation claim, or company notes here...",
        label_visibility="collapsed",
    )

st.markdown("</div>", unsafe_allow_html=True)

# 7. Action Button
btn_col1, btn_col2 = st.columns([3, 1], gap="medium")
with btn_col1:
    analyze_clicked = st.button("Analyze Job Offer", type="primary", use_container_width=True)
with btn_col2:
    reset_clicked = st.button("Reset Form", type="secondary", use_container_width=True)

if reset_clicked:
    st.session_state.analysis_result = None
    st.session_state.error_message = None
    st.session_state.voice_result = None
    st.session_state.voice_error = None
    st.session_state.auto_call_pending = False
    st.session_state.llm_summaries = {}
    st.rerun()


# 8. Pipeline Execution Handler
if analyze_clicked:
    st.session_state.error_message = None
    st.session_state.voice_result = None
    st.session_state.voice_error = None
    
    # Input validation prior to loading spinners
    has_text = bool(user_text_input and user_text_input.strip())
    has_img = uploaded_image is not None

    if not has_text and not has_img:
        st.session_state.error_message = (
            "Please provide an offer letter image, job description text, or both to analyze."
        )
        st.session_state.analysis_result = None
    else:
        with st.spinner("Analyzing offer through OCR, entity extraction, and credibility engine..."):
            try:
                reader = load_ocr_reader()
                retriever = load_retriever()

                result = handle_ui_submission(
                    image_file=uploaded_image,
                    user_text=user_text_input,
                    reader=reader,
                    retriever=retriever,
                )
                st.session_state.analysis_result = result
                st.session_state.error_message = None
                # High-risk offers go straight to phone verification
                st.session_state.auto_call_pending = bool(result.get("auto_call"))
            except Exception as err:
                st.session_state.analysis_result = None
                st.session_state.error_message = str(err)
                st.session_state.auto_call_pending = False


# 9. Error Display
if st.session_state.error_message:
    st.markdown(
        f"""
        <div class="error-box">
            <div class="error-box-title">Verification Notice</div>
            <div>{st.session_state.error_message}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# 10. Results Display
res = st.session_state.analysis_result

if res:
    st.markdown('<div class="ui-card"><div class="ui-card-title">2. Credibility Evaluation Results</div>', unsafe_allow_html=True)

    # Top Row: Trust Score + Risk Level Badge
    m_col1, m_col2 = st.columns([1, 1], gap="medium")

    with m_col1:
        st.markdown(
            f"""
            <div class="score-box">
                <div class="score-value">{res['trust_score']:.0f}<span style="font-size: 20px; font-weight: 500;">/100</span></div>
                <div class="score-label">Calibrated Confidence: Real</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with m_col2:
        risk_lvl = res.get("risk_level", "UNKNOWN").upper()
        if "LOW" in risk_lvl:
            badge_class = "badge-low-risk"
        elif "MED" in risk_lvl:
            badge_class = "badge-med-risk"
        else:
            badge_class = "badge-high-risk"

        st.markdown(
            f"""
            <div style="height: 100%; display: flex; flex-direction: column; justify-content: center;">
                <div class="{badge_class}">
                    RISK LEVEL: {risk_lvl}
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("<div style='height: 16px;'></div>", unsafe_allow_html=True)

    # Second Row: M-RAV Explanation & Raised Warning Flags
    exp_col, flags_col = st.columns([1, 1], gap="medium")

    with exp_col:
        st.markdown("<p style='font-weight: 700; margin-bottom: 8px;'>Credibility Assessment Explanation</p>", unsafe_allow_html=True)
        st.markdown(
            f"""
            <div style="background-color: #FFF1E7; border: 1px solid #326080; border-radius: 4px; padding: 14px; min-height: 110px;">
                <p style="margin: 0; line-height: 1.5; font-size: 14px;">{html.escape(res['explanation'])}</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with flags_col:
        st.markdown("<p style='font-weight: 700; margin-bottom: 8px;'>Warning Flags Raised</p>", unsafe_allow_html=True)
        flags = res.get("flags", [])
        if flags:
            flags_html = "".join([f'<div class="flag-item">{html.escape(f)}</div>' for f in flags])
            st.markdown(flags_html, unsafe_allow_html=True)
        else:
            st.markdown(
                '<div class="flag-empty">No warning flags raised. Listing appears consistent.</div>',
                unsafe_allow_html=True,
            )

    st.markdown("<div style='height: 16px;'></div>", unsafe_allow_html=True)

    # Objective 5: how each evidence source moved the verdict
    vexp = res.get("verdict_explanation")
    if vexp:
        st.markdown("<p style='font-weight: 700; margin-bottom: 8px;'>Why This Verdict</p>", unsafe_allow_html=True)
        st.markdown(
            "<p style='font-size: 13px; margin-bottom: 8px;'>Effect = how much the confidence that the offer is real "
            "changes when that source is left out (before any override). Positive points to a real offer, negative to a scam.</p>",
            unsafe_allow_html=True,
        )
        rows = "".join(
            f"<tr><th>{html.escape(f['label'])}</th>"
            f"<td style='color: {'#326080' if f['effect'] >= 0 else '#A33A2B'}; font-weight: 700;'>{f['effect'] * 100:+.0f} pts</td>"
            f"<td>{html.escape(f['detail'])}</td></tr>"
            for f in vexp["factors"]
        ) or "<tr><td>No evidence sources were combined.</td></tr>"
        st.markdown(f"<table class='entity-table'>{rows}</table>", unsafe_allow_html=True)

        comb = vexp["combination"]
        with st.expander("How the Evidence Was Combined", expanded=False):
            st.markdown(
                f"""
                <table class="entity-table">
                    <tr><th>Combined belief: Real</th><td>{comb['real']:.2f}</td></tr>
                    <tr><th>Combined belief: Fake</th><td>{comb['fake']:.2f}</td></tr>
                    <tr><th>Uncertain</th><td>{comb['uncertain']:.2f}</td></tr>
                    <tr><th>Conflict between sources (K)</th><td>{comb['conflict']:.2f}{' (high, confidence pulled towards 0.50)' if comb['high_conflict'] else ''}</td></tr>
                    <tr><th>Calibrated confidence (Platt)</th><td>{comb['calibrated_confidence']:.2f}</td></tr>
                    <tr><th>Override</th><td>{html.escape(vexp['override'] or 'None')}</td></tr>
                    <tr><th>Final confidence</th><td>{vexp['confidence']:.2f}, {html.escape(comb['band'])}</td></tr>
                </table>
                """,
                unsafe_allow_html=True,
            )

        offer_key = res.get("offer_id")
        if offer_key not in st.session_state.llm_summaries:
            if st.button("Explain in Plain Language (local LLM)", use_container_width=True):
                with st.spinner("Writing a plain-language explanation..."):
                    st.session_state.llm_summaries[offer_key] = summarize_with_llm(vexp)
                st.rerun()
        else:
            llm_text = st.session_state.llm_summaries[offer_key]
            if llm_text:
                body = html.escape(llm_text)
                note = "Written by the local LLM from the facts above; numbers checked against them."
            else:
                body = html.escape(vexp["summary"])
                note = "The local LLM was unavailable or its answer did not match the facts, so the standard summary is shown."
            st.markdown(
                f"""
                <div style="background-color: #FFF1E7; border: 1px solid #326080; border-radius: 4px; padding: 14px;">
                    <p style="margin: 0 0 6px 0; line-height: 1.5; font-size: 14px;">{body}</p>
                    <p style="margin: 0; font-size: 12px; color: #805232;">{note}</p>
                </div>
                """,
                unsafe_allow_html=True,
            )

        st.markdown("<div style='height: 16px;'></div>", unsafe_allow_html=True)

    # Extracted Entities Section
    st.markdown("<p style='font-weight: 700; margin-bottom: 8px;'>Extracted Job Offer Entities</p>", unsafe_allow_html=True)
    ent = res.get("extracted_entities", {})
    
    st.markdown(
        f"""
        <table class="entity-table">
            <tr><th>Company Name</th><td>{ent.get('company_name') or 'Not Specified'}</td></tr>
            <tr><th>Job Role</th><td>{ent.get('job_role') or 'Not Specified'}</td></tr>
            <tr><th>Recruiter Name</th><td>{ent.get('recruiter_name') or 'Not Specified'}</td></tr>
            <tr><th>Recruiter Email</th><td>{ent.get('recruiter_email') or 'Not Specified'}</td></tr>
            <tr><th>Website / Domain</th><td>{ent.get('domain_or_website') or 'Not Specified'}</td></tr>
            <tr><th>Location</th><td>{ent.get('location') or 'Not Specified'}</td></tr>
        </table>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("</div>", unsafe_allow_html=True)

    # 11. Phone Verification (Objective 3 + DST)
    st.markdown('<div class="ui-card"><div class="ui-card-title">3. Phone Verification</div>', unsafe_allow_html=True)

    if res.get("auto_call"):
        st.markdown(
            f"<p>Risk level is <b>{html.escape(res.get('risk_level', ''))}</b>. A verification call to the company is started automatically.</p>",
            unsafe_allow_html=True,
        )
    elif res.get("voice_recommended"):
        st.markdown(
            f"<p>Risk level is <b>{html.escape(res.get('risk_level', ''))}</b>. A verification call to the company is recommended.</p>",
            unsafe_allow_html=True,
        )
    else:
        st.markdown("<p>Risk level is low. A verification call is optional.</p>", unsafe_allow_html=True)

    st.markdown(
        "<p style='font-size: 13px;'>The call runs over WebRTC. Before starting, run "
        "<code>python whisper_Test/call_bridge.py</code> and <code>node webrtc-call/server/server.js</code>. "
        "The link to answer the call appears here once the questions are ready.</p>",
        unsafe_allow_html=True,
    )

    verify_clicked = st.button("Verify by Phone", type="primary", use_container_width=True)

    # Run once per analysis: either the user clicked, or Obj2 flagged high risk
    start_call = verify_clicked or st.session_state.auto_call_pending
    st.session_state.auto_call_pending = False

    if start_call:
        st.session_state.voice_result = None
        st.session_state.voice_error = None
        call_status = st.status("Generating verification questions — this can take a few minutes...", expanded=True)

        def on_call_progress(text, step):
            # Step 4 starts once the bridge has created the session with its questions
            if step == 4:
                call_status.update(label="Questions ready — waiting for you to answer the call...", state="running")
                call_status.markdown(
                    "Open <a href='http://localhost:3000' target='_blank'>http://localhost:3000</a>, "
                    "connect, and click <b>Start Session</b>.",
                    unsafe_allow_html=True,
                )

        try:
            st.session_state.voice_result = handle_phone_verification(res, on_progress=on_call_progress)
            if st.session_state.voice_result.get("voice_error"):
                call_status.update(label="Verification call failed", state="error", expanded=False)
            else:
                call_status.update(label="Verification call finished", state="complete", expanded=False)
        except Exception as err:
            st.session_state.voice_error = str(err)
            call_status.update(label="Verification call failed", state="error", expanded=False)

    if st.session_state.voice_error:
        st.markdown(
            f"""
            <div class="error-box">
                <div class="error-box-title">Phone Verification Notice</div>
                <div>{html.escape(st.session_state.voice_error)}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    vres = st.session_state.voice_result
    if vres:
        verdict = vres.get("final_verdict") or "UNCERTAIN"
        if "Fake" in verdict:
            verdict_class = "badge-high-risk"
        elif "Real" in verdict:
            verdict_class = "badge-low-risk"
        else:
            verdict_class = "badge-med-risk"

        v_col1, v_col2 = st.columns([1, 1], gap="medium")
        with v_col1:
            st.markdown(
                f"""
                <div class="{verdict_class}">COMBINED VERDICT: {html.escape(verdict.upper())}</div>
                <table class="entity-table" style="margin-top: 12px;">
                    <tr><th>Belief: Real</th><td>{(vres.get('final_belief_real') or 0.0) * 100:.1f}%</td></tr>
                    <tr><th>Belief: Fake</th><td>{(vres.get('final_belief_fake') or 0.0) * 100:.1f}%</td></tr>
                    <tr><th>Uncertain</th><td>{(vres.get('final_belief_uncertain') or 0.0) * 100:.1f}%</td></tr>
                </table>
                """,
                unsafe_allow_html=True,
            )

        with v_col2:
            call = vres.get("voice_verification") or {}
            signals = call.get("voice_signals") or {}
            if signals:
                call_html = f"<p style='margin: 0; line-height: 1.5; font-size: 14px;'>{html.escape(signals.get('response_summary') or 'No summary returned.')}</p>"
            else:
                reason = vres.get("voice_error") or call.get("skip_reason") or call.get("error") or "unknown"
                call_html = (
                    "<p style='margin: 0; line-height: 1.5; font-size: 14px;'>The call produced no usable signals "
                    f"(reason: {html.escape(str(reason))}). The verdict above uses database evidence only.</p>"
                )
            st.markdown("<p style='font-weight: 700; margin-bottom: 8px;'>Call Outcome</p>", unsafe_allow_html=True)
            st.markdown(
                f"""
                <div style="background-color: #FFF1E7; border: 1px solid #326080; border-radius: 4px; padding: 14px; min-height: 110px;">
                    {call_html}
                </div>
                """,
                unsafe_allow_html=True,
            )

        # Objective 5: what the call contributed to the combined verdict
        with st.expander("How the Call Changed the Verdict", expanded=False):
            pexp = explain_phone_verification(vres)
            st.markdown("".join(f"<div class='flag-item'>{html.escape(f)}</div>" for f in pexp["facts"]), unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)

    # 12. Expandable Verification Details
    ev_docs = res.get("retrieved_evidence", [])
    with st.expander(f"Retrieved Evidence from Database ({len(ev_docs)} match{'es' if len(ev_docs) != 1 else ''})", expanded=False):
        if ev_docs:
            for idx, doc in enumerate(ev_docs, 1):
                meta = doc.get("metadata", {})
                src = meta.get("source_type", "unknown")
                is_fraud = meta.get("fraudulent", None)
                fraud_badge = "FRAUDULENT RECORD" if is_fraud == 1 else ("LEGITIMATE RECORD" if is_fraud == 0 else "")
                dist = doc.get("distance", 0.0)
                title = meta.get("title") or meta.get("company_name") or "Record"

                st.markdown(
                    f"""
                    <div class="evidence-card">
                        <div class="evidence-meta">
                            [#{idx}] {src} | Dist: {dist:.4f} {f'| {fraud_badge}' if fraud_badge else ''}
                        </div>
                        <div style="font-weight: 700; color: #326080; margin-bottom: 4px;">{title}</div>
                        <div class="evidence-content">{doc.get('page_content', '')}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
        else:
            st.markdown("<div style='color: #805232;'>No matching evidence records returned from FAISS vector search.</div>", unsafe_allow_html=True)

    ocr_prev = res.get("ocr_preview", "")
    if ocr_prev:
        with st.expander("Extracted OCR Text Preview", expanded=False):
            st.markdown(
                f"""
                <div style="background-color: #FFF1E7; border: 1px solid #326080; border-radius: 4px; padding: 12px; font-family: monospace; font-size: 13px; line-height: 1.4; color: #326080;">
                    {ocr_prev}...
                </div>
                """,
                unsafe_allow_html=True,
            )
