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
from pipeline_bridge import (
    M4_LIMITATION,
    VERDICT_TO_RISK_LEVEL,
    get_detector,
    handle_phone_verification,
    handle_ui_submission,
)
from ocr_engine import init_ocr_reader
from mravplus_infer import ModelFilesMissing
from xai.render import bars_html, highlight_html, word_scores_from_ig
from mravplus_infer import clean as m4_clean

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
def load_detector():
    return get_detector()


# Optional structured fields: M4 was trained on structured postings, so filling these in
# brings poster input closer to what the model was validated on.
TRI_STATE = {"Unknown": None, "Yes": 1, "No": 0}


def _category_options(detector, column):
    try:
        cats = detector.ohe.categories_[list(detector.ohe.feature_names_in_).index(column)]
        return ["Unknown"] + [c for c in cats if c != "Unknown"]
    except Exception:
        return ["Unknown"]


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

if "deletion_result" not in st.session_state:
    st.session_state.deletion_result = None


# 5. Header Component
st.markdown(
    """
    <div class="app-header">
        <h1>🛡️ Fake Job & Internship Detection System</h1>
        <p>Upload a job offer letter image, paste the job offer text, or provide both. The MRAV+ M4 model
        (TF-IDF classifier + similar known postings + DistilBERT for borderline cases, fused with Dempster-Shafer
        theory) flags the offer and explains why. No LLM is used for the verdict.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

try:
    detector = load_detector()
    detector_error = None
except (ModelFilesMissing, ValueError) as err:
    detector, detector_error = None, str(err)

if detector_error:
    st.markdown(
        f"""
        <div class="error-box">
            <div class="error-box-title">Model files missing</div>
            <div>{html.escape(detector_error)}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.stop()

for w in detector.load_warnings:
    st.markdown(f'<div class="error-box"><div class="error-box-title">Environment warning</div><div>{html.escape(w)}</div></div>',
                unsafe_allow_html=True)


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

with st.expander("Optional: structured posting details (improves reliability)", expanded=False):
    st.markdown(
        "<p style='font-size: 13px;'>M4 was trained on structured job postings. Anything you fill in here is "
        "used as that posting's field; leave a field as Unknown / empty if you do not know it.</p>",
        unsafe_allow_html=True,
    )
    f_col1, f_col2, f_col3 = st.columns(3, gap="medium")
    with f_col1:
        f_title = st.text_input("Job title")
        f_location = st.text_input("Location (e.g. US, NY, New York)")
        f_salary = st.text_input("Salary range")
        f_department = st.text_input("Department")
    with f_col2:
        f_employment = st.selectbox("Employment type", _category_options(detector, "employment_type"))
        f_experience = st.selectbox("Required experience", _category_options(detector, "required_experience"))
        f_education = st.selectbox("Required education", _category_options(detector, "required_education"))
        f_industry = st.selectbox("Industry", _category_options(detector, "industry"))
        f_function = st.selectbox("Function", _category_options(detector, "function"))
    with f_col3:
        f_logo = st.radio("Company logo shown?", list(TRI_STATE), horizontal=True)
        f_remote = st.radio("Remote / telecommuting?", list(TRI_STATE), horizontal=True)
        f_questions = st.radio("Screening questions?", list(TRI_STATE), horizontal=True)
    f_profile = st.text_area("Company profile", height=80)

structured_fields = {
    "title": f_title, "location": f_location, "salary_range": f_salary, "department": f_department,
    "company_profile": f_profile,
    "employment_type": f_employment, "required_experience": f_experience, "required_education": f_education,
    "industry": f_industry, "function": f_function,
    "has_company_logo": TRI_STATE[f_logo], "telecommuting": TRI_STATE[f_remote], "has_questions": TRI_STATE[f_questions],
}
structured_fields = {k: v for k, v in structured_fields.items() if v not in (None, "", "Unknown")}

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
    st.session_state.deletion_result = None
    st.rerun()


# 8. Pipeline Execution Handler
if analyze_clicked:
    st.session_state.error_message = None
    st.session_state.voice_result = None
    st.session_state.voice_error = None
    st.session_state.deletion_result = None

    # Input validation prior to loading spinners
    has_text = bool(user_text_input and user_text_input.strip())
    has_img = uploaded_image is not None

    if not has_text and not has_img:
        st.session_state.error_message = (
            "Please provide an offer letter image, job description text, or both to analyze."
        )
        st.session_state.analysis_result = None
    else:
        with st.spinner("Analyzing offer: OCR, M4 scoring and explanation..."):
            try:
                reader = load_ocr_reader() if has_img else None

                result = handle_ui_submission(
                    image_file=uploaded_image,
                    user_text=user_text_input,
                    reader=reader,
                    detector=detector,
                    fields=structured_fields or None,
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
    st.markdown('<div class="ui-card"><div class="ui-card-title">2. M4 Detection Result</div>', unsafe_allow_html=True)

    # Top Row: Trust Score + Verdict / Risk Level Badge
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
                    {html.escape(res.get('final_verdict', ''))} &middot; {risk_lvl}
                </div>
                <p style="font-size: 13px; margin: 8px 0 0 0;">Calibrated P(fraud) = {res['p_fake']:.3f} &middot;
                zone {html.escape(res.get('zone', ''))}{' &middot; DistilBERT consulted' if res['m4'].get('used_distilbert') else ''}</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown(
        f"<p style='font-size: 12px; color: #805232; margin-top: 10px;'>{html.escape(res.get('limitations', M4_LIMITATION))}</p>",
        unsafe_allow_html=True,
    )

    # Second Row: Explanation summary & Raised Warning Flags
    exp_col, flags_col = st.columns([1, 1], gap="medium")

    with exp_col:
        st.markdown("<p style='font-weight: 700; margin-bottom: 8px;'>Why (plain-language summary, no LLM)</p>", unsafe_allow_html=True)
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

    # Extracted Entities Section
    st.markdown("<p style='font-weight: 700; margin-bottom: 8px;'>Extracted Job Offer Entities</p>", unsafe_allow_html=True)
    ent = res.get("extracted_entities", {})
    
    ent_rows = [("Company Name", "company_name"), ("Job Role", "job_role"), ("Recruiter Name", "recruiter_name"),
                ("Recruiter Email", "recruiter_email"), ("Recruiter Phone", "recruiter_phone"),
                ("Website / Domain", "domain_or_website"), ("Location", "location")]
    ent_html = "".join(
        f"<tr><th>{label}</th><td>{html.escape(str(ent.get(key) or 'Not Specified'))}</td></tr>" for label, key in ent_rows
    )
    st.markdown(f'<table class="entity-table">{ent_html}</table>', unsafe_allow_html=True)
    st.markdown(
        "<p style='font-size: 12px; margin-top: 6px;'>Entities are found with regular expressions and are used only "
        "for the phone call; they do not affect the verdict.</p>",
        unsafe_allow_html=True,
    )

    st.markdown("</div>", unsafe_allow_html=True)

    # 10b. Explanation (Objective 5)
    xai = res.get("xai")
    if xai:
        st.markdown('<div class="ui-card"><div class="ui-card-title">Explanation (Objective 5)</div>', unsafe_allow_html=True)
        for w in xai.get("warnings", []):
            st.markdown(f'<div class="flag-item">{html.escape(w)}</div>', unsafe_allow_html=True)

        m1 = xai["m1"]
        st.markdown(
            "<p style='font-weight: 700; margin-bottom: 4px;'>Words that drove the text classifier</p>"
            "<p style='font-size: 13px; margin-bottom: 8px;'>Exact contributions to the TF-IDF SVM's score: "
            "<span style='background: rgba(163, 58, 43, 0.5); padding: 0 3px;'>red</span> pushes towards FAKE, "
            "<span style='background: rgba(46, 125, 50, 0.5); padding: 0 3px;'>green</span> towards REAL. "
            f"Contributions + intercept reproduce the classifier's decision value "
            f"({m1['decision_function']:+.3f}; check error {m1['identity_error']:.1e}).</p>",
            unsafe_allow_html=True,
        )
        st.markdown(
            "<div style='background-color: #FFF1E7; border: 1px solid #326080; border-radius: 4px; padding: 14px; "
            "font-size: 14px; line-height: 1.8; max-height: 360px; overflow-y: auto;'>"
            f"{highlight_html(res.get('scoring_text', ''), m1['token_scores'])}</div>",
            unsafe_allow_html=True,
        )
        b_col1, b_col2 = st.columns([1, 1], gap="medium")
        with b_col1:
            st.markdown("<p style='font-weight: 700; margin: 12px 0 4px 0;'>Towards FAKE</p>", unsafe_allow_html=True)
            st.markdown(bars_html(m1["top_fraud_terms"]), unsafe_allow_html=True)
        with b_col2:
            st.markdown("<p style='font-weight: 700; margin: 12px 0 4px 0;'>Towards REAL</p>", unsafe_allow_html=True)
            st.markdown(bars_html(m1["top_legit_terms"]), unsafe_allow_html=True)
        if m1["metadata"]:
            st.markdown("<p style='font-weight: 700; margin: 12px 0 4px 0;'>Metadata contributions</p>", unsafe_allow_html=True)
            st.markdown(bars_html(m1["metadata"][:8], label_key="feature"), unsafe_allow_html=True)

        # DST evidence decomposition
        dst = xai["dst"]
        st.markdown("<p style='font-weight: 700; margin: 16px 0 4px 0;'>How the evidence sources were combined (Dempster-Shafer)</p>",
                    unsafe_allow_html=True)
        dst_rows = "".join(
            f"<tr><td>{html.escape(s['label'])}</td>"
            f"<td>{'' if s['says_p_fake'] is None else format(s['says_p_fake'], '.2f')}</td>"
            f"<td>{s['mass']['REAL']:.2f} / {s['mass']['FAKE']:.2f} / {s['mass']['UNCERTAIN']:.2f}</td>"
            f"<td>{'n/a' if s['effect_on_p_fake'] is None else format(s['effect_on_p_fake'] * 100, '+.1f') + ' pts'}</td></tr>"
            for s in dst["sources"]
        )
        st.markdown(
            "<table class='entity-table'><tr><th>Source</th><th>Says P(fraud)</th><th>Mass REAL / FAKE / Uncertain</th>"
            f"<th>Effect on P(fraud)</th></tr>{dst_rows}</table>"
            f"<p style='font-size: 13px; margin-top: 6px;'>Combined mass REAL {dst['combined']['REAL']:.2f}, "
            f"FAKE {dst['combined']['FAKE']:.2f}, residual ignorance {dst['residual_ignorance']:.2f}; "
            f"conflict K = {dst['conflict']:.2f}; calibrated with {dst['calibrator']}. "
            "&ldquo;Effect&rdquo; = P(fraud) with all sources minus P(fraud) with that source left out. "
            f"{html.escape(dst['note'])}</p>",
            unsafe_allow_html=True,
        )

        # DistilBERT (REVIEW zone only)
        bert = xai.get("distilbert")
        if bert:
            g = bert["gate_weights"]
            st.markdown(
                "<p style='font-weight: 700; margin: 16px 0 4px 0;'>DistilBERT (consulted because the case was borderline)</p>"
                f"<p style='font-size: 13px; margin-bottom: 8px;'>Integrated Gradients ({html.escape(bert['method'])}, "
                f"{bert['n_steps']} steps) on the fraud log-odds; the gate relied {g['text']:.0%} on text and "
                f"{g['metadata']:.0%} on metadata. Completeness residual {bert['completeness_residual']:+.3f} "
                f"({bert['completeness_relative']:.1%} of the output change). Attention weights are not used as an explanation."
                f"{' The text was truncated to 512 tokens.' if bert['truncated'] else ''}</p>",
                unsafe_allow_html=True,
            )
            st.markdown(
                "<div style='background-color: #FFF1E7; border: 1px solid #326080; border-radius: 4px; padding: 14px; "
                "font-size: 14px; line-height: 1.8; max-height: 300px; overflow-y: auto;'>"
                f"{highlight_html(res.get('scoring_text', ''), word_scores_from_ig(bert['words']), key=m4_clean)}</div>",
                unsafe_allow_html=True,
            )

        # Faithfulness check, on demand
        st.markdown("<p style='font-weight: 700; margin: 16px 0 4px 0;'>Faithfulness check</p>", unsafe_allow_html=True)
        st.markdown(
            "<p style='font-size: 13px;'>Deletes the top 5 / 10 fraud-leaning words and re-scores the posting with M4, "
            "next to deleting the same number of random words. A faithful explanation should lower P(fraud) more.</p>",
            unsafe_allow_html=True,
        )
        if st.button("Verify explanation", type="secondary"):
            from xai import deletion_test
            with st.spinner("Re-scoring with words removed..."):
                full = detector.predict(res.get("scoring_text", ""), res.get("fields") or None)
                full["xai"] = xai
                st.session_state.deletion_result = deletion_test(
                    detector, res.get("scoring_text", ""), res.get("fields") or None, full)
        dres = st.session_state.deletion_result
        if dres:
            drows = "".join(
                f"<tr><td>{r['k']}</td><td>{html.escape(', '.join(r['deleted']) or '-')}</td>"
                f"<td>{r['p_fake_before']:.3f} &rarr; {r['p_fake_after']:.3f} ({r['delta'] * 100:+.1f} pts)</td>"
                f"<td>{r['p_fake_after_random']:.3f} ({r['delta_random'] * 100:+.1f} pts)</td></tr>"
                for r in dres["results"]
            )
            st.markdown(
                "<table class='entity-table'><tr><th>k</th><th>Deleted words</th><th>P(fraud) after deleting them</th>"
                f"<th>After deleting k random words</th></tr>{drows}</table>",
                unsafe_allow_html=True,
            )

        st.markdown(
            "<p style='font-size: 12px; color: #805232; margin-top: 10px;'>" +
            " ".join(html.escape(c) for c in xai["summary"]["caveats"]) + "</p>",
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
        "<code>python whisper_Test/call_bridge.py</code> and <code>node webrtc-call/server/server.js</code> "
        "(or <code>start_all.sh</code> / <code>start_all.ps1</code> without <code>--no-voice</code>). "
        "The link to answer the call appears here once the questions are ready.</p>"
        "<p style='font-size: 12px; color: #805232;'>The combined post-call verdict (M4 text score + voice evidence, "
        "Dempster-Shafer, Objective 4 bands) is <b>not validated</b>: there is no labelled phone-call dataset. "
        "Only M4's text score is validated.</p>",
        unsafe_allow_html=True,
    )

    verify_clicked = st.button("Verify by Phone", type="primary", use_container_width=True)

    # Run once per analysis: either the user clicked, or M4 flagged high risk
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
        risk = VERDICT_TO_RISK_LEVEL.get(verdict)
        if risk is None:
            # DST-only verdict strings ("Likely Fake" / "Likely Real") when Obj2 was unavailable
            risk = "HIGH RISK" if "Fake" in verdict else "LOW RISK" if "Real" in verdict else "MEDIUM RISK"
        verdict_class = {"HIGH RISK": "badge-high-risk", "LOW RISK": "badge-low-risk"}.get(risk, "badge-med-risk")
        extra_rows = ""
        if vres.get("calibrated_confidence") is not None:
            extra_rows += f"<tr><th>Calibrated P(Real)</th><td>{vres['calibrated_confidence'] * 100:.1f}%</td></tr>"
        if vres.get("override_reason"):
            extra_rows += f"<tr><th>Override</th><td>{html.escape(vres['override_reason'])}</td></tr>"
        if vres.get("validated") is False:
            extra_rows += "<tr><th>Validation</th><td>NOT validated (no labelled call data)</td></tr>"

        v_col1, v_col2 = st.columns([1, 1], gap="medium")
        with v_col1:
            st.markdown(
                f"""
                <div class="{verdict_class}">COMBINED VERDICT (UNVALIDATED): {html.escape(verdict.upper())}</div>
                <table class="entity-table" style="margin-top: 12px;">
                    <tr><th>Belief: Real</th><td>{(vres.get('final_belief_real') or 0.0) * 100:.1f}%</td></tr>
                    <tr><th>Belief: Fake</th><td>{(vres.get('final_belief_fake') or 0.0) * 100:.1f}%</td></tr>
                    <tr><th>Uncertain</th><td>{(vres.get('final_belief_uncertain') or 0.0) * 100:.1f}%</td></tr>
                    {extra_rows}
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
                    f"(reason: {html.escape(str(reason))}). The verdict above uses M4's text score only.</p>"
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

    st.markdown("</div>", unsafe_allow_html=True)

    # 12. Expandable Verification Details
    nbrs = res.get("neighbours", [])
    n_fraud = sum(n["label"] for n in nbrs)
    with st.expander(f"Similar Known Postings ({len(nbrs)} nearest training postings, {n_fraud} fraudulent)", expanded=False):
        st.markdown(
            "<p style='font-size: 13px;'>Nearest EMSCAD training postings by MiniLM cosine similarity. Each one is a "
            "Dempster-Shafer source in M4 (weighted by similarity and label).</p>",
            unsafe_allow_html=True,
        )
        for n in nbrs:
            badge = "FRAUDULENT POSTING" if n["label"] == 1 else "LEGITIMATE POSTING"
            st.markdown(
                f"""
                <div class="evidence-card">
                    <div class="evidence-meta">[#{n['rank']}] training posting {n['index']} | similarity {n['similarity']:.3f} | {badge}</div>
                    <div style="font-weight: 700; color: #326080; margin-bottom: 4px;">{html.escape(n.get('title') or 'Untitled posting')}</div>
                    <div class="evidence-content">{html.escape(n.get('snippet') or '')}{'...' if n.get('snippet') else ''}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    ocr_prev = res.get("ocr_preview", "")
    if ocr_prev:
        with st.expander("Extracted OCR Text Preview", expanded=False):
            st.markdown(
                f"""
                <div style="background-color: #FFF1E7; border: 1px solid #326080; border-radius: 4px; padding: 12px; font-family: monospace; font-size: 13px; line-height: 1.4; color: #326080;">
                    {html.escape(ocr_prev)}...
                </div>
                """,
                unsafe_allow_html=True,
            )
