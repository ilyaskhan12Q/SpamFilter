from __future__ import annotations

import html
import json
from pathlib import Path

import joblib
import streamlit as st
import streamlit.components.v1 as components

from train_model import MODEL_PATH, VECTORIZER_PATH, train_and_save


ROOT = Path(__file__).resolve().parent
MAX_MESSAGE_LENGTH = 10_000


st.set_page_config(
    page_title="SpamFilter Pro",
    page_icon="[]",
    layout="centered",
    initial_sidebar_state="collapsed",
)


CSS = """
<style>
:root {
  --bg: #0d1117;
  --panel: #161b22;
  --panel-muted: #21262d;
  --border: #30363d;
  --border-soft: #21262d;
  --text: #c9d1d9;
  --muted: #8b949e;
  --green: #3fb950;
  --green-soft: rgba(63, 185, 80, 0.12);
  --red: #f85149;
  --red-soft: rgba(248, 81, 73, 0.12);
  --blue: #58a6ff;
  --blue-soft: rgba(88, 166, 255, 0.12);
  --yellow: #d29922;
  --yellow-soft: rgba(210, 153, 34, 0.12);
}

html, body, [data-testid="stAppViewContainer"] {
  background: var(--bg);
  color: var(--text);
  font-family: ui-monospace, SFMono-Regular, "SF Mono", Menlo, Consolas, monospace;
}

[data-testid="stHeader"], [data-testid="stToolbar"], [data-testid="stDecoration"] {
  display: none;
}

.block-container {
  max-width: 880px;
  padding-top: 46px;
  padding-bottom: 40px;
}

.sf-shell {
  background: var(--panel);
  border: 1px solid var(--border);
  border-radius: 6px;
  overflow: hidden;
}

.sf-topbar {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 16px;
  min-height: 44px;
  padding: 10px 14px;
  background: #0d1117;
  border-bottom: 1px solid var(--border);
}

.sf-title {
  margin: 0;
  color: var(--text);
  font-size: 15px;
  font-weight: 700;
}

.sf-subtitle {
  margin: 3px 0 0;
  color: var(--muted);
  font-size: 12px;
}

.sf-file {
  color: var(--muted);
  border: 1px solid var(--border);
  border-radius: 6px;
  padding: 4px 8px;
  font-size: 12px;
  white-space: nowrap;
}

.sf-section {
  padding: 16px;
}

.sf-counter {
  color: var(--muted);
  font-size: 12px;
  margin-top: -4px;
  margin-bottom: 12px;
}

.sf-status {
  display: grid;
  grid-template-columns: minmax(0, 1fr) auto;
  gap: 12px;
  align-items: center;
  background: #0d1117;
  border: 1px solid var(--border);
  border-left: 4px solid var(--blue);
  border-radius: 6px;
  padding: 12px 14px;
  margin-bottom: 14px;
}

.sf-status.ready {
  border-left-color: var(--blue);
}

.sf-status.waiting {
  border-left-color: var(--yellow);
}

.sf-status.ham {
  border-left-color: var(--green);
}

.sf-status.spam {
  border-left-color: var(--red);
}

.sf-status-main {
  min-width: 0;
}

.sf-status-kicker {
  color: var(--muted);
  font-size: 11px;
  font-weight: 700;
  letter-spacing: 0;
  text-transform: uppercase;
}

.sf-status-title {
  color: var(--text);
  font-size: 24px;
  font-weight: 800;
  line-height: 1.25;
  margin-top: 4px;
}

.sf-status-detail {
  color: var(--muted);
  font-size: 12px;
  line-height: 1.45;
  margin-top: 5px;
}

.sf-status-badge {
  justify-self: end;
  border: 1px solid var(--blue);
  border-radius: 999px;
  color: var(--blue);
  background: var(--blue-soft);
  font-size: 12px;
  font-weight: 800;
  line-height: 1;
  padding: 8px 10px;
  white-space: nowrap;
}

.sf-status-badge.waiting {
  border-color: var(--yellow);
  color: var(--yellow);
  background: var(--yellow-soft);
}

.sf-status-badge.ham {
  border-color: var(--green);
  color: var(--green);
  background: var(--green-soft);
}

.sf-status-badge.spam {
  border-color: var(--red);
  color: var(--red);
  background: var(--red-soft);
}

.sf-meter {
  background: var(--panel-muted);
  border: 1px solid var(--border-soft);
  border-radius: 999px;
  height: 8px;
  margin-top: 10px;
  overflow: hidden;
}

.sf-meter-fill {
  background: var(--green);
  height: 100%;
}

.sf-meter-fill.spam {
  background: var(--red);
}

.sf-result {
  background: #0d1117;
  border: 1px solid var(--border);
  border-left: 4px solid var(--green);
  border-radius: 6px;
  padding: 14px;
  margin-top: 14px;
}

.sf-result.spam {
  border-left-color: var(--red);
}

.sf-label {
  display: inline-flex;
  align-items: center;
  border: 1px solid currentColor;
  border-radius: 999px;
  padding: 4px 10px;
  font-size: 14px;
  font-weight: 700;
}

.sf-label.ham {
  color: var(--green);
}

.sf-label.spam {
  color: var(--red);
}

.sf-confidence {
  color: var(--text);
  font-size: 14px;
  margin: 10px 0 0;
}

.sf-preview {
  color: var(--muted);
  font-size: 13px;
  line-height: 1.55;
  margin: 10px 0 0;
  word-break: break-word;
}

.sf-alert {
  border: 1px solid rgba(248, 81, 73, 0.45);
  color: #ffa198;
  background: rgba(248, 81, 73, 0.08);
  border-radius: 6px;
  padding: 10px 12px;
  font-size: 13px;
}

div[data-testid="stTextArea"] textarea {
  min-height: 210px;
  background: #0d1117;
  color: var(--text);
  caret-color: var(--green);
  border: 1px solid var(--border);
  border-radius: 6px;
  font-family: ui-monospace, SFMono-Regular, "SF Mono", Menlo, Consolas, monospace;
  font-size: 14px;
  line-height: 1.6;
}

div[data-testid="stTextArea"] textarea:focus {
  border-color: var(--blue);
  box-shadow: 0 0 0 1px var(--blue);
}

div[data-testid="stTextArea"] label {
  color: var(--muted);
  font-size: 13px;
}

.stButton > button {
  width: 100%;
  min-height: 42px;
  background: var(--panel-muted);
  color: var(--text);
  border: 1px solid var(--border);
  border-radius: 6px;
  font-family: ui-monospace, SFMono-Regular, "SF Mono", Menlo, Consolas, monospace;
  font-weight: 700;
}

.stButton > button:hover {
  background: var(--border);
  color: var(--text);
  border-color: #8b949e;
}

.stButton > button:focus {
  box-shadow: 0 0 0 2px rgba(88, 166, 255, 0.35);
}

#MainMenu, footer {
  visibility: hidden;
}

@media (max-width: 640px) {
  .block-container {
    padding: 20px 12px 28px;
  }

  .sf-topbar {
    align-items: flex-start;
    flex-direction: column;
  }

  .sf-file {
    white-space: normal;
  }

  .sf-status {
    grid-template-columns: 1fr;
  }

  .sf-status-badge {
    justify-self: start;
  }
}
</style>
"""


COPY_COMPONENT = """
<!doctype html>
<html>
<head>
<style>
  :root {
    --panel: #161b22;
    --panel-muted: #21262d;
    --border: #30363d;
    --text: #c9d1d9;
    --green: #3fb950;
  }

  body {
    margin: 0;
    background: transparent;
    font-family: ui-monospace, SFMono-Regular, "SF Mono", Menlo, Consolas, monospace;
  }

  .copy-row {
    display: flex;
    align-items: center;
    gap: 10px;
    min-height: 36px;
  }

  button {
    display: inline-flex;
    align-items: center;
    gap: 7px;
    height: 34px;
    padding: 0 12px;
    background: var(--panel-muted);
    color: var(--text);
    border: 1px solid var(--border);
    border-radius: 6px;
    cursor: pointer;
    font: 700 13px ui-monospace, SFMono-Regular, "SF Mono", Menlo, Consolas, monospace;
  }

  button:hover {
    background: var(--border);
  }

  .icon {
    width: 14px;
    height: 14px;
  }

  .toast {
    opacity: 0;
    color: var(--green);
    font-size: 13px;
  }

  .toast.show {
    animation: copied 1.5s ease-in-out forwards;
  }

  @keyframes copied {
    0% { opacity: 0; transform: translateY(2px); }
    15% { opacity: 1; transform: translateY(0); }
    80% { opacity: 1; transform: translateY(0); }
    100% { opacity: 0; transform: translateY(-2px); }
  }
</style>
</head>
<body>
  <div class="copy-row">
    <button id="copyButton" type="button" aria-label="Copy classification result">
      <svg class="icon" viewBox="0 0 16 16" fill="currentColor" aria-hidden="true">
        <path d="M0 6.75C0 5.784.784 5 1.75 5h1.5a.75.75 0 0 1 0 1.5h-1.5a.25.25 0 0 0-.25.25v7.5c0 .138.112.25.25.25h7.5a.25.25 0 0 0 .25-.25v-1.5a.75.75 0 0 1 1.5 0v1.5A1.75 1.75 0 0 1 9.25 16h-7.5A1.75 1.75 0 0 1 0 14.25Z"></path>
        <path d="M5 1.75C5 .784 5.784 0 6.75 0h7.5C15.216 0 16 .784 16 1.75v7.5A1.75 1.75 0 0 1 14.25 11h-7.5A1.75 1.75 0 0 1 5 9.25Zm1.75-.25a.25.25 0 0 0-.25.25v7.5c0 .138.112.25.25.25h7.5a.25.25 0 0 0 .25-.25v-7.5a.25.25 0 0 0-.25-.25Z"></path>
      </svg>
      Copy
    </button>
    <span id="toast" class="toast" role="status" aria-live="polite">✓ Copied!</span>
  </div>
  <script>
    const resultText = __RESULT_TEXT__;
    const button = document.getElementById("copyButton");
    const toast = document.getElementById("toast");

    async function copyText() {
      try {
        await navigator.clipboard.writeText(resultText);
      } catch (error) {
        const area = document.createElement("textarea");
        area.value = resultText;
        area.style.position = "fixed";
        area.style.opacity = "0";
        document.body.appendChild(area);
        area.focus();
        area.select();
        document.execCommand("copy");
        area.remove();
      }

      toast.classList.remove("show");
      void toast.offsetWidth;
      toast.classList.add("show");
    }

    button.addEventListener("click", copyText);
  </script>
</body>
</html>
"""


@st.cache_resource(show_spinner=False)
def load_classifier():
    if not MODEL_PATH.exists() or not VECTORIZER_PATH.exists():
        train_and_save()
    return joblib.load(MODEL_PATH), joblib.load(VECTORIZER_PATH)


def predict_message(message: str) -> tuple[str, float]:
    model, vectorizer = load_classifier()
    features = vectorizer.transform([message])
    label = str(model.predict(features)[0])
    probabilities = model.predict_proba(features)[0]
    classes = list(model.classes_)
    confidence = float(probabilities[classes.index(label)])
    return label, confidence


def render_copy_button(result_text: str) -> None:
    payload = json.dumps(result_text)
    components.html(COPY_COMPONENT.replace("__RESULT_TEXT__", payload), height=42)


def render_status(
    state: str,
    title: str,
    detail: str,
    badge: str,
    confidence: float | None = None,
) -> None:
    meter = ""
    if confidence is not None:
        meter_class = "spam" if state == "spam" else "ham"
        meter = f"""
          <div class="sf-meter" aria-label="Confidence">
            <div class="sf-meter-fill {meter_class}" style="width: {confidence * 100:.1f}%"></div>
          </div>
        """

    st.markdown(
        f"""
        <div class="sf-status {state}">
          <div class="sf-status-main">
            <div class="sf-status-kicker">Classifier status</div>
            <div class="sf-status-title">{html.escape(title)}</div>
            <div class="sf-status-detail">{html.escape(detail)}</div>
            {meter}
          </div>
          <div class="sf-status-badge {state}">{html.escape(badge)}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_result(label: str, confidence: float, message: str) -> None:
    is_spam = label == "spam"
    tag = "SPAM" if is_spam else "NOT SPAM"
    status_class = "spam" if is_spam else "ham"
    escaped_message = html.escape(message)
    result_text = f"[{tag}] - Confidence: {confidence * 100:.1f}%\nText: \"{message}\""

    st.markdown(
        f"""
        <div class="sf-result {status_class}">
          <span class="sf-label {status_class}">[{tag}]</span>
          <p class="sf-confidence">Confidence: {confidence * 100:.1f}%</p>
          <p class="sf-preview">Text: "{escaped_message}"</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    render_copy_button(result_text)


st.markdown(CSS, unsafe_allow_html=True)
st.markdown(
    """
    <div class="sf-shell">
      <div class="sf-topbar">
        <div>
          <h1 class="sf-title">SpamFilter Pro</h1>
          <p class="sf-subtitle">GitHub-styled ML classifier with Streamlit UI</p>
        </div>
        <div class="sf-file">spam_filter_pro.py</div>
      </div>
      <div class="sf-section">
    """,
    unsafe_allow_html=True,
)

if "last_result" not in st.session_state:
    st.session_state.last_result = None

message = st.text_area(
    "Message",
    placeholder="Paste your message here...",
    label_visibility="collapsed",
    max_chars=MAX_MESSAGE_LENGTH,
)

st.markdown(f'<div class="sf-counter">⏤ {len(message):,} characters</div>', unsafe_allow_html=True)

status_slot = st.empty()
run_clicked = st.button("Run Classification", type="secondary", use_container_width=True)
cleaned = message.strip()
last_result = st.session_state.last_result

if run_clicked:
    if not cleaned:
        st.session_state.last_result = None
        st.markdown(
            '<div class="sf-alert">Paste a message before running classification.</div>',
            unsafe_allow_html=True,
        )
    else:
        with st.spinner("Classifying..."):
            prediction, probability = predict_message(cleaned)
        st.session_state.last_result = {
            "label": prediction,
            "confidence": probability,
            "message": cleaned,
        }
        last_result = st.session_state.last_result
        render_result(prediction, probability, cleaned)
elif last_result and last_result["message"] == cleaned:
    render_result(last_result["label"], last_result["confidence"], last_result["message"])

last_result = st.session_state.last_result
current_result = last_result if last_result and last_result["message"] == cleaned else None

with status_slot:
    if current_result:
        result_label = current_result["label"]
        result_confidence = current_result["confidence"]
        result_status = "spam" if result_label == "spam" else "ham"
        result_title = "SPAM" if result_status == "spam" else "NOT SPAM"
        result_detail = (
            f"Model confidence: {result_confidence * 100:.1f}%. "
            "This classifier is trained on SMS messages."
        )
        result_badge = "SPAM" if result_status == "spam" else "NOT SPAM"
        render_status(result_status, result_title, result_detail, result_badge, result_confidence)
    elif cleaned:
        render_status(
            "ready",
            "Ready to classify",
            "A message is loaded. Run classification to get the spam status.",
            "READY",
        )
    else:
        render_status(
            "waiting",
            "Waiting for message",
            "Paste SMS text to enable a meaningful spam check.",
            "INPUT NEEDED",
        )

st.markdown("</div></div>", unsafe_allow_html=True)
