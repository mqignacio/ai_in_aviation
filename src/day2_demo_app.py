"""
Day 2 Demo — Agentic Prescriptive Maintenance (Gradio chat app)

A messenger-style chat interface for `day2_agentic_maintenance.ipynb`,
designed for instructor-led live demos. The agent's reasoning (tool calls
and observations) streams into the conversation as distinct chat messages
so the audience can see the ReAct loop unfold in real time.

Same pipeline as the notebook: Day 1 RUL model (Gradient Boosting) + a
sample Engine Maintenance Program manual + a LangGraph ReAct agent backed
by a local Ollama model (qwen2.5:3b).

Features:
- Messenger-style chat with user bubbles (right) and assistant bubbles (left)
- Live streaming of tool calls (🔧) and observations (📋) as the agent reasons
- Suggestion chips for quick example queries
- Reset Chat button to clear the session

Run with:
    uv run python src/day2_demo_app.py
"""

import json
import re
import urllib.error
import urllib.request
from pathlib import Path

import gradio as gr
import joblib
import pandas as pd
from langchain.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent

# --- App Configuration ---
APP_TITLE = "Prescriptive Maintenance Agent — Day 2 Demo"
APP_DESCRIPTION = (
    "A messenger-style chat interface for the agentic maintenance system. "
    "Ask about an engine\u2019s health \u2014 the agent queries the RUL model, "
    "consults the maintenance manual, and recommends an action. "
    "All running locally, no cloud dependency."
)

# Color Palette (Option B / Palette 2 — Blue)
CARE_BLUE = "#0338A6"
MIDNIGHT_BLUE = "#04327B"
OCEAN_BLUE = "#22A2E4"
GRAY = "#58595B"

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "assets" / "models" / "rul_model.joblib"
TEST_PATH = BASE_DIR / "assets" / "cmapss" / "test_FD001.txt"
MANUAL_PATH = BASE_DIR / "assets" / "manuals" / "engine_maintenance_program.md"

OLLAMA_BASE_URL = "http://localhost:11434/v1"
OLLAMA_MODEL = "qwen2.5:3b"

COLUMN_NAMES = (
    ["unit_number", "cycle", "op_setting_1", "op_setting_2", "op_setting_3"]
    + [f"sensor_{i}" for i in range(1, 22)]
)

SYSTEM_PROMPT = (
    "You assist a powerplant engineer for commercial turbofan engines. "
    "Your job is to help analyze engine health data and recommend maintenance actions. "
    "Use the available tools to query engine RUL predictions, look up maintenance "
    "procedures from the manual, and assess health status. "
    "Always base your recommendations on data from the tools, not on assumptions. "
    "Structure your final recommendation clearly with: (1) Current engine status, "
    "(2) Relevant maintenance procedures, (3) Specific recommended action, "
    "and (4) Timeframe for execution."
)


# --- Startup: Ollama pre-flight check ---
def check_ollama() -> str:
    """Verify Ollama is running and the model is pulled. Returns a status string."""
    try:
        req = urllib.request.Request(
            f"{OLLAMA_BASE_URL}/models",
            headers={"User-Agent": "ai-in-aviation-demo"},
            method="GET",
        )
        with urllib.request.urlopen(req, timeout=10) as response:
            models = json.loads(response.read().decode())
            model_names = [m["id"] for m in models.get("data", [])]
            model_found = any(OLLAMA_MODEL.split(":")[0] in m for m in model_names)
            if model_found:
                return f"Ollama is running. Model '{OLLAMA_MODEL}' is available."
            return (
                f"Ollama is running, but '{OLLAMA_MODEL}' was not found. "
                f"Run: ollama pull {OLLAMA_MODEL}"
            )
    except urllib.error.URLError:
        return "Ollama is not running. Start it (menu bar icon) or run `ollama serve`."


# --- Load Day 1 RUL model + test data (once, at startup) ---
_loaded = joblib.load(MODEL_PATH)
rul_model = _loaded["model"]
rul_scaler = _loaded["scaler"]
window_size = _loaded["window_size"]
model_type = _loaded["model_type"]
model_rmse = _loaded["rmse"]

test_raw = pd.read_csv(
    TEST_PATH, sep=r"\s+", engine="python", header=None, names=COLUMN_NAMES
)
ENGINE_IDS = sorted(test_raw["unit_number"].unique().tolist())


def extract_features(df: pd.DataFrame, window: int) -> pd.DataFrame:
    """Sliding-window feature extraction (21 sensors x 4 rolling stats = 84 features)."""
    sensor_cols = [c for c in df.columns if c.startswith("sensor_")]
    frames = []
    for unit_id, group in df.groupby("unit_number"):
        group = group.sort_values("cycle").copy()
        rolling = group[sensor_cols].rolling(window=window, min_periods=1)
        roll_mean, roll_std = rolling.mean(), rolling.std().fillna(0)
        roll_min, roll_max = rolling.min(), rolling.max()
        features = pd.DataFrame()
        for col in sensor_cols:
            features[f"{col}_mean"] = roll_mean[col].values
            features[f"{col}_std"] = roll_std[col].values
            features[f"{col}_min"] = roll_min[col].values
            features[f"{col}_max"] = roll_max[col].values
        features["unit_number"] = group["unit_number"].values
        features["cycle"] = group["cycle"].values
        frames.append(features)
    return pd.concat(frames, ignore_index=True)


def predict_rul_for_engine(engine_df: pd.DataFrame) -> float:
    features = extract_features(engine_df, window_size)
    feature_cols = [c for c in features.columns if c not in ["unit_number", "cycle"]]
    last_row = features.tail(1)[feature_cols].values
    last_row_scaled = rul_scaler.transform(last_row)
    pred = rul_model.predict(last_row_scaled)[0]
    return max(0.0, min(pred, 125.0))


# --- Load and section the maintenance manual (once, at startup) ---
manual_text = MANUAL_PATH.read_text()
manual_sections: dict[str, str] = {}
_current_section, _current_content = "Introduction", []
for _line in manual_text.split("\n"):
    if _line.startswith("## ") and not _line.startswith("###"):
        if _current_content:
            manual_sections[_current_section] = "\n".join(_current_content).strip()
        _current_section = _line.strip("# ").strip()
        _current_content = []
    else:
        _current_content.append(_line)
if _current_content:
    manual_sections[_current_section] = "\n".join(_current_content).strip()


# --- Agent Tools (same logic as the notebook) ---
@tool
def query_rul(engine_id: int) -> str:
    """Query the Remaining Useful Life (RUL) prediction for a specific engine.

    Args:
        engine_id: The engine unit number (integer, e.g., 1, 5, 10).
    """
    try:
        engine_data = test_raw[test_raw["unit_number"] == engine_id].copy()
        if engine_data.empty:
            return f"Engine {engine_id} not found in the dataset."
        pred = predict_rul_for_engine(engine_data)
        last_cycle = engine_data["cycle"].max()
        return (
            f"Engine {engine_id} RUL Prediction:\n"
            f"  Predicted RUL: {pred:.1f} cycles\n"
            f"  Last observed cycle: {last_cycle}\n"
            f"  Model: {model_type} (RMSE: {model_rmse:.2f})\n"
            f"  Window size: {window_size} cycles"
        )
    except Exception as e:
        return f"Error querying RUL for Engine {engine_id}: {str(e)}"


@tool
def lookup_manual(query: str) -> str:
    """Search the engine maintenance manual for relevant procedures and thresholds.

    Useful queries: 'inspection', 'replacement', 'RUL', 'threshold',
    'HPC', 'fan', 'degradation', 'corrective', 'decision', 'maintenance'.

    Args:
        query: A keyword or phrase to search for in the manual.
    """
    query_lower = query.lower()
    query_words = [w for w in re.findall(r"\w+", query_lower) if len(w) > 2]

    best_section, best_score = None, -1
    for section_name, content in manual_sections.items():
        section_lower, content_lower = section_name.lower(), content.lower()
        score = 0
        if query_lower in section_lower:
            score += 20
        score += content_lower.count(query_lower) * 5
        for word in query_words:
            if word in section_lower:
                score += 5
            score += content_lower.count(word)
        if score > best_score:
            best_score, best_section = score, section_name

    if best_section and best_score > 0:
        content = manual_sections[best_section]
        if len(content) > 2000:
            content = content[:2000] + "\n... [content truncated]"
        return f"Manual Section: {best_section}\n\n{content}"

    default = manual_sections.get(
        "RUL-Based Maintenance Decision Matrix",
        manual_sections.get(
            "5. RUL-Based Maintenance Decision Matrix", "No matching section found."
        ),
    )
    return f"No exact match for '{query}'. Returning RUL decision matrix:\n\n{default}"


@tool
def assess_health_status(rul_cycles: float) -> str:
    """Assess the health status of an engine based on its RUL value.

    Args:
        rul_cycles: The predicted RUL in operating cycles (float).
    """
    rul = float(rul_cycles)
    if rul > 200:
        status, action = "HEALTHY", "Continue normal operations. Next scheduled A-Check applies."
    elif rul > 100:
        status, action = "MONITOR", "Increase sensor monitoring frequency. Review trend data at next A-Check."
    elif rul > 50:
        status, action = "WARNING", "Schedule a B-Check within the next 30 cycles. Prepare component replacement parts."
    elif rul > 30:
        status, action = "CAUTION", "Schedule a B-Check immediately. Conduct borescope inspection of HPC and turbine sections."
    elif rul > 15:
        status, action = "ADVISORY", "Plan for C-Check or targeted hot-section inspection. Consider engine removal if RUL drops below 20."
    else:
        status, action = "CRITICAL", "REMOVE ENGINE FROM SERVICE IMMEDIATELY. Schedule full C-Check or engine swap. Do not dispatch."
    return f"Health Assessment for RUL = {rul:.1f} cycles:\n  Status:  {status}\n  Action:  {action}"


TOOLS = [query_rul, lookup_manual, assess_health_status]

# --- Build the LLM connection + ReAct agent (once, at startup) ---
_ollama_status = check_ollama()
llm = ChatOpenAI(
    model=OLLAMA_MODEL,
    base_url=OLLAMA_BASE_URL,
    api_key="not-needed",
    temperature=0,
    timeout=120,
)
agent = create_react_agent(llm, TOOLS)


EXAMPLE_QUERIES = {
    "Engine 1 — general degradation": (
        "Engine 1 has been showing degraded performance in recent flights. "
        "What maintenance action do you recommend? Please check its RUL, "
        "look up the relevant maintenance procedures, and give me a clear recommendation."
    ),
    "Engine 5 — HPC degradation": (
        "I need a maintenance assessment for Engine 5. Check its RUL, look up "
        "the corrective actions for HPC degradation, and tell me what to do."
    ),
}


def run_agent_query(user_query: str, chat_history: list):
    """Generator: streams agent reasoning into chat history.

    Each tool call and observation is appended as a separate assistant message
    so the audience can see the ReAct loop unfold in real time. The final
    structured recommendation appears as the last assistant message.

    Yields (chat_history, "") tuples to update the chatbot and clear the input.
    """
    if not user_query or not user_query.strip():
        raise gr.Error("Please enter a maintenance question, or click one of the example buttons.")

    # Append user message (Gradio 6 MessageDict format)
    chat_history.append({"role": "user", "content": user_query})
    yield chat_history, ""

    input_messages = {
        "messages": [("system", SYSTEM_PROMPT), ("user", user_query)]
    }

    step_count = 0
    final_answer = ""
    last_ai_content = ""
    try:
        for event in agent.stream(input_messages, stream_mode="values"):
            messages = event["messages"]
            last_msg = messages[-1]

            # Capture any AI message content (even with tool calls — model may bundle both)
            if last_msg.type == "ai" and last_msg.content and len(last_msg.content) > 10:
                last_ai_content = last_msg.content
                # Only treat as final if no tool calls accompany it
                if not getattr(last_msg, "tool_calls", None):
                    final_answer = last_ai_content

            if getattr(last_msg, "tool_calls", None):
                step_count += 1
                for tc in last_msg.tool_calls:
                    args_str = ", ".join(f"{k}={v}" for k, v in tc["args"].items())
                    chat_history.append({"role": "assistant", "content": f"\U0001f527 **{tc['name']}({args_str})**"})
                yield chat_history, ""
            elif getattr(last_msg, "tool_call_id", None):
                step_count += 1
                content = last_msg.content if isinstance(last_msg.content, str) else str(last_msg.content)
                if len(content) > 600:
                    content = content[:600] + "..."
                chat_history.append({"role": "assistant", "content": f"\U0001f4cb {content}"})
                yield chat_history, ""
    except Exception as e:
        chat_history.append({"role": "assistant", "content": f"\u274c **Error:** {e}"})
        yield chat_history, ""
        return

    if final_answer:
        chat_history.append({"role": "assistant", "content": final_answer})
    elif last_ai_content and step_count > 0:
        # Model produced reasoning content alongside tool calls — use it
        chat_history.append({"role": "assistant", "content": last_ai_content})
    elif step_count == 0:
        # No tools were called — something went wrong
        chat_history.append({"role": "assistant", "content": "The agent did not produce a response. Please try rephrasing your question."})
    # else: tools were called and observations shown — the reasoning is visible in the chat
    yield chat_history, ""


def reset_chat() -> tuple:
    """Clear the chat history and input textbox."""
    return [], ""


def load_example(label: str) -> str:
    return EXAMPLE_QUERIES[label]


# --- UI Layout ---
CUSTOM_CSS = f"""
#header-banner {{
    background: linear-gradient(135deg, {CARE_BLUE} 0%, {MIDNIGHT_BLUE} 100%);
    padding: 20px 24px;
    border-radius: 12px;
    box-shadow: 0 4px 12px rgba(0,0,0,0.15);
    margin-bottom: 12px;
}}
#header-banner h1, #header-banner p {{
    color: #ffffff !important;
    margin: 0;
}}
#header-banner h1 {{ font-size: 1.5em; font-weight: 700; }}
#header-banner p {{ font-size: 0.95em; margin-top: 4px; opacity: 0.9; }}

#status-banner {{
    border: 1px solid {GRAY};
    border-radius: 8px;
    padding: 6px 12px;
    font-family: monospace;
    font-size: 0.85em;
    margin-bottom: 0;
}}

/* Reset button */
#reset-btn {{
    border-color: {GRAY} !important;
    color: {GRAY} !important;
    min-height: 32px !important;
    font-size: 0.85em !important;
}}

/* Suggestion chips */
.suggestion-btn {{
    border-color: {OCEAN_BLUE} !important;
    color: {CARE_BLUE} !important;
    font-size: 0.82em !important;
    padding: 4px 14px !important;
    border-radius: 20px !important;
    background: transparent !important;
    min-height: 32px !important;
}}
.suggestion-btn:hover {{
    background: {OCEAN_BLUE}18 !important;
}}

/* Chat bubble styling — user messages */
.chatbot-user-bubble {{
    background: {CARE_BLUE} !important;
    color: #ffffff !important;
    border-radius: 18px 18px 4px 18px !important;
}}

/* Chat bubble styling — assistant messages */
.chatbot-assistant-bubble {{
    background: #f0f2f5 !important;
    color: #1a1a1a !important;
    border-radius: 18px 18px 18px 4px !important;
    border-left: 3px solid {OCEAN_BLUE} !important;
}}

footer {{ display: none !important; }}
"""

with gr.Blocks(title=APP_TITLE, fill_height=True) as demo:
    gr.HTML(
        f"""
        <div id="header-banner">
            <h1>{APP_TITLE}</h1>
            <p>{APP_DESCRIPTION}</p>
        </div>
        """
    )

    with gr.Row():
        gr.Markdown(f"**Ollama:** {_ollama_status}", elem_id="status-banner")
        reset_btn = gr.Button("\U0001f5d1 Reset Chat", elem_id="reset-btn", size="sm")

    chatbot = gr.Chatbot(
        label="Conversation",
        scale=1,
        height=400,
        layout="bubble",
        buttons=["copy"],
    )

    with gr.Row():
        ex1_btn = gr.Button(
            "\U0001f527 Engine 1 \u2014 general degradation",
            elem_classes="suggestion-btn",
            size="sm",
        )
        ex2_btn = gr.Button(
            "\U0001f527 Engine 5 \u2014 HPC degradation",
            elem_classes="suggestion-btn",
            size="sm",
        )

    msg = gr.Textbox(
        placeholder="Ask about an engine\u2019s health...",
        submit_btn=True,
        container=False,
    )

    ex1_btn.click(
        fn=lambda: load_example("Engine 1 \u2014 general degradation"),
        outputs=msg,
    )
    ex2_btn.click(
        fn=lambda: load_example("Engine 5 \u2014 HPC degradation"),
        outputs=msg,
    )
    msg.submit(
        fn=run_agent_query,
        inputs=[msg, chatbot],
        outputs=[chatbot, msg],
    )
    reset_btn.click(
        fn=reset_chat,
        outputs=[chatbot, msg],
    )


if __name__ == "__main__":
    demo.launch(theme=gr.themes.Soft(primary_hue="blue"), css=CUSTOM_CSS)
