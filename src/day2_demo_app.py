"""
Day 2 Demo — Agentic Prescriptive Maintenance (Gradio web app)

A GUI version of `day2_agentic_maintenance.ipynb` for instructor-led live
demos where a notebook is not convenient (e.g. presenting on a projector,
running headless, or handing the mouse to a volunteer in the audience).

Same pipeline as the notebook: Day 1 RUL model (Gradient Boosting) + a
sample Engine Maintenance Program manual + a LangGraph ReAct agent backed
by a local Ollama model (qwen2.5:3b). The agent's reasoning (Thought /
Action / Observation) is streamed live into a terminal-style scrolling
panel so the audience can see the ReAct loop happen in real time, while
the final structured recommendation is surfaced separately with visual
priority.

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
    "Ask the agent about an engine's health. It queries the Day 1 RUL model, "
    "consults the maintenance manual, and recommends a maintenance action — "
    "all running locally, no cloud dependency."
)

# Color Palette (Option A / Palette 1)
MAROON = "#8D4F58"
GREEN = "#426556"
GOLD = "#F1AE35"
BLUE = "#1B587C"
GRAY = "#555555"

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


def run_agent_query(user_query: str):
    """Generator: streams the ReAct loop into a terminal-style log, then yields
    the final structured recommendation once the agent produces it.

    Yields (log_text, final_text) tuples so the Gradio UI updates both the
    scrolling terminal panel and the final-answer card as the agent progresses.
    """
    if not user_query or not user_query.strip():
        raise gr.Error("Please enter a maintenance question, or click one of the example buttons.")

    log_lines = [
        "=" * 70,
        f"USER QUERY: {user_query}",
        "=" * 70,
        "",
    ]
    yield "\n".join(log_lines), ""

    input_messages = {
        "messages": [("system", SYSTEM_PROMPT), ("user", user_query)]
    }

    step_count = 0
    final_answer = ""
    try:
        for event in agent.stream(input_messages, stream_mode="values"):
            messages = event["messages"]
            last_msg = messages[-1]

            if getattr(last_msg, "tool_calls", None):
                step_count += 1
                for tc in last_msg.tool_calls:
                    log_lines.append(f"\u2500\u2500\u2500 STEP {step_count}: ACTION \u2500\u2500\u2500")
                    log_lines.append(f"  Tool:  {tc['name']}")
                    log_lines.append(f"  Args:  {tc['args']}")
                    log_lines.append("")
                yield "\n".join(log_lines), ""
            elif getattr(last_msg, "tool_call_id", None):
                step_count += 1
                content = last_msg.content if isinstance(last_msg.content, str) else str(last_msg.content)
                if len(content) > 600:
                    content = content[:600] + "..."
                log_lines.append(f"\u2500\u2500\u2500 STEP {step_count}: OBSERVATION \u2500\u2500\u2500")
                log_lines.append(f"  Result: {content}")
                log_lines.append("")
                yield "\n".join(log_lines), ""
            elif last_msg.type == "ai" and not getattr(last_msg, "tool_calls", None):
                if step_count > 0 and last_msg.content and len(last_msg.content) > 50:
                    final_answer = last_msg.content
                    log_lines.append("=" * 70)
                    log_lines.append("FINAL RECOMMENDATION PRODUCED \u2014 see panel on the right.")
                    log_lines.append("=" * 70)
                    yield "\n".join(log_lines), final_answer
    except Exception as e:
        log_lines.append("")
        log_lines.append(f"\u274c ERROR: {e}")
        yield "\n".join(log_lines), "An error occurred. See the terminal log for details."
        return

    if not final_answer:
        final_answer = "Agent completed without a final structured answer."
        yield "\n".join(log_lines), final_answer


def load_example(label: str) -> str:
    return EXAMPLE_QUERIES[label]


# --- UI Layout ---
CUSTOM_CSS = f"""
#header-banner {{
    background: linear-gradient(135deg, {BLUE} 0%, {MAROON} 100%);
    padding: 24px 28px;
    border-radius: 12px;
    box-shadow: 0 4px 12px rgba(0,0,0,0.15);
    margin-bottom: 16px;
}}
#header-banner h1, #header-banner p {{
    color: #ffffff !important;
    margin: 0;
}}
#header-banner h1 {{ font-size: 1.6em; font-weight: 700; }}
#header-banner p {{ font-size: 1em; margin-top: 6px; opacity: 0.95; }}

#status-banner {{
    border: 1px solid {GRAY};
    border-radius: 8px;
    padding: 8px 14px;
    font-family: monospace;
    font-size: 0.9em;
}}

#terminal-log textarea {{
    background: #1E2227 !important;
    color: #7CFC9A !important;
    font-family: "Courier New", monospace !important;
    font-size: 0.85em !important;
    line-height: 1.4em;
    border-radius: 8px !important;
}}

#final-recommendation {{
    border: 2px solid {GOLD};
    border-radius: 12px;
    padding: 4px;
    box-shadow: 0 2px 8px rgba(0,0,0,0.08);
}}
#final-recommendation textarea {{
    font-size: 1.02em !important;
    line-height: 1.5em;
}}

button.primary {{
    background: {BLUE} !important;
    border-color: {BLUE} !important;
}}
button.secondary {{
    border-color: {GREEN} !important;
    color: {GREEN} !important;
}}

footer {{ display: none !important; }}
"""

with gr.Blocks(title=APP_TITLE) as demo:
    gr.HTML(
        f"""
        <div id="header-banner">
            <h1>{APP_TITLE}</h1>
            <p>{APP_DESCRIPTION}</p>
        </div>
        """
    )

    gr.Markdown(f"**Ollama status:** {_ollama_status}", elem_id="status-banner")

    with gr.Tabs():
        with gr.Tab("Live Demo"):
            with gr.Row():
                with gr.Column(scale=2):
                    query_box = gr.Textbox(
                        label="Maintenance question",
                        placeholder=(
                            "e.g. Engine 1 has been showing degraded performance. "
                            "What maintenance action do you recommend?"
                        ),
                        lines=3,
                    )
                    with gr.Row():
                        example1_btn = gr.Button(
                            "Load Example: Engine 1 (general degradation)",
                            variant="secondary",
                        )
                        example2_btn = gr.Button(
                            "Load Example: Engine 5 (HPC degradation)",
                            variant="secondary",
                        )
                    run_btn = gr.Button("Run Maintenance Analysis", variant="primary")

                    gr.Markdown("### Agent Reasoning Log (live ReAct loop)")
                    terminal_log = gr.Textbox(
                        label="",
                        lines=22,
                        max_lines=22,
                        autoscroll=True,
                        interactive=False,
                        elem_id="terminal-log",
                        buttons=["copy"],
                    )

                with gr.Column(scale=2):
                    gr.Markdown("### Final Recommendation")
                    final_output = gr.Textbox(
                        label="",
                        lines=24,
                        interactive=False,
                        elem_id="final-recommendation",
                        buttons=["copy"],
                    )

            example1_btn.click(fn=lambda: load_example("Engine 1 — general degradation"), outputs=query_box)
            example2_btn.click(fn=lambda: load_example("Engine 5 — HPC degradation"), outputs=query_box)
            run_btn.click(fn=run_agent_query, inputs=query_box, outputs=[terminal_log, final_output])

        with gr.Tab("About"):
            gr.Markdown(
                f"""
                ### What this demo shows

                This app is the GUI counterpart of `src/day2_agentic_maintenance.ipynb`
                (AI in Aviation — Day 2). It demonstrates **agentic AI** for
                prescriptive maintenance: an autonomous agent that combines a
                predictive model with domain knowledge to recommend action, not
                just predict a number.

                | Component | Role | Technology |
                |---|---|---|
                | Predictive Model | Engine health (RUL) | Day 1 Gradient Boosting model (RMSE {model_rmse:.2f}) |
                | Domain Knowledge | Maintenance procedures & thresholds | Engine Maintenance Program manual |
                | Agentic Orchestrator | Reasoning, tool selection, recommendation | LangChain + LangGraph ReAct loop, local Ollama ({OLLAMA_MODEL}) |

                **Available engines in the demo dataset:** {", ".join(str(e) for e in ENGINE_IDS)}

                Everything runs locally — no cloud API calls, no data leaves this machine.

                **The Calculator Lesson, revisited:** the agent can query data, read
                manuals, and produce recommendations, but the human maintenance
                engineer remains accountable for the final decision. The agent is
                an assistant, not a replacement.
                """
            )


if __name__ == "__main__":
    demo.launch(theme=gr.themes.Soft(primary_hue="blue"), css=CUSTOM_CSS)
