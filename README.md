# AI in Aviation: Practical Applications — Course Demo Materials

Welcome! This repository contains the hands-on demo materials for the **"AI in Aviation: Practical Applications"** course — two Jupyter notebooks and a small web app that show, step by step, how machine learning and agentic AI can support aircraft engine maintenance decisions.

**You do not need any prior programming or AI experience to run these demos.** This guide walks you through everything from scratch: installing the tools, downloading this repository, and running each demo.

If you get stuck at any point, see the [Troubleshooting](#troubleshooting) section near the end — most issues have a one-line fix.

---

## What's in This Repository

| Day | File | What it does |
|---|---|---|
| **Day 1** | `src/day1_cmapss_rul.ipynb` | Trains a machine learning model that **predicts** how many flight cycles remain before an engine needs maintenance — called Remaining Useful Life (RUL). |
| **Day 2** | `src/day2_agentic_maintenance.ipynb` | Builds an **agentic AI system** that uses the Day 1 model plus a sample maintenance manual to **recommend** a specific maintenance action — not just a number. |
| **Day 2 (bonus)** | `src/day2_demo_app.py` | The same Day 2 system, wrapped in a simple point-and-click web app — a preview of what this looks like as a deployed application, instead of a notebook. |

Supporting files:

- `assets/cmapss/` — the NASA engine sensor dataset used by both days (see [Dataset & License](#dataset--license) below).
- `assets/models/rul_model.joblib` — the trained Day 1 model, reused by Day 2.
- `assets/manuals/engine_maintenance_program.md` — a sample engine maintenance manual used by the Day 2 AI agent.
- `assets/*.png` — reference charts produced by the Day 1 notebook.
- `pyproject.toml` / `uv.lock` / `.python-version` — exact list of software this project needs, and the precise versions that are known to work together.

---

## Before You Start: What You Need

You need three things installed on your computer, once:

1. **Python 3.11 or newer** — most Macs and Linux machines already have some version of Python. Windows users can install it from [python.org](https://www.python.org/downloads/).
2. **`uv`** — a tool that automatically installs the correct Python packages for this project (pandas, scikit-learn, etc.), so you don't have to install each one by hand.
3. **Ollama** — only needed for the **Day 2** demos (the agentic AI). It runs a small AI model directly on your own computer — no internet connection or account needed once it's downloaded.

The sections below walk through installing each one.

### 1. Install `uv`

`uv` is a fast, modern tool for managing Python projects. Open a terminal (Terminal.app on macOS, or PowerShell on Windows) and run:

**macOS / Linux:**
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

**Windows (PowerShell):**
```powershell
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

Verify it installed correctly:
```bash
uv --version
```

### 2. Install Ollama (required for the Day 2 demos only)

Ollama lets you run an AI model locally on your own laptop — nothing is sent to the cloud.

| Your Operating System | How to Install |
|---|---|
| **macOS** | Download from [ollama.com/download/mac](https://ollama.com/download/mac), or run `brew install ollama` if you use Homebrew |
| **Windows** | Download the installer from [ollama.com/download/windows](https://ollama.com/download/windows) |
| **Linux** | Run `curl -fsSL https://ollama.com/install.sh \| sh`, or see [ollama.com/download/linux](https://ollama.com/download/linux) |

After installing, verify Ollama is working and download the AI model used in this course (about 1.9 GB, one-time download):

```bash
ollama --version
ollama pull qwen2.5:3b
```

> **About the model:** `qwen2.5:3b` is developed by Alibaba (China). It's a small, efficient open-weight model that runs comfortably on a laptop with 8 GB of RAM, while still being capable enough for the tool-calling demo in this course.

Make sure Ollama is **running** before you open the Day 2 notebook or app — on macOS/Windows it usually runs automatically in the background after installation (look for the Ollama icon in your menu bar / system tray). If it isn't running, start it from your Applications folder, or run `ollama serve` in a terminal.

### 3. Download this repository

If you have `git` installed:
```bash
git clone https://github.com/mqignacio/ai_in_aviation.git
cd ai_in_aviation
```

If you don't have `git`, click the green **"Code"** button on the GitHub repository page, choose **"Download ZIP"**, then unzip it and open a terminal inside the unzipped folder.

### 4. Install the project's Python packages

From inside the `ai_in_aviation` folder, run:

```bash
uv sync
```

This reads `pyproject.toml` and `uv.lock` and installs the exact versions of every package needed (pandas, scikit-learn, LangChain, LangGraph, Gradio, etc.) into a private project folder (`.venv`) that won't interfere with anything else on your computer. This may take a few minutes the first time.

---

## Running the Day 1 Demo (RUL Prediction)

Day 1 does not require Ollama — it's a standard machine learning notebook.

```bash
uv run jupyter lab
```

This opens Jupyter Lab in your web browser. In the file browser on the left, open `src/day1_cmapss_rul.ipynb`, then run the cells from top to bottom (Menu: **Run → Run All Cells**, or click each cell and press **Shift+Enter**).

**What you'll see:** the notebook loads real (simulated) jet engine sensor data, explains the model-building process in plain language, trains several models, and shows how accurately each one predicts when an engine will need maintenance.

---

## Running the Day 2 Demo (Agentic Maintenance Recommendations)

Day 2 requires Ollama to be running with the `qwen2.5:3b` model pulled (see step 2 above). There are two ways to experience it:

### Option A — The Notebook (recommended for learning how it works)

```bash
uv run jupyter lab
```

Open `src/day2_agentic_maintenance.ipynb` and run the cells from top to bottom. The notebook explains, cell by cell, how an AI "agent" can look up engine health data, read a maintenance manual, and reason its way to a recommendation — narrated for a non-programmer audience.

### Option B — The Web App (a preview of a deployed version)

This runs the exact same AI system as the notebook, but behind a **messenger-style chat interface** — showing what this might look like as a real, deployed application rather than a notebook.

```bash
uv run python src/day2_demo_app.py
```

This will print a local web address, for example:
```
* Running on local URL:  http://127.0.0.1:7860
```

Open that address in your web browser. You'll see:

- A status line confirming Ollama is running and the AI model is ready.
- A chat interface where you can type a maintenance question, or click one of the **suggestion chips** for a ready-made scenario.
- Press **Send** to start the analysis.
- The agent's reasoning streams into the chat in real time — 🔧 icons show tool calls, 📋 icons show observations.
- The final structured recommendation appears as the last assistant message.
- Use the **Reset Chat** button to clear the conversation and start fresh.

To stop the app, go back to the terminal window and press `Ctrl+C`.

---

## Dataset & License

Both demos use **NASA's C-MAPSS (Commercial Modular Aero-Propulsion System Simulation)** dataset — simulated turbofan engine sensor data, run to failure, produced by the NASA Prognostics Center of Excellence. This is one of the most widely used public benchmark datasets in aircraft engine health monitoring research.

This project uses the **FD001** subset: 100 training engines and 100 test engines, one operating condition, one fault mode (High-Pressure Compressor degradation).

- **License status:** Public domain, hosted on NASA's Open Data Portal for research and educational use. See `assets/cmapss/SOURCE.md` for the full verification notes.
- **Citation:** Saxena, A., Goebel, K., Simon, D., and Eklund, N., "Damage Propagation Modeling for Aircraft Engine Run-to-Failure Simulation," *Proc. 1st Int. Conf. on Prognostics and Health Management (PHM08)*, Denver, CO, Oct. 2008.

The sample `engine_maintenance_program.md` maintenance manual used in Day 2 is a fictional, course-authored document for teaching purposes — it does not represent any real airline's or manufacturer's maintenance program.

---

## Frequently Asked Questions

**Do I need to know how to code?**
No. Both notebooks are written with plain-language explanations before every step. You only need to click "Run" on each cell in order.

**Is any of my data sent to the cloud?**
No. Everything — the machine learning model and the Day 2 AI agent — runs entirely on your own computer. No internet connection is required after the initial downloads (dataset, packages, and the Ollama model are all included in or fetched once by this setup).

**Can I use a different AI model for Day 2?**
Yes. Pull any Ollama-compatible model (`ollama pull <model_name>`) and change the `OLLAMA_MODEL` variable near the top of the Day 2 notebook or `src/day2_demo_app.py`. Smaller models may struggle with tool-calling; `qwen2.5:3b` has been tested and works reliably.

**How accurate is the RUL prediction?**
The best model trained in the Day 1 notebook (Gradient Boosting) achieves a Root Mean Squared Error (RMSE) of about 19 cycles on held-out test engines — meaning predictions are typically off by about 19 flight cycles from the true remaining life. This is sufficient for maintenance planning, not for safety-critical, real-time decisions.

---

## Troubleshooting

| Problem | Fix |
|---|---|
| `uv: command not found` | Re-run the install command from [step 1](#1-install-uv), then close and reopen your terminal. |
| `uv sync` fails or hangs | Check your internet connection (packages are downloaded on first run). Try again with `uv sync --refresh`. |
| Day 2 notebook/app shows "Ollama is not running" | Start Ollama (menu bar icon on macOS, or run `ollama serve` in a terminal), then re-run the cell or restart the app. |
| Day 2 says the model was not found | Run `ollama pull qwen2.5:3b` in a terminal, then try again. |
| Web app doesn't open automatically | Copy the `http://127.0.0.1:7860` address printed in the terminal and paste it into your browser manually. |
| The AI agent takes a long time to respond | This is expected — each AI reasoning step takes 10–30 seconds on a laptop. A full recommendation can take 30–90 seconds. Please be patient; there is no error. |

---

## Acknowledgment

Course materials developed with the assistance of GitHub Copilot for code scaffolding, notebook narration drafting, and this README. All technical content, dataset citations, and educational framing were reviewed and verified by the course instructor.
