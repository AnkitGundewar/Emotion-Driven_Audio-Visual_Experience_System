# Emotion-Driven Audio-Visual Experience System

**Enhancing Human-Computer Interaction through Real-Time Multimodal Feedback**

Final project for **CS5170 (Spring 2025)** under Professor Stacy Marsella, Northeastern University.

The system has a short conversation with the user to work out how they feel and what they want (for example, to vent, calm down, or lift their mood). It then generates **music** and **visuals** to match. The user can give feedback, and the system detects their emotion again and regenerates the output. All LLM reasoning is grounded in psychology and media research papers through Retrieval-Augmented Generation (RAG).

---

## How It Works

```
┌──────────────────┐   3 open-ended    ┌──────────────┐
│  TemplateAgent   │ ───questions────▶ │     User     │
│ (RAG: emotion    │ ◀───answers────── │  (Streamlit) │
│  psychology)     │                   └──────▲───────┘
└────────┬─────────┘                          │ feedback loop
         │ emotional-state summary            │
         ▼                                    │
┌──────────────────┐                          │
│   PromptAgent    │  emotion + intent        │
│ (RAG: music- &   │ ──────────┬──────────────┤
│  visual-emotion) │           │              │
└──────────────────┘           ▼              │
             prompt_audio          prompt_visual
                  │                      │
                  ▼                      ▼
        ┌──────────────────┐   ┌────────────────────┐
        │   AudioAgent     │   │   TouchDesigner    │
        │ (Meta MusicGen)  │   │ (Stable Diffusion) │
        └────────┬─────────┘   └─────────┬──────────┘
                 └──── temp.wav ─────────┘ visual_prompt.txt
                        real-time audio-visual output
```

1. **Assessment**: `TemplateAgent` retrieves from emotion-science literature (`data/data_template/`) and writes three empathetic, open-ended questions.
2. **Emotion detection**: It analyzes the user's answers using frameworks such as appraisal theory and the affective circumplex, then summarizes their emotional state.
3. **Prompt generation**: `PromptAgent` identifies the primary **emotion** and the user's **intent**. It retrieves research on music-emotion (`data/data_prompt/audio/`) and visual-emotion (`data/data_prompt/visual/`) mappings, then returns JSON:
   ```json
   {"prompt_audio": "...", "prompt_visual": "..."}
   ```
4. **Audio generation**: `AudioAgent` passes `prompt_audio` to Meta's **MusicGen** (30 s clip at 32 kHz) and saves it as `temp.wav`.
5. **Visual generation**: `prompt_visual` is written to `visual_prompt.txt`, which a **TouchDesigner** network reads and renders with Stable Diffusion alongside the audio.
6. **Feedback loop**: The user can send free-text feedback at any time. The system re-detects the emotion and regenerates both outputs. Conversation memory is kept across turns.

---

## Repository Structure

| Path | Description |
|---|---|
| `App.py` | Streamlit UI ("Reality Architects"): questions → emotion → generation → feedback loop |
| `TemplateGenerator.py` | `TemplateAgent`: RAG-based question generation and emotional-state analysis |
| `PromptGenerator.py` | `PromptAgent`: RAG-based emotion/intent extraction and audio + visual prompt generation |
| `AudioGen.py` | `AudioAgent`: wrapper around Audiocraft MusicGen |
| `DocLoaders.py` | Loaders for PDF, TXT/MD, and web sources (LangChain) |
| `Runner.ipynb` | Notebook that runs the full pipeline step by step without the UI |
| `data/data_template/` | Emotion-science papers used by `TemplateAgent` |
| `data/data_prompt/audio/` | Music-emotion research used by `PromptAgent` |
| `data/data_prompt/visual/` | Visual-emotion research used by `PromptAgent` |
| `TD files/` | Sample outputs consumed by TouchDesigner (`temp.wav`, `visual_prompt.txt`) |
| `survey_data.csv` | User-study responses (5-point Likert scale) |
| `Project Report.pdf` / `Project Presentation.pptx` | Final report and presentation |

---

## Tech Stack

- **LLM / RAG**: LangChain, OpenAI `o1-mini`, OpenAI Embeddings, ChromaDB
- **Audio**: Meta Audiocraft MusicGen (`small` in the app, `large` in the notebook), PyTorch, torchaudio
- **Visuals**: TouchDesigner, Stable Diffusion
- **UI**: Streamlit

---

## Setup

> ⚠️ This repository mainly shares the source code. Several paths are hard-coded for the original Windows machine, and the TouchDesigner project isn't included (see below). You'll need to make the changes listed here to run it.

### 1. Requirements

- Python 3.10 (tested with PyTorch 2.1)
- NVIDIA GPU recommended. The demos ran on an RTX 4080 (12 GB VRAM) laptop GPU with an Intel Core Ultra 9 185H.
- [TouchDesigner](https://derivative.ca/) and a local Stable Diffusion setup for the visual part
- OpenAI API key

### 2. Install dependencies

```bash
conda create -n emotion-av python=3.10
conda activate emotion-av

pip install torch==2.1.0 torchaudio==2.1.0 --index-url https://download.pytorch.org/whl/cu121
pip install audiocraft==1.3.0 streamlit==1.44.0 chromadb==0.6.3 \
            langchain==0.3.21 langchain-community==0.3.20 langchain-core==0.3.49 \
            langchain-openai==0.3.11 openai==1.69.0 pypdf
```

> `requirements.txt` is a full environment export (UTF-16 encoded, with some local conda `file://` entries), so `pip install -r` may fail. Use the commands above, or install from it selectively.

MusicGen weights (`facebook/musicgen-small` / `-large`) download automatically from Hugging Face on first run.

### 3. Configure API keys

`PromptGenerator.py` and `TemplateGenerator.py` set placeholder keys at the top of each file. Replace them, or better, delete those lines and export the keys in your shell:

```bash
export OPENAI_API_KEY="sk-..."
export LANGCHAIN_API_KEY="..."   # optional, only for LangSmith tracing
```

If you aren't using LangSmith, set `LANGCHAIN_TRACING_V2` to `false`.

### 4. Set the output directory

The generated audio and visual prompt go to a folder that TouchDesigner watches. That folder is currently hard-coded as `D:\TouchDesigner projects`:

- `AudioGen.py`: `path = "D:\TouchDesigner projects"`
- `App.py`: `open("D:\\TouchDesigner projects\\visual_prompt.txt", ...)`

Change both to a directory on your machine, and point the TouchDesigner file inputs at the same place.

### 5. TouchDesigner project

The TouchDesigner files are larger than GitHub's 100 MB limit, so they're hosted on **[Google Drive](https://drive.google.com/drive/folders/1y-JZFFD8MT3kFVQhoCAvIkZFeGOpYXu4?usp=share_link)**. Download them, set up Stable Diffusion locally, and update any module paths that still use their defaults.

---

## Usage

**Streamlit app**

```bash
streamlit run App.py
```

1. Answer the three questions the system asks.
2. The system shows your detected emotional state and writes new audio and visual prompts.
3. TouchDesigner picks up `temp.wav` and `visual_prompt.txt` and renders the experience.
4. Type feedback (for example, "I want something calmer") to regenerate. Click **Reset** to start over.

**Notebook**: open `Runner.ipynb` to step through each agent without the UI.

---

## Results

The user study had 9 participants, scored on a 5-point Likert scale (from `survey_data.csv`):

| Statement | Mean |
|---|:---:|
| I would prefer a system that adapts to my emotions over one that does not | **4.89** |
| Audio/video adaptations felt empathetic toward my mood | 4.67 |
| Overall, I enjoyed using this adaptive system | 4.56 |
| Generated media felt fresh, creative, or novel | 4.56 |
| The iterative feedback process improved the output over time | 4.44 |
| My feedback genuinely influenced the final output | 4.33 |
| I felt highly immersed when the system adapted | 4.22 |
| The system accurately recognized my emotional state | 4.11 |
| Changes felt personally tailored to me | 4.11 |
| Adaptive media positively affected my emotional state | 4.11 |

See the [Project Report](Project%20Report.pdf) and [Presentation](Project%20Presentation.pptx) for full details.

---

## Demos

The demos were recorded on the hardware listed above and uploaded to YouTube as unlisted videos.

- [Demo 1](https://youtu.be/jDh-97PKf-g)
- [Demo 2: With Feedback](https://youtu.be/XiiXxGKiMjY)
- [Demo 3](https://youtu.be/ann_1mOyzaw)
- [Gameplay Demo 1](https://youtube.com/shorts/L_fuUXUeV-4)
- [Gameplay Demo 2](https://youtube.com/shorts/Vz14GaXR7AE)
- [Gameplay Demo 3](https://youtube.com/shorts/75xuLgsyzOM)

---

## Limitations & Future Work

- Output paths are hard-coded, and the Windows-only paths should move into a config or env var.
- API keys are set in source code and should be loaded from the environment.
- Vector stores are rebuilt from the PDFs on every request. Persisting ChromaDB would make responses much faster.
- A clean, cross-platform `requirements.txt` is still needed.
- A step-by-step guide for the TouchDesigner network is still to be written.
