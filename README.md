# Pitchdeck Analysis & RAG Query System

An AI-powered pitch deck analysis system that parses startup pitch decks (PDF), extracts structured insights using Google's Gemini models, and enables conversational question answering over the pitch deck content using a Retrieval-Augmented Generation (RAG) pipeline.

---

## 🌟 Architecture & Features

The project is structured into two main stages:

1. **Preprocessing Pipeline (`preprocessing/`)**
   - **Text & Image Extraction**: Extracts slide text and embedded images from pitch deck PDFs using PyMuPDF (`fitz`).
   - **Structured AI Analysis**: Sends extracted slide content with guided prompt schemas (`prompt.json`) to Google Gemini to produce structured slide-by-slide summaries, sections, and key takeaways.
   - **Outputs**: Generates `{deck_name}_parsed.json`, `{deck_name}_analysis.json`, and an images folder.

2. **RAG Pipeline Stage (`rag_pipeline/`)**
   - **Semantic Embeddings**: Uses Hugging Face (`sentence-transformers/all-MiniLM-L6-v2`) for fast, local vector embeddings.
   - **Vector Store**: FAISS in-memory vector indexing over structured slide contents and metadata.
   - **Reasoning LLM**: Integrates Google Gemini (`gemini-2.5-pro` / `gemini-1.5-pro`) via LangChain to provide context-grounded answers.
   - **Interactive CLI**: Query the deck interactively in the terminal.

---

## 📂 Project Structure

```text
├── .env.example                  # Environment variables template
├── .gitignore                    # Git ignore file (excludes secrets & venvs)
├── requirements.txt              # Project dependencies
├── preprocessing/                # PDF parsing and Gemini structuring
│   ├── prompt.json               # Structured prompt & JSON schema
│   ├── version0_1.py             # Preprocessing execution script
│   ├── pitch_decks/              # Directory for raw pitch deck PDFs
│   └── outputs/                  # Generated parsed JSON and images
└── rag_pipeline/                 # RAG query system
    ├── main.py                   # CLI entry point for Q&A
    ├── rag_query_agent.py        # Setup & query handler
    ├── rag_pipeline.py           # LangChain RAG chain definition
    ├── embedder.py               # Hugging Face embeddings configuration
    ├── vectorstore.py            # FAISS vector store creation
    ├── loader.py                 # JSON data loader and formatter
    └── test_embedder.py          # Embeddings test script
```

---

## 🚀 Getting Started

### 1. Prerequisites

- Python 3.10+
- Google Gemini API Key ([Get an API key here](https://aistudio.google.com/))

### 2. Installation

1. Clone the repository:
   ```bash
   git clone <YOUR_GITHUB_REPO_URL>
   cd Pitchdeck-Analysis-version-2
   ```

2. Create and activate a virtual environment:
   ```bash
   python -m venv .venv
   # On Windows:
   .venv\Scripts\activate
   # On macOS/Linux:
   source .venv/bin/activate
   ```

3. Install required packages:
   ```bash
   pip install -r requirements.txt
   ```

### 3. Environment Configuration

Create a `.env` file in the project root:
```bash
cp .env.example .env
```
Open `.env` and add your Gemini API Key:
```env
GEMINI_API_KEY=your_actual_gemini_api_key
```

---

## 💻 Usage

### Stage 1: Preprocess Pitch Decks
Place your pitch deck PDF into `preprocessing/pitch_decks/`, then run:
```bash
python preprocessing/version0_1.py
```
This extracts text, images, and generates structured analysis files in `preprocessing/outputs/`.

### Stage 2: Ask Questions (RAG System)
Run the interactive query assistant:
```bash
cd rag_pipeline
python main.py
```
Ask questions such as:
- *"What is the company's business model?"*
- *"What problem are they solving?"*
- *"Who are the competitors and what is their moat?"*

---

## 🛡️ License

This project is open source and available under the [MIT License](LICENSE).
