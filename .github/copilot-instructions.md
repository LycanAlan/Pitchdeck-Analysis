# AI Agent Instructions for Pitchdeck Analysis Project

## Project Overview
This project is a pitch deck analysis system that processes PDF pitch decks using Google's Gemini AI to extract and analyze their content. The system follows a two-stage pipeline:

1. **Preprocessing Stage** (`preprocessing/`)
   - Extracts text and images from PDF pitch decks
   - Uses Google Gemini AI for structured analysis
   - Outputs parsed JSON and image files

2. **RAG Pipeline Stage** (`rag_pipeline/`)
   - Loads processed pitch deck data
   - Implements RAG (Retrieval-Augmented Generation) for querying pitch deck content

## Key Components and Data Flow

### Preprocessing Pipeline
- Entry point: `preprocessing/version0_1.py`
- Input: PDF files in `pitch_decks/` directory
- Configuration: `prompt.json` for Gemini AI analysis instructions
- Output: Generated in `outputs/` directory
  - `{deck_name}_parsed.json`: Raw extracted text
  - `{deck_name}_analysis.json`: Structured Gemini analysis
  - `{deck_name}_images/`: Extracted deck images

### RAG Query System
- Core components:
  - `rag_pipeline/loader.py`: Loads and formats processed pitch deck data
  - `rag_pipeline/rag_query_agent.py`: Implements RAG query functionality

## Environment Setup
1. Create a `.env` file in the project root with:
   ```
   GEMINI_API_KEY=your_api_key_here
   ```
2. Required Python packages (see `dev/` virtual environment):
   - PyMuPDF (fitz)
   - google-generativeai
   - python-dotenv
   - langchain
   - faiss-cpu

## Project Conventions

### Data Structures

#### Gemini Analysis Output (`{deck_name}_analysis.json`)
```json
[
    {
        "slide": 1,
        "section": "Other",
        "summary": "Slide content summary",
        "key_points": [
            "Key point 1",
            "Key point 2",
            "Key point 3"
        ]
    }
]
```

#### RAG Pipeline Format (After processing by `loader.py`)
```python
{
    "id": "slide_number",
    "section": "slide_section",
    "text": "combined_summary_and_key_points"
}
```

### Error Handling
- Failed Gemini API responses are saved as `{deck_name}_error.txt`
- PDF processing errors are logged but don't halt pipeline execution

## Development Guidelines
1. Maintain JSON schema consistency in `prompt.json`
2. Use Path objects for file operations (`pathlib.Path`)
3. Keep PDF processing and AI analysis logic separated
4. Handle both text extraction and image processing for completeness