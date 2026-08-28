import json
from pathlib import Path


def load_pitchdeck_json(json_path: str):
    """Load and format labelled pitchdeck data for RAG (summary-only)."""
    path = Path(json_path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {json_path}")

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    slides = []
    for slide in data:
        text = slide["summary"]
        if slide.get("key_points"):
            text += "\n\nKey Points:\n" + "\n".join(f"- {p}" for p in slide["key_points"])
        slides.append({
            "id": slide["slide"],
            "section": slide["section"],
            "text": text,
            "source": "summary",
        })
    return slides


def load_pitchdeck_combined(analysis_path: str, parsed_path: str = None):
    """
    Load both Gemini summaries and raw slide text as separate retrievable units.

    Each unit has a 'source' field:
      - 'summary'  : Gemini-generated structured summary (clean embeddings, better semantic retrieval)
      - 'raw'      : Raw text extracted from the PDF (better for exact figures, quotes, proper nouns)

    Design Decision:
      Summary-only indexing produces tighter embeddings for semantic queries.
      Raw text improves recall for specific figures and proper nouns that summaries may omit.
      When both are indexed, retrieved slides are tagged with their source type in the CLI output.
    """
    # Always load summaries
    slides = load_pitchdeck_json(analysis_path)

    # Optionally layer in raw text from parsed JSON
    if parsed_path is not None:
        parsed_path = Path(parsed_path)
        if parsed_path.exists():
            with open(parsed_path, "r", encoding="utf-8") as f:
                parsed_data = json.load(f)

            raw_pages = parsed_data.get("pitchdeck", [])
            for page in raw_pages:
                raw_text = page.get("text", "").strip()
                if raw_text:
                    slides.append({
                        "id": page["page"],
                        "section": "raw",
                        "text": raw_text,
                        "source": "raw",
                    })
        else:
            print(f"⚠️  Parsed JSON not found at {parsed_path} — skipping raw text indexing.")

    return slides


def load_image_descriptions(image_desc_path: str):
    """
    Load Gemini-generated image descriptions for indexing.
    Returns a list of dicts with keys: id, section, text, source='image_description'.
    """
    path = Path(image_desc_path)
    if not path.exists():
        return []

    with open(path, "r", encoding="utf-8") as f:
        descriptions = json.load(f)

    slides = []
    for item in descriptions:
        desc_text = item.get("description", "").strip()
        if desc_text:
            slides.append({
                "id": item["page"],
                "section": item.get("section", "image"),
                "text": desc_text,
                "source": "image_description",
            })
    return slides
