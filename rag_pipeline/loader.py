import json
from pathlib import Path

def load_pitchdeck_json(json_path: str):
    """Load and format labelled pitchdeck data for RAG."""
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
            "text": text
        })
    return slides
