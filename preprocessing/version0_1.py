import os
import json
import fitz  # PyMuPDF
import google.generativeai as genai
from dotenv import load_dotenv

# 1. Load environment variables from a .env file
load_dotenv()

# 2. Get the API key from the environment variables
api_key = os.environ.get("GEMINI_API_KEY")

# 3. Check if the API key was loaded successfully
if not api_key:
    raise ValueError("GEMINI_API_KEY not found. Please create a .env file and add your API key to it.")

# 4. Configure the Gemini API with the loaded key
genai.configure(api_key=api_key)

print("Gemini API configured successfully.")



def extract_text_from_pdf(pdf_path):
    """Extract text from each page of the PDF using PyMuPDF (no OCR)."""
    try:
        pdf_document = fitz.open(pdf_path)
        pages = []
        for i, page in enumerate(pdf_document):
            text = page.get_text("text")
            pages.append({"page": i + 1, "text": text.strip()})
        pdf_document.close()
        return {"pitchdeck": pages}
    except fitz.FileNotFoundError:
        print(f"❌ PDF not found: {pdf_path}")
        return None
    except fitz.EmptyFileError:
        print(f"❌ The PDF file is empty: {pdf_path}")
        return None
    except Exception as e:
        print(f"❌ Could not read PDF '{pdf_path}': {e}")
        return None


def extract_images_from_pdf(pdf_path, images_output_dir):
    """Extracts all images from a PDF and saves them to a directory."""
    try:
        pdf_document = fitz.open(pdf_path)
        image_count = 0
        for page_index in range(len(pdf_document)):
            page = pdf_document[page_index]
            image_list = page.get_images(full=True)
            for img_index, img in enumerate(image_list, start=1):
                xref = img[0]
                base_image = pdf_document.extract_image(xref)
                image_bytes = base_image["image"]
                image_ext = base_image.get("ext", "png")
                
                image_filename = f"page_{page_index+1}_img_{img_index}.{image_ext}"
                image_path = os.path.join(images_output_dir, image_filename)
                
                with open(image_path, "wb") as img_file:
                    img_file.write(image_bytes)
                image_count += 1
        
        pdf_document.close()
        if image_count > 0:
            print(f"   Extracted {image_count} images to {images_output_dir}")
        else:
            print("   No embedded images found in the PDF.")
            
    except Exception as e:
        print(f"⚠️  Could not extract images from '{pdf_path}': {e}")


def generate_image_descriptions(pdf_path, deck_name, outputs_dir):
    """
    Generate Gemini multimodal descriptions for each page of the PDF (per-page approach).

    Renders each page as a 150-DPI PNG and sends it to gemini-2.5-flash with a
    structured prompt. Results are saved to {deck_name}_image_descriptions.json.

    Cost: ~1 Gemini API call per page (e.g. ~27 calls for a 27-page deck).

    Args:
        pdf_path: Path to the source PDF.
        deck_name: Base name used for the output file.
        outputs_dir: Directory where the output JSON will be saved.
    """
    output_path = os.path.join(outputs_dir, f"{deck_name}_image_descriptions.json")
    descriptions = []

    try:
        pdf_document = fitz.open(pdf_path)
        total_pages = len(pdf_document)
        print(f"   Generating image descriptions for {total_pages} pages...")

        model = genai.GenerativeModel("gemini-2.5-flash")

        for page_index in range(total_pages):
            page_num = page_index + 1
            try:
                page = pdf_document[page_index]

                # Render page to PNG at 150 DPI (matrix = scale of 150/72)
                mat = fitz.Matrix(150 / 72, 150 / 72)
                pix = page.get_pixmap(matrix=mat)
                img_bytes = pix.tobytes("png")

                prompt = (
                    "You are analyzing a slide from a startup pitch deck. "
                    "Describe what is shown visually on this slide: charts, diagrams, images, "
                    "logos, product photos, and any text that is part of graphics (not body text). "
                    "Be concise but specific — focus on visual content that wouldn't appear in "
                    "extracted slide text. Return only the description, no preamble."
                )

                response = model.generate_content([
                    {"mime_type": "image/png", "data": img_bytes},
                    prompt,
                ])

                description = response.text.strip() if response.text else ""
                print(f"   [Page {page_num}/{total_pages}] ✓ ({len(description)} chars)")

            except Exception as e:
                description = ""
                print(f"   [Page {page_num}/{total_pages}] ⚠️  Skipped: {e}")

            descriptions.append({
                "page": page_num,
                "section": "image_description",
                "description": description,
            })

        pdf_document.close()

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(descriptions, f, indent=4, ensure_ascii=False)
        print(f"   ✅ Image descriptions saved to {output_path}")

    except fitz.FileNotFoundError:
        print(f"❌ PDF not found for image description generation: {pdf_path}")
    except Exception as e:
        print(f"❌ Image description generation failed: {e}")


def save_json(data, path):
    """Save data to JSON file."""
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4, ensure_ascii=False)
    print(f"   Saved JSON to {path}")


def run_gemini_analysis(pitchdeck_data, prompt_data):
    """Pass pitchdeck JSON and a prompt JSON to Gemini for structured analysis, save result."""
    # 1. Configure the model to enforce JSON output.
    generation_config = {
      "response_mime_type": "application/json",
    }
    model = genai.GenerativeModel(
        "gemini-2.5-flash",
        generation_config=generation_config,
    )

    # 2. Combine the detailed instructions and the pitch deck data into one prompt.
    full_prompt = f"""
    You are an AI assistant analyzing a startup pitch deck.
    Your instructions are defined in the following JSON object:
    {json.dumps(prompt_data)}

    Now, apply these instructions to the following pitch deck data:
    {json.dumps(pitchdeck_data)}

    Your response must be ONLY the final, valid JSON array as specified in the instructions.
    """
    
    try:
        response = model.generate_content(full_prompt)
        return response.text.strip()
    except Exception as e:
        print(f"❌ Gemini API call failed: {e}")
        return None


def process_all_decks(decks_dir, prompt_path, outputs_dir, with_images: bool = False):
    """
    Find and process all PDF files in a given directory.

    Args:
        decks_dir: Directory containing PDF pitch decks.
        prompt_path: Path to the prompt JSON schema file.
        outputs_dir: Directory where outputs will be saved.
        with_images: If True, generate Gemini multimodal image descriptions per page.
                     Adds ~1 Gemini API call per page. Default: False.
    """
    try:
        with open(prompt_path, "r", encoding="utf-8") as f:
            prompt_info = json.load(f)
        print(f"Loaded analysis prompt from {prompt_path}")
    except FileNotFoundError:
        print(f"❌ Prompt file not found: {prompt_path}\n   Please ensure 'prompt.json' exists in the preprocessing directory.")
        return
    except json.JSONDecodeError as e:
        print(f"❌ Prompt file is not valid JSON: {prompt_path}\n   JSON error: {e}")
        return

    try:
        pdf_files = [f for f in os.listdir(decks_dir) if f.lower().endswith('.pdf')]
    except FileNotFoundError:
        print(f"❌ Pitch decks directory not found: {decks_dir}")
        return

    if not pdf_files:
        print(f"No PDF files found in '{decks_dir}'. Please add your pitch decks there.")
        return
    
    print(f"\nFound {len(pdf_files)} pitch deck(s) to analyze.")

    for pdf_filename in pdf_files:
        deck_name = os.path.splitext(pdf_filename)[0]
        pdf_path = os.path.join(decks_dir, pdf_filename)
        print(f"\n--- Processing: {deck_name} ---")

        # Define dynamically named output paths
        parsed_json_path = os.path.join(outputs_dir, f"{deck_name}_parsed.json")
        analysis_output_path = os.path.join(outputs_dir, f"{deck_name}_analysis.json")
        images_output_dir = os.path.join(outputs_dir, f"{deck_name}_images")

        # Create a dedicated folder for this deck's images
        os.makedirs(images_output_dir, exist_ok=True)

        # Step 1: Extract text from PDF
        extracted_data = extract_text_from_pdf(pdf_path)
        if not extracted_data:
            print(f"⚠️  Skipping '{deck_name}' — could not extract text.")
            continue
        save_json(extracted_data, parsed_json_path)

        # Step 2: Extract embedded images from PDF
        extract_images_from_pdf(pdf_path, images_output_dir)

        # Step 3 (optional): Generate Gemini multimodal image descriptions per page
        if with_images:
            print("   Running image description generation (--with-images)...")
            generate_image_descriptions(pdf_path, deck_name, outputs_dir)

        # Step 4: Run Gemini analysis on the extracted text
        response_text = run_gemini_analysis(extracted_data, prompt_info)
        if not response_text:
            print(f"⚠️  Skipping analysis save for '{deck_name}' — Gemini returned no response.")
            continue

        # Step 5: Save the structured analysis output
        try:
            gemini_output = json.loads(response_text)
            save_json(gemini_output, analysis_output_path)
            print(f"   ✅ Analysis for '{deck_name}' saved as structured JSON.")
        except json.JSONDecodeError:
            error_path = analysis_output_path.replace('.json', '_error.txt')
            with open(error_path, "w", encoding="utf-8") as f:
                f.write(response_text)
            print(
                f"⚠️  Gemini's response for '{deck_name}' was not valid JSON.\n"
                f"   Raw output saved to: {error_path}\n"
                f"   This can happen when the model exceeds its output length or the response is truncated."
            )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Preprocess pitch deck PDFs: extract text, images, and run Gemini structured analysis."
    )
    parser.add_argument(
        "--with-images",
        action="store_true",
        default=False,
        help=(
            "Generate Gemini multimodal image descriptions per page (one API call per page). "
            "Outputs {deck}_image_descriptions.json for use in the RAG pipeline. "
            "Not run by default to avoid unexpected API usage."
        ),
    )
    args = parser.parse_args()

    base_dir = os.path.dirname(os.path.abspath(__file__))
    
    # --- DIRECTORY SETUP ---
    prompt_json_path = os.path.join(base_dir, "prompt.json")
    decks_directory = os.path.join(base_dir, "pitch_decks")
    outputs_directory = os.path.join(base_dir, "outputs")

    # Create directories if they don't exist
    os.makedirs(decks_directory, exist_ok=True)
    os.makedirs(outputs_directory, exist_ok=True)
    
    process_all_decks(
        decks_directory,
        prompt_json_path,
        outputs_directory,
        with_images=args.with_images,
    )
    
    print("\n--- All decks processed. ---")
