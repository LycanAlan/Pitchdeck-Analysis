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
    except Exception as e:
        print(f"Error reading PDF for text extraction {pdf_path}: {e}")
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
            print(f"Extracted {image_count} images to {images_output_dir}")
        else:
            print("No embedded images found in the PDF.")
            
    except Exception as e:
        print(f"Error extracting images from {pdf_path}: {e}")


def save_json(data, path):
    """Save data to JSON file."""
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4, ensure_ascii=False)
    print(f"Saved JSON to {path}")


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
        print(f"An error occurred during the Gemini API call: {e}")
        return None


def process_all_decks(decks_dir, prompt_path, outputs_dir):
    """Finds and processes all PDF files in a given directory using a specified prompt file."""
    try:
        with open(prompt_path, "r", encoding="utf-8") as f:
            prompt_info = json.load(f)
        print(f"Loaded analysis prompt from {prompt_path}")
    except FileNotFoundError:
        print(f"Error: Prompt file not found at {prompt_path}. Please create it.")
        return
    except json.JSONDecodeError:
        print(f"Error: The prompt file at {prompt_path} is not valid JSON.")
        return

    pdf_files = [f for f in os.listdir(decks_dir) if f.lower().endswith('.pdf')]
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
            continue
        save_json(extracted_data, parsed_json_path)

        # Step 2: Extract images from PDF
        extract_images_from_pdf(pdf_path, images_output_dir)

        # Step 3: Run Gemini analysis on the extracted text
        response_text = run_gemini_analysis(extracted_data, prompt_info)
        if not response_text:
            continue

        # Step 4: Save the structured analysis output
        try:
            gemini_output = json.loads(response_text)
            save_json(gemini_output, analysis_output_path)
            print(f"Analysis for '{deck_name}' saved as structured JSON.")
        except json.JSONDecodeError:
            error_path = analysis_output_path.replace('.json', '_error.txt')
            with open(error_path, "w", encoding="utf-8") as f:
                f.write(response_text)
            print(f"Gemini response for '{deck_name}' was not valid JSON; raw output saved to {error_path}")


if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.abspath(__file__))
    
    # --- DIRECTORY SETUP ---
    # The prompt file is now expected in the main project directory.
    prompt_json_path = os.path.join(base_dir, "prompt.json")
    decks_directory = os.path.join(base_dir, "pitch_decks")
    outputs_directory = os.path.join(base_dir, "outputs")

    # Create directories if they don't exist
    os.makedirs(decks_directory, exist_ok=True)
    os.makedirs(outputs_directory, exist_ok=True)
    
    process_all_decks(decks_directory, prompt_json_path, outputs_directory)
    
    print("\n--- All decks processed. ---")

