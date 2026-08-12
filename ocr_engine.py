import os
import easyocr

def init_ocr_reader(lang_list=['en'], gpu=False):
    """Initializes the EasyOCR reader."""
    try:
        print("Initializing EasyOCR reader...")
        return easyocr.Reader(lang_list, gpu=gpu)
    except Exception as e:
        print(f"[Error] Failed to initialize EasyOCR: {e}")
        return None

def extract_text_from_image(reader, image_path: str) -> str:
    """Runs OCR on the given image path and returns the extracted text."""
    if not reader:
        return ""
        
    if not os.path.exists(image_path):
        print(f"[Warning] Image not found at path: {image_path}")
        return ""
    
    try:
        print(f"Running OCR on {image_path}...")
        results = reader.readtext(image_path)
        
        # Combine all detected text
        full_text = "\n".join(
            text for bounding_box, text, confidence in results
        )
        return full_text
    except Exception as e:
        print(f"[Error] Failed to process image {image_path}: {e}")
        return ""

def merge_text_and_ocr(user_text: str, ocr_text: str) -> str:
    """Merges user text and OCR text with source tags."""
    merged = ""
    if user_text:
        merged += f"[SOURCE: RAW_TEXT]\n{user_text.strip()}\n\n"
    if ocr_text:
        merged += f"[SOURCE: OCR_IMAGE]\n{ocr_text.strip()}\n"
    return merged.strip()

if __name__ == "__main__":
    pass
