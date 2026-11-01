import os
import re
import easyocr

def init_ocr_reader(lang_list=['en'], gpu=False):
    """Initializes the EasyOCR reader."""
    try:
        print("Initializing EasyOCR reader...")
        return easyocr.Reader(lang_list, gpu=gpu)
    except Exception as e:
        print(f"[Error] Failed to initialize EasyOCR: {e}")
        return None

def sanitize_ocr_text(text: str) -> str:
    """
    Fixes common OCR formatting glitches like missing dots in email domains
    (e.g., converts 'hr@richmondedu' -> 'hr@richmond.edu').
    """
    if not text:
        return ""
    # OCR often reads the dot before the TLD as a space: 'hr@gaotek com' -> 'hr@gaotek.com'
    text = re.sub(
        r'@([a-zA-Z0-9-]+) (edu|com|org|net|gov)\b',
        r'@\1.\2',
        text,
        flags=re.IGNORECASE
    )
    # Regex: Inserts missing dot before standard TLDs if OCR drops it
    text = re.sub(
        r'@([a-zA-Z0-9-]+)(edu|com|org|net|gov)\b',
        r'@\1.\2',
        text,
        flags=re.IGNORECASE
    )
    # ... and a space before the '@': 'hr @gaotek.com' -> 'hr@gaotek.com'
    text = re.sub(r'\b([A-Za-z0-9._%+-]+) @([a-zA-Z0-9-]+\.)', r'\1@\2', text)
    # 'www GAOTek com' / 'WWW.GAOTek com' -> 'www.GAOTek.com'
    text = re.sub(
        r'\b[wW]{3}[ .]([a-zA-Z0-9-]+)[ .](edu|com|org|net|gov)\b',
        r'www.\1.\2',
        text,
        flags=re.IGNORECASE
    )
    return text

SMALL_IMAGE_WIDTH = 1000  # WhatsApp/phone-compressed letters are often ~768 px wide


def _upscale_small(image_path: str):
    """
    Double the size of narrow images before OCR. On a 768x1024 WhatsApp letter this recovered
    '4 calendar days' and 'www GAOTek com', which EasyOCR misread at the original size.
    Returns a numpy array (or the path unchanged if the image is already wide enough).
    """
    import numpy as np
    from PIL import Image

    with Image.open(image_path) as im:
        if im.width >= SMALL_IMAGE_WIDTH:
            return image_path
        im = im.convert("RGB")
        return np.array(im.resize((im.width * 2, im.height * 2), Image.LANCZOS))


def extract_text_from_image(reader, image_path: str) -> str:
    """Runs OCR on the given image path and returns the extracted text."""
    if not reader:
        return ""
        
    if not os.path.exists(image_path):
        print(f"[Warning] Image not found at path: {image_path}")
        return ""
    
    try:
        print(f"Running OCR on {image_path}...")
        results = reader.readtext(_upscale_small(image_path))
        
        # Combine all detected text
        full_text = "\n".join(
            text for bounding_box, text, confidence in results
        )
        
        # Clean up common OCR typos (like missing dots in emails) before returning
        return sanitize_ocr_text(full_text)
        
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