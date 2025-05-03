import os
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "gemini-key.json"

import vertexai
from vertexai.preview.generative_models import GenerativeModel, Part
from PIL import Image
import io
import mimetypes
import fitz

vertexai.init(project="gdg-hackathon-458610", location="")

# def preprocess(file_path):
#     """
#     Takes a file path (PDF or image) and returns a list of PIL Image objects.
#     """
#     mime_type, _ = mimetypes.guess_type(file_path)
#
#     if mime_type == 'application/pdf':
#         return convert_pdf_to_images(file_path)
#     elif mime_type and mime_type.startswith('image/'):
#         return [Image.open(file_path).convert("RGB")]
#     else:
#         raise ValueError(f"Unsupported file type: {mime_type}")

def convert_pdf_to_images(file, dpi=100):
    images = []
    pdf_bytes = file.read()  # Read the uploaded file
    file.seek(0)
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")

    for page in doc:
        pix = page.get_pixmap(dpi=dpi)
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        images.append(img)

    return images

# from gemini import generate
#
# images = preprocess("IMG_7710.jpeg")
# prompt = "Describe the picture"
#
# for i, img in enumerate(images):
#     print(f"--- Page {i+1} ---")
#     print(generate(img, prompt))