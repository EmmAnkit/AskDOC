# gemini.py
from vertexai.preview.generative_models import GenerativeModel, Part
from PIL import Image
from io import BytesIO
import io
import re

import os
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "gemini-key.json"


def generate_with_image(image, prompt, model_name="gemini-2.0-flash", transpose_bbox=False):
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    img_bytes = buffer.getvalue()

    model = GenerativeModel(model_name)
    inputs = [
        Part.from_data(data=img_bytes, mime_type="image/png"),
        f"""
You are given a scanned document image. Your task is:
1. Answer the user's prompt.
2. If the answer corresponds to a visible region in the image, return its bounding box.
3. The bounding box must be in this format:
   Box: [x0, y0, x1, y1]
   - All coordinates are in a normalized 1000x1000 space (not pixels).
   - (0,0) is the top-left of the image.
   - x-axis goes left to right; y-axis goes top to bottom.
   - The box should tightly enclose the relevant content.

Prompt: {prompt}
"""
    ]

    response = model.generate_content(inputs)
    text = response.text.strip()

    # Extract
    answer_match = re.search(r"Answer:\s*(.*?)\nBox:", text, re.DOTALL)
    box_match = re.search(r"Box:\s*\[([\d., ]+)\]", text)

    answer = answer_match.group(1).strip() if answer_match else text
    bbox = None

    if box_match:
        coords = list(map(float, box_match.group(1).split(",")))
        if len(coords) == 4:
            x0, y0, x1, y1 = coords
            if transpose_bbox:
                # Optional transpose if the model returns coordinates in the wrong orientation
                bbox = (x0, y0, x1, y1)
            else:
                bbox = (y0, x0, y1, x1)
            return answer, True, bbox

    return answer, False, None

def generate(image: Image.Image, prompt: str, model_name="gemini-2.0-flash") -> str:
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    img_bytes = buffer.getvalue()

    model = GenerativeModel(model_name)

    inputs = [
        Part.from_data(data=img_bytes, mime_type="image/png"),
        prompt
    ]

    response = model.generate_content(inputs)
    return response.text

def generate_text_only(prompt, model_name="gemini-2.0-flash"):
    model = GenerativeModel(model_name)
    response = model.generate_content(prompt)
    return response.text.strip()

def is_handwritten(image, model_name="gemini-2.0-flash"):
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    img_bytes = buffer.getvalue()

    model = GenerativeModel(model_name)
    inputs = [
        Part.from_data(data=img_bytes, mime_type="image/png"),
        "Is the content in this image handwritten or printed? Just reply: handwritten or printed."
    ]

    response = model.generate_content(inputs)
    reply = response.text.strip().lower()

    if "handwritten" in reply:
        return True
    return False