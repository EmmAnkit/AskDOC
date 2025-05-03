import os
import io
import mimetypes
import re
import json
import pandas as pd
from PIL import Image, ImageDraw
import fitz  # PyMuPDF
from vertexai.preview.generative_models import GenerativeModel, Part
import vertexai
import streamlit as st


# --- 1. Setup ---
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "gemini-key.json"
vertexai.init(project="gdg-hackathon-458610", location="us-central1")


# --- 2. Preprocessing ---
def preprocess(file_path):
    mime_type, _ = mimetypes.guess_type(file_path)
    if mime_type == 'application/pdf':
        return convert_pdf_to_images(file_path)
    elif mime_type and mime_type.startswith('image/'):
        return [Image.open(file_path).convert("RGB")]
    else:
        raise ValueError(f"Unsupported file type: {mime_type}")


def convert_pdf_to_images(pdf_path, dpi=100):
    doc = fitz.open(pdf_path)
    images = []
    for page in doc:
        pix = page.get_pixmap(dpi=dpi)
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        images.append(img)
    return images


# --- 3. Gemini Image + Prompt ---
def generate_gemini_response(image: Image.Image, prompt: str, model_name="gemini-2.0-flash") -> str:
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    img_bytes = buffer.getvalue()

    model = GenerativeModel(model_name)
    inputs = [Part.from_data(data=img_bytes, mime_type="image/png"), prompt]
    response = model.generate_content(inputs)
    return response.text


# --- 4. Improved JSON Extraction with Error Handling ---
def extract_df_from_gemini_response(response_text: str, image_size: tuple) -> pd.DataFrame:
    # Check for error responses first
    error_phrases = [
        "Sorry, I'm unable to find those objects",
        "no bounding boxes",
        "no objects detected",
        "unable to locate"
    ]
    if any(phrase.lower() in response_text.lower() for phrase in error_phrases):
        return pd.DataFrame(columns=["text", "box_2d"])

    # Try multiple JSON extraction patterns
    json_patterns = [
        r"```json\s*(\[\s*{.*?}\s*])\s*```",  # Standard ```json ``` block
        r"\[\s*{.*?}\s*]",  # Bare JSON array
        r"{.*?}"  # Single JSON object
    ]

    json_data = None
    for pattern in json_patterns:
        match = re.search(pattern, response_text, re.DOTALL)
        if match:
            try:
                json_data = json.loads(match.group(0))
                if isinstance(json_data, dict):  # If single object, wrap in list
                    json_data = [json_data]
                break
            except json.JSONDecodeError:
                continue

    if not json_data:
        return pd.DataFrame(columns=["text", "box_2d"])

    width, height = image_size
    scale_x = width / 1000
    scale_y = height / 1000
    processed_data = []

    for d in json_data:
        try:
            # Handle different possible field names for bounding box
            box_key = next(
                (key for key in ["box_2d", "bounding_box", "bbox", "coordinates"]
                 if key in d), None)

            if not box_key:
                continue

            # Get coordinates - handle different formats
            if isinstance(d[box_key], str):
                # If coordinates are in string format like "x0,y0,x1,y1"
                coords = list(map(float, d[box_key].split(',')))
                x0, y0, x1, y1 = coords[:4]
            else:
                # Assume it's a list [x0, y0, x1, y1]
                x0, y0, x1, y1 = d[box_key][:4]

            # Handle different possible field names for text/label
            text = d.get("text", d.get("label", d.get("content", "")))

            # Transpose and scale coordinates
            y0, x0 = x0, y0  # Transpose
            y1, x1 = x1, y1  # Transpose

            x0 *= scale_x
            y0 *= scale_y
            x1 *= scale_x
            y1 *= scale_y

            processed_data.append({
                "text": text,
                "box_2d": [x0, y0, x1, y1]
            })

        except Exception as e:
            st.warning(f"Skipping invalid bounding box data: {e}")
            continue

    return pd.DataFrame(processed_data, columns=["text", "box_2d"])


# --- 5. Draw Boxes on Image ---
def draw_boxes(image: Image.Image, df: pd.DataFrame) -> Image.Image:
    if df.empty:
        return image

    draw = ImageDraw.Draw(image)
    for _, row in df.iterrows():
        try:
            box = row["box_2d"]
            if len(box) != 4:
                continue

            draw.rectangle(box, outline="red", width=3)
            draw.text((box[0], box[1] - 10), row["text"], fill="red")
        except Exception as e:
            st.warning(f"Could not draw box for: {row['text']} - {e}")
    return image


# --- 6. Answer Prompt for Entire Document ---
def answer_prompt_for_document(images, prompt):
    full_response = ""
    locations = []

    for i, img in enumerate(images):
        buffer = io.BytesIO()
        img.save(buffer, format="PNG")
        img_bytes = buffer.getvalue()

        model = GenerativeModel("gemini-2.0-flash")
        inputs = [Part.from_data(data=img_bytes, mime_type="image/png"),
                  f"{prompt}\nProvide a detailed answer first, then if locating specific elements, "
                  "provide their page numbers and coordinates in JSON format with bounding boxes."]

        response = model.generate_content(inputs)
        page_response = response.text

        # Extract locations if this is a "locate" type query
        if "locate" in prompt.lower() or "find" in prompt.lower():
            if "page" in page_response.lower() and (
                    "coordinate" in page_response.lower() or "position" in page_response.lower()):
                locations.append(f"Page {i + 1}: {page_response}")

        full_response += f"\n\n--- Page {i + 1} ---\n{page_response}"

    return full_response, locations


# --- 7. Full Pipeline ---
def process_and_annotate(uploaded_file, prompt: str):
    temp_file_path = f"temp_{uploaded_file.name}"
    with open(temp_file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    try:
        images = preprocess(temp_file_path)
        results = []

        # # First answer the prompt for the entire document
        # with st.spinner("Reading the Document"):
        #     document_answer, locations = answer_prompt_for_document(images, prompt)
        #
        #     st.subheader("Document Analysis Results")
        #     if locations:
        #         st.write("\n".join(locations))
        #     else:
        #         st.write(document_answer)

        st.divider()
        st.subheader("Annotations")

        # Then process each page for annotations
        for i, img in enumerate(images):
            with st.expander(f"Page {i + 1}", expanded=False):
                with st.spinner(f"Processing page {i + 1}..."):
                    annotation_prompt = (
                        f"Locate and annotate the relevant parts for: {prompt}\n"
                        "Return ONLY a JSON array with each object containing: "
                        "'text' (the content) and 'box_2d' (bounding box coordinates [x0,y0,x1,y1] in 1000x1000 space)"
                    )

                    response = generate_gemini_response(img, annotation_prompt)
                    df = extract_df_from_gemini_response(response, img.size)
                    annotated_img = draw_boxes(img.copy(), df)

                    st.image(annotated_img, caption=f"Annotated Page {i + 1}", use_container_width=True)

                    if not df.empty:
                        st.subheader("Extracted Data:")
                        st.dataframe(df)

                    results.append({
                        "page": i + 1,
                        "annotated": annotated_img,
                        "response": response,
                        "data": df
                    })

        return results
    finally:
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)


# --- 8. Streamlit App ---
def main(prompt, uploaded_file):
    # # --- Setup & Styling ---
    # st.set_page_config(page_title="AskDOC", layout="centered")

    # st.markdown("""<style>
    # .title { font-size: 98px; font-weight: 700; text-align: center; margin-top: 1rem; margin-bottom: 4rem; }
    # .title span { color: #2a7dd4; }
    # .subtitle { font-size: 20px; text-align: center; margin-bottom: 1.5rem; color: #444; }
    # .stTextInput > div > input { height: 42px; border-radius: 999px; font-size: 16px; padding: 0.6rem 1rem; }
    # .stButton > button { height: 42px !important; padding: 0px 14px; border-radius: 999px; font-size: 20px; }
    # .upload-label { font-weight: bold; margin-bottom: 0.5rem; }
    # .result-card { background: #f8f9fa; border-radius: 12px; padding: 1.5rem; margin-top: 1rem; }
    # </style>""", unsafe_allow_html=True)

    # # --- Header ---
    # st.markdown('<div class="title">Ask<span>DOC</span></div>', unsafe_allow_html=True)
    # st.markdown('<div class="subtitle">How can I help you today?</div>', unsafe_allow_html=True)

    # # --- Search Bar ---
    # col1, col2, col3 = st.columns([6, 0.5, 0.5])
    # with col1:
    #     prompt = st.text_input(" ", placeholder="Enter Prompt", label_visibility="collapsed")
    # with col2:
    #     ask_clicked = st.button("🔍", help="Ask")
    # with col3:
    #     reset_clicked = st.button("🔄", help="Reset")

    # # --- File Upload ---
    # st.markdown("**Upload Documents**", unsafe_allow_html=True)
    # uploaded_file = st.file_uploader("", type=["pdf", "png", "jpg", "jpeg"], label_visibility="collapsed")

    # # --- Reset Logic ---
    # if reset_clicked:
    #     st.session_state.clear()
    #     st.rerun()

    # --- Processing ---
    if True:
        with st.spinner("Further Analyzing document, May take a minute"):
            try:
                results = process_and_annotate(uploaded_file, prompt)

                # --- Results Display ---
                st.markdown('<div class="result-card">', unsafe_allow_html=True)
                # st.markdown("## 📌 Document Analysis")
                #
                # # Show document-level answer first
                # if results and hasattr(results[0], 'document_answer'):
                #     st.write(results[0].document_answer)
                #
                # # Then show annotations
                # st.markdown("---")
                # st.markdown("## 🔍  uyujnkAnnotations")
                # for result in results:
                #     with st.expander(f"📄 Page {result['page']}", expanded=False):
                #         st.image(result["annotated"], use_container_width=True)
                #         if not result["data"].empty:
                #             st.dataframe(result["data"])
                #         st.markdown("**Full Response:**")
                #         st.write(result["response"])

                st.markdown('</div>', unsafe_allow_html=True)

            except Exception as e:
                st.error(f"Error processing document: {str(e)}")

