import os
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "gemini-key.json"

import streamlit as st
from PIL import Image, ImageDraw
from fileLoader import convert_pdf_to_images
from gemini import generate_text_only, generate_with_image
import fitz  # PyMuPDF
import shutil, subprocess, tempfile
from pathlib import Path
from io import BytesIO


st.set_page_config(page_title="AskDOC", layout="centered")

if "submitted" not in st.session_state:
    st.session_state.submitted = False

# --- UI Styling ---
st.markdown(""" <style>
.title { font-size: 98px; font-weight: 700; text-align: center; margin-top: 1rem; margin-bottom: 4rem; }
.title span { color: #2a7dd4; }
.subtitle { font-size: 20px; text-align: center; margin-bottom: 1.5rem; color: #444; }
.stTextInput > div > input {
    height: 42px;
    border-radius: 999px;
    font-size: 16px;
    padding: 0.6rem 1rem;
}
.stButton > button {
    height: 42px !important;
    padding: 0px 14px;
    border-radius: 999px;
    font-size: 20px;
}
</style>""", unsafe_allow_html=True)

st.markdown('<div class="title">Ask<span>DOC</span></div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">How can I help you today?</div>', unsafe_allow_html=True)

st.markdown(""" <style>
.stTextInput > div > input {
    height: 48px;
    border-radius: 999px;
    font-size: 18px;
    padding: 0.75rem 1.2rem;
    width: 100% !important;
} </style>""", unsafe_allow_html=True)


with st.form(key="ask_form", clear_on_submit=False):

    col1, col2, col3 = st.columns([6, 0.5, 0.5])

    with col1:

        prompt = st.text_input(" ", placeholder="Enter Prompt", label_visibility="collapsed", key="prompt_input")

    with col2:

        ask_clicked = st.form_submit_button("🔍", help="Ask")

    with col3:

        reset_clicked = st.form_submit_button("🔄", help="Next Prompt")

# Label
st.markdown("**Upload Documents**", unsafe_allow_html=True)

st.markdown(""" <style>
.stFileUploader {
    width: 50% !important;
    margin: -3rem auto 0 auto !important;
} </style>""", unsafe_allow_html=True)

uploaded_file = st.file_uploader("", type=["pdf", "png", "jpg", "jpeg", "docx", 'pptx'])

if reset_clicked:
    st.session_state.prompt = ""
    st.session_state.file = None
    st.session_state.submitted = False
    st.experimental_set_query_params(reset="1")
    st.rerun()

# --- Core Logic ---

def extract_blocks_with_location_from_pdf(file):
    file.seek(0)
    doc = fitz.open(stream=file.read(), filetype="pdf")
    blocks_with_location = []
    for i, page in enumerate(doc):
        page_width, page_height = page.rect.width, page.rect.height
        blocks = page.get_text("blocks")
        for block in blocks:
            x0, y0, x1, y1, text, *_ = block
            if text.strip():
                norm_x0 = x0 / page_width * 1000
                norm_y0 = y0 / page_height * 1000
                norm_x1 = x1 / page_width * 1000
                norm_y1 = y1 / page_height * 1000
                blocks_with_location.append({
                    "page": i + 1,
                    "bbox": (norm_x0, norm_y0, norm_x1, norm_y1),
                    "text": text.strip()
                })
    return blocks_with_location

def office_to_pdf(file_obj):
    """
    Convert DOC(X)/PPT(X/M) -> PDF with *no* MS-Office dependency.
    Requires LibreOffice ≥ 6.0 in PATH (soffice) or pypandoc as a fallback.
    Returns raw PDF bytes.
    """
    suffix = Path(file_obj.name).suffix.lower()
    if suffix not in {".doc", ".docx", ".ppt", ".pptx", ".pptm"}:
        raise RuntimeError("Unsupported Office format")

    with tempfile.TemporaryDirectory() as tmp:
        in_path  = Path(tmp) / file_obj.name
        out_path = in_path.with_suffix(".pdf")
        in_path.write_bytes(file_obj.read())        # save upload

        soffice = shutil.which("soffice") or shutil.which("libreoffice")
        if soffice:                                 # ▶ 1️⃣  LibreOffice branch
            cmd = [
                soffice, "--headless", "--invisible",
                "--convert-to", "pdf", "--outdir", tmp, str(in_path)
            ]
            result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            if result.returncode != 0 or not out_path.exists():
                raise RuntimeError(f"LibreOffice failed:\n{result.stderr.decode()}")
            return out_path.read_bytes()

        # ▶ 2️⃣  Fallback: pypandoc (good for plain DOCX)
        try:
            import pypandoc
        except ImportError:  # no LibreOffice *and* no pandoc
            raise RuntimeError("Install LibreOffice or pypandoc for conversion")

        try:
            pdf_path = Path(tmp) / "pandoc_out.pdf"
            pypandoc.convert_file(str(in_path), to="pdf", outputfile=str(pdf_path))
            return pdf_path.read_bytes()
        except RuntimeError as e:
            raise RuntimeError(f"Pandoc conversion failed: {e}")


def annotate_handwritten(image, bbox, label=""):
    width, height = image.size
    # Assume bbox is in normalized PDF space (0–1000)
    x0 = bbox[0] / 1000 * width
    y0 = bbox[1] / 1000 * height
    x1 = bbox[2] / 1000 * width
    y1 = bbox[3] / 1000 * height

    draw = ImageDraw.Draw(image)
    draw.rectangle([x0, y0, x1, y1], outline="blue", width=4)
    if label:
        draw.text((x0, y0 - 15), label, fill="blue")
    return image

def annotate_image(image, bbox):
    # Pillow returns (width, height)
    width, height = image.size        # ❶ keep the right order

    # bbox = (x0, y0, x1, y1) in 0-1000 PDF space
    x0 = bbox[0] / 1000 * width       # ❷ map X coords with width
    y0 = bbox[1] / 1000 * height
    x1 = bbox[2] / 1000 * width
    y1 = bbox[3] / 1000 * height

    draw = ImageDraw.Draw(image)
    draw.rectangle([x0, y0, x1, y1], outline="red", width=4)
    return image
def build_gemini_block_prompt(user_prompt, blocks):
    formatted_blocks = "\n".join([
        f"[{i}] (Page {b['page']}) \"{b['text']}\"" for i, b in enumerate(blocks)
    ])
    return f"""User Question: {user_prompt}

Blocks:
{formatted_blocks}

Instructions:

* Answer the user’s question.
* Then return the index of the block that supports the answer.
* Format:
  Answer: ...
  Block Index: ...
"""

def parse_gemini_response(response):
    import re
    answer_match = re.search(r"Answer:\s*(.*?)\nBlock Index:", response, re.DOTALL)
    index_match = re.search(r"Block Index:\s*(\d+)", response)
    if answer_match and index_match:
        return answer_match.group(1).strip(), int(index_match.group(1))
    return response.strip(), None


if ask_clicked and prompt and uploaded_file:
    st.session_state.submitted = True
    st.session_state.prompt = prompt
    st.session_state.file = uploaded_file

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".gif", ".tif", ".tiff"}
OFFICE_EXTS = {".docx", ".doc", ".pptx", ".ppt", ".pptm"}
flag = False

if st.session_state.submitted:
    with st.spinner("Processing your document..."):
        uploaded = st.session_state.file
        uploaded.seek(0)
        suffix = Path(uploaded.name).suffix.lower()


        if suffix in OFFICE_EXTS:
            st.info("🔄 Converting Office file to PDF…")
            try:
                file_bytes = office_to_pdf(uploaded)
                uploaded_type = "application/pdf"  # pretend it was a PDF all along
                st.success("✅ Converted")
                suffix = ".pdf"
                flag = True
            except RuntimeError as err:
                st.error(f"Conversion failed: {err}")
                st.stop()
        if flag:
            uploaded = file_bytes
        else:
            uploaded = uploaded.read()

        if suffix == ".pdf":
            from gemini import is_handwritten
            if not flag:
                file_bytes = uploaded

            images = convert_pdf_to_images(BytesIO(uploaded))

            is_hw_pdf = any(is_handwritten(img) for img in images[:1])  # check 1st page

            if is_hw_pdf:
                st.info("🖋️ Detected handwritten PDF.")

                best_result = None
                for i, img in enumerate(images):
                    phrased_answer, should_highlight, bbox = generate_with_image(img, st.session_state.prompt)
                    print(f"Page {i+1}: highlight={should_highlight}, bbox={bbox}")  # optional debug

                    if should_highlight and bbox:
                        best_result = {
                            "page": i + 1,
                            "answer": phrased_answer,
                            "image": annotate_handwritten(img.copy(), bbox),
                            "bbox": bbox
                        }
                        # Don't break — keep latest valid

                if best_result:
                    st.markdown(f"## 📌 Answer\n{best_result['answer']}")
                    st.markdown(f"**📄 Source:** Page {best_result['page']}\n**📍 Coordinates:** {best_result['bbox']}")
                    st.image(best_result["image"], caption=f"Page {best_result['page']}")
                else:
                    st.warning("No answer with bounding box found in the handwritten PDF.")

            else:
                st.info("🖨️ Detected printed PDF.")

                blocks = extract_blocks_with_location_from_pdf(BytesIO(file_bytes))
                gemini_prompt = build_gemini_block_prompt(st.session_state.prompt, blocks)
                gemini_output = generate_text_only(gemini_prompt)
                phrased_answer, block_index = parse_gemini_response(gemini_output)

                if block_index is not None and 0 <= block_index < len(blocks):
                    best_block = blocks[block_index]
                    annotated = annotate_image(images[best_block['page'] - 1], best_block['bbox'])
                    st.markdown(f"## 📌 Answer\n{phrased_answer}")
                    st.markdown(f"**📄 Source:** Page {best_block['page']}\n\n**📍 Coordinates:** {best_block['bbox']}")
                    st.image(annotated, caption="Location in document")
                else:
                    st.markdown(f"## 📌 Answer\n{phrased_answer}")
                    st.warning("Could not identify an exact block, but answer is shown above.")

        elif suffix in IMAGE_EXTS:
            image = Image.open(BytesIO(uploaded)).convert("RGB")
            from gemini import is_handwritten

            is_hw = is_handwritten(image)
            st.info("🖋️ Detected handwritten input." if is_hw else "🖨️ Detected printed input.")

            phrased_answer, should_highlight, bbox = generate_with_image(image, st.session_state.prompt)
            st.markdown(f"## 📌 Answer\n{phrased_answer}")

            if should_highlight and bbox:
                if is_hw:
                    annotated = annotate_handwritten(image.copy(), bbox)
                else:
                    annotated = annotate_image(image.copy(), bbox)
                st.image(annotated, caption="Highlighted in image")
            else:
                st.image(image, caption="Uploaded image")




    click = False

    c1, c2, c3 = st.columns([2, 1.5, 2])  # middle column twice as wide
    with c2:
        click = st.button(
            "Further Analysis",
            key="further_analysis")  # no if-statement need

    if click:
        import FurtherAnal
        FurtherAnal.main(prompt, uploaded_file)
        # st.session_state.prompt = ""
        # st.session_state.file = None
        # st.session_state.submitted = False
        # st.experimental_set_query_params(reset="1")
        # st.rerun()
