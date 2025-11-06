import cv2
import numpy as np
import streamlit as st
from PIL import Image, ImageEnhance, ImageOps
import io

st.set_page_config(page_title="DocScan – Streamlit", layout="wide")

st.title("📄 DocScan – Streamlit Version")
st.write("Upload or capture a document, auto-detect edges, crop, enhance, and export!")

# --- Upload or Capture ---
col1, col2 = st.columns(2)
img_file = col1.file_uploader("Upload an Image", type=["jpg", "jpeg", "png"])
use_camera = col2.checkbox("📷 Use Camera")

if use_camera:
    camera_img = st.camera_input("Take a photo")
else:
    camera_img = None

image = None
if img_file is not None:
    image = Image.open(img_file).convert("RGB")
elif camera_img is not None:
    image = Image.open(camera_img).convert("RGB")

if image is not None:
    st.image(image, caption="Original Image", use_container_width=True)

    # --- Convert to OpenCV ---
    img_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

    # --- Auto-detect document edges ---
    if st.button("🪄 Auto Detect Document"):
        gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (5,5), 0)
        edged = cv2.Canny(gray, 50, 150)
        contours, _ = cv2.findContours(edged, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        contours = sorted(contours, key=cv2.contourArea, reverse=True)
        doc_cnt = None
        for c in contours:
            peri = cv2.arcLength(c, True)
            approx = cv2.approxPolyDP(c, 0.02 * peri, True)
            if len(approx) == 4:
                doc_cnt = approx
                break
        if doc_cnt is not None:
            pts = doc_cnt.reshape(4, 2)
            rect = np.zeros((4, 2), dtype="float32")
            s = pts.sum(axis=1)
            rect[0] = pts[np.argmin(s)]
            rect[2] = pts[np.argmax(s)]
            diff = np.diff(pts, axis=1)
            rect[1] = pts[np.argmin(diff)]
            rect[3] = pts[np.argmax(diff)]

            (tl, tr, br, bl) = rect
            widthA = np.linalg.norm(br - bl)
            widthB = np.linalg.norm(tr - tl)
            heightA = np.linalg.norm(tr - br)
            heightB = np.linalg.norm(tl - bl)
            maxW = int(max(widthA, widthB))
            maxH = int(max(heightA, heightB))
            dst = np.array([[0,0],[maxW-1,0],[maxW-1,maxH-1],[0,maxH-1]], dtype="float32")
            M = cv2.getPerspectiveTransform(rect, dst)
            warped = cv2.warpPerspective(img_cv, M, (maxW, maxH))
            image = Image.fromarray(cv2.cvtColor(warped, cv2.COLOR_BGR2RGB))
            st.success("Document Detected and Cropped!")
        else:
            st.warning("No document detected. Please crop manually.")
    else:
        warped = img_cv

    # --- Filters ---
    st.subheader("🎨 Enhancement Filters")
    mode = st.selectbox(
        "Choose Enhancement",
        ["Original", "Grayscale", "Clean B&W", "Magic Color", "Warm Tone", "Cool Tone", "Sketch"]
    )

    def enhance(img_pil, mode):
        if mode == "Original":
            return img_pil
        elif mode == "Grayscale":
            return ImageOps.grayscale(img_pil).convert("RGB")
        elif mode == "Clean B&W":
            img_cv = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2GRAY)
            bw = cv2.adaptiveThreshold(img_cv, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                       cv2.THRESH_BINARY, 21, 15)
            return Image.fromarray(bw).convert("RGB")
        elif mode == "Magic Color":
            img_cv = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
            lab = cv2.cvtColor(img_cv, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
            cl = clahe.apply(l)
            merged = cv2.merge([cl,a,b])
            colored = cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)
            sharp = cv2.addWeighted(colored, 1.4, cv2.GaussianBlur(colored, (0,0), 1.0), -0.4, 0)
            return Image.fromarray(cv2.cvtColor(sharp, cv2.COLOR_BGR2RGB))
        elif mode == "Warm Tone":
            r,g,b = img_pil.split()
            r = r.point(lambda i: min(255, i*1.1))
            b = b.point(lambda i: i*0.9)
            return Image.merge('RGB', (r,g,b))
        elif mode == "Cool Tone":
            r,g,b = img_pil.split()
            r = r.point(lambda i: i*0.9)
            b = b.point(lambda i: min(255, i*1.1))
            return Image.merge('RGB', (r,g,b))
        elif mode == "Sketch":
            gray = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2GRAY)
            inv = 255 - gray
            blur = cv2.GaussianBlur(inv, (21,21), 0)
            sketch = cv2.divide(gray, 255 - blur, scale=256)
            return Image.fromarray(sketch).convert("RGB")
        return img_pil

    enhanced = enhance(image, mode)
    st.image(enhanced, caption=f"Enhanced: {mode}", use_container_width=True)

    # --- Export to PDF ---
    if st.button("📄 Export as PDF"):
        buf = io.BytesIO()
        enhanced.save(buf, "PDF", resolution=300)
        st.download_button(
            label="Download PDF",
            data=buf.getvalue(),
            file_name="scanned_document.pdf",
            mime="application/pdf"
        )

