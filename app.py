import streamlit as st
from ocr_pipeline import ocr_image

st.title("Handwrite OCR")
uploaded_file = st.file_uploader("Upload an image", type=["png","jpg","jpeg"])

if uploaded_file:
    text = ocr_image(uploaded_file)
    st.text_area("Detected text", value=text, height=300)
