import streamlit as st
from ocr_pipeline import ocr_image_with_boxes
from PIL import Image
import numpy as np

st.set_page_config(page_title="Handwriting OCR", layout="wide")

st.markdown("""
    <style>
    .main-header {
        font-size: 3.5rem;
        font-weight: 700;
        margin-bottom: 0.2rem;
        color: #1a1a1a;
    }
    .subtitle {
        color: #666;
        font-size: 1.3rem;
        margin-bottom: 2.5rem;
        font-weight: 300;
    }
    .upload-section {
        background: #f8f9fa;
        padding: 2rem;
        border-radius: 12px;
        border: 2px dashed #ddd;
    }
    .output-section {
        background: #ffffff;
        padding: 2rem;
        border-radius: 12px;
        border: 1px solid #e0e0e0;
    }
    .stTextArea textarea {
        font-family: 'Courier New', monospace;
        font-size: 0.95rem;
    }
    </style>
    """, unsafe_allow_html=True)

st.markdown('<p class="main-header">handwriting reader</p>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">turn your scribbles into text</p>', unsafe_allow_html=True)

col1, col2 = st.columns([1.2, 1.5], gap="large")

with col1:
    st.markdown("### your image")
    uploaded_file = st.file_uploader(
        "drop or click to upload", 
        type=["png","jpg","jpeg"], 
        label_visibility="collapsed"
    )
    
    if uploaded_file:
        image = Image.open(uploaded_file)
        st.image(image, use_container_width=True, caption="original")
    else:
        st.markdown("""
            <div style='text-align: center; padding: 3rem; color: #999;'>
                <p style='font-size: 1.2rem;'>no image yet</p>
                <p>upload something to get started</p>
            </div>
        """, unsafe_allow_html=True)

with col2:
    st.markdown("### results")
    
    if uploaded_file:
        with st.spinner("reading..."):
            text, annotated_image = ocr_image_with_boxes(uploaded_file)
        
        tabs = st.tabs(["text output", "detection view"])
        
        with tabs[0]:
            st.text_area(
                "extracted text", 
                value=text, 
                height=400, 
                label_visibility="collapsed",
                placeholder="extracted text will appear here"
            )
            
            word_count = len(text.split())
            char_count = len(text)
            st.caption(f"{word_count} words · {char_count} characters")
        
        with tabs[1]:
            if annotated_image is not None:
                st.image(annotated_image, use_container_width=True, caption="detected regions")
    else:
        st.markdown("""
            <div style='text-align: center; padding: 3rem; color: #999;'>
                <p style='font-size: 1.2rem;'>waiting for input</p>
                <p>upload an image to see the magic</p>
            </div>
        """, unsafe_allow_html=True)

st.markdown("---")
st.caption("built with easyocr · works best with clear handwriting")