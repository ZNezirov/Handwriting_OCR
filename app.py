import streamlit as st
from ocr_pipeline import ocr_image
from PIL import Image

st.set_page_config(page_title="donut ocr", layout="wide")

st.title("donut ocr - handwriting extraction")
st.markdown("using donut transformer model for docs")

c1, c2 = st.columns(2)

with c1:
    st.subheader("upload image")
    f = st.file_uploader("pick an image with text", type=["png","jpg","jpeg"])
    if f:
        img = Image.open(f)
        st.image(img, caption="uploaded image", use_column_width=True)

with c2:
    st.subheader("extracted text")
    if f:
        with st.spinner("processing..."):
            f.seek(0)
            t = ocr_image(f)
        if t and t.strip():
            st.success("text got extracted")
            st.text_area("detected text", value=t, height=400)
            st.download_button("download txt", t, file_name="donut.txt", mime="text/plain")
        else:
            st.warning("no text found")
    else:
        st.info("upload something first")

st.markdown("---")
st.markdown("model: donut transformer")
