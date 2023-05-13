import streamlit as st
from PIL import Image


st.markdown("# HP Challenge Supply Chain 💻")
st.sidebar.markdown("# Home 🎈")


image = Image.open('LogoOptiFlow.jpeg')

st.image(image)

