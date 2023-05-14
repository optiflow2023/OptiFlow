import streamlit as st
from PIL import Image

st.markdown("# HP Challenge Supply Chain 💻")
st.sidebar.markdown("# Home 🎈")


image = Image.open("LogoOptiFlow.jpeg")

st.subheader("Welcome to our project!")

st.image(image, use_column_width=True)

st.subheader("About Us")
team_members = [
    {
        "name": "Nathaniel Mitrani",
        "degree": "Mathematics and Data Science and Engineering",
        "linkedin": "https://www.linkedin.com/in/nathaniel-mitrani-hadida-031b4021a/",
    },
    {
        "name": "Alex Serrano",
        "degree": "Mathematics and Data Science and Engineering",
        "linkedin": "https://www.linkedin.com/in/alexste/",
    },
    {
        "name": "Jan Tarrats",
        "degree": "Mathematics and Computer Engineering",
        "linkedin": "https://www.linkedin.com/in/jan-tarrats-castillo/",
    },
]

# Set up columns for horizontal layout
col1, col2, col3 = st.columns(3)
cols = [col1, col2, col3]

for member, col in list(zip(team_members, cols)):
    with col:
        st.write(f"### {member['name']}")
        st.markdown(f"<u>Degrees</u>: {member['degree']}", unsafe_allow_html=True)
        st.write(f"LinkedIn: [{member['name']}]({member['linkedin']})")

st.write("---")

st.subheader("Abstract")
st.write(
    "In this project, we propose a novel manner to optimize the robustness of a supply chain through stochastic modeling of its disruptions."
)
