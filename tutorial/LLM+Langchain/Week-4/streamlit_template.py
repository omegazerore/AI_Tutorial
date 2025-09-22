# app.py
import streamlit as st
import pandas as pd
import numpy as np

# Title and text
st.title("🌟 My First Streamlit App")
st.write("Hello! This is a basic Streamlit example.")

# Slider widget
x = st.slider("Select a value for x", 0, 100, 50)

# Display the chosen value
st.write(f"You selected x = {x}")

# Generate random data
data = pd.DataFrame(
    np.random.randn(20, 2) * x,
    columns=["Column 1", "Column 2"]
)

# Line chart
st.line_chart(data)

# Button
if st.button("Say hi"):
    st.success("👋 Hi there!")