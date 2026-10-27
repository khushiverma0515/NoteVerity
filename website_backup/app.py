import streamlit as st
import cv2
import numpy as np
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model.predict import detect_currency

st.set_page_config(
    page_title="Indian Currency Note Detection",
    page_icon="💵",
    layout="centered"
)

st.sidebar.title("📋About")

st.sidebar.info(
    """
    This project detects Indian currency notes
    using Artificial Intelligence (YOLOv8).

    Developed as a college mini project.
    """
)

st.sidebar.header("📖Instructions")

st.sidebar.write(
    """
    1. Upload an image of an Indian Currency Note.
    2. The AI model will detect the denomination.
    """
)

st.title("💵 Indian Currency Note Detection")

st.write(
    "Welcome! Upload an image of an Indian currency note and click 'Analyze Image' to detect its denomination."
)

st.header("📤 Upload Currency Note")

uploaded_file= st.file_uploader(
    "Choose an image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:

    st.success("Image uploaded successfully!")

    st.image(
        uploaded_file,
        caption="Uploaded Currency Note",
        use_container_width=True
    )

    st.subheader("📄File Details")

    st.write(f"**File Name:** {uploaded_file.name}")

    st.write(f"**File Type:** {uploaded_file.type}")

    st.write(f"**File Size:** {round(uploaded_file.size/1024,2)} KB")

analyze = st.button("🔍 Analyze Image")

st.header("🤖 Prediction")

if uploaded_file is not None:
    if analyze:
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)

        image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

        st.success("Analysis Complete!")

        st.divider()

        st.subheader("🤖 AI Detection Result")

        denomination, confidence, result = detect_currency(image)

        if denomination is not None:
            st.success(f"Detected Note: ₹{denomination}")

            confidence_percent = confidence * 100

            st.metric(
                "Confidence",
                f"{confidence_percent:.2f}%"
            )

            if confidence_percent >= 90:
                st.success("High confidence detection")

            elif confidence_percent >= 70:
                st.info("Good confidence detection")

            else:
                st.warning("Low confidence detection")

            st.subheader("📍 Detection Preview")

            annotated_image = result.plot()

            st.image(
                annotated_image,
                caption="AI Detection Result",
                use_container_width=True
            )

        else:
            st.warning("No currency note detected.")
            st.info(
                "Please upload a clear image of an Indian currency note"
                "and try again."
            )

    else:
        st.info("Click 'Analyze Image' to start detection")

else:
    st.warning("Please upload an image!")