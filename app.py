
import json
from pathlib import Path

import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image, ImageOps, UnidentifiedImageError


# --------------------------------------------------
# 1. APPLICATION CONFIGURATION
# --------------------------------------------------

st.set_page_config(
    page_title="Arecanut Disease Detection",
    page_icon="🌴",
    layout="wide"
)

BASE_DIR = Path(__file__).resolve().parent

MODEL_PATH = BASE_DIR / "mobilenetv2_finetuned_best.keras"
CLASS_PATH = BASE_DIR / "class_names.json"

IMG_SIZE = (224, 224)


# --------------------------------------------------
# 2. LOAD MODEL AND CLASS NAMES
# --------------------------------------------------

@st.cache_resource
def load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model file not found: {MODEL_PATH.name}"
        )

    return tf.keras.models.load_model(
        MODEL_PATH,
        compile=False
    )


@st.cache_data
def load_class_names():
    if not CLASS_PATH.exists():
        raise FileNotFoundError(
            f"Class file not found: {CLASS_PATH.name}"
        )

    with open(CLASS_PATH, "r", encoding="utf-8") as f:
        names = json.load(f)

    if not isinstance(names, list) or len(names) != 9:
        raise ValueError(
            "Expected a list containing exactly 9 class names."
        )

    return names


# --------------------------------------------------
# 3. IMAGE PREPROCESSING AND PREDICTION
# --------------------------------------------------

def predict_disease(image, model, classes):

    # Correct image orientation and RGB format
    image = ImageOps.exif_transpose(image).convert("RGB")

    # Resize to the model's input dimensions
    image = image.resize(
        IMG_SIZE,
        Image.Resampling.BILINEAR
    )

    # Convert image to a NumPy array
    image_array = np.asarray(
        image,
        dtype=np.float32
    )

    # Add batch dimension: (224, 224, 3)
    # becomes (1, 224, 224, 3)
    image_array = np.expand_dims(
        image_array,
        axis=0
    )

    # IMPORTANT:
    # The trained model already contains
    # MobileNetV2 preprocess_input().
    # Do not preprocess the image again here.

    probabilities = model.predict(
        image_array,
        verbose=0
    )[0]

    if (
        len(probabilities) != len(classes)
        or not np.all(np.isfinite(probabilities))
    ):
        raise ValueError(
            "The model returned invalid predictions."
        )

    predicted_index = int(np.argmax(probabilities))

    return predicted_index, probabilities


# --------------------------------------------------
# 4. WEB PAGE HEADER
# --------------------------------------------------

st.title("🌴 Arecanut Plant Disease Detection")

st.write(
    "Upload an arecanut plant image to obtain a "
    "prediction from the trained MobileNetV2 model."
)

st.info(
    "This research prototype predicts one of the nine "
    "classes learned during training. It is not a "
    "substitute for expert diagnosis."
)


# --------------------------------------------------
# 5. IMAGE UPLOAD
# --------------------------------------------------

uploaded_file = st.file_uploader(
    "Upload an arecanut image",
    type=["jpg", "jpeg", "png"]
)


# --------------------------------------------------
# 6. DISPLAY AND CLASSIFY
# --------------------------------------------------

if uploaded_file is not None:

    try:
        image = Image.open(uploaded_file)
        image = ImageOps.exif_transpose(image).convert("RGB")

        st.subheader("Uploaded Image")
        st.image(
            image,
            caption="Image selected for classification",
            use_container_width=True
        )

        if st.button(
            "Classify Image",
            type="primary",
            use_container_width=True
        ):

            try:
                with st.spinner("Classifying image..."):

                    model = load_model()
                    class_names = load_class_names()

                    predicted_index, probabilities = (
                        predict_disease(
                            image,
                            model,
                            class_names
                        )
                    )

                predicted_class = class_names[predicted_index]
                confidence = float(
                    probabilities[predicted_index]
                )

                st.divider()
                st.subheader("Prediction Result")

                st.success(
                    f"Predicted class: {predicted_class}"
                )

                st.metric(
                    "Model confidence",
                    f"{confidence * 100:.2f}%"
                )

                if confidence < 0.60:
                    st.warning(
                        "The model confidence is relatively low. "
                        "Try a clearer image or seek expert review. "
                        "This threshold is only a warning rule, "
                        "not a guarantee of correctness."
                    )

                st.subheader("Top Predictions")

                top_indices = np.argsort(
                    probabilities
                )[::-1][:3]

                for index in top_indices:
                    score = float(probabilities[index])

                    st.write(
                        f"**{class_names[index]}** — "
                        f"{score * 100:.2f}%"
                    )

                    st.progress(
                        float(np.clip(score, 0.0, 1.0))
                    )

                st.caption(
                    "Confidence values are model outputs, not "
                    "verified probabilities of disease. The "
                    "model may be less reliable on field images "
                    "that differ from its training dataset."
                )

            except Exception as error:
                st.error(
                    "Prediction failed. Check that the model, "
                    "class names, and preprocessing match."
                )
                st.exception(error)

    except (UnidentifiedImageError, OSError, ValueError):
        st.error(
            "Unable to read this image. Please upload a valid "
            "JPG, JPEG, or PNG file."
        )


# --------------------------------------------------
# 7. PROJECT INFORMATION
# --------------------------------------------------

with st.expander("About this project"):
    st.write(
        "This application uses transfer learning with "
        "MobileNetV2 to classify arecanut plant images "
        "into nine dataset classes."
    )
    st.write(
        "The model's test accuracy should be reported "
        "separately using a held-out, labelled test dataset."
    )
