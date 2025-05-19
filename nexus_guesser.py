import streamlit as st
from streamlit_drawable_canvas import st_canvas
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from PIL import Image, ImageOps
import pandas as pd 
import altair as alt 
import os

# Page configuration
# Page configuration
st.set_page_config(
    page_title="AI Hand Digit Classifier",
    layout="wide",
    page_icon="🦕",
    initial_sidebar_state="collapsed" 
)

#Side bar for Priject info
with st.sidebar:
    st.markdown("## 📋 Project Info")
    st.markdown("""
    **Title:** AI-Powered Hand Digit Classifier  
    **Author:** Abinanthan  
    **Model:** Convolutional Neural Network (CNN)  
    **Dataset:** MNIST Handwritten Digits  
    **Tooling:** TensorFlow, Streamlit, Altair  
    **Date:** May 2025  
    """)

    st.markdown("---")
    st.markdown("## 🔗 Links")
    st.markdown("""
    [📁 GitHub Repository](https://github.com/Abinanthan-CG/Hand-Digit-Recognition)  
    [📄 Project Report (UPDATING SOON...)](https://example.com)  
    """)

    st.markdown("---")


# --- Optimal Hardcoded Settings ---
OPTIMAL_STROKE_WIDTH = 18
OPTIMAL_CANVAS_WIDTH = 380 # Consider if this is too wide for portrait even with warning
OPTIMAL_CANVAS_HEIGHT = 380
OPTIMAL_DIGIT_FIT_SIZE = (16, 16)

# --- Model Loading ---
@st.cache_resource
def load_keras_model_silent(model_path):
    try:
        model = load_model(model_path)
        return model
    except Exception as e:
        print(f"[MODEL LOAD ERROR] Error loading model: {str(e)}")
        return None

MODELS_DIR = "models"
MODEL_FILENAME = "mnist_cnn_model.h5"
MODEL_PATH = os.path.join(MODELS_DIR, MODEL_FILENAME)
model = load_keras_model_silent(MODEL_PATH)

if 'model_load_status_shown' not in st.session_state:
    if model:
        st.toast("AI Model Ready!", icon="✅")
    else:
        st.toast("Error loading AI Model. Predictions may be unavailable.", icon="❌")
    st.session_state.model_load_status_shown = True

# --- Image Preprocessing ---
def preprocess_image_advanced(image_data, model_input_size=(28, 28), digit_fit_size=OPTIMAL_DIGIT_FIT_SIZE):
    if image_data is None or not image_data.any(): return None
    img_rgba = Image.fromarray(image_data.astype('uint8'), 'RGBA')
    alpha_channel = img_rgba.split()[3]; bbox = alpha_channel.getbbox()
    if bbox is None:
        img_final_gray = Image.new('L', model_input_size, color='black'); img_array = np.array(img_final_gray)
        img_normalized = img_array.astype('float32') / 255.0
        return img_normalized.reshape(1, model_input_size[0], model_input_size[1], 1)
    img_cropped = img_rgba.crop(bbox); current_width, current_height = img_cropped.size
    target_w, target_h = digit_fit_size
    ratio = min(target_w / current_width, target_h / current_height) if current_width > 0 and current_height > 0 else 1.0
    new_width = int(current_width * ratio); new_height = int(current_height * ratio)
    new_width = max(1, new_width); new_height = max(1, new_height)
    img_resized_digit = img_cropped.resize((new_width, new_height), Image.Resampling.LANCZOS)
    img_resized_digit_gray = img_resized_digit.convert('L'); img_resized_digit_inverted = ImageOps.invert(img_resized_digit_gray)
    final_model_img = Image.new('L', model_input_size, color='black')
    paste_x = (model_input_size[0] - new_width) // 2; paste_y = (model_input_size[1] - new_height) // 2
    final_model_img.paste(img_resized_digit_inverted, (paste_x, paste_y))
    img_array = np.array(final_model_img); img_normalized = img_array.astype('float32') / 255.0
    return img_normalized.reshape(1, model_input_size[0], model_input_size[1], 1)


# --- Streamlit App UI ---

# Add the landscape warning message here
st.markdown("""
<style>
@media (orientation: portrait) {
    .turn-device-message {
        display: flex !important; /* Use flex to center content */
    }
    /* Optional: Hide main content when message is shown in portrait */
    /* .main-content-area { display: none !important; } */
}
.turn-device-message {
    display: none; /* Hidden by default */
    position: fixed;
    top: 0; left: 0;
    width: 100vw; /* Cover full viewport width */
    height: 100vh; /* Cover full viewport height */
    background-color: rgba(0,0,0,0.95);
    color: white;
    z-index: 9999;
    align-items: center;
    justify-content: center;
    text-align: center;
    font-size: 1.2em; /* Adjusted for better fit */
    padding: 20px;
    box-sizing: border-box; /* Include padding in width/height */
}
</style>
<div class="turn-device-message">
    <p>🔄 For the best experience with this app,<br>please rotate your device to landscape mode. 🔄</p>
</div>
""", unsafe_allow_html=True)

# Start of main app content (you can wrap this in a div if you want to hide it)
# For example: st.markdown("<div class='main-content-area'>", unsafe_allow_html=True)

st.title("🤖 AI-Powered Hand Digit Classifier")
st.subheader("Put your handwriting to the test — draw any digit from 0 to 9!")

# New 3-column layout with specified ratios
col_canvas, col_prediction_display, col_probabilities = st.columns([0.3, 0.3, 0.4])

with col_canvas:
    st.markdown("#### Draw Here 👇")
    canvas_result = st_canvas(
        fill_color="rgba(255, 165, 0, 0.0)",
        stroke_width=OPTIMAL_STROKE_WIDTH,
        stroke_color="#000000",
        background_color="#FFFFFF",
        width=OPTIMAL_CANVAS_WIDTH,
        height=OPTIMAL_CANVAS_HEIGHT,
        drawing_mode="freedraw",
        key="main_canvas_3col_altair_warn", # New key for this version
        display_toolbar=True,
        update_streamlit=True,
    )

# Initialize variables for prediction display
display_digit = "-"
display_confidence_str = "Confidence: -"
prediction_probs = None
prediction_error_message = None

# Process data and determine what to display
if model is None:
    pass
elif canvas_result.image_data is not None and canvas_result.image_data.any():
    processed_img = preprocess_image_advanced(canvas_result.image_data)
    if processed_img is not None:
        try:
            prediction_output = model.predict(processed_img)
            prediction_probs = prediction_output[0]
            predicted_digit_val = np.argmax(prediction_probs)
            confidence_val = np.max(prediction_probs) * 100
            display_digit = str(predicted_digit_val)
            display_confidence_str = f"Confidence: {confidence_val:.1f}%"
        except Exception as e:
            prediction_error_message = f"Error during prediction: {e}"
            display_digit = "⚠️"
            display_confidence_str = "Prediction Error"

with col_prediction_display:
    st.markdown("#### AI's Guess 🤔")
    if model is None:
        st.error("Model not loaded.")
    elif prediction_error_message:
        st.error(prediction_error_message)
        st.markdown(f"""
        <div style="text-align: center; margin-top: 10px;">
            <p style="font-size: 10em; font-weight: bold; margin-bottom: -0.1em; line-height: 1;">
                {display_digit}
            </p>
            <p style="font-size: 1.1em; color: grey; margin-top: -0.3em;">
                {display_confidence_str}
            </p>
        </div>
        """, unsafe_allow_html=True)
    elif canvas_result.image_data is None or not canvas_result.image_data.any():
        st.info("Draw a digit on the left.")
        st.markdown(f"""
        <div style="text-align: center; margin-top: 10px;">
            <p style="font-size: 10em; font-weight: bold; margin-bottom: -0.1em; line-height: 1;">
                -
            </p>
            <p style="font-size: 1.1em; color: grey; margin-top: -0.3em;">
                Confidence: -
            </p>
        </div>
        """, unsafe_allow_html=True)
    elif processed_img is None and (canvas_result.image_data is not None and canvas_result.image_data.any()):
        st.warning("Could not process drawing. Try drawing more clearly.")
        st.markdown(f"""
        <div style="text-align: center; margin-top: 10px;">
            <p style="font-size: 10em; font-weight: bold; margin-bottom: -0.1em; line-height: 1;">
                -
            </p>
            <p style="font-size: 1.1em; color: grey; margin-top: -0.3em;">
                Confidence: -
            </p>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div style="text-align: center; margin-top: 10px;">
            <p style="font-size: 10em; font-weight: bold; margin-bottom: -0.1em; line-height: 1;">
                {display_digit}
            </p>
            <p style="font-size: 1.1em; color: grey; margin-top: -0.3em;">
                {display_confidence_str}
            </p>
        </div>
        """, unsafe_allow_html=True)

with col_probabilities:
    st.markdown("#### Probability Breakdown 📊")
    if model is None:
        st.caption("Model not loaded.")
    elif prediction_probs is not None and display_digit != "⚠️":
        data_for_altair = pd.DataFrame({
            'Digit': [str(i) for i in range(10)],
            'Probability': prediction_probs
        })
        chart = alt.Chart(data_for_altair).mark_bar().encode(
            x=alt.X('Probability:Q', axis=alt.Axis(format='%', title='Probability (%)', labelExpr="datum.value * 100"), scale=alt.Scale(domain=[0, 1])),
            y=alt.Y('Digit:N', sort=None, axis=alt.Axis(title='Digit')),
            tooltip=['Digit', alt.Tooltip('Probability:Q', format='.2%')]
        ).properties(
            height=300
        )
        st.altair_chart(chart, use_container_width=True)
    elif display_digit == "⚠️":
        st.caption("Probabilities N/A due to prediction error.")
    else:
        st.caption("Probabilities will appear here after drawing.")

# --- Footer ---
st.markdown("---")
st.markdown(
    "<p style='text-align: center; color: gray;'>© 2025 Abinanthan. All rights reserved.</p>",
    unsafe_allow_html=True
)
