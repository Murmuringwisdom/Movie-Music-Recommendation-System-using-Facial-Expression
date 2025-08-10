import streamlit as st
from deepface import DeepFace
import random
import pandas as pd
import cv2
from PIL import Image
import numpy as np

# Load Haar cascade for face detection
haar_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# Paths to the datasets
datasets = {
    "hollywood": {
        "movies": {
            "happy": r"C:\Users\Aniket Routray\OneDrive\Desktop\Dataset\Hollywood Movies\\Happy.csv",
            "sad": r"C:\Users\Aniket Routray\OneDrive\Desktop\Dataset\Hollywood Movies\\sad.csv",
            "angry": r"C:\Users\Aniket Routray\OneDrive\Desktop\Dataset\Hollywood Movies\\Angry.csv",
            "fear": r"C:\Users\Aniket Routray\OneDrive\Desktop\Dataset\Hollywood Movies\\Fear.csv",
            "neutral": r"C:\Users\Aniket Routray\OneDrive\Desktop\Dataset\Hollywood Movies\\neutral.csv",
            "surprise": r"C:\Users\Aniket Routray\OneDrive\Desktop\Dataset\Hollywood Movies\\surprise.csv"
        },
        "songs": {
            "happy": r"C:\Users\Aniket Routray\OneDrive\Desktop\Dataset\Hollywood Songs\\happysh.csv",
            "sad": r"C:\Users\Aniket Routray\OneDrive\Desktop\Dataset\Hollywood Songs\\sadsh.csv",
            "angry": r"C:\Users\Aniket Routray\OneDrive\Desktop\Dataset\Hollywood Songs\\angrysh.csv",
            "fear": r"C:\Users\Aniket Routray\OneDrive\Desktop\Dataset\Hollywood Songs\\fearsh.csv",
            "neutral": r"C:\Users\Aniket Routray\OneDrive\Desktop\Dataset\Hollywood Songs\\neutralsh.csv",
            "surprise": r"C:\Users\Aniket Routray\OneDrive\Desktop\Dataset\Hollywood Songs\\surprisesh.csv"
        }
    },
    "bollywood": {
        "movies": {
            "happy": r"C:\Users\Aniket Routray\OneDrive\Desktop\Dataset\Bollywood movies\\happyb.csv",
            "sad": r"C:\Users\Aniket Routray\OneDrive\Desktop\Dataset\Bollywood movies\\sadb.csv",
            "angry": r"C:\Users\Aniket Routray\OneDrive\Desktop\Dataset\Bollywood movies\\angryb.csv",
            "fear": r"C:\Users\Aniket Routray\OneDrive\Desktop\Dataset\Bollywood movies\\fearb.csv",
            "neutral": r"C:\Users\Aniket Routray\OneDrive\Desktop\Dataset\Bollywood movies\neutralb.csv",
            "surprise": r"C:\Users\Aniket Routray\OneDrive\Desktop\Dataset\Bollywood movies\\surpriseb.csv"
        },
        "songs": {
            "happy": r"C:\Users\Aniket Routray\OneDrive\Desktop\Dataset\Bollywood songs\\happybs.csv",
            "sad": r"C:\Users\Aniket Routray\OneDrive\Desktop\Dataset\Bollywood songs\\sadbs.csv",
            "angry": r"C:\Users\Aniket Routray\OneDrive\Desktop\Dataset\Bollywood songs\\angrybs.csv",
            "fear": r"C:\Users\Aniket Routray\OneDrive\Desktop\Dataset\Bollywood songs\\fearbs.csv",
            "neutral": r"C:\Users\Aniket Routray\OneDrive\Desktop\Dataset\Bollywood songs\\neutralbs.csv",
            "surprise": r"C:\Users\Aniket Routray\OneDrive\Desktop\Dataset\Bollywood songs\\surprisebs.csv"
        }
    }
}

def recommend_from_dataset(emotion, dataset_paths, column_name):
    try:
        file_path = dataset_paths.get(emotion, "")
        if file_path:
            df = pd.read_csv(file_path)
            if column_name in df.columns:
                items = df[column_name].dropna().tolist()
                return random.choice(items) if items else "No recommendations found."
            return f"The column '{column_name}' does not exist in the dataset."
        return "No dataset available for this emotion."
    except Exception as e:
        return f"Error reading the dataset: {e}"

# Streamlit-based UI
st.set_page_config(page_title="Emotion-Based Recommendations", layout="wide")

# Initialize session state
if "current_page" not in st.session_state:
    st.session_state.current_page = "start"
if "current_category" not in st.session_state:
    st.session_state.current_category = None
if "current_type" not in st.session_state:
    st.session_state.current_type = None
if "current_emotion" not in st.session_state:
    st.session_state.current_emotion = "neutral"

def start_page():
    st.title("Emotion-Based Recommendations")
    if st.button("Start"):
        st.session_state.current_page = "category"

def category_page():
    st.title("Choose a Category")
    if st.button("Hollywood"):
        st.session_state.current_category = "hollywood"
        st.session_state.current_page = "type"
    if st.button("Bollywood"):
        st.session_state.current_category = "bollywood"
        st.session_state.current_page = "type"
    if st.button("Back"):
        st.session_state.current_page = "start"

def type_page():
    st.title("Choose a Type")
    if st.button("Movies"):
        st.session_state.current_type = "movies"
        st.session_state.current_page = "recognition"
    if st.button("Songs"):
        st.session_state.current_type = "songs"
        st.session_state.current_page = "recognition"
    if st.button("Back"):
        st.session_state.current_page = "category"

def recognition_page():
    st.title("Emotion Recognition and Recommendation")
    st.text(f"Category: {st.session_state.current_category.title()}")
    st.text(f"Type: {st.session_state.current_type.title()}")

    # Camera functionality
    uploaded_image = st.camera_input("Capture an Image")

    if uploaded_image:
        file_bytes = np.asarray(bytearray(uploaded_image.read()), dtype=np.uint8)
        frame = cv2.imdecode(file_bytes, 1)

        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = haar_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        if len(faces) > 0:
            x, y, w, h = faces[0]  # Use the first detected face
            face_roi = frame[y:y + h, x:x + w]

            try:
                result = DeepFace.analyze(face_roi, actions=['emotion'], enforce_detection=False)
                if isinstance(result, list):
                    result = result[0]
                st.session_state.current_emotion = result.get("dominant_emotion", "neutral")
            except Exception as e:
                st.session_state.current_emotion = "neutral"
                st.error(f"Error in emotion detection: {e}")
        else:
            st.session_state.current_emotion = "neutral"

        st.success(f"Detected Emotion: {st.session_state.current_emotion.capitalize()}")

    # Recommendation
    dataset_paths = datasets.get(st.session_state.current_category, {}).get(st.session_state.current_type, {})
    column_name = "Movie" if st.session_state.current_type == "movies" else "Song"
    recommendation = recommend_from_dataset(st.session_state.current_emotion, dataset_paths, column_name)
    st.text(f"Recommendation: {recommendation}")

    if st.button("Back"):
        st.session_state.current_page = "type"

# Render the appropriate page
if st.session_state.current_page == "start":
    start_page()
elif st.session_state.current_page == "category":
    category_page()
elif st.session_state.current_page == "type":
    type_page()
elif st.session_state.current_page == "recognition":
    recognition_page()

