# 예시 코드 (파일 업로드 → 유사한 얼굴 찾기)
import streamlit as st
import face_recognition
import os
import numpy as np
from PIL import Image

st.title("닮은 독립운동가 찾기")

DATA_DIR = "data/patriots"
uploaded_file = st.file_uploader("얼굴 사진을 업로드하세요", type=["jpg", "png", "jpeg"])

if uploaded_file:
    user_img = face_recognition.load_image_file(uploaded_file)
    st.image(user_img, caption="업로드한 사진", use_column_width=True)

    try:
        user_encoding = face_recognition.face_encodings(user_img)[0]
    except IndexError:
        st.error("얼굴을 찾을 수 없습니다.")
    else:
        min_dist = float("inf")
        best_match = None
        for fname in os.listdir(DATA_DIR):
            try:
                known_img = face_recognition.load_image_file(os.path.join(DATA_DIR, fname))
                known_enc = face_recognition.face_encodings(known_img)[0]
                dist = np.linalg.norm(user_encoding - known_enc)
                if dist < min_dist:
                    min_dist = dist
                    best_match = fname
            except:
                continue
        if best_match:
            st.success(f"가장 닮은 인물: {best_match}")
            st.image(os.path.join(DATA_DIR, best_match), caption="유사 인물", use_column_width=True)
