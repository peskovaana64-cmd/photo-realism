import streamlit as st
import cv2
import numpy as np
from PIL import Image, ImageEnhance

st.title("🎬 Автоматический Фотореализм")
st.write("Загрузите кадр, и сайт мгновенно превратит его в эталонное фото с кинематографическим светом.")

uploaded_file = st.file_uploader("Выберите изображение...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Исходный кадр", use_container_width=True)
    
    if st.button("✨ Превратить в фотореализм"):
        with st.spinner("Обработка..."):
            img_cv = np.array(image)
            img_cv = cv2.cvtColor(img_cv, cv2.COLOR_RGB2BGR)
            
            denoised = cv2.fastNlMeansDenoisingColored(img_cv, None, h=5, hColor=5, templateWindowSize=7, searchWindowSize=21)
            gaussian = cv2.GaussianBlur(denoised, (0, 0), 2.0)
            sharpened = cv2.addWeighted(denoised, 1.3, gaussian, -0.3, 0)
            
            img_pil = Image.fromarray(cv2.cvtColor(sharpened, cv2.COLOR_BGR2RGB))
            img_pil = ImageEnhance.Color(img_pil).enhance(1.1)
            img_pil = ImageEnhance.Contrast(img_pil).enhance(1.05)
            
            np_img = np.array(img_pil)
            np_img[:, :, 0] = np.clip(np_img[:, :, 0].astype(np.int16) + 4, 0, 255).astype(np.uint8) 
            np_img[:, :, 2] = np.clip(np_img[:, :, 2].astype(np.int16) + 8, 0, 255).astype(np.uint8) 
            
            final_img = Image.fromarray(np_img)
            
            st.success("Готово!")
            st.image(final_img, caption="Результат коммерческого качества", use_container_width=True)
