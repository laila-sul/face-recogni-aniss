import cv2
import streamlit as st
import numpy as np
from PIL import Image
import io


face_cascade = cv2.CascadeClassifier('C:/Users/Aniss/PycharmProjects/pythonProject2.xml')

def detect_faces(image, scaleFactor, minNeighbors, color):
   gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
   faces = face_cascade.detectMultiScale(gray, scaleFactor=scaleFactor, minNeighbors=minNeighbors)
   for (x, y, w, h) in faces:
        cv2.rectangle(image, (x, y), (x+w, y+h), color, 2)
   return image, faces

def main():
    st.title("Face Detection")

    st.write(
        "1. Upload an image file.\n"
        "2. Adjust the parameters to fine-tune the face detection.\n"
        "3. Choose the color of the rectangles around detected faces.\n"
        "4. Click the 'Detect Faces' button to see the results.\n"
        "5. Download the image with detected faces using the 'Save Image' button."
    )

    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        image_np = np.array(image)

        st.image(image, caption="Uploaded Image", use_column_width=True)

        scaleFactor = st.slider("Scale Factor", min_value=1.1, max_value=2.0, value=1.3, step=0.1)
        minNeighbors = st.slider("Min Neighbors", min_value=1, max_value=10, value=3, step=1)
        color = st.color_picker("Rectangle Color", "#FF0000")
        color_bgr = [int(color[i:i+2], 16) for i in (1, 3, 5)]

        if st.button("Detect Faces"):
            result_image, faces = detect_faces(image_np, scaleFactor, minNeighbors, color_bgr)

            st.image(result_image, caption="Detected Faces", use_column_width=True)


            result_image_pil = Image.fromarray(result_image)
            buffer = io.BytesIO()
            result_image_pil.save(buffer, format="PNG")
            buffer.seek(0)

            st.download_button(
                label="Save Image",
                data=buffer,
                file_name="detected_faces.png",
                mime="image/png"
            )

if __name__ == "__main__":
    main()