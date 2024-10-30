import cv2
import mediapipe as mp
import time
import numpy as np
from keras.models import load_model
from keras.preprocessing.image import img_to_array
import tkinter as tk
from tkinter import filedialog
from tkinter import messagebox

# Load the age and gender model
from keras.metrics import MeanAbsoluteError

# Load the model
model = load_model('age_gender_model.h5', custom_objects={'mae': MeanAbsoluteError()})

# Function to preprocess the image for prediction
def preprocess_image(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # Convert to grayscale
    image = cv2.resize(image, (128, 128))  # Resize to 128x128
    image = img_to_array(image)  # Convert to array
    image = np.expand_dims(image, axis=0)  # Expand dimensions to match model input
    image = image / 255.0  # Normalize the image
    return image

# Function to process uploaded image
def process_image(img):
    mpFaceDetection = mp.solutions.face_detection
    faceDetection = mpFaceDetection.FaceDetection(0.75)

    predictions = {}
    Ptime = 0

    # Initialize default values for gender and age
    pred_gender = "Unknown"
    pred_age = "Unknown"

    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = faceDetection.process(imgRGB)

    if results.detections:
        for id, detection in enumerate(results.detections):
            h, w, ch = img.shape
            bboxC = detection.location_data.relative_bounding_box
            bbox = int(bboxC.xmin * w), int(bboxC.ymin * h), \
                   int(bboxC.width * w), int(bboxC.height * h)

            # Convert bbox to tuple for easy comparison
            bbox_tuple = tuple(bbox)

            # Check if the face has been predicted before
            if bbox_tuple in predictions:
                pred_gender, pred_age = predictions[bbox_tuple]
            else:
                cv2.rectangle(img, bbox, (255, 0, 0), 2)

                # Extract the face region for age and gender prediction
                face_img = img[bbox[1]:bbox[1] + bbox[3], bbox[0]:bbox[0] + bbox[2]]
                if face_img.size > 0:
                    preprocessed_face = preprocess_image(face_img)

                    # Predict age and gender
                    pred = model.predict(preprocessed_face)
                    pred_gender = "Male" if round(pred[0][0][0]) == 0 else "Female"
                    pred_age = round(pred[1][0][0])

                    # Store the predictions for this face
                    predictions[bbox_tuple] = (pred_gender, pred_age)

    # Display predictions (even if no faces are detected)
    cv2.putText(img, f'Gender: {pred_gender}', (20, 50), cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 0), 2)
    cv2.putText(img, f'Age: {pred_age}', (20, 80), cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 0), 2)

    cv2.imshow("Uploaded Image", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Function to upload image
def upload_image():
    file_path = filedialog.askopenfilename()
    if file_path:
        img = cv2.imread(file_path)
        if img is not None:
            img = cv2.resize(img, (800, 600))  # Resize for better visibility
            process_image(img)
        else:
            messagebox.showerror("Error", "Could not read the image.")

# Function to open camera
def open_camera():
    cap = cv2.VideoCapture(0)
    Ptime = 0

    # Initialize MediaPipe face detection
    mpFaceDetection = mp.solutions.face_detection
    faceDetection = mpFaceDetection.FaceDetection(0.75)

    predictions = {}

    while True:
        success, img = cap.read()
        if not success:
            print("Error: Could not read frame.")
            break

        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = faceDetection.process(imgRGB)

        if results.detections:
            for id, detection in enumerate(results.detections):
                h, w, ch = img.shape
                bboxC = detection.location_data.relative_bounding_box
                bbox = int(bboxC.xmin * w), int(bboxC.ymin * h), \
                       int(bboxC.width * w), int(bboxC.height * h)

                # Convert bbox to tuple for easy comparison
                bbox_tuple = tuple(bbox)

                # Check if the face has been predicted before
                if bbox_tuple in predictions:
                    pred_gender, pred_age = predictions[bbox_tuple]
                else:
                    cv2.rectangle(img, bbox, (255, 0, 0), 2)

                    # Extract the face region for age and gender prediction
                    face_img = img[bbox[1]:bbox[1] + bbox[3], bbox[0]:bbox[0] + bbox[2]]
                    if face_img.size > 0:
                        preprocessed_face = preprocess_image(face_img)

                        # Predict age and gender
                        pred = model.predict(preprocessed_face)
                        pred_gender = "Male" if round(pred[0][0][0]) == 0 else "Female"
                        pred_age = round(pred[1][0][0])

                        # Store the predictions for this face
                        predictions[bbox_tuple] = (pred_gender, pred_age)

                    # Display fixed predictions with smaller text
                    cv2.putText(img, f'Gender: {pred_gender} {int(detection.score[0]*100)}%', (bbox[0], bbox[1]-40), cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 0), 2)
                    cv2.putText(img, f'Age: {pred_age}', (bbox[0], bbox[1]-10), cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 0), 2)

        Ctime = time.time()
        fps = 1 / (Ctime - Ptime)
        Ptime = Ctime

        cv2.putText(img, f'FPS: {int(fps)}', (20, 70), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 3)
        cv2.imshow("Camera Feed", img)
        if cv2.waitKey(1) == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# Function to create a simple UI
def create_ui():
    root = tk.Tk()
    root.title("Age and Gender Prediction")
    root.geometry("300x200")

    btn_upload = tk.Button(root, text="Upload Image", command=upload_image, font=("Arial", 14))
    btn_upload.pack(pady=20)

    btn_camera = tk.Button(root, text="Open Camera", command=open_camera, font=("Arial", 14))
    btn_camera.pack(pady=20)

    root.mainloop()

# Run the UI
create_ui()
