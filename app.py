from flask import Flask, send_file, Response
import cv2
from cvzone.HandTrackingModule import HandDetector
from cvzone.ClassificationModule import Classifier
import numpy as np
import math
import pyttsx3
import threading
import time
import os

app = Flask(__name__)

# Initialize components
engine = pyttsx3.init()
engine.setProperty('rate', 150)  # Normal speaking rate
cap = cv2.VideoCapture(0)
detector = HandDetector(maxHands=1)
classifier = Classifier("Model/keras_model.h5", "Model/labels.txt")
labels = ["Bye", "Hello", "how are you", "I am fine", "Ok", "Same to u", "Take care", "Think"]

# Global variables
frame_size = (640, 480)
last_prediction = None
is_speaking = False

def speak_text(text):
    """Simple function to speak text"""
    global is_speaking
    if not is_speaking:
        is_speaking = True
        engine.say(text)
        engine.runAndWait()
        is_speaking = False

def process_frame(frame):
    global last_prediction
    imgOutput = frame.copy()
    hands, frame = detector.findHands(frame)
    
    if hands:
        hand = hands[0]
        x, y, w, h = hand['bbox']
        imgWhite = np.ones((300, 300, 3), np.uint8) * 255
        
        try:
            # Crop and process hand image
            imgCrop = frame[y-20:y+h+20, x-20:x+w+20]
            aspectRatio = h / w
            
            if aspectRatio > 1:
                k = 300 / h
                wCal = math.ceil(k * w)
                imgResize = cv2.resize(imgCrop, (wCal, 300))
                wGap = math.ceil((300-wCal)/2)
                imgWhite[:, wGap:wCal+wGap] = imgResize
            else:
                k = 300 / w
                hCal = math.ceil(k * h)
                imgResize = cv2.resize(imgCrop, (300, hCal))
                hGap = math.ceil((300-hCal)/2)
                imgWhite[hGap:hCal+hGap, :] = imgResize
            
            prediction, index = classifier.getPrediction(imgWhite, draw=False)
            current_text = labels[index]
            
            # Speak if prediction has changed
            if current_text != last_prediction:
                threading.Thread(target=speak_text, args=(current_text,), daemon=True).start()
                last_prediction = current_text
            
            # Draw on frame
            cv2.rectangle(imgOutput, (x-20, y-70), (x+280, y-20), (0,255,0), cv2.FILLED)
            cv2.putText(imgOutput, current_text, (x, y-30), cv2.FONT_HERSHEY_COMPLEX, 1.5, (0,0,0), 2)
            cv2.rectangle(imgOutput, (x-20, y-20), (x+w+20, y+h+20), (0,255,0), 4)
            
        except Exception as e:
            print(f"Error processing frame: {e}")
            
    else:
        # Reset prediction when no hands detected
        last_prediction = None
    
    return imgOutput

def generate_frames():
    while True:
        success, frame = cap.read()
        if not success:
            break
            
        # Resize frame for better performance
        frame = cv2.resize(frame, frame_size)
        
        # Process frame
        processed_frame = process_frame(frame)
        
        # Convert to jpg
        ret, buffer = cv2.imencode('.jpg', processed_frame)
        frame_bytes = buffer.tobytes()
        
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
        
        time.sleep(0.01)

@app.route('/')
def index():
    # Serve index.html from the main folder
    return send_file('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(),
                   mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    try:
        # Check if index.html exists in the main folder
        if not os.path.exists('index.html'):
            print("Warning: index.html not found in the main folder!")
        app.run(debug=True, threaded=True)
    finally:
        cap.release()
        cv2.destroyAllWindows()