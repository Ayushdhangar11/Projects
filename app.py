import cv2
import numpy as np
from ultralytics import YOLO
from flask import Flask, Response, render_template, request, redirect, url_for
import tempfile
import os

# Flask application setup
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = tempfile.gettempdir()

# Load YOLOv8 models for garbage and violence detection
garbage_model = YOLO('last.pt')  # Garbage detection model
violence_model = YOLO(r'D:\TY_EDI\Violence-Detection-Using-YOLOv8-Towards-Automated-Video-Surveillance-and-Public-Safety-main\best.pt')  # Violence detection model

# Load YOLOv3 model for crowd detection
def load_yolo_v3_model():
    config_path = r'D:\TY_EDI\Crowd-Management\yolov3.cfg'
    weights_path = r'D:\TY_EDI\Crowd-Management\yolov3.weights'
    names_path = r'D:\TY_EDI\Crowd-Management\coco.names'

    print(f"Loading YOLOv3 model from:\nConfig: {config_path}\nWeights: {weights_path}")
    net = cv2.dnn.readNetFromDarknet(config_path, weights_path)
    layers = net.getLayerNames()

    try:
        output_layers = [layers[i - 1] for i in net.getUnconnectedOutLayers()]
    except IndexError:
        output_layers = [layers[i - 1] for i in net.getUnconnectedOutLayers().flatten()]

    with open(names_path, 'r') as f:
        classes = [line.strip() for line in f.readlines()]

    return net, output_layers, classes

crowd_net, crowd_output_layers, crowd_classes = load_yolo_v3_model()

# Flag to indicate if the script should terminate
terminate_flag = False

# Function to process YOLOv3 (Crowd detection)
def detect_crowd(net, output_layers, frame):
    height, width, channels = frame.shape
    blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    detections = net.forward(output_layers)

    person_count = 0  # Initialize person count
    for detection in detections:
        for obj in detection:
            scores = obj[5:]
            class_id = int(np.argmax(scores))  # Use np.argmax to find the highest confidence
            confidence = scores[class_id]
            if confidence > 0.5:
                center_x, center_y, w, h = (obj[0:4] * np.array([width, height, width, height])).astype("int")
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)
                cv2.rectangle(frame, (x, y), (x + int(w), y + int(h)), (0, 255, 0), 2)
                cv2.putText(frame, crowd_classes[class_id], (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                # Increment the person count when 'person' (class ID 0) is detected
                if class_id == 0:
                    person_count += 1

    # Display crowd count and alert
    cv2.putText(frame, f"Crowd size: {person_count}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    if person_count > 10:
        cv2.putText(frame, "Crowd size exceeded!", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    return frame

# Define a generator function to stream video frames to the web page
def generate(file_path):
    print(f"File Path: {file_path}")
    if file_path == "camera":
        cap = cv2.VideoCapture(0)
    else:
        cap = cv2.VideoCapture(file_path)

    if not cap.isOpened():
        print("Error: Could not open video file or camera.")
        return

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            print("Error: Could not read frame.")
            break

        # Run garbage and violence detection using YOLOv8
        garbage_results = garbage_model(frame)
        violence_results = violence_model(frame)

        # Detect crowd using YOLOv3
        frame = detect_crowd(crowd_net, crowd_output_layers, frame)

        # Visualize YOLOv8 results
        garbage_annotated_frame = garbage_results[0].plot()
        violence_annotated_frame = violence_results[0].plot()

        # Combine frames
        combined_frame = cv2.addWeighted(garbage_annotated_frame, 0.5, violence_annotated_frame, 0.5, 0)

        # Encode the frame as JPEG
        ret, jpeg = cv2.imencode('.jpg', combined_frame)

        if not ret:
            print("Error: Could not encode frame to JPEG.")
            break

        # Yield the JPEG data to Flask
        yield (b'--frame\r\n'
              b'Content-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n')

        if cv2.waitKey(1) == 27 or terminate_flag:  # Exit when ESC key is pressed or terminate flag is set
            break

    cap.release()
    os._exit(0)  # Terminate the script when the video stream ends or terminate flag is set

# Define a route to serve the video stream
@app.route('/video_feed')
def video_feed():
    file_path = request.args.get('file')
    return Response(generate(file_path),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

# Define a route to serve the HTML page with the file upload form
@app.route('/', methods=['GET', 'POST'])
def index():
    global terminate_flag
    if request.method == 'POST':
        # Handle form submission
        if request.form.get("camera") == "true":
            file_path = "camera"
        elif 'file' in request.files:
            file = request.files['file']
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(file_path)
        else:
            file_path = None

        if file_path:
            return redirect(url_for('video_feed', file=file_path))  # Redirect to video feed with file path
        else:
            return render_template('index.html', message="Please select a video or camera.")
    return render_template('index.html')

@app.route('/stop', methods=['POST'])
def stop():
    global terminate_flag
    terminate_flag = True
    return "Process has been Terminated"

if __name__ == '__main__':
    app.run(debug=True)
