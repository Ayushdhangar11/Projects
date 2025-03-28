import cv2
from ultralytics import YOLO

# Load YOLOv8 models with error handling
try:
    garbage_model = YOLO('last.pt')  # Garbage detection model
except Exception as e:
    print(f"Error loading garbage detection model: {e}")
    garbage_model = None

try:
    crowd_model = YOLO('yolov8s.pt')  # Crowd detection model
except Exception as e:
    print(f"Error loading crowd detection model: {e}")
    crowd_model = None

try:
    violence_model = YOLO('D:\\TY_EDI\\Violence-Detection-Using-YOLOv8-Towards-Automated-Video-Surveillance-and-Public-Safety-main\\best.pt')  # Violence detection model
except Exception as e:
    print(f"Error loading violence detection model: {e}")
    violence_model = None

def process_video(file_path="camera"):
    # Check if all models loaded successfully
    if not (garbage_model and crowd_model and violence_model):
        print("One or more models failed to load. Exiting.")
        return

    # Open video capture
    cap = cv2.VideoCapture(0) if file_path == "camera" else cv2.VideoCapture(file_path)

    if not cap.isOpened():
        print(f"Error: Could not open video source {file_path}")
        return

    while cap.isOpened():
        success, frame = cap.read()
        if success:
            # Run garbage detection
            garbage_annotated_frame = frame
            if garbage_model:
                try:
                    garbage_results = garbage_model.predict(frame)
                    garbage_annotated_frame = garbage_results[0].plot()
                except Exception as e:
                    print(f"Error during garbage detection: {e}")

            # Run crowd detection
            crowd_annotated_frame = frame
            if crowd_model:
                try:
                    crowd_results = crowd_model.predict(frame)
                    crowd_annotated_frame = crowd_results[0].plot()
                except Exception as e:
                    print(f"Error during crowd detection: {e}")

            # Run violence detection
            violence_annotated_frame = frame
            if violence_model:
                try:
                    violence_results = violence_model.predict(frame)
                    violence_annotated_frame = violence_results[0].plot()
                except Exception as e:
                    print(f"Error during violence detection: {e}")

            # Combine the annotated frames
            combined_frame = cv2.addWeighted(garbage_annotated_frame, 0.33, crowd_annotated_frame, 0.33, 0)
            combined_frame = cv2.addWeighted(combined_frame, 0.5, violence_annotated_frame, 0.5, 0)

            # Display the combined frame
            cv2.imshow("Garbage, Crowd, and Violence Detection", combined_frame)

            # Exit on 'ESC' key press
            if cv2.waitKey(1) == 27:  # Press 'ESC' to exit
                print("Exiting...")
                break
        else:
            print("Error: Could not read frame")
            break

    cap.release()
    cv2.destroyAllWindows()  # Close all OpenCV windows

if __name__ == '__main__':
    # Path to video file or 'camera' to use webcam
    file_path = input("Enter video file path or type 'camera' for webcam: ").strip()

    # Run the detection on the selected input
    process_video(file_path)
