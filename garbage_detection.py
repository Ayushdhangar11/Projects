import cv2
from ultralytics import YOLO

# Load the YOLOv8 model
model = YOLO('last.pt')

# Function to process video or camera input
def process_video(file_path="camera"):
    # Open video capture
    if file_path == "camera":
        cap = cv2.VideoCapture(0)  # Open the default camera
    else:
        cap = cv2.VideoCapture(file_path)  # Open the video file

    if not cap.isOpened():
        print(f"Error: Could not open video source {file_path}")
        return

    while cap.isOpened():
        success, frame = cap.read()

        if success:
            # Run YOLOv8 inference on the frame
            results = model(frame)

            # Visualize the results on the frame
            annotated_frame = results[0].plot()

            # Display the frame in a separate window
            cv2.imshow("YOLOv8 Garbage Detection", annotated_frame)

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
