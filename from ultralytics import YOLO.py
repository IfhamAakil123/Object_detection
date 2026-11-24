import cv2
from ultralytics import YOLO

# Load the YOLOv8 model (downloads automatically the first time)
model = YOLO("yolov8n.pt")   # Nano model (fastest)

# Open webcam
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

while True:
    ret, frame = cap.read()

    if not ret:
        break

    # Perform object detection
    results = model(frame)

    # Draw detections on the frame
    annotated_frame = results[0].plot()

    # Display the result
    cv2.imshow("YOLOv8 Object Detection", annotated_frame)

    # Press Q to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()