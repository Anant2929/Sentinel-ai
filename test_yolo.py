import cv2
from ultralytics import YOLO

# YOLOv8 nano model load kar rahe hain (real-time ke liye sabse fast)
# Pehli baar run karne par ye thodi der download lega (~3-4 MB)
model = YOLO('yolov8n.pt') 

# Default webcam start karo (0 matlab laptop ka camera)
cap = cv2.VideoCapture(0)

print("YOLO Vision start ho raha hai... Camera window ko band karne ke liye 'q' press karein.")

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        print("Webcam detect nahi hua!")
        break

    # Frame ko YOLO model me bhejo aur detection karo
    # verbose=False kiya hai taaki terminal spam na ho
    results = model(frame, stream=True, verbose=False)

    # Frame par bounding boxes aur labels draw karo
    for r in results:
        annotated_frame = r.plot() 

    # Bounding boxes wala frame screen par dikhao
    cv2.imshow("Sentinel AI - YOLOv8 Object Detection", annotated_frame)

    # 'q' dabane par exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Camera aur windows close karo
cap.release()
cv2.destroyAllWindows()