import cv2
import mediapipe as mp

# MediaPipe Pose module load kar rahe hain
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)

cap = cv2.VideoCapture(0)
print("MediaPipe Pose start ho raha hai... Camera window band karne ke liye 'q' dabayein.")

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        print("Webcam detect nahi hua!")
        break
    
    # MediaPipe ko BGR ki jagah RGB image chahiye hoti hai
    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(image_rgb)
    
    # Agar frame me koi insaan dikha, toh uski haddiyan (skeleton) draw karo
    if results.pose_landmarks:
        mp_drawing.draw_landmarks(
            frame, 
            results.pose_landmarks, 
            mp_pose.POSE_CONNECTIONS,
            mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=2), # Dots
            mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2)  # Lines
        )
        
    cv2.imshow("Sentinel AI - MediaPipe Pose Tracking", frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()