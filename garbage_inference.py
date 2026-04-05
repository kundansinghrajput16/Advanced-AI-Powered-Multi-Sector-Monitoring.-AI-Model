import cv2
from ultralytics import YOLO
import os
import time

# Load trained model
model = YOLO("runs/detect/weapon_detector/weights/best.pt")

# Classes
class_names = ['1','2','3','4','5','6','7','8']

# Save path
save_path = "/Users/kundanrajsingh/Library/CloudStorage/GoogleDrive-kundansingh.iitm@gmail.com/My Drive/model_images_snapshot1/"
os.makedirs(save_path, exist_ok=True)

# Webcam
cap = cv2.VideoCapture("http://10.34.75.26:8080/video")

# Detection toggle
detect = True

# Store previous detections
previous_objects = set()

frame_count = 0

while True:

    ret, frame = cap.read()
    if not ret:
        break

    detected_objects = set()

    if detect:

        results = model(frame, conf=0.4)

        for r in results:
            for box in r.boxes:

                cls = int(box.cls[0])
                conf = float(box.conf[0])

                label = class_names[cls]
                detected_objects.add(label)

                x1, y1, x2, y2 = map(int, box.xyxy[0])

                cv2.rectangle(frame,(x1,y1),(x2,y2),(0,255,0),2)

                text = f"{label} {conf:.2f}"
                cv2.putText(frame,text,(x1,y1-10),
                            cv2.FONT_HERSHEY_SIMPLEX,0.6,(0,255,0),2)

        # Check for new objects
        new_objects = detected_objects - previous_objects

        if len(new_objects) > 0:

            frame_count += 1

            filename = os.path.join(save_path,f"detection_{frame_count}.jpg")

            cv2.imwrite(filename, frame)

            print("New garbage detected:", new_objects)

        previous_objects = detected_objects

    else:
        cv2.putText(frame,"Detection Paused",(20,40),
                    cv2.FONT_HERSHEY_SIMPLEX,1,(255,0,0),2)

    cv2.imshow("Garbage Detection", frame)

    key = cv2.waitKey(1) & 0xFF

    if key == ord('p'):
        detect = False
        print("Detection Paused")

    elif key == ord('s'):
        detect = True
        print("Detection Started")

    elif key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()