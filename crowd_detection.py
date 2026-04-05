import cv2
from ultralytics import YOLO
import time
import os

# Load YOLOv8 model
model = YOLO("yolov8n.pt")

# Open webcam
cap = cv2.VideoCapture("http://10.34.75.26:8080/video")

# Save folder path
save_path = "/Users/kundanrajsingh/Library/CloudStorage/GoogleDrive-kundansingh.iitm@gmail.com/My Drive/model_images_snapshot2/"
os.makedirs(save_path, exist_ok=True)

# Detection toggle
detect = True

# Alert cooldown
alert_cooldown = 10
last_alert = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    person_count = 0

    # Run detection only if enabled
    if detect:
        results = model(frame)

        for r in results:
            for box in r.boxes:
                cls = int(box.cls[0])

                # class 0 = person
                if cls == 0:
                    person_count += 1

                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    cv2.rectangle(frame,(x1,y1),(x2,y2),(0,255,0),2)

        # Display count
        cv2.putText(frame,f"Persons: {person_count}",(20,40),
                    cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,255),2)

        # Crowd alert
        if person_count > 3 and (time.time() - last_alert > alert_cooldown):

            filename = os.path.join(save_path, f"crowd_{int(time.time())}.jpg")
            cv2.imwrite(filename, frame)

            print("ALERT: Crowd detected! Saved:", filename)

            last_alert = time.time()

    else:
        cv2.putText(frame,"Detection Paused",(20,40),
                    cv2.FONT_HERSHEY_SIMPLEX,1,(255,0,0),2)

    cv2.imshow("Crowd Detection", frame)

    # Keyboard control
    key = cv2.waitKey(1) & 0xFF

    if key == ord('p'):
        detect = False
        print("Detection Paused")

    elif key == ord('s'):
        detect = True
        print("Detection Started")

    elif key == ord('q'):
        print("Exiting...")
        break

cap.release()
cv2.destroyAllWindows()