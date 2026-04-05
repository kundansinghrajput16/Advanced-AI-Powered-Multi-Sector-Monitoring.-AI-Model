import cv2
from ultralytics import YOLO
import time

# Load models
person_model = YOLO("yolov8n.pt")
weapon_model = YOLO("runs/detect/weapon_detector/weights/best.pt")

# Save path
drive_path = "/Users/kundanrajsingh/Library/CloudStorage/GoogleDrive-kundansingh.iitm@gmail.com/My Drive/model_images_snapshot/"

cap = cv2.VideoCapture("http://172.17.64.71:8080/video")

last_capture_time = 0
cooldown = 5

while True:

    ret, frame = cap.read()
    if not ret:
        break

    height, width, _ = frame.shape

    person_results = person_model(frame)
    weapon_results = weapon_model(frame)

    persons = []
    weapons = []

    # PERSON DETECTION
    for r in person_results:
        for box in r.boxes:

            cls = int(box.cls[0])
            label = person_model.names[cls]
            conf = float(box.conf[0])

            if label == "person" and conf > 0.6:

                x1, y1, x2, y2 = map(int, box.xyxy[0])
                person_height = y2 - y1

                # full body condition
                if person_height > height * 0.6 and x1 > 10 and x2 < width - 10:

                    persons.append((x1, y1, x2, y2))

                    cv2.rectangle(frame,(x1,y1),(x2,y2),(0,255,0),2)
                    cv2.putText(frame,"Person",(x1,y1-10),
                                cv2.FONT_HERSHEY_SIMPLEX,0.7,(0,255,0),2)

    # WEAPON DETECTION
    for r in weapon_results:
        for box in r.boxes:

            cls = int(box.cls[0])
            label = weapon_model.names[cls]
            conf = float(box.conf[0])

            if label in ["knife","handgun","rifle"] and conf > 0.6:

                x1, y1, x2, y2 = map(int, box.xyxy[0])

                weapons.append((x1, y1, x2, y2))

                cv2.rectangle(frame,(x1,y1),(x2,y2),(0,0,255),2)
                cv2.putText(frame,label,(x1,y1-10),
                            cv2.FONT_HERSHEY_SIMPLEX,0.7,(0,0,255),2)

    # DETERMINE ALERT TYPE
    person_detected = len(persons) > 0
    weapon_detected = len(weapons) > 0

    alert_type = None

    if person_detected and weapon_detected:
        alert_type = "person_with_weapon"
    elif person_detected:
        alert_type = "person"
    elif weapon_detected:
        alert_type = "weapon"

    # SAVE SNAPSHOT
    current_time = time.time()

    if alert_type and current_time - last_capture_time > cooldown:

        filename = drive_path + f"{alert_type}_{int(current_time)}.jpg"
        cv2.imwrite(filename, frame)

        print(f"Snapshot saved: {alert_type}")

        last_capture_time = current_time

    cv2.imshow("Security Detection", frame)

    if cv2.waitKey(1) == 27:
        break

cap.release()
cv2.destroyAllWindows()