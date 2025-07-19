
import cv2
import mediapipe as mp
import time
import numpy as np
import csv
import pyttsx3
from datetime import datetime
import matplotlib.pyplot as plt
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
import os

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
engine = pyttsx3.init()
engine.setProperty('rate', 160)

STANDARD_GESTURE = {
    4: (0.3, 0.3),
    8: (0.5, 0.2),
    12: (0.6, 0.2),
    16: (0.7, 0.2),
    20: (0.8, 0.2),
}
FINGER_NAMES = {
    4: "Thumb",
    8: "Index",
    12: "Middle",
    16: "Ring",
    20: "Pinky"
}
CSV_FILE = "rehab_records.csv"
history = []

def compute_finger_deviations(landmarks):
    deviations = {}
    for idx, (std_x, std_y) in STANDARD_GESTURE.items():
        if idx < len(landmarks):
            lm = landmarks[idx]
            distance = np.sqrt((lm.x - std_x)**2 + (lm.y - std_y)**2)
            deviations[idx] = distance
    return deviations

def overall_completion(deviations):
    if not deviations:
        return 0.0
    similarities = [max(0, 1 - d * 5) for d in deviations.values()]
    return round((sum(similarities) / len(similarities)) * 100, 2)

def speak(text):
    engine.say(text)
    engine.runAndWait()

def save_result_to_csv(timestamp, completion):
    with open(CSV_FILE, mode='a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([timestamp, completion])

def plot_history():
    if len(history) < 2:
        return
    times = list(range(1, len(history)+1))
    plt.figure()
    plt.plot(times, history, marker='o')
    plt.title("Rehab Completion History")
    plt.xlabel("Session")
    plt.ylabel("Completion (%)")
    plt.ylim(0, 100)
    plt.grid(True)
    plt.savefig("completion_history.png")
    plt.close()

with open(CSV_FILE, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(["Time", "Completion"])

cap = cv2.VideoCapture(0)

with mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7) as hands:
    while cap.isOpened():
        speak("Get ready for rehab training")
        start_time = time.time()
        while time.time() - start_time < 3:
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.flip(frame, 1)
            countdown = 3 - int(time.time() - start_time)
            cv2.putText(frame, f"Start in: {countdown}", (50, 100),
                        cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 255), 4)
            cv2.imshow("Hand Rehab - Preparation", frame)
            if cv2.waitKey(1) & 0xFF == 27:
                cap.release()
                cv2.destroyAllWindows()
                exit()

        speak("Start now")
        best_deviation = {}
        max_completion = 0.0
        best_frame = None
        capture_start = time.time()
        while time.time() - capture_start < 10:
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.flip(frame, 1)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(rgb)

            if results.multi_hand_landmarks:
                hand_landmarks = results.multi_hand_landmarks[0]
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                landmarks = hand_landmarks.landmark
                deviations = compute_finger_deviations(landmarks)
                completion = overall_completion(deviations)
                if completion > max_completion:
                    max_completion = completion
                    best_deviation = deviations.copy()
                    best_frame = frame.copy()

                y0 = 140
                for idx, dev in deviations.items():
                    color = (0, 255, 0) if dev < 0.05 else (0, 0, 255)
                    cv2.putText(frame, f"{FINGER_NAMES[idx]}: {dev:.3f}", (10, y0),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
                    y0 += 30

            time_left = 10 - int(time.time() - capture_start)
            cv2.putText(frame, f"Rehab Action ({time_left}s)", (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            cv2.putText(frame, f"Best Match: {max_completion:.2f}%", (10, 100),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 200, 255), 2)
            cv2.imshow("Hand Rehab - Tracking", frame)
            if cv2.waitKey(1) & 0xFF == 27:
                cap.release()
                cv2.destroyAllWindows()
                exit()

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        screenshot_file = f"screenshot_{timestamp}.png"
        if best_frame is not None:
            cv2.imwrite(screenshot_file, best_frame)

        save_result_to_csv(timestamp, max_completion)
        history.append(max_completion)
        plot_history()

        report_file = f"rehab_report_{timestamp}.pdf"
        c = canvas.Canvas(report_file, pagesize=letter)
        c.setFont("Helvetica-Bold", 16)
        c.drawString(50, 750, "Rehabilitation Training Report")
        c.setFont("Helvetica", 12)
        c.drawString(50, 730, f"Time: {timestamp}")
        c.drawString(50, 710, f"Completion: {max_completion:.2f}%")
        if best_frame is not None:
            c.drawString(50, 690, "Best Frame:")
            c.drawImage(screenshot_file, 50, 450, width=200, height=200)

        # ✅ 插入历史图时检查文件是否存在
        history_img = "completion_history.png"
        if os.path.exists(history_img):
            c.drawString(300, 690, "Completion History:")
            c.drawImage(history_img, 300, 450, width=200, height=200)
        else:
            c.drawString(300, 690, "Completion history not available")

        c.save()

        speak(f"Session completed. Your match is {int(max_completion)} percent")
        if max_completion > 90:
            speak("Excellent work!")
        elif max_completion > 70:
            speak("Good job. Keep improving.")
        else:
            speak("Try again. You can do it.")

        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.flip(frame, 1)
        y0 = 180
        for idx, dev in best_deviation.items():
            color = (0, 255, 0) if dev < 0.05 else (0, 0, 255)
            cv2.putText(frame, f"{FINGER_NAMES[idx]} Deviation: {dev:.3f}", (10, y0),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
            y0 += 30

        cv2.putText(frame, "Rehab Complete!", (10, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 0), 3)
        cv2.putText(frame, f"Final Completion: {max_completion:.2f}%", (10, 130),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 100), 3)
        cv2.putText(frame, "Press ENTER to continue or ESC to exit", (10, 460),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200, 200, 200), 2)
        cv2.imshow("Hand Rehab - Result", frame)

        while True:
            key = cv2.waitKey(0) & 0xFF
            if key == 13:
                break
            elif key == 27:
                cap.release()
                cv2.destroyAllWindows()
                exit()
