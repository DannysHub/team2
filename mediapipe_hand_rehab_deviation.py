
import cv2
import mediapipe as mp
import time
import numpy as np

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# 标准康复动作关键点（示意为五指张开）
STANDARD_GESTURE = {
    4: (0.3, 0.3),   # Thumb tip
    8: (0.5, 0.2),   # Index tip
    12: (0.6, 0.2),  # Middle tip
    16: (0.7, 0.2),  # Ring tip
    20: (0.8, 0.2),  # Pinky tip
}
FINGER_NAMES = {
    4: "Thumb",
    8: "Index",
    12: "Middle",
    16: "Ring",
    20: "Pinky"
}

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

cap = cv2.VideoCapture(0)

with mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7) as hands:
    while cap.isOpened():
        # Step 1: 3秒倒计时
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

        # Step 2: 10秒识别
        best_deviation = {}
        max_completion = 0.0
        capture_start = time.time()
        while time.time() - capture_start < 10:
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.flip(frame, 1)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(rgb)
            landmarks = None

            if results.multi_hand_landmarks:
                hand_landmarks = results.multi_hand_landmarks[0]
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                landmarks = hand_landmarks.landmark
                deviations = compute_finger_deviations(landmarks)
                completion = overall_completion(deviations)
                if completion > max_completion:
                    max_completion = completion
                    best_deviation = deviations.copy()

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

        # Step 3: 显示最终完成度和每个手指偏差
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

        cv2.putText(frame, f"Rehab Complete!", (10, 80),
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
