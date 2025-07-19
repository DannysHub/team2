
import cv2
import mediapipe as mp
import time
import numpy as np

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# 设定标准康复动作关键点（这里只是示例值，实际应从真实训练数据中提取）
STANDARD_GESTURE = {
    8: (0.5, 0.2),   # index fingertip
    12: (0.6, 0.2),  # middle fingertip
    16: (0.7, 0.2),  # ring fingertip
    20: (0.8, 0.2),  # pinky fingertip
    4: (0.3, 0.3),   # thumb tip
}

def compute_completion(landmarks):
    if landmarks is None:
        return 0.0

    total_similarity = 0
    count = 0
    for idx, (std_x, std_y) in STANDARD_GESTURE.items():
        if idx < len(landmarks):
            lm = landmarks[idx]
            distance = np.sqrt((lm.x - std_x)**2 + (lm.y - std_y)**2)
            similarity = max(0, 1 - distance * 5)  # 转换为相似度
            total_similarity += similarity
            count += 1
    return round((total_similarity / count) * 100, 2) if count > 0 else 0.0

cap = cv2.VideoCapture(0)

with mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7) as hands:
    while cap.isOpened():
        # Step 1: 显示3秒准备倒计时
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

        # Step 2: 开始10秒识别和分析
        final_completion = 0.0
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

            completion = compute_completion(landmarks)
            final_completion = max(final_completion, completion)  # 保存最好的一帧
            time_left = 10 - int(time.time() - capture_start)
            cv2.putText(frame, f"Complete the rehab action ({time_left})", (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(frame, f"Current Match: {completion:.2f}%", (10, 100),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 200, 255), 2)
            cv2.imshow("Hand Rehab - Matching", frame)
            if cv2.waitKey(1) & 0xFF == 27:
                cap.release()
                cv2.destroyAllWindows()
                exit()

        # Step 3: 显示最终完成度结果
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.flip(frame, 1)
        cv2.putText(frame, f"Rehab action completed!", (10, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 0), 3)
        cv2.putText(frame, f"Final Completion: {final_completion:.2f}%", (10, 140),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 100), 3)
        cv2.putText(frame, "Press ENTER for next round or ESC to exit", (10, 450),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200, 200, 200), 2)
        cv2.imshow("Hand Rehab - Result", frame)

        while True:
            key = cv2.waitKey(0) & 0xFF
            if key == 13:  # ENTER
                break
            elif key == 27:  # ESC
                cap.release()
                cv2.destroyAllWindows()
                exit()
