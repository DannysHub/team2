
import cv2
import mediapipe as mp
import random

# 初始化 MediaPipe Hands
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# 手势分类函数（根据手指张开数量）
def classify_gesture(landmarks):
    fingers = []
    tips_ids = [8, 12, 16, 20]  # 四个手指指尖 landmark index

    for tip in tips_ids:
        if landmarks[tip].y < landmarks[tip - 2].y:
            fingers.append(1)
        else:
            fingers.append(0)

    count = sum(fingers)
    if count == 0:
        return "rock"
    elif count == 2:
        return "scissors"
    elif count >= 4:
        return "paper"
    else:
        return "unknown"

# 胜负判断逻辑
def determine_winner(player, ai):
    if player == ai:
        return "Draw"
    elif (player == "rock" and ai == "scissors") or          (player == "scissors" and ai == "paper") or          (player == "paper" and ai == "rock"):
        return "You Win!"
    else:
        return "You Lose!"

# 启动摄像头
cap = cv2.VideoCapture(0)

with mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7) as hands:
    last_result = ""
    player_gesture = "None"
    ai_gesture = "None"

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                landmarks = hand_landmarks.landmark
                player_gesture = classify_gesture(landmarks)
                if player_gesture in ["rock", "paper", "scissors"]:
                    ai_gesture = random.choice(["rock", "paper", "scissors"])
                    last_result = determine_winner(player_gesture, ai_gesture)

        # 显示结果
        cv2.putText(frame, f"You: {player_gesture}", (10, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
        cv2.putText(frame, f"AI: {ai_gesture}", (10, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        cv2.putText(frame, last_result, (10, 130),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)

        cv2.imshow("Rock Paper Scissors - MediaPipe vs AI", frame)

        key = cv2.waitKey(10)
        if key == 27:  # ESC退出
            break

cap.release()
cv2.destroyAllWindows()
