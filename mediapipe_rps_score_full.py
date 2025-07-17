
import cv2
import mediapipe as mp
import random
import time

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

def classify_gesture(landmarks):
    fingers = []
    tips_ids = [8, 12, 16, 20]
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

def determine_winner(player, ai):
    if player == ai:
        return "Draw"
    elif (player == "rock" and ai == "scissors") or          (player == "scissors" and ai == "paper") or          (player == "paper" and ai == "rock"):
        return "You Win!"
    else:
        return "You Lose!"

cap = cv2.VideoCapture(0)
score = 0
wins = 0
losses = 0

with mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7) as hands:
    result_text = ""
    ai_gesture = "None"
    player_gesture = "None"

    while cap.isOpened():
        # === Step 1: 倒计时准备 ===
        start_time = time.time()
        while time.time() - start_time < 3:
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.flip(frame, 1)
            countdown = 3 - int(time.time() - start_time)
            cv2.putText(frame, f"Get Ready: {countdown}", (50, 100),
                        cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 255), 4)
            cv2.putText(frame, f"Score: {score} | Wins: {wins} | Losses: {losses}", (10, 470),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            cv2.imshow("Rock Paper Scissors - MediaPipe vs AI", frame)
            if cv2.waitKey(1) & 0xFF == 27:
                cap.release()
                cv2.destroyAllWindows()
                exit()

        # === Step 2: 玩家出拳阶段（10秒） ===
        gesture_captured = False
        player_gesture = "None"
        capture_start = time.time()
        while time.time() - capture_start < 10:
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
                    current_gesture = classify_gesture(landmarks)
                    if current_gesture in ["rock", "paper", "scissors"]:
                        player_gesture = current_gesture
                        gesture_captured = True

            time_left = 10 - int(time.time() - capture_start)
            cv2.putText(frame, f"Show your gesture ({time_left})", (30, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)
            cv2.putText(frame, f"Score: {score} | Wins: {wins} | Losses: {losses}", (10, 470),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            cv2.imshow("Rock Paper Scissors - MediaPipe vs AI", frame)
            if cv2.waitKey(1) & 0xFF == 27:
                cap.release()
                cv2.destroyAllWindows()
                exit()

        # === Step 3: AI 出拳并判断结果 ===
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.flip(frame, 1)
        ai_gesture = random.choice(["rock", "paper", "scissors"])
        if gesture_captured:
            result_text = determine_winner(player_gesture, ai_gesture)
            if result_text == "You Win!":
                score += 3
                wins += 1
            elif result_text == "You Lose!":
                score -= 1
                losses += 1
        else:
            player_gesture = "None"
            result_text = "No gesture detected!"

        # === Step 4: 显示结果，并提示按Enter继续或Esc退出 ===
        cv2.putText(frame, f"You: {player_gesture}", (10, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
        cv2.putText(frame, f"AI: {ai_gesture}", (10, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        cv2.putText(frame, result_text, (10, 130),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)
        cv2.putText(frame, f"Score: {score} | Wins: {wins} | Losses: {losses}", (10, 470),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.putText(frame, "Press ENTER to play next round, ESC to quit", (10, 430),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (180, 180, 180), 2)
        cv2.imshow("Rock Paper Scissors - MediaPipe vs AI", frame)

        while True:
            key = cv2.waitKey(0) & 0xFF
            if key == 13:  # Enter
                break
            elif key == 27:  # ESC
                cap.release()
                cv2.destroyAllWindows()
                exit()
