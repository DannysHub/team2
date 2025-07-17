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
    gesture_stable_timer = 0
    GESTURE_STABLE_TIME = 1.0  
    ai_played = False  
    last_player_gesture = "None"

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
            # ... [绘制关键点代码] ...
                landmarks = hand_landmarks.landmark
                current_gesture = classify_gesture(landmarks)

            # 只处理有效手势
                if current_gesture in ["rock", "paper", "scissors"]:
                # 手势识别阶段
                    if not ai_played:
                    # 更新当前识别到的手势
                       player_gesture = current_gesture
                    
                    # 手势稳定检测
                    if current_gesture == last_player_gesture:
                        gesture_stable_timer += 0.03  # 每帧约33ms
                    else:
                        gesture_stable_timer = 0
                        last_player_gesture = current_gesture
                    
                    # 显示倒计时条
                    progress_width = int((gesture_stable_timer / GESTURE_STABLE_TIME) * 200)
                    cv2.rectangle(frame, (50, 10), (50 + progress_width, 30), (0, 255, 0), -1)
                    cv2.putText(frame, "Hold your gesture...", (10, 170), 
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                    
                    # 手势稳定后触发AI出拳
                    if gesture_stable_timer >= GESTURE_STABLE_TIME:
                        ai_gesture = random.choice(["rock", "paper", "scissors"])
                        last_result = determine_winner(player_gesture, ai_gesture)
                        ai_played = True  # 标记已出拳
                
                    # 显示结果阶段
                    else:
                      cv2.putText(frame, last_result, (10, 130), 
                                 cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)
                    # 显示重置提示
                      cv2.putText(frame, "Release hand for next round", (10, 170), 
                                 cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                else:
                    # 检测到"unknown"手势时视为玩家放下手
                    ai_played = False
                    gesture_stable_timer = 0
                    last_player_gesture = "None"

        else:
        # 未检测到手时重置状态
            ai_played = False
            gesture_stable_timer = 0
            last_player_gesture = "None"
            player_gesture = "None"

        # 显示信息（总是显示）
        cv2.putText(frame, f"You: {player_gesture}", (10, 40), 
                 cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
        if ai_played:
            cv2.putText(frame, f"AI: {ai_gesture}", (10, 80), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        else:
            cv2.putText(frame, "AI: ?", (10, 80), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    
        cv2.imshow("Rock Paper Scissors - MediaPipe vs AI", frame)
    
        key = cv2.waitKey(10)
        if key == 27:  # ESC退出
            break

cap.release()
cv2.destroyAllWindows()
