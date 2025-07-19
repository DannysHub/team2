
import socket
import cv2
import mediapipe as mp
import random
import time

WIN_SCORE = 15          # 获胜分数阈值
cheat_mode = False      # 是否开启作弊模式
cheat_probability = 0.3  # 作弊模式基础触发概率
MAX_SCORE=15
remote_mode = False        # 是否处于远程对战模式
remote_connected = False   # 是否已成功建立远程连接 
sock = None                # 网络套接字
is_host = False            # 标识当前玩家是否是主机
HOST = '0.0.0.0'           # 默认监听地址
PORT = 65432               # 通信端口

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
    global cheat_mode, cheat_probability
    
    # 检查是否应该进入作弊模式
    if not remote_connected and score >= CHEAT_START_SCORE and not cheat_mode:
        print("AI注意到你领先了，开始认真对待了!")
        cheat_mode = True
    
    # 作弊模式下的特殊处理
    if cheat_mode and random.random() < cheat_probability:
        # 如果AI要作弊（赢你），则调整结果
        if player == "rock":
            ai = "paper"  # 出布赢石头
        elif player == "paper":
            ai = "scissors"  # 出剪刀赢布
        elif player == "scissors":
            ai = "rock"  # 出石头赢剪刀
    
    # 原始胜负判断逻辑保持不变
    if player == ai:
        return "Draw", ai
    elif (player == "rock" and ai == "scissors") or \
         (player == "scissors" and ai == "paper") or \
         (player == "paper" and ai == "rock"):
        return "You Win!", ai
    else:
        return "You Lose!", ai
def get_local_ip():
    """获取本地真实的IPv4地址"""
    try:
        # 创建临时的UDP连接（不发送实际数据）
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))  # 连接到Google的DNS服务器
        ip_address = s.getsockname()[0]  # 获取连接使用的本地IP
        s.close()
        return ip_address
    except:
        # 备选方案：枚举所有接口
        interfaces = socket.getaddrinfo(socket.gethostname(), None)
        for interface in interfaces:
            # 筛选IPv4地址
            if interface[0] == socket.AF_INET:
                return interface[4][0]
        return "127.0.0.1"  # 终极回退
def setup_connection(is_host):
    global sock, remote_connected
    
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        
        if is_host:
            # 注意这里使用全局定义的HOST和PORT！
            sock.bind((HOST, PORT))
            sock.listen(1)
            
            # 获取并显示IP地址
           
            ip_address =get_local_ip()
            print(f"主机已创建! 您的IP地址: {ip_address}")
            print(f"请将本机IP地址告知他人: {ip_address}")
            print(f"等待玩家连接...")
            
            conn, addr = sock.accept()
            sock = conn
            print(f"玩家已连接! IP: {addr[0]}")
            return True
        else:
            # 加入游戏的代码...
            host_ip = input("请输入主机IP地址: ")
            sock.connect((host_ip, PORT))
            print("成功连接到主机!")
            return True
            
    except Exception as e:
        print(f"连接失败: {str(e)}")
        return False
print("===== 游戏模式选择 =====")
print("1 - 单人模式")
print("2 - 双人对战")
mode_choice = input(">>> ").strip()
remote_mode=(mode_choice=='2')
# 如果是远程模式，尝试建立连接
if remote_mode:
    print("尝试连接远程玩家...")
    try:
        # 获取用户角色选择
        choice = input("选择: 1)创建主机 2)加入游戏 >>> ").strip()
        
        # 处理选择
        if choice == '1':
            remote_connected = setup_connection(is_host=True)
        elif choice == '2':
            remote_connected = setup_connection(is_host=False)
        else:
            print("! 无效选择，将使用单人模式")
            remote_mode = False
            
        # 检查连接状态
        if not remote_connected:
            print("! 连接失败，将使用单人模式")
            remote_mode = False
            
    except Exception as e:
        print(f"连接过程中出错: {str(e)}")
        remote_mode = False
# 非远程模式则保持本地AI
if not remote_mode:
    print("启动单人模式")
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
        if 'remote_mode' in globals() and remote_mode and remote_connected:
            try:
               # 发送自己的手势
               sock.sendall(player_gesture.encode('utf-8'))

               # 接收对手手势（10秒超时）
               sock.settimeout(10.0)
               ai_gesture = sock.recv(1024).decode('utf-8')

              # 验证收到的手势是否合法
               if ai_gesture not in ["rock", "paper", "scissors"]:
                  raise ValueError("无效手势")

            except Exception as e:
              print(f"网络错误: {e}")
              ai_gesture = random.choice(["rock", "paper", "scissors"])
              result_text = "网络中断! 使用AI替代"
        else:
           ai_gesture = random.choice(["rock", "paper", "scissors"])
        if gesture_captured:
            if cheat_mode:
               result_text, ai_gesture = determine_winner(player_gesture, ai_gesture)
                # 每局结束稍微提高作弊概率
               cheat_probability = min(0.8, cheat_probability + 0.05)
            else:
               # 原始公平模式
                if player_gesture == ai_gesture:
                   result_text = "Draw"
                elif (player_gesture == "rock" and ai_gesture == "scissors") or \
                     (player_gesture == "scissors" and ai_gesture == "paper") or \
                     (player_gesture == "paper" and ai_gesture == "rock"):
                   result_text = "You Win!"
                else:
                     result_text = "You Lose!"

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
        opponent_name = "对手" if remote_connected else "AI"
        cv2.putText(frame, f"{opponent_name}: {ai_gesture}", (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        cv2.putText(frame, result_text, (10, 130),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)
        cv2.putText(frame, f"Score: {score} | Wins: {wins} | Losses: {losses}", (10, 470),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.putText(frame, "Press ENTER to play next round, ESC to quit", (10, 430),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (180, 180, 180), 2)
        cv2.imshow("Rock Paper Scissors - MediaPipe vs AI", frame)
        if not remote_connected and score >= MAX_SCORE:
    # 显示恭喜通关信息
           cv2.putText(frame, "CONGRATULATIONS!",
                (frame.shape[1]//2 - 250, frame.shape[0]//2 - 50),
                cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 255), 4)
           cv2.putText(frame, f"You've reached {score} points!",
                (frame.shape[1]//2 - 250, frame.shape[0]//2),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 255), 3)
           cv2.putText(frame, "Press any key to exit",
                (frame.shape[1]//2 - 200, frame.shape[0]//2 + 100),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (180, 180, 255), 2)
    
           cv2.imshow("Rock Paper Scissors - MediaPipe vs AI", frame)
    
           # 等待按键退出
           cv2.waitKey(0)
           cap.release()
           cv2.destroyAllWindows()
           print("\n🎉 Congratulations! You've won the game with 15 points! 🎉")
           exit()

           cv2.imshow("Rock Paper Scissors - MediaPipe vs AI", frame)


        while True:
            key = cv2.waitKey(0) & 0xFF
            if key == 13:  # Enter
                break
            elif key == 27:  # ESC
                cap.release()
                if remote_connected and sock:
                   sock.close()
                   print("网络连接已关闭")
                cv2.destroyAllWindows()
                exit()
