#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Rock-Paper-Scissors + 手势完成度评分 (融合 simple_version_2.1 Rehab 逻辑)
---------------------------------------------------------------------------
功能:
  - RPS 游戏 (本地或远程)
  - 手势自动分类 (rock / paper / scissors)
  - 模板采集 (按 't') & 形状匹配 (指尖欧氏距离指数衰减)
  - 张开度 / 关闭度 / 中间偏好 / (可扩展 pinch)
  - 综合完成度分数 Total, 同时显示 Shape / Open%
  - 平滑窗口(默认5帧)降低波动
  - 每回合最佳完成度写入 rps_completion_records.csv
  - 可选截图 (刷新最佳时)

依赖:
  pip install mediapipe opencv-python numpy
"""

import socket
import cv2
import mediapipe as mp
import random
import time
import os
import csv
import json
import datetime
from collections import deque
from pathlib import Path
import math
import re

# ====================== RPS 基础参数 ======================
WIN_SCORE        = 15
MAX_SCORE        = 15
CHEAT_START_SCORE = 10
cheat_mode       = False
cheat_probability = 0.3

remote_mode      = False
remote_connected = False
sock             = None
HOST             = '0.0.0.0'
PORT             = 65432

# ====================== Rehab 评分参数（来自 simple_version_2.1） ======================
MODEL_TEMPLATE_FILE = "gesture_templates.json"
RPS_CSV_FILE        = "rps_completion_records.csv"

FINGERTIPS   = [4, 8, 12, 16, 20]          # 指尖
PALM_POINTS  = [0, 5, 9, 13, 17]           # 掌心近似点
SHAPE_WEIGHT = 0.7
OPENNESS_WEIGHT = 0.3
DEVIATION_GREEN_THRESHOLD = 0.05
SMOOTH_WINDOW = 5

# 手势映射 (RPS -> Rehab 基础手势类别)
RPS_TO_REHAB = {
    "rock": "fist",
    "paper": "open",
    "scissors": "scissors"   # 特例，在 gesture_total 内单独处理
}

# 是否在刷新最佳总分时截屏
SAVE_SNAPSHOT_ON_BEST = False

# ====================== MediaPipe Hands ======================
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# ====================== 模板加载/保存 ======================
def load_templates():
    if not os.path.exists(MODEL_TEMPLATE_FILE):
        with open(MODEL_TEMPLATE_FILE, "w") as f:
            json.dump({"open": {}, "fist": {}, "pinch": {}, "scissors": {}}, f, indent=2)
        return {"open": {}, "fist": {}, "pinch": {}, "scissors": {}}
    with open(MODEL_TEMPLATE_FILE, "r") as f:
        data = json.load(f)
    # 补齐键
    changed = False
    for k in ("open","fist","pinch","scissors"):
        if k not in data:
            data[k] = {}
            changed = True
    if changed:
        with open(MODEL_TEMPLATE_FILE, "w") as f:
            json.dump(data, f, indent=2)
    return data

def save_templates(tpl):
    with open(MODEL_TEMPLATE_FILE, "w") as f:
        json.dump(tpl, f, indent=2)

gesture_templates = load_templates()

def has_template(name):
    return name in gesture_templates and len(gesture_templates[name]) > 0

# ====================== Rehab 几何与评分函数 ======================
def palm_center_and_width(lm_list):
    cx = sum(lm_list[i].x for i in PALM_POINTS) / len(PALM_POINTS)
    cy = sum(lm_list[i].y for i in PALM_POINTS) / len(PALM_POINTS)
    w_ref = math.dist((lm_list[5].x, lm_list[5].y),
                      (lm_list[17].x, lm_list[17].y)) or 1e-6
    return cx, cy, w_ref

def normalize_landmarks(lm_list):
    cx, cy, w_ref = palm_center_and_width(lm_list)
    return {i: ((lm.x - cx)/w_ref, (lm.y - cy)/w_ref) for i, lm in enumerate(lm_list)}

def openness_ratio(lm_list):
    cx, cy, w_ref = palm_center_and_width(lm_list)
    ds = [math.dist((lm_list[t].x, lm_list[t].y), (cx, cy))/w_ref for t in FINGERTIPS]
    avg = sum(ds)/len(ds)
    return min(1.0, avg / 0.9)

def compute_devs(norm, template_dict):
    devs = {}
    for k, (tx, ty) in template_dict.items():
        idx = int(k)
        if idx in norm:
            x, y = norm[idx]
            devs[idx] = math.hypot(x - tx, y - ty)
    return devs

def finger_weight(idx, gesture):
    if gesture == "open":
        return 1.2 if idx == 4 else 1.0
    if gesture == "fist":
        return 1.1 if idx in (8, 12) else 1.0
    if gesture == "pinch":
        return 1.5 if idx in (4, 8) else 0.5
    if gesture == "scissors":
        # 让食指/中指更重要
        return 1.3 if idx in (8, 12) else 0.8
    return 1.0

def shape_score(devs, gesture):
    if not devs: return 0.0
    sims = []
    for idx, d in devs.items():
        s = math.exp(-8*d) * finger_weight(idx, gesture)
        sims.append(s)
    return (sum(sims)/len(sims))*100.0

def gesture_total(gesture, shape_part, open_part, lm_list):
    if gesture == "open":
        return SHAPE_WEIGHT*shape_part + OPENNESS_WEIGHT*open_part
    if gesture == "fist":
        closure = 100 - open_part
        return 0.6*shape_part + 0.4*closure
    if gesture == "pinch":
        pinch_score = 0.0
        if lm_list:
            pt4, pt8 = lm_list[4], lm_list[8]
            dist = math.hypot(pt4.x - pt8.x, pt4.y - pt8.y)
            _, _, w_ref = palm_center_and_width(lm_list)
            nd = dist / w_ref
            pinch_score = math.exp(-8*abs(nd - 0.15))*100.0
        mid_pref = 100 - abs(open_part - 50)
        return 0.5*shape_part + 0.3*pinch_score + 0.2*mid_pref
    if gesture == "scissors":
        mid_pref = 100 - abs(open_part - 50)
        return 0.5*shape_part + 0.5*mid_pref
    return SHAPE_WEIGHT*shape_part + OPENNESS_WEIGHT*open_part

# ====================== CSV 初始化/记录 ======================
def init_rps_csv():
    if not os.path.exists(RPS_CSV_FILE):
        with open(RPS_CSV_FILE, "w", newline="") as f:
            csv.writer(f).writerow([
                "Time","Round","PlayerGesture","OpponentGesture","Result",
                "Total","Shape","Open","Score","Wins","Losses","Draws"
            ])

def append_rps_csv(ts, rnd, player, opponent, result, total, shape, open_part, score, wins, losses, draws):
    with open(RPS_CSV_FILE, "a", newline="") as f:
        csv.writer(f).writerow([
            ts, rnd, player, opponent, result,
            f"{total:.2f}", f"{shape:.2f}", f"{open_part:.2f}",
            score, wins, losses, draws
        ])

# ====================== 其他辅助 ======================
def classify_gesture(landmarks):
    """简易基于指尖伸展数量的 RPS 分类"""
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
    global cheat_mode, cheat_probability, score
    if not remote_connected and score >= CHEAT_START_SCORE and not cheat_mode:
        print("AI注意到你领先了，开始认真对待了!")
        cheat_mode = True
    if cheat_mode and random.random() < cheat_probability:
        if player == "rock": ai = "paper"
        elif player == "paper": ai = "scissors"
        elif player == "scissors": ai = "rock"
    if player == ai:
        return "Draw", ai
    elif (player == "rock" and ai == "scissors") or \
         (player == "scissors" and ai == "paper") or \
         (player == "paper" and ai == "rock"):
        return "You Win!", ai
    else:
        return "You Lose!", ai

def get_local_ip():
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip_address = s.getsockname()[0]
        s.close()
        return ip_address
    except:
        return "127.0.0.1"

def setup_connection(is_host):
    global sock, remote_connected
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        if is_host:
            sock.bind((HOST, PORT))
            sock.listen(1)
            ip_address = get_local_ip()
            print(f"主机已创建! IP: {ip_address}")
            print("等待玩家连接...")
            conn, addr = sock.accept()
            sock = conn
            print(f"玩家已连接: {addr[0]}")
            return True
        else:
            host_ip = input("请输入主机IP地址: ")
            sock.connect((host_ip, PORT))
            print("成功连接到主机!")
            return True
    except Exception as e:
        print(f"连接失败: {e}")
        return False

def save_rehab_snapshot(frame, gesture, round_count, hand_bbox):
    try:
        desktop = Path.home() / "Desktop"
        desktop.mkdir(exist_ok=True)
        safe_gesture = re.sub(r'[\\/:*?"<>|]', "_", gesture)
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"rps_{round_count}_{timestamp}_{safe_gesture}.jpg"
        annotated = frame.copy()
        if hand_bbox:
            x1,y1,x2,y2 = hand_bbox
            cv2.rectangle(annotated,(x1,y1),(x2,y2),(0,255,0),2)
        cv2.putText(annotated, f"{gesture}", (10,30),
                    cv2.FONT_HERSHEY_SIMPLEX,0.8,(0,255,0),2)
        path = desktop / filename
        if cv2.imwrite(str(path), annotated):
            print(f"✅ 截图保存: {path}")
        else:
            print("❌ 截图失败")
    except Exception as e:
        print("截图异常:", e)

# ====================== 主流程 ======================
def main():
    global remote_mode, remote_connected, cheat_mode, cheat_probability, score

    print("===== 游戏模式选择 =====")
    print("1 - 单人模式")
    print("2 - 双人对战")
    mode_choice = input(">>> ").strip()
    remote_mode = (mode_choice == '2')

    if remote_mode:
        choice = input("选择: 1)创建主机 2)加入游戏 >>> ").strip()
        if choice == '1':
            remote_connected = setup_connection(is_host=True)
        elif choice == '2':
            remote_connected = setup_connection(is_host=False)
        else:
            print("无效选择，退回单人模式")
            remote_mode = False
        if not remote_connected:
            print("连接失败，退回单人模式")
            remote_mode = False
    if not remote_mode:
        print("启动单人模式")

    init_rps_csv()

    cap = cv2.VideoCapture(0)
    score = 0
    wins = losses = draws = 0
    round_count = 0

    total_window = deque(maxlen=SMOOTH_WINDOW)

    with mp_hands.Hands(max_num_hands=1,
                        min_detection_confidence=0.7,
                        min_tracking_confidence=0.5) as hands:

        while cap.isOpened():
            round_count += 1

            # -------- 准备阶段 --------
            start_time = time.time()
            while time.time() - start_time < 3:
                ret, frame = cap.read()
                if not ret: break
                frame = cv2.flip(frame, 1)
                countdown = 3 - int(time.time() - start_time)
                cv2.putText(frame, f"Round {round_count}  Get Ready: {countdown}",
                            (30,100), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0,255,255), 3)
                cv2.putText(frame, f"Score:{score} W:{wins} L:{losses} D:{draws}",
                            (10,470), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255,255,255), 2)
                cv2.imshow("RPS + Completion", frame)
                if cv2.waitKey(1) & 0xFF == 27:
                    cap.release(); cv2.destroyAllWindows(); return

            # -------- 出拳采集阶段 --------
            capture_start = time.time()
            player_gesture = "None"
            gesture_captured = False

            best_total = 0.0
            best_shape = 0.0
            best_open = 0.0
            best_frame = None
            best_rehab_gesture = None
            best_dev = {}

            while time.time() - capture_start < 10:
                ret, frame = cap.read()
                if not ret: break
                frame = cv2.flip(frame, 1)
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = hands.process(rgb)

                current_total = 0.0
                shape_part = 0.0
                open_part = 0.0
                devs = {}

                if results.multi_hand_landmarks:
                    lm_list = results.multi_hand_landmarks[0].landmark
                    cur_rps = classify_gesture(lm_list)
                    if cur_rps in ("rock","paper","scissors"):
                        player_gesture = cur_rps
                        gesture_captured = True

                    ktemp = cv2.waitKey(1) & 0xFF
                    if ktemp == ord('t') and player_gesture != "None":
                        rehab_type = RPS_TO_REHAB.get(player_gesture, "open")
                        norm_cap = normalize_landmarks(lm_list)
                        gesture_templates[rehab_type] = {
                            str(i): norm_cap[i] for i in FINGERTIPS if i in norm_cap
                        }
                        save_templates(gesture_templates)
                        print(f"[模板更新] {rehab_type}")

                    rehab_type = RPS_TO_REHAB.get(player_gesture, "open")
                    norm = normalize_landmarks(lm_list)
                    if has_template(rehab_type):
                        devs = compute_devs(norm, gesture_templates[rehab_type])
                        shape_part = shape_score(devs, rehab_type)
                    open_part = openness_ratio(lm_list) * 100.0
                    current_total = gesture_total(rehab_type, shape_part, open_part, lm_list)

                    total_window.append(current_total)
                    smooth_total = sum(total_window)/len(total_window)
                    current_total = smooth_total

                    if current_total > best_total:
                        best_total = current_total
                        best_shape = shape_part
                        best_open = open_part
                        best_frame = frame.copy()
                        best_rehab_gesture = rehab_type
                        best_dev = devs.copy()
                        if SAVE_SNAPSHOT_ON_BEST:
                            h,w,_ = frame.shape
                            xs = [lm.x*w for lm in lm_list]
                            ys = [lm.y*h for lm in lm_list]
                            x1,x2 = int(min(xs)), int(max(xs))
                            y1,y2 = int(min(ys)), int(max(ys))
                            save_rehab_snapshot(frame, player_gesture, round_count,
                                                (x1,y1,x2,y2))

                    mp_drawing.draw_landmarks(frame, results.multi_hand_landmarks[0], mp_hands.HAND_CONNECTIONS)

                    y0 = 170
                    for tip in FINGERTIPS:
                        if tip in devs:
                            d = devs[tip]
                            color = (0,255,0) if d < DEVIATION_GREEN_THRESHOLD else (0,0,255)
                            cv2.putText(frame, f"{tip}:{d:.3f}", (20,y0),
                                        cv2.FONT_HERSHEY_SIMPLEX,0.5,color,1)
                            y0 += 18

                    cv2.putText(frame, f"Shape:{shape_part:.1f}% Open:{open_part:.1f}%",
                                (20,140), cv2.FONT_HERSHEY_SIMPLEX,0.55,(0,200,255),2)
                    cv2.putText(frame, f"Total:{current_total:.1f}% (Best:{best_total:.1f}%)",
                                (20,115), cv2.FONT_HERSHEY_SIMPLEX,0.55,(255,255,255),2)

                time_left = 10 - int(time.time() - capture_start)
                cv2.putText(frame, f"Show your gesture ({time_left})",
                            (20,60), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,255,0), 2)
                cv2.putText(frame, f"Current: {player_gesture}",
                            (20,35), cv2.FONT_HERSHEY_SIMPLEX,0.7,(255,255,0),2)
                cv2.putText(frame, f"Score:{score} W:{wins} L:{losses} D:{draws}",
                            (10,470), cv2.FONT_HERSHEY_SIMPLEX,0.8,(255,255,255),2)
                cv2.putText(frame, "[t]模板  ESC退出",
                            (400,470), cv2.FONT_HERSHEY_SIMPLEX,0.6,(180,180,180),1)

                cv2.imshow("RPS + Completion", frame)
                if cv2.waitKey(1) & 0xFF == 27:
                    cap.release(); cv2.destroyAllWindows(); return

            # -------- AI / 对手出拳与判定 --------
            ret, frame = cap.read()
            if not ret: break
            frame = cv2.flip(frame, 1)

            if remote_mode and remote_connected:
                try:
                    sock.sendall(player_gesture.encode('utf-8'))
                    sock.settimeout(10.0)
                    ai_gesture = sock.recv(1024).decode('utf-8')
                    if ai_gesture not in ["rock","paper","scissors"]:
                        raise ValueError("无效手势")
                except Exception as e:
                    print("网络错误:", e)
                    ai_gesture = random.choice(["rock","paper","scissors"])
                    result_text = "网络中断! 使用AI替代"
            else:
                ai_gesture = random.choice(["rock","paper","scissors"])

            if gesture_captured and player_gesture in ("rock","paper","scissors"):
                if cheat_mode:
                    result_text, ai_gesture = determine_winner(player_gesture, ai_gesture)
                    cheat_probability = min(0.7, cheat_probability + 0.05)
                else:
                    if player_gesture == ai_gesture:
                        result_text = "Draw"
                    elif (player_gesture == "rock" and ai_gesture == "scissors") or \
                         (player_gesture == "scissors" and ai_gesture == "paper") or \
                         (player_gesture == "paper" and ai_gesture == "rock"):
                        result_text = "You Win!"
                    else:
                        result_text = "You Lose!"
                if result_text == "You Win!":
                    score += 3; wins += 1
                elif result_text == "You Lose!":
                    score -= 1; losses += 1
                else:
                    draws += 1
            else:
                player_gesture = "None"
                result_text = "No gesture detected!"

            show_total = best_total
            show_shape = best_shape
            show_open  = best_open

            cv2.putText(frame, f"You: {player_gesture}", (10, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 2)
            opponent_name = "对手" if remote_connected else "AI"
            cv2.putText(frame, f"{opponent_name}: {ai_gesture}", (10, 80),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
            cv2.putText(frame, result_text, (10, 130),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0,255,255), 3)
            cv2.putText(frame, f"Best Total:{show_total:.1f}%  Shape:{show_shape:.1f}%  Open:{show_open:.1f}%",
                        (10,170), cv2.FONT_HERSHEY_SIMPLEX,0.6,(255,255,255),2)
            cv2.putText(frame, f"Score:{score} W:{wins} L:{losses} D:{draws}",
                        (10,470), cv2.FONT_HERSHEY_SIMPLEX,0.9,(255,255,255),2)
            cv2.putText(frame, "Press ENTER next  ESC quit",
                        (10,440), cv2.FONT_HERSHEY_SIMPLEX,0.7,(180,180,180),2)

            ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            append_rps_csv(ts, round_count, player_gesture, ai_gesture, result_text,
                           show_total, show_shape, show_open, score, wins, losses, draws)

            cv2.imshow("RPS + Completion", frame)

            if not remote_connected and score >= MAX_SCORE:
                cv2.putText(frame, "CONGRATULATIONS!",
                            (frame.shape[1]//2 - 250, frame.shape[0]//2 - 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 2, (0,255,255), 4)
                cv2.putText(frame, f"You've reached {score} points!",
                            (frame.shape[1]//2 - 250, frame.shape[0]//2),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0,255,255), 3)
                cv2.putText(frame, "Press any key to exit",
                            (frame.shape[1]//2 - 200, frame.shape[0]//2 + 100),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (180,180,255), 2)
                cv2.imshow("RPS + Completion", frame)
                cv2.waitKey(0)
                break

            while True:
                key = cv2.waitKey(0) & 0xFF
                if key == 13:  # Enter
                    break
                elif key == 27:  # ESC
                    cap.release()
                    if remote_connected and sock:
                        sock.close()
                    cv2.destroyAllWindows()
                    return

        cap.release()
        if remote_connected and sock:
            sock.close()
        cv2.destroyAllWindows()
        print("结束。")

if __name__ == "__main__":
    main()
