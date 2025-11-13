import cv2
import mediapipe as mp
import pyautogui
import numpy as np
import time
import win32gui
import win32con

# Inicializações
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.6,
    min_tracking_confidence=0.6
)

screen_w, screen_h = pyautogui.size()
cap = cv2.VideoCapture(0)

LEFT_EYE = [33, 133, 159, 145]
RIGHT_EYE = [362, 263, 386, 374]
LEFT_EYE_CENTER = [33, 133]
RIGHT_EYE_CENTER = [362, 263]

SMOOTHING = 0.15
CALIBRATION_FRAMES = 30
BLINK_THRESHOLD = 0.18
BLINK_CONSEC_FRAMES = 2
CLICK_COOLDOWN = 1.0

calibrating = True
calib_x, calib_y = 0, 0
frame_count = 0
prev_x, prev_y = 0, 0
tracking_enabled = True
blink_counter_left = 0
blink_counter_right = 0
last_click_time = 0

window_name = "Mouse 2"
cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

# Funções auxiliares

def midpoint(p1, p2):
    return [(p1[0] + p2[0]) / 2, (p1[1] + p2[1]) / 2]

def eye_aspect_ratio(eye_indices, landmarks):
    p1 = np.array([landmarks[eye_indices[0]].x, landmarks[eye_indices[0]].y])
    p2 = np.array([landmarks[eye_indices[1]].x, landmarks[eye_indices[1]].y])
    p3 = np.array([landmarks[eye_indices[2]].x, landmarks[eye_indices[2]].y])
    p4 = np.array([landmarks[eye_indices[3]].x, landmarks[eye_indices[3]].y])
    horizontal = np.linalg.norm(p1 - p2)
    vertical = np.linalg.norm(p3 - p4)
    return vertical / horizontal

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb)
    h, w, _ = frame.shape

    if results.multi_face_landmarks:
        mesh = results.multi_face_landmarks[0].landmark

        left_eye = np.mean([[mesh[i].x, mesh[i].y] for i in LEFT_EYE_CENTER], axis=0)
        right_eye = np.mean([[mesh[i].x, mesh[i].y] for i in RIGHT_EYE_CENTER], axis=0)

        ear_left = eye_aspect_ratio(LEFT_EYE, mesh)
        ear_right = eye_aspect_ratio(RIGHT_EYE, mesh)
        head_center = midpoint(left_eye, right_eye)

        if calibrating:
            calib_x += head_center[0]
            calib_y += head_center[1]
            frame_count += 1
            if frame_count >= CALIBRATION_FRAMES:
                calib_x /= frame_count
                calib_y /= frame_count
                calibrating = False
                print("-> Calibração concluída")
            continue

        delta_x = head_center[0] - calib_x
        delta_y = head_center[1] - calib_y

        screen_x = np.interp(delta_x, [-0.05, 0.05], [0, screen_w])
        screen_y = np.interp(delta_y, [-0.05, 0.05], [0, screen_h])

        smooth_x = prev_x + (screen_x - prev_x) * SMOOTHING
        smooth_y = prev_y + (screen_y - prev_y) * SMOOTHING
        prev_x, prev_y = smooth_x, smooth_y

        if tracking_enabled:
            pyautogui.moveTo(smooth_x, smooth_y)

        current_time = time.time()

        left_closed = ear_left < BLINK_THRESHOLD
        right_closed = ear_right < BLINK_THRESHOLD

        blink_counter_left = blink_counter_left + 1 if left_closed else 0
        blink_counter_right = blink_counter_right + 1 if right_closed else 0

        if blink_counter_left >= BLINK_CONSEC_FRAMES and not right_closed:
            if current_time - last_click_time > CLICK_COOLDOWN:
                pyautogui.click(button="left")
                last_click_time = current_time
            blink_counter_left = 0

        elif blink_counter_right >= BLINK_CONSEC_FRAMES and not left_closed:
            if current_time - last_click_time > CLICK_COOLDOWN:
                pyautogui.click(button="right")
                last_click_time = current_time
            blink_counter_right = 0

        cx, cy = int(head_center[0] * w), int(head_center[1] * h)
        cv2.circle(frame, (cx, cy), 4, (255, 0, 0), -1)
        cv2.putText(frame, f"Tracking: {'ON' if tracking_enabled else 'OFF'} (T) | Recalibrar (R)",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    (0, 255, 0) if tracking_enabled else (0, 0, 255), 2)
        cv2.putText(frame, f"EAR L:{ear_left:.2f} R:{ear_right:.2f}",
                    (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,0), 2)
    elif not calibrating:
        cv2.putText(frame, "Sem rosto detectado", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    cv2.imshow(window_name, frame)

    # Apenas para manter a janela fixa sempre no topo
    hwnd = win32gui.FindWindow(None, window_name)
    if hwnd:
        win_width = int(screen_w * 0.3)
        win_height = int(screen_h * 0.4)
        win_x = 5
        win_y = 5
        win32gui.SetWindowPos(hwnd, win32con.HWND_TOPMOST, win_x, win_y, win_width, win_height, 0)

    key = cv2.waitKey(1) & 0xFF
    if key == 27: # Esc
        break
    elif key == ord('t'): # Tracking
        tracking_enabled = not tracking_enabled
    elif key == ord('r'): # Recalibração
        calibrating = True
        frame_count = 0
        calib_x, calib_y = 0, 0

cap.release()
cv2.destroyAllWindows()
