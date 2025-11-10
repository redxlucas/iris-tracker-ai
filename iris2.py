import cv2
import mediapipe as mp
import pyautogui
import numpy as np
import math
import time

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(max_num_faces=1, refine_landmarks=True)
screen_w, screen_h = pyautogui.size()
cap = cv2.VideoCapture(0)

LEFT_EYE = [33, 133, 159, 145]
RIGHT_EYE = [362, 263, 386, 374]
LEFT_IRIS = [474, 475, 476, 477]
RIGHT_IRIS = [469, 470, 471, 472]

SMOOTHING = 0.2
prev_x, prev_y = 0, 0
tracking_enabled = True
last_click_time = 0
CLICK_COOLDOWN = 1.0

def eye_aspect_ratio(eye_indices, landmarks):
    p1 = np.array([landmarks[eye_indices[0]].x, landmarks[eye_indices[0]].y])
    p2 = np.array([landmarks[eye_indices[1]].x, landmarks[eye_indices[1]].y])
    p3 = np.array([landmarks[eye_indices[2]].x, landmarks[eye_indices[2]].y])
    p4 = np.array([landmarks[eye_indices[3]].x, landmarks[eye_indices[3]].y])

    horizontal = np.linalg.norm(p1 - p2)
    vertical = np.linalg.norm(p3 - p4)
    return vertical / horizontal

BLINK_THRESHOLD = 0.18
BLINK_CONSEC_FRAMES = 2
blink_counter_left = 0
blink_counter_right = 0

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

        ear_left = eye_aspect_ratio(LEFT_EYE, mesh)
        ear_right = eye_aspect_ratio(RIGHT_EYE, mesh)

        def get_eye_ratio(eye_points, iris_points):
            x_eye = [mesh[p].x for p in eye_points]
            y_eye = [mesh[p].y for p in eye_points]
            x_iris = [mesh[p].x for p in iris_points]
            y_iris = [mesh[p].y for p in iris_points]

            eye_left, eye_right = np.min(x_eye), np.max(x_eye)
            iris_x = np.mean(x_iris)
            iris_y = np.mean(y_iris)

            ratio_x = (iris_x - eye_left) / (eye_right - eye_left)
            return ratio_x, np.mean(y_iris)

        left_ratio_x, left_ratio_y = get_eye_ratio(LEFT_EYE[:2], LEFT_IRIS)
        right_ratio_x, right_ratio_y = get_eye_ratio(RIGHT_EYE[:2], RIGHT_IRIS)

        avg_x = (left_ratio_x + right_ratio_x) / 2
        avg_y = (left_ratio_y + right_ratio_y) / 2

        screen_x = np.interp(avg_x, [0.35, 0.65], [0, screen_w])
        screen_y = np.interp(avg_y, [0.42, 0.52], [0, screen_h])

        smooth_x = prev_x + (screen_x - prev_x) * SMOOTHING
        smooth_y = prev_y + (screen_y - prev_y) * SMOOTHING
        prev_x, prev_y = smooth_x, smooth_y

        if tracking_enabled:
            pyautogui.moveTo(smooth_x, smooth_y)

        current_time = time.time()

        if ear_left < BLINK_THRESHOLD:
            blink_counter_left += 1
        else:
            if blink_counter_left >= BLINK_CONSEC_FRAMES and current_time - last_click_time > CLICK_COOLDOWN:
                pyautogui.click(button="left")
                last_click_time = current_time
            blink_counter_left = 0

        if ear_right < BLINK_THRESHOLD:
            blink_counter_right += 1
        else:
            if blink_counter_right >= BLINK_CONSEC_FRAMES and current_time - last_click_time > CLICK_COOLDOWN:
                pyautogui.click(button="right")
                last_click_time = current_time
            blink_counter_right = 0

        cv2.putText(frame, f"Tracking: {'ON' if tracking_enabled else 'OFF'} (press 'T')",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                    (0, 255, 0) if tracking_enabled else (0, 0, 255), 2)

        cv2.putText(frame, f"EAR L:{ear_left:.2f} R:{ear_right:.2f}",
                    (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,0), 2)

    cv2.imshow("Eye Tracker", frame)
    key = cv2.waitKey(1) & 0xFF
    if key == 27:
        break
    elif key == ord('t'):
        tracking_enabled = not tracking_enabled

cap.release()
cv2.destroyAllWindows()
