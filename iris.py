import cv2
import mediapipe as mp
import pyautogui
import numpy as np

# Configurações Iniciais
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)
pyautogui.FAILSAFE = False

cap = cv2.VideoCapture(0)
screen_w, screen_h = pyautogui.size()

LEFT_EYE = [33, 133, 159, 145]
RIGHT_EYE = [362, 263, 386, 374]

window_name = "Head Tracker"
cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

# Funções auxiliares
def eye_aspect_ratio(landmarks, eye_points, frame_w, frame_h):
    points = [(int(landmarks[p].x * frame_w), int(landmarks[p].y * frame_h)) for p in eye_points]
    vertical = np.linalg.norm(np.array(points[1]) - np.array(points[2]))
    horizontal = np.linalg.norm(np.array(points[0]) - np.array(points[3]))
    return vertical / horizontal if horizontal != 0 else 0

def get_head_position(landmarks, frame_w, frame_h):
    nose = landmarks[1]
    return int(nose.x * frame_w), int(nose.y * frame_h)

# Controle de piscadas e movimento
blink_counter_left = 0
blink_counter_right = 0
BLINK_THRESHOLD = 0.18
BLINK_FRAMES = 3

center_x, center_y = None, None
sensitivity = 1.5
control_enabled = True

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb)
    frame_h, frame_w, _ = frame.shape

    if results.multi_face_landmarks:
        landmarks = results.multi_face_landmarks[0].landmark
        nose_x, nose_y = get_head_position(landmarks, frame_w, frame_h)

        if center_x is None:
            center_x, center_y = nose_x, nose_y

        # Movimento da cabeça → movimento do mouse
        if control_enabled:
            dx = (nose_x - center_x) * sensitivity
            dy = (nose_y - center_y) * sensitivity
            pyautogui.moveRel(dx, dy, duration=0.05)
            center_x, center_y = nose_x, nose_y

        # Detecção de piscadas
        left_ear = eye_aspect_ratio(landmarks, LEFT_EYE, frame_w, frame_h)
        right_ear = eye_aspect_ratio(landmarks, RIGHT_EYE, frame_w, frame_h)

        # Olho esquerdo → clique esquerdo
        if left_ear < BLINK_THRESHOLD:
            blink_counter_left += 1
        else:
            if blink_counter_left >= BLINK_FRAMES:
                pyautogui.click(button='left')
            blink_counter_left = 0

        # Olho direito → clique direito
        if right_ear < BLINK_THRESHOLD:
            blink_counter_right += 1
        else:
            if blink_counter_right >= BLINK_FRAMES:
                pyautogui.click(button='right')
            blink_counter_right = 0

        # Indicador visual
        cv2.circle(frame, (nose_x, nose_y), 5, (0, 255, 0), -1)

    # Exibe instruções
    status = "ON" if control_enabled else "OFF"
    cv2.putText(frame, f"[R] Calibrar | [T] Controle: {status}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

    cv2.imshow(window_name, frame)

    # Mantém janela sempre no topo
    hwnd = win32gui.FindWindow(None, window_name)
    if hwnd:
        win32gui.SetWindowPos(hwnd, win32con.HWND_TOPMOST, 0, 0, 640, 480, 0)

    key = cv2.waitKey(1) & 0xFF

    # R = recalibrar centro
    if key == ord('r'):
        center_x, center_y = None, None

    # T = ativar/desativar movimento
    elif key == ord('t'):
        control_enabled = not control_enabled

    # ESC = sair
    elif key == 27:
        break

cap.release()
cv2.destroyAllWindows()
