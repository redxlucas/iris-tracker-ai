import numpy as np
import cv2

cap = cv2.VideoCapture(0)
face_cascade = cv2.CascadeClassifier('xml/haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('xml/haarcascade_eye.xml')

while True:
    ret, frame = cap.read()

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) # OpenCV precisa do tom cinza para o classificador de HaarCascade

    # ScaleFactor (Param 2) = Parâmetro que especifica o quanto o tamanho da imagem têm de ser reduzido para cada escala
    # MinNeighbors (Param 3) = Especifica quantos vizinhos cada retângulo candidato deve reter. Maior o valor reduz a detecção. Manter entre 3-6
    faces = face_cascade.detectMultiScale(gray, 1.3, 5) 

    for (x, y, w, h) in faces: 
        area = w * h
        frame_area = frame.shape[0] * frame.shape[1]
        confidence = min(100, (area / frame_area) * 20000)

        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
        cv2.putText(frame, f"Face: {confidence:.1f}%", (x, y - 10),
        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
        roi_gray = gray[y:y+w, x:x+w]
        roi_color = frame[y:y+h, x:x+w]

        eyes = eye_cascade.detectMultiScale(roi_gray, 1.3, 2)

        # for (ex, ey, ew, eh) in eyes:
        #     cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey+ eh), (0, 255, 255), 2)

    cv2.imshow('frame', frame)

    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()