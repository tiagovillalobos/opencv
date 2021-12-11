import cv2
import dlib
import time
from pathlib import Path
from os import path

cap = cv2.VideoCapture(0)

root = Path(path.dirname(path.realpath(__file__)))
predictor_path = str(root / "data/shape_predictor_81_face_landmarks.dat")

detector = dlib.get_frontal_face_detector()
sp = dlib.shape_predictor(predictor_path)

face_detected_count = 0
start = time.time()
elapsed_time = 0

while True:
    ret, frame = cap.read()
    
    # Grayscale
    grayscale = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    
    face_rects = detector(frame)

    if face_rects:
        face_detected_count = face_detected_count + 1

    if face_detected_count == 1 :
        elapsed_time = (time.time() - start)

    cv2.putText(frame, 'Elapsed Time:' + str(elapsed_time), (0, 20), cv2.FONT_HERSHEY_DUPLEX, 0.8, (255, 0, 0))
    
    for index, face in enumerate(face_rects):
        
        left = face.left()
        top = face.top()
        right = face.right()
        bottom = face.bottom()

        # Rosto
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)

        shape = sp(grayscale, face)

        # Olhos
        cv2.rectangle(frame, (shape.part(36).x, shape.part(37).y ), (shape.part(45).x, shape.part(46).y), (0, 255, 0), 2)

        # Olho esquerdo
        cv2.rectangle(frame, (shape.part(36).x, shape.part(37).y ), (shape.part(39).x, shape.part(40).y), (0, 0, 255), 2)

        # Olho direito
        cv2.rectangle(frame, (shape.part(42).x, shape.part(43).y ), (shape.part(45).x, shape.part(46).y), (0, 0, 255), 2)

        # Boca
        cv2.rectangle(frame, (shape.part(48).x, shape.part(49).y ), (shape.part(54).x, shape.part(55).y), (0, 0, 255), 2)

    cv2.imshow('frame', frame)
    if cv2.waitKey(1) == ord('q'):
        break

print(elapsed_time)

cap.release()
cv2.destroyAllWindows()