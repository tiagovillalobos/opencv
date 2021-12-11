import cv2
import time
from pathlib import Path
from os import path

cap = cv2.VideoCapture(0)

root = Path(path.dirname(path.realpath(__file__)))
paths = {
    'face': str(root / "data/haarcascades/haarcascade_frontalface_default.xml"),
    'eyes': str(root / "data/haarcascades/haarcascade_eye.xml"),
    'left_eye': str(root / "data/haarcascades/haarcascade_lefteye_2splits.xml"),
    'right_eye': str(root / "data/haarcascades/haarcascade_righteye_2splits.xml"),
    'smile': str(root / "data/haarcascades/haarcascade_smile.xml")
}

detectors = {}
for (name, path) in paths.items():
    detectors[name] = cv2.CascadeClassifier(path)

def getMultiScale(detector, roi):
    return detector.detectMultiScale(
            roi, scaleFactor=1.1, minNeighbors=10,
            minSize=(15, 15), flags=cv2.CASCADE_SCALE_IMAGE)

face_detected_count = 0
start = time.time()
elapsed_time = 0

while True:
    ret, frame = cap.read()

    if not ret :
        break

    grayscale = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    face = detectors['face'].detectMultiScale(
        grayscale, scaleFactor=1.05, minNeighbors=5, minSize=(30, 30),
        flags=cv2.CASCADE_SCALE_IMAGE)

    if len(face) > 0:
        face_detected_count = face_detected_count + 1

    if face_detected_count == 1 :
        elapsed_time = (time.time() - start)
    
    cv2.putText(frame, 'Elapsed Time:' + str(elapsed_time), (0, 20), cv2.FONT_HERSHEY_DUPLEX, 0.8, (255, 0, 0))

    for (fX, fY, fW, fH) in face:

        roi = grayscale[fY:fY+ fH, fX:fX + fW]

        eyes = getMultiScale(detectors['eyes'], roi)
        leftEye = getMultiScale(detectors['left_eye'], roi)
        rightEye = getMultiScale(detectors['right_eye'], roi)
        smile = getMultiScale(detectors['smile'], roi)

        for (eX, eY, eW, eH) in eyes:
            ptA = (fX + eX, fY + eY)
            ptB = (fX + eX + eW, fY + eY + eH)
            cv2.rectangle(frame, ptA, ptB, (0, 0, 255), 2)

        for (leX, leY, leW, leH) in leftEye:
            ptA = (fX + leX, fY + leY)
            ptB = (fX + leX + leW, fY + leY + leH)
            cv2.rectangle(frame, ptA, ptB, (0, 255, 0), 2)

        for (reX, reY, reW, reH) in rightEye:
            ptA = (fX + reX, fY + reY)
            ptB = (fX + reX + reW, fY + reY + reH)
            cv2.rectangle(frame, ptA, ptB, (0, 255, 0), 2)

        for (sX, sY, sW, sH) in smile:
            ptA = (fX + sX, fY + sY)
            ptB = (fX + sX + sW, fY + sY + sH)
            cv2.rectangle(frame, ptA, ptB, (255, 0, 0), 2)

        cv2.rectangle(frame, (fX, fY), (fX + fW, fY + fH),
            (0, 255, 0), 2)

    cv2.imshow('frame', frame)

    if cv2.waitKey(1) == ord('q'):
        break

print(elapsed_time)
cap.release()
cv2.destroyAllWindows()
