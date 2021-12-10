import cv2
from pathlib import Path
from os import path

cap = cv2.VideoCapture(0)

root = Path(path.dirname(path.realpath(__file__)))
paths = {
    'faces': str(root / "data/haarcascades/haarcascade_frontalface_default.xml"),
    'eyes': str(root / "data/haarcascades/haarcascade_eye.xml")
}

detectors = {}
for (name, path) in paths.items():
    detectors[name] = cv2.CascadeClassifier(path)

while True:
    ret, frame = cap.read()

    if not ret :
        break

    grayscale = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    results = detectors['faces'].detectMultiScale(
        grayscale,
        minSize=(210, 210)
    )
    
    for result in results:
        cv2.rectangle(frame, result, (0, 255, 0), 4)

        x, y, w, h = result

        face = grayscale[y:y+h, x:x+w]
        
        eyes = detectors['eyes'].detectMultiScale(
            face,
            minSize=(30, 30)
        )

        for eye in eyes:
            roi = eye
            roi[0:2] += result[0:2]
            cv2.rectangle(frame, roi, (255, 0, 0), 4)

    cv2.imshow('frame', frame)
    
    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
