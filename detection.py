from  ultralytics import YOLO
import cv2
model=YOLO('yolo5s.pt')
capture=cv2.VideoCapture(0)
while capture.isopened():
    ret,frame=capture.read()
    if not ret:
        break
    results=model.predict(frame)
    detection=results[0].plot()
    cv2.imshow('object detection',detection)
    if cv2.waitkey(1) & 0xFF ==ord('q'):
        break
    capture.release()
    cv2.destroyAllWindows()


