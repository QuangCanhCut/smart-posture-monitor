import cv2
import time

print('Testing camera 0 with DSHOW...')
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
if not cap.isOpened():
    print('Camera 0 could not be opened with DSHOW.')
else:
    time.sleep(1.0)
    ret, frame = cap.read()
    if ret:
        print(f'Camera 0 successfully read a frame of shape {frame.shape} with DSHOW.')
    else:
        print('Camera 0 opened, but failed to read a frame with DSHOW.')
    cap.release()
