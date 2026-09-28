import cv2
import time

for i in range(3):
    print(f'Testing camera {i}...')
    cap = cv2.VideoCapture(i)
    if not cap.isOpened():
        print(f'Camera {i} could not be opened.')
    else:
        time.sleep(0.5)
        ret, frame = cap.read()
        if ret:
            print(f'Camera {i} successfully read a frame of shape {frame.shape}')
        else:
            print(f'Camera {i} opened, but failed to read a frame.')
        cap.release()
