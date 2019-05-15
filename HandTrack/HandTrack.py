import dlib
import cv2
import time

detector = dlib.simple_object_detector('hand.svm')

cv2.namedWindow('frame', cv2.WINDOW_NORMAL)

cap = cv2.VideoCapture(0)
rscale = 2.0

while (True):
    start_time = time.time()
    ret, frame = cap.read()
    width, height, _ = frame.shape

    ft = cv2.resize(frame, (int(frame.shape[1] / rscale), int(frame.shape[0] / rscale)))
    dets = detector(ft)

    for d in dets:
        cv2.rectangle(frame, (int(d.left()*rscale),
                              int(d.top()*rscale)),
                              (int(d.right()*rscale),
                              int(d.bottom()*rscale)),
                              (255, 0, 0), 5)

    cv2.imshow('frame', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()