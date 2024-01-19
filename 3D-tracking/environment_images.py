import cv2

cap = cv2.VideoCapture(0)
cap2 = cv2.VideoCapture(2)
num = 0
while cap.isOpened():
    success1, img = cap.read()
    success2, img2 = cap2.read()
    k = cv2.waitKey(5)
    cv2.imshow('Img 1', img)
    cv2.imshow('Img 2', img2)
    if k == 27:
        break
    elif k == ord('q'):
        cv2.imwrite('images/stereoLeft/1.png', img)
        cv2.imwrite('images/stereoRight/2.png', img2)
        break


cap.release()
cap2.release()