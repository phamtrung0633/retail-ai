import cv2

camera = cv2.VideoCapture(0)
camera.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter.fourcc('m','j','p','g'))
camera.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter.fourcc('M','J','P','G'))
camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)


camera2 = cv2.VideoCapture(2)
camera2.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter.fourcc('m','j','p','g'))
camera2.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter.fourcc('M','J','P','G'))
camera2.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
camera2.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
num = 0
while camera.isOpened():
    success1, img = camera.read()
    success2, img2 = camera2.read()
    k = cv2.waitKey(5)
    cv2.imshow('Img 1', img)
    cv2.imshow('Img 2', img2)
    if k == 27:
        break
    elif k == ord('q'):
        cv2.imwrite('images/environmentLeft/1.png', img)
        cv2.imwrite('images/environmentRight/2.png', img2)
        break


camera.release()
camera2.release()