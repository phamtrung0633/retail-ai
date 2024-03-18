import cv2
import time

camera = cv2.VideoCapture(0)
camera.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter.fourcc('m','j','p','g'))
camera.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter.fourcc('M','J','P','G'))
camera.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)


camera2 = cv2.VideoCapture(2)
camera2.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter.fourcc('m','j','p','g'))
camera2.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter.fourcc('M','J','P','G'))
camera2.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
camera2.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

num = 8
start = time.time()
while True:
    success1, img = camera.read()
    success2, img2 = camera2.read()
    k = cv2.waitKey(1)
    if k == 27:
        break
    elif k == ord('q'):
        '''current = time.time()
        if current - start >= 5:'''
        cv2.imwrite('images/stereoLeft/imageL' + str(num) + '.png', img)
        cv2.imwrite('images/stereoRight/imageR' + str(num) + '.png', img2)
        print("Images saved!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        num += 1
        #start = current
    img = cv2.resize(img, (640, 480))
    img2 = cv2.resize(img2, (640, 480))
    cv2.imshow('Img 1', img)
    cv2.imshow('Img 2', img2)

camera.release()
camera2.release()
