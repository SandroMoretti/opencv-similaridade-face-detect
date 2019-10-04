# -*- coding: UTF-8 -*-
import numpy as np
import cv2
from matplotlib import pyplot as plt


frameBase = cv2.imread('imagem/Elizabeth.png')
#frameBase = cv2.imread('imagem/Raymond.png')
cap = cv2.VideoCapture('imagem/video.mp4')

cont = 0
flag = 0

plt.rcParams['figure.figsize'] = (224, 224)

face_cascade = cv2.CascadeClassifier('modelo/haarcascade_frontalface_default.xml')



while(cap.isOpened()):
    flag = flag+1
    ret, frame = cap.read()
    if ret==True:
        img = frame
        if flag <= 820:
            continue
        
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        

        faces = face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30),
        )
        
        for (x,y,w,h) in faces:
            try:
                crop_img = img[y:y+h, x:x+w]
                
                res = cv2.matchTemplate(crop_img, frameBase, cv2.TM_CCOEFF_NORMED) # verifica a similiaridade
                min_val, similaridade, min_loc, max_loc = cv2.minMaxLoc(res)
                
                if similaridade >= 0.60:
                    print("Encontrou")
                    # marcar de verde
                    cv2.rectangle(img, (x, y), (x+w, y+h), (0,255,0), 2)
                else:
                    # marcar de vermelho
                    cv2.rectangle(img, (x, y), (x+w, y+h), (0,0,255), 2)
            except Exception as exp:
                print("ERROR AO ANALISAR")
                continue
        cv2.imshow('Video', img)
        cv2.waitKey(1)
    else:
        break

cv2.destroyAllWindows()