#importing dependencies:
import cv2
#importing trained and stored data from xml files::
car_tracker = cv2.CascadeClassifier('cars.xml')       #trained data of cars
pedes_tracker = cv2.CascadeClassifier('body.xml')     #trained data for human long figures
trained_faced_data = cv2.CascadeClassifier('df.xml')  #trained data for human frontal face(side face excluded)
webcam = cv2.VideoCapture('lol_xd.mp4')               #importing video file
try :
    while True:                                           #creating infnite loop
            success_frame_read, frame = webcam.read()         #reading the vide file frame by frame
            if success_frame_read:
                gray_scaled_video = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)   #if the video reading is succeccfull turn the each video frame in grayscale
            else :
                break                                                            #if the video reading is unsuccessfull break the loop
            car_coordinates = car_tracker.detectMultiScale(gray_scaled_video,scaleFactor = 1.2, minNeighbors = 4)                    #finding the co ordinates of cars from gray scaled video
            pedes_coordinates = pedes_tracker.detectMultiScale(gray_scaled_video,scaleFactor = 1.2, minNeighbors = 1)                #finding the co ordinates of people  from gray scaled video
            face_coordinates = trained_faced_data.detectMultiScale(gray_scaled_video,scaleFactor = 1.2, minNeighbors = 4)            #finding the co ordinates of people's faces from gray scaled video
        # scale factor is used to basically blur the vid or image so that when we perdict using our model some details of the vid/photo gets removed and hence the chances of detecting wrong feature as a right one decreases
         # minNeighbours actuallly sees number of layers of neighbours saying its actually the correct one or not and this way accuracy increases more and more
            for (x, y, a, b) in car_coordinates:                                                            #creatingrectangles around the detected cars
                cv2.rectangle(frame,(x,y),(x+a,y+b),(255,0,0),2)
                cv2.putText(frame,'CAR',(x,y+b+4), fontFace=cv2.FONT_HERSHEY_DUPLEX, fontScale=1, color=(25,50,55))                  #adding texts to the detected rectangle 
            for (x, y, a, b) in pedes_coordinates:                                                          #creatingrectangles around the detected people
                cv2.rectangle(frame,(x,y),(x+a,y+b),(0,0,0),2)
                cv2.putText(frame,'pedestrians',(x,y+b+4), fontFace=cv2.FONT_HERSHEY_DUPLEX , fontScale=1, color=(255,125,0))        #adding texts to the detected rectangle
            for (x, y, a, b) in face_coordinates:                                                           #creatingrectangles around the detected people's faces
                cv2.rectangle(frame,(x,y),(x+a,y+b),(255,140,150),2)
                cv2.putText(frame,'face',(x,y+b+4), fontFace=cv2.FONT_HERSHEY_DUPLEX, fontScale=1, color=(10,0,25))                  #adding texts to the detected rectangle
            cv2.imshow('video cam',frame)                    #showing the video after detectiom frame by frame
            cv2.waitKey(1)    
except KeyboardInterrupt:
        print("The program has been terminated") #On pressing ctrl-c in the terminal this msg will be shown
        pass     