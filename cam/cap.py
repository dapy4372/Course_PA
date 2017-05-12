#! /usr/bin/python2.7
# import the necessary packages
import sys, os
sys.path.append("/usr/local/lib/python2.7/dist-packages/")
import cv2

THRESHOLD = 0.9
CAP_FRAME_SPACING = 15

faceCascade = cv2.CascadeClassifier('/home/pi/opencv/data/haarcascades/haarcascade_frontalface_default.xml')

fourcc = cv2.VideoWriter_fourcc('m','p','4','v')

video_filename = sys.argv[1]
total_time = int(sys.argv[2]) #ms
date=os.path.basename(video_filename).split('.')[0]

cap = cv2.VideoCapture(video_filename)

tablefile = open("./data/table/%s.txt" % (date), "w")

frame_count= 0
prev_frame_count = -1000
total_frame = cap.get(cv2.CAP_PROP_FRAME_COUNT)
time_per_frame = total_time / total_frame

prev_image = None
while cap.isOpened():
    ret, image = cap.read()
    if (frame_count == total_frame):
        break
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

    sim = 0
    if prev_image != None:
        hist1 = cv2.calcHist([image],[0],None,[256],[0,256])
        hist2 = cv2.calcHist([prev_image],[0],None,[256],[0,256])
        sim = cv2.compareHist(hist1, hist2, 0)

    if prev_image == None or sim < THRESHOLD:

        faces = faceCascade.detectMultiScale(image)

        # Draw a rectangle around the faces
        for (x, y, w, h) in faces:
            if(frame_count - prev_frame_count > CAP_FRAME_SPACING):
                tablefile.write("%s_%d.jpg %d\n" % (date, frame_count, frame_count * time_per_frame))
                cv2.imwrite("./data/image/%s_%d.jpg" % (date, frame_count), image)
                cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)
                prev_frame_count = frame_count

        key = cv2.waitKey(1) & 0xFF
    frame_count = frame_count + 1
    prev_image = image
