import cv2

cam=cv2.VideoCapture(0) #creation of object which links to webcam
model=cv2.CascadeClassifier("haarcascade_frontalface_default.xml") #creation of an object which contains imortant function detectMultiScale

while True:
	ret,frame=cam.read()
	if ret==True:
		gray_frame=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY) #it converts bgr image to grayscale because detectMultiScale always accepts 
		faces=model.detectMultiScale(gray_frame,1.3,5) #1.3 is the scale with which image is to be resized and 5 is minimum number of neighbors required or minimum no. of faces/glows detected at that point to be considered it as a face

		for face in faces:
			x,y,w,h=face
			cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2) #to make a rectangle around the face inside frame

		cv2.imshow("Window",frame) #to print the frame with name window
		key=cv2.waitKey(1) #to hold the screen having frame
		if key==ord("q"):
			break
	else:
		break

cam.release()
cv2.destroyAllWindows()