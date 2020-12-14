# importing the necessary packages
from scipy.spatial import distance as dist
from imutils import face_utils
import numpy as np
import pygame #For playing sound
import imutils
import dlib
import cv2

#Initialize Pygame and load music
pygame.mixer.init()
pygame.mixer.music.load('alarm.wav')

#calculating eye aspect ratio
def eye_aspect_ratio(eye):
	# compute the euclidean distances between the vertical
	A = dist.euclidean(eye[1], eye[5])
	B = dist.euclidean(eye[2], eye[4])

	# compute the euclidean distance between the horizontal
	C = dist.euclidean(eye[0], eye[3])
	# compute the eye aspect ratio
	ear = (A + B) / (2.0 * C)
	return ear

#calculating mouth aspect ratio
def mouth_aspect_ratio(mou):
	# compute the euclidean distances between the horizontal
	D   = dist.euclidean(mou[0], mou[6])
	# compute the euclidean distances between the vertical
	E1  = dist.euclidean(mou[2], mou[10])
	E2  = dist.euclidean(mou[4], mou[8])
	# taking average
	E   = (E1+E2)/2.0
	# compute mouth aspect ratio
	mar = E/D
	return mar
	
#Start webcam video capture
camera = cv2.VideoCapture(0)
camera.set(10,50)
predictor_path = 'shape_predictor_68_face_landmarks.dat'

# define constants for aspect ratios
EYE_AR_THRESH = 0.25
EYE_AR_CONSEC_FRAMES = 8
MOUTH_AR_THRESH = 0.75
MOUTH_AR_CONSEC_FRAMES = 8

#Counts no. of consecutuve frames below threshold value
COUNTER = 0
yawnStatus = False
yawns = 0

#Load face cascade which will be used to draw a rectangle around detected faces.
face_cascade = cv2.CascadeClassifier("haarcascades/haarcascade_frontalface_default.xml")

# initialize dlib's face detector (HOG-based) and then create
# the facial landmark predictor
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(predictor_path)

# grab the indexes of the facial landmarks for the left and right eye
# also for the mouth
(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]
(mStart, mEnd) = face_utils.FACIAL_LANDMARKS_IDXS["mouth"]

# loop over captuing video
while True:
	# grab the frame from the camera, resize
	# it, and convert it to grayscale
	# channels)
	ret, frame = camera.read()
	frame = imutils.resize(frame, width=640)
	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
	prev_yawn_status = yawnStatus
	# detect faces in the grayscale frame
	rects = detector(gray, 0)

	#Detect faces through haarcascade_frontalface_default.xml
	face_rectangle = face_cascade.detectMultiScale(gray, 1.3, 5)

	#Draw rectangle around each face detected
	for (x,y,w,h) in face_rectangle:
            cv2.rectangle(frame,(x,y),(x+w,y+h),(0, 204, 51),2)

	# loop over the face detections
	for rect in rects:
		# determine the facial landmarks for the face region, then
		# convert the facial landmark (x, y)-coordinates to a NumPy
		# array
		shape = predictor(gray, rect)
		shape = face_utils.shape_to_np(shape)

		# extract the left and right eye coordinates, then use the
		# coordinates to compute the eye aspect ratio for both eyes
		leftEye = shape[lStart:lEnd]
		rightEye = shape[rStart:rEnd]
		mouth = shape[mStart:mEnd]
		##Calculate aspect ratio of both eyes and mouth
		leftEAR = eye_aspect_ratio(leftEye)
		rightEAR = eye_aspect_ratio(rightEye)
		mouEAR = mouth_aspect_ratio(mouth)
		# average the eye aspect ratio together for both eyes
		ear = (leftEAR + rightEAR) / 2.0

		# compute the convex hull for the left and right eye, then
		# visualize each of the eyes
		leftEyeHull = cv2.convexHull(leftEye)
		rightEyeHull = cv2.convexHull(rightEye)
		mouthHull = cv2.convexHull(mouth)
		cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 255), 1)
		cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 255), 1)
		cv2.drawContours(frame, [mouthHull], -1, (0, 255, 0), 1)

		# check to see if the eye aspect ratio is below the blink
		# threshold, and if so, increment the blink frame counter
		if ear < EYE_AR_THRESH or mouEAR > MOUTH_AR_THRESH:
			COUNTER += 1
            #If no. of frames is greater than threshold frames,
			# if the eyes were closed for a sufficient number of
			if COUNTER >= EYE_AR_CONSEC_FRAMES or COUNTER >= MOUTH_AR_CONSEC_FRAMES :
				# draw an alarm on the frame			
				pygame.mixer.music.play(-1)
				cv2.putText(frame, "DROWSINESS ALERT!", (10, 50),
					cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 153), 2)

		# otherwise, the eye aspect ratio is not below the blink
		# threshold, so reset the counter and alarm
		else: 
			COUNTER = 0
			pygame.mixer.music.stop()

		cv2.putText(frame, "EAR: {:.2f}".format(ear), (480, 30),
			cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 153), 2)
		cv2.putText(frame, "MAR: {:.2f}".format(mouEAR), (480, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 153), 2)

	# show the frame
	cv2.imshow("Video", frame)
	key = cv2.waitKey(1) & 0xFF

	# if the `q` key was pressed, break from the loop
	if key == ord("q"):
		break

# do a bit of cleanup
camera.release()
cv2.destroyAllWindows()
