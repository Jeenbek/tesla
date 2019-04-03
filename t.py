import numpy as np
import cv2
# Get user supplied values
imagePath = 'e.jpg'
cascPath = 'cascade.xml'

faceCascade = cv2.CascadeClassifier(cascPath)

# Read the image
image = cv2.imread(imagePath)
#resize = cv2.resize(image, (640, 640))
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Detect faces in the image
faces = faceCascade.detectMultiScale(
    gray,
    scaleFactor=1.01,
    minNeighbors=20,
    minSize=(250, 250),
    flags=cv2.CASCADE_SCALE_IMAGE
)
faces2 = cv2.groupRectangles(list(faces), 1)
print "Found {0} faces!".format(len(faces))
crop_img = image[faces[0][1]:faces[0][1]+faces[0][3], faces[0][0]:faces[0][0]+faces[0][2]]
for (x, y, w, h) in faces:
    cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)

cv2.imshow("Faces found", image)
cv2.waitKey(0)
