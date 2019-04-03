import cv2

cascPath = 'haar.xml'
cap = cv2.VideoCapture(0)
faceCascade = cv2.CascadeClassifier(cascPath)

while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()

    # Our operations on the frame come here
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Display the resulting frame
    faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor=1.2,
        minNeighbors=10,
        minSize=(150, 150),
        flags = cv2.CASCADE_SCALE_IMAGE
    )
    for (x, y, w, h) in faces:
       cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
    cv2.imshow("Faces found", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cascPath = 'haar.xml'

faceCascade = cv2.CascadeClassifier(cascPath)
