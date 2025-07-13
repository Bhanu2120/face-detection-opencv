import cv2
import mediapipe as mp


img = cv2.VideoCapture(0)
#detect faces
mp_face_detection = mp.solutions.face_detection
face_detection = mp_face_detection.FaceDetection(model_selection=0,min_detection_confidence=0.5) #model_detection is used to detect small distance of face

while True:
    ret,frame = img.read()
    H,W,_ = frame.shape
    if not ret:
        break
    img_rgb = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
    out = face_detection.process(img_rgb) #detects our measurements of face
    if out.detections is not None:
        for detection in out.detections:
            location_data = detection.location_data
            bbox = location_data.relative_bounding_box

            x1,y1,w,h = bbox.xmin,bbox.ymin,bbox.width,bbox.height
            x1 = int(x1*W)
            y1 = int(y1*H)
            w = int(w*W)
            h = int(h*H)

            frame = cv2.rectangle(frame,(x1,y1),(x1+w,y1+h),(0,255,0),5)

    cv2.imshow('Webcam',frame)
    if cv2.waitKey(1) & 0xFF == ord('b'):
        break
img.release()
cv2.destroyAllWindows()