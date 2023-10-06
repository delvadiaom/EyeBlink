import threading
import cv2
import dlib
import time
from imutils import face_utils
from scipy.spatial import distance as dist
from deepface import DeepFace
import pyrebase
import tkinter as tk
from tkinter import ttk
import face_recognition
import numpy as np

config = {
  "apiKey": "AIzaSyDziuQZe29c-QwANamrubsZvXkotZqeMiw",
  "authDomain": "eyeblink-1f72b.firebaseapp.com",
  "projectId": "eyeblink-1f72b",
  "storageBucket": "eyeblink-1f72b.appspot.com",
  "messagingSenderId": "225024441415",
  "appId": "1:225024441415:web:8d920c28751e31e046c562",
  "measurementId": "G-EYZL0RX4P7",
  "serviceAccount": "serviceAccount.json",
  "databaseURL": "https://eyeblink-1f72b-default-rtdb.firebaseio.com/"
}

detector = dlib.get_frontal_face_detector()
firebase = pyrebase.initialize_app(config)
face_region = None
face_region_db = None
frame_jpg_db = None
frame_region_jpg = None

def EAR_cal(eye):
    #----vertical----#
    v1 = dist.euclidean(eye[1],eye[5])
    v2 = dist.euclidean(eye[2],eye[4])

    #-------horizontal----#
    h1 = dist.euclidean(eye[0],eye[3])

    ear = (v1+v2)/h1
    return ear

def capture_frame_and_upload():
    cap = cv2.VideoCapture(0)  # Use the default camera (0) or specify your camera's index
    cv2.waitKey(10)
    ret, frame = cap.read()
    global face_region_db
    global face_region
    global frame_jpg_db
    global frame_region_jpg

    if ret:
        img_gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
        #-----facedetection----#
        faces = detector(img_gray)
        for face in faces:
            x1 = face.left()
            y1 = face.top()
            x2 = face.right()
            y2= face.bottom()
            cv2.rectangle(frame,(x1,y1),(x2,y2),(200),2)

            face_region_db = frame[y1:y2, x1:x2]
            frame_jpg_db = cv2.imencode('frame1.jpg',face_region_db)[1].tobytes()
        cv2.imwrite("frame1.jpg", face_region_db)

        # Upload the captured frame to Firebase Storage
        storage = firebase.storage()
        storage.child("images/frame1.jpg").put(frame_jpg_db)
        storage.child("images/frame1.jpg").download(path="gs://eyeblink-1f72b.appspot.com/images/frame1.jpg",filename="frame1.jpg")
        status_label.config(text="Frame captured and uploaded to Firebase Storage successfully!")
        cap.release()
        # reference_img = cv2.imread("omimage.jpg")

        cam = cv2.VideoCapture(0)
        _,fram = cam.read()
        img_gray = cv2.cvtColor(fram,cv2.COLOR_BGR2GRAY)
        #-----facedetection----#
        faces = detector(img_gray)
        for face in faces:
            x1 = face.left()
            y1 = face.top()
            x2 = face.right()
            y2= face.bottom()
            cv2.rectangle(fram,(x1,y1),(x2,y2),(200),2)

            face_region = fram[y1:y2, x1:x2]
            cv2.imwrite("frame2.jpg", face_region)

        # Load the image files
        image1 = face_recognition.load_image_file("frame1.jpg")
        image2 = face_recognition.load_image_file("frame2.jpg")

        # Extract face encodings
        face_encoding1 = face_recognition.face_encodings(image1)[0]
        face_encoding2 = face_recognition.face_encodings(image2)[0]

        # Calculate the Euclidean distance between the encodings
        distance = np.linalg.norm(face_encoding1 - face_encoding2)

        # Set a similarity threshold (you may need to tune this threshold)
        threshold = 0.5

        cam.release()
        blink_thresh=0.5
        tt_frame = 1
        count=0
        ptime = 0
        blink_count = 0
        lm_model = dlib.shape_predictor('Model/shape_predictor_68_face_landmarks.dat')

        #--Eye ids ---#
        (L_start, L_end) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
        # print(L_start,L_end)
        (R_start, R_end) = face_utils.FACIAL_LANDMARKS_IDXS['right_eye']
        # print(R_start,R_end)
        camera = cv2.VideoCapture(0)
        while blink_count<=20:
            if camera.get(cv2.CAP_PROP_POS_FRAMES) == camera.get(cv2.CAP_PROP_FRAME_COUNT) :
                    camera.set(cv2.CAP_PROP_POS_FRAMES,0)
            _,frameee = camera.read()
            img_gray = cv2.cvtColor(frameee,cv2.COLOR_BGR2GRAY)
            #--------fps --------#
            ctime = time.time()
            fps = 1/(ctime-ptime)
            ptime= ctime
            cv2.putText(
                frameee,
                f'FPS:{int(fps)}',
                (50,50),
                cv2.FONT_HERSHEY_DUPLEX,
                1,
                (0,0,100),
                1
            )

            #-----facedetection----#
            faces = detector(img_gray)

            for face in faces:
                x1 = face.left()
                y1 = face.top()
                x2 = face.right()
                y2= face.bottom()
                cv2.rectangle(frameee,(x1,y1),(x2,y2),(200),2)


                #---------Landmarks------#
                shapes = lm_model(img_gray,face)
                shape = face_utils.shape_to_np(shapes)

                #-----Eye landmarks---#
                lefteye = shape[L_start:L_end]
                righteye = shape[R_start:R_end]

                for Lpt,rpt in zip(lefteye,righteye):
                    cv2.circle(frameee,Lpt,2,(200,200,0),2)
                    cv2.circle(frameee, rpt, 2, (200,200, 0), 2)

                left_EAR = EAR_cal(lefteye)
                right_EAR= EAR_cal(righteye)

                avg =(left_EAR+right_EAR)/2
                if avg<blink_thresh :
                    count+=1

                else :
                    if (count>tt_frame) and (distance<threshold):
                        blink_count+=1
                        cv2.putText(frame,f'BLINK Detected',(frameee.shape[1]//2 - 300,frameee.shape[0]//2),cv2.FONT_HERSHEY_SIMPLEX,2,(0,200,0),2)
                        print('you are allowed to access')
                    elif (distance<threshold) and (count<=tt_frame):
                        cv2.putText(frame,f'Please blink your eye',(frameee.shape[1]//2 - 300,frameee.shape[0]//2),cv2.FONT_HERSHEY_SIMPLEX,2,(0,200,0),2)
                        print('Blink is not detecting')
                    else :
                        count=0
                        print('You are not allowed to access')
                        camera.release()

            cv2.imshow("Video" ,frameee)
            if cv2.waitKey(3) & 0xFF == ord('q'):
                break

        camera.release()
    else:
        status_label.config(text="Failed to capture frame!")


app = tk.Tk()
app.title("Frame Capture and Upload App")

capture_button = ttk.Button(app, text="Capture Frame and Upload", command=capture_frame_and_upload)
capture_button.pack(pady=20)

# new_window_button = ttk.Button(app, text="Next", command=check_face_from_database())
# new_window_button.pack()

status_label = ttk.Label(app, text="")
status_label.pack()

app.mainloop()

