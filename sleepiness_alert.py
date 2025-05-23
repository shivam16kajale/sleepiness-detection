import cv2
import mediapipe as mp
import numpy as np
import winsound

fm = mp.solutions.face_mesh

fm_model = fm.FaceMesh(
static_image_mode = False,
max_num_faces=1,
refine_landmarks=True,
min_detection_confidence = 0.5,
min_tracking_confidence = 0.5
)

#p1 - p6
right_index=[362, 385, 387, 263, 373, 380]
#p1 - p6
left_index=[33, 160, 158, 133, 153, 144]

lips_index=[61,81,13,311,291,402,14,178]

def get_ear(eye_index,landmark,img_w,img_h):
    p = [(int(landmarks[i].x * img_w),int(landmark[i].y*img_h)) for i in eye_index]
    #p=[0,1,2,3,4,5]
    p1 = np.linalg.norm(np.array(p[1])-np.array(p[5]))
    p2 = np.linalg.norm(np.array(p[2])-np.array(p[4]))
    p3 = np.linalg.norm(np.array(p[0])-np.array(p[3]))
    
    ear = (p1+p2)/(2.0*p3)
    return ear

def get_mar(lips_index,landmark,img_w,img_h):
    p = [(int(landmarks[i].x * img_w),int(landmark[i].y*img_h)) for i in lips_index]
    #p=[0,1,2,3,4,5,6,7]
    p1 = np.linalg.norm(np.array(p[1])-np.array(p[7]))
    p2 = np.linalg.norm(np.array(p[2])-np.array(p[6]))
    p3 = np.linalg.norm(np.array(p[3])-np.array(p[5]))
    p4 = np.linalg.norm(np.array(p[0])-np.array(p[4]))
    
    mar = (p1+p2+p3)/(2.0*p4)
    return mar

sleep_ear = 0.15
eye_lid_count = 20 #assume value

#eye lid
count=0

#yawn value
yawn_val = 0.89
yawn = False
# yawm count
yawn_count =0

sound_playing = False 

vid = cv2.VideoCapture(0)


while True:
    s,f = vid.read()

    if s==False:
        break

    rgb_img = cv2.cvtColor(f,cv2.COLOR_BGR2RGB)

    result = fm_model.process(rgb_img)
    h,w,_ = rgb_img.shape
    if result.multi_face_landmarks:
        landmarks = result.multi_face_landmarks[0].landmark
        left_ear = get_ear(left_index,landmarks,w,h)
        right_ear = get_ear(right_index,landmarks,w,h)

        mar = get_mar(lips_index,landmarks,w,h)

        avg_ear=(left_ear+right_ear)/2.0

        #display EAR
        cv2.putText(f, f"EAR: {avg_ear:.2f}", (30, 30),
        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        
        #display EAR
        cv2.putText(f, f"MAR: {mar:.2f}", (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        if avg_ear<sleep_ear:
            count+=1
            if count>=eye_lid_count:
                    cv2.putText(f,"SLEEP DETECTED!",(30,100),cv2.FONT_HERSHEY_COMPLEX,0.9,(0,0,255),2)
                    if not sound_playing:
                        print("DROWSINESS ALERT!")
                        winsound.PlaySound("alert.wav", winsound.SND_FILENAME | winsound.SND_ASYNC)
                        sound_playing = True
        else:
             count=0
             winsound.PlaySound(None, winsound.SND_PURGE)
             sound_playing=False
        
        #counting yawm - how much times
        if mar >= yawn_val:
             if not yawn:
                  yawn_count+=1
                  yawn = True
        else:
             yawn=False
        cv2.putText(f,f"Yawn count: {yawn_count}",(30,150),cv2.FONT_HERSHEY_COMPLEX,0.9,(0,0,255),2)
             

    cv2.imshow("Monitor",f)
    if cv2.waitKey(1) & 255 ==ord("q"):
        break

vid.release()
cv2.destroyAllWindows()