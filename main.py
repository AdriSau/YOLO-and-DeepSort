import os
import random
from collections import deque
import cv2
from ultralytics import YOLO

from tracker import Tracker

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

video_path = "testingFootage/Clip_TopView_720p.mp4"
video_out_path = "testingFootage/out.mp4"

cap = cv2.VideoCapture(video_path)
ret, frame = cap.read()

#cap_out = cv2.VideoWriter(video_out_path, cv2.VideoWriter_fourcc('m', 'p', '4', 'v'), cap.get(cv2.CAP_PROP_FPS),(frame.shape[1], frame.shape[0]))

model = YOLO("best.pt")

tracker = Tracker()

colors = [(random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)) for j in range(10)]
data_deque = {}
detection_threshold = 0.5
while ret:
    results = model(frame)
    aux = 0
    for result in results:
        detections = []
        for r in result.boxes.data.tolist():
            x_top_L, y_top_L, x_bot_R, y_bot_R, conf_value, class_id = r
            x_top_L = int(x_top_L)
            x_bot_R = int(x_bot_R)
            y_top_L = int(y_top_L)
            y_bot_R = int(y_bot_R)
            class_id = int(class_id)
            if conf_value > detection_threshold:
                detections.append([x_top_L, y_top_L, x_bot_R, y_bot_R, conf_value])
        tracker.update(frame, detections)
        frame_height, frame_width, nChannnel = frame.shape
        cv2.rectangle(frame, (int(frame_width - 400), int(0)), (int(frame_width), int(28)), (0, 0, 0), thickness=-1)
        cv2.putText(frame, "Last detection: ", (int(frame_width - 395), int(22)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), thickness=2)
        for track in tracker.tracks:
            aux+=1
            bbox = track.bbox
            x_top_L, y_top_L, x_bot_R, y_bot_R = bbox
            track_id = track.track_id
            print(track)
            print(tracker.tracks)
            object_center_X = (int(x_top_L)+int(x_bot_R))/2
            object_center_Y = (int(y_top_L) + int(y_bot_R)) / 2
            print('car detected: Id: '+str(track_id)+' Detected in cordinates: ('+str(object_center_X)+','+str(object_center_Y)+')')
            cv2.rectangle(frame, (int(x_top_L), int(y_top_L)), (int(x_bot_R), int(y_bot_R)), (colors[track_id % len(colors)]), 3)
            cv2.circle(frame,( int(object_center_X),int(object_center_Y)),2,(colors[track_id % len(colors)]),thickness = -1)
            cv2.rectangle(frame,  (int(x_top_L), int(y_top_L-15)), (int(x_top_L+120), int(y_top_L)), (colors[track_id % len(colors)]), thickness=-1)
            if (aux > 0 )&(aux == len(tracker.tracks)):
                cv2.putText(frame, "Last detection: Id:" + str(track_id) + " posX: " + str(object_center_X) + " posY: "+str(object_center_Y), (int(frame_width - 395), int(22)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), thickness=2)
            rounded_score = round(conf_value, 3)
            #creo lista
            if track_id not in data_deque:
                data_deque[track_id] = deque(maxlen=15)
            #añado el centro
            if len(data_deque[track_id]) >= 15:
                data_deque[track_id].pop()
                data_deque[track_id].appendleft((object_center_X,object_center_Y))
            else:
                data_deque[track_id].append((object_center_X,object_center_Y))
            for i in range(1, len(data_deque[track_id])):
                punto1_x =  data_deque[track_id][i - 1][0]
                punto1_y =  data_deque[track_id][i - 1][1]
                punto2_x = data_deque[track_id][i][0]
                punto2_y = data_deque[track_id][i][1]
                cv2.line(frame,( int(punto1_x),int(punto1_y)), ( int(punto2_x),int(punto2_y)), (colors[track_id % len(colors)]), 2)
            cv2.putText(frame, "id:" + str(track_id) + " conf:" + str(rounded_score), (int(x_top_L), int(y_top_L)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), thickness=2)
    aux = 0

        #fuera del for un bucle para elminar los ids perdidos
    cv2.imshow('display_video',frame)
    cv2.waitKey(30)
    #cap_out.write(frame)
    ret, frame = cap.read()

cap.release()
#cap_out.release()
cv2.destroyAllWindows()
