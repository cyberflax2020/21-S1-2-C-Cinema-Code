import pickle
from collections import Counter
import numpy as np
import cv2
import mediapipe as mp
import pandas as pd


def facial_exp_detect(if_show_video=False):
    mp_drawing = mp.solutions.drawing_utils  # Drawing helpers
    mp_holistic = mp.solutions.holistic  # Mediapipe Solutions
    emotion_list = []
    with open('facial_exp.pkl', 'rb') as f:
        model = pickle.load(f)

    str_source = "age+gender.mp4"
    cap = cv2.VideoCapture(str_source)

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter('output_facial.mp4', fourcc, float(cap.get(cv2.CAP_PROP_FPS)),
                          (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))))

    # Initiate holistic model
    with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
        while cap.isOpened():
            ret, frame = cap.read()

            # Recolor Feed
            if frame is None:
                break
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image.flags.writeable = False

            # Make Detections
            results = holistic.process(image)

            # Recolor image back to BGR for rendering
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            # Draw face landmarks
            mp_drawing.draw_landmarks(image, results.face_landmarks, mp_holistic.FACE_CONNECTIONS,
                                      mp_drawing.DrawingSpec(color=(80, 110, 10), thickness=1, circle_radius=1),
                                      mp_drawing.DrawingSpec(color=(80, 256, 121), thickness=1, circle_radius=1)
                                      )

            # Export coordinates
            try:

                # Extract Face landmarks
                face = results.face_landmarks.landmark
                face_row = list(
                    np.array(
                        [[landmark.x, landmark.y, landmark.z, landmark.visibility] for landmark in face]).flatten())

                row = face_row

                # Make Detections
                X = pd.DataFrame([row])
                body_language_class = model.predict(X)[0]
                body_language_prob = model.predict_proba(X)[0]

                emotion_list.append(str(body_language_class))

                print(body_language_class, body_language_prob)

                if round(body_language_prob[np.argmax(body_language_prob)], 2) > 0.5:
                    # Get status box
                    cv2.rectangle(image, (0, 100), (250, 160), (245, 117, 16), -1)

                    # Display Class
                    cv2.putText(image, 'CLASS'
                                , (95, 112), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
                    cv2.putText(image, body_language_class.split(' ')[0]
                                , (90, 140), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

                    # Display Probability
                    cv2.putText(image, 'PROB'
                                , (15, 112), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
                    cv2.putText(image, str(round(body_language_prob[np.argmax(body_language_prob)], 2))
                                , (10, 140), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

            except:
                pass

            out.write(image)
            if if_show_video:
                cv2.imshow('Video', image)

            if cv2.waitKey(10) & 0xFF == ord('q'):
                break

    if len(emotion_list) == 0:
        f = open("all_labels.txt", "r")
        list_of_lines = f.readlines()
        if len(list_of_lines) == 3:
            list_of_lines.append("emotion: " + "No face detected" + "\n")
        else:
            list_of_lines[3] = "emotion: " + "No face detected" + "\n"
    else:
        counter = Counter(emotion_list)
        maxKey = max(counter, key=counter.get)
        print(maxKey)
        f = open("all_labels.txt", "r")
        list_of_lines = f.readlines()
        if len(list_of_lines) == 3:
            list_of_lines.append("emotion: " + maxKey + "\n")
        else:
            list_of_lines[3] = "emotion: " + maxKey + "\n"
    f = open("all_labels.txt", "w")
    f.writelines(list_of_lines)
    f.close()
    out.release()
    cap.release()
    cv2.destroyAllWindows()
    print("Facial expression detection finished")
