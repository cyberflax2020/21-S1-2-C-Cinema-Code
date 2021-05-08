import pickle
import numpy as np
import cv2
import mediapipe as mp
import pandas as pd
from collections import Counter


def action_detect(if_show_video=False):
    mp_drawing = mp.solutions.drawing_utils  # Drawing helpers
    mp_holistic = mp.solutions.holistic  # Mediapipe Solutions

    body_list = []

    with open('body_exp.pkl', 'rb') as f:
        model = pickle.load(f)

    str_source = "output_facial.mp4"
    cap = cv2.VideoCapture(str_source)

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter('output_body.mp4', fourcc, float(cap.get(cv2.CAP_PROP_FPS)),
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

            # Right hand
            mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
                                      mp_drawing.DrawingSpec(color=(80, 22, 10), thickness=2, circle_radius=4),
                                      mp_drawing.DrawingSpec(color=(80, 44, 121), thickness=2, circle_radius=2)
                                      )

            # Left Hand
            mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
                                      mp_drawing.DrawingSpec(color=(121, 22, 76), thickness=2, circle_radius=4),
                                      mp_drawing.DrawingSpec(color=(121, 44, 250), thickness=2, circle_radius=2)
                                      )

            # Pose Detections
            mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS,
                                      mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=4),
                                      mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2)
                                      )
            # Export coordinates
            try:
                # Extract Pose landmarks
                pose = results.pose_landmarks.landmark
                pose_row = list(
                    np.array(
                        [[landmark.x, landmark.y, landmark.z, landmark.visibility] for landmark in pose]).flatten())

                row = pose_row

                # Make Detections
                X = pd.DataFrame([row])
                body_language_class = model.predict(X)[0]
                body_language_prob = model.predict_proba(X)[0]

                body_list.append(str(body_language_class))

                print(body_language_class, body_language_prob)

                if round(body_language_prob[np.argmax(body_language_prob)], 2) > 0.5:
                    # Grab ear coords
                    coords = tuple(np.multiply(
                        np.array(
                            (results.pose_landmarks.landmark[mp_holistic.PoseLandmark.LEFT_EAR].x,
                             results.pose_landmarks.landmark[mp_holistic.PoseLandmark.LEFT_EAR].y))
                        , [int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))]).astype(
                        int))

                    cv2.rectangle(image,
                                  (coords[0], coords[1] + 5),
                                  (coords[0] + len(body_language_class) * 20, coords[1] - 30),
                                  (245, 117, 16), -1)
                    cv2.putText(image, body_language_class, coords,
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

                    # Get status box
                    cv2.rectangle(image, (0, 0), (250, 60), (245, 117, 16), -1)

                    # Display Class
                    cv2.putText(image, 'CLASS'
                                , (95, 12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
                    cv2.putText(image, body_language_class.split(' ')[0]
                                , (90, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

                    # Display Probability
                    cv2.putText(image, 'PROB'
                                , (15, 12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)

                    cv2.putText(image, str(round(body_language_prob[np.argmax(body_language_prob)], 2))
                                , (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

            except:
                pass

            out.write(image)

            if if_show_video:
                cv2.imshow('Video', image)

            if cv2.waitKey(10) & 0xFF == ord('q'):
                break

    if len(body_list) == 0:
        f = open("all_labels.txt", "r")
        list_of_lines = f.readlines()
        if len(list_of_lines) == 4:
            list_of_lines.append("activity: " + "No body detected" + "\n")
        else:
            list_of_lines[4] = "activity: " + "No body detected" + "\n"
    else:
        counter = Counter(body_list)
        maxKey = max(counter, key=counter.get)
        print(maxKey)
        f = open("all_labels.txt", "r")
        list_of_lines = f.readlines()
        if len(list_of_lines) == 4:
            list_of_lines.append("activity:  " + maxKey + "\n")
        else:
            list_of_lines[4] = "activity:  " + maxKey + "\n"

    f = open("all_labels.txt", "w")
    f.writelines(list_of_lines)
    f.close()
    out.release()
    cap.release()
    cv2.destroyAllWindows()
    print("Action detection finished")
    print("All process finished")