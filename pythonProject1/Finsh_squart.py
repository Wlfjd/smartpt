import cv2  # 영상 관련 라이브러리
import mediapipe as mp  # MEDIAPIPE 라는 솔루션 : 누군가가 만든거야
import numpy as np  # 숫자, 계산 관련 라이브러리
import pigpio  # servomotor
import time  # time
import pygame  # sound
from gpiozero import Motor  # motor

# Initial condition setting

pi = pigpio.pi()

mp_drawing = mp.solutions.drawing_utils  #
mp_pose = mp.solutions.pose

pygame.mixer.init()

motorL = Motor(forward=20, backward=21)
motorR = Motor(forward=19, backward=26)

p_s = pygame.mixer.Sound('wjdaus.wav')
p_s1 = pygame.mixer.Sound('djdejddl.wav')
p_s2 = pygame.mixer.Sound('gjfl.wav')
p_s4 = pygame.mixer.Sound('dhfmsWhr.wav')
p_s5 = pygame.mixer.Sound('dnsehd.wav')

p = pygame.mixer.Sound('1.wav')
p1 = pygame.mixer.Sound('2.wav')
p2 = pygame.mixer.Sound('3.wav')
p3 = pygame.mixer.Sound('4.wav')
p4 = pygame.mixer.Sound('5.wav')

c_p = [p, p1, p2, p3, p4]

# video read

cap = cv2.VideoCapture(0)

# Initiallization
v_i = 2500
s = 0
squart_count = 0
state = None
pi.set_servo_pulsewidth(17, 2500)
p_s.play()
n = 1


def calculate_angle(a, b, c):
    a = np.array(a)  # First
    b = np.array(b)  # Mid
    c = np.array(c)  # End

    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    c_angle = np.abs(radians * 180.0 / np.pi)

    if c_angle > 180.0:
        c_angle = 360 - c_angle

    return c_angle


## Setup mediapipe instance

with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
    while cap.isOpened():
        ret, frame = cap.read()
        # Recolor image to RGB

        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False

        # Make detection
        results = pose.process(image)

        # Recolor back to BGR
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        try:

            landmarks = results.pose_landmarks.landmark

            # Get coordinates

            left_foot_index_x = [landmarks[mp_pose.PoseLandmark.LEFT_FOOT_INDEX.value].x]
            left_heel_x = [landmarks[mp_pose.PoseLandmark.LEFT_HEEL.value].x]
            left_knee_x = [landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x]
            left_foot_index_y = [landmarks[mp_pose.PoseLandmark.LEFT_FOOT_INDEX.value].y]
            left_knee_y = [landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y]

            left_shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,
                             landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]

            left_hip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x,
                        landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]

            left_knee = [landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x,
                         landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y]

            left_ankle = [landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].x,
                          landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y]

            if left_foot_index_y[0] < 0.9:
                motorL.forward(0.5)
                motorR.forward(0.5)
            if left_foot_index_y[0] > 1:
                motorL.backward(0.5)
                motorR.backward(0.5)
            if 0.90 < left_foot_index_y[0] < 1:
                pi.set_servo_pulsewidth(17, v_i)
                motorL.stop()
                motorR.stop()
                if s == 0:
                    p_s4.play()
                    print(left_heel_x)
                    print(left_foot_index_x)

                    s = 1
                    continue
                else:
                    pass

                if s == 1:
                    h_x = left_heel_x[0]
                    f_i_x = left_foot_index_x[0]
                    print(h_x - f_i_x)
                    if (h_x - f_i_x) > 0.06:
                        p_s5.play()
                        s = 2
                    print(left_heel_x, left_foot_index_x)
                    pass
            # LEFT, RIGHT Calculate angle
            left_body_angle = calculate_angle(left_shoulder, left_hip, left_knee)
            left_leg_angle = calculate_angle(left_hip, left_knee, left_ankle)

            # cv2.putText(image, str(int(left_angle)),
            #            tuple(np.multiply(left_knee, [960, 240]).astype(int)),
            #            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA
            #            )
            if left_leg_angle > 160:
                state = 'stand'
            if 90 < left_leg_angle < 110 and state == 'stand':
                state = 'squart'
                squart_count += 1

                m = left_knee_x[0] - left_foot_index_x[0]
                if m < -0.02:
                    p_s1.play()
                    squart_count -= 1
                elif (left_body_angle < 95):
                    p_s2.play()
                    squart_count -= 1
                else:
                    c_p[squart_count - 1].play()
        except:
            motorL.backward(0.5)
            motorR.backward(0.5)

        cv2.rectangle(image, (0, 0), (225, 73), (255, 255, 255), -1)

        # Rep data
        cv2.putText(image, 'REPS', (15, 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
        cv2.putText(image, str(squart_count),
                    (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 0), 2, cv2.LINE_AA)

        # Stage data
        cv2.putText(image, 'STATE', (65, 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
        cv2.putText(image, state,
                    (60, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 0), 2, cv2.LINE_AA)

        # Render detections
        mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                  mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=2),
                                  mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2)
                                  )

        cv2.imshow('SMART PT', image)

        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
