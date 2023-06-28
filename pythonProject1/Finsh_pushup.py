import cv2  # 영상 관련 라이브러리
import mediapipe as mp  # MEDIAPIPE 라는 솔루션 : 누군가가 만든거야
import numpy as np  # 숫자, 계산 관련 라이브러리
import pigpio  # servomotor
import time  # time
import pygame  # sound
from gpiozero import Motor  # motor

# Initial condition setting

pi = pigpio.pi()

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

pygame.mixer.init()

motorL = Motor(forward=20, backward=21)
motorR = Motor(forward=19, backward=26)

p_s = pygame.mixer.Sound('wjdaus.wav')
p_s1 = pygame.mixer.Sound('djvemfu.wav')
p_s2 = pygame.mixer.Sound('dnsehd.wav')
p_s3 = pygame.mixer.Sound('djdejddl2_2.wav')

p = pygame.mixer.Sound('1.wav')
p1 = pygame.mixer.Sound('2.wav')
p2 = pygame.mixer.Sound('3.wav')
p3 = pygame.mixer.Sound('4.wav')
p4 = pygame.mixer.Sound('5.wav')

c_p = [p, p1, p2, p3, p4]

cap = cv2.VideoCapture(0)

# Initiallization
v_i = 2500
s = 0
n = 1
pi.set_servo_pulsewidth(17, v_i)
p_s.play()
pushup_count = 0
state = 'UP'


def calculate_angle(a, b, c):
    a = np.array(a)  # First
    b = np.array(b)  # Mid
    c = np.array(c)  # End

    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    left_c_angle = np.abs(radians * 180.0 / np.pi)

    if left_c_angle > 180.0:
        left_c_angle = 360 - left_c_angle

    return left_c_angle


with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
    while cap.isOpened():
        ret, frame = cap.read()

        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False

        results = pose.process(image)

        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        try:

            landmarks = results.pose_landmarks.landmark

            # Get coordinates

            left_heel_x = [landmarks[mp_pose.PoseLandmark.LEFT_HEEL.value].x]
            left_wrist_y = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]
            left_foot_index_x = [landmarks[mp_pose.PoseLandmark.LEFT_FOOT_INDEX.value].x]
            left_foot_index_y = [landmarks[mp_pose.PoseLandmark.LEFT_FOOT_INDEX.value].y]
            left_knee_x = [landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x]
            left_knee_y = [landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y]

            left_shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,
                             landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
            left_elbow = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x,
                          landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
            left_wrist = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x,
                          landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]
            left_hip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x,
                        landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]
            left_knee = [landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x,
                         landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y]
            left_ankle = [landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].x,
                          landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y]
            right_hip = [landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x,
                         landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y]
            right_knee = [landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].x,
                          landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].y]
            right_ankle = [landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].x,
                           landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].y]

            left_elbow_angle = calculate_angle(left_shoulder, left_elbow, left_wrist)
            left_waist_angle = calculate_angle(left_shoulder, left_hip, left_knee)
            left_leg_angle = calculate_angle(left_hip, left_knee, left_ankle)
            right_leg_angle = calculate_angle(right_hip, right_knee, right_ankle)

            print(left_foot_index_y[0])
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
                    p_s1.play()
                    # print(left_heel_x)
                    # print(left_foot_index_x)
                    s = 1
                    continue
                else:
                    pass
                if s == 1:
                    h_x = left_heel_x[0]
                    f_i_x = left_foot_index_x[0]
                    # print(h_x - f_i_x)
                    # print(left_wrist_y)
                    if left_wrist_y[0] > 0.8 and left_leg_angle > 100 and right_leg_angle > 100:
                        p_s2.play()
                        s = 2

            if left_leg_angle > 100 and left_waist_angle > 150 and left_elbow_angle > 160 and state == 'DOWN' and 0.8 < \
                    left_wrist_y[0] < 1:
                state = 'UP'
                pushup_count += 1
                print(pushup_count)
                if n == 1:
                    print("UP")
                    c_p[pushup_count - 1].play()
            elif left_leg_angle > 100 and left_waist_angle > 150 and left_elbow_angle < 100 and state == 'UP' and 0.8 < \
                    left_wrist_y[0] < 1:
                n = 1
                state = 'DOWN'
                print("DOWN")
                print(left_waist_angle)
                if left_waist_angle < 150:
                    p_s3.play()
                    print("djdejddl")
                    pushup_count -= 1
                    n = 0
                else:
                    pass
            else:
                pass
        except:
            motorL.backward(0.5)
            motorR.backward(0.5)

        cv2.rectangle(image, (0, 0), (225, 73), (255, 255, 255), -1)

        cv2.putText(image, 'REPS', (15, 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
        cv2.putText(image, str(pushup_count),
                    (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 0), 2, cv2.LINE_AA)
        cv2.putText(image, 'STATE', (65, 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
        cv2.putText(image, state,
                    (60, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 0), 2, cv2.LINE_AA)

        mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                  mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=2),
                                  mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2)
                                  )
        cv2.imshow('PUSH UP POSE', image)

        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
