from typing import Set
from flask import Flask, request, session, render_template, redirect, url_for, flash
from flask_sqlalchemy import SQLAlchemy
import sqlite3
from models import db
import os
from models import Fcuser, Setting_pushup, Setting_squat, Pushup, Squat, Pushup_clear, Time_p, Squat_clear, Time_s
from flask import Flask, render_template, Response
import cv2
import numpy as np
import mediapipe as mp
import sys
from datetime import datetime
# from gpiozero import Motor
# import pigpio
# import pygame
cpp = Flask(__name__)
# Load a sample picture and learn how to recognize it.

process_this_frame = True

#pi = pigpio.pi()


def gen_frames_s():
    conn = sqlite3.connect('db.sqlite')
    cursor = conn.cursor()
    sql = 'SELECT s_set_u, s_count_u, s_time_u, lo FROM Setting_squat ORDER BY ROWID DESC LIMIT 1'
    cursor.execute(sql)
    rows = cursor.fetchall()
    for es in rows:
        s_set_u = es[0]
        s_count_u = es[1]
        s_time_u = es[2]
        lo = es[3]

    mp_drawing = mp.solutions.drawing_utils  #
    mp_pose = mp.solutions.pose

    # pygame.mixer.init()
    #
    # motorL = Motor(forward=20, backward=21)
    # motorR = Motor(forward=19, backward=26)
    #
    # p_s = pygame.mixer.Sound('wjdaus.wav')
    # p_s1 = pygame.mixer.Sound('djdejddl.wav')
    # p_s2 = pygame.mixer.Sound('gjfl.wav')
    # p_s4 = pygame.mixer.Sound('dhfmsWhr.wav')
    # p_s5 = pygame.mixer.Sound('dnsehd.wav')
    #
    # p = pygame.mixer.Sound('1.wav')
    # p1 = pygame.mixer.Sound('2.wav')
    # p2 = pygame.mixer.Sound('3.wav')
    # p3 = pygame.mixer.Sound('4.wav')
    # p4 = pygame.mixer.Sound('5.wav')
    
    # c_p = [p, p1, p2, p3, p4]

    # video read

    camera = cv2.VideoCapture(0)

    # Initiallization
    v_i = 2250
    s = 0
    s_set_o = 0
    s_count_o = 0
    state = None
    # pi.set_servo_pulsewidth(17, 2500)
    # p_s.play()
    n = 1
    def left_calculate_angle(a, b, c):
        a = np.array(a)  # First
        b = np.array(b)  # Mid
        c = np.array(c)  # End

        radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
        left_c_angle = np.abs(radians * 180.0 / np.pi)

        if left_c_angle > 180.0:
            left_c_angle = 360 - left_c_angle

        return left_c_angle

    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        while True:
            success, frame = camera.read()  # read the camera frame
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame.flags.writeable = False

            # Make detection
            results = pose.process(frame)

            # Recolor back to BGR
            frame.flags.writeable = True
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

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
                # if left_foot_index_y[0] < 0.9:
                     # motorL.forward(0.5)
                     # motorR.forward(0.5)
                #if left_foot_index_y[0] > 1:
                     # motorL.backward(0.5)
                     # motorR.backward(0.5)
                
                if 0.90 < left_foot_index_y[0] < 1:
                    # motorL.stop()
                    # motorR.stop()
                    if s == 0:
                        # p_s4.play()
                        s = 1
                        continue
                    else:
                        pass

                    if s == 1:
                        h_x = left_heel_x[0]
                        f_i_x = left_foot_index_x[0]
                        
                        # if (h_x - f_i_x) > 0.06:
                        #     p_s5.play()
                        #     s = 2
                        
                        pass
                # LEFT, RIGHT Calculate angle
                left_body_angle = left_calculate_angle(left_shoulder, left_hip, left_knee)
                left_leg_angle = left_calculate_angle(left_hip, left_knee, left_ankle)

                # cv2.putText(image, str(int(left_angle)),
                #            tuple(np.multiply(left_knee, [960, 240]).astype(int)),
                #            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA
                #            )
                if left_leg_angle > 160:
                    state = 'stand'
                if 90 < left_leg_angle < 110 and state == 'stand':
                    
                    state = 'squart'
                    s_count_o += 1
                    if (s_count_o == s_count_u):
                        s_set_o += 1
                        s_count_o = 0
                    m = left_knee_x[0] - left_foot_index_x[0]
                    if m < -0.02:
                        # p_s1.play()
                        s_count_o -= 1
                        if (s_count_o == -1):
                            s_count_o = 0
                    elif (left_body_angle < 80):
                        
                        # p_s2.play()
                        s_count_o -= 1
                        if (s_count_o == -1):
                            s_count_o = 0


                        # c_p[s_count_o - 1].play()
            except:
                # motorL.backward(0.5)
                # motorR.backward(0.5)
                cv2.rectangle(frame, (0, 0), (225, 73), (255, 255, 255), -1)
            cv2.putText(frame, 'SET', (15, 12),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
            cv2.putText(frame, str(s_set_o),
                        (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 0), 2, cv2.LINE_AA)
            cv2.putText(frame, 'REPS', (65, 12),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
            cv2.putText(frame, str(s_count_o),
                        (65, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 0), 2, cv2.LINE_AA)
            cv2.putText(frame, 'STATE', (120, 12),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
            cv2.putText(frame, state,
                        (120, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 0), 2, cv2.LINE_AA)# squart_count
            mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                      mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=2),
                                      mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2)
                                      )

            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()

            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

            squat = Squat()
            squat.s_set_s = s_set_o  # models의 FCuser 클래스를 이용해 db에 입력한다.
            squat.s_count_s = s_count_o
            squat.lo = lo
            db.session.add(squat)
            db.session.commit()
            if (s_set_o == s_set_u):
                camera.release()
def gen_frames_p():
    conn = sqlite3.connect('db.sqlite')
    cursor = conn.cursor()
    sql = 'SELECT p_set_u, p_count_u, lo From setting_pushup'
    cursor.execute(sql)
    rows = cursor.fetchall()
    for es in rows:
        p_set_u = es[0]
        p_count_u = es[1]
        lo = es[2]

    mp_drawing = mp.solutions.drawing_utils
    mp_pose = mp.solutions.pose

    
    # pygame.mixer.init()
    
    # motorL = Motor(forward=20, backward=21)
    # motorR = Motor(forward=19, backward=26)
    #
    # p_s = pygame.mixer.Sound('wjdaus.wav')
    # p_s1 = pygame.mixer.Sound('djvemfu.wav')
    # p_s2 = pygame.mixer.Sound('dnsehd.wav')
    # p_s3 = pygame.mixer.Sound('djdejddl2_2.wav')
    #
    # p = pygame.mixer.Sound('1.wav')
    # p1 = pygame.mixer.Sound('2.wav')
    # p2 = pygame.mixer.Sound('3.wav')
    # p3 = pygame.mixer.Sound('4.wav')
    # p4 = pygame.mixer.Sound('5.wav')
    
    # c_p = [p, p1, p2, p3, p4]
    # pi.set_servo_pulsewidth(17, v_i)
    # p_s.play()

    camera = cv2.VideoCapture(0)

    # Initiallization
    v_i = 2400
    s = 1
    n = 1
    p_count_o = 0
    p_set_o = 0
    state = 'UP'
    i = 0
    def left_calculate_angle(a, b, c):
        a = np.array(a)  # First
        b = np.array(b)  # Mid
        c = np.array(c)  # End

        radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
        left_c_angle = np.abs(radians * 180.0 / np.pi)

        if left_c_angle > 180.0:
            left_c_angle = 360 - left_c_angle

        return left_c_angle

    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        while True:
            success, frame = camera.read()  # read the camera frame
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame.flags.writeable = False

            # Make detection
            results = pose.process(frame)

            # Recolor back to BGR
            frame.flags.writeable = True
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

            try:
                landmarks = results.pose_landmarks.landmark

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
                elbow_angle = left_calculate_angle(left_shoulder, left_elbow, left_wrist)
                left_waist_angle = left_calculate_angle(left_shoulder, left_hip, left_knee)
                left_leg_angle = left_calculate_angle(left_hip, left_knee, left_ankle)
                right_leg_angle = left_calculate_angle(right_hip, right_knee, right_ankle)

                
                # if left_foot_index_y[0] < 0.9:
                #     motorL.forward(0.5)
                #     motorR.forward(0.5)
                # if left_foot_index_y[0] > 1:
                #     motorL.backward(0.5)
                #     motorR.backward(0.5)
                if 0.90 < left_foot_index_y[0] < 1:
                    # pi.set_servo_pulsewidth(17, 2300)
                    # motorL.stop()
                    # motorR.stop()
                    if s == 0:
                        # p_s1.play()
                        s = 1
                        continue
                    else:
                        pass
                if s == 1:
                        h_x = left_heel_x[0]
                        f_i_x = left_foot_index_x[0]
                       
                        
                        #if left_leg_angle > 100 and right_leg_angle > 100:
                        if left_wrist_y[0] > 0.8 and left_leg_angle > 100 and right_leg_angle > 100:
                            # p_s2.play()
                            s = 2
                if s == 2:
                    # if left_wrist_y[0] > 1:
                        # motorL.backward()
                        # motorR.backward()
                    if (p_count_o == p_count_u):
                        p_set_o += 1
                        p_set_s += 1
                        p_count_o = 0
                        p_set_s = 0
                    # if left_leg_angle > 100 and left_waist_angle > 150 and elbow_angle > 140 and state == 'DOWN':
                    #if left_leg_angle > 100 and left_waist_angle > 150 and elbow_angle > 140 and state == 'DOWN' :
                    if left_leg_angle > 100 and left_waist_angle > 150 and elbow_angle > 140 and state == 'DOWN' \
                                 and 0.8 < left_wrist_y[0]:

                        state = 'UP'
                        p_count_o += 1
                        p_count_s += 1

                        #print(pushup_count)
                        # if n == 1:
                        # print("UP")
                        # c_p[p_count_o - 1].play()
                    #elif left_leg_angle > 100 and elbow_angle < 120 and state == 'UP':
                    elif left_leg_angle > 100 and elbow_angle < 120 and state == 'UP' and 0.8 < left_wrist_y[0]:
                        n = 1
                        state = 'DOWN'
                        # print("DOWN")
                        #print(left_leg_angle)
                        #print(elbow_angle)
                        #print(left_waist_angle)
                        if left_waist_angle < 150:
                            # p_s3.play()
                            # print("djdejddl")
                            p_count_o -= 1
                            if (p_count_o == -1):
                                p_count_o = 0
                                p_count_s = 0
                            n = 0
                        else:
                            pass
                    else:
                        pass
            except:
                # pi.set_servo_pulsewidth(17, 2250)
                # motorL.backward(0.5)
                # motorR.backward(0.5)
                
                pass
            cv2.rectangle(frame, (0, 0), (300, 73), (255, 255, 255), -1)
            cv2.putText(frame, 'SET', (15, 12),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
            cv2.putText(frame, str(p_set_o),
                        (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 0), 2, cv2.LINE_AA)
            cv2.putText(frame, 'REPS', (65, 12),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
            cv2.putText(frame, str(p_count_o),
                        (65, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 0), 2, cv2.LINE_AA)
            cv2.putText(frame, 'STATE', (180, 12),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
            cv2.putText(frame, state,
                        (120, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 0), 2, cv2.LINE_AA)
            mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                      mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=2),
                                      mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2)
                                      )
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

            conn = sqlite3.connect('db.sqlite')
            cursor = conn.cursor()
            sql = 'SELECT p_set_u, p_count_u, lo  FROM Setting_pushup ORDER BY ROWID DESC LIMIT 1'
            cursor.execute(sql)
            rows = cursor.fetchall()
            for es in rows:
                p_set_u = es[0]
                p_count_u = es[1]
                lo = es[2]

            p_set_s = p_set_o
            p_count_s = p_count_o
            pushup = Pushup()
            pushup.p_set_s = p_set_s  # models의 FCuser 클래스를 이용해 db에 입력한다.
            pushup.p_count_s = p_count_s
            pushup.lo = lo
            db.session.add(pushup)
            db.session.commit()

            if (p_set_o == p_set_u):
                camera.release()


@cpp.route('/squart', methods=['GET', 'POST'])
def index_s():
    if request.method == 'GET':
        conn = sqlite3.connect('db.sqlite')
        cursor = conn.cursor()
        sql = 'SELECT s_time_u FROM setting_squat ORDER BY ROWID DESC LIMIT 1'
        cursor.execute(sql)
        r = cursor.fetchall()
        for i in r:
            h = i[0]
        return render_template('squart.html',num=h)


@cpp.route('/video_feed_s', methods=['GET', 'POST'])
def video_feed_s():
    if request.method == 'GET':
        return Response(gen_frames_s(), mimetype='multipart/x-mixed-replace; boundary=frame')

@cpp.route('/pushup', methods=['GET'])
def index_p():
    if request.method == 'GET':
        conn = sqlite3.connect('db.sqlite')
        cursor = conn.cursor()
        sql = 'SELECT p_time_u FROM setting_pushup ORDER BY ROWID DESC LIMIT 1'
        cursor.execute(sql)
        r = cursor.fetchall()
        for i in r:
            t = i[0]
        return render_template('pushup.html',num=t)



@cpp.route('/video_feed_p', methods=['GET', 'POST'])
def video_feed_p():
    if request.method == 'GET':
        return Response(gen_frames_p(), mimetype='multipart/x-mixed-replace; boundary=frame')


@cpp.route('/')
def root():
    session.clear()
    return render_template('main.html')


@cpp.route('/main2', methods=['GET', 'POST'])
def main():
    if request.method == 'POST':
        return render_template('main2.html')
    else:
        temp = request.args.get('hide')
        conn = sqlite3.connect('db.sqlite')
        cursor = conn.cursor()
        sql = 'SELECT p_set_clear, p_count_clear, lo, today_p FROM pushup_clear ORDER BY ROWID DESC LIMIT 1'
        cursor.execute(sql)
        rows = cursor.fetchall()
        for e in rows:
            p_set_c = e[0]
            p_count_c =e[1]
            lo = e[2]
            today_p = e[3]
        time_p = Time_p()
        time_p.temp_p = temp
        time_p.p_set_c=p_set_c
        time_p.p_count_c =p_count_c
        time_p.today_p =today_p
        time_p.lo = lo
        db.session.add(time_p)
        db.session.commit()
        sql = 'DELETE FROM time_p WHERE temp_p is NULL'
        cursor.execute(sql)
        conn.commit()
        sql4 = 'SELECT s_set_clear, s_count_clear, lo, today_s FROM squat_clear ORDER BY ROWID DESC LIMIT 1'
        cursor.execute(sql4)
        rows = cursor.fetchall()
        for e in rows:
            s_set_c = e[0]
            s_count_c = e[1]
            lo = e[2]
            today_s = e[3]
        time_s = Time_s()
        time_s.temp_s = temp
        time_s.s_set_c = s_set_c
        time_s.s_count_c = s_count_c
        time_s.today_s = today_s
        time_s.lo = lo
        db.session.add(time_s)
        db.session.commit()
        sql5 = 'DELETE FROM time_s WHERE temp_s is NULL'
        cursor.execute(sql5)
        conn.commit()
        sql2 = 'DELETE FROM pushup'
        cursor.execute(sql2)
        conn.commit()
        sql3 = 'DELETE FROM squat'
        cursor.execute(sql3)
        conn.commit()
        return render_template('main2.html',hide=temp)


@cpp.route('/login_proc', methods=['POST'])
def login_proc():
    if request.method == 'POST':
        userid = request.form['userid']
        password = request.form['password']
        if userid == "":
            flash("Please Input USERID")
            return render_template("main.html")
        if password == "":
            flash("Please Input PASSWORD")
            return render_template("main.html")

        conn = sqlite3.connect('db.sqlite')
        cursor = conn.cursor()
        sql = 'SELECT id, userid, password, username FROM fcuser WHERE userId = ?'
        cursor.execute(sql, (userid,))
        rows = cursor.fetchall()
        for rs in rows:
            if userid == rs[1] and password == rs[2]:
                session['logFlag'] = True
                session['id'] = rs[0]
                session['userid'] = userid
                session['password'] = password
                session['username'] = rs[3]
                flash("welcome")
                return render_template('main2.html')
            else:
                flash("incorrect")
                return redirect(url_for('main'))
    else:
        return render_template("main.html")  # 메소드를 호출


@cpp.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'GET':
        return render_template("register.html")
    else:
        # 회원정보 생성
        userid = request.form.get('userid')
        username = request.form.get('username')
        password = request.form.get('password')
        re_password = request.form.get('re_password')
        print(password)  # 들어오나 확인해볼 수 있다.
        if not (userid and username and password and re_password):
            return "모두 입력해주세요"
        elif password != re_password:
            return "비밀번호를 확인해주세요"
        else:  # 모두 입력이 정상적으로 되었다면 밑에명령실행(DB에 입력됨)
            fcuser = Fcuser()
            fcuser.password = password  # models의 FCuser 클래스를 이용해 db에 입력한다.
            fcuser.userid = userid
            fcuser.username = username
            db.session.add(fcuser)
            db.session.commit()
            return redirect(url_for('root'))


@cpp.route('/setting_s', methods=['GET', 'POST'])
def setting_s():
    if request.method == 'GET':
        return render_template("setting_s.html")
    else:
        # 회원정보 생성
        s_set_u = request.form.get('s_set_u')
        s_count_u = request.form.get('s_count_u')
        s_time_u = request.form.get('s_time_u')
        if not (s_set_u and s_count_u and s_time_u):
            return "모두 입력해주세요"
        else:  # 모두 입력이 정상적으로 되었다면 밑에명령실행(DB에 입력됨)
            setting_squat = Setting_squat()
            setting_squat.s_time_u = s_time_u  # models의 FCuser 클래스를 이용해 db에 입력한다.
            setting_squat.s_set_u = s_set_u
            setting_squat.s_count_u = s_count_u
            setting_squat.lo = session['id']
            db.session.add(setting_squat)
            db.session.commit()
            flash("complete")
            return redirect(url_for('index_s'))


@cpp.route('/setting_p', methods=['GET', 'POST'])
def setting_p():
    if request.method == 'GET':
        conn = sqlite3.connect('db.sqlite')
        cursor = conn.cursor()
        sql5 = 'DELETE FROM pushup_clear WHERE p_set_clear is 999'
        cursor.execute(sql5)
        conn.commit()
        sql6 = 'DELETE FROM squat_clear WHERE s_set_clear is 999'
        cursor.execute(sql6)
        conn.commit()
        return render_template("setting_p.html")
    else:
        # 회원정보 생성
        p_set_u = request.form.get('p_set_u')
        p_count_u = request.form.get('p_count_u')
        p_time_u = request.form.get('p_time_u')
        lo = session['id']

        if not (p_set_u and p_count_u and p_time_u):
            return "모두 입력해주세요"
        else:  # 모두 입력이 정상적으로 되었다면 밑에명령실행(DB에 입력됨)
            setting_pushup = Setting_pushup()
            setting_pushup.p_time_u = p_time_u  # models의 FCuser 클래스를 이용해 db에 입력한다.
            setting_pushup.p_set_u = p_set_u
            setting_pushup.p_count_u = p_count_u
            setting_pushup.lo = lo
            db.session.add(setting_pushup)
            db.session.commit()
            flash("complete")
            return redirect(url_for('index_p'))
@cpp.route('/challenge_s1', methods=['GET', 'POST'])
def challenge_s1():
    if request.method == 'GET':
            setting_squat = Setting_squat()
            setting_squat.s_time_u = 90  # models의 FCuser 클래스를 이용해 db에 입력한다.
            setting_squat.s_set_u = 3
            setting_squat.s_count_u = 3
            setting_squat.lo = session['id']
            db.session.add(setting_squat)
            db.session.commit()
            return redirect(url_for('index_s'))

@cpp.route('/challenge_s2', methods=['GET', 'POST'])
def challenge_s2():
    if request.method == 'GET':
            setting_squat = Setting_squat()
            setting_squat.s_time_u = 150  # models의 FCuser 클래스를 이용해 db에 입력한다.
            setting_squat.s_set_u = 3
            setting_squat.s_count_u = 8
            setting_squat.lo = session['id']
            db.session.add(setting_squat)
            db.session.commit()
            return redirect(url_for('index_s'))

@cpp.route('/challenge_s3', methods=['GET', 'POST'])
def challenge_s3():
    if request.method == 'GET':
            setting_squat = Setting_squat()
            setting_squat.s_time_u = 300  # models의 FCuser 클래스를 이용해 db에 입력한다.
            setting_squat.s_set_u = 5
            setting_squat.s_count_u = 10
            setting_squat.lo = session['id']
            db.session.add(setting_squat)
            db.session.commit()
            return redirect(url_for('index_s'))


@cpp.route('/challenge_p1', methods=['GET', 'POST'])
def challenge_p1():
    if request.method == 'GET':
            setting_pushup = Setting_pushup()
            setting_pushup.p_time_u = 90  # models의 FCuser 클래스를 이용해 db에 입력한다.
            setting_pushup.p_set_u = 3
            setting_pushup.p_count_u = 3
            setting_pushup.lo = session['id']
            db.session.add(setting_pushup)
            db.session.commit()
            return redirect(url_for('index_p'))


@cpp.route('/challenge_p2', methods=['GET', 'POST'])
def challenge_p2():
    if request.method == 'GET':
            setting_pushup = Setting_pushup()
            setting_pushup.p_time_u = 150  # models의 FCuser 클래스를 이용해 db에 입력한다.
            setting_pushup.p_set_u = 3
            setting_pushup.p_count_u = 8
            setting_pushup.lo = session['id']
            db.session.add(setting_pushup)
            db.session.commit()
            return redirect(url_for('index_p'))


@cpp.route('/challenge_p3', methods=['GET', 'POST'])
def challenge_p3():
    if request.method == 'GET':
            setting_pushup = Setting_pushup()
            setting_pushup.p_time_u = 300  # models의 FCuser 클래스를 이용해 db에 입력한다.
            setting_pushup.p_set_u = 5
            setting_pushup.p_count_u = 10
            setting_pushup.lo = session['id']
            db.session.add(setting_pushup)
            db.session.commit()
            return redirect(url_for('index_p'))


@cpp.route('/record1', methods=['GET', 'POST'])
def record1():
    if request.method == 'GET':
        conn = sqlite3.connect('db.sqlite')
        cursor = conn.cursor()
        sql = 'SELECT p_set_c, p_count_c, temp_p , lo , today_p FROM time_p'
        cursor.execute(sql)
        data_list_p = cursor.fetchall()
        sql1 = 'SELECT s_set_c, s_count_c, temp_s, lo , today_s FROM time_s'
        cursor.execute(sql1)
        data_list_s = cursor.fetchall()
        return render_template('record1.html', data_list_p=data_list_p, data_list_s=data_list_s)

@cpp.route('/result_p',methods=['GET','POST'])
def result_p():
    temp = request.args.get('hide')
    conn = sqlite3.connect('db.sqlite')
    cursor = conn.cursor()
    sql = 'SELECT p_set_s, p_count_s, lo FROM pushup ORDER BY ROWID DESC LIMIT 1'
    cursor.execute(sql)
    rows = cursor.fetchall()
    for e in rows:
        q = e[0]
        r = e[1]
        m = e[2]
    sql2 = 'INSERT INTO squat_clear(s_set_clear, s_count_clear, s_time_clear, lo, today_s) VALUES(NULL,NULL,NULL,NULL,NULL);'
    cursor.execute(sql2)
    conn.commit()
    now = datetime.now()
    current_time = now.strftime("%Y.%m.%d %H:%M:%S")
    pushup_clear = Pushup_clear()
    pushup_clear.p_set_clear = q
    pushup_clear.p_count_clear = r
    pushup_clear.lo = m
    pushup_clear.today_p = current_time
    db.session.add(pushup_clear)
    db.session.commit()
    print(temp, file=sys.stderr)
    sql1 = 'SELECT p_set_u, p_count_u, p_time_u FROM setting_pushup ORDER BY ROWID DESC LIMIT 1'
    cursor.execute(sql1)
    rows = cursor.fetchall()
    for e in rows:
        p = e[0]
        t = e[1]
        s = e[2]
    x = s
    v = t
    j = q * r

    if j == 0:
        if q == 0:
            j = r
        if r == 0:
             j = q * v
    else:
        j = q * v + r

    return render_template('result_p.html',p_set=p,p_count=v,p_set_c=q,p_count_c=j,hide=temp,num1=x)


@cpp.route('/result_s',methods=['GET','POST'])
def result_s():
    temp = request.args.get('hide')
    conn = sqlite3.connect('db.sqlite')
    cursor = conn.cursor()
    sql = 'SELECT s_set_s, s_count_s, lo FROM squat ORDER BY ROWID DESC LIMIT 1'
    cursor.execute(sql)
    rows = cursor.fetchall()
    for e in rows:
        q = e[0]
        r = e[1]
        m = e[2]
    sql2 = 'INSERT INTO pushup_clear(p_set_clear, p_count_clear, p_time_clear, lo, today_p) VALUES(NULL,NULL,NULL,NULL,NULL);'
    cursor.execute(sql2)
    conn.commit()
    now = datetime.now()
    current_time = now.strftime("%Y.%m.%d %H:%M:%S")
    squat_clear = Squat_clear()
    squat_clear.s_set_clear = q
    squat_clear.s_count_clear = r
    squat_clear.lo = m
    squat_clear.today_s = current_time
    db.session.add(squat_clear)
    db.session.commit()
    sql1 = 'SELECT s_set_u, s_count_u, s_time_u FROM setting_squat ORDER BY ROWID DESC LIMIT 1'
    cursor.execute(sql1)
    rows = cursor.fetchall()
    for e in rows:
        p = e[0]
        t = e[1]
        s = e[2]
    x = s
    v = t
    j = q * r

    if j == 0:
        if q == 0:
            j = r
        if r == 0:
             j = q * v
    else:
        j = q * v + r

    return render_template('result_s.html',p_set=p,p_count=v,p_set_c=q,p_count_c=j,hide=temp,num1=x)


@cpp.route('/logout')
def logout():
    session.clear()
    return redirect('/')


if __name__ == '__main__':
    cpp.secret_key = '20200601'
    cpp.debug = True
    basedir = os.path.abspath(os.path.dirname(__file__))  # database 경로를 절대경로로 설정함
    dbfile = os.path.join(basedir, 'db.sqlite')  # 데이터베이스 이름과 경로
    cpp.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///' + dbfile
    cpp.config['SQLALCHEMY_COMMIT_ON_TEARDOWN'] = True  # 사용자에게 원하는 정보를 전달완료했을때가 TEARDOWN, 그 순간마다 COMMIT을 하도록 한다.라는 설정
    # 여러가지 쌓아져있던 동작들을 Commit을 해주어야 데이터베이스에 반영됨. 이러한 단위들은 트렌젝션이라고함.
    cpp.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False  # True하면 warrnig메시지 유발,
    db.init_app(cpp)  # 초기화 후 db.app에 app으로 명시적으로 넣어줌
    db.app = cpp
    # db.create_all()  # 이 명령이 있어야 생성됨. DB가
#     cpp.run(host="192.168.0.112", port="8000",debug=True)
    cpp.run(port=10001, debug=True)
