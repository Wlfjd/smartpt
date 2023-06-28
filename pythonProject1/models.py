from typing import Set
from flask_sqlalchemy import SQLAlchemy

db = SQLAlchemy()           #SQLAlchemy를 사용해 데이터베이스 저장

class Fcuser(db.Model): 
    __tablename__ = 'fcuser'   #테이블 이름 : fcuser
    id = db.Column(db.Integer, primary_key = True)   #id를 프라이머리키로 설정
    password = db.Column(db.String(64))     #패스워드를 받아올 문자열길이 
    userid = db.Column(db.String(32))       #이하 위와 동일
    username = db.Column(db.String(8))
      
class Setting_pushup(db.Model):
    __tablename__ = 'setting_pushup'
    id = db.Column(db.Integer, primary_key = True) 
    p_set_u = db.Column(db.Integer)
    p_count_u = db.Column(db.Integer)
    p_time_u = db.Column(db.Integer)
    lo = db.Column(db.Integer)

class Setting_squat(db.Model):
    __tablename__ = 'setting_squat'
    id = db.Column(db.Integer, primary_key = True)
    s_set_u = db.Column(db.Integer)
    s_count_u = db.Column(db.Integer)
    s_time_u = db.Column(db.Integer)
    lo = db.Column(db.Integer)

class Pushup(db.Model):
    __tablename__ = 'pushup'
    id = db.Column(db.Integer, primary_key=True)
    p_set_s = db.Column(db.Integer)
    p_count_s = db.Column(db.Integer)
    p_time_s = db.Column(db.Integer)
    lo = db.Column(db.Integer)

class Pushup_clear(db.Model):
    __tablename__ = 'pushup_clear'
    id = db.Column(db.Integer, primary_key=True)
    p_set_clear = db.Column(db.Integer)
    p_count_clear = db.Column(db.Integer)
    p_time_clear = db.Column(db.Integer)
    lo = db.Column(db.Integer)
    today_p = db.Column(db.Integer)

class Squat_clear(db.Model):
    __tablename__ = 'squat_clear'
    id = db.Column(db.Integer, primary_key=True)
    s_set_clear = db.Column(db.Integer)
    s_count_clear = db.Column(db.Integer)
    s_time_clear = db.Column(db.Integer)
    lo = db.Column(db.Integer)
    today_s = db.Column(db.Integer)

class Squat(db.Model):
    __tablename__ = 'squat'
    id = db.Column(db.Integer, primary_key = True)
    s_set_s = db.Column(db.Integer)
    s_count_s = db.Column(db.Integer)
    s_time_s = db.Column(db.Integer)
    lo = db.Column(db.Integer)

class Time_p(db.Model):
    __tablename__ = 'time_p'
    id = db.Column(db.Integer, primary_key = True)
    p_set_c = db.Column(db.Integer)
    p_count_c = db.Column(db.Integer)
    temp_p = db.Column(db.Integer)
    today_p = db.Column(db.Integer)
    lo = db.Column(db.Integer)

class Time_s(db.Model):
    __tablename__ = 'time_s'
    id = db.Column(db.Integer, primary_key = True)
    s_set_c = db.Column(db.Integer)
    s_count_c = db.Column(db.Integer)
    temp_s = db.Column(db.Integer)
    today_s = db.Column(db.Integer)
    lo = db.Column(db.Integer)