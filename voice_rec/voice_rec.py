# -*- coding:utf-8 -*-
import wave
import requests
import time
import base64
import numpy as np
from pyaudio import PyAudio, paInt16
import time
from playsound import playsound
import os
import sys


framerate = 16000  #
num_samples = 2000
channels = 1
sampwidth = 2  #
FILEPATH = './voice_rec/speech.wav'

base_url = "https://openapi.baidu.com/oauth/2.0/token?grant_type=client_credentials&client_id=%s&client_secret=%s"
APIKey = "xquGU6uUM5EUMmnjbWGkkGUG"
SecretKey = "nfhYce3srBPwc6VQGbYL6KhGv3Cuwoo7"

HOST = base_url % (APIKey, SecretKey)


def getToken(host):
    try:
        res = requests.post(host)
        # time.sleep(1)
        return res.json()['access_token']
    except:
        print("return res.json()['access_token'] fail !")
    else:
        print("return res.json()['access_token'] success !")


def save_wave_file(filepath, data):
    wf = wave.open(filepath, 'wb')
    wf.setnchannels(channels)
    wf.setsampwidth(sampwidth)
    wf.setframerate(framerate)
    wf.writeframes(b''.join(data))
    wf.close()


def my_record():
    pa = PyAudio()
    stream = pa.open(format=paInt16, channels=channels,
                     rate=framerate, input=True, frames_per_buffer=num_samples)
    my_buf = []
    volume_flag = False
    time_flag = True

    while True:
        string_audio_data = stream.read(num_samples)
        try:
            audio_data = np.fromstring(string_audio_data, dtype=np.short)
        except:
            pass
        qiangdu = np.max(audio_data)
        qiangdu1 = qiangdu//100  # 声音强度值
        print('当前声音强度值：'+str(qiangdu1))

        if qiangdu1 < 80:
            if volume_flag:
                if time_flag:
                    t0 = time.perf_counter()
                    time_flag = False
                else:
                    t1 = time.perf_counter()
                    if ((t1-t0) >= 1.5):  # 声音强度<80后1.5s结束, 这个值可能需要现场调试
                        break
                my_buf.append(string_audio_data)
            continue
        else:
            volume_flag = True
            my_buf.append(string_audio_data)

        # if qiangdu1 < 50:
        #     if volume_flag:
        #         if time_flag:
        #             t0 = time.perf_counter()
        #             time_flag = False
        #         else:
        #             t1 = time.perf_counter()
        #             if ((t1-t0) >= 2):  # 声音强度<50后2s结束, 这个值可能需要现场调试
        #                 break
        #     continue
        # elif qiangdu1 > 20:
        #     volume_flag = True
        #     my_buf.append(string_audio_data)

    # print('录音结束.')
    save_wave_file(FILEPATH, my_buf)
    stream.close()


def get_audio(file):
    with open(file, 'rb') as f:
        data = f.read()
    return data


def speech2text(speech_data, token, dev_pid=1537):
    FORMAT = 'wav'
    RATE = '16000'
    CHANNEL = 1
    CUID = '*******'
    SPEECH = base64.b64encode(speech_data).decode('utf-8')

    data = {
        'format': FORMAT,
        'rate': RATE,
        'channel': CHANNEL,
        'cuid': CUID,
        'len': len(speech_data),
        'speech': SPEECH,
        'token': token,
        'dev_pid': dev_pid
    }
    url = 'https://vop.baidu.com/server_api'
    headers = {'Content-Type': 'application/json'}
    # r=requests.post(url,data=json.dumps(data),headers=headers)
    print('识别中...')
    r = requests.post(url, json=data, headers=headers)
    Result = r.json()
    if 'result' in Result:
        return Result['result'][0]
    else:
        return Result

def one_record():
    print('************  您请说：')
    my_record()  # 进行录音
    TOKEN = getToken(HOST)
    speech = get_audio(FILEPATH)
    result = speech2text(speech, TOKEN, int(1537))
    if type(result) == str:
        return result
    else:
        return '未听到您说话^-^'

if __name__ == '__main__':
    while True:
        print('************  您请说：')
        my_record()  # 进行录音
        TOKEN = getToken(HOST)
        speech = get_audio(FILEPATH)
        result = speech2text(speech, TOKEN, int(1537))
        if type(result) == str:
            print('rec result:'+result)
        else:
            print('未听到您说话^-^')