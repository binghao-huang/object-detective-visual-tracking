# -*- coding:utf-8 -*-
import pyaudio
import wave
from aip import AipSpeech
import time

class AudioRecognition(object):
    def __init__(self):
        p = pyaudio.PyAudio()
        self.dir=p.get_device_info_by_index(0)
        self.chunk = 1024
        self.sample_format = pyaudio.paInt16
        self.channels = self.dir['maxInputChannels']  #检测当前麦克风的最大声道
        self.fs = 8000   #采样频率
        self.seconds = 4   #每次录制时间
        self.filename = "output.wav" #输出文件名
        self.result = '未识别'  #识别结果
        #百度aip接口
        self.APP_ID = '16007034'
        self.API_KEY = '9cVZDkCrl0sZP3wpQlMeqZq2'
        self.SECRET_KEY = 'lGTYdBrcomGUAgfPCt2jrYO9Rg68IMAB'

    def record(self):#录入
        p = pyaudio.PyAudio()  # Create an interface to PortAudio
        stream = p.open(format=self.sample_format,
                        channels=self.channels,
                        rate=self.fs,
                        frames_per_buffer=self.chunk,
                        input=True,
                        )
        frames = []
        for i in range(0, int(self.fs / self.chunk * self.seconds)):
            data = stream.read(self.chunk)
            frames.append(data)
            if i % 5 == 0:
                print("*")
        stream.stop_stream()
        stream.close()
        p.terminate()
        wf = wave.open(self.filename, 'wb')
        wf.setnchannels(self.channels)
        wf.setsampwidth(p.get_sample_size(self.sample_format))
        wf.setframerate(self.fs)
        wf.writeframes(b''.join(frames))
        wf.close()

    def recognition(self): #识别
        client = AipSpeech(self.APP_ID, self.API_KEY, self.SECRET_KEY)

        # 读取文件
        def get_file_content(file_path):
            with open(file_path, 'rb') as fp:
                return fp.read()

        # 识别本地文件
        result = client.asr(get_file_content(self.filename), 'wav', 16000, {
            'dev_pid': 1537,  # 默认1537（普通话 输入法模型）
        })
        self.result = result['result'][0]

    def microphone(self): #设备识别,打印系统音频设备参数
        p = pyaudio.PyAudio()
        print(p)
        for i in range(p.get_device_count()):
            print(p.get_device_info_by_index(i))
        print(p.get_device_info_by_index)

a=AudioRecognition()

print("开始录制")
#a.microphone()
a.record()
print("正在识别......")
a.recognition()
print("结果："+a.result)

