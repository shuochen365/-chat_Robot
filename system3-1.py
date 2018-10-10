import sys
sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')  #主要针对my system

import json
from os.path import join, dirname
from watson_developer_cloud import SpeechToTextV1
from watson_developer_cloud.websocket import RecognizeCallback
import os
import watson_developer_cloud
from watson_developer_cloud import TextToSpeechV1
from threading import Thread
import time
import cv2
import pyaudio
import wave
import multiprocessing

speech_to_text = SpeechToTextV1(
    username='ec624e5b-a01e-47a0-acbc-89813c22f412',
    password='zOHsOZYfssKl',
    url='https://stream.watsonplatform.net/speech-to-text/api')


conversation = watson_developer_cloud.ConversationV1(username='b1623c3f-9feb-4150-8dea-c9259951452e',
	                                             password='CzBVl0KTt4Ye',
	                                             version='2017-04-21')

workspace_id='31965070-5130-4f8b-8253-c337ca71eb2b'

text_to_speech = TextToSpeechV1(
    username='4e316583-1dad-4f75-bbd3-1d85c1548378',
    password='YL6eG10XhRhb')

from statistics import mode
from keras.models import load_model
import numpy as np

from utils.datasets import get_labels
from utils.inference import detect_faces
from utils.inference import draw_text
from utils.inference import draw_bounding_box
from utils.inference import apply_offsets
from utils.inference import load_detection_model
from utils.preprocessor import preprocess_input



###########################################################
#功能：心情分析、性别分析、界面显示
#说明：子线程
#操作：无
###########################################################

def face_gender(name):                                   #休眠5ms

	cv2.namedWindow('AI3',0)               #创建窗口
	cv2.resizeWindow('AI3', 800, 400);   #创建一个640*480大小的窗口
	cap = cv2.VideoCapture('1.avi')
	currentFrame = 0
	totalFrame = cap.get(7)
	print (totalFrame)

	# parameters for loading data and images
	detection_model_path = './trained_models/detection_models/haarcascade_frontalface_default.xml'
	emotion_model_path = './trained_models/emotion_models/fer2013_mini_XCEPTION.102-0.66.hdf5'
	gender_model_path = './trained_models/gender_models/simple_CNN.81-0.96.hdf5'
	emotion_labels = get_labels('fer2013')
	gender_labels = get_labels('imdb')
	font = cv2.FONT_HERSHEY_SIMPLEX

	# hyper-parameters for bounding boxes shape
	frame_window = 10
	gender_offsets = (30, 60)
	emotion_offsets = (20, 40)

	# loading models
	face_detection = load_detection_model(detection_model_path)
	emotion_classifier = load_model(emotion_model_path, compile=False)
	gender_classifier = load_model(gender_model_path, compile=False)

	# getting input model shapes for inference
	emotion_target_size = emotion_classifier.input_shape[1:3]
	gender_target_size = gender_classifier.input_shape[1:3]

	# starting lists for calculating modes
	gender_window = []
	emotion_window = []

	# starting video streaming
	#cv2.namedWindow('window_frame')
	video_capture = cv2.VideoCapture(0)
#图像大小为400 * 400
	while 1:
		time.sleep(0.01)
		ret,frame = cap.read()  # 获取图像
		#cv2.imshow("AI", frame)  # 显示帧
		currentFrame += 1
		#print (currentFrame)
		if currentFrame >= (totalFrame - 1):
			currentFrame = 0
			cap.set(cv2.CAP_PROP_POS_FRAMES, 1)
		cv2.waitKey(1)

		bgr_image = video_capture.read()[1]
		gray_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2GRAY)
		rgb_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)
		faces = detect_faces(face_detection, gray_image)

		for face_coordinates in faces:

			x1, x2, y1, y2 = apply_offsets(face_coordinates, gender_offsets)
			rgb_face = rgb_image[y1:y2, x1:x2]

			x1, x2, y1, y2 = apply_offsets(face_coordinates, emotion_offsets)
			gray_face = gray_image[y1:y2, x1:x2]
			try:
				rgb_face = cv2.resize(rgb_face, (gender_target_size))
				gray_face = cv2.resize(gray_face, (emotion_target_size))
			except:
				continue
			gray_face = preprocess_input(gray_face, False)
			gray_face = np.expand_dims(gray_face, 0)
			gray_face = np.expand_dims(gray_face, -1)
			emotion_label_arg = np.argmax(emotion_classifier.predict(gray_face))
			emotion_text = emotion_labels[emotion_label_arg]
			emotion_window.append(emotion_text)

			rgb_face = np.expand_dims(rgb_face, 0)
			rgb_face = preprocess_input(rgb_face, False)
			gender_prediction = gender_classifier.predict(rgb_face)
			gender_label_arg = np.argmax(gender_prediction)
			gender_text = gender_labels[gender_label_arg]
			gender_window.append(gender_text)

			if len(gender_window) > frame_window:
			    emotion_window.pop(0)
			    gender_window.pop(0)
			try:
			    emotion_mode = mode(emotion_window)
			    gender_mode = mode(gender_window)
			except:
			    continue

			if gender_text == gender_labels[0]:
			    color = (0, 0, 255)
			else:
			    color = (255, 0, 0)

			draw_bounding_box(face_coordinates, rgb_image, color)
			draw_text(face_coordinates, rgb_image, gender_mode,
				  color, 0, -20, 1, 1)
			draw_text(face_coordinates, rgb_image, emotion_mode,
				  color, 0, -45, 1, 1)

		bgr_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)
		size = (int(400), int(400))  
		bgr_image = cv2.resize(bgr_image, size, interpolation=cv2.INTER_AREA) 
		frame = cv2.resize(frame, size, interpolation=cv2.INTER_AREA) 
		htitch= np.hstack((bgr_image, frame))
		#cv2.imshow('window_frame', bgr_image)
		cv2.imshow('AI3', htitch)
		if cv2.waitKey(1) & 0xFF == ord('q'):
			break

def robot(name):                                   #休眠5ms
	while 1:                                         #休眠5ms
		time.sleep(0.01)

		CHUNK = 1024                        #块
		FORMAT = pyaudio.paInt16            #音频格式
		CHANNELS = 2                        #设置通道数
		RATE = 44100                        #帧数
		RECORD_SECONDS = 5                 
		WAVE_OUTPUT_FILENAME = "output.wav" #
		p = pyaudio.PyAudio()

		frames = []
		recordFlag = 0

		while 1:
			if cv2.waitKey(10) == 98:           #代表'b'推出录音
				recordFlag = 1
			if cv2.waitKey(10) == 97:           #代表'a'开始录音
				stream = p.open(format=FORMAT,
			 	       channels=CHANNELS,
			 	       rate=RATE,
			 	       input=True,
			 	       frames_per_buffer=CHUNK)
				recordFlag = 2
				print("* recording")
			if recordFlag == 2:
				data = stream.read(CHUNK)
				frames.append(data)
			if recordFlag == 1:
				recordFlag = 0
				break

		print("* done recording")

		stream.stop_stream()
		stream.close()
		p.terminate()

		wf = wave.open(WAVE_OUTPUT_FILENAME, 'wb')
		wf.setnchannels(CHANNELS)
		wf.setsampwidth(p.get_sample_size(FORMAT))
		wf.setframerate(RATE)
		wf.writeframes(b''.join(frames))
		wf.close()

###########################################################
#功能：语音转化为文字
#操作：
###########################################################
		while 1:

			with open('output.wav','rb') as audio_file:


				speechTxt = speech_to_text.recognize(
						audio=audio_file,
						content_type='audio/wav',
						timestamps=False,
						word_confidence=False)['results']
				if speechTxt == []:                              #解决语音内容为空的问题
					break
				res = json.dumps(
						speechTxt[0]['alternatives'][0]['transcript'],
						indent=2)

				print (res)


###########################################################
#功能：机器人对话
#操作：
###########################################################
			input = {'text': res}
			response = conversation.message(workspace_id=workspace_id,
				                input=input)
			print (response['output']['text'])
			if response['output']['text'] == []:                   #解决无法匹配意图
				print (response)
				break
			#print (response['output']['text'][0])

###########################################################
#功能：文本转化为语音
#操作：
###########################################################
			text = response['output']['text'][0]
			print (text)
			with open(join(dirname(__file__), 'output2.wav'),
				  'wb') as audio_file:
			    audio_file.write(
				text_to_speech.synthesize(text, accept='audio/wav',
						          voice="en-US_AllisonVoice").content)

			wf = wave.open('output2.wav', 'rb')

			p = pyaudio.PyAudio()

			stream = p.open(format=p.get_format_from_width(wf.getsampwidth()),
					channels=wf.getnchannels(),
					rate=wf.getframerate(),
					output=True)

			data = wf.readframes(CHUNK)

			while len(data) > 0:
			    stream.write(data)
			    data = wf.readframes(CHUNK)
			    if data == '':
			    	print (data)
			stream.stop_stream()
			stream.close()
			p.terminate()
			break                                               #使程序跳出当前循环，否则一直循环上次内容

	cv2.destroyAllWindows()


###########################################################
#功能：界面显示
#操作：无
###########################################################
		#img = cv2.imread('1.jpg',cv2.IMREAD_GRAYSCALE)
		#cv2.namedWindow('AI',0)               #创建窗口
		#cv2.resizeWindow("AI", 640, 480);   #创建一个500*500大小的窗口
		#cv2.imshow('AI',img)        
		#cv2.waitKey(10)
		#print('%s say hello' %name)

###########################################################
#功能：国际巨星理发店
#说明：主线程
#操作：无
###########################################################
if __name__ == '__main__':
	#t=Thread(target=face_gender,args=('hh',))
	#t.start()
	#t1=Thread(target=face_gender,args=('hh',))
	#t2=Thread(target=robot,args=('hh',))
	#t1.start()
	#t2.start()


	p1 = multiprocessing.Process(target=face_gender, args=('hh',))
	p1.start()
	p2 = multiprocessing.Process(target=robot, args=('hh',))
	p2.start()
	p1.join()
	p2.join()





















