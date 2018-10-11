
import sys
sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')  # mainly aimed at  my system

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
from multiprocessing import Process, Queue

speech_to_text = SpeechToTextV1(
    username='ec624e5b-a01e-47a0-acbc-89813c22f412',
    password='zOHsOZYfssKl',
    url='https://stream.watsonplatform.net/speech-to-text/api')


conversation = watson_developer_cloud.ConversationV1(username='b1623c3f-9feb-4150-8dea-c9259951452e',
	                                             password='CzBVl0KTt4Ye',
	                                             version='2017-04-21')

workspace_id='10fc0028-a960-44ed-af41-f16d5270356c'

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

#htitch = im_out = np.zeros((400, 800, 3), np.uint8);
dict = {'longOrShort': 0, 'color': 0}  #0: no haircut  1:long hair 2:short hair  0: no hair dyed  1:red 2:yellow
menPath = {'1': './men/1.jpg', '2': './men/2.jpg', '3': './men/3.jpg', '4': './men/4.jpg', '5': './men/5.jpg', '6': './men/7.jpg'}
###########################################################
#function： mood analysis, gender analysis, interface display 
#explain：thread
###########################################################

def face_gender(inf, inf3):                     #sleep 5ms

	#cv2.namedWindow('AI3',0)               #new window
	#cv2.resizeWindow('AI3', 800, 400);     #640*480
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
	runFlag = 0
	displayTimes = 0
	pictureNum = 0
	menNumber = 0
	womenNumber = 0
	showHairStatus = 0

	while 1:

		time.sleep(0.01)
		#cv2.imshow("AI", frame)  # show fps
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
				  color, 0, -20, 1, 2)
			draw_text(face_coordinates, rgb_image, emotion_mode,
				  color, 0, -45, 1, 2)
			#print (gender_mode)
			if gender_mode == 'man':
				menNumber += 1
			if  gender_mode == 'woman':
				womenNumber += 1
			if menNumber > 10000:
				menNumber = 5000
			if womenNumber > 10000:
				womenNumber = 5000
		bgr_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)
		size = (int(400), int(400))  
		if inf3.empty() == False:
			pictureNum = inf3.get()
			runFlag = 1
			print (pictureNum)
		if runFlag == 1:
#			pictureNum += 1
#			pictureNum < 100:
			if womenNumber > menNumber:
				if pictureNum == 1:
					frame = cv2.imread('./woman/1.jpg')
				if pictureNum == 2:
					frame = cv2.imread('./woman/2.jpg')
				if pictureNum == 3:
					frame = cv2.imread('./woman/3.jpg')
				if pictureNum == 4:
					frame = cv2.imread('./woman/4.jpg')
				if pictureNum == 5:
					frame = cv2.imread('./woman/5.jpg')
				if pictureNum == 6:
					frame = cv2.imread('./woman/6.jpg')
			else:
				if pictureNum == 1:
					frame = cv2.imread('./man/1.jpg')
				if pictureNum == 2:
					frame = cv2.imread('./man/2.jpg')
				if pictureNum == 3:
					frame = cv2.imread('./man/3.jpg')
				if pictureNum == 4:
					frame = cv2.imread('./man/4.jpg')
				if pictureNum == 5:
					frame = cv2.imread('./man/5.jpg')
				if pictureNum == 6:
					frame = cv2.imread('./man/6.jpg')	
			#else:
			runFlag = 0
			pictureNum = 0
			womenNumber = 0
			menNumber = 0
			showHairStatus = 1
		if showHairStatus == 0:
			ret,frame = cap.read()  # get image
		if showHairStatus == 1:
			displayTimes += 1
			if displayTimes >= 150:
				displayTimes = 0
				showHairStatus = 0

		bgr_image = cv2.resize(bgr_image, size, interpolation=cv2.INTER_AREA) 

		#if frame.empty() == False:
		frame = cv2.resize(frame, size, interpolation=cv2.INTER_AREA) 
		htitch= np.hstack((bgr_image, frame))
		cv2.imshow('AI', htitch)
		#inf.put(htitch)
		if cv2.waitKey(1) & 0xFF == ord('q'):
			break

def robot(inf, inf2, inf3):                        
	resultNumber = 0 
	while 1:                                   #sleep 5ms
		time.sleep(0.01)
		CHUNK = 1024                       #block
###########################################################
#function：speech to text
###########################################################
		while inf2.get():
			time.sleep(0.01)
			with open('output.wav','rb') as audio_file:


				speechTxt = speech_to_text.recognize(
						audio=audio_file,
						content_type='audio/wav',
						timestamps=False,
						word_confidence=False)['results']
				if speechTxt == []:                              # voice to text is empty
					print ('Restart:')
					break
				res = json.dumps(
						speechTxt[0]['alternatives'][0]['transcript'],
						indent=2)

				print (res)


###########################################################
#function：chat robot
###########################################################
			input = {'text': res}
			response = conversation.message(workspace_id=workspace_id,
				                input=input)
			if response['output']['text'] == []:                   #solution no intention
				#print (response)
				print ('Restart:')
				break
			#print (response['intents'][0]['intent'])
			if (response['intents']) == []:                     #intention is null  
				print ('Restart:')				
				break
			if response['intents'][0]['intent'] == 'longHair':
				dict['longOrShort'] = 4
			if response['intents'][0]['intent'] == 'shortHair':
				dict['longOrShort'] = 1
			if response['intents'][0]['intent'] == 'noColorHair':
				dict['color'] = 0
				resultNumber = dict['color'] + dict['longOrShort']
				print (resultNumber)
				inf3.put(resultNumber)
			if response['intents'][0]['intent'] == 'yellowHair':
				dict['color'] = 1
				resultNumber = dict['color'] + dict['longOrShort']
				print (resultNumber)
				inf3.put(resultNumber)
			if response['intents'][0]['intent'] == 'redHair':
				dict['color'] = 2
				resultNumber = dict['color'] + dict['longOrShort']
				print (resultNumber)
				inf3.put(resultNumber)
				

#***********************string comparison**********************


###########################################################
#function：text to speech
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
			    #if data == '':
			    #	print (data)

			stream.stop_stream()
			stream.close()
			p.terminate()
			print ('Restart:')
			break                                               #break

	cv2.destroyAllWindows()


def speechQicuk(inf, inf2):                              #sleep 5ms
	global htitch
	while 1:

		time.sleep(0.01)
		CHUNK = 1024                        #block
		FORMAT = pyaudio.paInt16            #
		CHANNELS = 2                        #
		RATE = 44100                        #fps
		RECORD_SECONDS = 5                 
		WAVE_OUTPUT_FILENAME = "output.wav" #
		p = pyaudio.PyAudio()

		frames = []
		recordFlag = 0
		firstDisposeFlag = 0

		cv2.namedWindow('control',0)            
		cv2.resizeWindow('control', 200, 200);  
		speechImg = cv2.imread('1.jpg')
		while 1:

			time.sleep(0.005)
			if inf.empty() == False:
				firstDisposeFlag = 1
			#if firstDisposeFlag == 1:
			#	cv2.imshow('AI', inf.get())
			cv2.waitKey(1)
			cv2.imshow('control', speechImg)
			if cv2.waitKey(1) == 98:           #'b' stop recording
				if recordFlag == 2:
					recordFlag = 1
			if cv2.waitKey(1) == 97:           #'a' start recording
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
		inf2.put(1)

###########################################################
#fuction：Tony Plus
#explain：main thread
###########################################################
if __name__ == '__main__':
	dataQueue = Queue()
	dataQueue1 = Queue()
	dataQueue2 = Queue()
	p1 = multiprocessing.Process(target=face_gender,  args=(dataQueue,dataQueue2,))
	p1.start()
	p2 = multiprocessing.Process(target=robot,  args=(dataQueue,dataQueue1,dataQueue2,))
	p2.start()
	p3 = multiprocessing.Process(target=speechQicuk,  args=(dataQueue,dataQueue1,))
	p3.start()
	p1.join()
	p2.join()
	p3.join()

	#t1 = Thread(target=face_gender)
	#t2 = Thread(target=robot)
	#t3 = Thread(target=speechQicuk)

	#t1.start()
	#t2.start()
	#t3.start()


