from django.shortcuts import render

# Create your views here.
from django.http import HttpResponse
from django.template import loader
from django.core.files.storage import FileSystemStorage
from django.conf import settings
import os

import tensorflow as tf
from tensorflow.keras.models import load_model
import cv2
import numpy as np
from moviepy.editor import VideoFileClip



model_file = None
video_file = None    
def upload_model(request):
    global model_file
    #template = loader.get_template('fer/index.html')
    try:
        if request.method == 'POST' and request.FILES['myfile']:
            
            myfile = request.FILES['myfile']
            fs = FileSystemStorage()
            filename = fs.save(myfile.name, myfile)
            uploaded_file_url = fs.url(filename)
            
            model_file = uploaded_file_url
            
            return render(request, 'fer/upload.html', {'uploaded_file_url': uploaded_file_url})
    except:
        #print('Upload file first')
        if model_file:
            return render(request, 'fer/upload.html', {'uploaded_file_url': model_file})
        else:        
            return render(request, 'fer/upload.html', {'uploaded_file_url': 'Upload file first'})
    model_file = None
    return render(request, 'fer/upload.html')
    
def upload_video(request):
    global video_file
    #template = loader.get_template('fer/index.html')
    try:
        if request.method == 'POST' and request.FILES['myfile']:
            myfile = request.FILES['myfile']
            fs = FileSystemStorage()
            filename = fs.save(myfile.name, myfile)
            uploaded_file_url = fs.url(filename)
            
            video_file = uploaded_file_url
            return render(request, 'fer/play_video.html', {'uploaded_file_url': uploaded_file_url})
    except:
        #print('Upload file first')
        if video_file:
            return render(request, 'fer/play_video.html', {'uploaded_file_url': video_file})
        else:
            return render(request, 'fer/upload.html', {'uploaded_file_url': 'Upload file first'})
    video_file = None
    return render(request, 'fer/upload.html')
    '''
    context = {}
    return HttpResponse(template.render(context, request))    
    #return HttpResponse("Hello, world. You're at the polls index.")
    '''
def predict(request):
    '''
    #template = loader.get_template('fer/index.html')
    try:
        if request.method == 'POST' and request.FILES['myfile']:
            myfile = request.FILES['myfile']
            fs = FileSystemStorage()
            filename = fs.save(myfile.name, myfile)
            uploaded_file_url = fs.url(filename)
            return render(request, 'fer/index.html', {'uploaded_file_url': uploaded_file_url})
    except:
        #print('Upload file first')
        return render(request, 'fer/upload.html', {'uploaded_file_url': 'Upload file first'})
    '''
    '''
    if model: print(model)
    if video: print(video)
    return render(request, 'base.html')    
    '''
    global loaded_model, model_file, video_file

    print(model_file)
    print(settings.BASE_DIR)
    model_file_ = settings.BASE_DIR + '/' + model_file#os.path.join(settings.BASE_DIR, model_file)
    print('****', settings.BASE_DIR + '/' + model_file)
    loaded_model = load_model(model_file_)
    print(video_file)
    video_file_ = settings.BASE_DIR + '/' + video_file#os.path.join(settings.BASE_DIR, video_file)
    video_output = predict_video(video_file_)
    return render(request, 'fer/play_video.html', {'uploaded_file_url': video_output})


calssifiers =['haarcascade_frontalface_default.xml',
          'haarcascade_frontalface_alt2.xml',
          'haarcascade_frontalcatface_extended.xml',
          'haarcascade_frontalcatface.xml',
          'haarcascade_frontalface_alt_tree.xml',
          'haarcascade_frontalface_alt.xml']


def detectFace(img):
    for classifier in calssifiers:
        #face_cascade = cv2.CascadeClassifier('/fer/static/face_detectors/' + classifier)
        #face_cascade = cv2.CascadeClassifier(os.path.join(settings.BASE_DIR, 'fer',settings.STATIC_URL, 'face_detectors', classifier))
        face_cascade = cv2.CascadeClassifier(settings.BASE_DIR + '/fer/static/face_detectors/' + classifier)
        faces = face_cascade.detectMultiScale(img, 1.3, 1)
        if (len(faces) > 0):
            return faces
    faces = []
    return faces

def process_image_challenge(frame):
    global calcOpticalFlow, old_gray, count_frames, loaded_model
    count_frames += 1
    #if isinstance(frame, type(temp_image)) and not isinstance(frame, type(none_image)) :
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    #print(frame_gray.shape)
    faces = detectFace(frame_gray)
    if len(faces) == 0:
        calcOpticalFlow = 0
        print("Old frame due to no faces")
        return frame #return original frame of video

    (x,y,w,h)  = faces[0]
    frame_gray = frame_gray[y:y+h, x:x+w]
    roi_color = frame[y:y+h, x:x+w]

    if(frame_gray.shape[0] < 90 or roi_color.shape[0] < 90 ):
        print("Old frame due to small size")
        calcOpticalFlow = 0
        return frame #return original frame of video

    frame_gray = cv2.resize(frame_gray, (100,100))
    roi_color = cv2.resize(roi_color, (100,100))

    calcOpticalFlow += 1
    hsv = np.zeros_like(roi_color)
    hsv[...,1] = 255

    if (calcOpticalFlow > 1 ):
        flow = cv2.calcOpticalFlowFarneback(old_gray,frame_gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)            
        mag, ang   = cv2.cartToPolar(flow[...,0], flow[...,1])
        hsv[...,0] = ang*180/np.pi/2
        hsv[...,2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
        rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)

        flowImg = cv2.resize(rgb, (48,48))
        flowImg = flowImg.reshape(-1, 48,48,3)
        roi_color = cv2.cvtColor(roi_color, cv2.COLOR_BGR2RGB)
        spatialImg = cv2.resize(roi_color, (48,48))
        spatialImg = spatialImg.reshape(-1, 48,48,3)
        spatialImg = spatialImg *1 / 255.0
        flowImg = flowImg * 1 / 255.0 

        with tf.device('/cpu:0'):
            out = loaded_model.predict([spatialImg, flowImg])

        org = (50, 50)   
        # fontScale 
        fontScale = 1 
        font = cv2.FONT_HERSHEY_SIMPLEX      
        # Blue color in BGR 
        color = (255, 0, 0)   
        # Line thickness of 2 px 
        thickness = 2    
        # Using cv2.putText() method 
        #txt = 'Ground Truth: {}, Out: {}'.format(vidLabel, emotions[np.argmax(out[0])])
        emotions = ['Anger', 'Sadness', 'Surprise', 'Disgust', 'Fear', 'Happiness']
        txt = 'Prediction: {}'.format(emotions[np.argmax(out[0])])
        #print ('True: {}, Out: {}, Prob.: {}'.format(vidLabel, emotions[np.argmax(out[0])], out[0]))
        frame = cv2.putText(frame, txt, org, font,  
                          fontScale, color, thickness, cv2.LINE_AA) 

    old_gray = frame_gray.copy()
    return frame

def predict_video(video_file):
    temp_image = np.array([])
    none_image = None

    
    vid = video_file
    vidName = vid.split('/')[-1].split('.')[0]
    video_output = settings.MEDIA_URL + 'output.mp4'#os.path.join(settings.MEDIA_URL, 'output.mp4') #'/media/output.mp4'
    cap = cv2.VideoCapture(vid)
    frameCount = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))  
    ret, old_frame = cap.read()
    old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)
    old_gray = cv2.resize(old_gray, (100,100))

    clip = VideoFileClip(vid)
    global count_frames, calcOpticalFlow
    count_frames = 0
    calcOpticalFlow = -1
    challenge_clip = clip.fl_image(process_image_challenge)
    #%time challenge_clip.write_videofile(video_output, audio=False)
    challenge_clip.write_videofile(settings.BASE_DIR + '/' + video_output, audio=False)
    return video_output
    
    
