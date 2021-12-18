# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# print(device)

def point_score(outputs, imgs):    
    loss_func_mse = nn.MSELoss(reduction='none')
    error = loss_func_mse((outputs[0]+1)/2,(imgs[0]+1)/2)
    normal = (1-torch.exp(-error))
    score = (torch.sum(normal*loss_func_mse((outputs[0]+1)/2,(imgs[0]+1)/2)) / torch.sum(normal)).item()

    return score

def score_sum(list1, list2, alpha):
    list_result = []
    for i in range(len(list1)):
        list_result.append((alpha*list1[i]+(1-alpha)*list2[i]))
        
    return list_result

def anomaly_score_list(psnr_list):
    anomaly_score_list = list()
    for i in range(len(psnr_list)):
        #print(i)
        anomaly_score_list.append(anomaly_score(psnr_list[i], np.max(psnr_list), np.min(psnr_list)))
        
    return anomaly_score_list

def anomaly_score_list_inv(psnr_list):
    anomaly_score_list = list()
    for i in range(len(psnr_list)):
        anomaly_score_list.append(anomaly_score_inv(psnr_list[i], np.max(psnr_list), np.min(psnr_list)))
        
    return anomaly_score_list

def psnr(mse):

    return 10 * math.log10(1 / mse)

def anomaly_score(psnr, max_psnr, min_psnr):

    return ((psnr - min_psnr) / (max_psnr-min_psnr))

def anomaly_score_inv(psnr, max_psnr, min_psnr):

    return (1.0 - ((psnr - min_psnr) / (max_psnr-min_psnr)))

import numpy as np
import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torch.nn.init as init
import torch.utils.data as data
import torch.utils.data.dataset as dataset
import torchvision.datasets as dset
import torchvision.transforms as transforms
from torch.autograd import Variable
import torchvision.utils as v_utils
import matplotlib.pyplot as plt
import cv2
import math
from collections import OrderedDict
import copy
import time
from model.utils import DataLoader
from model.final_future_prediction_with_memory_spatial_sumonly_weight_ranking_top1 import *
from model.Reconstruction import *
from sklearn.metrics import roc_auc_score
#from utils import *
import random
import glob
import argparse

# parser = argparse.ArgumentParser(description="MNAD")
# parser.add_argument('--video_path', nargs='+', type=str, help='gpus')

# args = parser.parse_args()
# args.video_path
#model = torch.load("/home/abnormal_detection/VuNgocTu/MNAD_w_mem_1/exp/VNAnomaly/pred/log/model.pth")
#model.eval()
##model.cuda()
#m_items = torch.load("/home/abnormal_detection/VuNgocTu/MNAD_w_mem_1/exp/VNAnomaly/pred/log/keys.pt")
#m_items_test = m_items.clone()
model = torch.load("/content/model.pth", map_location='cpu')
model.eval()
model.cuda()
m_items = torch.load("/content/keys.pt")
m_items_test = m_items.clone()
import psutil
print(psutil.virtual_memory().percent)
fourcc = cv2.VideoWriter_fourcc('X', 'V', 'I', 'D')
log_file=open('test.npy', 'wb') 
def demo(video_path):
  cap = cv2.VideoCapture(video_path)
  fourcc = cv2.VideoWriter_fourcc('X', 'V', 'I', 'D')

  video_writer = cv2.VideoWriter(f'temp_video.avi', fourcc, 30, (256,256))
  labels_list = []
  label_length = 0
  psnr_list = []
  feature_distance_list = []
  cnt = 0
  outputs=None
  
  loss_func_mse = nn.MSELoss(reduction='none')
  batch=[]
  pred_frame=None
  result_frames=[]
  while cap.isOpened():
    ret, frame = cap.read()
    if ret is False:
      break
    cnt=cnt+1
    frame = cv2.resize(frame, (256, 256))
    frame = frame.astype(dtype=np.float32)
    frame = (frame / 127.5) - 1.0
    frame=transforms.ToTensor()(frame)
    print(cnt)
    #prev_frame=None
    if cnt<5:
      batch.append(frame)
    elif cnt==5:
      batch.append(frame)
      batch=np.concatenate(batch, axis=0) 
      imgs=torch.from_numpy(batch)
      imgs=imgs.unsqueeze(0).cuda()
    elif cnt>5:
      # print(imgs.shape)
      imgs[:,0:12]=imgs[:,3:15].clone()
      imgs[:,12:15]=frame.unsqueeze(0).cuda()
      # print(imgs.shape)
    if cnt>=5:
      # print(torch.cuda.memory_allocated(0))
      if pred_frame is not None:
          print(pred_frame.shape)
          prev_frame=torch.from_numpy(pred_frame)
          prev_frame=prev_frame.unsqueeze(0).cuda()
          imgs[:,9:3*4]=prev_frame[:,0:3]*0.25+imgs[:,9:3*4]*0.75
      outputs, feas, updated_feas, m_items_test, _, _, _, _, _, compactness_loss = model.forward(imgs[:,0:3*4], m_items, False)
      mse_imgs = torch.mean(loss_func_mse((outputs[0]+1)/2, (imgs[0,3*4:]+1)/2)).item()
      mse_feas = compactness_loss.item()
      # print(psutil.virtual_memory().percent)
      #prev_frame=outputs.clone()
      pred_frame=outputs.cpu().detach().numpy()
      #np.save(log_file, pred_frame)
      predict_frame=((pred_frame[0] + 1) * 127.5).transpose(1, 2, 0).astype('uint8')
      video_writer.write(predict_frame)
      # Calculating the threshold for updating at the test time
      point_sc = point_score(outputs, imgs[:,3*4:])
      del outputs
      
      if  point_sc < 0.01:
        query = F.normalize(feas, dim=1)
        query = query.permute(0,2,3,1) #s b X h X w X d
        m_items_test = model.memory.update(query, m_items_test, False)
      #del prev_frame
      psnr_list.append(psnr(mse_imgs))
      feature_distance_list.append(mse_feas)
  cap.release()

  anomaly_score_total_list = []
  anomaly_score_total_list += score_sum(anomaly_score_list(psnr_list), 
                                  anomaly_score_list_inv(feature_distance_list), 0.6)
  anomaly_score_total_list = np.asarray(anomaly_score_total_list)
  cap = cv2.VideoCapture("temp_video.avi")
  cnt=0
  
  fourcc = cv2.VideoWriter_fourcc(* 'mp4v')
  video_writer = cv2.VideoWriter(f'result_video.mp4', fourcc, 30, (256,256))

  while cap.isOpened():
    ret, frame = cap.read()
    if ret is False:
      break
    if anomaly_score_total_list[cnt]>0.5:
      video_writer.write(frame)
    cnt=cnt+1
  cap.release()   
  print(cnt)
  
# demo("/content/Right_Robbery_97.mp4")

!cp -r /content/drive/MyDrive/Colab\ Notebooks/KLTN/Street_final/Normal /content

import os

path = '/content/Normal/Left'
file_lst = []

for video_name in os.listdir(path):
  print(video_name)
  demo(path + '/' + video_name)
  if os.path.isfile('/content/result_video.mp4'):
    file_lst.append(video_name)
  os.remove('/content/result_video.mp4')

import csv

file = open('Left.csv', 'w+', newline ='')

with file:
    write = csv.writer(file)
    write.writerows(file_lst)

"""# Web

"""

!pip install flask
!pip install flask_ngrok
!pip install pyngrok
!ngrok authtoken 222aH6229Hgl4E7yo4n20fLume0_2QrJeqjVpYW1ioswUgpKN

!cp -r /content/drive/MyDrive/Colab\ Notebooks/KLTN/Web/template /content
!cp -r /content/drive/MyDrive/Colab\ Notebooks/KLTN/Web/data /content

import os
import os.path
from flask import Flask, flash, request, redirect, url_for, render_template, Response
from werkzeug.utils import secure_filename
from flask_ngrok import run_with_ngrok

# UPLOAD_FOLDER = '/content/static/uploads/'
UPLOAD_FOLDER = '/content/data'

app = Flask(__name__, template_folder='/content/template')
app.secret_key = "secret key"
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
run_with_ngrok(app)

def gen_frames():
	vs = cv2.VideoCapture('/content/result_video.mp4')  
	while True:
		success, frame = vs.read()
		if not success:
			break
		else:
			cv2.waitKey(100)
			ret, buffer = cv2.imencode('.jpg', frame)
			frame = buffer.tobytes()
			yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
	vs.release()
	os.remove('/content/result_video.mp4')
	
@app.route('/')
def upload_form():
	return render_template('upload.html')

@app.route('/', methods=['POST'])
def run_model():
	if 'file' not in request.files:
		flash('No file part')
		return redirect(request.url)
	file = request.files['file']
	if file.filename == '':
		flash('No image selected for uploading')
		return redirect(request.url)
	else:
		flag = False
		filename = secure_filename(file.filename)
		file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
		# print(filename)
		flash('Video successfully uploaded')
		demo("/content/data/" + str(filename))
		if os.path.isfile('/content/result_video.mp4'):
			flag = True
		return render_template('upload.html', filename=filename, flag=flag)

@app.route('/video_feed')
def video_feed():
	return Response(gen_frames(), mimetype = "multipart/x-mixed-replace; boundary=frame")

if __name__ == "__main__":
    app.run()

video = cv2.VideoCapture('/content/result_video.mp4')

if (video.isOpened() == False):
	print("Loi mo file")
else:
	fps = video.get(5)
	print('fps: ', fps)

	frame_count = video.get(7)
	print('Frame count: ', frame_count)

while(video.isOpened()):
	ret, frame = video.read()
	if ret == True:
		cv2_imshow(frame)
		
		if key == ord('q'):
			break
	else:
		break

video.release()