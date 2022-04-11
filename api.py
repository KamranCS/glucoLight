import warnings
warnings.filterwarnings("ignore", message=r"Passing", category=FutureWarning)
import warnings
from flask import Flask,jsonify,request
warnings.filterwarnings('ignore')
from flask import Flask, render_template, request, session
from flask import flash, render_template
from flask import Flask
from flask import Flask, flash, request, redirect, url_for, render_template
from flask_session import Session
from flaskext.mysql import MySQL
from werkzeug.utils import secure_filename
import re
import os.path
import os
import matplotlib.pyplot as plt
from PIL import Image
from flask import Flask, render_template, request
from werkzeug.utils import secure_filename
from flask import Flask, redirect, url_for, request, render_template, Response, jsonify, redirect
from tensorflow.keras.layers import Layer, Lambda
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import BatchNormalization
import os
from tensorflow.keras.optimizers import Adam
from keras import optimizers
import warnings
warnings.filterwarnings('ignore')
from keras.models import Sequential, Model
from keras.layers import Activation, Dropout, Flatten, Dense, Input
from keras.layers import Dense, Conv2D, MaxPool2D, Flatten, LeakyReLU, BatchNormalization
import dlib
import base64
from imutils import face_utils
app = Flask(__name__)
mysql = MySQL()
app.config['MYSQL_DATABASE_USER'] = 'root'
app.config['MYSQL_DATABASE_PASSWORD'] = ''
app.config['MYSQL_DATABASE_DB'] = 'gaze'
app.config['MYSQL_DATABASE_HOST'] = 'localhost'
app.config['JSONIFY_PRETTYPRINT_REGULAR'] = True
app.secret_key = "elp"
app.config["SESSION_PERMANENT"] = False
app.config["SESSION_TYPE"] = "filesystem"
Session(app)
mysql.init_app(app)
UPLOAD_FOLDER='C:\\xampp\htdocs\gaze/uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


def create_convolution_layers1(input_img):
   model = Conv2D(32, (3, 3), activation='relu')(input_img)
   model = MaxPool2D(2, 2)(model)
   model = BatchNormalization()(model)

   model = Conv2D(64, (3, 3), activation='relu')(model)

   model = MaxPool2D(2, 2)(model)
   model = BatchNormalization()(model)

   model = Conv2D(128, (3, 3), activation='relu')(model)
   model = MaxPool2D(2, 2)(model)
   model = BatchNormalization()(model)

   model = Conv2D(256, (3, 3), activation='relu')(model)
   model = MaxPool2D(2, 2)(model)
   model = BatchNormalization()(model)

   return model


def final(input_img1,input_img_1):
   d5 = Flatten()(input_img1)
   d2 = Dense(512, activation='relu')(d5)
   d2 = Dense(2)(d2)
   d3 = Activation('linear')(d2)
   model = Model(inputs=[input_img_1], outputs=[d3])
   return model

def predict():
   files2 = os.listdir('C:\\xampp\htdocs\gaze/uploads/')
   files=[]
   for f in files2:
       if session["id"] in f:
           files.append(f)
   counter=1
   for f in files:
       if (counter==4):
           left_x = []
           left_y = []
           right_x = []
           right_y = []
           mi = 1
           ia = 0
           count = 0
           import cv2
           import numpy as np
           import time
           cap = cv2.VideoCapture('C:\\xampp\htdocs\gaze/uploads/' + f)
           i = 0
           start_time = time.time()
           while (cap.isOpened()):
               ret, frame = cap.read()
               if ret == False:
                   break
               tyt = str(session["id"]) +"_frames" + str(counter)
               if not os.path.exists(tyt):
                   os.mkdir(tyt)
               cv2.imwrite('C:\\xampp\htdocs\gaze/' + str(tyt) + "/" + str("%04d" % i) + '.jpg', frame)
               i += 1

           cap.release()
           cv2.destroyAllWindows()
           input_img_1 = Input(shape=(90, 90, 3))
           model_1 = create_convolution_layers1(input_img_1)
           model_3 = final(model_1, input_img_1)
           model_3.compile(optimizers.adam(lr=0.0001), loss='mae', metrics=['accuracy'])
           model_3.load_weights('C:\\xampp\htdocs\gaze\Models/MyGaze-5 (1).h5')
           detector = dlib.get_frontal_face_detector()
           predictor = dlib.shape_predictor(
               'C:\\xampp\htdocs\gaze\Models/shape_predictor_68_face_landmarks.dat')
           files1 = os.listdir('C:\\xampp\htdocs\gaze/' + str(tyt) + "/")
           for f in files1:
               img = cv2.imread('C:\\xampp\htdocs\gaze/' + str(tyt) + "/" + f)
               image = img
               gray = image
               rects = detector(gray, 1)
               for (i, rect) in enumerate(rects):
                   shape = predictor(gray, rect)
                   shape = face_utils.shape_to_np(shape)
                   for (name, (i, j)) in face_utils.FACIAL_LANDMARKS_IDXS.items():
                       p = 1
                       if name == 'right_eye':
                           image = image.copy()
                           for (x, y) in shape[i:j]:
                               (x, y, w, h) = cv2.boundingRect(np.array([shape[i:j]]))
                               roi = image[y - 20:y + h + 28, x - 10:x + w + 10]
                               roi = cv2.resize(roi, (90, 90))
                               pred = model_3.predict(roi.reshape((1, 90, 90, 3)))
                               a = pred[0][0]
                               b = pred[0][1]
                               a = round(a, 1)
                               b = round(b, 1)
               mi = mi + 1
               left_x.append(a)
               left_y.append(b)
               count = count + 1
               ia = ia + 1
           y = []
           y.append(4)
           for i in range(4, 17, 2):
               y.append(i)
           y.append(17)
           x = [11 for i in range(len(y))]
           x_prev = [9 for i in range(len(y))]
           x_next = [13 for i in range(len(y))]
           f = []
           f.append(11)
           for i in range(11, 24, 2):
               f.append(i)
           f.append(23)

           y_center = [10 for i in range(len(f))]
           y_prev = [8 for i in range(len(f))]
           y_next = [12 for i in range(len(f))]
           g = []
           g.append(4)
           for i in range(4, 17, 2):
               g.append(i)
           g.append(17)
           p = [23 for i in range(len(g))]
           p_prev = [21 for i in range(len(g))]
           p_next = [25 for i in range(len(g))]

           plt.plot(x, y, color="yellow")
           plt.plot(x_prev, y, color="red")
           plt.plot(x_next, y, color="darkgreen")

           plt.plot(f, y_center, color="yellow")
           plt.plot(f, y_prev, color="red")
           plt.plot(f, y_next, color="darkgreen")

           plt.plot(p, g, color="yellow")
           plt.plot(p_prev, g, color="red")
           plt.plot(p_next, g, color="darkgreen")
           minimum1 = int(min(left_x))
           maximum1 = int(max(left_x))
           maximum2 = int(max(left_y))
           minimum2 = int(min(left_y))
           predictedx = []
           predictedx = [minimum1, maximum1]
           predictedyabove = [maximum2 for i in range(len(predictedx))]
           predictedybelow = [minimum2 for i in range(len(predictedx))]
           plt.scatter(left_x, left_y, color="blue")
           predictedyabove = [maximum2 for i in range(len(predictedx))]
           predictedybelow = [minimum2 for i in range(len(predictedx))]
           plt.plot(predictedx, predictedyabove, color='darkorange')
           plt.plot(predictedx, predictedybelow, color='darkorange')
           plt.fill_between(predictedx, predictedyabove, predictedybelow, color='#539ecd', alpha=0.5)
           y = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]
           plt.yticks(y, fontsize=8)
           x = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27,
                28,
                29,
                30,
                31, 32, 33, 34, 35, 36]
           plt.xticks(x, fontsize=8)
           plt.grid()

           filename20 = str(session["id"]) +"_lefteye" + str(counter) + '.png'
           plt.savefig('C:\\xampp\htdocs\gaze/static/uploads/' + filename20)
           plt.close()
           import pandas as pd
           list_dict = {'x': left_x, 'y': left_y}
           df = pd.DataFrame(list_dict)
           csv_file1='C:\\xampp\htdocs\gaze/static/uploads/' + str(session["id"]) +"_"+ str(counter) + '_left.csv'
           df.to_csv('C:\\xampp\htdocs\gaze/static/uploads/' + str(session["id"]) +"_"+ str(counter) + '_left.csv',
                     index=False,
                     header=False)
           model_3.load_weights('C:\\xampp\htdocs\gaze\Models/MyGaze-6.h5')
           detector = dlib.get_frontal_face_detector()
           predictor = dlib.shape_predictor(
               'C:\\xampp\htdocs\gaze\Models/shape_predictor_68_face_landmarks.dat')
           files1 = os.listdir('C:\\xampp\htdocs\gaze/' + str(tyt) + "/")
           for f in files1:
               img = cv2.imread('C:\\xampp\htdocs\gaze/' + str(tyt) + "/" + f)
               image = img
               gray = image
               rects = detector(gray, 1)
               for (i, rect) in enumerate(rects):
                   shape = predictor(gray, rect)
                   shape = face_utils.shape_to_np(shape)
                   for (name, (i, j)) in face_utils.FACIAL_LANDMARKS_IDXS.items():
                       p = 1
                       if name == 'left_eye':
                           image = image.copy()
                           for (x, y) in shape[i:j]:
                               (x, y, w, h) = cv2.boundingRect(np.array([shape[i:j]]))
                               roi = image[y - 20:y + h + 28, x - 10:x + w + 10]
                               roi = cv2.resize(roi, (90, 90))
                               pred = model_3.predict(roi.reshape((1, 90, 90, 3)))
                               a = pred[0][0]
                               b = pred[0][1]
                               a = round(a, 1)
                               b = round(b, 1)
               mi = mi + 1
               right_x.append(a)
               right_y.append(b)
               count = count + 1
               ia = ia + 1
           y = []
           y.append(4)
           for i in range(4, 17, 2):
               y.append(i)
           y.append(17)
           x = [11 for i in range(len(y))]
           x_prev = [9 for i in range(len(y))]
           x_next = [13 for i in range(len(y))]
           f = []
           f.append(11)
           for i in range(11, 24, 2):
               f.append(i)
           f.append(23)
           y_center = [10 for i in range(len(f))]
           y_prev = [8 for i in range(len(f))]
           y_next = [12 for i in range(len(f))]
           g = []
           g.append(4)
           for i in range(4, 17, 2):
               g.append(i)
           g.append(17)
           p = [23 for i in range(len(g))]
           p_prev = [21 for i in range(len(g))]
           p_next = [25 for i in range(len(g))]

           plt.plot(x, y, color="yellow")
           plt.plot(x_prev, y, color="red")
           plt.plot(x_next, y, color="darkgreen")

           plt.plot(f, y_center, color="yellow")
           plt.plot(f, y_prev, color="red")
           plt.plot(f, y_next, color="darkgreen")

           plt.plot(p, g, color="yellow")
           plt.plot(p_prev, g, color="red")
           plt.plot(p_next, g, color="darkgreen")
           minimum1 = int(min(right_x))
           maximum1 = int(max(right_x))
           maximum2 = int(max(right_y))
           minimum2 = int(min(right_y))
           predictedx = []
           predictedx = [minimum1, maximum1]
           predictedyabove = [maximum2 for i in range(len(predictedx))]
           predictedybelow = [minimum2 for i in range(len(predictedx))]
           plt.scatter(right_x, right_y, color="blue")
           predictedyabove = [maximum2 for i in range(len(predictedx))]
           predictedybelow = [minimum2 for i in range(len(predictedx))]
           plt.plot(predictedx, predictedyabove, color='darkorange')
           plt.plot(predictedx, predictedybelow, color='darkorange')
           plt.fill_between(predictedx, predictedyabove, predictedybelow, color='#539ecd', alpha=0.5)
           y = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]
           plt.yticks(y, fontsize=8)
           x = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27,
                28,
                29,
                30,
                31, 32, 33, 34, 35, 36]
           plt.xticks(x, fontsize=8)
           plt.grid()
           filename21 = str(session["id"]) +"_righteye" + str(counter) + '.png'
           plt.savefig('C:\\xampp\htdocs\gaze/static/uploads/' + filename21)
           plt.close()
           import pandas as pd
           list_dict = {'x': right_x, 'y': right_y}
           df = pd.DataFrame(list_dict)
           csv_file2='C:\\xampp\htdocs\gaze/static/uploads/' + str(session["id"]) +"_"+ str(counter) + '_right.csv'
           df.to_csv('C:\\xampp\htdocs\gaze/static/uploads/' + str(session["id"]) +"_"+ str(counter) + '_right.csv',
                     index=False,
                     header=False)
       if (counter==5):
           left_x = []
           left_y = []
           right_x = []
           right_y = []
           mi = 1
           ia = 0
           count = 0
           import cv2
           import numpy as np
           import time
           cap = cv2.VideoCapture('C:\\xampp\htdocs\gaze/uploads/' + f)
           i = 0
           start_time = time.time()
           while (cap.isOpened()):
               ret, frame = cap.read()
               if ret == False:
                   break
               tyt = str(session["id"]) +"_frames" + str(counter)
               if not os.path.exists(tyt):
                   os.mkdir(tyt)
               cv2.imwrite('C:\\xampp\htdocs\gaze/' + str(tyt) + "/" + str("%04d" % i) + '.jpg', frame)
               i += 1

           cap.release()
           cv2.destroyAllWindows()
           input_img_1 = Input(shape=(90, 90, 3))
           model_1 = create_convolution_layers1(input_img_1)
           model_3 = final(model_1, input_img_1)
           model_3.compile(optimizers.adam(lr=0.0001), loss='mae', metrics=['accuracy'])
           model_3.load_weights('C:\\xampp\htdocs\gaze\Models/MyGaze-5 (1).h5')
           detector = dlib.get_frontal_face_detector()
           predictor = dlib.shape_predictor(
               'C:\\xampp\htdocs\gaze\Models/shape_predictor_68_face_landmarks.dat')
           files1 = os.listdir('C:\\xampp\htdocs\gaze/' + str(tyt) + "/")
           for f in files1:
               img = cv2.imread('C:\\xampp\htdocs\gaze/' + str(tyt) + "/" + f)
               image = img
               gray = image
               rects = detector(gray, 1)
               for (i, rect) in enumerate(rects):
                   shape = predictor(gray, rect)
                   shape = face_utils.shape_to_np(shape)
                   for (name, (i, j)) in face_utils.FACIAL_LANDMARKS_IDXS.items():
                       p = 1
                       if name == 'right_eye':
                           image = image.copy()
                           for (x, y) in shape[i:j]:
                               (x, y, w, h) = cv2.boundingRect(np.array([shape[i:j]]))
                               roi = image[y - 20:y + h + 28, x - 10:x + w + 10]
                               roi = cv2.resize(roi, (90, 90))
                               pred = model_3.predict(roi.reshape((1, 90, 90, 3)))
                               a = pred[0][0]
                               b = pred[0][1]
                               a = round(a, 1)
                               b = round(b, 1)
               mi = mi + 1
               left_x.append(a)
               left_y.append(b)
               count = count + 1
               ia = ia + 1
           x = []
           x = [16, 18, 20]
           minimum1 = int(min(left_x))
           maximum1 = int(max(left_x))
           maximum2 = int(max(left_y))
           minimum2 = int(min(left_y))
           predictedx = [minimum1, maximum1]
           y = [10 for i in range(len(x))]
           y1 = [12 for i in range(len(x))]
           y2 = [8 for i in range(len(x))]
           predictedyabove = [maximum2 for i in range(len(predictedx))]
           predictedybelow = [minimum2 for i in range(len(predictedx))]
           plt.plot(x, y, color="yellow")
           plt.plot(x, y1, color="red")
           plt.plot(x, y2, color="darkgreen")
           plt.scatter(left_x, left_y, color="red")
           plt.plot(predictedx, predictedyabove, color='darkorange')
           plt.plot(predictedx, predictedybelow, color='darkorange')
           plt.fill_between(x, y1, y2, color='k', alpha=.2)
           plt.fill_between(predictedx, predictedyabove, predictedybelow, color='#539ecd', alpha=0.5)
           plt.title('Screen size 36cm*20cm (W*H)', fontsize=10)
           y = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]
           plt.yticks(y, fontsize=8)
           x = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27,
                28, 29, 30, 31, 32, 33, 34, 35, 36]
           plt.xticks(x, fontsize=8)
           plt.grid()
           filename16 = str(session["id"]) +"_lefteye" + str(counter) + '.png'
           plt.savefig('C:\\xampp\htdocs\gaze/static/uploads/' + filename16)
           plt.close()
           import pandas as pd
           list_dict = {'x': left_x, 'y': left_y}
           df = pd.DataFrame(list_dict)
           csv_file3='C:\\xampp\htdocs\gaze/static/uploads/' + str(session["id"]) + "_"+ str(counter) + '_left.csv'
           df.to_csv('C:\\xampp\htdocs\gaze/static/uploads/' + str(session["id"]) + "_"+ str(counter) + '_left.csv', index=False,
                     header=False)
           model_3.load_weights('C:\\xampp\htdocs\gaze\Models/MyGaze-6.h5')
           detector = dlib.get_frontal_face_detector()
           predictor = dlib.shape_predictor(
               'C:\\xampp\htdocs\gaze\Models/shape_predictor_68_face_landmarks.dat')
           files1 = os.listdir('C:\\xampp\htdocs\gaze/' + str(tyt) + "/")
           for f in files1:
               img = cv2.imread('C:\\xampp\htdocs\gaze/' + str(tyt) + "/" + f)
               image = img
               gray = image
               rects = detector(gray, 1)
               for (i, rect) in enumerate(rects):
                   shape = predictor(gray, rect)
                   shape = face_utils.shape_to_np(shape)
                   for (name, (i, j)) in face_utils.FACIAL_LANDMARKS_IDXS.items():
                       p = 1
                       if name == 'left_eye':
                           image = image.copy()
                           for (x, y) in shape[i:j]:
                               (x, y, w, h) = cv2.boundingRect(np.array([shape[i:j]]))
                               roi = image[y - 20:y + h + 28, x - 10:x + w + 10]
                               roi = cv2.resize(roi, (90, 90))
                               pred = model_3.predict(roi.reshape((1, 90, 90, 3)))
                               a = pred[0][0]
                               b = pred[0][1]
                               a = round(a, 1)
                               b = round(b, 1)
               mi = mi + 1
               right_x.append(a)
               right_y.append(b)
               count = count + 1
               ia = ia + 1
           x = []
           x = [16, 18, 20]
           minimum1 = int(min(right_x))
           maximum1 = int(max(right_x))
           maximum2 = int(max(right_y))
           minimum2 = int(min(right_y))
           predictedx = [minimum1, maximum1]
           y = [10 for i in range(len(x))]
           y1 = [12 for i in range(len(x))]
           y2 = [8 for i in range(len(x))]
           predictedyabove = [maximum2 for i in range(len(predictedx))]
           predictedybelow = [minimum2 for i in range(len(predictedx))]
           plt.plot(x, y, color="yellow")
           plt.plot(x, y1, color="red")
           plt.plot(x, y2, color="darkgreen")
           plt.scatter(right_x, right_y, color="red")
           plt.plot(predictedx, predictedyabove, color='darkorange')
           plt.plot(predictedx, predictedybelow, color='darkorange')
           plt.fill_between(x, y1, y2, color='k', alpha=.2)
           plt.fill_between(predictedx, predictedyabove, predictedybelow, color='#539ecd', alpha=0.5)
           plt.title('Screen size 36cm*20cm (W*H)', fontsize=10)
           y = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]
           plt.yticks(y, fontsize=8)
           x = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27,
                28, 29, 30, 31, 32, 33, 34, 35, 36]
           plt.xticks(x, fontsize=8)
           plt.grid()
           filename17 = str(session["id"]) +"_righteye" + str(counter) + '.png'
           plt.savefig('C:\\xampp\htdocs\gaze/static/uploads/' + filename17)
           plt.close()
           import pandas as pd
           list_dict = {'x': right_x, 'y': right_y}
           df = pd.DataFrame(list_dict)
           csv_file4='C:\\xampp\htdocs\gaze/static/uploads/' + str(session["id"]) +"_"+ str(counter) + '_right.csv'
           df.to_csv('C:\\xampp\htdocs\gaze/static/uploads/' + str(session["id"]) +"_"+ str(counter) + '_right.csv', index=False,
                     header=False)
       if (counter==3):

           left_x = []
           left_y = []
           right_x = []
           right_y = []
           mi = 1
           ia = 0
           count = 0
           import cv2
           import numpy as np
           import time
           cap = cv2.VideoCapture('C:\\xampp\htdocs\gaze/uploads/' + f)
           i = 0
           start_time = time.time()
           while (cap.isOpened()):
               ret, frame = cap.read()
               if ret == False:
                   break
               tyt = str(session["id"]) +"_frames" + str(counter)
               if not os.path.exists(tyt):
                   os.mkdir(tyt)
               cv2.imwrite('C:\\xampp\htdocs\gaze/' + str(tyt) + "/" + str("%04d" % i) + '.jpg', frame)
               i += 1

           cap.release()
           cv2.destroyAllWindows()
           input_img_1 = Input(shape=(90, 90, 3))
           model_1 = create_convolution_layers1(input_img_1)
           model_3 = final(model_1, input_img_1)
           model_3.compile(optimizers.adam(lr=0.0001), loss='mae', metrics=['accuracy'])
           model_3.load_weights('C:\\xampp\htdocs\gaze\Models/MyGaze-5 (1).h5')
           detector = dlib.get_frontal_face_detector()
           predictor = dlib.shape_predictor(
               'C:\\xampp\htdocs\gaze\Models/shape_predictor_68_face_landmarks.dat')
           files1 = os.listdir('C:\\xampp\htdocs\gaze/' + str(tyt) + "/")
           len1=len(files1)
           len2=len1/2
           len2=int(len2)
           files3=files1[0:len2]
           for f in files3:
               img = cv2.imread('C:\\xampp\htdocs\gaze/' + str(tyt) + "/" + f)
               image = img
               gray = image
               rects = detector(gray, 1)
               for (i, rect) in enumerate(rects):
                   shape = predictor(gray, rect)
                   shape = face_utils.shape_to_np(shape)
                   for (name, (i, j)) in face_utils.FACIAL_LANDMARKS_IDXS.items():
                       p = 1
                       if name == 'right_eye':
                           image = image.copy()
                           for (x, y) in shape[i:j]:
                               (x, y, w, h) = cv2.boundingRect(np.array([shape[i:j]]))
                               roi = image[y - 20:y + h + 28, x - 10:x + w + 10]
                               roi = cv2.resize(roi, (90, 90))
                               pred = model_3.predict(roi.reshape((1, 90, 90, 3)))
                               a = pred[0][0]
                               b = pred[0][1]
                               a = round(a, 1)
                               b = round(b, 1)
               mi = mi + 1
               left_x.append(a)
               left_y.append(b)
               count = count + 1
               ia = ia + 1
           x = [3, 34]
           y = [3, 17]
           a = []
           for i in x:
               a.append(i - 2)
           y1 = []
           for i in y:
               y1.append(i - 2)
           b = []
           for i in x:
               b.append(i + 2)
           y2 = []
           for i in y:
               y2.append(i + 2)
           plt.plot(x, y, color="yellow")
           plt.plot(a, y, color="red")
           plt.plot(b, y, color="darkgreen")
           minimum1 = (min(left_x))
           maximum1 = (max(left_x))
           maximum2 = (max(left_y))
           minimum2 = (min(left_y))
           plt.scatter(left_x, left_y, color="red")
           index1 = left_x.index(min(left_x))
           index2 = left_x.index(max(left_x))
           index3 = left_y.index(min(left_y))
           index4 = left_y.index(max(left_y))
           predictedx = []
           predictedy = []
           predictedy1 = []
           predictedx1 = []
           predictedx.append(minimum1)
           predictedx.append(maximum1)
           predictedy.append(left_y[index1])
           predictedy.append(left_y[index2])
           predictedy1.append(minimum2)
           predictedy1.append(maximum2)
           plt.plot(predictedx, predictedy, color='darkorange')
           plt.plot(predictedx, predictedy1, color='darkorange')
           plt.fill_between(predictedx, predictedy, predictedy1, color='#539ecd')
           plt.title('Screen size 36cm*20cm (W*H)', fontsize=10)
           y = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]
           plt.yticks(y, fontsize=8)
           x = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27,
                28, 29, 30, 31, 32, 33, 34, 35, 36]
           plt.xticks(x, fontsize=8)
           plt.grid()
           filename14 = str(session["id"]) +"_Part_1_lefteye" + str(counter) + '.png'
           plt.savefig('C:\\xampp\htdocs\gaze/static/uploads/' + filename14)
           plt.close()
           import pandas as pd
           list_dict = {'x': left_x, 'y': left_y}
           df = pd.DataFrame(list_dict)
           csv_file5='C:\\xampp\htdocs\gaze/static/uploads/' + str(session["id"]) +"_Part_1"+ str(counter) + '_left.csv'
           df.to_csv('C:\\xampp\htdocs\gaze/static/uploads/' + str(session["id"]) +"_Part_1"+ str(counter) + '_left.csv', index=False,
                     header=False)
           model_3.load_weights('C:\\xampp\htdocs\gaze\Models/MyGaze-6.h5')
           detector = dlib.get_frontal_face_detector()
           predictor = dlib.shape_predictor(
               'C:\\xampp\htdocs\gaze\Models/shape_predictor_68_face_landmarks.dat')
           files1 = os.listdir('C:\\xampp\htdocs\gaze/' + str(tyt) + "/")
           len1 = len(files1)
           len2 = len1 / 2
           len2= int(len2)
           files3 = files1[0:len2]
           for f in files3:
               img = cv2.imread('C:\\xampp\htdocs\gaze/' + str(tyt) + "/" + f)
               image = img
               gray = image
               rects = detector(gray, 1)
               for (i, rect) in enumerate(rects):
                   shape = predictor(gray, rect)
                   shape = face_utils.shape_to_np(shape)
                   for (name, (i, j)) in face_utils.FACIAL_LANDMARKS_IDXS.items():
                       p = 1
                       if name == 'left_eye':
                           image = image.copy()
                           for (x, y) in shape[i:j]:
                               (x, y, w, h) = cv2.boundingRect(np.array([shape[i:j]]))
                               roi = image[y - 20:y + h + 28, x - 10:x + w + 10]
                               roi = cv2.resize(roi, (90, 90))
                               pred = model_3.predict(roi.reshape((1, 90, 90, 3)))
                               a = pred[0][0]
                               b = pred[0][1]
                               a = round(a, 1)
                               b = round(b, 1)
               mi = mi + 1
               right_x.append(a)
               right_y.append(b)
               count = count + 1
               ia = ia + 1
           x = [3, 34]
           y = [3, 17]
           a = []
           for i in x:
               a.append(i - 2)
           y1 = []
           for i in y:
               y1.append(i - 2)
           b = []
           for i in x:
               b.append(i + 2)
           y2 = []
           for i in y:
               y2.append(i + 2)
           plt.plot(x, y, color="yellow")
           plt.plot(a, y, color="red")
           plt.plot(b, y, color="darkgreen")
           minimum1 = (min(right_x))
           maximum1 = (max(right_x))
           maximum2 = (max(right_y))
           minimum2 = (min(right_y))
           plt.scatter(right_x, right_y, color="red")
           index1 = right_x.index(min(right_x))
           index2 = right_x.index(max(right_x))
           index3 = right_y.index(min(right_y))
           index4 = right_y.index(max(right_y))
           predictedx = []
           predictedy = []
           predictedy1 = []
           predictedx1 = []
           predictedx.append(minimum1)
           predictedx.append(maximum1)
           predictedy.append(right_y[index1])
           predictedy.append(right_y[index2])
           predictedy1.append(minimum2)
           predictedy1.append(maximum2)
           plt.plot(predictedx, predictedy, color='darkorange')
           plt.plot(predictedx, predictedy1, color='darkorange')
           plt.fill_between(predictedx, predictedy, predictedy1, color='#539ecd')
           plt.title('Screen size 36cm*20cm (W*H)', fontsize=10)
           y = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]
           plt.yticks(y, fontsize=8)
           x = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27,
                28, 29, 30, 31, 32, 33, 34, 35, 36]
           plt.xticks(x, fontsize=8)
           plt.grid()
           filename15 = str(session["id"]) +"_Part_1_righteye" + str(counter) + '.png'
           plt.savefig('C:\\xampp\htdocs\gaze/static/uploads/' + filename15)
           plt.close()
           import pandas as pd
           list_dict = {'x': right_x, 'y': right_y}
           df = pd.DataFrame(list_dict)
           csv_file6='C:\\xampp\htdocs\gaze/static/uploads/' + str(session["id"]) +"_Part-1_"+ str(counter) + '_right.csv'
           df.to_csv('C:\\xampp\htdocs\gaze/static/uploads/' + str(session["id"]) +"_Part-1_"+ str(counter) + '_right.csv', index=False,
                     header=False)
       if (counter==3):
           left_x = []
           left_y = []
           right_x = []
           right_y = []
           mi = 1
           ia = 0
           count = 0
           import cv2
           import numpy as np
           import time
           cap = cv2.VideoCapture('C:\\xampp\htdocs\gaze/uploads/' +f)
           i = 0
           start_time = time.time()
           while (cap.isOpened()):
               ret, frame = cap.read()
               if ret == False:
                   break
               tyt = str(session["id"]) +"frames" + str(counter)
               if not os.path.exists(tyt):
                   os.mkdir(tyt)
               cv2.imwrite('C:\\xampp\htdocs\gaze/' + str(tyt) + "/" + str("%04d" % i) + '.jpg', frame)
               i += 1

           cap.release()
           cv2.destroyAllWindows()
           input_img_1 = Input(shape=(90, 90, 3))
           model_1 = create_convolution_layers1(input_img_1)
           model_3 = final(model_1, input_img_1)
           model_3.compile(optimizers.adam(lr=0.0001), loss='mae', metrics=['accuracy'])
           model_3.load_weights('C:\\xampp\htdocs\gaze\Models/MyGaze-5 (1).h5')
           detector = dlib.get_frontal_face_detector()
           predictor = dlib.shape_predictor(
               'C:\\xampp\htdocs\gaze\Models/shape_predictor_68_face_landmarks.dat')
           files1 = os.listdir('C:\\xampp\htdocs\gaze/' + str(tyt) + "/")
           len1 = len(files1)
           len2 = len1 / 2
           len2 = int(len2)
           files3 = files1[len2:len1]
           for f in files3:
               img = cv2.imread('C:\\xampp\htdocs\gaze/' + str(tyt) + "/" + f)
               image = img
               gray = image
               rects = detector(gray, 1)
               for (i, rect) in enumerate(rects):
                   shape = predictor(gray, rect)
                   shape = face_utils.shape_to_np(shape)
                   for (name, (i, j)) in face_utils.FACIAL_LANDMARKS_IDXS.items():
                       p = 1
                       if name == 'right_eye':
                           image = image.copy()
                           for (x, y) in shape[i:j]:
                               (x, y, w, h) = cv2.boundingRect(np.array([shape[i:j]]))
                               roi = image[y - 20:y + h + 28, x - 10:x + w + 10]
                               roi = cv2.resize(roi, (90, 90))
                               pred = model_3.predict(roi.reshape((1, 90, 90, 3)))
                               a = pred[0][0]
                               b = pred[0][1]
                               a = round(a, 1)
                               b = round(b, 1)
               mi = mi + 1
               left_x.append(a)
               left_y.append(b)
               count = count + 1
               ia = ia + 1

           x = [34, 3]
           y = [3, 17]
           a = []

           for i in x:
               a.append(i - 2)

           y1 = []

           for i in y:
               y1.append(i - 2)

           b = []

           for i in x:
               b.append(i + 2)

           y2 = []

           for i in y:
               y2.append(i + 2)
           plt.plot(x, y, color="yellow")
           plt.plot(a, y, color="red")
           plt.plot(b, y, color="darkgreen")
           minimum1 = (min(left_x))
           maximum1 = (max(left_x))
           maximum2 = (max(left_y))
           minimum2 = (min(left_y))
           plt.scatter(left_x, left_y, color="red")
           index1 = left_x.index(min(left_x))
           index2 = left_x.index(max(left_x))
           index3 = left_y.index(min(left_y))
           index4 = left_y.index(max(left_y))
           predictedx = []
           predictedy = []
           predictedy1 = []
           predictedx1 = []
           predictedx.append(minimum1)
           predictedx.append(maximum1)
           predictedy.append(left_y[index1])
           predictedy.append(left_y[index2])
           predictedx.reverse()
           predictedy.reverse()
           predictedy1.append(minimum2)
           predictedy1.append(maximum2)
           plt.plot(predictedx, predictedy, color='darkorange')
           plt.plot(predictedx, predictedy1, color='darkorange')
           plt.fill_between(predictedx, predictedy, predictedy1, color='#539ecd')
           plt.title('Screen size 36cm*20cm (W*H)', fontsize=10)
           y = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]
           y.reverse()
           plt.yticks(y, fontsize=8)
           x = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27,
                28, 29, 30, 31, 32, 33, 34, 35, 36]
           x.reverse()
           plt.xticks(x, fontsize=8)
           plt.grid()
           filename12 = str(session["id"]) +"_Part_2_lefteye" + str(counter) + '.png'
           plt.savefig('C:\\xampp\htdocs\gaze/static/uploads/' + filename12)
           plt.close()
           import pandas as pd
           list_dict = {'x': left_x, 'y': left_y}
           df = pd.DataFrame(list_dict)
           csv_file7='C:\\xampp\htdocs\gaze/static/uploads/' + str(session["id"]) + "_Part_2_"+str(counter) + '_left.csv'
           df.to_csv('C:\\xampp\htdocs\gaze/static/uploads/' + str(session["id"]) + "_Part_2_"+str(counter) + '_left.csv', index=False,
                     header=False)
           model_3.load_weights('C:\\xampp\htdocs\gaze\Models/MyGaze-6.h5')
           detector = dlib.get_frontal_face_detector()
           predictor = dlib.shape_predictor(
               'C:\\xampp\htdocs\gaze\Models/shape_predictor_68_face_landmarks.dat')
           files1 = os.listdir('C:\\xampp\htdocs\gaze/' + str(tyt) + "/")
           len1 = len(files1)
           len2 = len1 / 2
           len2 = int(len2)
           files3 = files1[len2:len1]
           for f in files3:
               img = cv2.imread('C:\\xampp\htdocs\gaze/' + str(tyt) + "/" + f)
               image = img
               gray = image
               rects = detector(gray, 1)
               for (i, rect) in enumerate(rects):
                   shape = predictor(gray, rect)
                   shape = face_utils.shape_to_np(shape)
                   for (name, (i, j)) in face_utils.FACIAL_LANDMARKS_IDXS.items():
                       p = 1
                       if name == 'left_eye':
                           image = image.copy()
                           for (x, y) in shape[i:j]:
                               (x, y, w, h) = cv2.boundingRect(np.array([shape[i:j]]))
                               roi = image[y - 20:y + h + 28, x - 10:x + w + 10]
                               roi = cv2.resize(roi, (90, 90))
                               pred = model_3.predict(roi.reshape((1, 90, 90, 3)))
                               a = pred[0][0]
                               b = pred[0][1]
                               a = round(a, 1)
                               b = round(b, 1)
               mi = mi + 1
               right_x.append(a)
               right_y.append(b)
               count = count + 1
               ia = ia + 1
           x = [34, 3]
           y = [3, 17]
           a = []

           for i in x:
               a.append(i - 2)

           y1 = []

           for i in y:
               y1.append(i - 2)

           b = []

           for i in x:
               b.append(i + 2)

           y2 = []

           for i in y:
               y2.append(i + 2)
           plt.plot(x, y, color="yellow")
           plt.plot(a, y, color="red")
           plt.plot(b, y, color="darkgreen")
           minimum1 = (min(right_x))
           maximum1 = (max(right_x))
           maximum2 = (max(right_y))
           minimum2 = (min(right_y))
           plt.scatter(right_x, right_y, color="red")
           index1 = right_x.index(min(right_x))
           index2 = right_x.index(max(right_x))
           index3 = right_y.index(min(right_y))
           index4 = right_y.index(max(right_y))
           predictedx = []
           predictedy = []
           predictedy1 = []
           predictedx1 = []
           predictedx.append(minimum1)
           predictedx.append(maximum1)
           predictedy.append(right_y[index1])
           predictedy.append(right_y[index2])
           predictedx.reverse()
           predictedy.reverse()
           predictedy1.append(minimum2)
           predictedy1.append(maximum2)
           plt.plot(predictedx, predictedy, color='darkorange')
           plt.plot(predictedx, predictedy1, color='darkorange')
           plt.fill_between(predictedx, predictedy, predictedy1, color='#539ecd')
           plt.title('Screen size 36cm*20cm (W*H)', fontsize=10)
           y = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]
           y.reverse()
           plt.yticks(y, fontsize=8)
           x = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27,
                28, 29, 30, 31, 32, 33, 34, 35, 36]
           x.reverse()
           plt.xticks(x, fontsize=8)
           plt.grid()
           filename13 = str(session["id"]) +"_Part2_righteye" + str(counter) + '.png'
           plt.savefig('C:\\xampp\htdocs\gaze/static/uploads/' + filename13)
           plt.close()
           import pandas as pd
           list_dict = {'x': right_x, 'y': right_y}
           df = pd.DataFrame(list_dict)
           csv_file8='C:\\xampp\htdocs\gaze/static/uploads/' + str(session["id"]) + "_Part_2_"+str(counter) + '_right.csv'
           df.to_csv('C:\\xampp\htdocs\gaze/static/uploads/' + str(session["id"]) + "_Part_2_"+str(counter) + '_right.csv', index=False,
                     header=False)

       if (counter==2):
           left_x = []
           left_y = []
           right_x = []
           right_y = []
           mi = 1
           ia = 0
           count = 0
           import cv2
           import numpy as np
           import time
           cap = cv2.VideoCapture('C:\\xampp\htdocs\gaze/uploads/' + f)
           i = 0
           start_time = time.time()
           while (cap.isOpened()):
               ret, frame = cap.read()
               if ret == False:
                   break
               tyt = str(session["id"]) +"_frames" + str(counter)
               if not os.path.exists(tyt):
                   os.mkdir(tyt)
               cv2.imwrite('C:\\xampp\htdocs\gaze/' + str(tyt) + "/" + str("%04d" % i) + '.jpg', frame)
               i += 1

           cap.release()
           cv2.destroyAllWindows()
           input_img_1 = Input(shape=(90, 90, 3))
           model_1 = create_convolution_layers1(input_img_1)
           model_3 = final(model_1, input_img_1)
           model_3.compile(optimizers.adam(lr=0.0001), loss='mae', metrics=['accuracy'])
           model_3.load_weights('C:\\xampp\htdocs\gaze\Models/MyGaze-5 (1).h5')
           detector = dlib.get_frontal_face_detector()
           predictor = dlib.shape_predictor(
               'C:\\xampp\htdocs\gaze\Models/shape_predictor_68_face_landmarks.dat')
           files1 = os.listdir('C:\\xampp\htdocs\gaze/' + str(tyt) + "/")
           for f in files1:
               img = cv2.imread('C:\\xampp\htdocs\gaze/' + str(tyt) + "/" + f)
               image = img
               gray = image
               rects = detector(gray, 1)
               for (i, rect) in enumerate(rects):
                   shape = predictor(gray, rect)
                   shape = face_utils.shape_to_np(shape)
                   for (name, (i, j)) in face_utils.FACIAL_LANDMARKS_IDXS.items():
                       p = 1
                       if name == 'right_eye':
                           image = image.copy()
                           for (x, y) in shape[i:j]:
                               (x, y, w, h) = cv2.boundingRect(np.array([shape[i:j]]))
                               roi = image[y - 20:y + h + 28, x - 10:x + w + 10]
                               roi = cv2.resize(roi, (90, 90))
                               pred = model_3.predict(roi.reshape((1, 90, 90, 3)))
                               a = pred[0][0]
                               b = pred[0][1]
                               a = round(a, 1)
                               b = round(b, 1)
               mi = mi + 1
               left_x.append(a)
               left_y.append(b)
               count = count + 1
               ia = ia + 1

           y = []
           y.append(3.5)
           for i in range(3, 17, 2):
               y.append(i)
           y.append(17)
           plt.scatter(left_x, left_y, color="red")
           minimum1 = (min(left_x))
           maximum1 = (max(left_x))
           maximum2 = (max(left_y))
           minimum2 = (min(left_y))
           predictedx = [minimum1, maximum1]
           x = [17 for i in range(len(y))]
           a = [19 for i in range(len(y))]
           b = [15 for i in range(len(y))]
           predictedyabove = [maximum2 for i in range(len(predictedx))]
           predictedybelow = [minimum2 for i in range(len(predictedx))]
           plt.plot(x, y, color="yellow")
           plt.plot(a, y, color="red")
           plt.plot(b, y, color="darkgreen")
           plt.plot(predictedx, predictedyabove, color='darkorange')
           plt.plot(predictedx, predictedybelow, color='darkorange')
           plt.fill_between(predictedx, predictedyabove, predictedybelow, color='#539ecd', alpha=0.5)
           plt.title('Screen size 36cm*20cm (W*H)', fontsize=10)
           y = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]
           plt.yticks(y, fontsize=8)
           x = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27,
                28, 29, 30, 31, 32, 33, 34, 35, 36]
           plt.xticks(x, fontsize=8)
           plt.grid()
           filename10 = str(session["id"]) +"_lefteye" + str(counter) + '.png'
           plt.savefig('C:\\xampp\htdocs\gaze/static/uploads/' + filename10)
           import pandas as pd
           plt.close()
           list_dict = {'x': left_x, 'y': left_y}
           df = pd.DataFrame(list_dict)
           csv_file9='C:\\xampp\htdocs\gaze/static/uploads/' + str(session["id"]) +"_"+ str(counter) + '_left.csv'
           df.to_csv('C:\\xampp\htdocs\gaze/static/uploads/' + str(session["id"]) +"_"+ str(counter) + '_left.csv', index=False,
                     header=False)
           model_3.load_weights('C:\\xampp\htdocs\gaze\Models/MyGaze-6.h5')
           detector = dlib.get_frontal_face_detector()
           predictor = dlib.shape_predictor(
               'C:\\xampp\htdocs\gaze\Models/shape_predictor_68_face_landmarks.dat')
           files = os.listdir('C:\\xampp\htdocs\gaze/' + str(tyt) + "/")
           for f in files:
               img = cv2.imread('C:\\xampp\htdocs\gaze/' + str(tyt) + "/" + f)
               image = img
               gray = image
               rects = detector(gray, 1)
               for (i, rect) in enumerate(rects):
                   shape = predictor(gray, rect)
                   shape = face_utils.shape_to_np(shape)
                   for (name, (i, j)) in face_utils.FACIAL_LANDMARKS_IDXS.items():
                       p = 1
                       if name == 'left_eye':
                           image = image.copy()
                           for (x, y) in shape[i:j]:
                               (x, y, w, h) = cv2.boundingRect(np.array([shape[i:j]]))
                               roi = image[y - 20:y + h + 28, x - 10:x + w + 10]
                               roi = cv2.resize(roi, (90, 90))
                               pred = model_3.predict(roi.reshape((1, 90, 90, 3)))
                               a = pred[0][0]
                               b = pred[0][1]
                               a = round(a, 1)
                               b = round(b, 1)
               mi = mi + 1
               right_x.append(a)
               right_y.append(b)
               count = count + 1
               ia = ia + 1
           y = []
           y.append(3.5)
           for i in range(3, 17, 2):
               y.append(i)
           y.append(17)
           plt.scatter(right_x, right_y, color="red")
           minimum1 = (min(right_x))
           maximum1 = (max(right_x))
           maximum2 = (max(right_y))
           minimum2 = (min(right_y))
           predictedx = [minimum1, maximum1]
           x = [17 for i in range(len(y))]
           a = [19 for i in range(len(y))]
           b = [15 for i in range(len(y))]
           predictedyabove = [maximum2 for i in range(len(predictedx))]
           predictedybelow = [minimum2 for i in range(len(predictedx))]
           plt.plot(x, y, color="yellow")
           plt.plot(a, y, color="red")
           plt.plot(b, y, color="darkgreen")
           plt.plot(predictedx, predictedyabove, color='darkorange')
           plt.plot(predictedx, predictedybelow, color='darkorange')
           plt.fill_between(predictedx, predictedyabove, predictedybelow, color='#539ecd', alpha=0.5)
           plt.title('Screen size 36cm*20cm (W*H)', fontsize=10)
           y = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]
           plt.yticks(y, fontsize=8)
           x = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27,
                28, 29, 30, 31, 32, 33, 34, 35, 36]
           plt.xticks(x, fontsize=8)
           plt.grid()
           filename11 = str(session["id"]) +"_righteye" + str(counter) + '.png'
           plt.savefig('C:\\xampp\htdocs\gaze/static/uploads/' + filename11)
           plt.close()
           import pandas as pd
           list_dict = {'x': right_x, 'y': right_y}
           df = pd.DataFrame(list_dict)
           csv_file10='C:\\xampp\htdocs\gaze/static/uploads/' + str(session["id"]) + "_"+str(counter) + '_right.csv'
           df.to_csv('C:\\xampp\htdocs\gaze/static/uploads/' + str(session["id"]) + "_"+str(counter) + '_right.csv', index=False,
                     header=False)


       if (counter==1):
         left_x = []
         left_y = []
         right_x = []
         right_y = []
         mi = 1
         ia = 0
         count = 0
         import cv2
         import numpy as np
         import time
         cap = cv2.VideoCapture('C:\\xampp\htdocs\gaze/uploads/' + f)
         i = 0
         start_time = time.time()
         while (cap.isOpened()):
             ret, frame = cap.read()
             if ret == False:
                 break
             tyt = str(session["id"]) +"_frames" + str(counter)
             if not os.path.exists(tyt):
                 os.mkdir(tyt)
             cv2.imwrite('C:\\xampp\htdocs\gaze/' + str(tyt) + "/" + str("%04d" % i) + '.jpg', frame)
             i += 1

         cap.release()
         cv2.destroyAllWindows()
         input_img_1 = Input(shape=(90, 90, 3))
         model_1 = create_convolution_layers1(input_img_1)
         model_3 = final(model_1, input_img_1)
         model_3.compile(optimizers.adam(lr=0.0001), loss='mae', metrics=['accuracy'])
         model_3.load_weights('C:\\xampp\htdocs\gaze\Models/MyGaze-5 (1).h5')
         detector = dlib.get_frontal_face_detector()
         predictor = dlib.shape_predictor(
             'C:\\xampp\htdocs\gaze\Models/shape_predictor_68_face_landmarks.dat')
         files1 = os.listdir('C:\\xampp\htdocs\gaze/' + str(tyt) + "/")
         for f in files1:
             img = cv2.imread('C:\\xampp\htdocs\gaze/' + str(tyt) + "/" + f)
             image = img
             gray = image
             rects = detector(gray, 1)
             for (i, rect) in enumerate(rects):
                 shape = predictor(gray, rect)
                 shape = face_utils.shape_to_np(shape)
                 for (name, (i, j)) in face_utils.FACIAL_LANDMARKS_IDXS.items():
                     p = 1
                     if name == 'right_eye':
                         image = image.copy()
                         for (x, y) in shape[i:j]:
                             (x, y, w, h) = cv2.boundingRect(np.array([shape[i:j]]))
                             roi = image[y - 20:y + h + 28, x - 10:x + w + 10]
                             roi = cv2.resize(roi, (90, 90))
                             pred = model_3.predict(roi.reshape((1, 90, 90, 3)))
                             a = pred[0][0]
                             b = pred[0][1]
                             a = round(a, 1)
                             b = round(b, 1)
             mi = mi + 1
             left_x.append(a)
             left_y.append(b)
             count = count + 1
             ia = ia + 1

         x = []
         x.append(3)
         for i in range(3, 34, 2):
             x.append(i)
         x.append(34)
         minimum1 = int(min(left_x))
         maximum1 = int(max(left_x))
         maximum2 = int(max(left_y))
         minimum2 = int(min(left_y))
         predictedx = []
         predictedx = [minimum1, maximum1]
         y = [8 for i in range(len(x))]
         y1 = [10 for i in range(len(x))]
         y2 = [6 for i in range(len(x))]
         predictedyabove = [maximum2 for i in range(len(predictedx))]
         predictedybelow = [minimum2 for i in range(len(predictedx))]
         plt.plot(x, y, color="yellow")
         plt.plot(x, y1, color="red")
         plt.plot(x, y2, color="red")
         plt.scatter(left_x, left_y, color="blue")
         plt.title('Screen size 36*20cm (W*H)', fontsize=10)
         predictedyabove = [maximum2 for i in range(len(predictedx))]
         predictedybelow = [minimum2 for i in range(len(predictedx))]
         plt.plot(predictedx, predictedyabove, color='darkorange')
         plt.plot(predictedx, predictedybelow, color='darkorange')
         plt.fill_between(predictedx, predictedyabove, predictedybelow, color='#539ecd', alpha=0.5)
         y = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]
         plt.yticks(y, fontsize=8)
         x = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28,
              29, 30, 31, 32, 33, 34, 35, 36]
         plt.xticks(x, fontsize=8)
         plt.grid()
         filename18 = str(session["id"]) +"_lefteye" + str(counter) + '.png'
         plt.savefig('C:\\xampp\htdocs\gaze/static/uploads/' + filename18)
         plt.close()
         import pandas as pd
         list_dict = {'x': left_x, 'y': left_y}
         df = pd.DataFrame(list_dict)
         csv_file11='C:\\xampp\htdocs\gaze/static/uploads/' + str(session["id"]) + "_"+ str(counter) + '_left.csv'
         df.to_csv('C:\\xampp\htdocs\gaze/static/uploads/' + str(session["id"]) + "_"+ str(counter) + '_left.csv', index=False,
                   header=False)
         model_3.load_weights('C:\\xampp\htdocs\gaze\Models/MyGaze-6.h5')
         detector = dlib.get_frontal_face_detector()
         predictor = dlib.shape_predictor(
             'C:\\xampp\htdocs\gaze\Models/shape_predictor_68_face_landmarks.dat')
         files1 = os.listdir('C:\\xampp\htdocs\gaze/' + str(tyt) + "/")
         for f in files1:
             img = cv2.imread('C:\\xampp\htdocs\gaze/' + str(tyt) + "/" + f)
             image = img
             gray = image
             rects = detector(gray, 1)
             for (i, rect) in enumerate(rects):
                 shape = predictor(gray, rect)
                 shape = face_utils.shape_to_np(shape)
                 for (name, (i, j)) in face_utils.FACIAL_LANDMARKS_IDXS.items():
                     p = 1
                     if name == 'left_eye':
                         image = image.copy()
                         for (x, y) in shape[i:j]:
                             (x, y, w, h) = cv2.boundingRect(np.array([shape[i:j]]))
                             roi = image[y - 20:y + h + 28, x - 10:x + w + 10]
                             roi = cv2.resize(roi, (90, 90))
                             pred = model_3.predict(roi.reshape((1, 90, 90, 3)))
                             a = pred[0][0]
                             b = pred[0][1]
                             a = round(a, 1)
                             b = round(b, 1)
             mi = mi + 1
             right_x.append(a)
             right_y.append(b)
             count = count + 1
             ia = ia + 1
         x = []
         x.append(3)
         for i in range(3, 34, 2):
             x.append(i)
         x.append(34)
         minimum1 = int(min(right_x))
         maximum1 = int(max(right_x))
         maximum2 = int(max(right_y))
         minimum2 = int(min(right_y))
         predictedx = []
         predictedx = [minimum1, maximum1]
         y = [8 for i in range(len(x))]
         y1 = [10 for i in range(len(x))]
         y2 = [6 for i in range(len(x))]
         predictedyabove = [maximum2 for i in range(len(predictedx))]
         predictedybelow = [minimum2 for i in range(len(predictedx))]
         plt.plot(x, y, color="yellow")
         plt.plot(x, y1, color="red")
         plt.plot(x, y2, color="red")
         plt.scatter(right_x, right_y, color="blue")
         plt.title('Screen size 36*20cm (W*H)', fontsize=10)
         predictedyabove = [maximum2 for i in range(len(predictedx))]
         predictedybelow = [minimum2 for i in range(len(predictedx))]
         plt.plot(predictedx, predictedyabove, color='darkorange')
         plt.plot(predictedx, predictedybelow, color='darkorange')
         plt.fill_between(predictedx, predictedyabove, predictedybelow, color='#539ecd', alpha=0.5)
         y = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]
         plt.yticks(y, fontsize=8)
         x = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28,
              29,
              30,
              31, 32, 33, 34, 35, 36]
         plt.xticks(x, fontsize=8)
         plt.grid()
         filename19 = str(session["id"]) +"_righteye" + str(counter) + '.png'
         plt.savefig('C:\\xampp\htdocs\gaze/static/uploads/' + filename19)
         plt.close()
         import pandas as pd

         list_dict = {'x': right_x, 'y': right_y}
         df = pd.DataFrame(list_dict)
         csv_file12='C:\\xampp\htdocs\gaze/static/uploads/' + str(session["id"]) +"_"+ str(counter) + '_right.csv'
         df.to_csv('C:\\xampp\htdocs\gaze/static/uploads/' + str(session["id"]) +"_"+ str(counter) + '_right.csv',
                   index=False, header=False)

       counter=counter+1
   return filename10,filename11,filename12,filename13,filename14,filename15,filename16,filename17,filename18,filename19,filename20,filename21,csv_file1,csv_file2,\
                csv_file3,csv_file4,csv_file5,csv_file6,csv_file7,csv_file8,csv_file9,csv_file10,csv_file11,csv_file12


@app.route('/animation1',methods=["GET" ,"POST"])
def submit3():
        ###########Get Session ID ########
        data=request.form["session_id"]
        session["id"] = data
        ###########Query To Get values of time for left key ########
        conn = mysql.connect()
        cursor = conn.cursor()
        cursor.execute("""SELECT left_arrow FROM video WHERE session_id= %s LIMIT 1""", session["id"])
        left = cursor.fetchone()
        left = str(left)
        left = re.sub("[()]", "", left)
        left = re.sub(",", "", left)
        left = re.sub("'", "", left)
        conn.commit()
        cursor.close()
        ###########Query To Get values of time for right key ########

        conn = mysql.connect()
        cursor = conn.cursor()
        cursor.execute("""SELECT right_arrow FROM video WHERE session_id= %s LIMIT 1""", session["id"])
        right= cursor.fetchone()
        right = str(right)
        right = re.sub("[()]", "", right)
        right= re.sub(",", "", right)
        right = re.sub("'", "", right)

        conn.commit()
        cursor.close()

        ###########Query To Get values of time for down key ########
        conn = mysql.connect()
        cursor = conn.cursor()
        cursor.execute("""SELECT down_arrow FROM video WHERE session_id= %s  LIMIT 1""", session["id"])
        down = cursor.fetchone()
        down= str(down)
        down = re.sub("[()]", "", down)
        down = re.sub(",", "", down)
        down = re.sub("'", "", down)

        conn.commit()
        cursor.close()

        ###########Compute Time ########
        if left=="":
            leftvalue="You not press the key"
        if left!="":
            leftvalue=abs(int(left)-1)

        if right == "":
            rightvalue = "You not press the key"
        if right != "":
            rightvalue = abs(int(right) - 6)

        if down == "":
            downvalue = "You not press the key"
        if down != "":
            downvalue= abs(int(down) - 11)

        return render_template('mainpage.html',id=session["id"],left=leftvalue,right=rightvalue,down=downvalue)


@app.route('/')
def mainpage():
    return render_template("mainpage.html")

@app.route('/test', methods=['GET','POST'])
def test():
        conn = mysql.connect()
        cursor = conn.cursor()
        cursor.execute("""SELECT * FROM video WHERE session_id= %s""", session["id"])
        data = cursor.fetchall()
        # check whether user has done 5 experiments or not (i-e 5 videos)
        if len(data) == 5:
            conn.commit()
            cursor.close()
            filename10,filename11,filename12,filename13,filename14,filename15,filename16,filename17,filename18,filename19,filename20,filename21,csv_file1,csv_file2,\
                csv_file3,csv_file4,csv_file5,csv_file6,csv_file7,csv_file8,csv_file9,csv_file10,csv_file11,csv_file12=predict()
            # To check whether all files of the current session of the user are saved in directory or not ( for each user there exists total files)
            if os.path.isfile('C:\\xampp\htdocs\gaze/static/uploads/'+filename10)\
                    and os.path.isfile('C:\\xampp\htdocs\gaze/static/uploads/'+filename11)\
                    and os.path.isfile('C:\\xampp\htdocs\gaze/static/uploads/'+filename12) \
                    and os.path.isfile('C:\\xampp\htdocs\gaze/static/uploads/' + filename13) \
                    and os.path.isfile('C:\\xampp\htdocs\gaze/static/uploads/' + filename14) \
                    and os.path.isfile('C:\\xampp\htdocs\gaze/static/uploads/' + filename15) \
                    and os.path.isfile('C:\\xampp\htdocs\gaze/static/uploads/' + filename16) \
                    and os.path.isfile('C:\\xampp\htdocs\gaze/static/uploads/' + filename17) \
                    and os.path.isfile('C:\\xampp\htdocs\gaze/static/uploads/' + filename18) \
                    and os.path.isfile('C:\\xampp\htdocs\gaze/static/uploads/' + filename19) \
                    and os.path.isfile('C:\\xampp\htdocs\gaze/static/uploads/' + filename20) \
                    and os.path.isfile('C:\\xampp\htdocs\gaze/static/uploads/' + filename21) \
                    and os.path.isfile('C:\\xampp\htdocs\gaze/static/uploads/' + filename21) \
                    and os.path.isfile(csv_file1) \
                    and os.path.isfile(csv_file2) \
                    and os.path.isfile(csv_file3)\
                    and os.path.isfile(csv_file4) \
                    and os.path.isfile(csv_file5) \
                    and os.path.isfile(csv_file6) \
                    and os.path.isfile(csv_file7) \
                    and os.path.isfile(csv_file8) \
                    and os.path.isfile(csv_file9) \
                    and os.path.isfile(csv_file10) \
                    and os.path.isfile(csv_file11) \
                    and os.path.isfile(csv_file12):
                data = {
                    "Success is ": True,
                    "  Animation2 Graph (Left Eye)  ": 'C:\\xampp\htdocs\gaze/static/uploads/' + filename18,
                    "  Animation2 Graph (Right Eye)  ": 'C:\\xampp\htdocs\gaze/static/uploads/' +  filename19,
                    "  Animation3 Graph (Left Eye)  ": 'C:\\xampp\htdocs\gaze/static/uploads/' + filename10,
                    "  Animation3 Graph (Right Eye)  ": 'C:\\xampp\htdocs\gaze/static/uploads/' + filename11,
                    "  Animation4-Part-1 Graph (Left Eye)  ":'C:\\xampp\htdocs\gaze/static/uploads/' +  filename13,
                    "  Animation4-Part-1 Graph (Right Eye)  ": 'C:\\xampp\htdocs\gaze/static/uploads/' + filename12,
                    "  Animation4-Part-2 Graph (Left Eye)  ": 'C:\\xampp\htdocs\gaze/static/uploads/' + filename14,
                    "  Animation4-Part-2 Graph (Right Eye)  ":'C:\\xampp\htdocs\gaze/static/uploads/' +  filename15,
                    "  Animation5 Graph (Left Eye)  ": 'C:\\xampp\htdocs\gaze/static/uploads/' + filename16,
                    "  Animation5 Graph (Right Eye)  ":'C:\\xampp\htdocs\gaze/static/uploads/' +  filename17,
                    "  Animation5 Graph (Left Eye)  ": 'C:\\xampp\htdocs\gaze/static/uploads/' + filename20,
                    "  Animation5 Graph (Right Eye)  ":'C:\\xampp\htdocs\gaze/static/uploads/' +  filename21,
                    "  Animation2 CSV file (Left Eye)  ": csv_file11,
                    "  Animation2 CSV file  (Right Eye)  ": csv_file12,
                    "  Animation3 CSV file  (Left Eye)  ": csv_file9,
                    "  Animation3 CSV file  (Right Eye)  ": csv_file10,
                    "  Animation4-Part-1 CSV file  (Left Eye)  ": csv_file7,
                    "  Animation4-Part-1 CSV file  (Right Eye)  ": csv_file8,
                    "  Animation4-Part-2 CSV file  (Left Eye)  ": csv_file5,
                    "  Animation4-Part-2 CSV file  (Right Eye)  ": csv_file6,
                    "  Animation5 CSV file  (Left Eye)  ": csv_file3,
                    "  Animation5 CSV file  (Right Eye)  ":csv_file4,
                    "  Animation5 CSV file (Left Eye)  ": csv_file1,
                    "  Animation5 CSV file  (Right Eye)  ": csv_file2,
                }
                # Return JSON Object
            return jsonify(data)

if __name__ == '__main__':
    app.run(threaded = True)