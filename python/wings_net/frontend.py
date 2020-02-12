from __future__ import print_function

import sys, os
sys.path.append('./python/wings_net')

from keras.models import Model, load_model
from keras.optimizers import SGD, Adam, RMSprop, Nadam
from keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard, LearningRateScheduler
from keras.losses import categorical_crossentropy
from keras.layers import *
from backend import *
from keras import backend as K
from keras import metrics
# from preprocessing import BatchGenerator, TextImageGenerator
# from utils import decode_netout, compute_overlap, compute_ap
from keras.utils.generic_utils import CustomObjectScope
import keras
import itertools
from scipy.ndimage.filters import gaussian_filter1d
from glob import glob
from sklearn.utils import class_weight
import math
from keras import regularizers
import time
from keras.applications import MobileNetV2

import tensorflow as tf
import numpy as np
import os
import cv2


class OCR(object):
    def __init__(self, labels, input_size=188):
        self.input_size = input_size
        self.labels   = list(labels)
        self.nb_class = len(self.labels)
        self.class_wt = np.ones(self.nb_class, dtype='float32')
        
        sq1x1  = "squeeze1x1"
        exp1x1 = "expand1x1"
        sq1x3 = "squeeze1x3"
        sq3x1 = "squeeze3x1"

        def fire_module(x, fire_id, squeeze=16, expand=64):
            s_id = 'fire' + str(fire_id) + '/'

            x = Conv2D(squeeze, (1, 1), padding='valid', name=s_id + sq1x1)(x)
            x = BatchNormalization()(x)
            x = LeakyReLU()(x)    

            x  = Conv2D(squeeze,  (3, 1), padding='same', name=s_id + sq1x3)(x)
            x = BatchNormalization()(x)
            x = LeakyReLU()(x)    

            x = Conv2D(squeeze,  (1, 3), padding='same',  name=s_id + sq3x1)(x)
            x = BatchNormalization()(x)
            x = LeakyReLU()(x)    
                        
            x = Conv2D(expand,  (1, 1), padding='valid',  name=s_id + exp1x1)(x)
            x = BatchNormalization()(x)
            x = LeakyReLU()(x)    

            return x

        input_image     = Input(shape=(self.input_size, 48, 3),name='input_data')
        
        left = AveragePooling2D(pool_size=(2,2), strides=(2,2))(input_image)        
        left = Conv2D(32, (5,5), strides=(2,2),
                kernel_initializer='he_normal', padding='same')(left)
           
        right = Conv2D(32, (5,5), strides=(4,4),
                kernel_initializer='he_normal', padding='same')(input_image)
        locnet = Concatenate()([left,right])
        locnet = Dropout(0.5)(locnet)
        locnet = Conv2D(32, (5,5), strides=(2,2),
                kernel_initializer='he_normal', padding='same')(locnet)
        locnet = Dropout(0.5)(locnet)
        locnet = Flatten()(locnet)
        locnet = Dense(32, activation='tanh')(locnet)
        weights = get_initial_weights(32)
        locnet = Dense(6, activation='tanh', weights=weights)(locnet)
        warped_image = BilinearInterpolation((188,48), name='bill')([input_image, locnet])
        
        inner = Conv2D(64, (3,3), padding='same', strides=(1,1),
                       kernel_initializer='he_normal',
                       kernel_regularizer=regularizers.l2(0.01),
                       name='conv1')(warped_image)
        inner = BatchNormalization()(inner)
        inner = LeakyReLU() (inner)
        inner_r = MaxPooling2D(pool_size=(3, 3), strides=(1,1), name='max1', padding='same')(inner)    
        

        inner_2 = fire_module(inner_r, fire_id=2, squeeze=16, expand=64)
        inner = Add()([inner_r,inner_2])
        inner = BatchNormalization()(inner)
        inner = LeakyReLU() (inner)    
        inner = MaxPooling2D(pool_size=(3, 3), strides=(2,1), name='max2', padding='same')(inner)     
        
        inner = fire_module(inner, fire_id=3, squeeze=64, expand=256)
        inner = BatchNormalization()(inner)
        inner_r = LeakyReLU() (inner) 

        inner_3 = fire_module(inner_r, fire_id=4, squeeze=64, expand=256)
        inner = Add()([inner_r,inner_3])
        inner = BatchNormalization()(inner)
        inner = LeakyReLU() (inner) 
        inner = MaxPooling2D(pool_size=(3, 3), strides=(2,1), name='max3', padding='same')(inner) 

        inner = Conv2D(256, (4,1), padding='same', strides=(1,1),
                kernel_initializer='he_normal',
                name='conv5')(inner)
        inner = Dropout(0.5)(inner)
        inner = BatchNormalization()(inner)
        inner_r = LeakyReLU() (inner)

        inner = Conv2D(256, (8,1), padding='same', strides=(1,1),
                kernel_initializer='he_normal',
                name='conv5_res')(inner_r)
        inner = Dropout(0.5)(inner)
        inner = BatchNormalization()(inner)
        inner = LeakyReLU() (inner)
        inner = Add(name='add1')([inner_r,inner])
        
        inner = Conv2D(self.nb_class+1, (1,18), padding='same', strides=(1,1),
                kernel_initializer='he_normal',
                       kernel_regularizer=regularizers.l2(0.01),
                name='conv6')(inner)
        inner = LeakyReLU() (inner)
        inner = Lambda(lambda x: x ** 2)(inner)
        c_inner = GlobalAveragePooling2D()(inner)
        inner = Lambda(self.division)([inner, c_inner])
        
        inner_1 = AveragePooling2D(pool_size=(4, 1), strides=(4,1), name='avg1', padding='same')(warped_image) 
        inner_1 = Lambda(lambda x: x ** 2)(inner_1)
        c_inner1 = GlobalAveragePooling2D()(inner_1)
        inner_1 = Lambda(self.division)([inner_1, c_inner1])

        
        inner_2 = AveragePooling2D(pool_size=(4, 1), strides=(4,1), name='avg2', padding='same')(inner_2) 
        inner_2 = Lambda(lambda x: x ** 2)(inner_2)
        c_inner2 = GlobalAveragePooling2D()(inner_2)
        inner_2 = Lambda(self.division)([inner_2, c_inner2])

        
        inner_3 = AveragePooling2D(pool_size=(2, 1), strides=(2,1), name='avg3', padding='same')(inner_3) 
        inner_3 = Lambda(lambda x: x ** 2)(inner_3)
        c_inner3 = GlobalAveragePooling2D()(inner_3)
        inner_3 = Lambda(self.division)([inner_3, c_inner3])
        
        inner = Concatenate(name='concat2')([inner,inner_1,inner_2,inner_3])
        
        inner = Conv2D(self.nb_class+1, (3,3), strides=(2,2),
                kernel_initializer='he_normal',
                name='conv7')(inner)
        
        inner_max = MaxPooling2D(pool_size=(1, 23)) (inner)
        inner_avg = AveragePooling2D(pool_size=(1, 23)) (inner)
        inner_conv = Conv2D(self.nb_class+1, (1,23), strides=(1,1),
                kernel_initializer='he_normal',
                name='conv7_c')(inner)
        inner = Concatenate(axis=2)([inner_max,inner_avg,inner_conv])

        inner = Conv2D(self.nb_class+1, (1,3), strides=(1,1),
                kernel_initializer='he_normal',
                name='conv8')(inner)

        inner = Reshape(target_shape=(23,37))(inner)
        
        y_pred = Activation('softmax', name='softmax')(inner)

        out_labels = Input(name='the_labels', shape=[9], dtype='float32')
        input_length = Input(name='input_length', shape=[1], dtype='int64')
        label_length = Input(name='label_length', shape=[1], dtype='int64')
        loss_out = Lambda(self.ctc_lambda_func, output_shape=(1,), name='ctc')([y_pred, out_labels, input_length, label_length])
        self.model = Model(inputs=[input_image, out_labels, input_length, label_length], outputs=loss_out)
        self.intermediate_layer_model = Model(inputs=self.model.get_layer('input_data').input,
                                         outputs=self.model.get_layer('softmax').output)

        self.first_lp = True
        self.output_lp = None
        
        for layer in self.model.layers:
            layer.trainable = False

        tt = Model(inputs=self.model.input,outputs=self.model.get_layer('bill').output)
        for layer in tt.layers:
            layer.trainable = True

        self.model.summary()

    
    
    def division(self,args):
        x,cx = args        
        
        cx = Reshape(target_shape=(1,1,cx.shape[1].value))(cx)
        cx= UpSampling2D(size=(47,48))(cx)

        return x/cx

    def ctc_lambda_func(self,args):
        y_pred, out_labels, input_length, label_length = args

        y_pred = y_pred[:, 2:, :]
        return K.ctc_batch_cost(out_labels, y_pred, input_length, label_length)


    def load_weights(self, weight_path):
        self.model.load_weights(weight_path)  

    def train(self, train_images,     
                    valid_images,    
                    epochs,     
                    lr_rate,  
                    batch_size,     
                    saved_weights_name='best_weights.h5',
                    train_annot_folder='fail',
                    debug=False):     

        model_json = self.model.to_json()
        with open(saved_weights_name[:-3]+".json", "w") as json_file:
            json_file.write(model_json)
            
        self.batch_size = batch_size
        self.debug = debug

        generator_config = {
            'IMAGE_H'         : 48, 
            'IMAGE_W'         : self.input_size,
            'LABELS'          : self.labels,
            'CLASS'           : len(self.labels),
            'BATCH_SIZE'      : self.batch_size,
        }    

        train_generator = TextImageGenerator(train_images, 
                                        generator_config, 
                                        max_text_len = 9)
        valid_generator = TextImageGenerator(valid_images, 
                                        generator_config, 
                                        max_text_len = 9,
                                        jitter=False)   
                                             
        optimizer = Adam(lr=lr_rate, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)

        self.model.compile(loss={'ctc': lambda y_true, y_pred: y_pred}, optimizer=optimizer)
        
        checkpoint = ModelCheckpoint(saved_weights_name, 
                                        monitor='val_loss', 
                                        verbose=1, 
                                        save_best_only=False, 
                                        mode='min', 
                                        period=1)
        tensorboard = TensorBoard(log_dir=os.path.expanduser('logs/'), 
                                    histogram_freq=0, 
                                    write_graph=True, 
                                    write_images=False)
     
        all_label = []
        for each in glob(train_annot_folder+'*.txt'):
            with open(each,"r") as f:
                content = f.readlines()
            content = [x.strip() for x in content]
            plate = content[0]
            for c in plate:
                all_label.append(c)
        class_weights = class_weight.compute_class_weight('balanced',
                                                 np.unique(all_label),
                                                 all_label)
        print(class_weights)
        self.model.fit_generator(generator        = train_generator, 
                                    steps_per_epoch  = len(train_generator) , 
                                    epochs           = epochs, 
                                    verbose          = 1,
                                    validation_data  = valid_generator,
                                    validation_steps = len(valid_generator),
                                    callbacks        = [checkpoint, tensorboard], 
                                    workers          = 2,
                                    max_queue_size   = 8,
                                    class_weight     = class_weights)  

      
    def decode_batch(self,out):
        ret = []
        for j in range(out.shape[0]):
            out_best = list(np.argmax(out[j, 2:], 1))
            count = 2
            startNum = 0
            endNum = 0
            out_best2 = [k for k, g in itertools.groupby(out_best)]
            short_best = []
            prob_best = []
            k = 0
            l = 0
            for c in out_best: 
                if k < len(out_best2) and out_best2[k] == c:
                    if c < len(self.labels):
                        short_best.append(c)
                        prob_best.append(out[j,count,c])
                        if c < 10:
                            if startNum == 0:
                                startNum = l
                        if c > 9 and c < 36:
                            if(startNum != 0 and endNum == 0):
                                endNum = l
                        l += 1
                    k += 1
                count += 1
            if endNum == 0:
                endNum = len(short_best)-1
            np.asarray(prob_best)

            prob_char1 = np.argsort(prob_best[:startNum],0)
            prob_number = np.argsort(prob_best[startNum:endNum-1],0)
            prob_char2 = np.argsort(prob_best[endNum:],0)
            max_char1 = 0
            max_number = 0 
            max_char2 = 0

            if len(prob_char1) > 2:
                max_char1 = prob_best[prob_char1[len(prob_char1)-3]]
            if len(prob_number) > 4:
                max_number = prob_best[startNum+prob_number[len(prob_char1)-5]]
            if len(prob_char2) > 3:
                max_char2 = prob_best[endNum+prob_char2[len(prob_char1)-4]]

            k = 0
            outstr = ''
            final_prob = []
            for c in short_best:
                if k < startNum:
                    if prob_best[k] > max_char1:
                        outstr += self.labels[c]
                        final_prob.append(prob_best[k])
                elif k >= startNum and k < endNum:
                    if prob_best[k] > max_number:
                        outstr += self.labels[c]
                        final_prob.append(prob_best[k])
                elif k >= endNum:
                    if prob_best[k] > max_char2:
                        outstr += self.labels[c]
                        final_prob.append(prob_best[k])
                k += 1
            ret.append(outstr)

        return ret, final_prob

    def setFirst(self, first):
        self.first_lp = first
        
    def predict(self, image):

        image_h, image_w, _ = image.shape
        if (image_h == 0 or image_w == 0):
            return '', 0.0
        image = cv2.resize(image, (self.input_size, 48),interpolation=cv2.INTER_CUBIC)
        image = np.transpose(image, (1, 0, 2))
        image = image.astype(np.float32)
        image = image / 127.5 - 1.

        input_image = np.expand_dims(image, 0)        
        intermediate_output = self.intermediate_layer_model.predict(input_image)        

        self.output_lp = intermediate_output
            
        pred_texts, prob_texts = self.decode_batch(self.output_lp)

        return pred_texts, prob_texts


def ocr(input_dir):
    cwd = os.getcwd()
    labels = ['0','1','2','3','4','5','6','7','8','9','A','B','C','D','E','F','G','H','I','J','K','L','M','N','O','P','Q','R','S','T','U','V','W','X','Y','Z']

    ocr_net = OCR(labels=labels)
    ocr_net.load_weights(cwd + '/python/wings_net/WingsNet_char.h5')
    imgs_paths = glob('%s/*lp.png' % input_dir)
    for i,img_path in enumerate(imgs_paths):
        print("IMG_PATH ", img_path)
        predicted_text, prob = ocr_net.predict(cv2.imread(img_path))
        filename = img_path.split("/")[6].split(".")[0] + "_ocr.txt"

        filepath = os.path.join(cwd + '/output', filename)
        f = open(filepath, "w")
        f.write(predicted_text[0])

        print('PREDICTED TEXT =', predicted_text)
        print('PROB =', prob)

    return predicted_text