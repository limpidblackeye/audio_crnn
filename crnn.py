
import sys
# import cPickle
import Pickle
import numpy as np
import argparse
import glob
import time
import os

import keras
from keras import backend as K
from keras.models import Sequential,Model, load_model
from keras.layers.core import Dense, Dropout, Activation, Flatten, Reshape, Permute,Lambda, RepeatVector
from keras.layers.convolutional import ZeroPadding2D, AveragePooling2D, Conv2D,MaxPooling2D, Convolution1D,MaxPooling1D
from keras.layers.pooling import GlobalMaxPooling2D
from keras.layers import Merge, Input, merge
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, LearningRateScheduler
from keras.layers import LSTM, SimpleRNN, GRU, TimeDistributed, Bidirectional
from keras.layers.normalization import BatchNormalization
import h5py
from keras.layers.merge import Multiply
from sklearn import preprocessing
import random

import config as cfg
from prepare_data import create_folder, load_hdf5_data, do_scale
from data_generator import RatioDataGenerator
from evaluation import io_task4, evaluate
np.random.seed(1001)

###====================================================###
###=================== data load ======================###
###====================================================###

train = pd.read_csv("../../data/train.csv")
test = pd.read_csv("./sample_submission.csv")
train.head()
print("Number of training examples=", train.shape[0], "  Number of classes=", len(train.label.unique()))
print(train.label.unique())

LABELS = list(train.label.unique())
label_idx = {label: i for i, label in enumerate(LABELS)}
train.set_index("fname", inplace=True)
test.set_index("fname", inplace=True)
train["label_idx"] = train.label.apply(lambda x: label_idx[x])

###====================================================###
###================== prepare data ====================###
###====================================================###

def prepare_data(df, config, data_dir):
    X = np.empty(shape=(df.shape[0], config.dim[0], config.dim[1], 1))
    input_length = config.audio_length
    for i, fname in enumerate(df.index):
        # print(fname)
        if ".wav" in str(fname):
            file_path = data_dir + fname
            data, _ = librosa.core.load(file_path, sr=config.sampling_rate, res_type="kaiser_fast")

            # Random offset / Padding
            if len(data) > input_length:
                max_offset = len(data) - input_length
                offset = np.random.randint(max_offset)
                data = data[offset:(input_length+offset)]
            else:
                if input_length > len(data):
                    max_offset = input_length - len(data)
                    offset = np.random.randint(max_offset)
                else:
                    offset = 0
                data = np.pad(data, (offset, input_length - len(data) - offset), "constant")

            data = librosa.feature.mfcc(data, sr=config.sampling_rate, n_mfcc=config.n_mfcc)
            data = np.expand_dims(data, axis=-1)
            X[i,] = data
    return X

print("start preparing data ...")
X_train = prepare_data(train, config, '../../data/audio_train/')
X_test = prepare_data(test, config, '../../data/audio_test/')
y_train = to_categorical(train.label_idx, num_classes=config.n_classes)

# #### Normalization
mean = np.mean(X_train, axis=0)
std = np.std(X_train, axis=0)

X_train = (X_train - mean)/std
X_test = (X_test - mean)/std

###
def audio_norm(data):
    max_data = np.max(data)
    min_data = np.min(data)
    data = (data-min_data)/(max_data-min_data+1e-6)
    return data-0.5


###====================================================###
###==================== Conv-GLU ======================###
###====================================================###

# CNN with Gated linear unit (GLU) block
def block(input):
    cnn = Conv2D(128, (3, 3), padding="same", activation="linear", use_bias=False)(input)
    cnn = BatchNormalization(axis=-1)(cnn)

    cnn1 = Lambda(slice1, output_shape=slice1_output_shape)(cnn)
    cnn2 = Lambda(slice2, output_shape=slice2_output_shape)(cnn)

    cnn1 = Activation('linear')(cnn1)
    cnn2 = Activation('sigmoid')(cnn2)

    out = Multiply()([cnn1, cnn2])
    return out

def slice1(x):
    return x[:, :, :, 0:64]

def slice2(x):
    return x[:, :, :, 64:128]

def slice1_output_shape(input_shape):
    return tuple([input_shape[0],input_shape[1],input_shape[2],64])

def slice2_output_shape(input_shape):
    return tuple([input_shape[0],input_shape[1],input_shape[2],64])

# Attention weighted sum
def outfunc(vects):
    cla, att = vects    # (N, n_time, n_out), (N, n_time, n_out)
    att = K.clip(att, 1e-7, 1.)
    out = K.sum(cla * att, axis=1) / K.sum(att, axis=1)     # (N, n_out)
    return out

# Train model
def train(args):
    num_classes = cfg.num_classes
    
    # Load training & testing data
    (tr_x, tr_y, tr_na_list) = load_hdf5_data(args.tr_hdf5_path, verbose=1)
    (te_x, te_y, te_na_list) = load_hdf5_data(args.te_hdf5_path, verbose=1)
    print("tr_x.shape: %s" % (tr_x.shape,))

    # Scale data
    tr_x = do_scale(tr_x, args.scaler_path, verbose=1)
    te_x = do_scale(te_x, args.scaler_path, verbose=1)
    
    # Build model
    (_, n_time, n_freq) = tr_x.shape    # (N, 240, 64)
    input_logmel = Input(shape=(n_time, n_freq), name='in_layer')   # (N, 240, 64)
    a1 = Reshape((n_time, n_freq, 1))(input_logmel) # (N, 240, 64, 1)
    
    a1 = block(a1)
    a1 = block(a1)
    a1 = MaxPooling2D(pool_size=(1, 2))(a1) # (N, 240, 32, 128)
    
    a1 = block(a1)
    a1 = block(a1)
    a1 = MaxPooling2D(pool_size=(1, 2))(a1) # (N, 240, 16, 128)
    
    a1 = block(a1)
    a1 = block(a1)
    a1 = MaxPooling2D(pool_size=(1, 2))(a1) # (N, 240, 8, 128)
    
    a1 = block(a1)
    a1 = block(a1)
    a1 = MaxPooling2D(pool_size=(1, 2))(a1) # (N, 240, 4, 128)
    
    a1 = Conv2D(256, (3, 3), padding="same", activation="relu", use_bias=True)(a1)
    a1 = MaxPooling2D(pool_size=(1, 4))(a1) # (N, 240, 1, 256)
    
    a1 = Reshape((240, 256))(a1) # (N, 240, 256)
    
    # Gated BGRU
    rnnout = Bidirectional(GRU(128, activation='linear', return_sequences=True))(a1)
    rnnout_gate = Bidirectional(GRU(128, activation='sigmoid', return_sequences=True))(a1)
    a2 = Multiply()([rnnout, rnnout_gate])
    
    # Attention
    cla = TimeDistributed(Dense(num_classes, activation='sigmoid'), name='localization_layer')(a2)
    att = TimeDistributed(Dense(num_classes, activation='softmax'))(a2)
    out = Lambda(outfunc, output_shape=(num_classes,))([cla, att])
    
    model = Model(input_logmel, out)
    model.summary()
    
    # Compile model
    model.compile(loss='binary_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])
    return model

    # Save model callback
    filepath = os.path.join(args.out_model_dir, "gatedAct_rationBal44_lr0.001_normalization_at_cnnRNN_64newMel_240fr.{epoch:02d}-{val_acc:.4f}.hdf5")
    create_folder(os.path.dirname(filepath))
    save_model = ModelCheckpoint(filepath=filepath,
                                 monitor='val_acc', 
                                 verbose=0,
                                 save_best_only=False,
                                 save_weights_only=False,
                                 mode='auto',
                                 period=1)  

    # Data generator
    gen = RatioDataGenerator(batch_size=44, type='train')

    # Train
    model.fit_generator(generator=gen.generate([tr_x], [tr_y]), 
                        steps_per_epoch=100,    # 100 iters is called an 'epoch'
                        epochs=31,              # Maximum 'epoch' to train
                        verbose=1, 
                        callbacks=[save_model], 
                        validation_data=(te_x, te_y))

'''
# Run function in mini-batch to save memory. 
def run_func(func, x, batch_size):
    pred_all = []
    batch_num = int(np.ceil(len(x) / float(batch_size)))
    for i1 in xrange(batch_num):
        batch_x = x[batch_size * i1 : batch_size * (i1 + 1)]
        [preds] = func([batch_x, 0.])
        pred_all.append(preds)
    pred_all = np.concatenate(pred_all, axis=0)
    return pred_all
'''

# Recognize and write probabilites. 
def recognize(args, at_bool):
    (te_x, te_y, te_na_list) = load_hdf5_data(args.te_hdf5_path, verbose=1)
    x = te_x
    y = te_y
    na_list = te_na_list
    
    x = do_scale(x, args.scaler_path, verbose=1)
    
    fusion_at_list = []
    # fusion_sed_list = []
    for epoch in range(20, 30, 1):
        t1 = time.time()
        [model_path] = glob.glob(os.path.join(args.model_dir, 
            "*.%02d-0.*.hdf5" % epoch))
        model = load_model(model_path)
        
        # Audio tagging
        if at_bool:
            pred = model.predict(x)
            fusion_at_list.append(pred)

###################################################################
        
        # # Sound event detection
        # if sed_bool:
        #     in_layer = model.get_layer('in_layer')
        #     loc_layer = model.get_layer('localization_layer')
        #     func = K.function([in_layer.input, K.learning_phase()], 
        #                       [loc_layer.output])
        #     pred3d = run_func(func, x, batch_size=20)
        #     fusion_sed_list.append(pred3d)
        
        print("Prediction time: %s" % (time.time() - t1,))
    
    # Write out AT probabilities
    if at_bool:
        fusion_at = np.mean(np.array(fusion_at_list), axis=0)
        print("AT shape: %s" % (fusion_at.shape,))
        io_task4.at_write_prob_mat_to_csv(
            na_list=na_list, 
            prob_mat=fusion_at, 
            out_path=os.path.join(args.out_dir, "at_prob_mat.csv.gz"))
    
    # # Write out SED probabilites
    # if sed_bool:
    #     fusion_sed = np.mean(np.array(fusion_sed_list), axis=0)
    #     print("SED shape:%s" % (fusion_sed.shape,))
    #     io_task4.sed_write_prob_mat_list_to_csv(
    #         na_list=na_list, 
    #         prob_mat_list=fusion_sed, 
    #         out_path=os.path.join(args.out_dir, "sed_prob_mat_list.csv.gz"))
            
    print("Prediction finished!")

'''
# Get stats from probabilites. 
def get_stat(args, at_bool, sed_bool):
    lbs = cfg.lbs
    step_time_in_sec = cfg.step_time_in_sec
    max_len = cfg.max_len
    thres_ary = [0.3] * len(lbs)

    # Calculate AT stat
    if at_bool:
        pd_prob_mat_csv_path = os.path.join(args.pred_dir, "at_prob_mat.csv.gz")
        at_stat_path = os.path.join(args.stat_dir, "at_stat.csv")
        at_submission_path = os.path.join(args.submission_dir, "at_submission.csv")
        
        at_evaluator = evaluate.AudioTaggingEvaluate(
            weak_gt_csv="meta_data/groundtruth_weak_label_testing_set.csv", 
            lbs=lbs)
        
        at_stat = at_evaluator.get_stats_from_prob_mat_csv(
                        pd_prob_mat_csv=pd_prob_mat_csv_path, 
                        thres_ary=thres_ary)
                        
        # Write out & print AT stat
        at_evaluator.write_stat_to_csv(stat=at_stat, 
                                       stat_path=at_stat_path)
        at_evaluator.print_stat(stat_path=at_stat_path)
        
        # Write AT to submission format
        io_task4.at_write_prob_mat_csv_to_submission_csv(
            at_prob_mat_path=pd_prob_mat_csv_path, 
            lbs=lbs, 
            thres_ary=at_stat['thres_ary'], 
            out_path=at_submission_path)
               
    # Calculate SED stat
    if sed_bool:
        sed_prob_mat_list_path = os.path.join(args.pred_dir, "sed_prob_mat_list.csv.gz")
        sed_stat_path = os.path.join(args.stat_dir, "sed_stat.csv")
        sed_submission_path = os.path.join(args.submission_dir, "sed_submission.csv")
        
        sed_evaluator = evaluate.SoundEventDetectionEvaluate(
            strong_gt_csv="meta_data/groundtruth_strong_label_testing_set.csv", 
            lbs=lbs,
            step_sec=step_time_in_sec,
            max_len=max_len)
                            
        # Write out & print SED stat
        sed_stat = sed_evaluator.get_stats_from_prob_mat_list_csv(
                    pd_prob_mat_list_csv=sed_prob_mat_list_path, 
                    thres_ary=thres_ary)
                    
        # Write SED to submission format
        sed_evaluator.write_stat_to_csv(stat=sed_stat, 
                                        stat_path=sed_stat_path)                     
        sed_evaluator.print_stat(stat_path=sed_stat_path)
        
        # Write SED to submission format
        io_task4.sed_write_prob_mat_list_csv_to_submission_csv(
            sed_prob_mat_list_path=sed_prob_mat_list_path, 
            lbs=lbs, 
            thres_ary=thres_ary, 
            step_sec=step_time_in_sec, 
            out_path=sed_submission_path)
                                                        
    print("Calculating stat finished!")
'''

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="")
    subparsers = parser.add_subparsers(dest='mode')
    
    parser_train = subparsers.add_parser('train')
    parser_train.add_argument('--tr_hdf5_path', type=str)
    parser_train.add_argument('--te_hdf5_path', type=str)
    parser_train.add_argument('--scaler_path', type=str)
    parser_train.add_argument('--out_model_dir', type=str)
    
    parser_recognize = subparsers.add_parser('recognize')
    parser_recognize.add_argument('--te_hdf5_path', type=str)
    parser_recognize.add_argument('--scaler_path', type=str)
    parser_recognize.add_argument('--model_dir', type=str)
    parser_recognize.add_argument('--out_dir', type=str)
    
    # parser_get_stat = subparsers.add_parser('get_stat')
    # parser_get_stat.add_argument('--pred_dir', type=str)
    # parser_get_stat.add_argument('--stat_dir', type=str)
    # parser_get_stat.add_argument('--submission_dir', type=str)
    
    args = parser.parse_args()
    
    if args.mode == 'train':
        train(args)
    elif args.mode == 'recognize':
        recognize(args, at_bool=True)
    # elif args.mode == 'get_stat':
    #     get_stat(args, at_bool=True, sed_bool=True)
    else:
        raise Exception("Incorrect argument!")