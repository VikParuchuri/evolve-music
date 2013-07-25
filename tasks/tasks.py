from __future__ import division
from itertools import chain
import numpy as np
import pandas as pd
import re
import collections
import math
from percept.tasks.base import Task
from percept.fields.base import Complex, List, Dict, Float
from inputs.inputs import MusicFormats
from percept.utils.models import RegistryCategories, get_namespace
from percept.conf.base import settings
import os
from percept.tasks.train import Train
from sklearn.ensemble import RandomForestClassifier
import pickle
import random
import sqlite3
from pandas.io import sql
from collections import namedtuple
from scikits.audiolab import oggread
import random
import math
import operator
from scipy.fftpack import dct
from scikits.audiolab import oggwrite
from time import gmtime, strftime

import logging
log = logging.getLogger(__name__)

MAX_FEATURES = 500
DISTANCE_MIN=1
CHARACTER_DISTANCE_MIN = .2
RESET_SCENE_EVERY = 5

def make_df(datalist, labels, name_prefix=""):
    df = pd.DataFrame(datalist).T
    if name_prefix!="":
        labels = [name_prefix + "_" + l for l in labels]
    labels = [l.replace(" ", "_").lower() for l in labels]
    df.columns = labels
    df.index = range(df.shape[0])
    return df

def read_sound(fpath):
    try:
        data, fs, enc = oggread(fpath)
        upto = fs* settings.MUSIC_TIME_LIMIT
    except IOError:
        log.exception("Could not read file")
        raise IOError
    if data.shape[0]<upto:
        log.error("Music file not long enough.")
        raise ValueError
    try:
        if data.shape[1]!=2:
            log.error("Invalid dimension count. Do you have left and right channel audio?")
            raise ValueError
    except Exception:
        log.error("Invalid dimension count. Do you have left and right channel audio?")
        raise ValueError
    data = data[0:upto,:]
    return data, fs, enc

class ProcessMusic(Task):
    data = Complex()

    data_format = MusicFormats.dataframe

    category = RegistryCategories.preprocessors
    namespace = get_namespace(__module__)

    help_text = "Process sports events."

    def train(self, data, target, **kwargs):
        """
        Used in the training phase.  Override.
        """
        self.data = self.predict(data, **kwargs)

    def predict(self, data, **kwargs):
        """
        Used in the predict phase, after training.  Override
        """
        d = []
        labels = []
        encs = []
        fss = []
        fnames = []
        if not os.path.isfile(settings.FEATURE_PATH):
            for (z,p) in enumerate(data):
                log.info("On file {0}".format(z))
                try:
                    data , fs, enc = read_sound(p['newpath'])
                except Exception:
                    continue
                try:
                    features = process_song(data,fs)
                except Exception:
                    log.exception("Could not get features")
                    continue
                d.append(features)
                labels.append(p['type'])
                fss.append(fs)
                encs.append(enc)
                fnames.append(p['newpath'])
            frame = pd.DataFrame(d)
            frame['labels']  = labels
            frame['fs'] = fss
            frame['enc'] = encs
            frame['fname'] = fnames
            label_dict = {
                'classical' : 1,
                'electronic' : 0
            }
            frame['label_code'] = [label_dict[i] for i in frame['labels']]
            frame.to_csv(settings.FEATURE_PATH)
        else:
            frame = pd.read_csv(settings.FEATURE_PATH)

        return frame

def calc_slope(x,y):
    x_mean = np.mean(x)
    y_mean = np.mean(y)
    x_dev = np.sum(np.abs(np.subtract(x,x_mean)))
    y_dev = np.sum(np.abs(np.subtract(y,y_mean)))

    slope = (x_dev*y_dev)/(x_dev*x_dev)
    return slope

def get_indicators(vec):
    mean = np.mean(vec)
    slope = calc_slope(np.arange(len(vec)),vec)
    std = np.std(vec)
    return mean, slope, std

def calc_u(vec):
    fft = np.fft.fft(vec)
    return np.sum(np.multiply(fft,vec))/np.sum(vec)

def calc_mfcc(fft):
    ps = np.abs(fft) ** 2
    fs = np.dot(ps, mel_filter(ps.shape[0]))
    ls = np.log(fs)
    ds = dct(ls, type=2)
    return ds

def mel_filter(blockSize):
    numBands = 13
    maxMel = int(freqToMel(24000))
    minMel = int(freqToMel(10))

    filterMatrix = np.zeros((numBands, blockSize))

    melRange = np.array(xrange(numBands + 2))

    melCenterFilters = melRange * (maxMel - minMel) / (numBands + 1) + minMel

    aux = np.log(1 + 1000.0 / 700.0) / 1000.0
    aux = (np.exp(melCenterFilters * aux) - 1) / 22050
    aux = 0.5 + 700 * blockSize * aux
    aux = np.floor(aux)  # Arredonda pra baixo
    centerIndex = np.array(aux, int)  # Get int values

    for i in xrange(numBands):
        start, center, end = centerIndex[i:(i+3)]
        k1 = np.float32(center - start)
        k2 = np.float32(end - center)
        up = (np.array(xrange(start, center)) - start) / k1
        down = (end - np.array(xrange(center, end))) / k2

        filterMatrix[i][start:center] = up
        try:
            filterMatrix[i][center:end] = down
        except ValueError:
            pass

    return filterMatrix.transpose()

def freqToMel(freq):
    return 1127.01048 * math.log(1 + freq / 700.0)

def melToFreq(freq):
    return 700 * (math.exp(freq / 1127.01048 - 1))

def calc_features(vec,freq):
    #bin count
    bc = settings.MUSIC_TIME_LIMIT
    bincount = list(range(bc))
    #framesize
    fsize = 512
    #mean
    m = np.mean(vec)
    #spectral flux
    sf = np.mean(vec-np.roll(vec,fsize))
    mx = np.max(vec)
    mi = np.min(vec)
    sdev = np.std(vec)
    binwidth = len(vec)/bc
    bins = []

    for i in xrange(0,bc):
        bins.append(vec[(i*binwidth):(binwidth*i + binwidth)])
    peaks = [np.max(i) for i in bins]
    mins = [np.min(i) for i in bins]
    amin,smin,stmin = get_indicators(mins)
    apeak, speak, stpeak = get_indicators(peaks)
    #fft = np.fft.fft(vec)
    bin_fft = []
    for i in xrange(0,bc):
        bin_fft.append(np.fft.fft(vec[(i*binwidth):(binwidth*i + binwidth)]))

    mel = [list(calc_mfcc(j)) for (i,j) in enumerate(bin_fft) if i%3==0]
    mels = list(chain.from_iterable(mel))

    cepstrums = [np.fft.ifft(np.log(np.abs(i))) for i in bin_fft]
    inter = [get_indicators(i) for i in cepstrums]
    acep,scep, stcep = get_indicators([i[0] for i in inter])
    aacep,sscep, stsscep = get_indicators([i[1] for i in inter])

    zero_crossings = np.where(np.diff(np.sign(vec)))[0]
    zcc = len(zero_crossings)
    zccn = zcc/freq

    u = [calc_u(i) for i in bins]
    spread = np.sqrt(u[-1] - u[0]**2)
    skewness = (u[0]**3 - 3*u[0]*u[5] + u[-1])/spread**3

    #Spectral slope
    #ss = calc_slope(np.arange(len(fft)),fft)
    avss = [calc_slope(np.arange(len(i)),i) for i in bin_fft]
    savss = calc_slope(bincount,avss)
    mavss = np.mean(avss)

    features = [m,sf,mx,mi,sdev,amin,smin,stmin,apeak,speak,stpeak,acep,scep,stcep,aacep,sscep,stsscep,zcc,zccn,spread,skewness,savss,mavss] + mels + [i[0] for (j,i) in enumerate(inter) if j%5==0]

    for i in xrange(0,len(features)):
        try:
            features[i] = features[i].real
        except Exception:
            pass
    return features

def extract_features(sample,freq):
    left = calc_features(sample[:,0],freq)
    right = calc_features(sample[:,1],freq)
    return left+right

def process_song(vec,f):
    try:
        features = extract_features(vec,f)
    except Exception:
        log.exception("Cannot generate features")
        return None

    return features

class RandomForestTrain(Train):
    """
    A class to train a random forest
    """
    colnames = List()
    clf = Complex()
    category = RegistryCategories.algorithms
    namespace = get_namespace(__module__)
    algorithm = RandomForestClassifier
    args = {'n_estimators' : 300, 'min_samples_leaf' : 4, 'compute_importances' : True}

    help_text = "Train and predict with Random Forest."

class EvolveMusic(Task):
    data = Complex()
    clf = Complex()
    importances = Complex()

    data_format = MusicFormats.dataframe

    category = RegistryCategories.preprocessors
    namespace = get_namespace(__module__)
    args = {
        'non_predictors' : ["labels","label_code","fs","enc","fname"],
        'target' : 'label_code'
    }

    def train(self,data,**kwargs):
        non_predictors = kwargs.get('non_predictors')
        target = kwargs.get('target')

        data.index = range(data.shape[0])

        alg = RandomForestTrain()
        good_names = [i for i in data.columns if i not in non_predictors]
        for c in good_names:
            data[c] = data[c].astype(float)

        for c in good_names:
            data[c] = data[c].real

        clf = alg.train(np.asarray(data[good_names]),data[target],**alg.args)
        importances = clf.feature_importances_

        for i in xrange(0,data.shape[0]):
            fname = data['fname'][i]
            vec, fs, enc = read_sound(fname)
            label = data["labels"][i]
            if label=="classical":
                name = fname.split("/")[-1]
                feats = process_song(vec,fs)
                initial_quality = clf.predict_proba(feats)[0,1]
                if initial_quality<.5:
                    final_quality = initial_quality + .1
                else:
                    final_quality = initial_quality - .1
                for z in xrange(0,1000):
                    if z%100==0 or z==0:
                        print("Song {0} iteration {1}".format(i,z))
                        v2ind = random.randint(0,data.shape[0]-1)
                        v2fname = data['fname'][v2ind]
                        vec2, v2fs, v2enc = read_sound(v2fname)
                        feats = process_song(vec,fs)
                        quality = clf.predict_proba(feats)[0,1]
                        #nearest_match, min_dist = find_nearest_match(feats, data[good_names])
                        if abs(quality-final_quality)<=.1 and z!=0:
                            time = strftime("%m-%d-%Y-%H%M%S", gmtime())
                            fname = time+name
                            dir_path = settings.MUSIC_STORE_PATH
                            if not os.path.isdir(dir_path):
                                os.mkdir(dir_path)
                            fpath = os.path.abspath(os.path.join(dir_path,fname))
                            oggwrite(vec,fpath,fs,enc)
                            break
                    vec = alter(vec,vec2)

def random_effect(vec,func):
    vec_len = len(vec)
    for i in xrange(1,10):
        randint = random.randint(0,vec_len)
        effect_area = math.floor((random.random()+.3)*100)
        if randint + vec_len/effect_area > vec_len:
            randint = vec_len - vec_len/effect_area
        vec[randint:(randint+vec_len/effect_area)]= func(vec[randint:(randint+vec_len/effect_area)],random.random())
    return vec

def subtract_random(vec):
    return random_effect(vec,operator.sub)

def multiply_random(vec):
    return random_effect(vec,operator.mul)

def add_random(vec):
    return random_effect(vec,operator.add)

def mix_random(vec,vec2):
    vec_len = len(vec)
    for i in xrange(1,10):
        randint = random.randint(0,vec_len)
        effect_area = math.floor((random.random()+.3)*100)
        if randint + vec_len/effect_area > vec_len:
            randint = vec_len - vec_len/effect_area
        vec[randint:(randint+vec_len/effect_area)]+=vec2[randint:(randint+vec_len/effect_area)]
    return vec

def alter(vec,vec2):
    op = random.randint(0,3)
    if op==0:
        vec = subtract_random(vec)
    elif op==1:
        vec = add_random(vec)
    elif op==2:
        #vec = multiply_random(vec)
        pass
    else:
        vec = mix_random(vec,vec2)
    return vec

def find_nearest_match(features, matrix):
    features = np.asarray(features)
    distances = [euclidean(u, features) for u in matrix]
    nearest_match = distances.index(min(distances))
    return nearest_match, min(distances)

def euclidean(v1, v2):
    return np.sqrt(np.sum(np.square(np.subtract(v1,v2))))
