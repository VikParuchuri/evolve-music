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

row_types = ["id","info","start","play","sub","data"]

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
        for p in data:
            data, fs, enc = oggread(p['newpath'])
            upto = fs* settings.MUSIC_TIME_LIMIT
            if data.shape[0]<upto:
                continue
            try:
                if data.shape[1]!=2:
                    continue
            except Exception:
                log.error("Invalid dimension count. Do you have left and right channel audio?")
                continue
            data = data[0:upto,:]
            try:
                features = process_song(data,fs)
            except Exception:
                log.exception("Could not get features")
                continue
            d.append(features)
            labels.append(p['type'])
        frame = pd.DataFrame(d)
        frame['labels']  = labels
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

def calc_features(vec,freq):
    #bin count
    bc = settings.MUSIC_TIME_LIMIT * 3
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

    return [m,sf,mx,mi,sdev,amin,smin,stmin,apeak,speak,stpeak,acep,scep,stcep,aacep,sscep,stsscep,zcc,zccn,spread,skewness,savss,mavss,[i[0] for (j,i) in enumerate(inter) if j%5==0]]

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

    data_format = MusicFormats.dataframe

    category = RegistryCategories.preprocessors
    namespace = get_namespace(__module__)
    args = {
        'non_predictors' : ["labels"]
    }

    def train(self,data,**kwargs):
        non_predictors = kwargs.get('non_predictors')
        alg = RandomForestTrain()
        good_names = [i for i in data.columns if i not in non_predictors]

        clf = alg.train(data[good_names],data['labels'])

        

    def random_effect(self,vec,func):
        vec_len = len(vec)
        for i in xrange(1,10):
            randint = random.randint(0,vec_len)
            effect_area = math.floor((random.random()+.3)*100)
            if randint + vec_len/effect_area > vec_len:
                randint = vec_len - vec_len/effect_area
            vec[randint:(randint+vec_len/effect_area)]= func(vec[randint:(randint+vec_len/effect_area)],random.random())
        return vec

    def subtract_random(self,vec):
        return self.random_effect(vec,operator.sub)

    def multiply_random(self,vec):
        return self.random_effect(vec,operator.mul)

    def add_random(self,vec):
        return self.random_effect(vec,operator.add)

    def mix_random(self,vec,d):
        #Mix two songs randomly
        pass

    def alter(self,vec):
        op = random.randint(0,4)
        if op==0:
            vec = self.subtract_random(vec)
        elif op==1:
            vec = self.add_random(vec)
        elif op==2:
            vec = self.multiply_random(vec)
        return vec
