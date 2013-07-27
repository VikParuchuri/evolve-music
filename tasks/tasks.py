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
import random
import math
import operator
from scipy.fftpack import dct
from scikits.audiolab import oggwrite, play, oggread
from time import gmtime, strftime
import subprocess
from midiutil.MidiFile import MIDIFile
import midi

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

def read_sound(fpath, limit = settings.MUSIC_TIME_LIMIT):
    try:
        data, fs, enc = oggread(fpath)
        upto = fs* limit
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
        'non_predictors' : ["labels","label_code","fs","enc","fname","Unnamed: 0"],
        'target_var' : 'label_code',
    }

    def train(self,data,target, **kwargs):
        non_predictors = kwargs.get('non_predictors')
        target = kwargs.get('target_var')

        data.index = range(data.shape[0])

        alg = RandomForestTrain()
        good_names = [i for i in data.columns if i not in non_predictors]
        for c in good_names:
            data[c] = data[c].astype(float)

        for c in good_names:
            data[c] = data[c].real

        clf = alg.train(np.asarray(data[good_names]),data[target],**alg.args)
        importances = clf.feature_importances_

        counter = 0
        for i in xrange(0,data.shape[0]):
            fname = data['fname'][i]
            vec, fs, enc = read_sound(fname)
            label = data["labels"][i]
            if counter>10:
                break
            if label=="classical":
                counter+=1
                name = fname.split("/")[-1]
                feats = process_song(vec,fs)
                initial_quality = clf.predict_proba(feats)[0,1]
                headers = "song_index,iteration,quality,distance,splice_song_index,splice_song"
                v2s = [headers,"{0},{1},{2},{3},{4},{5}".format(i,-1,initial_quality,0,0,"N/A")]
                print(headers)
                for z in xrange(0,10):
                    if z%10==0 or z==0:
                        v2ind = random.randint(0,data.shape[0]-1)
                        v2fname = data['fname'][v2ind]
                        vec2, v2fs, v2enc = read_sound(v2fname)
                        feats = process_song(vec,fs)
                        quality = clf.predict_proba(feats)[0,1]
                        nearest_match, min_dist = find_nearest_match(feats, data[good_names])
                        descriptor = "{0},{1},{2},{3},{4},{5}".format(i,z,quality,min_dist,v2ind,v2fname.split("/")[-1])
                        v2s.append(descriptor)
                        print(descriptor)
                        if min_dist>.35 and (abs(quality-0)<=.1 or abs(1-quality)<=.1) and z!=0:
                            write_file(name,vec,fs,enc,v2s)
                    vec = alter(vec,vec2,fs,v2fs,clf)
                write_file(name,vec,fs,enc,v2s)

def open_song(i,data):
    fname = data['fname'][i]
    d, fs, enc = read_sound(fname)
    return d

def write_file(name,vec,fs,enc,v2s):
    time = strftime("%m-%d-%Y-%H%M%S", gmtime())
    fname = time+name
    dir_path = settings.MUSIC_STORE_PATH
    if not os.path.isdir(dir_path):
        os.mkdir(dir_path)
    fpath = os.path.abspath(os.path.join(dir_path,fname))
    oggwrite(vec,fpath,fs,enc)
    desc_path = os.path.abspath(os.path.join(dir_path,time + "desc.txt"))
    dfile = open(desc_path,'wb')
    for item in v2s:
        dfile.write("{0}\n".format(item))

def random_effect(vec,func):
    vec_len = len(vec)
    for i in xrange(1,10):
        randint = random.randint(0,vec_len)
        effect_area = math.floor((random.random()+.3)*100)
        if randint + vec_len/effect_area > vec_len:
            randint = vec_len - vec_len/effect_area
        vec[randint:(randint+vec_len/effect_area)]= func(vec[randint:(randint+vec_len/effect_area)],random.random())
    return vec

def mix_random(vec,vec2):
    vec_len = len(vec)
    for i in xrange(1,10):
        randint = random.randint(0,vec_len)
        effect_area = math.floor((random.random()+.3)*100)
        if randint + vec_len/effect_area > vec_len:
            randint = vec_len - vec_len/effect_area
        vec[randint:(randint+vec_len/effect_area)]+=vec2[randint:(randint+vec_len/effect_area)]
    return vec

def extract_note(vec,fs,clf):
    quality = .5
    best_note = None
    best_quality = .5
    counter=0
    note = None
    timeslice = 1
    while abs(quality-round(quality,0))>.1 and counter<30:
        counter+=1
        note_start = random.randint(0,len(vec)-1)
        if note_start + timeslice*fs > len(vec)-1:
            note_start = len(vec) - 1 - timeslice*fs
        note = vec[note_start:(timeslice*fs + note_start)]
        quality = find_quality(note,fs,clf)
        if quality<best_quality:
            best_note = note
            best_quality = quality
    if note is None:
        note = best_note
    return note

def find_quality(vec,fs,clf):
    feats = extract_features(vec,fs)
    quality = clf.predict_proba(feats)[0,1]
    return quality

def splice(vec,vec2,fs1,fs2,clf):
    note = extract_note(vec2,fs2,clf)
    vec_len = len(vec)
    insertions = 10
    insertion_gap = math.floor(vec_len/insertions)
    insert_point = random.randint(0,insertion_gap-1)
    for i in xrange(0,insertions):
        if int(insert_point + len(note)) < len(vec):
            vec[insert_point:int(insert_point + len(note))]=note
        insert_point+=insertion_gap
    return vec

def alter(vec,vec2,fs1,fs2,clf):
    vec = splice(vec,vec2,fs1,fs2,clf)
    return vec

def find_nearest_match(features, matrix):
    features = np.asarray(features)
    matrix = np.asarray(matrix)
    distances = [euclidean(u, features) for u in matrix]
    nearest_match = distances.index(min(distances))
    return nearest_match, min(distances)/len(features)

def euclidean(v1, v2):
    return np.sqrt(np.sum(np.square(np.subtract(v1,v2)/(v2+.1))))


def get_matrix():
    non_predictors = ["labels","label_code","fs","enc","fname","Unnamed: 0"]
    frame = pd.read_csv(settings.FEATURE_PATH)
    good_names = [i for i in frame.columns if i not in non_predictors]
    data = frame[good_names]
    data['labels'] = frame['labels']
    folder_path = os.path.abspath(os.path.join(settings.DATA_PATH,"highlights"))
    ogg_files = [os.path.abspath(os.path.join(folder_path,f)) for f in os.listdir(folder_path) if f.endswith(".ogg")]
    d = []
    for o in ogg_files:
        try:
            data , fs, enc = read_sound(o)
        except Exception:
            continue
        try:
            features = process_song(data,fs)
        except Exception:
            log.exception("Could not get features")
            continue
        d.append(features)
    gframe = pd.DataFrame(d)
    gframe['labels']  = ["generated" for i in xrange(0,gframe.shape[0])]
    gframe.columns = list(xrange(0,gframe.shape[1]))
    frame.columns = list(xrange(0,frame.shape[1]))
    full_frame = pd.concat([frame,gframe],axis=0)
    full_frame.to_csv(settings.VIZ_PATH)

def generate_note(mfile):
    generate_midi(mfile)
    ofile = convert_to_ogg(mfile)
    try:
        data, fs, enc = oggread(ofile)
    except Exception:
        log.exception("Could not read sound file.")
        raise IOError
    return data, fs, enc

def convert_to_ogg(mfile):
    file_end = mfile.split("/")[-1].split(".")[0]
    oggfile = file_end + ".ogg"
    wavfile = file_end + ".wav"
    oggpath = os.path.abspath(os.path.join(settings.MIDI_PATH,oggfile))
    wavpath = os.path.abspath(os.path.join(settings.MIDI_PATH,wavfile))
    if not os.path.isfile(oggpath):
        subprocess.call(['fluidsynth', '-i','-n', '-F', wavpath, settings.SOUNDFONT_PATH, mfile])
        subprocess.call(['oggenc', wavpath])
        os.remove(wavpath)
    return oggpath

def additive_transform(pitch,msign=1):
    multiplier = random.randint(1,3)*msign
    return [p+(i*multiplier) for (i,p) in enumerate(pitch)]

def generate_pitch(track_length):
    pitch = list(range(track_length))
    o = random.randint(0,4)
    if o==0:
        pitch = additive_transform(pitch)
    elif o==1:
        pitch = additive_transform(pitch,msign=-1)
    elif o==2:
        pitch = additive_transform(pitch)
        pitch.reverse()
    elif o==3:
        pitch = additive_transform(pitch,msign=-1)
        pitch.reverse()
    return pitch

def add_track(midi,track,length,time=0,pitch_min=0,pitch_max=127):
    tempo = random.randint(30,480)

    beats_per_second = tempo/60
    tempo = int(math.floor(beats_per_second)*60)
    track_length = int(length * math.floor(beats_per_second))
    midi.addTrackName(track,time,"Sample Track")
    midi.addTempo(track,time,tempo)
    pitches = generate_pitch(track_length)
    # Add a note. addNote expects the following information:
    for i in xrange(0,track_length):
        channel = 0
        pitch = pitches[i]
        time = i
        duration = 1
        volume = 100
        if pitch>pitch_max:
            pitch = pitch_max
        if pitch<pitch_min:
            pitch = pitch_min

        # Now add the note.
        midi.addNote(track,channel,pitch,time,duration,volume)
    return midi


def generate_midi(filename,length=60):
    track_count = random.randint(1,3)
    midi = MIDIFile(track_count)

    for i in xrange(0,track_count):
        midi = add_track(midi,i,length)

    binfile = open(filename, 'wb')
    midi.writeFile(binfile)
    binfile.close()


class ProcessMidi(Task):
    data = Complex()

    data_format = MusicFormats.dataframe

    category = RegistryCategories.preprocessors
    namespace = get_namespace(__module__)

    help_text = "Process midi files."

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
        if not os.path.isfile(settings.MIDI_FEATURE_PATH):
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
                'modern' : 0
            }
            frame['label_code'] = [label_dict[i] for i in frame['labels']]
            frame.to_csv(settings.MIDI_FEATURE_PATH)
        else:
            frame = pd.read_csv(settings.MIDI_FEATURE_PATH)

        return frame

def process_midifile(m,notes,tempos):
    for track in m:
        for e in track:
            if isinstance(e,midi.events.NoteOnEvent):
                channel = e.channel
                pitch, velocity = e.data
                tick = e.tick
                if channel not in notes:
                    notes[channel] = {'pitch' : [], 'velocity' : [] ,'tick' : []}
                notes[channel]['pitch'].append(pitch)
                notes[channel]['velocity'].append(velocity)
                notes[channel]['tick'].append(tick)
            elif isinstance(e,midi.events.SetTempoEvent):
                tick = e.tick
                tick = round(tick/10)*10
                mpqn = e.mpqn
                tempos['tick'].append(tick)
                tempos['mpqn'].append(mpqn)
        tempos['mpqn'].append(0)
        tempos['tick'].append(0)
    return notes, tempos

def generate_matrix(seq):
    seq = round(seq/5)*5
    unique_seq = list(set(seq))
    unique_seq.sort()
    mat = np.zeros(len(unique_seq),len(unique_seq))
    for i in xrange(0,len(seq)-1):
        i_ind = unique_seq.index(seq[i])
        i_1_ind = unique_seq.index(seq[i+1])
        mat[i_ind,i_1_ind] = mat[i_ind,i_1_ind] + 1
    for i in xrange(0,mat.shape[0]):
        mat[i,:] = mat[i,:]/np.sum(mat[i,:])
    return {'mat': mat, 'inds' : unique_seq}

def generate_matrices(notes,tempos):
    tm = {}
    nm = {}
    tm['tick'] = generate_matrix(tempos['tick'])
    tm['mpqn'] = generate_matrix(tempos['mpqn'])

    for k in notes:
        nm[k] = {}
        for sk in notes[k]:
            nm[k][sk] = generate_matrix(notes[k][sk])
    return nm, tm

def pick_proba(vec):
    choices = []
    for i in xrange(0,len(vec)):
        choices += [i] * int(math.floor(vec[i]*100))
    choice = random.choice(choices)
    return choice

def generate_markov_seq(m,inds,length):
    start = random.choice(inds)
    seq = []
    seq.append(start)
    for i in xrange(1,length):
        ind = inds.index(seq[i-1])
        sind = pick_proba(m[ind,:])
        seq.append(inds[sind])
    return seq

def generate_audio_track(notes,length):
    channel = random.choice(notes.keys())
    pitch = generate_markov_seq(notes[channel]['pitch']['mat'],notes[channel]['pitch']['inds'],length)
    velocity = generate_markov_seq(notes[channel]['velocity']['mat'],notes[channel]['velocity']['inds'],length)
    tick = generate_markov_seq(notes[channel]['tick']['mat'],notes[channel]['tick']['inds'],length)
    track = midi.Track()
    track.append(midi.TrackNameEvent())
    for i in xrange(0,length):
        on = midi.NoteOnEvent(channel=channel)
        on.set_pitch(pitch[i])
        on.set_velocity(velocity[i])
        on.tick = tick[i]
        track.append(on)
    track.append(midi.EndOfTrackEvent)
    return track

def generate_tempo_track(tempos,length):
    tick = generate_markov_seq(tempos['tick']['mat'],tempos['tick']['inds'],length)
    mpqn = generate_markov_seq(tempos['mpqn']['mat'],tempos['mpqn']['inds'],length)

    track = midi.Track()
    track.append(midi.TrackNameEvent())
    track.append(midi.TextMetaEvent())
    for i in xrange(0,length):
        te = midi.SetTempoEvent()
        te.tick = tick[i]
        te.set_mpqn(mpqn[i])
    track.append(midi.EndOfTrackEvent())
    return track

def evaluate_midi_quality(pattern,clf):
    midi_path = os.path.abspath(os.path.join(settings.MIDI_STORE_PATH,"tmp.mid"))
    midi.write_midifile(midi_path,pattern)
    oggpath = convert_to_ogg(midi_path)
    data, fs, enc = oggread(oggpath)
    features = process_song(data,fs)
    quality = clf.predict_proba(features)[0,1]
    return quality

def generate_pattern(tracks):
    pat = midi.Pattern(tracks=tracks)
    return pat

class GenerateTransitionMatrix(Task):
    data = Complex()

    data_format = MusicFormats.dataframe

    category = RegistryCategories.preprocessors
    namespace = get_namespace(__module__)

    help_text = "Process midi files."

    def train(self, data, target, **kwargs):
        """
        Used in the training phase.  Override.
        """
        self.data = self.predict(data, **kwargs)

    def predict(self, data, **kwargs):
        """
        Used in the predict phase, after training.  Override
        """
        tempos = {'tick' : [], 'mpqn' : []}
        notes = {}
        for (z,p) in enumerate(data):
            log.info("On file {0}".format(z))
            try:
                m = midi.read_midifile(p['path'])
            except Exception:
                continue
            try:
                notes, tempos = process_midifile(m,notes,tempos)
            except Exception:
                log.exception("Could not get features")
                continue
        nm, tm = generate_matrices(notes,tempos)

        data = {'files' : data, 'notes' : notes, 'tempos' : tempos, 'nm' : nm, 'tm': tm}

        return data

class GenerateMarkovTracks(Task):
    data = Complex()

    data_format = MusicFormats.dataframe

    category = RegistryCategories.preprocessors
    namespace = get_namespace(__module__)

    help_text = "Process midi files."

    args = {
        'non_predictors' : ["labels","label_code","fs","enc","fname","Unnamed: 0"],
        'target_var' : 'label_code',
    }

    def train(self, data, target, **kwargs):
        """
        Used in the training phase.  Override.
        """
        self.data = self.predict(data, **kwargs)

    def predict(self, data, **kwargs):
        """
        Used in the predict phase, after training.  Override
        """
        frame1 = pd.read_csv(settings.MIDI_FEATURE_PATH)
        frame2 = pd.read_csv(settings.FEATURE_PATH)

        frame = pd.concat([frame1,frame2],axis=0)
        non_predictors = kwargs.get('non_predictors')
        target = kwargs.get('target_var')

        frame.index = range(frame.shape[0])

        alg = RandomForestTrain()
        good_names = [i for i in frame.columns if i not in non_predictors]
        for c in good_names:
            frame[c] = frame[c].astype(float)

        for c in good_names:
            frame[c] = frame[c].real

        clf = alg.train(np.asarray(frame[good_names]),frame[target],**alg.args)
        track_pool = []
        for i in xrange(0,100):
            track_pool.append(generate_audio_track(data['nm'],1000))

        tempo_pool = []
        for i in xrange(0,100):
            tempo_pool.append(generate_tempo_track(data['tm'],1000))

        pattern_pool = []
        for i in xrange(0,100):
            track_count = random.randint(1,8)
            tempo_track = random.choice(tempo_pool)
            tracks = [tempo_track]
            for i in xrange(0,track_count):
                tracks.append(random.choice(track_pool))
            pattern_pool.append(generate_pattern(tracks))

        quality = []
        for p in pattern_pool:
            quality.append(evaluate_midi_quality(p,clf))

        return data