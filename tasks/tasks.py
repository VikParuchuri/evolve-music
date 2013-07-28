from __future__ import division
from itertools import chain
import numpy as np
import pandas as pd
import math
from percept.tasks.base import Task
from percept.fields.base import Complex, List, Dict, Float
from inputs.inputs import MusicFormats
from percept.utils.models import RegistryCategories, get_namespace
from percept.conf.base import settings
import os
from percept.tasks.train import Train
from sklearn.ensemble import RandomForestClassifier
import random
from scipy.fftpack import dct
from scikits.audiolab import oggwrite, play, oggread
from time import gmtime, strftime
import subprocess
from midiutil.MidiFile import MIDIFile
import midi
from multiprocessing import Pool
from functools import partial

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
    instruments = []
    for track in m:
        instrument = None
        for e in track:
            if isinstance(e, midi.events.ProgramChangeEvent):
                instrument = e.data[0]
                instruments.append(instrument)
            if isinstance(e,midi.events.NoteOnEvent) and instrument is not None:
                pitch, velocity = e.data
                tick = e.tick
                if instrument not in notes:
                    notes[instrument] = {'pitch' : [], 'velocity' : [] ,'tick' : []}
                notes[instrument]['pitch'].append(pitch)
                notes[instrument]['velocity'].append(velocity)
                notes[instrument]['tick'].append(tick)
            elif isinstance(e,midi.events.SetTempoEvent):
                tick = e.tick
                tick = round(tick/10)*10
                mpqn = e.mpqn
                tempos['tick'].append(tick)
                tempos['mpqn'].append(mpqn)
        tempos['mpqn'].append(0)
        tempos['tick'].append(0)
    return notes, tempos, instruments

def generate_matrix(seq):
    seq = [round(i/5)*5 for i in seq]
    unique_seq = list(set(seq))
    unique_seq.sort()
    mat = np.zeros((len(unique_seq),len(unique_seq)))
    for i in xrange(0,len(seq)-1):
        i_ind = unique_seq.index(seq[i])
        i_1_ind = unique_seq.index(seq[i+1])
        mat[i_ind,i_1_ind] = mat[i_ind,i_1_ind] + 1
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
    randint = random.uniform(0,np.sum(vec))
    choice = None
    try:
        total = 0.0
        for i in xrange(0,len(vec)):
            total+=vec[i]
            if total>randint:
                choice=i
                break
    except Exception:
        return 0
    if choice is None:
        choice = len(vec)-1
    return choice

def generate_markov_seq(m,inds,length):
    inds = [int(i) for i in inds]
    start = inds[pick_proba(np.divide(np.sum(m,axis=1),1000000))]
    seq = []
    seq.append(start)
    for i in xrange(1,length):
        ind = inds.index(seq[i-1])
        try:
            sind = pick_proba(m[ind,:]/np.sum(m[ind,:]))
            seq.append(inds[sind])
        except:
            seq.append(random.choice(inds))
    return seq

def find_closest_element(e,l):
    dists = [abs(i-e) for i in l]
    ind = dists.index(min(dists))
    return l[ind]

def generate_tick_seq(m,inds,length):
    inds = [int(i) for i in inds]
    tick_max = 4000
    tick_max = int(find_closest_element(tick_max,inds))
    start = inds[pick_proba(np.divide(np.sum(m,axis=1),1000000))]
    log.info("Start {0}".format(start))
    if start > tick_max:
        start = tick_max
    seq = []
    seq.append(start)
    sofar = 0
    i = 1
    zeros_count = 0
    while sofar<length:
        ind = inds.index(seq[i-1])
        sind = pick_proba(m[ind,:]/np.sum(m[ind,:]))
        t = inds[sind]
        if t>tick_max:
            t = tick_max
        if zeros_count>5:
            t = int(find_closest_element(100,inds))
        seq.append(int(t))
        sofar += t
        i+=1

        if t==0:
            zeros_count+=1
        else:
            zeros_count = 0
    if sofar>length:
        seq[-1]-=(sofar-length)
    return seq

def generate_audio_track(notes,length,all_instruments= None):
    if all_instruments is None:
        instrument = random.choice(notes.keys())
    else:
        instrument = random.choice(all_instruments)
    tick = generate_tick_seq(notes[instrument]['tick']['mat'],notes[instrument]['tick']['inds'],length)
    length = len(tick)
    pitch = generate_markov_seq(notes[instrument]['pitch']['mat'],notes[instrument]['pitch']['inds'],length)
    velocity = generate_markov_seq(notes[instrument]['velocity']['mat'],notes[instrument]['velocity']['inds'],length)
    track = midi.Track()
    track.append(midi.TrackNameEvent())
    prog = midi.ProgramChangeEvent()
    prog.set_value(instrument)
    track.append(prog)
    for i in xrange(0,length):
        on = midi.NoteOnEvent(channel=0)
        on.set_pitch(pitch[i])
        on.set_velocity(velocity[i])
        on.tick = tick[i]
        track.append(on)
    track.append(midi.EndOfTrackEvent())
    return track

def generate_tempo_track(tempos,length):
    tick = generate_tick_seq(tempos['tick']['mat'],tempos['tick']['inds'],length)
    length = len(tick)
    mpqn = generate_markov_seq(tempos['mpqn']['mat'],tempos['mpqn']['inds'],length)

    track = midi.Track()
    track.append(midi.TrackNameEvent())
    track.append(midi.TextMetaEvent())
    for i in xrange(0,length):
        if mpqn[i]!=0:
            te = midi.SetTempoEvent()
            te.tick = tick[i]
            te.set_mpqn(mpqn[i])
            track.append(te)
    track.append(midi.EndOfTrackEvent())
    return track

def write_midi_to_file(pattern,name="tmp.mid"):
    midi_path = os.path.abspath(os.path.join(settings.MIDI_STORE_PATH,name))
    midi.write_midifile(midi_path,pattern)
    return midi_path

def convert_to_ogg_tmp(mfile):
    file_end = mfile.split("/")[-1].split(".")[0]
    oggfile = file_end + ".ogg"
    wavfile = file_end + ".wav"
    oggpath = os.path.abspath(os.path.join(settings.MIDI_STORE_PATH,oggfile))
    wavpath = os.path.abspath(os.path.join(settings.MIDI_STORE_PATH,wavfile))
    subprocess.call(['fluidsynth', '-i','-n', '-F', wavpath, settings.SOUNDFONT_PATH, mfile])
    subprocess.call(['oggenc', wavpath])
    os.remove(wavpath)
    return oggpath

def write_and_convert(pattern,name="tmp.mid"):
    midi_path = write_midi_to_file(pattern,name)
    oggpath = convert_to_ogg_tmp(midi_path)

def evaluate_midi_quality(pattern, clf):
    midi_path = write_midi_to_file(pattern)
    oggpath = convert_to_ogg_tmp(midi_path)
    data, fs, enc = oggread(oggpath)
    maxl = fs * settings.MUSIC_TIME_LIMIT
    upto = fs * 2
    if upto>len(data):
        log.error("Input data is too short")
        raise Exception
    data = data[:maxl,:]
    features = process_song(data,fs)
    quality = clf.predict_proba(features)[0,1]
    os.remove(oggpath)
    os.remove(midi_path)
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
        all_instruments = []
        for (z,p) in enumerate(data):
            log.info("On file {0}".format(z))
            try:
                m = midi.read_midifile(p['path'])
            except Exception:
                continue
            try:
                notes, tempos, instruments = process_midifile(m,notes,tempos)
                all_instruments.append(instruments)
            except Exception:
                log.exception("Could not get features")
                continue
        nm, tm = generate_matrices(notes,tempos)

        data = {'files' : data, 'notes' : notes, 'tempos' : tempos, 'nm' : nm, 'tm': tm, 'in' : list(chain.from_iterable(all_instruments))}

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

        evolutions = 20
        track_count = 100
        patterns_to_pick = int(math.floor(track_count/4))
        remixes_to_make = int(math.floor(track_count/4))
        additions_to_make = int(math.floor(track_count/4))
        patterns = generate_patterns(track_count,data)
        for i in xrange(0,evolutions):
            new_quality, quality, patterns = rate_tracks(patterns, clf)
            patterns = patterns[0:patterns_to_pick]
            for i in xrange(0,remixes_to_make):
                patterns.append(remix(random.choice(patterns[:patterns_to_pick]), patterns[:patterns_to_pick]))
            for i in xrange(0,additions_to_make):
                patterns.append(add_song(random.choice(patterns[:patterns_to_pick]), patterns[:patterns_to_pick]))
            patterns += generate_patterns(track_count - len(patterns), data)
        new_quality, quality, patterns = rate_tracks(patterns, clf)

        return data

def add_song(song1,song2):
    min_len = min([len(song1),len(song2)])
    tracks =[]
    for i in xrange(0,min_len):
        new_song1 = [e for e in song1[i] if not isinstance(e,midi.events.TrackNameEvent) and not isinstance(e,midi.events.EndOfTrackEvent)]
        new_song2 = [e for e in song2[i] if not isinstance(e,midi.events.TrackNameEvent) and not isinstance(e,midi.events.ProgramChangeEvent) and not isinstance(e,midi.events.TextMetaEvent)]
        tracks.append(new_song1 + new_song2)
    return midi.Pattern(tracks)

def remix(song1, song2):
    songs = [song1,song2]
    all_tracks = song1[1:] + song2[1:]
    track_len = len(song1) + len(song2) -2
    tempo_track = random.randint(0,1)
    tempo = songs[tempo_track][0]
    new_len = random.randint(1,track_len)
    tracks = []
    for i in xrange(0,new_len):
        choice = random.randint(0,len(all_tracks)-1)
        tracks.append(all_tracks[choice])
        del all_tracks[choice]
        if len(all_tracks)==0:
            break
    return midi.Pattern([tempo] + tracks)

def generate_patterns(track_count,data):
    track_pool = []
    for i in xrange(0,track_count):
        log.info("On track {0}".format(i))
        track_pool.append(generate_audio_track(data['nm'],2000,data['in']))

    tempo_pool = []
    for i in xrange(0,int(math.floor(track_count/4))):
        log.info("On tempo {0}".format(i))
        tempo_pool.append(generate_tempo_track(data['tm'],2000))

    pattern_pool = []
    all_instruments = []
    for t in track_pool:
        for e in t:
            if isinstance(e, midi.events.ProgramChangeEvent):
                all_instruments.append(e.data[0])
    all_instruments = list(set(all_instruments))
    all_instruments.sort()

    for i in xrange(0,int(math.floor(track_count))):
        log.info("On pattern {0}".format(i))
        track_number = random.randint(1,3)
        tempo_track = random.choice(tempo_pool)
        tracks = [tempo_track]
        instruments = []
        for i in xrange(0,track_number):
            dist, instrument = maximize_distance(instruments,all_instruments)
            if dist <=8:
                break
            sel_track_pool = []
            for t in track_pool:
                for e in t:
                    if isinstance(e, midi.events.ProgramChangeEvent) and e.data[0]==instrument:
                        sel_track_pool.append(t)
                        break
            if len(sel_track_pool)==0:
                sel_track_pool = track_pool
            sel_track = random.choice(sel_track_pool)
            for e in sel_track:
                try:
                    e.channel = i
                except:
                    pass
            tracks.append(sel_track)
            instruments.append(instrument)
        pattern_pool.append(generate_pattern(tracks))
    return pattern_pool

def rate_tracks(pattern_pool, clf):
    quality = []
    good_patterns = []

    for i in range(len(pattern_pool)):
        log.info("On pattern {0}".format(i))
        try:
            qual = evaluate_midi_quality(pattern_pool[i],clf)
            quality.append(qual)
            good_patterns.append(pattern_pool[i])
        except Exception:
            log.exception("Could not get quality")
            continue

    new_quality = [abs(i-round(i)) for i in quality]

    new_quality, quality, good_patterns = (list(x) for x in zip(*sorted(zip(new_quality, quality, good_patterns))))
    return new_quality, quality, good_patterns

def generate_and_rate_tracks(track_count,data,clf):
    pattern_pool = generate_patterns(track_count,data)
    new_quality, quality, good_patterns = rate_tracks(pattern_pool, clf)

    return new_quality, quality, good_patterns

def maximize_distance(existing,possible):
    try:
        max_dists = []
        for p in possible:
            max_dists.append(min([abs(p-e) for e in existing]))
        max_dist = max(max_dists)
        max_dist_index = max_dists.index(max_dist)
    except ValueError:
        return 10, random.choice(range(len(possible)))

    return max_dist, max_dist_index