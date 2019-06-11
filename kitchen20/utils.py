from os.path import join
import numpy as np
import subprocess
import librosa
import random
import torch
import glob
import os


def compute_mfcc(sound, rate, frame=512):
    '''MFCC computation with default settings
    (2048 FFT window length, 512 hop length, 128 bands)'''
    melspectrogram = librosa.feature.melspectrogram(sound,
                                                    sr=rate,
                                                    hop_length=frame)
    logamplitude = librosa.amplitude_to_db(melspectrogram)
    mfcc = librosa.feature.mfcc(S=logamplitude, n_mfcc=13).transpose()
    return mfcc
 

def group(iterator, count):
    '''Group an iterator (like a list) in chunks of <count>'''
    itr = iter(iterator)
    while True:
        yield tuple([next(itr) for i in range(count)])


def compute_zcr(sound, frame_size=512):
    '''Compute zero crossing rate'''
    zcr = []
    for frame in group(sound, frame_size):
        zcr.append(np.nanmean(0.5 * np.abs(np.diff(np.sign(frame)))))

    zcr = np.asarray(zcr)
    return zcr


def convert_ar(src_path, dst_path, ar):
    if not os.path.isfile(dst_path):
        cmd = 'ffmpeg -i "{}" -ac 1 -ar {} -loglevel error -y "{}"'.format(
            src_path, ar, dst_path)
        subprocess.call(cmd, shell=True)


def filter_silent_audio(sound,
                        fs,
                        m_section_t=0.010,
                        m_section_engy_thr=0.0005,
                        n_micro_section_thr=3,
                        M_frame_attention_size=1.1,
                        is_debug=False):
    '''Compute energy of micro-sections (of 10ms) and remove sound when
    large section mostly doesn't have energy
    
    sound: raw sound
    fs: audio sampling frequency
    m_section_t: time of a micro-section in ms
    m_section_engy: threshold energy of a micro section (mean of squares)
    n_micro_section_thr: minimum amount of active micro-section to have to consider a sound
    M_frame_attention_size: macro-section's frame of attention (in sec.)
    '''
    
    def _micro_section_energy_over_thr(micro_section):
        micro_section = micro_section / 25000.
        _sum = (micro_section ** 2).mean()
        if _sum < m_section_engy_thr:
            return False
        return True

    n_to_keep = 0
    # Check energy of micro-sections
    while n_to_keep == 0:
        m_section_len = int(m_section_t * fs)
        engy_over_thr = []
        for i in range(0, len(sound) - m_section_len, m_section_len):
            start = i
            end = i + m_section_len
            micro_section = sound[start: end]
            engy_over_thr.append(
                _micro_section_energy_over_thr(micro_section))
        engy_over_thr = np.array(engy_over_thr)

        # Search for high energy Macro_sections (of M_section_len pts)
        # True if it has <n_micro_section_thr> micro-sections active
        M_frame_attention_size_pts = M_frame_attention_size * fs
        M_section_len = int(M_frame_attention_size_pts / m_section_len)
        to_keep = []
        for i in range(0, len(engy_over_thr) - M_section_len):
            pad = m_section_len  # pad start and end of array
            if i == 0 or i == len(engy_over_thr) - M_section_len - 1:
                pad = (M_section_len / 2 + 1) * m_section_len
            if (engy_over_thr[i: i + M_section_len]).sum() > n_micro_section_thr:
                    to_keep.extend([True,] * pad)
            else :
                    to_keep.extend([False,] * pad)
        to_keep = np.array(to_keep)

        # Only keep active sections
        n_to_keep = int(to_keep.sum())
        if n_to_keep == 0:  # increase sound if nothing is detected
            sound = sound.astype(float) * 1.2

    new_sound = sound.copy() * 0
    sound_to_keep = sound[: len(to_keep)][to_keep]
    for i in range(0, len(new_sound) - n_to_keep, n_to_keep):
        new_sound[i: i + n_to_keep] = sound_to_keep
    new_sound[i + n_to_keep:] = sound_to_keep[: len(new_sound) - i - n_to_keep]

    return sound


# Default data augmentation
def padding(pad):
    def f(sound):
        return np.pad(sound, pad, 'constant')

    return f


def random_crop(size):
    def f(sound):
        org_size = len(sound)
        start = random.randint(0, org_size - size)
        return sound[start: start + size]

    return f


def normalize(factor):
    def f(sound):
        return sound / factor

    return f


# For strong data augmentation
def random_scale(max_scale, interpolate='Linear'):
    def f(sound):
        scale = np.power(max_scale, random.uniform(-1, 1))
        output_size = int(len(sound) * scale)
        ref = np.arange(output_size) / scale
        if interpolate == 'Linear':
            ref1 = ref.astype(np.int32)
            ref2 = np.minimum(ref1 + 1, len(sound) - 1)
            r = ref - ref1
            scaled_sound = sound[ref1] * (1 - r) + sound[ref2] * r
        elif interpolate == 'Nearest':
            scaled_sound = sound[ref.astype(np.int32)]
        else:
            raise Exception('Invalid interpolation mode {}'.format(
                interpolate))

        return scaled_sound

    return f


def random_gain(db):
    def f(sound):
        return sound * np.power(10, random.uniform(-db, db) / 20.0)

    return f


def noiseAugment(noise_path):
    dataset = dict(np.load(noise_path).items())
    train, valid = dataset['train'][0], dataset['valid'][0]
    valid = (valid / np.percentile(train, 95)).clip(-1, 1)
    train = (train / np.percentile(train, 95)).clip(-1, 1)

    def f(is_train, audio_len):
        ds = train if is_train else valid
        ds = ds.astype(np.float32)
        rand_idx = np.random.randint(0, len(ds) - audio_len - 1)
        return ds[rand_idx: rand_idx + audio_len]

    return f


def random_flip():
    def f(sound):
        do_flip = bool(random.getrandbits(1))
        if do_flip:
            sound = - sound
        return sound

    return f


# For testing phase
def multi_crop(input_length, n_crops):
    def f(sound):
        stride = (len(sound) - input_length) // (n_crops - 1)
        sounds = [sound[stride * i: stride * i + input_length]
                  for i in range(n_crops)]
        return np.array(sounds)

    return f


# For BC learning
def a_weight(fs, n_fft, min_db=-80.0):
    freq = np.linspace(0, fs // 2, n_fft // 2 + 1)
    freq_sq = np.power(freq, 2)
    freq_sq[0] = 1.0
    weight = 2.0 + 20.0 * (2 * np.log10(12194) + 2 * np.log10(freq_sq)
                           - np.log10(freq_sq + 12194 ** 2)
                           - np.log10(freq_sq + 20.6 ** 2)
                           - 0.5 * np.log10(freq_sq + 107.7 ** 2)
                           - 0.5 * np.log10(freq_sq + 737.9 ** 2))
    weight = np.maximum(weight, min_db)

    return weight


def compute_gain(sound, fs, min_db=-80.0, mode='A_weighting'):
    if 15000 < fs and fs < 17000:  # fs == 16000:
        n_fft = 2048
    elif 43000 < fs and fs < 45000:  # fs == 44100:
        n_fft = 4096
    else:
        raise Exception('Invalid fs {}'.format(fs))
    stride = n_fft // 2

    gain = []
    for i in range(0, len(sound) - n_fft + 1, stride):
        if mode == 'RMSE':
            g = np.mean(sound[i: i + n_fft] ** 2)
        elif mode == 'A_weighting':
            spec = np.fft.rfft(np.hanning(n_fft + 1)[:-1] * sound[i: i + n_fft])
            power_spec = np.abs(spec) ** 2
            a_weighted_spec = power_spec * np.power(10, a_weight(fs, n_fft) / 10)
            g = np.sum(a_weighted_spec)
        else:
            raise Exception('Invalid mode {}'.format(mode))
        gain.append(g)

    gain = np.array(gain)
    gain = np.maximum(gain, np.power(10, min_db / 10))
    gain_db = 10 * np.log10(gain)

    return gain_db


def mix(sound1, sound2, r, fs):
    gain1 = np.max(compute_gain(sound1, fs))  # Decibel
    gain2 = np.max(compute_gain(sound2, fs))
    t = 1.0 / (1 + np.power(10, (gain1 - gain2) / 20.) * (1 - r) / r)
    sound = ((sound1 * t + sound2 * (1 - t)) / np.sqrt(t ** 2 + (1 - t) ** 2))

    return sound


def kl_divergence(y, t):
    nonzero_idx = torch.nonzero(t) 
    entropy = - torch.sum(t[nonzero_idx] * torch.log(t[nonzero_idx]))
    crossEntropy = - torch.sum(t * torch.log_softmax(y))
    # import chainer.functions as F
    # entropy = - F.sum(t[t.data.nonzero()] * F.log(t[t.data.nonzero()]))
    # crossEntropy = - F.sum(t * F.log_softmax(y))

    return (crossEntropy - entropy) / y.shape[0]


# Convert time representation
def to_hms(time):
    h = int(time // 3600)
    m = int((time - h * 3600) // 60)
    s = int(time - h * 3600 - m * 60)
    if h > 0:
        line = '{}h{:02d}m'.format(h, m)
    else:
        line = '{}m{:02d}s'.format(m, s)

    return line
