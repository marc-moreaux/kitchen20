import os
import torch
import wavio
import shutil
import random
import numpy as np
import pandas as pd
from os.path import join
from torch.utils.data import Dataset, DataLoader
from utils import filter_silent_audio, convert_ar
import utils as U


class Kitchen20(Dataset):
    """Kitchen20 dataset accessor for pytorch"""

    def __init__(self,
                 root,
                 csv_file='kitchen20.csv',
                 threshold_sound=0,
                 audio_rate=16000,
                 overwrite=False,
                 folds=[1,2,3,4,5],
                 transforms=None,
                 use_bc_learning=False,
                 strong_augment=True):
        """
        Args:
            root (string): Root directory of dataset where directory
                ``kitchen20`` exists
            csv_file (string): name of the csv_file
            folds (array, int): folds to call for.
            transform (callable, optional): Optional transform to be applied
                on a sample.
            use_bc_learning (bool): Use the between classes learning approch
            audio_rate (int): audio rate to use for the learning
            overwrite (bool): overwrite existing npz file
        """
        self.root = root

        # Dataset generation
        self.csv_file = csv_file
        self.df = pd.read_csv(join(root, csv_file))
        self.audio_rate = audio_rate
        self.threshold_sound = threshold_sound
        self.db_path = join(self.root, './audio/{}.npz'.format(audio_rate))
        self.classes = self._ordered_classes()
        self.nClasses = len(self.classes)

        # Maybe create dataset
        if not os.path.isfile(self.db_path) or overwrite:
            self._create_dataset()

        # Get item processing
        self.folds = folds
        self.transforms = transforms
        self.use_bc_learning = use_bc_learning
        self.strongAugment = strong_augment
        self.get_db_folds()  # Fils self.sounds and self.labels

    def _ordered_classes(self):
        classes = set(zip(self.df.target, self.df.category))
        classes = sorted(classes, key=lambda x: x[0])
        classes = [c[1] for c in classes]
        return classes

    def __len__(self):
        return len(self.sounds)

    def get_db_folds(self):
        full_db = np.load(self.db_path)
        self.sounds = []
        self.labels = []
        for fold in self.folds:
            self.sounds.extend(
                full_db['fold{}'.format(fold)].item()['sounds'])
            self.labels.extend(
                full_db['fold{}'.format(fold)].item()['labels'])

    def preprocess(self, sound):
        for f in self.transforms:
            sound = f(sound)
        return sound

    def __getitem__(self, idx):
        sound = self.sounds[idx]
        label = self.labels[idx]
        sound = self.preprocess(sound)

        if self.use_bc_learning:  # Training phase of BC learning
            # Select and preprocess another training example
            while True:
                rand_idx = random.randint(0, len(self) - 1)
                label2 = self.labels[rand_idx]
                if label != label2:
                    break
            sound2 = self.sounds[rand_idx]
            sound2 = self.preprocess(sound2)

            # Mix two examples
            r = np.array(random.random())
            sound = U.mix(sound, sound2, r, self.audio_rate).astype(np.float32)
            eye = np.eye(self.nClasses)
            label = (eye[label] * r + eye[label2] * (1 - r)).astype(np.float32)

        else:  # Training phase of standard learning or testing phase
            sound = sound.astype(np.float32)
            label = np.array(label, dtype=np.int32)
        
        if self.strongAugment:
            sound = U.random_gain(6)(sound).astype(np.float32)
        
        return sound, label

    def _create_dataset(self):
        # Convert audio rates
        print('Converting sounds to {}Hz...'.format(
            self.audio_rate))
        for idx, row in self.df.iterrows():
            src_path = join(self.root, row.path)
            dst_path = src_path.replace('audio/', 'tmp/')
            os.makedirs(join(self.root, 'tmp'), exist_ok=True)
            convert_ar(src_path, dst_path, self.audio_rate)

        # Create npz file
        print('Creating corresponding npz file...')
        kitchen20 = {}
        for fold in range(1, 6):
            kitchen20['fold{}'.format(fold)] = {}
            kitchen20['fold{}'.format(fold)]['sounds'] = []
            kitchen20['fold{}'.format(fold)]['labels'] = []

            for idx, row in self.df[self.df.fold == fold - 1].iterrows():
                wav_file = row.path.replace('audio/', 'tmp/')
                wav_file = join(self.root, wav_file)
                sound = wavio.read(wav_file).data.T[0]
                if self.threshold_sound == 0:  # Remove silent sections
                    start = sound.nonzero()[0].min()
                    end = sound.nonzero()[0].max()
                    sound = sound[start: end + 1]
                else:
                    sound = filter_silent_audio(
                        sound,
                        self.audio_rate,
                        m_section_engy_thr=self.threshold_sound)
                label = row.target
                kitchen20['fold{}'.format(fold)]['sounds'].append(sound)
                kitchen20['fold{}'.format(fold)]['labels'].append(label)

        print('Saving')
        np.savez(self.db_path, **kitchen20)
        shutil.rmtree(join(self.root, 'tmp'))
        print('Finished')


if __name__ == '__main__':
    inputLength = 48000
    nCrops = 5
    transforms = []
    transforms += [U.random_scale(1.25)]  # Strong augment
    transforms += [U.padding(inputLength // 2)]  # Padding
    transforms += [U.random_crop(inputLength)]  # Random crop
    transforms += [U.normalize(float(2 ** 16 / 2))]  # 16 bit signed
    transforms += [U.random_flip()]  # Random +-

    train = Kitchen20(root='../',
                      transforms=transforms,
                      overwrite=False,
                      use_bc_learning=True)
    train[3]

    transforms = []
    transforms += [U.padding(inputLength // 2)]  # Long audio
    transforms += [U.normalize(float(2 ** 16 / 2))]  # 16 bit signed
    transforms += [U.multi_crop(inputLength, nCrops)]  # Multiple crops
    transforms += [U.random_flip()]  # Random +-

    test = Kitchen20(root='../',
                     transforms=transforms,
                     overwrite=False)

    def mtest(idx):
        import matplotlib.pyplot as plt
        import sounddevice as sd
        from pprint import pprint
        audio, lbls = train[idx]
        pprint(list(zip(list(lbls), test.classes)))
        sd.play(audio, 16000)
        plt.plot(audio)
        plt.show()
    mtest(3)
