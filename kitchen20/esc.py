import os
import wavio
import shutil
import random
import numpy as np
import pandas as pd
from os.path import join
from torch.utils.data import Dataset
from torchaudio import transforms
import torch
try:
    from . import utils as U
except Exception:
    import utils as U
from abc import ABC
import configparser


package_dir, _ = os.path.split(os.path.abspath(__file__))

config = configparser.ConfigParser()
for fName in ['config_local.ini', 'config.ini']:
    f_path = join(package_dir, fName)
    if os.path.isfile(f_path):
        config.read(f_path)
        continue


class ESC(Dataset, ABC):
    """Abstract class for the ESC datasets accessible with pytorch"""

    def __init__(self,
                 csv_file,
                 root,
                 threshold_sound=0,
                 audio_rate=16000,
                 overwrite=False,
                 folds=[1, 2, 3, 4, 5],
                 transforms=[],
                 use_bc_learning=False,
                 strong_augment=False,
                 compute_features=False):
        """
        Args:
            root (string): Root directory of dataset where directory
                ``esc`` exists
            csv_file (string): name of the csv_file
            folds (array, int): folds to call for.
            transform (callable, optional): Optional transform to be applied
                on a sample.
            use_bc_learning (bool): Use the between classes learning approch
            audio_rate (int): audio rate to use for the learning
            overwrite (bool): overwrite existing npz file
        """

        # Dataset generation
        self.root = root
        self.audio_rate = audio_rate
        self.threshold_sound = threshold_sound
        self.csv_file = csv_file

        # Process root and related values
        if type(root) is list:
            root = root[0]
        self.db_path = join(root, './audio/{}_{}.npz'.format(
            type(self).__name__, audio_rate))

        # Read the csv and process it
        if 'df' not in self.__dir__():
            self.df = pd.read_csv(join(root, csv_file))
        self.dfs = self.df
        if type(self.df) is list:
            self.df = self.df[0].append(self.df[1:])
        self.classes = self._ordered_classes()
        self.nClasses = len(self.classes)

        # Maybe create dataset
        if not os.path.isfile(self.db_path) or overwrite:
            self._create_dataset()

        # Maybe merge df

        # Get item processing
        self.folds = folds
        self.transforms = transforms
        self.use_bc_learning = use_bc_learning
        self.strongAugment = strong_augment
        self.get_db_folds()  # Fils self.sounds and self.labels

        # Maybe compute mfcc and zcr
        if compute_features:
            self.compute_features()

    def compute_features(self):
        print('Computing features...')
        self.mfcc = []
        self.zcr = []
        for s in self.sounds:
            s = s / float(2 ** 16 / 2)
            self.mfcc.append(U.compute_mfcc(s, self.audio_rate, 512))
            self.zcr.append(U.compute_zcr(s, 512))

    def _ordered_classes(self):
        '''Retrieve classes from df ignoring target if there is more than
        one category name per target'''
        df = self.df
        if(set(zip(df.target, df.category)) > set(df.target)):
            # More than a category per target, create new ordering (alphabetic)
            classes = sorted(set(df.category))
        else:
            # No problem in target and category, create a list of classes
            # in odered by target value
            classes = set(zip(df.target, df.category))
            classes = sorted(classes, key=lambda x: x[0])
            classes = [c[1] for c in classes]
        return classes

    def __len__(self):
        return len(self.sounds)

    def get_db_folds(self):
        full_db = np.load(self.db_path, allow_pickle=True)
        self.sounds = []
        self.labels = []
        self.folds_nb = []
        for fold in self.folds:
            fold_name = 'fold{}'.format(fold)
            print('loading ', fold_name)
            sounds = full_db[fold_name].item()['sounds']
            labels = full_db[fold_name].item()['labels']
            self.sounds.extend(sounds)
            self.labels.extend(labels)
            self.folds_nb.extend([fold, ] * len(labels))

        self.sounds = [torch.tensor(s).unsqueeze(0) for s in self.sounds]

    def preprocess(self, sound):
        sound = self.transforms(sound)
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
            label = torch.tensor(label).int()

        if self.strongAugment:
            sound = U.random_gain(6)(sound).astype(np.float32)

        return sound, label

    def _transform_silent_section(self, sound):
        if self.threshold_sound == 0:  # Remove silent sections
            start = sound.nonzero()[0].min()
            end = sound.nonzero()[0].max()
            sound = sound[start: end + 1]
        else:
            sound = U.filter_silent_audio(
                sound,
                self.audio_rate,
                m_section_engy_thr=self.threshold_sound)
        return sound

    def _create_dataset(self):

        # Root and df to lists
        roots = self.root
        dfs = self.dfs
        if type(roots) is not list:
            roots = [roots]
        if type(dfs) is not list:
            dfs = [dfs]

        # Convert audio
        print('Converting sounds to {}Hz...'.format(
            self.audio_rate))
        for root, df in zip(roots, dfs):
            for idx, row in df.iterrows():
                src_path = join(root, row.path)
                dst_path = src_path.replace('audio/', 'tmp/')
                os.makedirs(join(root, 'tmp'), exist_ok=True)
                U.convert_ar(src_path, dst_path, self.audio_rate)

        # Create npz file
        print('Creating corresponding npz file...')
        dataset = {}
        for fold in range(1, 6):
            fold_name = 'fold{}'.format(fold)
            dataset[fold_name] = {}
            dataset[fold_name]['sounds'] = []
            dataset[fold_name]['labels'] = []
            for root, df in zip(roots, dfs):
                for idx, row in df[df.fold == fold].iterrows():
                    wav_file = row.path.replace('audio/', 'tmp/')
                    wav_file = join(root, wav_file)
                    sound = wavio.read(wav_file).data.T[0]
                    sound = self._transform_silent_section(sound)
                    label = self.classes.index(row.category)
                    dataset[fold_name]['sounds'].append(sound)
                    dataset[fold_name]['labels'].append(label)

        print('Saving')
        np.savez(self.db_path, **dataset)
        for root in roots:
            shutil.rmtree(join(root, 'tmp'))


class ESC50(ESC):
    def __init__(self,
                 csv_file='meta/esc50.csv',
                 root=config['Paths']['ESC50'],
                 threshold_sound=0,
                 audio_rate=16000,
                 overwrite=False,
                 folds=[1,  2, 3, 4, 5],
                 transforms=[],
                 use_bc_learning=False,
                 strong_augment=False,
                 compute_features=False):
        # Fix path in df
        self.df = pd.read_csv(join(root, csv_file))
        self.df['path'] = 'audio/' + self.df.filename

        super().__init__(
            csv_file=csv_file,
            root=root,
            threshold_sound=threshold_sound,
            audio_rate=audio_rate,
            overwrite=overwrite,
            folds=folds,
            transforms=transforms,
            use_bc_learning=use_bc_learning,
            strong_augment=strong_augment,
            compute_features=compute_features)


class ESC10(ESC):
    def __init__(self,
                 csv_file='meta/esc50.csv',
                 root=config['Paths']['ESC10'],
                 threshold_sound=0,
                 audio_rate=16000,
                 overwrite=False,
                 folds=[1, 2, 3, 4, 5],
                 transforms=[],
                 use_bc_learning=False,
                 strong_augment=False,
                 compute_features=False):
        # Fix path in df
        self.df = pd.read_csv(join(root, csv_file))
        self.df = self.df[self.df.esc10 == True]
        self.df['path'] = 'audio/' + self.df.filename

        super().__init__(
            csv_file=csv_file,
            root=root,
            threshold_sound=threshold_sound,
            audio_rate=audio_rate,
            overwrite=overwrite,
            folds=folds,
            transforms=transforms,
            use_bc_learning=use_bc_learning,
            strong_augment=strong_augment,
            compute_features=compute_features)


class ESC70(ESC):
    def __init__(self,
                 csv_file=['meta/esc50.csv',
                           'kitchen20.csv'],
                 root=[config['Paths']['ESC50'],
                       config['Paths']['KITCHEN20']],
                 threshold_sound=0,
                 audio_rate=16000,
                 overwrite=False,
                 folds=[1, 2, 3, 4, 5],
                 transforms=[],
                 use_bc_learning=False,
                 strong_augment=False,
                 compute_features=False):
        # Array of 2 dataframes for ESC50 and kitchen20
        dfs = []
        for r, f in zip(root, csv_file):
            df = pd.read_csv(join(r, f))
            if 'path' not in df.columns:
                df['path'] = 'audio/' + df.filename
            dfs.append(df)
        self.df = dfs

        super().__init__(
            csv_file=csv_file,
            root=root,
            threshold_sound=threshold_sound,
            audio_rate=audio_rate,
            overwrite=overwrite,
            folds=folds,
            transforms=transforms,
            use_bc_learning=use_bc_learning,
            strong_augment=strong_augment,
            compute_features=compute_features)


class Kitchen20(ESC):
    def __init__(self,
                 csv_file='kitchen20.csv',
                 root=config['Paths']['KITCHEN20'],
                 threshold_sound=0,
                 audio_rate=16000,
                 overwrite=False,
                 folds=[1, 2, 3, 4, 5],
                 transforms=[],
                 use_bc_learning=False,
                 strong_augment=False,
                 compute_features=False):
        super().__init__(
            csv_file=csv_file,
            root=root,
            threshold_sound=threshold_sound,
            audio_rate=audio_rate,
            overwrite=overwrite,
            folds=folds,
            transforms=transforms,
            use_bc_learning=use_bc_learning,
            strong_augment=strong_augment,
            compute_features=compute_features)


if __name__ == '__main__':
    from torch.utils.data import DataLoader
    from torch import nn
    from torch import optim

    input_time = 1.5
    audio_rate = 16384
    input_length = int(audio_rate * input_time)

    audio_set = Kitchen20(
        root='/media/data/dataest/kitchen20/',
        folds=[1, 2, 3, 4],
        transforms=transforms.Compose([
            transforms.RandomStretch(1.25),
            transforms.Scale(2 ** 16 / 2),
            transforms.Pad(input_length // 2),
            transforms.RandomCrop(input_length),
            transforms.RandomOpposite()]),
        overwrite=False,
        use_bc_learning=False,
        audio_rate=audio_rate)

    audio_loader = DataLoader(audio_set, batch_size=2,
                              shuffle=True, num_workers=4)

    # Load a sample network
    net = nn.Sequential(
        nn.Conv1d(1, 32, 9, 3), nn.ReLU(), nn.BatchNorm1d(32),
        nn.Conv1d(32, 32, 9, 3), nn.ReLU(), nn.BatchNorm1d(32),
        nn.Conv1d(32, 32, 9, 3), nn.ReLU(), nn.BatchNorm1d(32),
        nn.Conv1d(32, 20, 9, 3), nn.ReLU(),
        nn.AdaptiveAvgPool1d(1)
    )
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.01, momentum=0.9)

    # Training loop
    n_epochs = 5
    summary = {'loss': [[] for _ in range(n_epochs)]}
    for e in range(n_epochs):
        for i, (sounds, labels) in enumerate(audio_loader):

            # Zero the grads
            optimizer.zero_grad()

            # Run the Net
            x = net(sounds)
            x = x.view(x.size()[:-1])

            # Optimize net
            loss = criterion(x, labels.long())
            loss.backward()
            optimizer.step()
            summary['loss'][e].append(loss.item())
        print(np.mean(summary['loss'][e]))
