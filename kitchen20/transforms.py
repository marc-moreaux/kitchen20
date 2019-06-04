import numpy as np
import numbers


def _check_audio(audio):
    if not isinstance(audio, (np.array, np.ndarray, list)):
        raise TypeError('audio is nor list, nor np.array ({})'.format(
            type(audio)))
    if len(np.array(audio).shape) > 2:
        raise TypeError('audio has not a correct shape: {}'.format(
            np.array(audio).shape))


def pad(audio, padding, fill=0, padding_mode='constant'):
    '''Pad the given audio on both sides with specified padding mode 
    and fill value

    Args:
        padding_mode: Type of padding.
            Should be: constant, edge, reflect or symmetric. Default is constant.
            - constant: pads with a constant value, this value is specified with fill
            - edge: pads with the last value on the edge of the audio
            - reflect: pads with reflection of audio (without repeating the last value on the edge)
                       padding [1, 2, 3, 4] with 2 elements on both sides in reflect mode
                       will result in [3, 2, 1, 2, 3, 4, 3, 2]
            - symmetric: pads with reflection of audio (repeating the last value on the edge)
                         padding [1, 2, 3, 4] with 2 elements on both sides in symmetric mode
                         will result in [2, 1, 1, 2, 3, 4, 4, 3]
    '''
    # Check inserted values
    _check_audio(audio)
    if not isinstance(padding, (numbers.Number, tuple)):
        raise TypeError('Got inappropriate padding arg')
    if not isinstance(fill, (numbers.Number, str, tuple)):
        raise TypeError('Got inappropriate fill arg')
    if not isinstance(padding_mode, str):
        raise TypeError('Got inappropriate padding_mode arg')
    if isinstance(padding, Sequence) and len(padding) != 2:
        raise ValueError("Padding must be an int or a 2 element tuple," +
                         " not a {} element tuple".format(len(padding)))
    assert padding_mode in ['constant', 'edge', 'reflect', 'symmetric'], \
        'Padding mode should be either constant, edge, reflect or symmetric'

    # Adjust parameters 
    if isinstance(padding, Sequence):
        pad_left, pad_right = padding
    else:
        pad_left = pad_right = padding

    #TODO: check if left right isn't top left !
    return np.pad(audio, (pad_left, pad_right), padding_mode)


def crop(audio, start, end):
    """Crop the given audio
    Args:
        audio (np.array, list): Image to be cropped.
        i: Upper pixel coordinate.
        j: Left pixel coordinate.
        h: Height of the cropped image.
        w: Width of the cropped image.
    Returns:
        PIL Image: Cropped image.
    """
    if not _is_pil_image(img):
        raise TypeError('img should be PIL Image. Got {}'.format(type(img)))

    return img.crop((j, i, j + w, i + h))


class Filter_silent_audio(object):
    """Compute energy of micro-sections (of 10ms) and remove sound when
       large section mostly doesn't have energy

    Inputs:
        tensor (Tensor): Tensor of audio of size (c x n) or (n x c)
        fs (int): audio sampling frequency
        m_section_t (float): time of a micro-section in ms
        m_section_engy (float): threshold energy of a micro section
            (mean of squares)
        n_micro_section_thr (int): minimum amount of active micro-section to 
            have to consider a sound
        M_frame_attention_size (float): macro-section's frame of attention
            (in sec.)

    Returns:
        tensor (Tensor) (Samples x 1):

    """
    def __init__(self, fs,
                 m_section_t=0.010,
                 m_section_engy_thr=0.0005,
                 n_micro_section_thr=3,
                 M_frame_attention_size=1.1):
        self.fs = fs
        self.m_section_t = m_section_t
        self.m_section_engy_thr = m_section_engy_thr
        self.n_micro_section_thr = n_micro_section_thr
        self.M_frame_attention_size = M_frame_attention_size
    
    def _micro_section_energy_over_thr(self, micro_section):
        '''Check micro_section's energy
        '''
        micro_section = micro_section / 25000.
        _sum = (micro_section ** 2).mean()
        if _sum < self.m_section_engy_thr:
            return False
        return True

    def __call__(self):
        n_to_keep = 0
        # Check energy of micro-sections
        while n_to_keep == 0:
            self.m_section_len = int(self.m_section_t * fs)
            engy_over_thr = []
            for i in range(0, len(sound) - self.m_section_len, self.m_section_len):
                start = i
                end = i + self.m_section_len
                micro_section = sound[start: end]
                engy_over_thr.append(
                    _micro_section_energy_over_thr(micro_section))
            engy_over_thr = np.array(engy_over_thr)

            # Search for high energy Macro_sections (of M_section_len pts)
            # True if it has <n_micro_section_thr> micro-sections active
            M_frame_attention_size_pts = self.M_frame_attention_size * fs
            M_section_len = int(M_frame_attention_size_pts / self.m_section_len)
            to_keep = []
            for i in range(0, len(engy_over_thr) - M_section_len):
                pad = self.m_section_len  # pad start and end of array
                if i == 0 or i == len(engy_over_thr) - M_section_len - 1:
                    pad = (M_section_len / 2 + 1) * self.m_section_len
                if (engy_over_thr[i: i + M_section_len]).sum() > self.n_micro_section_thr:
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
