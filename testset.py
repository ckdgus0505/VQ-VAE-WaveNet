import tensorflow as tf
import numpy as np
import librosa, json
from tqdm import tqdm
from scipy.io import wavfile
from utils import get_speaker_to_int


class Dataset():
    def __init__(self):
        self.x = None
        self.y = None
        self.init = None
        self.all_files = None
        self.total = None
        self.speaker_to_int = None

    def read_files(self, filename):
        with open(filename) as file:
            files = file.readlines()
        np.random.shuffle(files)
        return [f.strip() for f in files]

    def trim_silence(self, audio, threshold=0.01, frame_length=2048):
        '''Removes silence at the beginning and end of a sample.'''
        if audio.size < frame_length:
            frame_length = audio.size
        energy = librosa.feature.rms(audio, frame_length=frame_length)
        frames = np.nonzero(energy > threshold)
        indices = librosa.core.frames_to_samples(frames)[1]
        # Note: indices can be an empty array, if the whole audio was silence.
        return audio[indices[0]:indices[-1]] if indices.size else audio[0:0]

    def generator(self, max_len, data_dir, sr):
        indices = list(range(len(self.all_files)))
        # if sr is 16k, use gen, since wavfile is way faster than librosa
        def gen():
            while True:
                indices = np.random.shuffle(indices)
                for i in indices:
                    filename = self.all_files[i]
                    _, wav = wavfile.read(data_dir + filename)
                    wav = (wav + 0.5) / 32767.5
                    start = np.random.randint(low=0, 
                                          high=len(wav) - max_len)
                    wav = wav[start: start + max_len]
                    wav = np.reshape(wav, [max_len, 1])
                    speaker = self.split_func(filename)
                    speaker_id = np.reshape(self.speaker_to_int[speaker], [1])
                    yield wav, speaker_id
        # if sr is other than 16k, force librosa to read as 16k
        # otherwise you should modify the corresponding receptive field
        def gen_sr():
            while True:
                for i in indices:
                    filename = self.all_files[i]
                    wav, _ = librosa.load(data_dir + filename, sr=16000)
 #               start = np.random.randint(low=0, 
 #                                         high=len(wav) - max_len)
                    start = 8126
                    wav = wav[start: start + max_len]
                    wav = np.reshape(wav, [max_len, 1])
                    speaker = self.split_func(filename)
                    speaker_id = np.reshape(self.speaker_to_int[speaker], [1])
                    yield wav, speaker_id
        return gen if sr == 16000 else gen_sr

    def make_iterator(self, relative_path, max_len, sr, batch_size):
        filename = relative_path + self.filename
        speaker_file = relative_path + self.speaker_file
        data_dir = relative_path + self.data_dir

        self.speaker_to_int = get_speaker_to_int(speaker_file)
        self.num_speakers = len(self.speaker_to_int)
        self.all_files = self.read_files(filename)

        gen = self.generator(max_len, data_dir, sr)
        dataset = tf.data.Dataset.from_generator(gen, 
            (tf.float32, tf.int32), ([max_len, 1], [1]))
        self.total = len(self.all_files)

        dataset = dataset.batch(batch_size, drop_remainder=False)
        # dataset = dataset.prefetch(4)
        iterator = dataset.make_initializable_iterator()
        self.init = iterator.initializer
        self.x, y = iterator.get_next()
        self.y = tf.one_hot(y, depth=self.num_speakers, dtype=tf.float32)

class VCTK_test(Dataset):
    def __init__(self, batch_size=1, max_len=5120, sr=48000, relative_path=''):
        super(VCTK_test, self).__init__()

        self.filename = 'vctk_test.txt'
        self.speaker_file = 'vctk_speakers.txt'
        self.data_dir = 'VCTK-Corpus/wav48/'
        self.split_func = lambda s: s.split('/')[0]
        self.make_iterator(relative_path, max_len, sr, batch_size)
