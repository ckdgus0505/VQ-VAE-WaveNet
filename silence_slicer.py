# coding: utf-8
import os
from pydub import AudioSegment
from pydub.silence import detect_nonsilent

def silence_slicer(path_in, path_out, format="wav"):
    sound = AudioSegment.from_file(path_in, format=format)
    non_sil_times = detect_nonsilent(sound, min_silence_len=50, silence_thresh=sound.dBFS * 1.5)
    if len(non_sil_times) > 0:
        non_sil_times_concat = [non_sil_times[0]]
        if len(non_sil_times) > 1:
            for t in non_sil_times[1:]:
                if t[0] - non_sil_times_concat[-1][-1] < 200:
                    non_sil_times_concat[-1][-1] = t[1]
                else:
                    non_sil_times_concat.append(t)
        non_sil_times = [t for t in non_sil_times_concat if t[1] - t[0] > 350]
        sound[non_sil_times[0][0]: non_sil_times[-1][1]].export(path_out, format='wav')


relative_path = './data/'
file_path = 'VCTK-Corpus/wav48/'
f = open('error.txt', mode = 'w', encoding = 'utf-8')
for subdir, dirs, files in os.walk(relative_path+file_path):
    for wav_file in files:
        tmp = subdir.split('/')[-1]
        save_path = './sliced_data/VCTK_Corpus/wav48/'+tmp+'/'
        os.makedirs(save_path, exist_ok=True)
        try:
            silence_slicer(subdir+'/'+wav_file, save_path+wav_file)
        except Exception as ex:
            copyfile(subdir+'/'+wav_file, save_path+wav_file)

            f.write(wav_file + '\n')
f.close()

