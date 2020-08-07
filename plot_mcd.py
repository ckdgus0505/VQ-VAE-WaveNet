import numpy as np
import matplotlib.pyplot as plt
import librosa
from scipy.spatial.distance import euclidean
from fastdtw import fastdtw

def getMFCC(src, n_mfcc = 24):
    # src : audio file path
    y, sr = librosa.load(src)
    data = librosa.feature.mfcc(y=y, sr=sr, n_fft=4096, n_mfcc=n_mfcc)

    return data

def cal_mcd(C, C_hat):
    if C.ndim==2:
        K = 10 * np.sqrt(2) / np.log(10)
        return K * np.mean(np.sqrt(np.sum((C - C_hat) ** 2, axis = 1)))
    elif C.ndim==1:
        K = 10 * np.sqrt(2) / np.log(10)
        return K * np.mean(np.sqrt(np.sum((C - C_hat) ** 2)))
        

mcd_list = list()

original_path = './original.wav'
path = list()
path.append("./100000_p225.wav")
path.append("./2.wav")
path.append("./3.wav")
path.append("./2.wav")
path.append("./4.wav")



C = getMFCC(original_path)
ll, l1 = C.shape


for i in range(0, len(path)):
    C_hat = getMFCC( path[i] )
    lll, l2 = C_hat.shape
    a, b = fastdtw(C.T, C_hat.T, dist=cal_mcd)
    b = np.array(b)
    print(a)
    if (l1 > l2):
        fdtw_C = np.zeros(shape=(l1,ll))
        fdtw_C_hat = np.zeros(shape=(l1,ll))
        for j in range(0, l1):
            fdtw_C[j] = C.T[b[j][0]]
            fdtw_C_hat[j] = C_hat.T[b[j][1]]

    else:
        fdtw_C = np.zeros(shape=(l2,ll))
        fdtw_C_hat = np.zeros(shape=(l2,ll))
        for j in range(0, l2):
            fdtw_C[j] = C.T[b[j][0]]
            fdtw_C_hat[j] = C_hat.T[b[j][1]]
            

    mcd = cal_mcd(fdtw_C, fdtw_C_hat)
    mcd_list.append(mcd)

#    '''
x = range(1, len(mcd_list)+1)
plt.title("mcd")
plt.xlabel("epoch")
plt.xticks(np.arange(1, len(mcd_list)+1, step=1))
plt.ylabel("mcd")
# plt.figure(figsize=(15,20))
plt.plot(x, mcd_list)

plt.savefig("./mcd.png", dpi=400)
#    '''
