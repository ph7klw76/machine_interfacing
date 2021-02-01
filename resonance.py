import numpy as np
from scipy.io import wavfile
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

rate, audio = wavfile.read('bell.wav')
audio = np.mean(audio, axis=1)
N = audio.shape[0]
L = N / rate

print(f'Audio length: {L:.2f} seconds')

f, ax = plt.subplots(2,2)
ax[0][0].plot(np.arange(N) / rate, audio)
ax[0][0].set_xlabel('Time(s)')
ax[0][0].set_ylabel('Amplitude(a.u)')

from skimage import util
NN=1
M = 1024*NN

slices = util.view_as_windows(audio, window_shape=(M,), step=100)
print(f'Audio shape: {audio.shape}, Sliced audio shape: {slices.shape}')
win = np.hanning(M + 1)[:-1]
slices = slices * win
slices = slices.T
print('Shape of `slices`:', slices.shape)
spectrum = np.fft.fft(slices, axis=0)[:M // 2 + 1:-1]
spectrum = np.abs(spectrum)
S = np.abs(spectrum)


Z=np.sum(S,axis=1)
x=[i/10.625*NN for i in range(len(Z))]
ax[0][1].plot(x,Z)
ax[0][1].loglog()
ax[0][1].set_xlabel('Frequency (Hz)')
ax[0][1].set_ylabel('Amplitude(a.u)')

def datapoint(t):
    return t*rate

startpoint=15.05
endpoint=15.8
x1=int(datapoint(startpoint))
x2=int(datapoint(endpoint))
amplitude=[]
number=200
for i in range((x2-x1)//number):
    y=np.max(audio[x1+i*number:x1+(i+1)*number])
    xx=np.where(audio[x1+i*number:x1+(i+1)*number] == np.amax(y))
    if i==0:
        x0= (xx[0][0]+i*number)
        x=0
        y0=y
        y=1
    else:
        x= ((xx[0][0]+i*number)-x0)/rate
        y=y/y0
    amplitude.append([x,y])
    
x=[amplitude[i][0] for i in range(len(amplitude))]
y=[amplitude[i][1] for i in range(len(amplitude))]


def func(t,k):  # defination of a function
    return (np.exp(-k*t))

def drawgraph(func,x,y):
    popt, pcov = curve_fit(func, x, y)                                                             
    predicted=func(x, popt) 
    r=np.corrcoef(y, predicted)
    r2=r[0][1]**2
    print('coefficient of determination:', r2)
    ii=1
    for i in popt:
        print(i," ", ii ,"parameter")
        ii+=1   
    return popt, r2
 
popt, r2=drawgraph(func,x,y)


newx=np.linspace(0,max(x),100)
ax[1][0].plot(newx, func(newx, popt),color='red') 
ax[1][0].set_ylim(0,max(y)*1.1)
ax[1][0].set_xlabel('Time(s)')
ax[1][0].set_ylabel('Amplitude(a.u)')
ax[1][0].set_title('Title here')
ax[1][0].scatter(x, y, label='scatter')           


ax[1][1].plot(np.arange(N) / rate, audio)
ax[1][1].set_xlabel('Time(s)')
ax[1][1].set_ylabel('Amplitude(a.u)')         
ax[1][1].set_xlim(startpoint,endpoint)    
