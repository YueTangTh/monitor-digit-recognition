import cv2
import random
import numpy as np
import matplotlib.pyplot as plt

fr =3

d = np.load('results/data/v2/144.npy')
x = np.linspace(0, len(d)-1, len(d))
plt.plot(x/float(fr)/60.0, d)
plt.xlabel("Time(min)")
plt.ylabel("Value")
plt.title('Pulse')
plt.savefig('results/4th/Pulse')
plt.clf()

d = np.load('results/data/v2/58.npy')
x = np.linspace(0, len(d)-1, len(d))
plt.plot(x/float(fr)/60.0, d)
plt.xlabel("Time(min)")
plt.ylabel("Value")
plt.title('ABP_mean')
plt.savefig('results/4th/ABP')
plt.clf()

d = np.load('results/data/v2/48.npy')
x = np.linspace(0, len(d)-1, len(d))
plt.plot(x/float(fr)/60.0, d)
plt.xlabel("Time(min)")
plt.ylabel("Value")
plt.title('ART_mean')
plt.savefig('results/4th/ART')
plt.clf()

d = np.load('results/data/v2/55.npy')
x = np.linspace(0, len(d)-1, len(d))
plt.plot(x/float(fr)/60.0, d)
plt.xlabel("Time(min)")
plt.ylabel("Value")
plt.title('ABS')
plt.savefig('results/4th/ABS')
plt.clf()

d1 = np.load('results/data/v2/37.npy')
d2 = np.load('results/data/v2/5.npy')
d = []

for i in range(len(d1)):
  d.append( float(d1[i])+float(d2[i])/10.0 )

x = np.linspace(0, len(d)-1, len(d))
plt.plot(x/float(fr)/60.0, d)
plt.xlabel("Time(min)")
plt.ylabel("Value")
plt.title('Temperature')
plt.savefig('results/4th/Temperature')
plt.clf()

dsp = np.load('results/data/98.npy')

dsp2 = dsp

for i in range(len(dsp)):
  if dsp[i] == 10:
    dsp[i] =100

x2= np.linspace(0, len(dsp)-1, len(dsp))
plt.plot(x2/float(fr)/60.0, dsp)
plt.plot(x2/float(fr)/60.0, dsp2)
plt.legend()
plt.xlabel("Time(min)")
plt.ylabel("Value")
plt.title('SpO2')
plt.savefig('results/4th/SpO2')
plt.clf()