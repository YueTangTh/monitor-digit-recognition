import os
import time
import cv2
import random
import scipy.io
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from ImageProcessing import FrameProcessor

#=====================================
#====== changeable parameters ========
#=====================================
version = '_2_3'     #recognition network version, please see in 'knn' folder
video_tag = '3_12_5' #cropped video name, sheep3, 12 o'clock, phase 5
frameRate = 3        #fps
shift = (24*60+35)*frameRate #time-shift, in this case, the video starts at 24min 35sec
test_folder = 'test/'+video_tag #path of the video

#=====================================
#========== initialization ===========
#=====================================
#first convert all the images into standard size
std_height = 140 #standard height
trio = [0, 0, 0]
red = []   #three color channels
green = []
blue = []

#=====================================
#=========== nomenclature ============
#=====================================

#In each phase, the data varies under the same index
#Note: ABS refers to the index on the left monitor
#10_1
'''temperature = [380]
pulse = [104]
spo2 = [100, 82]#[fetal, maternal]
art =[44, 49, 52]#[low, mean, high]
abp =[57, 68, 80]#[low, mean, high]
ABS =[101]'''

#10_2
'''temperature = [379]
pulse = [180]
spo2 = [100, 84]#[fetal, maternal]
abp =[56, 68, 79]#[low, mean, high] red
art =[46, 49, 52]#[low, mean, high] yellow
ABS =[101]'''

#10_3
'''temperature = [379]
pulse = [196]
spo2 = [100, 81]#[fetal, maternal]
abp =[55, 66, 78]#[low, mean, high] red
art =[40, 47, 50]#[low, mean, high] yellow
ABS =[101]'''

#10_4
'''temperature = [378]
pulse = [186]
spo2 = [100, 79]#[fetal, maternal]
abp =[40, 42, 44]#[low, mean, high] red
art =[39, 48, 59]#[low, mean, high] yellow
ABS =[92]'''

#11_1
'''temperature = [378]
pulse = [207]
spo2 = [100, 72]#[fetal, maternal]
abp =[33, 34, 36]#[low, mean, high] red
art =[42, 51, 61]#[low, mean, high] yellow
ABS =[50]'''

#11_2
'''temperature = [378]
pulse = [214]
spo2 = [99, 71]#[fetal, maternal]
abp =[34, 35, 37]#[low, mean, high] red
art =[42, 51, 60]#[low, mean, high] yellow
ABS =[1]
'''

#12_1
'''temperature = [376]
pulse = [209]
spo2 = [100, 85]#[fetal, maternal]
abp =[41, 44, 48]#[low, mean, high] red
art =[49, 56, 64]#[low, mean, high] yellow
ABS =[102]'''

#12_2
'''temperature = [376]
pulse = [209]
spo2 = [100, 79]#[fetal, maternal]
abp =[42, 47, 50]#[low, mean, high] red
art =[46, 53, 60]#[low, mean, high] yellow
ABS =[92]'''

#12_3
'''temperature = [376]
pulse = [220]
spo2 = [100, 74]#[fetal, maternal]
abp =[39, 43, 46]#[low, mean, high] red
art =[13, 45, 53]#[low, mean, high] yellow
ABS =[93]'''

#12_4
'''temperature = [376]
pulse = [210]
spo2 = [100, 72]#[fetal, maternal]
abp =[38, 42, 45]#[low, mean, high] red
art =[21, 47, 53]#[low, mean, high] yellow
ABS =[103]'''

#12_5
temperature = [376]
pulse = [207]
spo2 = [100, 70]#[fetal, maternal]
abp =[32, 35, 38]#[low, mean, high] red
art =[45, 49, 52]#[low, mean, high] yellow
ABS =[92]

#13
'''temperature = [375]
pulse = [144]
spo2 = [100, 88]#[fetal, maternal]
art =[45, 48, 50]#[low, mean, high] red
abp =[47, 58, 69]#[low, mean, high] yellow
ABS =[55]'''

#=====================================
#======= function definition =========
#=====================================

frameProcessor = FrameProcessor(std_height, version, False, write_digits=False)  

#random_int_list: 
#function: generate 'length' random integers from 'start' to 'stop'
def random_int_list(start, stop, length):
    start, stop = (int(start), int(stop)) if start <= stop else (int(stop), int(start))
    length = int(abs(length)) if length else 0
    random_list = []
    for i in range(length):
        random_list.append(random.randint(start, stop))
    return random_list

#get_expected_from_filename
#function: obtain the digit from the filename.
#          for example, '51.jpg' => '51'
def get_expected_from_filename(filename):
    expected = filename.split('.')[0]
    expected = expected.replace('A', '.')
    expected = expected.replace('Z', '')
    return expected

#signalprocess
#function: signal smoothing, in case of abrupt surge during signal jumps
def signalprocess(cal, exp_int, trio):
    trio[0] = trio[1]
    trio[1] = trio[2]
    trio[2] = cal
    if (trio[0] - trio[1])*(trio[1] - trio[2]) <0 :
      trio[1] = (trio[0] + trio[2]) / 2
    return int(trio[1])

#finetune
#function: manually set different condition to finetune extracted signals
#          such as invalidation (set -1), calculation (cal/10) and signal delay 
#          (cal_last, the last calculation number, AKA, signal delay from the last time)
#          based on digit values, time period and video tag.

#return: cal, the finetuned predicted signals

def finetune(cal, cal_last, exp_int, count):
    if video_tag == '3_13':
      if (exp_int == 375 and (cal<373 or cal > 377)) or\
         (exp_int == 45 and (cal<31 or cal>55)) or\
         (exp_int ==48 and cal<40) or\
         (exp_int ==50 and cal<10) or\
         (exp_int ==48 and cal>70):
        cal = -1
      if exp_int ==69 and cal>400:
        cal=cal/10
    
    if video_tag =='3_11_1':
      if (exp_int == 34 and cal>50) or\
        (exp_int ==36 and cal>45) or\
        (exp_int ==42 and (cal>45 or cal<20)) or\
        (exp_int ==51 and (cal<45 or cal>56)):
        cal = -1
      if exp_int ==33 and (cal<25 or cal>40):
        cal = cal_last #the last calculation number, AKA, signal delay from the last time
      if exp_int ==33 and cal<30 and (count>120*3 and count<180*3):
        cal = cal_last
        
    if video_tag =='3_11_2':
      if ((exp_int ==34 or exp_int ==35) and cal>80) or \
        (exp_int==51 and cal>65) or \
        (exp_int==60 and cal>90) or \
        ((exp_int==34 or exp_int==35) and count>3*(15*60+35) and count<3*(20*60+40)) or \
        (exp_int==42 and ((count>145*3 and count<190*3)or(count>440*3 and count<462*3) or cal>60)):
        cal = -1
      if ((exp_int ==34 or exp_int==35) and count<15*60*3 and (cal>40 or cal<20)) or\
      (exp_int ==378 and cal<250):
        cal = cal_last
      if exp_int ==37 and cal>400:
        cal = cal/10
      if exp_int ==37 and count > 40*60*3 and count<50*60*3 and cal<30:
        cal = cal+20
     
    if video_tag =='3_10_1':
      if (exp_int == 44 and (count>1*3 and count<24*3) ) or\
      (exp_int ==44 and count>30*3 and count<59*3) or \
      (exp_int ==44 and cal>50) or\
      (exp_int ==49 and (count>30*3 and count <3*60))or\
      (exp_int ==104 and count<60*3):
        cal = -1
        
    if video_tag =='3_10_2':
      if exp_int ==52 and count>99*3 and count<119*3:
        cal = 48
      if exp_int ==56 and count>95*3 and count<165*3:
        cal = 56
    
    if video_tag =='3_10_3':
      if exp_int ==100 and cal>200:
        cal = 100
      if (exp_int ==196 and count >229*3 and count<303*3) or\
      ((exp_int ==40 or exp_int==50) and ((count<136*3) or (count>220*3 and count<302*3) or (count>12*60*3 and count<758*3) or (count>941*3 and count<968*3) or (count>1090 and count<1120)or(count>(18*60+54)*3 and count < (19*60+3)*3)or(count>(22*60+50)*3 and count<(23*60+15)*3) or(count>(26*60+10)*3 and count<(26*60+40)*3)or(count>(28*60+45)*3 and count<(29*60+15)*3))) or\
      (exp_int ==47 and count>381 and count<390) or\
      (exp_int ==47 and count>425 and count<428 and cal ==81)or\
      (exp_int ==55 and count>(12*60+35)*3 and count<(12*60+37)*3):
       cal = -1 
      if exp_int in abp:
        if cal >100:
          cal = cal-cal/100*100
        if exp_int ==66:
          if count>320*3 and cal>80:
            cal=cal-30
      if exp_int in temperature:
        if cal<100:
          cal=cal+300 
        if cal ==377:
          cal =379
    
    if video_tag =='3_10_4':
      if (exp_int ==186 and count>198*3 and count<245*3) or\
      (exp_int ==39 and ((count>141*3 and count<143*3)or(count>300*3 and count<340*3)))or\
      (exp_int ==48 and ((count>101*3 and count<128*3)or(count>300*3 and count<340*3) or(count>438*3)))or\
      (exp_int ==39 and cal>60):
        cal=-1
      if exp_int==42 and cal>100:
        cal=cal-cal/100*100
    
    if video_tag =='3_12_1':
        if count > 95*3:
          cal=cal_last
    
    if video_tag =='3_12_2':
        if (exp_int ==209 and count<180 and cal<50)or\
        (exp_int ==53 and ((count>113*3 and count<166*3) or (count>428*3 and count<450*3))):
          cal=-1
          
    if video_tag =='3_12_3':
        if (exp_int ==13 and count>35*3 and count<120)or\
        (exp_int ==45 and count>15*3):
          cal=-1
    
    if video_tag =='3_12_4':
        if (exp_int ==21 and count>(8*60+13)*3 and count<(8*60+24)*3)or\
        (exp_int ==47 and count>(10*60+18)*3 and count<(11*60+29)*3) or\
        (exp_int ==53 and count>(7*60+48)*3 and count <9*60*3):
          cal=-1
        if exp_int ==53 and count<40*3 and cal >80:
          cal=cal-30
    
    if video_tag =='3_12_5':
        if ((exp_int ==45 or exp_int ==49) and ((count>105*3 and count<171*3) or (count>300*3 and count<403*3) or (count>455*3 and count<510*3)))or\
        (exp_int ==32 and cal>200)or\
        (exp_int ==52 and count>326*3 and count<334*3):
          cal=-1
        if ((exp_int ==207 and count>12*60*3) or\
        (exp_int ==45 and cal>600)):
          cal=cal_last
        if (exp_int ==35 and cal>500) or (exp_int ==38 and cal>600)or\
        (exp_int ==49 and cal>400) or(exp_int ==52 and cal>500):
          cal=cal-cal/100*100
        if exp_int ==376 and cal>600:
          cal=cal-300
                
    return cal

#test_img
#function: take images from cropped videos frame by frame,
#          use knn network for digit recognition,
#          finetune the results manually
#return: signal(main finetuned signals)
#        gray_buff, pred(not really used)
def test_img(path, expected, switch, show_result=True):

  #initialization
  global frame, cal_last 
  
  frame = 0
  signal =[]
  grayscale =[]
  pred =[]
  
  gray_buff =[] #buff array for organizing predicted signals
  cal_buff =[]
  pred_buff =[]
  
  #color channel distributions
  red = np.append(abp, ABS)  
  green = np.append(art, temperature)
  blue = np.append(pulse, spo2)
  
  cal_last = int(expected)#the last calculation number, AKA, signal delay from the last time
  
  cap = cv2.VideoCapture(path)
  exp_int = int(expected)
  trio = [exp_int, exp_int, exp_int]
  
  rd = random_int_list(600*3, 800*3, 100)#random list to take out images, for evaluation
  count = 0
  
  #image extraction from the videos
  while(cap.isOpened()):
    ret, img = cap.read()
    count = count + 1
  
    if ret == True:
      #if the image number is in the random list, print out the image for further evaluation
      if count in rd:
        switch = 1 #print the image
      else:
        switch = 0 #not print the image
      
      frameProcessor.set_image(img)#processor initialization
      predict, aver_gray, cal = frameProcessor.process_image(switch, expected, red, green, blue)#main recognition function
      gray_buff.append(aver_gray) #save the grayscales, for the purpose of grayscale distribution
      
      if cal != '' and cal != '-1':
        cal = int(cal)
        cal = finetune(cal, cal_last, exp_int, count)
        cal_last = signalprocess(cal, exp_int, trio)
        signal.append(cal_last)
      else:
        if exp_int in abp: #11_1
            signal.append(cal_last)
            continue
        if video_tag =='3_11_2':
          if exp_int in temperature: #11_2
            signal.append(cal_last)
            continue
        if video_tag =='3_10_3':
          if exp_int in temperature:#10_3
            signal.append(cal_last)
        if video_tag =='3_12_5':
          if exp_int in temperature:#12_5
            signal.append(cal_last) 
            continue  
          if exp_int in ABS:
            signal.append(cal_last) 
            continue  
        signal.append(-1)#otherwise, invalidate the digits
        
      if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    else:
      break
  return signal, gray_buff, pred

#run_tests
#function: extract signals from individual cropped videos
#          and plot out the extracted results
def run_tests(show_result=True):
    count = 0
    switch = 0
    
    start_time = time.time()
    for file_name in os.listdir(test_folder):
        # Skip hidden files
        if not file_name.startswith('.'):
            count += 1
            expected = get_expected_from_filename(file_name) #obtain labeled digit, as the expected result
            print(expected)
            
            #main fuction of image processing and digit recognition, more details in 'ImageProcessing/FrameProcessor.py'
            signal, grayscale, pred = test_img(test_folder + '/' + file_name, expected, switch, show_result)
            np.save("result/data/"+video_tag+'/' + expected + ".npy", signal)

            x = np.linspace(0, len(signal)-1, len(signal))
            std = np.linspace(1,1, len(signal))
            x = x+std*shift #change the timestamp accordingly
            
            plt.xlabel("Time(min)")
            plt.ylabel("Value")
            plt.plot(x/frameRate/60, signal) # x-axis(min)
            plt.title("Number " + expected)
              
            plt.savefig('result/image/'+video_tag +'/'+ expected + '.png')#plot extracted signals for further evaluations
            plt.clf()

    return x

#main
#function: calculate recognition time,
#          convert extracted signals from .npy to .mat
def main():
    start_time = time.time()
    x = run_tests() #signal extraction function
    length=len(x)/franeRate 
    print("time length: %d" % length + 's')
    print("--- %s seconds ---" % (time.time() - start_time))
    
    # convert data from .npy into .mat
    np_folder = 'result/data/' + video_tag #.npy folder path
    temp=pulse_np=spo2_f_np=spo2_m_np=abs_np=[]
    art_low=art_mean=art_high=abp_low=abp_mean=abp_high=[]
    
    for np_filename in os.listdir(np_folder):
      if not np_filename.startswith('.'):
        path = os.path.join(np_folder, np_filename)
        npy = np.load(path) #obtain .npy files
        
        fn = get_expected_from_filename(np_filename)
        fn_int = int(fn)
        
        if fn_int in temperature:
          for i in range(len(npy)):
            temp.append(npy[i])
        
        #if the tag fits the noclamenture, put the data into corresponding index
        if fn_int in pulse: pulse_np = npy
        if fn_int == spo2[0]: spo2_f_np = npy #fetal
        if fn_int == spo2[1]: spo2_m_np = npy #maternal
        if fn_int in ABS: abs_np = npy
        
        #arterial pressure: low,mean,high
        if fn_int == art[0]: art_low = npy
        if fn_int == art[1]: art_mean = npy
        if fn_int == art[2]: art_high = npy
        if fn_int == abp[0]: abp_low = npy
        if fn_int == abp[1]: abp_mean = npy
        if fn_int == abp[2]: abp_high = npy
    
    #save .mat file 
    scipy.io.savemat('result/'+ video_tag+'.mat', dict(frame = x, temperature = temp, \
    pulse = pulse_np, spo2_f = spo2_f_np, spo2_m = spo2_m_np,\
    art_low = art_low, art_mean = art_mean, art_high = art_high, \
    abp_low = abp_low, abp_mean = abp_mean, abp_high = abp_high))
 
if __name__ == "__main__":
    main()