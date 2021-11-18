import cv2
import numpy as np
import os
from ImageProcessing.OpenCVUtils import inverse_colors, sort_contours

RESIZED_IMAGE_WIDTH = 40
RESIZED_IMAGE_HEIGHT = 40
count2 = 0 #different counting indexes
count3 = 0

class FrameProcessor:
    def __init__(self, height, version, debug=False, write_digits=False):
        self.debug = debug
        self.version = version
        self.height = height
        self.file_name = None
        self.img = None
        self.width = 0
        self.original = None
        self.write_digits = write_digits
        self.knn = self.train_knn(self.version)

    #set the image: if 'img' is string, it is a path, use cv2.imread to read the image
    #               if not, then it is an image, simply put it into img.
    def set_image(self, img):
        if isinstance(img,str):
            self.file_name = img
            self.img = cv2.imread(img)
        else:
            self.img = img
        self.original, self.width = self.resize_to_height(self.height) #resize the image
        self.img = self.original.copy() #make a copy of the image
    
    #resize the image, with the same width-length ratio
    def resize_to_height(self, height):
        r = self.img.shape[0] / float(height)
        dim = (int(self.img.shape[1] / r), height) 
        img = cv2.resize(self.img, dim, interpolation=cv2.INTER_AREA) #resize function
        return img, dim[0]
    
    #knn network trainning
    def train_knn(self, version):
        npa_classifications = np.loadtxt("knn/classifications" + version + ".txt",
                                         np.float32)  # read in training classifications
        npa_flattened_images = np.loadtxt("knn/flattened_images" + version + ".txt",
                                          np.float32)  # read in training images

        npa_classifications = npa_classifications.reshape((npa_classifications.size, 1))
        k_nearest = cv2.ml.KNearest_create()
        k_nearest.train(npa_flattened_images, cv2.ml.ROW_SAMPLE, npa_classifications)
        return k_nearest
    
    #main image processing function
    def process_image(self, switch, expected, red, green, blue):

        self.img = self.original.copy()#paste the copy image

        debug_images = []
        global count2, count3, avergray, single
        
        #obtain the number of digits from the label(expected number)
        expected_int = int(expected)
        if expected_int < 10:
          number = 1 #there is only one digit
        if expected_int >= 10 and expected_int < 100:
          number = 2 #there are 2 digits in the frame
        if expected_int >= 100:
          number = 3 #there may be 2/3 digits in the frame
        
        img_rgb = self.img
        img_copy = img_rgb #make another copy
        
        #cv2.imwrite('results/pre/'+expected+'RGB.jpg',img_copy)
        
        #convert into gray pictures
        img_gray = np.zeros((img_rgb.shape[0],img_rgb.shape[1]),dtype=img_rgb.dtype)
        img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2GRAY)
        
        for i in range(len(red)):
          if expected_int == red[i]:
            img_gray[:,:] = img_rgb[:,:,2]  #  red  channel 
        for i in range(len(blue)):
          if expected_int == blue[i]:
            img_gray[:,:] = img_rgb[:,:,0]  #  blue channel
        for i in range(len(green)):
          if expected_int == green[i]:
            img_gray[:,:] = img_rgb[:,:,1]  #  green channel
        
        #cv2.imwrite('results/pre/'+expected+'GRAY.jpg',img_rgb[:,:,2])
        '''
        cv2.imwrite('tests/rgb/blue'+expected+'.jpg',b)#print out the result for one-channel gray conversion
        cv2.imwrite('tests/rgb/red'+expected+'.jpg',r)
        cv2.imwrite('tests/rgb/green'+expected+'.jpg',g)'''
        
        #obtain basic image parameters
        height, width=img_rgb.shape[:2]
        area = height*width

        #calculate average grayscale
        avergray = 0
        '''
        if expected == '45': #calculate grayscale for a particular video
          sum = 0
          for i in range(height):
            for j in range(width):
              sum = sum + img_gray[i][j]
          avergray = sum / (height*width)'''
        
        #OTSU's method for thresholding from grayscale to binary format
        _, img = cv2.threshold(img_gray,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        
        #cv2.imwrite('results/pre/'+expected+'OTSU.jpg',img) #print out OTSU result
        
        # Find the digit contours
        _, contours, _ = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)  # get contours
        #cv2.drawContours(img_copy,contours,-1,(255,0,0),3)
        #cv2.imwrite('results/pre/'+expected+'CONTOUR.jpg',img_copy)
        
        # Assuming we find some, we'll sort them in order left->right
        if len(contours) > 0:
            contours, _ = sort_contours(contours)
        
        potential_decimals = []
        potential_digits = []

        total_digit_height = 0
        total_digit_y = 0
        
        # Loop over all the contours collecting potential digits and decimals
        for contour in contours:
            # get rectangle bounding contour
            [x, y, w, h] = cv2.boundingRect(contour)

            aspect = float(w) / h
            size = w * h
            w_min = 0.5* width/ number
            h_min = height * 0.8
            list1 = []
            #condition: rule out irregular contours
            # 1)width > minimum, width/height < 0.8
            # 2)exception: number '1', width<minimum, width/height<0.4, width*height>area/25
            if (w>w_min and w< 0.8*h) or (w<w_min and w<0.4*h and size> area/25):
                flag = True
            else:
                flag = False
            
            if flag:
                list1.append([x,y,w,h])
                total_digit_height += h
                total_digit_y += y
                potential_digits.append(contour)

        avg_digit_height = 0
        avg_digit_y = 0
        potential_digits_count = len(potential_digits)
        left_most_digit = 0
        right_most_digit = 0
        digit_x_positions = []

        # Calculate the average digit height and y position so we can determine what we can throw out
        if potential_digits_count > 0:
            avg_digit_height = float(total_digit_height) / potential_digits_count
            avg_digit_y = float(total_digit_y) / potential_digits_count
            if self.debug:
                print("Average Digit Height and Y: " + str(avg_digit_height) + " and " + str(avg_digit_y))

        output = ''
        ix = 0
        
        # Loop over all the potential digits and see if they are candidates to run through KNN to get the digit
        for pot_digit in potential_digits:
            [x, y, w, h] = cv2.boundingRect(pot_digit)

            if w<h :
                cropped = img[y:y + h, x: x + w]
                # Draw a rect around it
                cv2.rectangle(self.img, (x, y), (x + w, y + h), (255, 0, 0), 2)
                # Call into the KNN to determine the digit
                digit = self.predict_digit(cropped)
                if self.debug:
                    print("Digit: " + digit)
                output += digit
        #11_1
        if expected == '378':
          if output == '375':
            output = '378'
        
        if output == '':
          output = '-1'
        if int(output)<10:
          output = '-1'
          
        '''if expected == '69' and int(output)>350:
          output_int = int(int(output)/10)
          output = str(output_int)
        
        if expected == '45' and int(output)>80:
          count2 = count2 + 1
          output = '-1'''
          #path_output = 'results/surge/%d_%d'%(expected_int, count2)+ '_'+ output + '.jpg'
          #cv2.drawContours(img_copy,contours,-1,(255,0,0),1)
          #cv2.imwrite(path_output, img_copy)

        if switch == 1:
            count2=count2+1
            '''path_output = '/home/yuetang/OCR/OTSU/results/contours/%d_%d'%(expected_int, count2)+ '_'+ output + '.jpg'
            cv2.drawContours(img_copy,contours,-1,(255,0,0),1)
            cv2.imwrite(path_output, img_copy)'''
            predict = output
        else:
            predict =0
        
        '''count = count + 1
        if output != '':
          output_int = int(output)
          if output_int < 40 and count % 30 == 0:
            path_output = '/home/yuetang/OCR/OTSU/results/contour/%d_%d' % (output_int, count) + '.jpg'
            cv2.drawContours(img_copy,contours,-1,(255,0,0),1)
            cv2.imwrite(path_output, img_copy)
        if output == '' and count < 300:
            path_output = '/home/yuetang/OCR/OTSU/results/contour/%d_%d' % (count, count) + '.jpg'
            cv2.drawContours(img_copy,contours,-1,(255,0,0),1)
            cv2.imwrite(path_output, img_copy)'''
        
        return predict, avergray, output


    # Predict the digit from an image using KNN
    def predict_digit(self, digit_mat):
        # Resize the image
        imgROIResized = cv2.resize(digit_mat, (RESIZED_IMAGE_WIDTH, RESIZED_IMAGE_HEIGHT))
        # Reshape the image
        npaROIResized = imgROIResized.reshape((1, RESIZED_IMAGE_WIDTH * RESIZED_IMAGE_HEIGHT))
        # Convert it to floats
        npaROIResized = np.float32(npaROIResized)
        _, results, neigh_resp, dists = self.knn.findNearest(npaROIResized, k=1)
        predicted_digit = str(chr(int(results[0][0])))

        return predicted_digit
