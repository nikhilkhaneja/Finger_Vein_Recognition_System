from django.shortcuts import render
from skimage.measure import compare_ssim
import matplotlib.pyplot as plt
import numpy as np
import cv2
import glob


def mse(imageA, imageB):

	err = np.sum((imageA.astype("float") - imageB.astype("float")) ** 2)
	err /= float(imageA.shape[0] * imageA.shape[1])
	return err
def compare_images(imageA, imageB, title):
	
    m = mse(imageA, imageB)
    grayA = cv2.cvtColor(imageA, cv2.COLOR_BGR2GRAY)
    grayB = cv2.cvtColor(imageB, cv2.COLOR_BGR2GRAY)
    (score, diff) = compare_ssim(grayA, grayB, full=True,multichannel=True)
    diff = (diff * 255).astype("uint8")
    #if(score == 1.0):
    print("SSIM: {}".format(score))
def button(request):
    return render(request,'index.html')

def output(request):
    
    im2 = cv2.imread(r'C:\Users\NIKHIL\OneDrive\Desktop\raw_12.bmp')
    path = r"D:\Projects\ThumbPe\dbfiltered\*.*"

    print (os.path.join(srcFiles + "\raw_" + "1" + ".bmp"));
 
    for file in glob.glob(path):
        #f = os.path.join(srcFiles + "\raw_" + str(i) + ".bmp")
        dir_path = Path("D:\dbfiltered")
        im1= cv2.imread(file)
        fig = plt.figure("Images")
        images = ("Image 1", im1), ("Image 2", im2)
   
        print(file)
        res = compare_images(im1,im2, "path, vs. Original")
        datas = "You can Continue the Transaction"
        #ti= end-start

        if res > 0.95:
            return render(request,'index.html',{'data':datas})
