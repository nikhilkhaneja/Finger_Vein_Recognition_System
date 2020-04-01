
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
	
    #fig = plt.figure(title)
   
    #ax = fig.add_subplot(1, 2, 1)
    #plt.imshow(imageA, cmap = plt.cm.gray)
    #plt.axis("off")

    #ax = fig.add_subplot(1, 2, 2)
    #plt.imshow(imageB, cmap = plt.cm.gray)
    #plt.axis("off")
    #3plt.show()
im2 = cv2.imread(r'C:\Users\NIKHIL\OneDrive\Desktop\raw_12.bmp')
path = r"D:\Projects\ThumbPe\dbfiltered\*.*"

for file in glob.glob(path):
#im1 = cv2.imread(r'D:\Projects\ThumbPe\dbfiltered\raw_11.bmp')
    im1= cv2.imread(file)
    fig = plt.figure("Images")
    images = ("Image 1", im1), ("Image 2", im2)
    #for (i, (name, image)) in enumerate(images):
        #ax = fig.add_subplot(1, 2, i + 1)
        #ax.set_title(name)
        #plt.imshow(image, cmap = plt.cm.gray)
        #plt.axis("off")
        #plt.show()
    print(file)
    res = compare_images(im1,im2, "path, vs. Original")
