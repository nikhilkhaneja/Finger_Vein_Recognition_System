from __future__ import print_function
from skimage.measure import structural_similarity as ssim
from skimage.measure import compare_ssim
import matplotlib.pyplot as plt
import numpy as np
import cv2
from skimage import exposure
import sys
import numpy as np
import skimage.color
import skimage.filters
import skimage.io
import skimage.viewer



import numpy as np
import cv2
from multiprocessing.pool import ThreadPool

def build_filters():
    filters = []
    ksize = 4000
    for theta in np.arange(0, np.pi, np.pi / 10):
        kern = cv2.getGaborKernel((ksize, ksize), 4.0, theta, 10.0, 3.5, 0, ktype=cv2.CV_32F)
        kern /= 1.5*kern.sum()
        filters.append(kern)
    return filters

def process(img, filters):
    accum = np.zeros_like(img)
    for kern in filters:
        fimg = cv2.filter2D(img, cv2.CV_8UC3, kern)
        np.maximum(accum, fimg, accum)
    return accum

def process_threaded(img, filters, threadn = 8):
    accum = np.zeros_like(img)
    def f(kern):
        return cv2.filter2D(img, cv2.CV_8UC3, kern)
    pool = ThreadPool(processes=threadn)
    for fimg in pool.imap_unordered(f, filters):
        np.maximum(accum, fimg, accum)
    return accum

im1 = cv2.imread(r'D:\Projects\ThumbPe\DatabaseRaw\raw1.bmp')
fig = plt.figure(figsize = (15,15))
#plt.suptitle("MSE: %.2f, SSIM: %.2f" % (m, s))
#show first image
ax = fig.add_subplot(3, 3, 1)
plt.imshow(im1,cmap = plt.cm.gray)
#HIstogram
hist = cv2.calcHist([im1],[0],None,[64],[0,256])
hist,bins = np.histogram(im1,64,[0,256])
ax = fig.add_subplot(3, 3, 2)
plt.hist(im1.ravel(),64,[0,256])
#RGB->GRay
im2 = cv2.cvtColor(im1, cv2.COLOR_BGR2GRAY)
#histogram equalization   - > histeq() matlab wale me
equ = exposure.equalize_hist(im1)
equ_ = exposure.equalize_hist(im2)
ax = fig.add_subplot(3, 3, 3)
plt.imshow(equ) 
ax = fig.add_subplot(3, 3, 4)
plt.imshow(equ_)

kernel = np.ones((5,5), np.uint8) 

# The first parameter is the original image, 
# kernel is the matrix with which image is 
# convolved and third parameter is the number 
# of iterations, which will determine how much 
# you want to erode/dilate a given image

img_erosion = cv2.erode(equ, kernel, iterations=1) 
ax = fig.add_subplot(3, 3, 7)
plt.imshow(img_erosion)
img_dilation = cv2.dilate(equ, kernel, iterations=1) 
ax = fig.add_subplot(3, 3, 8)
plt.imshow(img_dilation)

filters = build_filters()

res1 = process(img_dilation, filters)
ax = fig.add_subplot(3, 3, 9)
plt.imshow(res1)


#blur = skimage.filters.gaussian(im2, sigma=1)

# perform adaptive thresholding
#t = skimage.filters.threshold_otsu(blur)
mask = res1 < 1
#ax = fig.add_subplot(3,3,6)
#plt.imshow(mask)
sel = np.zeros_like(img_dilation)
sel[mask] = img_dilation[mask]

#sel =cv2.cvtColor(sel,cv2.COLOR_GRAY2RGB)
ax = fig.add_subplot(3,3,5)
plt.imshow(sel)
cv2.imshow('result', sel)



kernel = np.ones((3,3), np.uint8) 

# The first parameter is the original image, 
# kernel is the matrix with which image is 
# convolved and third parameter is the number 
# of iterations, which will determine how much 
# you want to erode/dilate a given image

img_erosion = cv2.erode(sel, kernel, iterations=1) 
#ax = fig.add_subplot(3, 3, 7)
#plt.imshow(img_erosion)
img1_dilation = cv2.dilate(sel, kernel, iterations=1) 
#ax = fig.add_subplot(3, 3, 6)
#plt.imshow(img1_dilation)




ax = fig.add_subplot(3, 3, 6)
plt.imshow(img1_dilation)   
#np.set_printoptions(threshold=np.inf)
#data = np.asarray(img_dilation)
#print(data)
#f = open("abcccc.txt","w")
#f.write(str(data))
#f.close()

cropim = img1_dilation[62:195,5:430]
ax = fig.add_subplot(3, 3, 6)
plt.imshow(cropim)
axq = cv2.imwrite('D:\OutPut\oap_1.bmp',img_dilation)

print(axq)
#abc = cv2.medianBlur(im2,5)
#ret,th1 = cv2.threshold(equ,127,255,cv2.THRESH_BINARY)
#th3 = cv2.adaptiveThreshold(equ_,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,11,2)

#abc = threshold ( grey_image, bin_image, 0, 255, THRESH_BINARY | THRESH_OTSU );




 