from PIL import Image
import os
import math
import numpy as np
import scipy
from numpy import *
from scipy import misc, ndimage
from scipy.fftpack import fft, dct, idct
import statistics
from numpy import linalg as LA
from skimage.util import random_noise

np.set_printoptions(threshold=np.inf)

image1 = Image.open("host_image.jpeg")

image1_array = np.asarray(image1)
tup1 = image1_array.shape
print("*************************************************************************************************")
print("                                WATERMARKING USING DCT                                    ")
print("*************************************************************************************************")
print("")
print("******************************* DIMENSION AND SIZE OF COVER IMAGE *******************************")
print("")
print("Dimension of the host image martix: "+str(tup1))
image1_size = tup1[0]*tup1[1]*tup1[2]
print("Size of the host image matrix:"+str(image1_size))
print("")
image2 = Image.open("copyright_image.jpeg")

#convert image to array
image2_array = np.asarray(image2)
tup2 = image2_array.shape
print("******************************* DIMENSION AND SIZE OF COPYRIGHT *********************************")
print("")
print("Dimension of the copyright image matrix:"+str(tup2))
#image size   
image2_size = tup2[0]*tup2[1]*tup2[2]

print("Size of the copyright image matrix: "+str(image2_size))
print("")

#Converting the host image to frequency domain
k = image2_size
alpha = 0.1

image1_dct = dct(dct(dct( image1_array, axis=0, norm='ortho' ), axis=1, norm='ortho' ), axis=2, norm='ortho')

dvec = image1_dct.flatten().T

c = np.sort(abs(dvec))[::-1]
ind = np.argsort(abs(dvec))[::-1]

ind = ind[1:k+1]
cw = dvec[ind]

cr = image2_array.flatten().T
wx = cr
w = (cr)/60

ci = cw+(alpha*w)

dvec[ind] = ci

dhat = np.reshape(dvec,(tup1[0],tup1[1],tup1[2]))
out = idct(idct(idct( dhat, axis=0 , norm='ortho'), axis=1 , norm='ortho'),axis=2 , norm='ortho')
out = np.clip(out,a_min=0,a_max=255)

out = out.astype("uint8")

q = Image.fromarray(out)

q.save('watermarked_image.jpeg')

num1_matrix = image1_array
num2_matrix = out
print("******************************* PSNR CALCULATION ************************************************")
print("")
mse = np.mean((num1_matrix-num2_matrix)**2)
if(mse==0):
	print("PSNR value is: 100")
PIXEL_MAX=255.0
psnr = 20*math.log10(PIXEL_MAX/math.sqrt(mse))
print("PSNR of the cover and watermarked is: "+str(psnr))
print("")

img = (image1-out)

max1 = statistics.mean((img.flatten().T).astype("float"))
print("******************************* CHANGED PIXEL BY ************************************************")
print("")
print("On average we are changing the pixel by not more than "+str(max1)+" gray levels")
print("")

img = np.uint8(img)
q = Image.fromarray(img)

q.save('watermarkspot.jpeg')

d = dct(dct(dct(out, axis=0, norm='ortho' ), axis=1, norm='ortho'),axis=2, norm='ortho')
dvec = d.flatten().T
testc = dvec[ind];
what = ((testc-cw))/alpha;
what = what*60
what = np.round(what)
what1 = what.reshape((-1, 1))
what1 = what1.T
print("*********** SIMILARITY / ROBUSTNESS BETWEEN THE EXTRACTED IMAGE AND ORIGNAL COPYRIGHT ***********")
print("")

cos_sim = np.dot(what1,wx)/(LA.norm(what1)*LA.norm(wx))
print("Cosine similarity of the extracted and original copyright: "+str(cos_sim))
print("")
print("**************************************************************************************************")
what  = np.reshape(what,(tup2[0],tup2[1],tup2[2]))
what = np.clip(what,0,255)
what = what.astype("uint8")
new_img = image2_array - what
what = what+new_img
q = Image.fromarray(what)
q.save('extracted_image.jpeg')