import moviepy.editor as mpe
from glob import glob

import sys, os
import numpy as np
import scipy

import cv2

import matplotlib.pyplot as plt

from sklearn import decomposition

# MAX_ITERS = 10
TOL = 1.0e-8

video = mpe.VideoFileClip("media/Video_003.avi")

video.subclip(0,50).ipython_display(width=300)
print(video.duration)

def create_data_matrix_from_video(clip, k, dims):
    return np.vstack([cv2.resize(rgb2gray(clip.get_frame(i/float(k))), dims)
                      .flatten() for i in range(k * int(clip.duration))]).T

def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.299, 0.587, 0.114])

def plt_images(M, A, E, index_array, dims, filename=None):
    f = plt.figure(figsize=(15, 10))
    r = len(index_array)
    pics = r * 3
    for k, i in enumerate(index_array):
        for j, mat in enumerate([M, A, E]):
            sp = f.add_subplot(r, 3, 3*k + j + 1)
            sp.axis('Off')
            pixels = mat[:,i]
            if isinstance(pixels, scipy.sparse.csr_matrix):
                pixels = pixels.todense()
            plt.imshow(np.reshape(pixels, dims), cmap='gray')
    return f

def plots(ims, dims, figsize=(15,20), rows=1, interp=False, titles=None):
    if type(ims[0]) is np.ndarray:
        ims = np.array(ims)
    f = plt.figure(figsize=figsize)
    for i in range(len(ims)):
        sp = f.add_subplot(rows, len(ims)//rows, i+1)
        sp.axis('Off')
        plt.imshow(np.reshape(ims[i], dims), cmap="gray")

scale = 50   # Adjust scale to change resolution of image
dims = (int(240 * (scale/100)), int(320 * (scale/100)))

M = create_data_matrix_from_video(video, 100, (dims[1], dims[0]))
np.save("50_res_surveillance_matrix.npy", M)
#M = np.load("50_res_surveillance_matrix.npy")

#M_high = create_data_matrix_from_video(video, 100)
#np.save("high_res_surveillance_matrix.npy", M)
#M_high = np.load("high_res_surveillance_matrix.npy")

print(dims, M.shape)

# Do not use with high res video, it cannot plot
#plt.figure(figsize=(12, 12))
#plt.imshow(M, cmap='gray')
#plt.imsave(fname="image1.jpg", arr=np.reshape(M[:,140], dims), cmap='gray')

u, s, v = decomposition.randomized_svd(M, 2)
print(u.shape, s.shape, v.shape)
low_rank = u @ np.diag(s) @ v
print(low_rank.shape)

plt.figure(figsize=(12, 12))
plt.imshow(low_rank, cmap='gray')

# Low rank
plt.imshow(np.reshape(low_rank[:,140], dims), cmap='gray')
plt.imsave(fname="low_rank.jpg", arr=np.reshape(low_rank[:,140], dims), cmap='gray')

# M - low rank
plt.imshow(np.reshape(M[:,140] - low_rank[:,140], dims), cmap='gray')


# Other pictures
img = cv2.imread("./images/car.png", cv2.IMREAD_GRAYSCALE)
u, s, v = decomposition.randomized_svd(img, 5)
low_rank = u @ np.diag(s) @ v
print(np.linalg.matrix_rank(low_rank))
plt.imsave(fname="car_bg.jpg", arr=low_rank, cmap='gray')
plt.imsave(fname="car_fg.jpg", arr=img-low_rank, cmap='gray')

img = cv2.imread("./images/man_beach.png", cv2.IMREAD_GRAYSCALE)
u, s, v = decomposition.randomized_svd(img, 5)
low_rank = u @ np.diag(s) @ v
print(np.linalg.matrix_rank(low_rank))
plt.imsave(fname="man_beach_bg.jpg", arr=low_rank, cmap='gray')
plt.imsave(fname="man_beach_fg.jpg", arr=img-low_rank, cmap='gray')

img = cv2.imread("./images/mans.jpg", cv2.IMREAD_GRAYSCALE)
u, s, v = decomposition.randomized_svd(img, 5)
low_rank = u @ np.diag(s) @ v
print(np.linalg.matrix_rank(low_rank))
plt.imsave(fname="mans_bg.jpg", arr=low_rank, cmap='gray')
plt.imsave(fname="mans_fg.jpg", arr=img-low_rank, cmap='gray')