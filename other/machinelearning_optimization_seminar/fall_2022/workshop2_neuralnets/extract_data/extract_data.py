import numpy as np
import matplotlib.pyplot as plt
import skimage as img
import imageio.v3 as iio

# Load video (the portion with good side view)
frames = []
for i in range(200,232):
    frame = iio.imread("youtube_video.mp4",plugin="pyav",index=i)
    # plt.imshow(frame)
    # plt.show()
    frames.append(frame)

# %%
# Convert to grayscale
test = frames[0]
# 
test = (test[:,:,1] > test[:,:,0]) & (test[:,:,1] > test[:,:,2]) & (test[:,:,1] > 190)
plt.imshow(test, cmap="gray")
plt.show()
# test = img.color.rgb2gray(test)

# Edge find in image
# test = img.feature.canny(test)
