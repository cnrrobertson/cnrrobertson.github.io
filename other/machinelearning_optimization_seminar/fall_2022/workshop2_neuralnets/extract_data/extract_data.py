import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as anim
import skimage as img
import imageio.v3 as iio

# Load video (the portion with good side view)
frames = []
heights = []
cut = (160,200)
for i in range(200,232):
    frame = iio.imread("youtube_video.mp4",plugin="pyav",index=i)

    # Cut the image to focus only on the wave portion
    frame = frame[cut[0]:cut[1],:,:]

    # Find where the image is more green than red or blue and very bright green
    mean_green = np.mean(frame[:,:,1])
    std_green = np.std(frame[:,:,1])
    frame = (frame[:,:,1] > frame[:,:,0]) & (frame[:,:,1] > frame[:,:,2]) & (frame[:,:,1] > mean_green+std_green)

    # Find an approximate height of the wave by averaging the y-locations of the bright green areas
    height = np.zeros(frame.shape[1])
    for j in range(frame.shape[1]):
        height[j] = np.mean(np.where(frame[:,j] == 1)[0])

    frames.append(frame)
    heights.append(height)

# Adjust images and heights for an un-leveled camera
im_width = len(heights[16])
slope = (heights[16][-1] - heights[16][0]) / im_width
for i in range(len(heights)):
    frame = frames[i]
    height = heights[i]

    # Adjust
    for j in range(len(height)):
        shift = int(slope*(im_width-j))
        # Move frame pixels per column
        frame[:,j] = np.roll(frame[:,j], shift)
        # Move height of wave
        height[j] += shift
    frames[i] = frame
    heights[i] = height

frames = np.array(frames)
heights = np.array(heights)
# %%
# Show animation for visual validation
fig = plt.figure(figsize=(10,4))
im = plt.imshow(frames[0], cmap="gray")
line = plt.plot(heights[0], color="red")[0]

def animation_function(i):
    im.set_array(frames[i])
    line.set_ydata(heights[i])
    return [im,line]

wave_animation = anim.FuncAnimation(fig, animation_function, frames=range(len(frames)), blit=True)
plt.show()

# %%
# Save data
# Video portion is about 2 seconds long
times = np.linspace(0,2,len(heights))
np.save("../data/video_wave_images.npy",frames)
np.save("../data/video_wave_heights.npz",h=heights,)
