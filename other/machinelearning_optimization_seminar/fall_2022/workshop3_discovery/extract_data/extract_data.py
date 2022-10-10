import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as anim
import skimage as img
import imageio.v3 as iio

# Load video (the portion with good side view)
raw_frames = []
frames = []
heights = []
cut = (160,200)
for i in range(200,232):
    frame = iio.imread("youtube_video.mp4",plugin="pyav",index=i)

    # Cut the image to focus only on the wave portion
    raw_frame = frame[cut[0]:cut[1],:,:]
    frame = frame[cut[0]:cut[1],:,:]

    # Find where the image is more green than red or blue and very bright green
    mean_green = np.mean(frame[:,:,1])
    std_green = np.std(frame[:,:,1])
    frame = (frame[:,:,1] > frame[:,:,0]) & (frame[:,:,1] > frame[:,:,2]) & (frame[:,:,1] > mean_green+std_green)

    # Find an approximate height of the wave by averaging the y-locations of the bright green areas
    height = np.zeros(frame.shape[1])
    for j in range(frame.shape[1]):
        height[j] = np.mean(np.where(frame[:,j] == 1)[0])

    raw_frames.append(raw_frame)
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
raw_frames = np.array(raw_frames)
heights = np.array(heights)

# Drop heights to baseline
base = heights[16, 0]
heights -= base

# %%
# Show animation for visual validation
fig = plt.figure(figsize=(10,4))
im = plt.imshow(frames[0], cmap="gray")
line = plt.plot(base+heights[0], color="red",lw=3)[0]
line2 = plt.plot([0,heights.shape[1]], [31,31], color="orange", ls="--")[0]

def animation_function(i):
    im.set_array(frames[i])
    line.set_ydata(base+heights[i])
    return [im,line,line2]

wave_animation = anim.FuncAnimation(fig, animation_function, frames=range(len(frames)), blit=True)
plt.show()

# %%
# Show animation on original image
fig = plt.figure(figsize=(10,4))
im = plt.imshow(raw_frames[0])
line = plt.plot(base+heights[0], color="red",lw=4)[0]

def animation_function(i):
    im.set_array(raw_frames[i])
    line.set_ydata(base+heights[i])
    return [im,line]

plt.axis(False)
wave_animation = anim.FuncAnimation(fig, animation_function, frames=range(len(frames)), blit=True)
plt.show()

# %%
# Save data
# Video portion is about 2 seconds long
times = np.linspace(0,2,len(heights))
x_domain = len(heights[0])
np.save("../data/video_wave_images.npy",raw_frames)
np.savez("../data/video_wave_heights.npz",h=heights,x=x_domain,t=times)
