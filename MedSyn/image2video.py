# use "pip install opencv-python" if cv2 is not found
import cv2
import glob
import numpy as np
import imageio
 
# Get a list of PNG images on the "test_images" folder
images = glob.glob("../test_images/*.png")
# You can also use the following two lines to get PNG files in a folder.
# import os
# images = [os.path.join("../test_images", file) for file in os.listdir("../test_images") if file.endswith(".png")]
# Sort images by name. Optional step.
images = sorted(images)
# Define codec and create a VideoWriter object
# cv2.VideoWriter_fourcc(*"mp4v") or cv2.VideoWriter_fourcc("m", "p", "4", "v")
fourcc = cv2.VideoWriter_fourcc(*"XVID") #XVID
video = cv2.VideoWriter(
    filename="output.avi", fourcc=fourcc, fps=30.0, frameSize=(400, 500)
)


images = np.load('ddpm.npy') ## 1000 images with resolution:( 144, 192, 144)
print(f'there are {len(images)}')
with imageio.get_writer('output1.gif', mode='I', fps=6) as writer:
    for image in images:
        frame = np.rot90(image[0,0,:,96,:])##.T
        # frame = np.flip(image[0,0,70,:,:].T)
        writer.append_data(((frame)* 255).astype('uint8'))

        # writer.append_data((frame * 255).astype('uint8'))
exit()
# Read each image and write it to the video
for image in images:
    # read the image using OpenCV
    print(f"the image size is {image.shape}")
    frame1 = image[0,0,:,:,72] #cv2.imread(image)
    # frame2 = image[0,0,:,96,:] #cv2.imread(image)
    # frame3 = image[0,0,72,:,:] #cv2.imread(image)
    ##
    frame = cv2.resize(frame1[...,None], dsize=(400, 500))


    # Optional step to resize the input image to the dimension stated in the
    # VideoWriter object above
    # frame = cv2.resize(frame, dsize=(400, 500))
    video.write(frame)
 
# Exit the video writer
video.release()