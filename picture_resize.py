import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import scipy.misc
import numpy as np
import PIL
from PIL import Image

InputDir = "data/hide_specific_target_dataset_noise_picture/"
OutDir = "data/hide_specific_target_dataset_noise_picture_resized/"

def load_image(path):
    image = PIL.Image.open(path)
    image = image.resize((416, 416))
    img = np.asarray(image).astype(np.float32)
    return img



image_paths = sorted([os.path.join(InputDir, i) for i in os.listdir(InputDir)])
print(image_paths)

for i,path in enumerate(image_paths,2):
    # print(i,path)
    image = load_image(path)
    scipy.misc.imsave(os.path.join(OutDir, os.path.basename(path)), image)






