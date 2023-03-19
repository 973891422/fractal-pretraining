import os
import pickle
import random
from PIL import Image

import numpy as np
import torch
from fractal_learning.fractals import ifs, diamondsquare

seed = 123
def init_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    random.seed(seed)
    
ifs_file = './ifs-10k.pkl'

sample_images = 18000
sample_codes = 2
iterations = 1000
image_size = 32
save_dir = f'./ifs-10k_n{sample_images}_i{sample_codes}_iter{iterations}_seed{seed}/'


with open(ifs_file, 'rb') as f:
	fractal_systems = pickle.load(f)
num_systems = len(fractal_systems["params"])
print(num_systems)
print(fractal_systems["params"][0]['system'])


images = []
for k in range(sample_images):
    indices = np.random.choice(num_systems, size=sample_codes, replace=False)
    # print(indices)
    black_backgrounds = []
    black_backgrounds = np.zeros((2, image_size, image_size, 3), dtype=np.uint8)
    
    for i in range(sample_codes):
        system = fractal_systems["params"][indices[i]]['system']
        for j in range(2):
            points = ifs.iterate(system, 100000)
            gray_image = ifs.render(points, s=image_size, binary=False)
            color_image = ifs.colorize(gray_image)
            black_background = black_backgrounds[j]
            # print(black_background.shape)
            black_background[gray_image.nonzero()] = color_image[gray_image.nonzero()]
            # background = np.zeros((image_size, image_size, 3), dtype=np.uint8)
            # background[gray_image.nonzero()] = color_image[gray_image.nonzero()]
            # Image.fromarray(background).save(save_dir + f'k{k}_i{i}_j{j}.jpg')
    
    for j in range(2):
        os.makedirs(save_dir + f'{j}/', exist_ok=True)
        Image.fromarray(black_backgrounds[j]).save(save_dir + f'{j}/{k}.jpg')
        
# a_system = fractal_systems["params"][0]['system']
# a_points = ifs.iterate(a_system, 100000)
# a_gray_image = ifs.render(a_points, s=image_size, binary=False)
# a_color_image = ifs.colorize(a_gray_image)
# a_black_background = np.zeros((image_size, image_size, 3), dtype=np.uint8)
# a_black_background[a_gray_image.nonzero()] = a_color_image[a_gray_image.nonzero()]
# Image.fromarray(a_black_background).save("./a.jpg")

# b_system = fractal_systems["params"][1]['system']
# b_points = ifs.iterate(b_system, 100000)
# b_gray_image = ifs.render(b_points, s=image_size, binary=False)
# b_color_image = ifs.colorize(b_gray_image)

# b_black_background = np.zeros((image_size, image_size, 3), dtype=np.uint8)
# b_black_background[b_gray_image.nonzero()] = b_color_image[b_gray_image.nonzero()]
# Image.fromarray(b_black_background).save("./b.jpg")

# black_background = np.zeros((image_size, image_size, 3), dtype=np.uint8)
# black_background[a_gray_image.nonzero()] = a_color_image[a_gray_image.nonzero()]
# black_background[b_gray_image.nonzero()] = b_color_image[b_gray_image.nonzero()]

# print(black_background.dtype)
# print(black_background.shape)
# img_pil = Image.fromarray(black_background)
# # img_pil.save("./a.jpg")
# img_pil.save("./all.jpg")
# # img_pil.save("./all1.jpg")
