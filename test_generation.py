import os
import pickle
import random
from PIL import Image

import numpy as np
import torch
import torchvision
from fractal_learning.fractals import ifs, diamondsquare

def tensor_to_np(tensor):
    img = tensor.mul(255).byte()
    img = img.cpu().numpy().transpose((1, 2, 0))
    return img

seed = 123
def init_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    random.seed(seed)
    
ifs_file = './ifs-10k.pkl'
statmix_file = None
# statmix_file = 'statmix_cifar10_n20_step5_0.2_0.1_0.7_imb0.3_dirichlet0.3_seed123.pkl'

sample_images = 18000
sample_codes = [1,2,3]
iterations = 1000
iteration_times = 2
image_size = 224
use_color_background = True
dataset_name = f'ifs-10k_n{sample_images}_i{"_".join([str(i) for i in sample_codes])}_iter{iterations}_seed{seed}_toy'
print(dataset_name)

if statmix_file is not None:
    dataset_name = 'statmix-' + dataset_name

if use_color_background:
    dataset_name = 'color-' + dataset_name

save_dir = f'./{dataset_name}/'
print(save_dir)

if statmix_file is not None:
    statmix_all_clients = pickle.load(open(statmix_file, 'rb'))
    print(len(statmix_all_clients['mu']))
    print(statmix_all_clients['mu'][0])

with open(ifs_file, 'rb') as f:
	fractal_systems = pickle.load(f)
num_systems = len(fractal_systems["params"])
print(num_systems)
print(fractal_systems["params"][0]['system'])


toTensor = torchvision.transforms.ToTensor()
toPIL = torchvision.transforms.ToPILImage()
images = []
num_codes_all_images = np.random.choice(sample_codes, size=sample_images, replace=True)
for k in range(sample_images):
    
    
    
    if use_color_background:
        colorized_background = diamondsquare.colorized_ds(size=32).copy()
        backgrounds = np.stack([colorized_background, colorized_background.copy()], axis=0)
    else:
        backgrounds = np.zeros((2, image_size, image_size, 3), dtype=np.uint8)
    
    num_codes = num_codes_all_images[k]
    
    indices = np.random.choice(num_systems, size=num_codes, replace=False)
    # print(indices)
    for i in range(num_codes):
        system = fractal_systems["params"][indices[i]]['system']
        for j in range(2):
            points = ifs.iterate(system, 100000)
            gray_image = ifs.render(points, s=image_size, binary=False)
            color_image = ifs.colorize(gray_image)
            if statmix_file is not None:
                
                color_image_tensor = toTensor(color_image)
                
                index_random = np.random.choice(len(statmix_all_clients['mu']), size=1)[0]
                mu_random = statmix_all_clients['mu'][index_random].unsqueeze(1).unsqueeze(2)
                std_random = statmix_all_clients['std'][index_random].unsqueeze(1).unsqueeze(2)
                # print(mu_random)
                # print(std_random)
                
                mu_fps = torch.mean(color_image_tensor, dim=[1, 2]).unsqueeze(1).unsqueeze(2)
                std_fps = torch.std(color_image_tensor, dim=[1, 2]).unsqueeze(1).unsqueeze(2)
                # print(mu_fps)
                # print(std_fps)
                color_image_tensor_statmix = torch.div(color_image_tensor - mu_fps, std_fps)
                color_image_tensor_statmix = (color_image_tensor_statmix * std_random) + mu_random
                color_image_statmix = tensor_to_np(color_image_tensor_statmix)
                
                
                
                background = backgrounds[j]
                # print(black_background.shape)
                background[gray_image.nonzero()] = color_image_statmix[gray_image.nonzero()]
                
                
                # background = np.zeros((image_size, image_size, 3), dtype=np.uint8)
                # background[gray_image.nonzero()] = color_image_statmix[gray_image.nonzero()]
                # Image.fromarray(background).save(save_dir + f'k{k}_i{i}_j{j}_statmix.jpg')
            else:
                background = backgrounds[j]
                # print(black_background.shape)
                background[gray_image.nonzero()] = color_image[gray_image.nonzero()]
                
                
                # background = np.zeros((image_size, image_size, 3), dtype=np.uint8)
                # background[gray_image.nonzero()] = color_image[gray_image.nonzero()]
                # Image.fromarray(background).save(save_dir + f'k{k}_i{i}_j{j}.jpg')
            
    for j in range(2):
        os.makedirs(os.path.join(save_dir, f'{j}'), exist_ok=True)
        Image.fromarray(backgrounds[j]).save(os.path.join(save_dir, f'{j}', f'{k}.jpg'))
        
        
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
