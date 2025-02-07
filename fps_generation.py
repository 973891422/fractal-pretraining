import os
import pickle
import random
from PIL import Image
from tqdm import tqdm
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

ifs_name = 'ifs-100k'  
ifs_file = f'./{ifs_name}.pkl'

sample_images = 50000
sample_codes = 2
iterations = 1000
image_size = 32
dataset_name = f'{ifs_name}_i{sample_codes}_n{sample_images}_iter{iterations}_size{image_size}_seed{seed}'


use_random_affine = True
randomAffine = torchvision.transforms.RandomAffine(degrees=(-180, 180), translate=(0.2, 0.2), scale=(0.6, 0.85))
if use_random_affine:
    dataset_name = 'affined-' + dataset_name

statmix_file = None
# statmix_file = 'statmix_cifar10_n20_step5_0.2_0.1_0.7_imb0.3_dirichlet0.3_seed123.pkl'
use_color_background = False


# if statmix_file is not None:
#     dataset_name = 'statmix-' + dataset_name
#     statmix_all_clients = pickle.load(open(statmix_file, 'rb'))
#     print(len(statmix_all_clients['mu']))
#     print(statmix_all_clients['mu'][0])

# if use_color_background:
#     dataset_name = 'color-' + dataset_name

save_dir = f'./{dataset_name}/'
print(save_dir)


with open(ifs_file, 'rb') as f:
	fractal_systems = pickle.load(f)
num_systems = len(fractal_systems["params"])
print(num_systems)
print(fractal_systems["params"][0]['system'])


toTensor = torchvision.transforms.ToTensor()
toPIL = torchvision.transforms.ToPILImage()
images = []
for k in tqdm(range(sample_images)):
    indices = np.random.choice(num_systems, size=sample_codes, replace=False)
    # print(indices)
    
    backgrounds = np.zeros((2, image_size, image_size, 3), dtype=np.uint8)
    # if use_color_background:
    #     colorized_background = diamondsquare.colorized_ds(size=image_size).copy()
    #     backgrounds = np.stack([colorized_background, colorized_background.copy()], axis=0)
    # else:
    #     backgrounds = np.zeros((2, image_size, image_size, 3), dtype=np.uint8)
    
    for i in range(sample_codes):
        system = fractal_systems["params"][indices[i]]['system']
        for j in range(2):
            points = ifs.iterate(system, iterations)
            gray_image = ifs.render(points, s=image_size, binary=False)
            color_image = ifs.colorize(gray_image)
            # if statmix_file is not None:
                
            #     color_image_tensor = toTensor(color_image)
                
            #     index_random = np.random.choice(len(statmix_all_clients['mu']), size=1)[0]
            #     mu_random = statmix_all_clients['mu'][index_random].unsqueeze(1).unsqueeze(2)
            #     std_random = statmix_all_clients['std'][index_random].unsqueeze(1).unsqueeze(2)
            #     # print(mu_random)
            #     # print(std_random)
                
            #     mu_fps = torch.mean(color_image_tensor, dim=[1, 2]).unsqueeze(1).unsqueeze(2)
            #     std_fps = torch.std(color_image_tensor, dim=[1, 2]).unsqueeze(1).unsqueeze(2)
            #     # print(mu_fps)
            #     # print(std_fps)
            #     color_image_tensor_statmix = torch.div(color_image_tensor - mu_fps, std_fps)
            #     color_image_tensor_statmix = (color_image_tensor_statmix * std_random) + mu_random
            #     color_image_statmix = tensor_to_np(color_image_tensor_statmix)
                
                
                
            #     background = backgrounds[j]
            #     # print(black_background.shape)
            #     background[gray_image.nonzero()] = color_image_statmix[gray_image.nonzero()]
            
            if use_random_affine: 
                affined_images = torch.cat([toTensor(gray_image), toTensor(color_image)], dim=0)
                affined_images = randomAffine(affined_images)
                
                # print(affined_images[0,:,:].shape)
                # print(affined_images[1:,:,:].shape)
                gray_image = tensor_to_np(affined_images[0,:,:].unsqueeze(0)).squeeze()
                color_image = tensor_to_np(affined_images[1:,:,:])
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
