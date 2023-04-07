import os
import matplotlib.pyplot as plt
from PIL import Image

data_dir = 'ifs-10k_n100_i1_iter100_size1024_seed123'

fractal_files = [
    os.path.join(data_dir, '1', '27.jpg'),
    os.path.join(data_dir, '1', '77.jpg'),
    os.path.join(data_dir, '1', '82.jpg'),
    os.path.join(data_dir, '0', '79.jpg'),
    os.path.join(data_dir, '0', '64.jpg'),
    os.path.join(data_dir, '1', '54.jpg'),
    os.path.join(data_dir, '0', '95.jpg'),
    
    os.path.join(data_dir, '0', '93.jpg'),
]


# 指定行列数
rows = 2
cols = 4

# 指定图片大小
fig, ax = plt.subplots(rows, cols)

# 循环绘制图片
for i in range(rows):
    for j in range(cols):
        # 计算当前图片的索引
        index = i * cols + j
        
        # 如果图片数量不足，则退出循环
        if index >= len(fractal_files):
            break
        
        # 读取图片
        img = plt.imread(fractal_files[index])
        
        # 显示图片
        ax[i, j].imshow(img)
        ax[i, j].axis('off')
plt.subplots_adjust(wspace=0.01, hspace=0.01, left=0.01, right=0.99, bottom=0.01, top=0.99)
plt.show()





