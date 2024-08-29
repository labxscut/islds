import cv2
import matplotlib.pyplot as plt
import numpy as np

# 读取图像
image_path = "0.png"  # 替换为你的图像路径
image = cv2.imread(image_path)

# 将图像从BGR转换为RGB颜色空间
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# 展平图像数组以便绘制直方图
pixels = image_rgb.reshape((-1, 3))

# 绘制RGB颜色直方图
plt.figure(figsize=(10, 4))

# 绘制三条线
for i, color in enumerate(['red', 'green', 'blue']):
    hist = np.histogram(pixels[:, i], bins=256, range=(0, 256))[0]
    plt.plot(hist, color=color, label=f'{color.capitalize()} Channel', linewidth=2)

# 添加图例
plt.legend()

# 设置标题和标签
plt.title('RGB Color Histogram')
plt.xlabel('Pixel Value')
plt.ylabel('Frequency')

# 添加网格线
plt.grid(True, linestyle='--', alpha=0.7)

# 美化图像
plt.style.use('seaborn-darkgrid')

# 显示图形
plt.show()
