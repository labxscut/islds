import cv2
import numpy as np
import matplotlib.pyplot as plt

# 读取图片
img = cv2.imread('10.jpg')

# 降噪
denoised = cv2.fastNlMeansDenoisingColored(img, None, 10, 10, 7, 21)

# 转换为HSV颜色空间
hsv_image = cv2.cvtColor(denoised, cv2.COLOR_BGR2HSV)

# 增强颜色
lower_red = np.array([0, 100, 100])  # 调整这里的数值以突出更深红色
upper_red = np.array([10, 255, 255])
mask = cv2.inRange(hsv_image, lower_red, upper_red)
enhanced_color = cv2.bitwise_and(denoised, denoised, mask=mask)

# 转换为灰度图
gray_image = cv2.cvtColor(enhanced_color, cv2.COLOR_BGR2GRAY)

# 边缘检测
edges = cv2.Canny(gray_image, 30, 150)

# 阈值分割
_, thresholded = cv2.threshold(gray_image, 50, 255, cv2.THRESH_BINARY)

# 查找轮廓
contours, _ = cv2.findContours(thresholded, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# 获取苹果的几何坐标
apple_coordinates = []
for contour in contours:
    M = cv2.moments(contour)
    if M["m00"] != 0:
        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])
        # 将坐标系调整为从上往下
        apple_coordinates.append((cx, cy))

apple_count = len(apple_coordinates)

# 显示处理结果
plt.figure(figsize=(12, 8), dpi=300)  # 调整图形大小

# Original Image
plt.subplot(2, 2, 1)
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
plt.title("Original Image")

# Denoised Image
plt.subplot(2, 2, 2)
plt.imshow(cv2.cvtColor(denoised, cv2.COLOR_BGR2RGB))
plt.title("Denoised Image")

# Enhanced Color Image
plt.subplot(2, 2, 3)
plt.imshow(cv2.cvtColor(enhanced_color, cv2.COLOR_BGR2RGB))
plt.title("Enhanced Color Image")

# Edges Image
plt.subplot(2, 2, 4)
plt.imshow(edges, cmap='gray')
plt.title("Edges Image")

plt.tight_layout()

# 保存处理结果图像
plt.savefig('processed_images.png')

# 提取散点图
plt.figure(figsize=(4, 3.5), dpi=300)
plt.scatter(*zip(*apple_coordinates), color=(1, 0, 0, 0.8), marker='o', s=50, label='Apple Centers')
plt.title(f"Detected Apples: {apple_count} ")
plt.gca().invert_yaxis()  # 反转 Y 轴坐标
plt.legend()

# 保存散点图
plt.savefig('scatter_plot.png')

# 显示图像和散点图
plt.show()

print(f"苹果数量：{apple_count}")
print("苹果几何坐标：", apple_coordinates)
