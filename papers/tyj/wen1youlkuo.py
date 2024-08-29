import cv2
import numpy as np
import matplotlib.pyplot as plt
import time

# 读取图片
img = cv2.imread('10.jpg')

# 记录开始时间
start_time = time.time()

# 降噪
denoised = cv2.fastNlMeansDenoisingColored(img, None, 10, 10, 7, 21)

# 转换为HSV颜色空间
hsv_image = cv2.cvtColor(denoised, cv2.COLOR_BGR2HSV)

# 增强颜色
lower_red = np.array([0, 50, 50])  # 调整这里的数值以突出更深红色
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

# 记录结束时间
end_time = time.time()

# 绘制轮廓并统计苹果数量
img_with_contours = denoised.copy()
cv2.drawContours(img_with_contours, contours, -1, (0, 255, 0), 2)

apple_count = len(contours)

# 显示处理结果
plt.imshow(cv2.cvtColor(img_with_contours, cv2.COLOR_BGR2RGB))
plt.title(f"Detected Apples:{apple_count}  apples found")
plt.savefig('lunkuo2.png', dpi=300)  # 保存生成的图像
plt.show()

# 打印运行时间
print(f"苹果数量：{apple_count}")
print(f"运行时间: {end_time - start_time} 秒")
