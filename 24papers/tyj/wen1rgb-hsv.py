import cv2
import numpy as np
import matplotlib.pyplot as plt


def extract_red_objects(image_path, hue_threshold=(0, 10), saturation_threshold=(100, 255), value_threshold=(100, 255)):
    # 读取图像
    image = cv2.imread(image_path)

    # 将图像从BGR转换到HSV
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # 定义红色的HSV范围
    lower_red = np.array([hue_threshold[0], saturation_threshold[0], value_threshold[0]])
    upper_red = np.array([hue_threshold[1], saturation_threshold[1], value_threshold[1]])

    # 创建掩码，标记红色区域
    red_mask = cv2.inRange(hsv_image, lower_red, upper_red)

    # 将原始图像和红色目标掩码进行按位与操作
    result_image = cv2.bitwise_and(image, image, mask=red_mask)

    return result_image


def plot_comparison(original_image, processed_image, title1, title2):
    # 显示原始图像和处理后的图像
    plt.figure(figsize=(8, 3))

    plt.subplot(1, 2, 1)
    plt.title(title1)
    plt.imshow(cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB))

    plt.subplot(1, 2, 2)
    plt.title(title2)
    plt.imshow(cv2.cvtColor(processed_image, cv2.COLOR_BGR2RGB))
    plt.savefig('hsv2.png', dpi=300)
    plt.show()


# 图片路径
image_path = '1.jpg'

# 提取红色目标
extracted_red_result = extract_red_objects(image_path)

# 显示原始图像和提取红色目标后的图像对比
original_image = cv2.imread(image_path)
plot_comparison(original_image, extracted_red_result, 'Original Image', 'Extracted Red Objects')
