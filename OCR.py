from PIL import Image
import pytesseract
import os
import re

# 配置 Tesseract OCR 的路径
pytesseract.pytesseract.tesseract_cmd = r'F:\Tesseract-OCR\tesseract.exe'

def check_text_in_image(image_path, target_text="无需餐具"):
    # 打开图像
    image = Image.open(image_path)

    # 使用 Tesseract OCR 提取图像中的文字
    extracted_text = pytesseract.image_to_string(image, lang='chi_sim')  # 使用中文语言包

    # 去除多余的空格和换行符
    cleaned_text = re.sub(r'\s+', '', extracted_text)

    # 检查提取的文字中是否包含目标文字
    if target_text in cleaned_text:
        print(f"图片 '{os.path.basename(image_path)}' 中包含目标文字：'{target_text}'")
        return True
    else:
        print(f"图片 '{os.path.basename(image_path)}' 中不包含目标文字：'{target_text}'")
        return False

# 获取脚本所在目录
script_dir = os.path.dirname(os.path.abspath(__file__))

# 预测文件夹路径
predict_dir = os.path.join(script_dir, 'predict')

# 遍历 predict 文件夹中的所有图片
for img_file in os.listdir(predict_dir):
    if img_file.endswith(('.png', '.jpg')):
        img_path = os.path.join(predict_dir, img_file)
        check_text_in_image(img_path)
