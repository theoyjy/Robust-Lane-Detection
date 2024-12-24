from PIL import Image

# 加载图像
image_path = "D:/Code/Robust-Lane-Detection/data/trainset/truth/origin/clips_13_truth/0313-1/60.jpg"
image = Image.open(image_path)

# 输出图像模式和通道数
print("Image mode:", image.mode)
print("Channels:", len(image.getbands()))  # 通道数
