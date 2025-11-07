"""诊断脚本 - 检查图片旋转和EXIF"""
import cv2
from PIL import Image
from PIL.ExifTags import TAGS

IMAGE_PATH = r"C:\Users\DELL\Desktop\image_understanding\project2\picture\100NIKON-DSCN0008_DSCN0008.JPG"

print("=" * 60)
print("诊断报告")
print("=" * 60)

# 1. 原始文件尺寸
img_cv = cv2.imread(IMAGE_PATH)
print(f"\n1. cv2.imread 读取的尺寸: {img_cv.shape[1]} x {img_cv.shape[0]}")

# 2. EXIF Orientation
try:
    img_pil = Image.open(IMAGE_PATH)
    exif = img_pil._getexif()
    orientation = 1
    focal_mm = None
    
    if exif:
        for tag_id, value in exif.items():
            tag = TAGS.get(tag_id, tag_id)
            if tag == "Orientation":
                orientation = value
                print(f"\n2. EXIF Orientation: {orientation}")
            if tag == "FocalLength":
                if isinstance(value, tuple):
                    focal_mm = value[0] / value[1]
                else:
                    focal_mm = float(value)
                print(f"3. EXIF 焦距: {focal_mm}mm")
except Exception as e:
    print(f"\n读取EXIF失败: {e}")

# 3. 测试旋转
print(f"\n4. Orientation={orientation} 的含义:")
if orientation == 1:
    print("   → 无需旋转")
elif orientation == 6:
    print("   → 需要顺时针旋转90°")
elif orientation == 8:
    print("   → 需要逆时针旋转90°")
else:
    print(f"   → 其他方向: {orientation}")

# 4. 执行旋转测试
print(f"\n5. 旋转测试:")
print(f"   旋转前: {img_cv.shape[1]} x {img_cv.shape[0]}")

if orientation == 8:
    img_rotated = cv2.rotate(img_cv, cv2.ROTATE_90_COUNTERCLOCKWISE)
    print(f"   旋转后: {img_rotated.shape[1]} x {img_rotated.shape[0]}")
    print(f"   ✓ 旋转成功！")
else:
    print(f"   Orientation={orientation}，不需要旋转")

# 5. 焦距计算
sensor_width = 6.17
print(f"\n6. 焦距计算:")
print(f"   传感器宽度: {sensor_width}mm")

if orientation == 8:
    # 旋转后应该用短边
    sensor_pixels = img_rotated.shape[1]  # 旋转后的宽度（原始的高度）
    focal_px = focal_mm * (sensor_pixels / sensor_width)
    print(f"   传感器对应像素: {sensor_pixels}px (旋转后的宽度)")
    print(f"   计算焦距: {focal_mm}mm × ({sensor_pixels} / {sensor_width}) = {focal_px:.0f}px")
else:
    sensor_pixels = img_cv.shape[1]
    focal_px = focal_mm * (sensor_pixels / sensor_width)
    print(f"   传感器对应像素: {sensor_pixels}px")
    print(f"   计算焦距: {focal_mm}mm × ({sensor_pixels} / {sensor_width}) = {focal_px:.0f}px")

print("\n" + "=" * 60)
print("诊断完成")
print("=" * 60)
