"""
圆柱全景图拼接 - 最终完整版
解决问题：
1. EXIF Orientation信息丢失，需要强制旋转
2. 图片从横向(2048x1536)旋转为竖向(1536x2048)
3. 竖条从左到右拼接成横向长条全景图
"""

import cv2
import numpy as np
import os
import glob
from PIL import Image
from PIL.ExifTags import TAGS


# ==================== 配置区域 ====================
# 👇👇👇 修改这里的路径 👇👇👇
IMAGE_FOLDER = r"C:\Users\DELL\Desktop\image_understanding\project2\picture"
OUTPUT_FILE = "panorama.jpg"

# 传感器参数（尼康 Coolpix 1/2.3" 传感器）
SENSOR_WIDTH_MM = 6.17   # mm
SENSOR_HEIGHT_MM = 4.63  # mm

# 强制旋转（因为EXIF Orientation=1不正确）
FORCE_ROTATE = True  # True=强制逆时针旋转90°

# 手动焦距（None=自动计算，或手动指定如 1800）
MANUAL_FOCAL = None
# ==================== 配置结束 ====================


def get_focal_from_exif(image_path):
    """从EXIF读取焦距(mm)"""
    try:
        img = Image.open(image_path)
        exif = img._getexif()
        if exif:
            for tag_id, value in exif.items():
                tag = TAGS.get(tag_id, tag_id)
                if tag == "FocalLength":
                    if isinstance(value, tuple):
                        return value[0] / value[1]
                    else:
                        return float(value)
    except:
        pass
    return None


def load_and_rotate_images(folder_path, force_rotate=False):
    """加载并旋转图像"""
    extensions = ['*.JPG', '*.jpg', '*.JPEG', '*.jpeg', '*.PNG', '*.png']
    files = []
    
    for ext in extensions:
        pattern = os.path.join(folder_path, ext)
        files.extend(glob.glob(pattern))
    
    # 去重（Windows不区分大小写）
    files = list(set(files))
    files = sorted(files)
    
    if not files:
        raise FileNotFoundError(f"在 {folder_path} 中未找到图片文件")

    print(f"找到 {len(files)} 个图片文件\n")
    
    images = []
    for f in files:
        img = cv2.imread(f)
        if img is not None:
            h_orig, w_orig = img.shape[:2]
            
            if force_rotate:
                # 逆时针旋转90度
                img = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
                h_new, w_new = img.shape[:2]
                print(f"✓ {os.path.basename(f)}: {w_orig}x{h_orig} → {w_new}x{h_new} (已旋转)")
            else:
                print(f"✓ {os.path.basename(f)}: {w_orig}x{h_orig}")
            
            images.append(img)

    if not images:
        raise ValueError("无法加载任何有效图片")

    print(f"\n✅ 加载了 {len(images)} 张图片")
    print(f"📐 最终尺寸: {images[0].shape[1]} x {images[0].shape[0]}\n")
    return images, files


def cylindrical_warp(img, focal):
    """圆柱投影变换"""
    h, w = img.shape[:2]
    
    # 创建映射
    y_i, x_i = np.indices((h, w))
    
    # 归一化坐标
    X = np.stack([
        (x_i - w / 2) / focal,
        (y_i - h / 2) / focal,
        np.ones_like(x_i)
    ], axis=-1)
    
    # 圆柱投影公式
    theta = np.arctan(X[..., 0])
    h_ = X[..., 1] / np.sqrt(X[..., 0] ** 2 + 1)
    
    # 映射到新坐标
    x_ = focal * theta + w / 2
    y_ = focal * h_ + h / 2
    
    # 重映射
    mapx = x_.astype(np.float32)
    mapy = y_.astype(np.float32)
    return cv2.remap(img, mapx, mapy, cv2.INTER_LINEAR)


def extract_features(img):
    """SIFT特征提取"""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    sift = cv2.SIFT_create(nfeatures=1000)
    kp, des = sift.detectAndCompute(gray, None)
    return kp, des


def match_features(des1, des2):
    """特征匹配"""
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(des1, des2, k=2)
    good = []
    for m, n in matches:
        if m.distance < 0.75 * n.distance:
            good.append(m)
    return good


def ransac_horizontal(pts1, pts2, max_iter=1000, thresh=3.0):
    """RANSAC估算水平位移"""
    best_inliers = []
    best_trans = (0, 0)
    n = len(pts1)
    
    for _ in range(max_iter):
        idx = np.random.randint(n)
        dx = pts2[idx, 0] - pts1[idx, 0]
        
        pred_x = pts1[:, 0] + dx
        errs = np.abs(pred_x - pts2[:, 0])
        inliers = np.where(errs < thresh)[0]
        
        if len(inliers) > len(best_inliers):
            dx_median = np.median(pts2[inliers, 0] - pts1[inliers, 0])
            best_trans = (dx_median, 0)
            best_inliers = inliers
    
    return best_trans, len(best_inliers)


def align_images(warped_images):
    """对齐相邻图像"""
    print("=" * 60)
    print("特征提取与配准")
    print("=" * 60)
    
    features = []
    for i, img in enumerate(warped_images):
        kp, des = extract_features(img)
        features.append((kp, des))
        print(f"图像 {i}: {len(kp)} 特征点")
    
    print("\n相邻帧配准:")
    translations = []
    
    for i in range(len(warped_images) - 1):
        kp1, des1 = features[i]
        kp2, des2 = features[i + 1]
        
        matches = match_features(des1, des2)
        
        if len(matches) < 10:
            print(f"  {i}↔{i+1}: ⚠ 匹配点不足")
            translations.append((0, 0, 0))
            continue
        
        pts1 = np.float32([kp1[m.queryIdx].pt for m in matches])
        pts2 = np.float32([kp2[m.trainIdx].pt for m in matches])
        
        trans, inliers = ransac_horizontal(pts1, pts2)
        translations.append((trans[0], trans[1], inliers))
        
        print(f"  {i}↔{i+1}: dx={trans[0]:6.1f}px, 内点={inliers:3d}")
    
    print()
    return translations


def blend_images(warped_images, translations):
    """图像融合"""
    print("=" * 60)
    print("图像融合")
    print("=" * 60)
    
    h, w = warped_images[0].shape[:2]
    
    # 计算累计偏移
    x_offsets = [0]
    cum_x = 0
    for dx, _, _ in translations:
        cum_x += dx
        x_offsets.append(cum_x)
    
    x_offsets = np.array(x_offsets)
    min_x = int(np.floor(x_offsets.min()))
    max_x = int(np.ceil(x_offsets.max())) + w
    
    x_offsets = (x_offsets - min_x).astype(int)
    
    canvas_w = max_x - min_x
    canvas_h = h
    
    print(f"画布大小: {canvas_w} x {canvas_h}")
    
    canvas = np.zeros((canvas_h, canvas_w, 3), dtype=np.float32)
    count = np.zeros((canvas_h, canvas_w, 1), dtype=np.float32)
    
    # 羽化融合
    fade_width = min(50, w // 4)
    
    for i, img in enumerate(warped_images):
        x_off = x_offsets[i]
        y1, y2 = 0, h
        x1, x2 = x_off, x_off + w
        
        # 创建权重（羽化）
        weight = np.ones((h, w, 1), dtype=np.float32)
        for j in range(fade_width):
            alpha = j / fade_width
            weight[:, j] = alpha
            weight[:, -(j + 1)] = alpha
        
        canvas[y1:y2, x1:x2] += img.astype(np.float32) * weight
        count[y1:y2, x1:x2] += weight
    
    count[count == 0] = 1
    panorama = (canvas / count).astype(np.uint8)
    
    print("✓ 融合完成\n")
    return panorama


def crop_black_borders(pano):
    """裁剪黑边"""
    print("=" * 60)
    print("裁剪黑边")
    print("=" * 60)
    
    gray = cv2.cvtColor(pano, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        print("未检测到有效区域\n")
        return pano
    
    x, y, w, h = cv2.boundingRect(max(contours, key=cv2.contourArea))
    
    margin = 5
    x = max(0, x + margin)
    y = max(0, y + margin)
    w = w - 2 * margin
    h = h - 2 * margin
    
    cropped = pano[y:y+h, x:x+w]
    
    print(f"原始: {pano.shape[1]} x {pano.shape[0]}")
    print(f"裁剪后: {cropped.shape[1]} x {cropped.shape[0]}\n")
    
    return cropped


def main():
    print("\n" + "=" * 60)
    print("圆柱全景图拼接 - 最终完整版")
    print("=" * 60 + "\n")
    
    print(f"📁 输入文件夹: {IMAGE_FOLDER}")
    print(f"💾 输出文件: {OUTPUT_FILE}")
    print(f"🔄 强制旋转: {'是' if FORCE_ROTATE else '否'}\n")
    
    if not os.path.exists(IMAGE_FOLDER):
        print(f"❌ 错误: 文件夹不存在!")
        print(f"   {IMAGE_FOLDER}")
        return
    
    # ========== 步骤 1: 加载图像 ==========
    print("=" * 60)
    print("步骤 1: 加载图像")
    print("=" * 60)
    
    try:
        images, files = load_and_rotate_images(IMAGE_FOLDER, FORCE_ROTATE)
    except Exception as e:
        print(f"❌ 错误: {e}")
        return
    
    if len(images) < 2:
        print("❌ 错误: 至少需要2张图片!")
        return
    
    # ========== 步骤 2: 计算焦距 ==========
    print("=" * 60)
    print("步骤 2: 圆柱投影")
    print("=" * 60)
    
    if MANUAL_FOCAL:
        focal = MANUAL_FOCAL
        print(f"⚠ 使用手动焦距: {focal}px\n")
    else:
        focal_mm = get_focal_from_exif(files[0])
        if focal_mm is None:
            focal_mm = 8.2
            print(f"⚠ 使用默认焦距: {focal_mm}mm")
        
        # 旋转后，水平方向是1536px
        # 对应传感器宽度6.17mm（使用原始图片的宽度对应关系）
        rotated_width = images[0].shape[1]  # 1536
        focal = focal_mm * (rotated_width / SENSOR_WIDTH_MM)
        
        print(f"✓ 焦距: {focal_mm}mm → {focal:.0f}px")
        print(f"  (基于旋转后宽度 {rotated_width}px)\n")
    
    # 圆柱投影
    warped = []
    total = len(images)
    for i, img in enumerate(images):
        w = cylindrical_warp(img, focal)
        warped.append(w)
        if i % 5 == 0 or i == total - 1:
            print(f"  投影进度: {i+1}/{total}")
    print()
    
    # ========== 步骤 3: 特征匹配 ==========
    translations = align_images(warped)
    
    # ========== 步骤 4: 图像融合 ==========
    panorama = blend_images(warped, translations)
    
    # ========== 步骤 5: 裁剪黑边 ==========
    panorama = crop_black_borders(panorama)
    
    # ========== 步骤 6: 保存 ==========
    cv2.imwrite(OUTPUT_FILE, panorama)
    
    print("=" * 60)
    print("✅ 拼接完成!")
    print("=" * 60)
    print(f"✓ 全景图已保存: {OUTPUT_FILE}")
    print(f"  尺寸: {panorama.shape[1]} x {panorama.shape[0]}")
    print(f"  焦距: {focal:.0f}px")
    print("=" * 60 + "\n")
    
    # 显示统计
    total_dx = sum([t[0] for t in translations])
    print(f"📊 统计信息:")
    print(f"  图片数量: {len(images)}")
    print(f"  累计位移: {total_dx:.1f}px")
    print(f"  平均位移: {total_dx/(len(images)-1):.1f}px/帧")
    print()


if __name__ == "__main__":
    main()
