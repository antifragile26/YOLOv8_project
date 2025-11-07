"""
圆柱全景图拼接 - 增强版
新增功能：
1. 曝光补偿（增益补偿）
2. 色彩均衡
3. 增强特征匹配
4. 多频段融合（可选）
5. 接缝优化
"""

import cv2
import numpy as np
import os
import glob
from PIL import Image
from PIL.ExifTags import TAGS


# ==================== 配置区域 ====================
IMAGE_FOLDER = r"C:\Users\DELL\Desktop\image_understanding\project2\picture"
OUTPUT_FILE = "panorama.jpg"

# 传感器参数
SENSOR_WIDTH_MM = 6.17
SENSOR_HEIGHT_MM = 4.63

# 强制旋转
FORCE_ROTATE = True

# 焦距（None=自动）
MANUAL_FOCAL = None

# 羽化宽度
BLEND_WIDTH = 300

# ⭐⭐ 新增：图像调整选项 ⭐⭐
ENABLE_EXPOSURE_COMPENSATION = True  # 曝光补偿
ENABLE_COLOR_BALANCE = True          # 色彩均衡
EXPOSURE_METHOD = "gain"             # "gain" 或 "histogram"

# ⭐⭐ 特征匹配参数 ⭐⭐
SIFT_FEATURES = 2000  # SIFT特征点数量（增加）
MATCH_RATIO = 0.7     # 匹配比率（更严格）
RANSAC_THRESHOLD = 4.0  # RANSAC阈值

# 调试输出
VERBOSE = True  # 显示详细信息
SAVE_DEBUG = False  # 保存调试图像
# ==================== 配置结束 ====================


def log(msg):
    """日志输出"""
    if VERBOSE:
        print(msg)


def get_focal_from_exif(image_path):
    """从EXIF读取焦距"""
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
    
    files = list(set(files))
    files = sorted(files)
    
    if not files:
        raise FileNotFoundError(f"在 {folder_path} 中未找到图片文件")

    log(f"找到 {len(files)} 个图片文件\n")
    
    images = []
    for f in files:
        img = cv2.imread(f)
        if img is not None:
            h_orig, w_orig = img.shape[:2]
            
            if force_rotate:
                img = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
                h_new, w_new = img.shape[:2]
                log(f"✓ {os.path.basename(f)}: {w_orig}x{h_orig} → {w_new}x{h_new}")
            else:
                log(f"✓ {os.path.basename(f)}: {w_orig}x{h_orig}")
            
            images.append(img)

    if not images:
        raise ValueError("无法加载任何有效图片")

    print(f"\n✅ 加载了 {len(images)} 张图片")
    print(f"📐 最终尺寸: {images[0].shape[1]} x {images[0].shape[0]}\n")
    return images, files


def compute_brightness(img):
    """计算图像亮度"""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return np.mean(gray)


def exposure_compensation_gain(images):
    """曝光补偿 - 增益方法"""
    print("=" * 60)
    print("曝光补偿（增益调整）")
    print("=" * 60)
    
    # 计算每张图片的平均亮度
    brightnesses = [compute_brightness(img) for img in images]
    target_brightness = np.median(brightnesses)
    
    log(f"目标亮度: {target_brightness:.1f}")
    
    compensated = []
    for i, img in enumerate(images):
        gain = target_brightness / brightnesses[i]
        # 限制增益范围，避免过曝或过暗
        gain = np.clip(gain, 0.5, 2.0)
        
        adjusted = cv2.convertScaleAbs(img, alpha=gain, beta=0)
        compensated.append(adjusted)
        
        log(f"  图像 {i}: 亮度 {brightnesses[i]:.1f} → 增益 {gain:.2f}")
    
    print("✓ 曝光补偿完成\n")
    return compensated


def color_balance(images):
    """色彩均衡 - 灰度世界算法"""
    print("=" * 60)
    print("色彩均衡")
    print("=" * 60)
    
    balanced = []
    
    # 计算参考图像的通道均值
    ref_img = images[len(images) // 2]  # 使用中间图像作为参考
    ref_means = cv2.mean(ref_img)[:3]
    
    log(f"参考RGB均值: ({ref_means[2]:.1f}, {ref_means[1]:.1f}, {ref_means[0]:.1f})")
    
    for i, img in enumerate(images):
        means = cv2.mean(img)[:3]
        
        # 计算每个通道的缩放因子
        scales = [ref_means[j] / (means[j] + 1e-6) for j in range(3)]
        scales = [np.clip(s, 0.7, 1.5) for s in scales]  # 限制范围
        
        # 应用缩放
        result = img.astype(np.float32)
        for c in range(3):
            result[:, :, c] = np.clip(result[:, :, c] * scales[c], 0, 255)
        
        balanced.append(result.astype(np.uint8))
        
        log(f"  图像 {i}: RGB缩放 ({scales[2]:.2f}, {scales[1]:.2f}, {scales[0]:.2f})")
    
    print("✓ 色彩均衡完成\n")
    return balanced


def cylindrical_warp(img, focal):
    """圆柱投影"""
    h, w = img.shape[:2]
    
    y_i, x_i = np.indices((h, w))
    X = np.stack([
        (x_i - w / 2) / focal,
        (y_i - h / 2) / focal,
        np.ones_like(x_i)
    ], axis=-1)
    
    theta = np.arctan(X[..., 0])
    h_ = X[..., 1] / np.sqrt(X[..., 0] ** 2 + 1)
    
    x_ = focal * theta + w / 2
    y_ = focal * h_ + h / 2
    
    mapx = x_.astype(np.float32)
    mapy = y_.astype(np.float32)
    return cv2.remap(img, mapx, mapy, cv2.INTER_LINEAR)


def extract_features(img):
    """增强SIFT特征提取"""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # 使用更多特征点
    sift = cv2.SIFT_create(
        nfeatures=SIFT_FEATURES,
        contrastThreshold=0.03,  # 降低对比度阈值，提取更多特征
        edgeThreshold=10,
        sigma=1.6
    )
    
    kp, des = sift.detectAndCompute(gray, None)
    return kp, des


def match_features(des1, des2):
    """增强特征匹配"""
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(des1, des2, k=2)
    
    # Lowe's ratio test（更严格）
    good = []
    for m, n in matches:
        if m.distance < MATCH_RATIO * n.distance:
            good.append(m)
    
    return good


def ransac_horizontal(pts1, pts2, max_iter=2000, thresh=RANSAC_THRESHOLD):
    """RANSAC估算水平位移 - 增强版"""
    best_inliers = []
    best_trans = (0, 0)
    best_err = float('inf')
    n = len(pts1)
    
    if n < 4:
        return (0, 0), 0
    
    for _ in range(max_iter):
        # 随机选择样本
        idx = np.random.randint(n)
        dx = pts2[idx, 0] - pts1[idx, 0]
        
        # 计算误差
        pred_x = pts1[:, 0] + dx
        errs = np.abs(pred_x - pts2[:, 0])
        inliers = np.where(errs < thresh)[0]
        
        if len(inliers) > len(best_inliers):
            # 使用内点重新计算
            dx_refined = np.median(pts2[inliers, 0] - pts1[inliers, 0])
            pred_x_refined = pts1[inliers, 0] + dx_refined
            err = np.mean(np.abs(pred_x_refined - pts2[inliers, 0]))
            
            best_trans = (dx_refined, 0)
            best_inliers = inliers
            best_err = err
    
    return best_trans, len(best_inliers)


def align_images(warped_images):
    """对齐相邻图像 - 增强版"""
    print("=" * 60)
    print("特征提取与配准")
    print("=" * 60)
    
    features = []
    for i, img in enumerate(warped_images):
        kp, des = extract_features(img)
        features.append((kp, des))
        log(f"图像 {i}: {len(kp)} 特征点")
    
    print("\n相邻帧配准:")
    translations = []
    
    for i in range(len(warped_images) - 1):
        kp1, des1 = features[i]
        kp2, des2 = features[i + 1]
        
        matches = match_features(des1, des2)
        
        if len(matches) < 10:
            print(f"  {i}↔{i+1}: ⚠ 匹配点不足 ({len(matches)})")
            translations.append((0, 0, 0))
            continue
        
        pts1 = np.float32([kp1[m.queryIdx].pt for m in matches])
        pts2 = np.float32([kp2[m.trainIdx].pt for m in matches])
        
        trans, inliers = ransac_horizontal(pts1, pts2)
        translations.append((trans[0], trans[1], inliers))
        
        # 显示详细信息
        match_rate = inliers / len(matches) * 100 if len(matches) > 0 else 0
        print(f"  {i}↔{i+1}: dx={trans[0]:7.1f}px, 匹配={len(matches):3d}, 内点={inliers:3d} ({match_rate:.1f}%)")
    
    print()
    return translations


def multiband_blend(img1, img2, mask1, mask2, levels=4):
    """多频段融合（拉普拉斯金字塔）"""
    # 构建高斯金字塔
    gpA = [img1]
    gpB = [img2]
    gpM1 = [mask1]
    gpM2 = [mask2]
    
    for i in range(levels):
        gpA.append(cv2.pyrDown(gpA[i]))
        gpB.append(cv2.pyrDown(gpB[i]))
        gpM1.append(cv2.pyrDown(gpM1[i]))
        gpM2.append(cv2.pyrDown(gpM2[i]))
    
    # 构建拉普拉斯金字塔
    lpA = [gpA[levels]]
    lpB = [gpB[levels]]
    
    for i in range(levels, 0, -1):
        size = (gpA[i-1].shape[1], gpA[i-1].shape[0])
        LA = cv2.subtract(gpA[i-1], cv2.pyrUp(gpA[i], dstsize=size))
        LB = cv2.subtract(gpB[i-1], cv2.pyrUp(gpB[i], dstsize=size))
        lpA.append(LA)
        lpB.append(LB)
    
    # 融合
    LS = []
    for la, lb, m1, m2 in zip(lpA, lpB, gpM1[::-1], gpM2[::-1]):
        ls = la * m1 + lb * m2
        LS.append(ls)
    
    # 重建
    result = LS[0]
    for i in range(1, len(LS)):
        size = (LS[i].shape[1], LS[i].shape[0])
        result = cv2.add(cv2.pyrUp(result, dstsize=size), LS[i])
    
    return result


def blend_images_advanced(warped_images, translations):
    """高级图像融合"""
    print("=" * 60)
    print("图像融合（高级）")
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
    
    # 增强羽化
    fade_width = min(BLEND_WIDTH, w // 2)
    print(f"羽化宽度: {fade_width}px")
    
    for i, img in enumerate(warped_images):
        x_off = x_offsets[i]
        y1, y2 = 0, h
        x1, x2 = x_off, x_off + w
        
        # 创建渐变权重（余弦插值）
        weight = np.ones((h, w, 1), dtype=np.float32)
        
        # 左边羽化
        for j in range(min(fade_width, w)):
            alpha = 0.5 * (1 - np.cos(j / fade_width * np.pi))
            weight[:, j] = alpha
        
        # 右边羽化
        for j in range(min(fade_width, w)):
            alpha = 0.5 * (1 - np.cos(j / fade_width * np.pi))
            col_idx = w - 1 - j
            if col_idx >= 0:
                weight[:, col_idx] = alpha
        
        # 在重叠区域应用接缝优化
        if i > 0:
            # 查找前一张图的边界
            prev_x_off = x_offsets[i-1]
            overlap_start = max(x_off, prev_x_off)
            overlap_end = min(x_off + w, prev_x_off + w)
            
            if overlap_end > overlap_start:
                # 在重叠区域增强羽化
                overlap_width = overlap_end - overlap_start
                for x in range(overlap_width):
                    img_x = x + (overlap_start - x_off)
                    if 0 <= img_x < w:
                        # 使用距离到边界的比例作为权重
                        alpha = x / overlap_width
                        weight[:, img_x] = alpha
        
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
    print("圆柱全景图拼接 - 增强版")
    print("=" * 60 + "\n")
    
    print(f"📁 输入: {IMAGE_FOLDER}")
    print(f"💾 输出: {OUTPUT_FILE}")
    print(f"🔄 旋转: {'是' if FORCE_ROTATE else '否'}")
    print(f"🎨 曝光补偿: {'是' if ENABLE_EXPOSURE_COMPENSATION else '否'}")
    print(f"🌈 色彩均衡: {'是' if ENABLE_COLOR_BALANCE else '否'}\n")
    
    if not os.path.exists(IMAGE_FOLDER):
        print(f"❌ 错误: 文件夹不存在!")
        return
    
    # 步骤 1: 加载
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
    
    # 步骤 2: 曝光补偿
    if ENABLE_EXPOSURE_COMPENSATION:
        images = exposure_compensation_gain(images)
    
    # 步骤 3: 色彩均衡
    if ENABLE_COLOR_BALANCE:
        images = color_balance(images)
    
    # 步骤 4: 圆柱投影
    print("=" * 60)
    print("步骤 4: 圆柱投影")
    print("=" * 60)
    
    if MANUAL_FOCAL:
        focal = MANUAL_FOCAL
        print(f"⚠ 使用手动焦距: {focal}px\n")
    else:
        focal_mm = get_focal_from_exif(files[0])
        if focal_mm is None:
            focal_mm = 8.2
        
        rotated_width = images[0].shape[1]
        focal = focal_mm * (rotated_width / SENSOR_WIDTH_MM)
        print(f"✓ 焦距: {focal_mm}mm → {focal:.0f}px\n")
    
    warped = []
    total = len(images)
    for i, img in enumerate(images):
        w = cylindrical_warp(img, focal)
        warped.append(w)
        if i % 5 == 0 or i == total - 1:
            log(f"  投影进度: {i+1}/{total}")
    print()
    
    # 步骤 5: 特征匹配
    translations = align_images(warped)
    
    # 步骤 6: 高级融合
    panorama = blend_images_advanced(warped, translations)
    
    # 步骤 7: 裁剪
    panorama = crop_black_borders(panorama)
    
    # 步骤 8: 保存
    cv2.imwrite(OUTPUT_FILE, panorama)
    
    print("=" * 60)
    print("✅ 拼接完成!")
    print("=" * 60)
    print(f"✓ 保存: {OUTPUT_FILE}")
    print(f"  尺寸: {panorama.shape[1]} x {panorama.shape[0]}")
    print(f"  焦距: {focal:.0f}px")
    print("=" * 60 + "\n")
    
    # 统计
    total_dx = sum([t[0] for t in translations])
    avg_dx = total_dx / (len(images) - 1) if len(images) > 1 else 0
    print(f"📊 统计:")
    print(f"  图片: {len(images)}")
    print(f"  特征点: {SIFT_FEATURES}")
    print(f"  累计位移: {total_dx:.1f}px")
    print(f"  平均位移: {avg_dx:.1f}px/帧")
    print()


if __name__ == "__main__":
    main()
