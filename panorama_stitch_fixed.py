"""
圆柱全景图拼接 - 完全修复版
1. 自动旋转竖向拍摄的图片
2. 修复Windows文件重复识别bug
3. 防止焦距异常
"""

import cv2
import numpy as np
import os
import glob
from PIL import Image
from PIL.ExifTags import TAGS


# ==================== 配置区域 ====================
# 👇 修改这里的路径
IMAGE_FOLDER = r"C:\Users\DELL\Desktop\image_understanding\project2\picture"
OUTPUT_FILE = "panorama.jpg"
SENSOR_WIDTH = 6.17  # mm (尼康 Coolpix 1/2.3" 传感器)
# ==================== 配置结束 ====================


def get_exif_data(image_path):
    """读取 EXIF 信息"""
    try:
        img = Image.open(image_path)
        exif = img._getexif()
        focal_mm = None
        orientation = 1
        if exif:
            for tag_id, value in exif.items():
                tag = TAGS.get(tag_id, tag_id)
                if tag == "FocalLength":
                    if isinstance(value, tuple):
                        focal_mm = value[0] / value[1]
                    else:
                        focal_mm = float(value)
                if tag == "Orientation":
                    orientation = value
        return focal_mm, orientation
    except Exception as e:
        print(f"⚠ 读取 EXIF 失败: {e}")
        return None, 1


def rotate_image_by_orientation(img, orientation):
    """根据EXIF Orientation旋转图像"""
    if orientation == 1:
        return img
    elif orientation == 3:
        return cv2.rotate(img, cv2.ROTATE_180)
    elif orientation == 6:
        return cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
    elif orientation == 8:
        return cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
    else:
        return img


def load_images(folder_path):
    """加载图像（已修复Windows重复识别bug）"""
    extensions = ['*.JPG', '*.jpg', '*.JPEG', '*.jpeg', '*.PNG', '*.png']
    files = []
    
    for ext in extensions:
        pattern = os.path.join(folder_path, ext)
        files.extend(glob.glob(pattern))
    
    # ⭐ 关键修复：去重（Windows系统不区分大小写）
    files = list(set(files))
    files = sorted(files)
    
    if not files:
        raise FileNotFoundError(f"在 {folder_path} 中未找到图片文件")

    images = []
    valid_files = []
    orientations = []
    
    print(f"找到 {len(files)} 个图片文件\n")
    
    for f in files:
        img = cv2.imread(f)
        if img is not None:
            _, orientation = get_exif_data(f)
            img = rotate_image_by_orientation(img, orientation)
            images.append(img)
            valid_files.append(f)
            orientations.append(orientation)
            
            status = f" (旋转 Orientation={orientation})" if orientation != 1 else ""
            print(f"✓ {os.path.basename(f)}{status}")

    if not images:
        raise ValueError("无法加载任何有效图片")

    print(f"\n✅ 成功加载 {len(images)} 张图片")
    print(f"📐 图片尺寸: {images[0].shape[1]} x {images[0].shape[0]}\n")
    return images, valid_files, orientations


def cylindrical_warp(img, f):
    """圆柱投影变换"""
    h, w = img.shape[:2]
    y_i, x_i = np.indices((h, w))
    X = np.stack([(x_i - w / 2) / f, (y_i - h / 2) / f, np.ones_like(x_i)], axis=-1)
    theta = np.arctan(X[..., 0])
    h_ = X[..., 1] / np.sqrt(X[..., 0] ** 2 + 1)
    x_ = f * theta + w / 2
    y_ = f * h_ + h / 2
    mapx = x_.astype(np.float32)
    mapy = y_.astype(np.float32)
    return cv2.remap(img, mapx, mapy, cv2.INTER_LINEAR)


def extract_features(img):
    """SIFT 特征提取"""
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


def ransac_translation_horizontal(pts1, pts2, max_iter=1000, thresh=3.0):
    """RANSAC 只估算水平位移"""
    best_inliers = []
    best_trans = (0, 0)
    best_err = float("inf")
    n = len(pts1)
    
    for _ in range(max_iter):
        idx = np.random.randint(n)
        dx = pts2[idx, 0] - pts1[idx, 0]
        pred_x = pts1[:, 0] + dx
        errs = np.abs(pred_x - pts2[:, 0])
        inliers = np.where(errs < thresh)[0]

        if len(inliers) > len(best_inliers):
            dx_ref = np.median(pts2[inliers, 0] - pts1[inliers, 0])
            pred_x_ref = pts1[inliers, 0] + dx_ref
            err = np.mean(np.abs(pred_x_ref - pts2[inliers, 0]))
            best_trans = (dx_ref, 0)
            best_inliers = inliers
            best_err = err

    return best_trans, len(best_inliers), best_err


def align_neighboring_pairs(warped_images):
    """对齐相邻图像"""
    print("=" * 50)
    print("特征提取与配准")
    print("=" * 50)

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
            print(f"  {i}↔{i+1}: 匹配点不足!")
            translations.append((0, 0, 0, 999))
            continue

        pts1 = np.float32([kp1[m.queryIdx].pt for m in matches])
        pts2 = np.float32([kp2[m.trainIdx].pt for m in matches])
        trans, inliers, err = ransac_translation_horizontal(pts1, pts2)
        translations.append((trans[0], trans[1], inliers, err))
        print(f"  {i}↔{i+1}: dx={trans[0]:.1f}, dy={trans[1]:.1f}, 内点={inliers}, 误差={err:.2f}px")

    return translations, features


def save_translation_list(translations, output_file="trans.txt"):
    """保存平移列表"""
    with open(output_file, "w", encoding='utf-8') as f:
        f.write("# i  j  dx  dy  inliers  reproj_err\n")
        for i, (dx, dy, inliers, err) in enumerate(translations):
            f.write(f"{i}  {i+1}  {dx:.2f}  {dy:.2f}  {inliers}  {err:.3f}\n")
    print(f"\n✓ 平移列表已保存: {output_file}\n")


def match_first_last(warped_images, features):
    """匹配首尾图像"""
    kp1, des1 = features[0]
    kp2, des2 = features[-1]
    matches = match_features(des1, des2)
    if len(matches) < 10:
        return (0, 0)
    pts1 = np.float32([kp1[m.queryIdx].pt for m in matches])
    pts2 = np.float32([kp2[m.trainIdx].pt for m in matches])
    trans, _, _ = ransac_translation_horizontal(pts1, pts2)
    return trans


def correct_drift(translations, first_last_trans, focal):
    """漂移校正"""
    print("=" * 50)
    print("漂移校正")
    print("=" * 50)

    cumsum = sum([t[0] for t in translations])
    actual = first_last_trans[0]
    gap_dx = cumsum - actual
    theta_g = gap_dx / focal

    print(f"累计位移: {cumsum:.1f}px")
    print(f"首尾位移: {actual:.1f}px")
    print(f"角度缺口: {np.degrees(theta_g):.2f}°")

    if abs(np.degrees(theta_g)) > 180.0:
        print("⚠ 角度缺口异常大，跳过漂移校正\n")
        return translations, focal

    if abs(np.degrees(theta_g)) < 5.0:
        print("缺口较小，跳过校正\n")
        return translations, focal

    n = len(translations) + 1
    theta_per = theta_g / n
    dx_per = theta_per * focal
    print(f"每帧校正: {np.degrees(theta_per):.4f}° ({dx_per:.2f}px)")

    corrected = []
    for i, (dx, dy, inliers, err) in enumerate(translations):
        correction = (i + 1) * dx_per
        corrected.append((dx - correction, dy, inliers, err))

    focal_new = focal * (1 - theta_g / (2 * np.pi))
    
    if focal_new < focal * 0.5 or focal_new > focal * 1.5:
        print(f"⚠ 焦距变化异常，保持原焦距\n")
        return translations, focal
    
    print(f"焦距更新: {focal:.1f} → {focal_new:.1f}px\n")
    return corrected, focal_new


def blend_images(warped_images, translations):
    """图像融合"""
    print("=" * 50)
    print("图像融合")
    print("=" * 50)

    h, w = warped_images[0].shape[:2]
    x_offsets = [0]
    cum_x = 0

    for dx, _, _, _ in translations:
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

    for i, img in enumerate(warped_images):
        x_off = x_offsets[i]
        y1, y2 = 0, h
        x1, x2 = x_off, x_off + w

        weight = np.ones((h, w, 1), dtype=np.float32)
        fade = min(50, w // 4)
        for j in range(fade):
            alpha = j / fade
            weight[:, j] = alpha
            weight[:, -(j + 1)] = alpha

        canvas[y1:y2, x1:x2] += img.astype(np.float32) * weight
        count[y1:y2, x1:x2] += weight

    count[count == 0] = 1
    panorama = (canvas / count).astype(np.uint8)
    print("✓ 融合完成\n")
    return panorama


def crop_panorama(pano):
    """裁剪黑边"""
    print("=" * 50)
    print("裁剪黑边")
    print("=" * 50)

    gray = cv2.cvtColor(pano, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        return pano

    x, y, w, h = cv2.boundingRect(max(contours, key=cv2.contourArea))
    margin = 5
    x, y = max(0, x + margin), max(0, y + margin)
    w, h = w - 2 * margin, h - 2 * margin
    cropped = pano[y : y + h, x : x + w]
    print(f"裁剪: {pano.shape} → {cropped.shape}\n")
    return cropped


def main():
    print("\n" + "=" * 60)
    print("圆柱全景图拼接 - 完全修复版")
    print("=" * 60 + "\n")
    print(f"📁 输入文件夹: {IMAGE_FOLDER}")
    print(f"💾 输出文件: {OUTPUT_FILE}\n")

    if not os.path.exists(IMAGE_FOLDER):
        print(f"❌ 错误: 文件夹不存在: {IMAGE_FOLDER}")
        print("请修改代码开头的 IMAGE_FOLDER 路径")
        return

    # 步骤 1: 加载
    print("=" * 60)
    print("步骤 1: 加载图像")
    print("=" * 60)
    
    try:
        images, files, orientations = load_images(IMAGE_FOLDER)
    except Exception as e:
        print(f"❌ 错误: {e}")
        return

    if len(images) < 2:
        print("❌ 错误: 至少需要 2 张图片!")
        return

    # 步骤 2: 圆柱投影
    print("=" * 60)
    print("步骤 2: 圆柱投影")
    print("=" * 60)

    try:
        focal_mm, _ = get_exif_data(files[0])
    except:
        focal_mm = None
    
    if focal_mm is None:
        focal_mm = 8.2
        print(f"⚠ 无法读取EXIF焦距，使用默认值: {focal_mm}mm")
    
    image_width = images[0].shape[1]
    focal = focal_mm * (image_width / SENSOR_WIDTH)
    print(f"✓ 焦距: {focal_mm:.1f}mm → {focal:.0f}px (图像宽度={image_width}px)\n")

    warped = []
    for i, img in enumerate(images):
        w = cylindrical_warp(img, focal)
        warped.append(w)
        if i % 5 == 0 or i == len(images) - 1:
            print(f"✓ 已投影 {i+1}/{len(images)}")
    print()

    # 步骤 3-4: 配准
    translations, features = align_neighboring_pairs(warped)
    save_translation_list(translations, "trans.txt")

    # 步骤 5: 漂移校正
    first_last = match_first_last(warped, features)
    translations_corrected, focal_new = correct_drift(translations, first_last, focal)

    if abs(focal_new - focal) > 1.0:
        print("=" * 60)
        print("使用新焦距重新投影")
        print("=" * 60)
        warped = []
        for i, img in enumerate(images):
            w = cylindrical_warp(img, focal_new)
            warped.append(w)
            if i % 5 == 0 or i == len(images) - 1:
                print(f"✓ 已投影 {i+1}/{len(images)}")
        print()
        save_translation_list(translations_corrected, "trans_corrected.txt")
        final_trans = translations_corrected
    else:
        final_trans = translations

    # 步骤 6: 融合
    panorama = blend_images(warped, final_trans)

    # 步骤 7: 裁剪
    panorama = crop_panorama(panorama)

    # 保存
    cv2.imwrite(OUTPUT_FILE, panorama)

    print("=" * 60)
    print("✅ 完成!")
    print("=" * 60)
    print(f"✓ 全景图已保存: {OUTPUT_FILE}")
    print(f"  尺寸: {panorama.shape[1]} x {panorama.shape[0]}")
    print(f"  最终焦距: {focal_new:.0f}px")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
