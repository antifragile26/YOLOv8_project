# 修复说明 - 竖向拍摄问题

## 问题诊断

根据您的运行日志，发现以下关键问题：

### 1. **焦距变成负数** ❌
```
焦距更新: 2041.4 → -391.6px
```
这是致命错误，导致第二次圆柱投影完全错误。

### 2. **角度缺口异常大** ❌
```
角度缺口: 429.05°
```
超过了一整圈（360°），说明首尾匹配出现严重错误。

### 3. **图像被严重裁剪** ❌
```
画布大小: 117042 x 2048
裁剪: (2048, 117042, 3) → (2038, 2367, 3)
```
从117042px宽裁到只有2367px，说明大部分区域是黑色的。

### 4. **根本原因** 🔍
- 您的图片是**竖向拍摄**的 (1536 x 2048)
- 虽然代码检测到了 `Orientation=8`（需要逆时针旋转90度）
- **但在圆柱投影前没有实际旋转图像**
- 导致投影方向错误，拼接完全混乱

## 修复内容

### ✅ 修复1: 添加图像旋转功能

```python
def rotate_image_by_orientation(img, orientation):
    """根据EXIF Orientation旋转图像"""
    if orientation == 1:
        return img  # 正常，不需要旋转
    elif orientation == 3:
        return cv2.rotate(img, cv2.ROTATE_180)
    elif orientation == 6:
        return cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
    elif orientation == 8:
        return cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
    else:
        return img
```

### ✅ 修复2: 在加载时旋转图像

修改 `load_images()` 函数：
- 读取每张图片的EXIF Orientation信息
- **立即旋转图像**到正确方向
- 返回旋转后的图像用于后续处理

### ✅ 修复3: 基于旋转后的尺寸计算焦距

```python
# 使用旋转后的图像宽度计算焦距
image_width = images[0].shape[1]  # 旋转后是2048px
focal = focal_mm * (image_width / sensor_width)
```

### ✅ 修复4: 防止焦距异常

在 `correct_drift()` 函数中添加保护：
```python
# 如果角度缺口超过180度，说明首尾匹配可能有问题，跳过校正
if abs(np.degrees(theta_g)) > 180.0:
    print("⚠ 角度缺口异常大，可能是首尾匹配错误，跳过漂移校正\n")
    return translations, focal

# 防止焦距变成负数或异常值
if focal_new < focal * 0.5 or focal_new > focal * 1.5:
    print(f"⚠ 焦距变化异常，保持原焦距\n")
    return translations, focal
```

## 预期效果

修复后，运行应该会看到：

```
✓ 100NIKON-DSCN0008_DSCN0008.JPG (旋转 Orientation=8)
✓ 100NIKON-DSCN0009_DSCN0009.JPG (旋转 Orientation=8)
...

加载了 18 张图片
图片尺寸: 2048 x 1536    <-- 注意：宽度>高度

✓ 焦距: 8.2mm → 2041px (旋转后图像宽度=2048px)

相邻帧配准:
  0↔1: dx=约400-600, dy=0.0, ...    <-- dx应该变小
  ...

角度缺口: <10°    <-- 应该很小
```

## 使用方法

直接运行修复后的代码：

```bash
python cylindrical_panorama_stitching.py --input /path/to/images --output panorama.jpg
```

不需要任何额外参数，程序会自动：
1. 检测图片方向
2. 旋转到正确方向
3. 计算正确的焦距
4. 拼接全景图

## 技术细节

### EXIF Orientation 值含义
- `1`: 正常 (0°)
- `3`: 旋转180°
- `6`: 顺时针旋转90° (相机竖向拍摄，顶部在右侧)
- `8`: 逆时针旋转90° (相机竖向拍摄，顶部在左侧)

### 为什么竖向拍摄会出问题？

圆柱全景拼接假设：
- 图像的**宽度方向**对应场景的**水平方向**
- 图像的**高度方向**对应场景的**垂直方向**

竖向拍摄时（Orientation=8）：
- 原始图像：1536(宽) x 2048(高)
- 实际拍摄：相机旋转了90度
- **如果不旋转图像**，程序会把场景的"垂直"当成"水平"
- 导致投影和拼接完全错误

## 测试建议

1. 运行修复后的代码
2. 检查输出日志中的：
   - 图片尺寸是否正确 (应该是 2048 x 1536)
   - 角度缺口是否合理 (<10°)
   - 最终焦距是否为正数
3. 查看 `panorama.jpg` 是否符合预期

如果还有问题，请提供新的运行日志！
