import torch
import cv2
import os
from models.experimental import attempt_load
from utils.general import non_max_suppression

def scale_coords(img1_shape, coords, img0_shape):
    # img1_shape: 模型输入图像的尺寸 (640, 640)
    # coords: 检测框的坐标 (x1, y1, x2, y2)
    # img0_shape: 原始图像的尺寸 (h, w)
    gain = min(img1_shape[0] / img0_shape[0], img1_shape[1] / img0_shape[1])
    pad = (img1_shape[1] - img0_shape[1] * gain, img1_shape[0] - img0_shape[0] * gain)  # wh padding
    coords[:, [0, 2]] -= pad[0] / 2  # x padding
    coords[:, [1, 3]] -= pad[1] / 2  # y padding
    coords[:, :4] /= gain
    clip_coords(coords, img0_shape)
    return coords

def clip_coords(boxes, img_shape):
    # 将坐标限制在图像范围内
    boxes[:, 0].clamp_(0, img_shape[1])  # x1
    boxes[:, 1].clamp_(0, img_shape[0])  # y1
    boxes[:, 2].clamp_(0, img_shape[1])  # x2
    boxes[:, 3].clamp_(0, img_shape[0])  # y2
    return boxes


# 加载训练好的模型
weights = 'weights/best.pt'  # 或者 'weights/last.pt'
model = attempt_load(weights)

# 读取图片
image_folder = 'image'
image_files = [f for f in os.listdir(image_folder) if f.endswith('.jpeg')]
for image_file in image_files:
    image_path = os.path.join(image_folder, image_file)
    img0 = cv2.imread(image_path)

    # 图片尺寸调整为模型输入的尺寸
    img = cv2.resize(img0, (640, 640))

    # 图片转换为RGB格式
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # 图片转换为PyTorch张量
    img = torch.from_numpy(img.transpose((2, 0, 1))).float()

    # 标准化图像
    img /= 255.0

    # 添加批次维度
    img = img.unsqueeze(0)

    # 模型推理
    pred = model(img)[0]
    pred = non_max_suppression(pred, conf_thres=0.5, iou_thres=0.5)[0]

    # 打印识别的坐标结果
    print(f"识别图片 {image_file} 的结果：")
    if pred is not None and len(pred):
        for det in pred:
            # 提取坐标信息
            xyxy = det[:4].cpu().numpy()  # 提取坐标信息
            print("原始坐标信息：", xyxy)

            # 将坐标映射到原始图像尺寸
            h, w, _ = img0.shape
            xyxy_scaled = scale_coords((640, 640), xyxy, (h, w))
            print("映射后的坐标信息：", xyxy_scaled)

            # 绘制边界框和标签
            for x1, y1, x2, y2, conf, cls in xyxy_scaled:
                cv2.rectangle(img0, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 2)
                cv2.putText(img0, f"{cls}: {conf:.2f}", (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                            (255, 0, 0), 2)

    # 保存绘制了坐标的新图像
    output_path = os.path.join('output', f'detected_{image_file}')
    cv2.imwrite(output_path, img0)
    print(f"已将识别结果保存到 {output_path}")
