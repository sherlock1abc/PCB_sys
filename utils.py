import cv2
import numpy as np
import pandas as pd
import streamlit as st

# 定义一些好看的颜色 (B, G, R) - 虽然我们现在强制用红色，但保留列表以备后用
COLORS = [
    (255, 0, 0), (0, 255, 0), (0, 0, 255),
    (255, 255, 0), (255, 0, 255), (0, 255, 255),
    (128, 0, 0), (0, 128, 0), (0, 0, 128),
    (128, 128, 0), (128, 0, 128), (0, 128, 128)
]


def draw_boxes(image, boxes, classes, confs, class_names):
    """
    在图像上绘制检测框
    修改点：强制使用红色框，加大字体和线条
    """
    # 如果没有检测到目标，直接返回原图
    if boxes is None or len(boxes) == 0:
        return image

    # 确保 boxes 是整数坐标
    boxes = boxes.astype(np.int32)

    # --- 设置样式参数 ---
    color = (0, 0, 255)  # 强制红色 (B, G, R)
    line_thickness = 3   # 线条加粗
    font_scale = 2     # 字体放大 (原为 0.6)
    font_thickness = 2   # 文字加粗

    for i, box in enumerate(boxes):
        # 获取类别和置信度
        cls_id = int(classes[i])
        conf = confs[i]

        # 获取标签名称
        label = f"{class_names[cls_id]} {conf:.2f}"

        # 获取坐标
        x1, y1, x2, y2 = box

        # 1. 绘制矩形框 (红色，加粗)
        cv2.rectangle(image, (x1, y1), (x2, y2), color, thickness=line_thickness)

        # 2. 计算文本大小
        (text_width, text_height), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness)

        # 3. 绘制标签背景 (红色实心矩形)
        # 确保背景框在图片范围内
        y_bg_top = max(y1 - text_height - 10, 0)
        cv2.rectangle(image, (x1, y_bg_top), (x1 + text_width, y1), color, -1)

        # 4. 绘制标签文字 (白色，大字号)
        cv2.putText(image, label, (x1, y_bg_top + text_height), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), font_thickness)

    return image


def parse_results(results, class_names, inference_time):
    """
    解析 YOLO 结果并返回 DataFrame
    包含：检测用时、目标数、以及每个目标的详细坐标(Xmin, Xmax, Ymin, Ymax)
    """
    data = []
    target_count = 0

    # results 是一个列表，通常包含一张图的结果
    # 我们取第一个元素 results[0]
    if len(results) > 0:
        result = results[0]

        # 获取检测框信息
        boxes = result.boxes.xyxy.cpu().numpy()
        confs = result.boxes.conf.cpu().numpy()
        classes = result.boxes.cls.cpu().numpy()

        target_count = len(boxes)

        for i in range(target_count):
            x1, y1, x2, y2 = boxes[i]

            # 确保 Xmin < Xmax, Ymin < Ymax
            x_min, x_max = sorted([x1, x2])
            y_min, y_max = sorted([y1, y2])

            data.append({
                "目标编号": i + 1,
                "类别": class_names[int(classes[i])],
                "置信度": f"{confs[i]:.4f}",
                "Xmin": int(x_min),
                "Xmax": int(x_max),
                "Ymin": int(y_min),
                "Ymax": int(y_max)
            })

    # 创建 DataFrame
    df = pd.DataFrame(data)

    # 如果没有检测到目标，也要显示用时信息
    if df.empty:
        st.info(f"检测完成，用时: {inference_time:.2f}秒，未发现目标。")
        return pd.DataFrame([{"检测信息": f"用时: {inference_time:.2f}秒", "状态": "未发现目标"}])

    return df