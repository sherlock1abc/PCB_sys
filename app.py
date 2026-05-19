import streamlit as st
import cv2
import numpy as np
from PIL import Image
from ultralytics import YOLO
import pandas as pd
import os
import time
from datetime import datetime

# 导入数据库模块 (保留 txt 原有功能)
from database import init_db, save_record, get_all_records
from utils import draw_boxes, parse_results

# --- 初始化数据库 ---
init_db()


# ==========================================
# 🚀 新增功能：辅助函数 (来自第二段代码)
# ==========================================

def get_next_file_index():
    """扫描 result 文件夹，返回下一个可用的序号"""
    if not os.path.exists("result"):
        os.makedirs("result")
        return 1

    files = os.listdir("result")
    indices = []
    for f in files:
        # 兼容图片和 Excel 的序号逻辑
        if f.endswith(".xlsx") or f.endswith(('.jpg', '.jpeg', '.png', '.bmp')):
            try:
                idx = int(f.split("_")[0])
                indices.append(idx)
            except ValueError:
                continue
    if not indices:
        return 1
    return max(indices) + 1


def save_to_excel(results_data, filename_prefix):
    """
    将检测结果保存为 Excel
    results_data: 包含检测信息的列表字典
    filename_prefix: 文件名前缀
    """
    if not results_data:
        return None

    # 1. 生成文件名
    current_idx = get_next_file_index()
    time_str = datetime.now().strftime("%m%d_%H%M")
    excel_name = f"{current_idx}_{time_str}.xlsx"
    excel_path = os.path.join("result", excel_name)

    # 2. 转换为 DataFrame
    df = pd.DataFrame(results_data)

    # 3. 保存
    try:
        df.to_excel(excel_path, index=False)
        return excel_path
    except Exception as e:
        st.error(f"保存 Excel 失败: {e}")
        return None


# --- 页面配置 ---
st.set_page_config(
    page_title="PCB 智能缺陷检测系统",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- 自定义 CSS 样式 ---
st.markdown("""
<style>
.main { background-color: #f5f5f5; }
.stButton>button { width: 100%; border-radius: 5px; height: 3em; background-color: #0066cc; color: white; }
.stButton>button:hover { background-color: #0052a3; }
</style>
""", unsafe_allow_html=True)

# --- 侧边栏设置 ---
with st.sidebar:
    st.header("⚙️ 系统设置")

    # 1. 模型选择 (保留 txt 原有逻辑)
    weights_dir = "weights"
    model = None
    if os.path.exists(weights_dir):
        available_models = [f for f in os.listdir(weights_dir) if f.endswith('.pt')]
        if available_models:
            selected_model_name = st.selectbox("选择模型文件", available_models)
            model_path = os.path.join(weights_dir, selected_model_name)


            @st.cache_resource
            def load_model(path):
                return YOLO(path)


            try:
                model = load_model(model_path)
                st.success("✅ 模型加载成功")
            except Exception as e:
                st.error(f"❌ 模型加载失败: {e}")
        else:
            st.warning(f"⚠️ {weights_dir} 文件夹下未找到 .pt 模型文件")
    else:
        st.error(f"❌ 未找到 {weights_dir} 文件夹，请确保模型存放在该目录下")

    st.divider()

    # 2. 检测参数
    st.subheader("检测参数")
    confidence_threshold = st.slider("置信度阈值", 0.0, 1.0, 0.5, 0.05)
    iou_threshold = st.slider("IoU 阈值", 0.0, 1.0, 0.45, 0.05)

    st.divider()

    # 3. 类别映射
    if model:
        class_names = model.names
        st.write(f"**检测类别:** {len(class_names)} 类")
        cols = st.columns(2)
        for idx, (k, v) in enumerate(class_names.items()):
            with cols[idx % 2]:
                st.text(f"- {v}")
    else:
        class_names = {}


# --- 辅助函数：缓存转换结果 (保留 txt 原有功能) ---
@st.cache_data
def convert_df_to_excel(df):
    import io
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        df.to_excel(writer, index=False, sheet_name='检测记录')
    processed_data = output.getvalue()
    return processed_data


# --- 主界面 ---
st.title("🔍 PCB 缺陷智能检测系统")
st.markdown("基于 YOLO 深度学习算法的工业级缺陷检测解决方案")

tab1, tab2, tab3 = st.tabs(["📷 图片检测", "🎥 视频/摄像头检测", "📊 历史记录与导出"])

# ==========================================
# 选项卡 1: 图片检测 (合并逻辑)
# ==========================================
with tab1:
    if model is None:
        st.error("请先在侧边栏选择并加载模型。")
    else:
        st.info("💡 提示：你可以按住 Ctrl (Windows) 或 Command (Mac) 键选择多个文件，或者直接拖入整个文件夹。")
        uploaded_files = st.file_uploader(
            "上传 PCB 图片 (支持多选)", type=['jpg', 'jpeg', 'png', 'bmp'], accept_multiple_files=True
        )

        if uploaded_files:
            if not isinstance(uploaded_files, list):
                uploaded_files = [uploaded_files]

            progress_bar = st.progress(0)
            status_text = st.empty()
            total_images = len(uploaded_files)
            inference_times = []

            # 确保 result 文件夹存在
            result_dir = "result"
            os.makedirs(result_dir, exist_ok=True)

            for i, uploaded_file in enumerate(uploaded_files):
                status_text.text(f"正在检测 ({i + 1}/{total_images}): {uploaded_file.name}")

                # 读取图片
                image = Image.open(uploaded_file)
                image_np = np.array(image)
                image_cv = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)

                # 执行检测
                start_time = time.time()
                results = model(image_cv, conf=confidence_threshold, iou=iou_threshold, verbose=False)
                inference_time = time.time() - start_time
                inference_times.append(inference_time)
                current_fps = 1.0 / inference_time if inference_time > 0 else 0.0

                # 解析结果 (使用 txt 中的 parse_results)
                df_results = parse_results(results, class_names, inference_time)

                # 绘图
                boxes = results[0].boxes.xyxy.cpu().numpy()
                classes = results[0].boxes.cls.cpu().numpy()
                confs = results[0].boxes.conf.cpu().numpy()
                annotated_img = draw_boxes(image_cv, boxes, classes, confs, class_names)
                annotated_img_rgb = cv2.cvtColor(annotated_img, cv2.COLOR_BGR2RGB)

                # === 功能 1: 保存图片 (保留 txt 原有逻辑) ===
                file_count = len(os.listdir(result_dir)) + 1
                timestamp = datetime.now().strftime("%m%d_%H%M%S")
                save_name = f"{file_count}_{timestamp}_{uploaded_file.name}"
                save_path = os.path.join(result_dir, save_name)
                cv2.imwrite(save_path, annotated_img)

                # === 功能 2: 自动保存 Excel (来自第二段代码的新增功能) ===
                # 将 df_results 转换为字典列表以便保存
                detection_list = df_results.to_dict('records')
                if detection_list:
                    # 添加源文件名信息
                    for item in detection_list:
                        item['源文件名'] = uploaded_file.name

                    # 调用新增的函数自动保存
                    excel_path = save_to_excel(detection_list, uploaded_file.name)
                    if excel_path:
                        st.success(f"✅ 报告已自动生成: `{os.path.basename(excel_path)}`")
                else:
                    st.info("ℹ️ 未检测到缺陷，不生成报告")

                # --- 显示单张图片结果 ---
                st.markdown(f"### 🖼️ 图片 {i + 1}: {uploaded_file.name}")
                col1, col2 = st.columns(2)
                with col1:
                    st.subheader("原始图像")
                    st.image(image, use_column_width=True)
                with col2:
                    st.subheader("检测结果")
                    st.image(annotated_img_rgb, use_column_width=True)

                defect_count = len(df_results)
                if defect_count > 0:
                    st.warning(f"⚠️ 发现 **{defect_count}** 处缺陷")
                else:
                    st.success("✅ 未发现缺陷")

                st.info(f"⏱️ 耗时: **{inference_time:.3f}s** | 🚀 速度: **{current_fps:.2f} FPS**")
                st.dataframe(df_results, use_container_width=True)
                st.divider()

                progress_bar.progress((i + 1) / total_images)

                # === 功能 3: 保存记录到数据库 (保留 txt 原有逻辑) ===
                save_record(
                    filename=uploaded_file.name,
                    defect_count=defect_count,
                    duration=inference_time,
                    fps=current_fps,
                    details=str(df_results.to_dict('records'))
                )

            status_text.text("✅ 所有图片检测完成！")
            avg_time = np.mean(inference_times)
            avg_fps = 1.0 / avg_time if avg_time > 0 else 0.0
            st.success(f"🎉 检测结束！共处理 **{total_images}** 张图片。")
            stat_col1, stat_col2 = st.columns(2)
            with stat_col1:
                st.metric(label="平均推理耗时", value=f"{avg_time:.3f} 秒")
            with stat_col2:
                st.metric(label="系统平均速度", value=f"{avg_fps:.2f} FPS")

# ==========================================
# 选项卡 2: 视频/摄像头检测 (保留 txt 原有逻辑)
# ==========================================
with tab2:
    if model is None:
        st.error("请先在侧边栏选择并加载模型。")
    else:
        st.info("💡 注意：如果是远程服务器部署，请选择上传图片方式。本地运行可直接使用摄像头。")
        detection_mode = st.radio("选择检测模式", ["上传图片进行视频检测", "使用摄像头 (实时拍照检测)"])

        if detection_mode == "使用摄像头 (实时拍照检测)":
            st.write("### 📸 实时摄像头检测")
            camera_image = st.camera_input("点击按钮拍照进行检测")
            if camera_image:
                image = Image.open(camera_image)
                image_np = np.array(image)
                image_cv = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
                with st.spinner("正在分析..."):
                    start_time = time.time()
                    results = model(image_cv, conf=confidence_threshold, iou=iou_threshold, verbose=False)
                    inference_time = time.time() - start_time
                    current_fps = 1.0 / inference_time
                    annotated_frame = results[0].plot()
                    annotated_frame_rgb = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
                    st.image(annotated_frame_rgb, channels="RGB", use_column_width=True)
                    st.success(f"✅ 检测完成 | 耗时: {inference_time:.3f}s | 速度: {current_fps:.2f} FPS")
        else:
            st.write("### 🎥 视频文件检测")
            uploaded_video = st.file_uploader("上传视频文件", type=['mp4', 'avi', 'mov'])
            if uploaded_video and st.button("开始视频分析"):
                tfile = open("temp_video.mp4", "wb")
                tfile.write(uploaded_video.read())
                tfile.close()
                cap = cv2.VideoCapture("temp_video.mp4")
                st_frame = st.empty()
                status_text = st.empty()
                frame_count = 0
                total_time = 0.0
                while cap.isOpened():
                    ret, frame = cap.read()
                    if not ret:
                        break
                    frame_count += 1
                    start_time = time.time()
                    results = model(frame, conf=confidence_threshold, iou=iou_threshold, verbose=False)
                    inference_time = time.time() - start_time
                    total_time += inference_time
                    annotated_frame = results[0].plot()
                    annotated_frame_rgb = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
                    st_frame.image(annotated_frame_rgb, channels="RGB", use_column_width=True)
                cap.release()
                status_text.text("⏹️ 视频分析结束")
                if frame_count > 0:
                    avg_video_fps = frame_count / total_time if total_time > 0 else 0
                    st.info(f"视频共 {frame_count} 帧，平均处理速度: {avg_video_fps:.2f} FPS")
                os.remove("temp_video.mp4")

# ==========================================
# 选项卡 3: 历史记录与导出 (保留 txt 原有逻辑)
# ==========================================
with tab3:
    st.subheader("📊 历史检测记录")
    df_history = get_all_records()
    if not df_history.empty:
        st.dataframe(
            df_history.drop(columns=['id', 'details']),
            use_container_width=True,
            hide_index=True
        )

        st.markdown("### 📥 导出数据")
        excel_data = convert_df_to_excel(df_history)

        result_dir = "result"
        os.makedirs(result_dir, exist_ok=True)
        file_count = len(os.listdir(result_dir)) + 1
        timestamp = datetime.now().strftime("%m%d_%H%M%S")
        file_name = f"{file_count}_{timestamp}.xlsx"

        st.download_button(
            label="点击下载 Excel 报告",
            data=excel_data,
            file_name=file_name,
            mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
        )
        st.info(f"文件将保存为: **{file_name}** (保存在服务器/本地的 result 文件夹中)")
    else:
        st.info("暂无历史记录，请先进行检测。")