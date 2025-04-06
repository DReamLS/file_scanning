# -*- coding: utf-8 -*-
import fitz as PyMuPDF  # PyMuPDF
from PIL import Image
from paddleocr import PaddleOCR
import pdfplumber
from models import details_abstract
from models import logic_search
from models import flexible_area_abstract
from models import miner

import os
import io
import cv2
import numpy as np
import shutil
import logging

# 配置日志记录
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

ocr = PaddleOCR(use_angle_cls=True, lang="ch", use_gpu=False)  # 启用 GPU 加速时设置 use_gpu=True


# 获取当前文件所在目录
current_dir = os.path.dirname(os.path.abspath(__file__))

# 构建 stopwords.txt 和 must_words.txt 的绝对路径
stopwords_path = os.path.join(current_dir, '..', 'static', 'stopwords.txt')
must_words_path = os.path.join(current_dir, '..', 'static', 'must_words.txt')


#通用函数

def build_logic_tree(input_dict):
    """
    构建逻辑树
    """
    logic_tree = {}

    def insert_into_tree(tree, path, final_value):
        if not path:
            return

        if len(path) == 1:
            current_key = path[0]
            if current_key in tree:
                if isinstance(tree[current_key], list):
                    tree[current_key].append(final_value)
                else:
                    tree[current_key] = [tree[current_key], final_value]
            else:
                tree[current_key] = [final_value]
        else:
            current_key = path[-1]
            if current_key not in tree or not isinstance(tree[current_key], dict):
                tree[current_key] = {}
            insert_into_tree(tree[current_key], path[:-1], final_value)

    for key, value_lists in input_dict.items():
        if not isinstance(value_lists, (list, tuple)):
            value_lists = [value_lists]
        for value_list in value_lists:
            if not isinstance(value_list, list):
                value_list = [value_list]
            insert_into_tree(logic_tree, value_list, key)

    return logic_tree


def linking(mode, scanned, approximate):
    """
    模板与扫描文件匹配函数
    """
    result = {}
    output_dir = "debug_images"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    dx0, dy0, dx1, dy1 = approximate

    try:
        for num in range(min(len(mode), len(scanned))):
            img = scanned[num]

            for (x0, y0, x1, y1), tree in mode[num].items():
                x0, y0, x1, y1 = map(int, (x0 + dx0, y0 + dy0, x1 + dx1, y1 + dy1))
                cropped_img = img.crop((x0, y0, x1, y1))
                temp_path = os.path.join(output_dir, f"page_{num}_area_{x0}_{y0}_{x1}_{y1}.png")
                cropped_img.save(temp_path)

                raw_text = ocr.ocr(temp_path, cls=True)
                text = raw_text[0][0][1][0] if raw_text[0] else "none"
                text = text + f" 行：{y0}，列：{x0}"
                result[text] = tree

                if os.path.exists(temp_path):
                    os.remove(temp_path)

    except Exception as e:
        logging.error(f"模板与扫描文件匹配失败：{e}")

    finally:
        if os.path.exists(output_dir):
            shutil.rmtree(output_dir)

    return result


def detect_border(image):
    """
    边界检测函数
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 50, 150)

    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        raise ValueError("未检测到轮廓，请检查参考图片是否正确。")

    largest_contour = max(contours, key=cv2.contourArea)
    epsilon = 0.1 * cv2.arcLength(largest_contour, True)
    approx = cv2.approxPolyDP(largest_contour, epsilon, True)

    if len(approx) != 4:
        raise ValueError("检测到的轮廓不是四边形，请检查参考图片是否正确。")

    pts_border = np.squeeze(approx).astype(np.float32)

    def order_points(pts):
        rect = np.zeros((4, 2), dtype=np.float32)
        s = pts.sum(axis=1)
        rect[0] = pts[np.argmin(s)]  # 左上
        rect[2] = pts[np.argmax(s)]  # 右下
        diff = np.diff(pts, axis=1)
        rect[1] = pts[np.argmin(diff)]  # 右上
        rect[3] = pts[np.argmax(diff)]  # 左下
        return rect

    pts_border = order_points(pts_border)
    canvas_height, canvas_width = image.shape[:2]
    return pts_border, (canvas_width, canvas_height)


#电子模板处理函数

def mode_process1(pdf_path):
    """
    模板处理函数
    """
    result = {}
    details_num = {}
    try:
        with pdfplumber.open(pdf_path) as pdf:
            doc = PyMuPDF.open(pdf_path)
            for page_num in range(len(doc)):
                page_plum = pdf.pages[page_num]
                details, fixed_area, unfixed_area, extension = details_abstract.abstract(
                    page_plum, stopwords_path=stopwords_path, must_words_path=must_words_path
                )
                details_num[page_num] = details
                logic = logic_search.search(fixed_area)
                flexible_area = flexible_area_abstract.flexible_abstract(logic, fixed_area, unfixed_area)
                result[page_num] = flexible_area
            doc.close()
    except Exception as e:
        logging.error(f"模板处理失败：{e}")
    return result, extension


def scanned_process1(mode_path, pdf_path, mode_extension):
    result = {}

    # 打开PDF文档
    doc = PyMuPDF.open(pdf_path)
    mode = PyMuPDF.open(mode_path)

    for page_num in range(len(mode)):
        logging.info(f"正在处理扫描文件第 {page_num + 1} 页...")

        # 加载模板页面并转换为图像
        m_page = mode.load_page(page_num)
        m_pix = m_page.get_pixmap()

        # 使用 BytesIO 在内存中处理图像
        m_img = Image.frombytes("RGB", [m_pix.width, m_pix.height], m_pix.samples)
        img_byte_arr = io.BytesIO()
        m_img.save(img_byte_arr, format='PNG')
        img_byte_arr.seek(0)
        reference_img = cv2.imdecode(np.frombuffer(img_byte_arr.getvalue(), dtype=np.uint8), cv2.IMREAD_COLOR)

        # 检测参考图片的边界线和画布大小
        try:
            pts_ref_border, (ref_canvas_width, ref_canvas_height) = detect_border(reference_img)
        except ValueError as e:
            logging.warning(f"页 {page_num + 1} 边界检测失败：{e}")
            continue

        # 加载扫描页面并转换为图像
        page = doc.load_page(page_num)
        pix = page.get_pixmap()
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)

        # 将PIL图像转换为OpenCV格式
        img_cv = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

        # 提取感兴趣区域，并确保坐标为整数
        x0, y0, x1, y1 = map(int, mode_extension)


        if x0 < 0 or y0 < 0 or x1 > ref_canvas_width or y1 > ref_canvas_height:
            logging.warning(f"页 {page_num + 1} 的模式扩展超出边界，跳过...")
            continue

        cropped_img = img_cv[y0:y1, x0:x1]

        # 转换为灰度图并进行边缘检测
        gray = cv2.cvtColor(cropped_img, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blurred, 50, 150)

        # 查找轮廓
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if len(contours) == 0:
            logging.warning(f"页 {page_num + 1} 未检测到轮廓，跳过...")
            continue

        largest_contour = max(contours, key=cv2.contourArea)

        # 近似多边形拟合轮廓，得到四个角点
        epsilon = 0.1 * cv2.arcLength(largest_contour, True)
        approx = cv2.approxPolyDP(largest_contour, epsilon, True)

        if len(approx) != 4:
            logging.warning(f"页 {page_num + 1} 检测到的轮廓不是四边形，跳过...")
            continue

        pts_src = np.squeeze(approx).astype(np.float32)

        # 确保 pts_src 的点按顺时针顺序排列（左上、右上、右下、左下）
        def order_points(pts):
            rect = np.zeros((4, 2), dtype=np.float32)
            s = pts.sum(axis=1)
            rect[0] = pts[np.argmin(s)]  # 左上
            rect[2] = pts[np.argmax(s)]  # 右下
            diff = np.diff(pts, axis=1)
            rect[1] = pts[np.argmin(diff)]  # 右上
            rect[3] = pts[np.argmax(diff)]  # 左下
            return rect

        pts_src = order_points(pts_src)

        # 定义目标坐标为参考图片的边界线坐标
        pts_dst = pts_ref_border

        # 计算透视变换矩阵
        matrix = cv2.getPerspectiveTransform(pts_src, pts_dst)

        # 创建一个空白画布，大小为参考图片的画布大小
        canvas = np.zeros((ref_canvas_height, ref_canvas_width, 3), dtype=np.uint8)

        # 进行透视变换
        warped_img = cv2.warpPerspective(cropped_img, matrix, (ref_canvas_width, ref_canvas_height))

        # 将透视变换后的图像叠加到画布上
        mask = warped_img != 0
        canvas[mask] = warped_img[mask]

        # 将OpenCV图像转换回PIL格式
        processed_img = Image.fromarray(cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB))

        # 存储结果
        result[page_num] = processed_img

    doc.close()
    mode.close()

    return result


def process1(mode_path, scanned_path):
    """
    主处理函数
    """
    try:
        mode_result, mode_extension = mode_process1(mode_path)
        logging.info(f"模板处理结果：{mode_result}")

        scanned_result = scanned_process1(mode_path, scanned_path, mode_extension)

        approximate = (0, 0, 0, 5)
        result = linking(mode_result, scanned_result, approximate)

        logic_tree = build_logic_tree(result)
        return logic_tree

    except Exception as e:
        logging.error(f"主处理流程失败：{e}")


#纸质模板处理函数

def mode_process2(pdf_path):
    """
    模板处理函数
    """
    result = {}
    try:
        doc = PyMuPDF.open(pdf_path)
        for page_num in range(len(doc)):
            fixed_area, unfixed_area = miner.process_pdf_page1(pdf_path, page_num,hands = 0.1)
            logic = logic_search.search(fixed_area)
            flexible_area = flexible_area_abstract.flexible_abstract(logic, fixed_area, unfixed_area)
            result[page_num] = flexible_area
        doc.close()
    except Exception as e:
        logging.error(f"模板处理失败：{e}")

    return result


def transform_image2(img, canvas_size=(1000, 1000), margin=80):
    img_cv = np.array(img)

    # 假设detect_border函数返回图像的边界点和建议的画布宽度和高度
    try:
        pts_border, (original_width, original_height) = detect_border(img_cv)  # 请确保这个函数存在并正确工作
    except ValueError as e:
        print(e)
        return None

    # 计算比例并调整目标canvas尺寸
    ratio = min((canvas_size[0] - 2 * margin) / original_width, (canvas_size[1] - 2 * margin) / original_height)
    target_width = int(original_width * ratio)
    target_height = int(original_height * ratio)

    # 定义目标点，使图像居中
    dst_pts = np.array([[margin, margin], [target_width + margin, margin],
                        [target_width + margin, target_height + margin], [margin, target_height + margin]],
                       dtype="float32")

    # 计算变换矩阵
    M = cv2.getPerspectiveTransform(pts_border.astype("float32"), dst_pts)
    transformed_img = cv2.warpPerspective(img_cv, M, (canvas_size[0], canvas_size[1]), borderValue=(255, 255, 255))

    # 转换回PIL图像并返回
    return Image.fromarray(cv2.cvtColor(transformed_img, cv2.COLOR_BGR2RGB))


def scanned_process2(pdf_path):
    result = {}

    # 打开PDF文档
    doc = PyMuPDF.open(pdf_path)

    for page_num in range(len(doc)):
        logging.info(f"正在处理扫描文件第 {page_num + 1} 页...")

        # 加载扫描页面并转换为图像
        page = doc.load_page(page_num)
        pix = page.get_pixmap()

        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)

        processed_img =  transform_image2(img)

        # 存储结果
        result[page_num] = processed_img

    doc.close()

    return result


def process2(mode_path, scanned_path):
    """
    主处理函数
    """
    try:
        mode_result = mode_process2(mode_path)
        print(mode_result)
        logging.info(f"模板处理结果：{mode_result}")

        scanned_result = scanned_process2(scanned_path)

        approximate = (0, 0, 0, 5)
        result = linking(mode_result, scanned_result, approximate)

        logic_tree = build_logic_tree(result)
        return logic_tree

    except Exception as e:
        logging.error(f"主处理流程失败：{e}")


#无模板处理函数

def mode_process3(pdf_path):
    """
    模板处理函数
    """
    result = {}
    try:
        doc = PyMuPDF.open(pdf_path)
        for page_num in range(len(doc)):
            fixed_area, unfixed_area = miner.process_pdf_page2(pdf_path, page_num,hands = 0.1)
            logic = logic_search.search(fixed_area)
            flexible_area = flexible_area_abstract.flexible_abstract(logic, fixed_area, unfixed_area)
            result[page_num] = flexible_area
        doc.close()
    except Exception as e:
        logging.error(f"模板处理失败：{e}")

    return result


def transform_image3(img, canvas_size=(1000, 1000), margin=80):
    img_cv = np.array(img)

    # 假设detect_border函数返回图像的边界点和建议的画布宽度和高度
    try:
        pts_border, (original_width, original_height) = detect_border(img_cv)  # 请确保这个函数存在并正确工作
    except ValueError as e:
        print(e)
        return None

    # 计算比例并调整目标canvas尺寸
    ratio = min((canvas_size[0] - 2 * margin) / original_width, (canvas_size[1] - 2 * margin) / original_height)
    target_width = int(original_width * ratio)
    target_height = int(original_height * ratio)

    # 定义目标点，使图像居中
    dst_pts = np.array([[margin, margin], [target_width + margin, margin],
                        [target_width + margin, target_height + margin], [margin, target_height + margin]],
                       dtype="float32")

    # 计算变换矩阵
    M = cv2.getPerspectiveTransform(pts_border.astype("float32"), dst_pts)
    transformed_img = cv2.warpPerspective(img_cv, M, (canvas_size[0], canvas_size[1]), borderValue=(255, 255, 255))

    # 转换回PIL图像并返回
    return Image.fromarray(cv2.cvtColor(transformed_img, cv2.COLOR_BGR2RGB))

def scanned_process3(pdf_path):
    result = {}

    # 打开PDF文档
    doc = PyMuPDF.open(pdf_path)

    for page_num in range(len(doc)):
        logging.info(f"正在处理扫描文件第 {page_num + 1} 页...")

        # 加载扫描页面并转换为图像
        page = doc.load_page(page_num)
        pix = page.get_pixmap()

        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)

        processed_img =  transform_image3(img)
        processed_img.save("xxxx.png")

        # 存储结果
        result[page_num] = processed_img

    doc.close()

    return result


def process3(scanned_path):
    """
    主处理函数
    """
    try:
        mode_result = mode_process3(scanned_path)
        print(mode_result)
        logging.info(f"模板处理结果：{mode_result}")

        scanned_result = scanned_process3(scanned_path)

        approximate = (0, 0, 0, 5)
        result = linking(mode_result, scanned_result, approximate)

        logic_tree = build_logic_tree(result)
        return logic_tree

    except Exception as e:
        logging.error(f"主处理流程失败：{e}")




if __name__ == "__main__":
    mode_path = r"../static/scanned-test_Page1.pdf"
    scanned_path = r"../static/scanned-test_Page1.pdf"
    process3(scanned_path)