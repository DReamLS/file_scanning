from io import BytesIO
import fitz  # PyMuPDF
import cv2
import numpy as np
from PIL import Image
import re
from paddleocr import PaddleOCR


# 获取过滤正则表达式
def get_filter_reg(stopwords_path):
    with open(stopwords_path, 'r', encoding='utf-8') as f:
        stopwords = [line.strip() for line in f if line.strip()]
    filter_reg = r'\b(?:' + r'|'.join(re.escape(word) for word in stopwords) + r')\b'
    return filter_reg


# 获取必须包含的正则表达式
def get_must_reg(must_words_path):
    with open(must_words_path, 'r', encoding='utf-8') as f:
        must_words = [line.strip() for line in f if line.strip()]
    must_reg = r'\b(?:' + r'|'.join(re.escape(word) for word in must_words) + r')\b'
    return must_reg


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


def transform_image(img):
    img_cv = np.array(img)

    # 使用detect_border函数获取边界点
    try:
        pts_border, (canvas_width, canvas_height) = detect_border(img_cv)
    except ValueError as e:
        print(e)
        return None

    # 目标点定义一个与原图比例相同的矩形区域，
    dst_pts = np.array([[80, 80], [canvas_width+80, 80], [canvas_width+80, canvas_height+80], [80, canvas_height+80]], dtype="float32")

    # 计算变换矩阵
    M = cv2.getPerspectiveTransform(pts_border, dst_pts)
    transformed_img = cv2.warpPerspective(img_cv, M, (canvas_width+160, canvas_height+160), borderValue=(255, 255, 255))

    # 转换回PIL图像并返回
    return Image.fromarray(cv2.cvtColor(transformed_img, cv2.COLOR_BGR2RGB))


def preprocess_image(image):
    # 灰度转换
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # 使用更大的高斯模糊核，减少噪声同时尽量保留边缘信息
    blurred = cv2.GaussianBlur(gray, (7 , 7), 0)

    # 自适应阈值处理，对于光照变化有更好的鲁棒性
    binary = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY_INV, 11, 2)

    # 形态学操作，用于去噪和连接字符
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

    return binary


#竖直axis是1
def merge_similar_lines_and_choose_longest(lines, axis, threshold=10):
    if len(lines) == 0:
        return []

    # 首先根据line[axis]排序，以便后续分组
    sorted_lines = sorted(lines, key=lambda line: line[1-axis])

    def process_group(group):
        longest_line = max(group, key=lambda l: abs(l[2+axis] - l[axis]))
        merged = []
        remaining = []
        temp = {}
        temp[axis] = longest_line[axis]
        temp[2+axis] = longest_line[2+axis]
        for line in group:
            if line != longest_line:
                if (longest_line[axis]-10 <= line[axis] <= longest_line[axis+2]+10) or \
                   (longest_line[axis]-10 <= line[axis+2] <= longest_line[axis+2]+10):
                    merged.append(line)
                else:
                    remaining.append(line)

        # 如果merged不为空，则合并这些线条（这里简单地选择最长线代表合并结果）
        if merged:
            for mline in merged:
                if mline[axis]<temp[axis]: temp[axis] = mline[axis]
                if mline[2+axis]>temp[2+axis]: temp[2+axis] = mline[2+axis]
            if axis == 0:
                result = [(temp[axis],longest_line[1-axis],temp[axis+2],longest_line[1-axis])]
            else:
                result = [(longest_line[1-axis],temp[axis],longest_line[1-axis],temp[axis+2])]


        else:
            result = []

        # 对剩余的线条递归处理
        if remaining:
            result.extend(process_group(remaining))

        return result

    # 分组并处理每个组
    groups = []
    current_group = []
    last_center = None

    for line in sorted_lines:
        center = line[1-axis]
        if last_center is not None and abs(center - last_center) > threshold:
            if current_group:
                groups.append(current_group)
                current_group = []
        current_group.append(line)
        last_center = center

    if current_group:  # 最后一组
        groups.append(current_group)

    merged_lines = []
    for group in groups:
        merged_lines.extend(process_group(group))

    # 处理合并后的直线，使其成为标准形式
    straight_lines = []
    for line in merged_lines:
        x0, y0, x1, y1 = line
        if axis == 0:  # 水平线
            avg_y = (y0 + y1) // 2
            straight_lines.append((x0, avg_y, x1, avg_y))
        elif axis == 1:  # 垂直线
            avg_x = (x0 + x1) // 2
            straight_lines.append((avg_x, y0, avg_x, y1))

    return straight_lines


def detect_and_merge_lines(binary, merge_threshold=8):  # 调整合并阈值为8
    horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (35, 1))
    detected_horizontal_lines = cv2.morphologyEx(binary, cv2.MORPH_OPEN, horizontal_kernel, iterations=2)

    vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 20))  # 减小垂直内核高度到15
    detected_vertical_lines = cv2.morphologyEx(binary, cv2.MORPH_OPEN, vertical_kernel, iterations=2)

    horizontal_lines = cv2.HoughLinesP(detected_horizontal_lines,
                                       rho=1,
                                       theta=np.pi / 180,
                                       threshold=15,  # 降低霍夫变换阈值到15
                                       minLineLength=10,  # 减少最小线段长度到10
                                       maxLineGap=100)  # 增加最大线段间隙到20

    vertical_lines = cv2.HoughLinesP(detected_vertical_lines,
                                     rho=1,
                                     theta=np.pi / 180,
                                     threshold=15,  # 降低霍夫变换阈值到15
                                     minLineLength=10,  # 减少最小线段长度到10
                                     maxLineGap=100)  # 增加最大线段间隙到20

    # 交换horizontal_lines中的y0和y1
    horizontal_lines = [(x0, y1, x1, y0) if y0 > y1 else (x0, y0, x1, y1) for [[x0, y0, x1, y1]] in horizontal_lines]
    # 交换vertical_lines中的y0和y1
    vertical_lines = [(x0, y1, x1, y0) if y0 > y1 else (x0, y0, x1, y1) for [[x0, y0, x1, y1]] in vertical_lines]

    merged_horizontal_lines = merge_similar_lines_and_choose_longest(horizontal_lines, axis=0,
                                                                     threshold=merge_threshold)
    merged_vertical_lines = merge_similar_lines_and_choose_longest(vertical_lines, axis=1, threshold=merge_threshold)

    return merged_horizontal_lines, merged_vertical_lines


def build_cells(horizontal_lines, vertical_lines, overlap_threshold=15):
    cells = []

    # 对竖线进行排序
    vertical_lines.sort(key=lambda x: x[0])  # 根据x坐标排序

    for i in range(len(vertical_lines)-1):  # 遍历竖线
        left_line = vertical_lines[i]

        for j in range(i + 1, len(vertical_lines)):  # 遍历右侧竖线
            right_line = vertical_lines[j]

            # 检测竖线是否满足y坐标范围有重叠部分
            y_min = max(left_line[1], right_line[1])
            y_max = min(left_line[3], right_line[3])

            if y_max-y_min <= overlap_threshold:
                continue  # 如果没有重叠部分，则跳过当前竖线

            middle_lines = []
            for k in range(i + 1, j):
                middle_line = vertical_lines[k]
                middle_y_min = max(y_min, middle_line[1])
                middle_y_max = min(y_max, middle_line[3])
                overlap_length = max(0, middle_y_max - middle_y_min)

                if overlap_length >= overlap_threshold:
                    middle_lines.append((middle_line, middle_y_min, middle_y_max))
            new_y_ranges = [(y_min, y_max)]
            if middle_lines:
                middle_lines.sort(key=lambda x: x[1])  # 按y坐标排序
                for middle_line, my_min, my_max in middle_lines:
                    for current_y_min, current_y_max in new_y_ranges:
                        if current_y_min < my_min and my_max < current_y_max:
                            new_y_ranges.append((current_y_min, my_min))
                            new_y_ranges.append((my_max, current_y_max))
                            new_y_ranges.remove((current_y_min, current_y_max))
                        elif current_y_min >= my_min and current_y_min < my_max < current_y_max:
                            new_y_ranges.append((my_max, current_y_max))
                            new_y_ranges.remove((current_y_min, current_y_max))
                        elif current_y_min < my_min < current_y_max and my_max >= current_y_max:
                            new_y_ranges.append((current_y_min, my_min))
                            new_y_ranges.remove((current_y_min, current_y_max))
                        elif current_y_min >= my_min and my_max >= current_y_max:
                            new_y_ranges.remove((current_y_min, current_y_max))
                        else:
                            continue
            for current_y_min, current_y_max in new_y_ranges:
                if current_y_max - current_y_min <=5:
                    new_y_ranges.remove((current_y_min, current_y_max))
            # 找到所有在这两条竖线的x坐标范围内的横线，并且y坐标在剩余的y范围内
            relevant_horizontal_lines = []
            for y_range in new_y_ranges:
                y_min, y_max = y_range
                relevant_horizontal_lines.extend([
                    line for line in horizontal_lines
                    if (line[0] <= left_line[0]+10 and right_line[0]-10 <= line[2] and y_min-10 <= line[1] <= y_max+10)
                ])

            if len(relevant_horizontal_lines) < 2:
                continue  # 如果没有足够的横线形成单元格，则跳过这对竖线

            # 对相关横线按y坐标排序
            relevant_horizontal_lines.sort(key=lambda x: x[1])

            # 确定单元格
            for h1 in range(len(relevant_horizontal_lines) - 1):
                top_line = relevant_horizontal_lines[h1]
                bottom_line = relevant_horizontal_lines[h1 + 1]

                # 确定单元格的边界
                left_x = left_line[0]
                right_x = right_line[0]
                top_y = top_line[1]
                bottom_y = bottom_line[1]

                cells.append((left_x, top_y, right_x, bottom_y))

    return cells


# 处理PDF页面
def process_pdf_page(pdf_path, page_num, hands=0.1):
    doc = fitz.open(pdf_path)
    page = doc.load_page(page_num)
    pix = page.get_pixmap()
    img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
    img = transform_image(img)  # 应用你的图像转换逻辑
    image = np.array(img)  # 将PIL图像转换为numpy数组用于OpenCV
    binary = preprocess_image(image)
    horizontal_lines, vertical_lines = detect_and_merge_lines(binary)
    cells = build_cells(horizontal_lines, vertical_lines)

    ocr = PaddleOCR(use_angle_cls=True, lang="ch", use_gpu=False)

    unfixed_area = []
    fixed_area = {}

    for (x0, y0, x1, y1) in cells:
        # 将浮点数坐标转换为整数
        x0, y0, x1, y1 = map(int, (x0, y0, x1, y1))
        # 根据坐标裁剪图像
        cropped_img = img.crop((x0, y0, x1, y1))

        # 使用BytesIO在内存中处理图像而不是保存到磁盘
        buffer = BytesIO()
        cropped_img.save(buffer, format="PNG")
        buffer.seek(0)  # 移动到字节流的开始
        content = buffer.getvalue()  # 获取字节流中的数据

        # 使用opencv从内存加载图像
        img_array = np.asarray(bytearray(content), dtype=np.uint8)
        cropped_cv_image = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

        # 识别图片中的文字
        raw_text = ocr.ocr(cropped_cv_image, cls=True)  # 直接传入OpenCV格式的图像

        if raw_text and isinstance(raw_text[0], list) and len(raw_text[0]) > 0:  # 确保有检测结果
            text_info = raw_text[0][0]  # 获取第一个检测结果
            if isinstance(text_info, list) and len(text_info) > 1:
                text, possibility = text_info[1]  # 解包出text和置信度分数
                if possibility >= hands:
                    fixed_area[text + f" 行：{y0}，列：{x0}"] = (x0, y0, x1, y1)
                else:
                    unfixed_area.append((x0, y0, x1, y1))
            else:
                unfixed_area.append((x0, y0, x1, y1))
        else:
            unfixed_area.append((x0, y0, x1, y1))

    return fixed_area, unfixed_area

# 示例调用
if __name__ == "__main__":
    pdf_path = "../static/mode_Page1.pdf"
    for page_num in range(1):  # 假设我们处理第一页
        fixed_area, unfixed_area = process_pdf_page(pdf_path, page_num)
        print(f"Page {page_num} - Fixed Area:", fixed_area)
        print(f"Page {page_num} - Unfixed Area:", unfixed_area)