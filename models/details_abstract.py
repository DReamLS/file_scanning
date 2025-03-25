# -*- coding: utf-8 -*-
import re


def get_filter_reg(stopwords_path):
    """
    从文件读取停用词并生成正则表达式过滤规则。
    :param stopwords_path: 包含停用词的文本文件路径
    :return: 正则表达式字符串
    """
    with open(stopwords_path, 'r', encoding='utf-8') as f:
        stopwords = [line.strip() for line in f if line.strip()]
    filter_reg = r'\b(?:' + r'|'.join(re.escape(word) for word in stopwords) + r')\b'
    return filter_reg


def get_must_reg(must_words_path):
    """
    从文件读取必须词并生成正则表达式匹配规则。
    :param must_words_path: 包含必须词的文本文件路径
    :return: 正则表达式字符串
    """
    with open(must_words_path, 'r', encoding='utf-8') as f:
        must_words = [line.strip() for line in f if line.strip()]
    must_reg = r'\b(?:' + r'|'.join(re.escape(word) for word in must_words) + r')\b'
    return must_reg


def fixed_abstract(page, filter_reg, must_reg):
    """
    处理给定页面中的表格单元格，过滤无效信息，返回有效文本及其对应单元格坐标的字典。
    :param page: pdfplumber.Page 对象
    :param filter_reg: 用于过滤无效信息的正则表达式
    :param must_reg: 用于匹配必须词的正则表达式
    :return: 包含有效文本为detail键，单元格坐标列表为value的字典
    """
    fixed_area = {}

    table_settings = {
        "vertical_strategy": "lines",
        "horizontal_strategy": "lines",
    }

    tables = page.find_tables(table_settings)
    x_min = 1000
    y_min = 1000
    x_max = 0
    y_max = 0
    for table_idx, table in enumerate(tables, 1):
        cells = table.cells
        for cell in cells:
            x0, y0, x1, y1 = cell
            x_min = min(x_min,x0)
            y_min = min(y_min,y0)
            x_max = max(x_max,x1)
            y_max = max(y_max,y1)
            if cell is not None:


                cell_width = x1 - x0
                cell_height = y1 - y0
                aspect_ratio = cell_width / cell_height if cell_height != 0 else float('inf')

                if (cell_width < 15 or cell_height < 8 or
                        aspect_ratio > 15 or aspect_ratio < 0.05):
                    continue

                cell_text = page.crop(bbox=(x0, y0, x1, y1)).extract_text()
                if cell_text is None:
                    continue

                cleaned_text = cell_text.replace('\n', '').replace(' ', '')

                # 如果单元格文本包含任何停用词，则跳过该单元格
                if not cleaned_text or cleaned_text.isspace() or re.search(filter_reg, cleaned_text, re.IGNORECASE):
                    continue

                # 构造新的键值对
                detail_key = f"{cleaned_text}"
                fixed_area[detail_key] = (x0, y0, x1, y1)

    return fixed_area , (x_min,y_min,x_max,y_max)


def unfixed_abstract(page, fixed_area):
    """
    根据已有的固定区域，返回页面中其他单元格组成的列表。
    :param page: pdfplumber.Page 对象
    :param fixed_area: 包含有效文本及其对应单元格坐标的字典
    :return: 不在固定区域中的单元格组成的列表
    """
    all_cells = []
    table_settings = {
        "vertical_strategy": "lines",
        "horizontal_strategy": "lines",
    }
    tables = page.find_tables(table_settings)

    for table in tables:
        for cell in table.cells:
            if cell is not None:
                all_cells.append(cell)

    fixed_cell_set = set(fixed_area.values())

    unfixed_cells = [cell for cell in all_cells if tuple(cell) not in fixed_cell_set]

    return unfixed_cells


def find_and_fix_remaining_must_words(page, unfixed_cells, must_reg, fixed_area):
    """
    查找未固定的单元格中包含必须词的单元格，并将其加入到fixed_area中。
    :param page: pdfplumber.Page 对象
    :param unfixed_cells: 未固定的单元格列表
    :param must_reg: 用于匹配必须词的正则表达式
    :param fixed_area: 已固定的区域字典
    :return: 更新后的固定区域字典
    """
    for cell in unfixed_cells:
        x0, y0, x1, y1 = cell
        cell_text = page.crop(bbox=(x0, y0, x1, y1)).extract_text()
        if cell_text is None:
            continue

        cleaned_text = cell_text.replace('\n', '').replace(' ', '')

        # 如果单元格文本包含任何必须词，则直接加入固定区域
        if re.search(must_reg, cleaned_text, re.IGNORECASE):
            detail_key = f"{cleaned_text}"
            fixed_area[detail_key] = (x0, y0, x1, y1)

    return fixed_area


def abstract(page, stopwords_path, must_words_path):
    """
    抽取给定页面中的有效文本信息，并返回details, fixed_area, unfixed_area。
    :param page: pdfplumber.Page 对象
    :param stopwords_path: 包含停用词的文本文件路径
    :param must_words_path: 包含必须词的文本文件路径
    :return: 包含有效文本及其原始文本的字典, 固定区域字典, 非固定区域列表
    """
    filter_reg = get_filter_reg(stopwords_path)
    must_reg = get_must_reg(must_words_path)

    # 执行原始逻辑
    fixed_area, (x_min,y_min,x_max,y_max)= fixed_abstract(page, filter_reg, must_reg)
    unfixed_area = unfixed_abstract(page, fixed_area)

    # 在原始逻辑结束后，查找并固定剩余的必须词所在的单元格
    fixed_area = find_and_fix_remaining_must_words(page, unfixed_area, must_reg, fixed_area)

    # 更新非固定区域
    unfixed_area = unfixed_abstract(page, fixed_area)

    # 构造details字典
    details = {text: text for text, location in fixed_area.items()}

    return details, fixed_area, unfixed_area , (x_min,y_min,x_max,y_max)