def detect_grid_lines(fixed_area, threshold=5):
    """检测可能的网格线位置"""
    vertical_lines = set()  # x坐标
    horizontal_lines = set()  # y坐标

    for key, rect in fixed_area.items():
        x0, y0, x1, y1 = rect
        vertical_lines.add(x0)
        vertical_lines.add(x1)
        horizontal_lines.add(y0)
        horizontal_lines.add(y1)

    # 过滤相近的边界（合并为一条网格线）
    def merge_lines(lines, threshold):
        sorted_lines = sorted(lines)
        merged = []
        for line in sorted_lines:
            if not merged or line - merged[-1] > threshold:
                merged.append(line)
        return merged

    return merge_lines(vertical_lines, threshold), merge_lines(horizontal_lines, threshold)


def partition_cells(fixed_area, vertical_lines, horizontal_lines):
    """基于网格线划分单元格区域"""
    cells = []
    for i in range(len(horizontal_lines) - 1):
        for j in range(len(vertical_lines) - 1):
            y0 = horizontal_lines[i]
            y1 = horizontal_lines[i + 1]
            x0 = vertical_lines[j]
            x1 = vertical_lines[j + 1]

            # 查找落入此区域的矩形
            contained_rects = []
            for key, rect in fixed_area.items():
                rx0, ry0, rx1, ry1 = rect
                # 检查矩形是否大部分在单元格内
                if (rx0 >= x0 and rx1 <= x1 and ry0 >= y0 and ry1 <= y1):
                    contained_rects.append((key, rect))

            if contained_rects:
                cells.append({
                    "bounds": (x0, y0, x1, y1),
                    "rects": contained_rects,
                    "row": i,
                    "col": j
                })

    return cells


def identify_header_and_data(cells, format_info):
    """识别表头和数据区域"""
    headers = []
    data = []

    if not cells:
        return headers, data

    # 假设第一行为表头
    first_row = min(cell['row'] for cell in cells)

    for cell in cells:
        if cell['row'] == first_row:
            # 检查是否有格式特征表明是表头（如加粗、较大字体）
            has_header_format = False
            for key, _ in cell['rects']:
                fmt = format_info.get(key, {})
                if fmt.get('bold', False) or fmt.get('font_size', 0) > 12:
                    has_header_format = True
                    break
            if has_header_format:
                headers.append(cell)
            else:
                data.append(cell)  # 如果第一行没有表头特征，则视为数据
        else:
            data.append(cell)

    return headers, data


def detect_merged_cells(cells):
    """检测合并单元格"""
    merged_cells = []

    # 创建行-列到单元格的映射
    cell_map = {}
    for cell in cells:
        cell_map[(cell['row'], cell['col'])] = cell

    # 获取表格维度
    max_row = max(cell['row'] for cell in cells)
    max_col = max(cell['col'] for cell in cells)

    # 遍历所有可能的单元格位置
    for row in range(max_row + 1):
        for col in range(max_col + 1):
            # 如果当前位置没有单元格，可能是合并单元格的一部分
            if (row, col) not in cell_map:
                # 检查是否已被其他合并单元格包含
                is_merged = False
                for mc in merged_cells:
                    if (mc['start_row'] <= row <= mc['end_row'] and
                            mc['start_col'] <= col <= mc['end_col']):
                        is_merged = True
                        break

                if not is_merged:
                    # 尝试找到可能的合并单元格
                    mc = find_possible_merge(row, col, cell_map, max_row, max_col)
                    if mc:
                        merged_cells.append(mc)

    return merged_cells


def find_possible_merge(row, col, cell_map, max_row, max_col):
    """查找包含指定位置的可能合并单元格"""
    # 尝试向右扩展
    end_col = col
    while end_col < max_col and (row, end_col + 1) not in cell_map:
        end_col += 1

    # 尝试向下扩展
    end_row = row
    while end_row < max_row:
        valid = True
        for c in range(col, end_col + 1):
            if (end_row + 1, c) in cell_map:
                valid = False
                break
        if valid:
            end_row += 1
        else:
            break

    if end_row > row or end_col > col:
        return {
            "start_row": row,
            "start_col": col,
            "end_row": end_row,
            "end_col": end_col
        }

    return None