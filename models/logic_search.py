def sort_fixed_area_into_lines(fixed_area):
    """将固定区域按y0坐标分组，并在组内按x0坐标排序"""
    all_rects = [(key, rect) for key, rect in fixed_area.items()]
    sorted_rects = sorted(all_rects, key=lambda item: (item[1][1], item[1][0]))
    lines_dict = {}
    for key, rect in sorted_rects:
        y0 = rect[1]
        if y0 not in lines_dict:
            lines_dict[y0] = []
        lines_dict[y0].append((key, rect))

    print(f"行分组完成：共{len(lines_dict)}行")
    for y0, rects in lines_dict.items():
        print(f"  y0={y0}: {len(rects)}个矩形")

    return lines_dict


def filter_fences(lines_dict, x_threshold=50, y_threshold=30):
    """过滤行内和行间的无效间隔（围栏）"""
    filtered_lines = {}
    for y0, rects in lines_dict.items():
        if len(rects) <= 1:
            filtered_lines[y0] = rects
            continue

        grouped_rects = [[rects[0]]]
        for i in range(1, len(rects)):
            prev_rect = grouped_rects[-1][-1][1]
            curr_rect = rects[i][1]
            horizontal_gap = curr_rect[0] - prev_rect[2]
            if horizontal_gap > x_threshold:
                grouped_rects.append([rects[i]])
            else:
                grouped_rects[-1].append(rects[i])

        filtered_lines[y0] = [rect for group in grouped_rects for rect in group]

    final_lines = {}
    for y0, rects in filtered_lines.items():
        if len(rects) >= 2:
            final_lines[y0] = rects

    print(f"围栏过滤：从{len(lines_dict)}行减少到{len(final_lines)}行")
    for y0, rects in final_lines.items():
        print(f"  保留行y0={y0}: {len(rects)}个矩形")

    return final_lines


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
    last_row = max(cell['row'] for cell in cells)

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

    # 进一步分析：如果表头太少，可能需要重新考虑
    if len(headers) < len(data) * 0.2:
        print("警告：检测到的表头过少，可能需要重新评估")
        # 可以实现更复杂的表头识别逻辑

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


# 策略权重配置
STRATEGY_WEIGHTS = {
    "position": 0.6,  # 位置关系权重
    "format": 0.3,  # 格式特征权重
    "distance": 0.2,  # 距离度量权重
    "table": 0.3  # 表格结构权重（新增）
}

# 最小置信度阈值
MIN_CONFIDENCE = 0.3


def calculate_position_confidence(rect_i, rect_j):
    """计算基于位置关系的置信度"""
    xi0, yi0, xi1, yi1 = rect_i
    xj0, yj0, xj1, yj1 = rect_j

    confidence = 0.0

    # 规则1：右侧且y1更低（同级或子级）
    if xj0 > xi1 and yj1 < yi1:
        confidence += 0.8
        print(f"  位置规则1匹配: 矩形{rect_i} -> {rect_j}, 置信度+0.8")

    # 规则2：坐标包含（父级关系）
    if xi0 < xj0 and xi1 > xj1 and yi0 < yj0 and yi1 > yj1:
        confidence += 0.9
        print(f"  位置规则2匹配: 矩形{rect_i} -> {rect_j}, 置信度+0.9")

    # 规则3：垂直重叠（同级关系）
    if xj0 > xi1 and ((yj0 >= yi0 and yj0 <= yi1) or (yj1 >= yi0 and yj1 <= yi1)):
        confidence += 0.7
        print(f"  位置规则3匹配: 矩形{rect_i} -> {rect_j}, 置信度+0.7")

    print(f"  位置总置信度: {confidence}")
    return confidence


def calculate_format_confidence(fmt_i, fmt_j):
    """计算基于格式特征的置信度"""
    confidence = 0.0

    # 检查是否有格式信息
    if not fmt_i or not fmt_j:
        print("  格式信息缺失，置信度为0")
        return confidence

    # 父级标题通常更大更粗
    if fmt_i.get('bold', False) and (fmt_i.get('font_size', 0) > fmt_j.get('font_size', 0)):
        confidence += 0.6
        print(f"  格式规则1匹配: 字体{fmt_i} -> {fmt_j}, 置信度+0.6")

    # 相同颜色可能属于同一组
    if fmt_i.get('color') == fmt_j.get('color'):
        confidence += 0.3
        print(f"  格式规则2匹配: 颜色{fmt_i} -> {fmt_j}, 置信度+0.3")

    print(f"  格式总置信度: {confidence}")
    return confidence


def calculate_distance_confidence(rect_i, rect_j, avg_distance):
    """计算基于距离的置信度（距离越近，置信度越高）"""
    xi0, yi0, xi1, yi1 = rect_i
    xj0, yj0, xj1, yj1 = rect_j

    # 计算水平距离
    h_distance = xj0 - xi1 if xj0 > xi1 else xi0 - xj1

    # 距离越近，置信度越高
    normalized_distance = min(1.0, h_distance / (avg_distance * 2))
    confidence = 1.0 - normalized_distance

    print(f"  距离: {h_distance}, 平均距离: {avg_distance}, 置信度: {confidence}")
    return confidence


def calculate_table_confidence(cell_i, cell_j, headers, data):
    """计算基于表格结构的置信度"""
    confidence = 0.0

    # 如果两个矩形在同一个单元格中
    if cell_i and cell_j and cell_i == cell_j:
        confidence += 0.8
        print(f"  表格规则1匹配: 同一单元格, 置信度+0.8")
        return confidence

    # 如果是表头与数据的关系
    if cell_i in headers and cell_j in data:
        if cell_i['col'] == cell_j['col']:  # 同一列
            confidence += 0.9
            print(
                f"  表格规则2匹配: 表头({cell_i['row']},{cell_i['col']}) -> 数据({cell_j['row']},{cell_j['col']}), 置信度+0.9")
        else:
            confidence += 0.2  # 不同列，但仍是表头与数据
            print(f"  表格规则3匹配: 表头与数据, 置信度+0.2")

    # 如果是同一行的数据
    if cell_i in data and cell_j in data and cell_i['row'] == cell_j['row']:
        confidence += 0.7
        print(f"  表格规则4匹配: 同行数据, 置信度+0.7")

    # 如果是同一列的数据
    if cell_i in data and cell_j in data and cell_i['col'] == cell_j['col']:
        confidence += 0.6
        print(f"  表格规则5匹配: 同列数据, 置信度+0.6")

    print(f"  表格结构置信度: {confidence}")
    return confidence


def find_lower_right_rects(rects_with_format, avg_distance, table_structure=None):
    """基于多算法加权置信度构建矩形层次结构"""
    print(f"构建层次结构: {len(rects_with_format)}个带格式矩形")
    if not rects_with_format:
        print("警告：没有有效矩形用于构建层次结构")
        return {}

    # 构建矩形到单元格的映射
    rect_to_cell = {}
    if table_structure:
        headers = table_structure["headers"]
        data = table_structure["data"]
        all_cells = headers + data
        for cell in all_cells:
            for key, _ in cell["rects"]:
                rect_to_cell[key] = cell

    hierarchy = {}
    for i, (key_i, rect_i, fmt_i) in enumerate(rects_with_format):
        print(f"\n分析矩形 {key_i}: {rect_i}")
        related_rects = []
        for j, (key_j, rect_j, fmt_j) in enumerate(rects_with_format):
            if i == j:
                continue

            print(f"  与矩形 {key_j}: {rect_j} 比较")

            # 计算各策略的置信度
            pos_confidence = calculate_position_confidence(rect_i, rect_j)
            fmt_confidence = calculate_format_confidence(fmt_i, fmt_j)
            dist_confidence = calculate_distance_confidence(rect_i, rect_j, avg_distance)

            # 新增：表格结构置信度
            cell_i = rect_to_cell.get(key_i)
            cell_j = rect_to_cell.get(key_j)
            table_confidence = 0.0
            if table_structure and cell_i and cell_j:
                table_confidence = calculate_table_confidence(
                    cell_i, cell_j,
                    table_structure["headers"],
                    table_structure["data"]
                )

            # 加权综合置信度（包含表格结构）
            total_confidence = (
                    pos_confidence * STRATEGY_WEIGHTS["position"] +
                    fmt_confidence * STRATEGY_WEIGHTS["format"] +
                    dist_confidence * STRATEGY_WEIGHTS["distance"] +
                    table_confidence * STRATEGY_WEIGHTS["table"]
            )

            # 归一化处理
            weight_sum = (
                    STRATEGY_WEIGHTS["position"] +
                    STRATEGY_WEIGHTS["format"] +
                    STRATEGY_WEIGHTS["distance"] +
                    (STRATEGY_WEIGHTS["table"] if table_structure else 0)
            )
            total_confidence /= weight_sum

            print(
                f"  总置信度: {total_confidence:.4f} (位置:{pos_confidence:.4f}, 格式:{fmt_confidence:.4f}, 距离:{dist_confidence:.4f}, 表格:{table_confidence:.4f})")

            # 只有置信度超过阈值才认为有关系
            if total_confidence >= MIN_CONFIDENCE:
                related_rects.append((key_j, total_confidence))
                print(f"  ✅ 建立关系: {key_i} -> {key_j} (置信度:{total_confidence:.4f})")

        # 按置信度排序
        related_rects.sort(key=lambda x: x[1], reverse=True)
        hierarchy[key_i] = [key for key, _ in related_rects]
        print(f"  {key_i} 的关联矩形: {hierarchy[key_i]}")

    return hierarchy


def calculate_average_distance(rects):
    """计算相邻矩形的平均水平距离"""
    distances = []
    for i in range(len(rects) - 1):
        curr_rect = rects[i][1]
        next_rect = rects[i + 1][1]
        if curr_rect[1] == next_rect[1]:  # 同一行
            distances.append(next_rect[0] - curr_rect[2])  # x0_next - x1_curr

    if not distances:
        print("警告：无法计算平均距离，返回默认值50")
        return 50  # 默认值
    avg = sum(distances) / len(distances)
    print(f"平均水平距离: {avg} (基于{len(distances)}个测量值)")
    return avg


def fixed_to_flexible(fixed_area, format_info=None):
    """将固定区域转换为灵活的逻辑层次结构"""
    print("\n==== 开始固定区域到灵活结构的转换 ====")
    print(f"输入: {len(fixed_area)}个固定区域")

    # 1. 行分组和围栏过滤
    lines_by_y0 = sort_fixed_area_into_lines(fixed_area)
    lines_by_y0 = filter_fences(lines_by_y0)

    if not lines_by_y0:
        print("错误：围栏过滤后没有剩余有效行")
        return {"hierarchy": {}, "table_structure": {}}

    # 2. 表格结构分析
    vertical_lines, horizontal_lines = detect_grid_lines(fixed_area)
    cells = partition_cells(fixed_area, vertical_lines, horizontal_lines)

    if not cells:
        print("警告：未能检测到表格单元格")
        table_structure = {}
    else:
        headers, data = identify_header_and_data(cells, format_info or {})
        merged_cells = detect_merged_cells(cells)

        table_structure = {
            "headers": headers,
            "data": data,
            "merged_cells": merged_cells,
            "grid": {
                "vertical_lines": vertical_lines,
                "horizontal_lines": horizontal_lines
            },
            "dimensions": {
                "rows": max(cell['row'] for cell in cells) + 1 if cells else 0,
                "cols": max(cell['col'] for cell in cells) + 1 if cells else 0
            }
        }

        print(f"表格结构分析: {len(headers)}个表头, {len(data)}个数据单元格, {len(merged_cells)}个合并单元格")

    # 3. 添加格式信息
    rects_with_format = []
    for y0, rects in lines_by_y0.items():
        for key, rect in rects:
            fmt = format_info.get(key, {}) if format_info else {}
            rects_with_format.append((key, rect, fmt))

    print(f"最终用于分析的矩形: {len(rects_with_format)}")

    # 4. 计算平均距离
    avg_distance = calculate_average_distance(rects_with_format)

    # 5. 构建层次结构（结合表格结构）
    overall_hierarchy = find_lower_right_rects(rects_with_format, avg_distance, table_structure)

    print("\n最终层次结构:")
    for key, related in overall_hierarchy.items():
        print(f"  {key} -> {related}")

    return {
        "hierarchy": overall_hierarchy,
        "table_structure": table_structure
    }


def search(fixed_area, format_info=None):
    """搜索并返回固定区域的逻辑关系"""
    print("\n==== 开始搜索逻辑关系 ====")
    print(f"输入: {len(fixed_area)}个固定区域")

    if format_info:
        print(f"格式信息: {len(format_info)}个条目")
    else:
        print("警告：未提供格式信息")

    result = fixed_to_flexible(fixed_area, format_info)

    if not result["hierarchy"]:
        print("⚠️ 警告：未找到任何逻辑关系")

    return result