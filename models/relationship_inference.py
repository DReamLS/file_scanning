# 策略权重配置
STRATEGY_WEIGHTS = {
    "position": 0.6,  # 位置关系权重
    "format": 0.3,  # 格式特征权重
    "distance": 0.2,  # 距离度量权重
    "table": 0.3  # 表格结构权重
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

    # 规则2：坐标包含（父级关系）
    if xi0 < xj0 and xi1 > xj1 and yi0 < yj0 and yi1 > yj1:
        confidence += 0.9

    # 规则3：垂直重叠（同级关系）
    if xj0 > xi1 and ((yj0 >= yi0 and yj0 <= yi1) or (yj1 >= yi0 and yj1 <= yi1)):
        confidence += 0.7

    return confidence


def calculate_format_confidence(fmt_i, fmt_j):
    """计算基于格式特征的置信度"""
    confidence = 0.0

    # 检查是否有格式信息
    if not fmt_i or not fmt_j:
        return confidence

    # 父级标题通常更大更粗
    if fmt_i.get('bold', False) and (fmt_i.get('font_size', 0) > fmt_j.get('font_size', 0)):
        confidence += 0.6

    # 相同颜色可能属于同一组
    if fmt_i.get('color') == fmt_j.get('color'):
        confidence += 0.3

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

    return confidence


def calculate_table_confidence(cell_i, cell_j, headers, data):
    """计算基于表格结构的置信度"""
    confidence = 0.0

    # 如果两个矩形在同一个单元格中
    if cell_i and cell_j and cell_i == cell_j:
        confidence += 0.8
        return confidence

    # 如果是表头与数据的关系
    if cell_i in headers and cell_j in data:
        if cell_i['col'] == cell_j['col']:  # 同一列
            confidence += 0.9
        else:
            confidence += 0.2  # 不同列，但仍是表头与数据

    # 如果是同一行的数据
    if cell_i in data and cell_j in data and cell_i['row'] == cell_j['row']:
        confidence += 0.7

    # 如果是同一列的数据
    if cell_i in data and cell_j in data and cell_i['col'] == cell_j['col']:
        confidence += 0.6

    return confidence


def find_lower_right_rects(rects_with_format, avg_distance, table_structure=None):
    """基于多算法加权置信度构建矩形层次结构"""
    if not rects_with_format:
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
        related_rects = []
        for j, (key_j, rect_j, fmt_j) in enumerate(rects_with_format):
            if i == j:
                continue

            # 计算各策略的置信度
            pos_confidence = calculate_position_confidence(rect_i, rect_j)
            fmt_confidence = calculate_format_confidence(fmt_i, fmt_j)
            dist_confidence = calculate_distance_confidence(rect_i, rect_j, avg_distance)

            # 表格结构置信度
            cell_i = rect_to_cell.get(key_i)
            cell_j = rect_to_cell.get(key_j)
            table_confidence = 0.0
            if table_structure and cell_i and cell_j:
                table_confidence = calculate_table_confidence(
                    cell_i, cell_j,
                    table_structure["headers"],
                    table_structure["data"]
                )

            # 加权综合置信度
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

            # 只有置信度超过阈值才认为有关系
            if total_confidence >= MIN_CONFIDENCE:
                related_rects.append((key_j, total_confidence))

        # 按置信度排序
        related_rects.sort(key=lambda x: x[1], reverse=True)
        hierarchy[key_i] = [key for key, _ in related_rects]

    return hierarchy