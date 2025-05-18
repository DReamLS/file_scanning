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


# 策略权重配置
STRATEGY_WEIGHTS = {
    "position": 0.6,  # 位置关系权重
    "format": 0.3,  # 格式特征权重
    "distance": 0.2  # 距离度量权重
}

# 降低最小置信度阈值，便于调试
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


def find_lower_right_rects(rects_with_format, avg_distance):
    """基于多算法加权置信度构建矩形层次结构"""
    print(f"构建层次结构: {len(rects_with_format)}个带格式矩形")
    if not rects_with_format:
        print("警告：没有有效矩形用于构建层次结构")
        return {}

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

            # 加权综合置信度
            total_confidence = (
                    pos_confidence * STRATEGY_WEIGHTS["position"] +
                    fmt_confidence * STRATEGY_WEIGHTS["format"] +
                    dist_confidence * STRATEGY_WEIGHTS["distance"]
            )

            print(
                f"  总置信度: {total_confidence} (位置:{pos_confidence}, 格式:{fmt_confidence}, 距离:{dist_confidence})")

            # 只有置信度超过阈值才认为有关系
            if total_confidence >= MIN_CONFIDENCE:
                related_rects.append((key_j, total_confidence))
                print(f"  ✅ 建立关系: {key_i} -> {key_j} (置信度:{total_confidence})")

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

    lines_by_y0 = sort_fixed_area_into_lines(fixed_area)
    lines_by_y0 = filter_fences(lines_by_y0)

    if not lines_by_y0:
        print("错误：围栏过滤后没有剩余有效行")
        return {}

    # 添加格式信息（如果有）
    rects_with_format = []
    for y0, rects in lines_by_y0.items():
        for key, rect in rects:
            fmt = format_info.get(key, {}) if format_info else {}
            rects_with_format.append((key, rect, fmt))

    print(f"最终用于分析的矩形: {len(rects_with_format)}")

    # 计算平均距离用于距离度量
    avg_distance = calculate_average_distance(rects_with_format)

    overall_hierarchy = {}
    for y0, rects in lines_by_y0.items():
        print(f"\n处理行 y0={y0}: {len(rects)}个矩形")

        # 提取当前行的矩形及格式信息
        line_rects_with_format = [
            (key, rect, fmt)
            for key, rect, fmt in rects_with_format
            if rect[1] == y0  # y0匹配当前行
        ]

        if not line_rects_with_format:
            print(f"  行y0={y0}没有有效矩形")
            continue

        # 构建该行内的层次结构
        line_hierarchy = find_lower_right_rects(line_rects_with_format, avg_distance)

        # 合并层次结构
        for key, related_keys in line_hierarchy.items():
            if key in overall_hierarchy:
                overall_hierarchy[key].extend(related_keys)
            else:
                overall_hierarchy[key] = related_keys

    print("\n最终层次结构:")
    for key, related in overall_hierarchy.items():
        print(f"  {key} -> {related}")

    return overall_hierarchy


def search(fixed_area, format_info=None):
    """搜索并返回固定区域的逻辑关系"""
    print("\n==== 开始搜索逻辑关系 ====")
    print(f"输入: {len(fixed_area)}个固定区域")

    if format_info:
        print(f"格式信息: {len(format_info)}个条目")
    else:
        print("警告：未提供格式信息")

    logic = fixed_to_flexible(fixed_area, format_info)

    if not logic:
        print("⚠️ 警告：未找到任何逻辑关系")

    return logic

