def sort_fixed_area_into_lines(fixed_area):
    """将固定区域按y0坐标分组，并在组内按x0坐标排序"""
    all_rects = [(key, rect) for key, rect in fixed_area.items()]
    # 先按y0分组，再在组内按x0排序（从左到右）
    sorted_rects = sorted(all_rects, key=lambda item: (item[1][1], item[1][0]))
    lines_dict = {}
    for key, rect in sorted_rects:
        y0 = rect[1]  # 获取y0坐标
        if y0 not in lines_dict:
            lines_dict[y0] = []
        lines_dict[y0].append((key, rect))
    return lines_dict


def filter_fences(lines_dict, x_threshold=50, y_threshold=30):
    """过滤行内和行间的无效间隔（围栏）"""
    filtered_lines = {}
    for y0, rects in lines_dict.items():
        if len(rects) <= 1:
            filtered_lines[y0] = rects
            continue

        # 行内水平围栏过滤（x方向间隔过大的矩形）
        grouped_rects = [[rects[0]]]
        for i in range(1, len(rects)):
            prev_rect = grouped_rects[-1][-1][1]  # 前一个矩形的坐标
            curr_rect = rects[i][1]
            horizontal_gap = curr_rect[0] - prev_rect[2]  # 当前矩形x0 - 前矩形x1
            if horizontal_gap > x_threshold:
                grouped_rects.append([rects[i]])  # 新建组
            else:
                grouped_rects[-1].append(rects[i])  # 加入当前组

        # 合并分组为列表（每组内矩形连续，组间有围栏）
        filtered_lines[y0] = [rect for group in grouped_rects for rect in group]

    # 行间垂直围栏过滤（删除包含矩形过少的行）
    final_lines = {}
    for y0, rects in filtered_lines.items():
        if len(rects) >= 2:  # 保留至少包含2个矩形的行
            final_lines[y0] = rects

    return final_lines


def find_lower_right_rects(rects):
    """基于位置关系构建矩形层次结构，支持多级父级识别"""
    hierarchy = {}
    for i, (key_i, rect_i) in enumerate(rects):
        related_rects = []
        xi0, yi0, xi1, yi1 = rect_i  # 当前矩形坐标
        for j in range(i + 1, len(rects)):
            key_j, rect_j = rects[j]
            xj0, yj0, xj1, yj1 = rect_j

            # 规则1：右侧且y1更低（同级或子级）
            if xj0 > xi1 and yj1 < yi1:
                related_rects.append(key_j)

            # 规则2：当前矩形包含右侧矩形（父级关系）
            elif xi0 < xj0 and xi1 > xj1 and yi0 < yj0 and yi1 > yj1:
                related_rects.append(key_j)

            # 规则3：右侧矩形与当前矩形垂直方向重叠（同级关系）
            elif xj0 > xi1 and (
                    (yj0 >= yi0 and yj0 <= yi1) or  # 右侧矩形y0在当前矩形范围内
                    (yj1 >= yi0 and yj1 <= yi1)  # 右侧矩形y1在当前矩形范围内
            ):
                related_rects.append(key_j)

        hierarchy[key_i] = related_rects
    return hierarchy


def fixed_to_flexible(fixed_area):
    """将固定区域转换为灵活的逻辑层次结构"""
    lines_by_y0 = sort_fixed_area_into_lines(fixed_area)
    # 新增围栏过滤
    lines_by_y0 = filter_fences(lines_by_y0)

    overall_hierarchy = {}
    for y0, area in lines_by_y0.items():
        # 对每一行中的矩形按x0进行排序（确保顺序）
        sorted_area = sorted(area, key=lambda item: item[1][0])
        # 构建该行内的层次结构
        line_hierarchy = find_lower_right_rects(sorted_area)
        # 将该行的层次结构加入总体层次结构中
        for key, related_keys in line_hierarchy.items():
            if key in overall_hierarchy:
                overall_hierarchy[key].extend(related_keys)
            else:
                overall_hierarchy[key] = related_keys

    return overall_hierarchy


def search(fixed_area):
    """搜索并返回固定区域的逻辑关系"""
    logic = fixed_to_flexible(fixed_area)
    return logic




