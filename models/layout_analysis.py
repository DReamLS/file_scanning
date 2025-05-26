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

    return final_lines


def calculate_average_distance(rects):
    """计算相邻矩形的平均水平距离"""
    distances = []
    for i in range(len(rects) - 1):
        curr_rect = rects[i][1]
        next_rect = rects[i + 1][1]
        if curr_rect[1] == next_rect[1]:  # 同一行
            distances.append(next_rect[0] - curr_rect[2])  # x0_next - x1_curr

    if not distances:
        return 50  # 默认值
    return sum(distances) / len(distances)