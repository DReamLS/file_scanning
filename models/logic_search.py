def sort_fixed_area_into_lines(fixed_area):
    all_rects = [(key, rect) for key, rect in fixed_area.items()]
    sorted_rects = sorted(all_rects, key=lambda item: (item[1][1], item[1][0]))
    lines_dict = {}
    for key, rect in sorted_rects:
        y0 = rect[1]  # 获取矩形的 y0 坐标
        if y0 not in lines_dict:
            lines_dict[y0] = []  # 如果 y0 不在字典中，则创建一个新的空列表
        lines_dict[y0].append((key, rect))  # 将矩形添加到对应 y0 的列表中
    return lines_dict


def find_lower_right_rects(rects):

    """
    找出同一行内每个矩形右侧且 y1 更小的矩形。

    参数:
    rects (list of tuple): 每个元素是 (key, (x0, y0, x1, y1)) 的元组列表，
                           已按照 x0 排序。

    返回:
    hierarchy (dict): 矩形之间的层次关系，键为矩形的 key，值为位于其右侧且更高的矩形 keys 列表。
    """
    hierarchy = {}
    for i, (key_i, rect_i) in enumerate(rects):
        higher_right_rects = []
        for j in range(i + 1, len(rects)):
            key_j, rect_j = rects[j]
            if rect_j[3] < rect_i[3]:  # 如果右侧矩形的 y1 小于当前矩形的 y1
                higher_right_rects.append(key_j)
            else:
                break  # 因为已经按 x0 排序，所以可以在这里停止搜索
        hierarchy[key_i] = higher_right_rects
    return hierarchy


def fixed_to_flexible(fixed_area):
    lines_by_y0 = sort_fixed_area_into_lines(fixed_area)
    overall_hierarchy = {}

    for y0, area in lines_by_y0.items():
        # 对每一行中的矩形按 x0 进行排序
        sorted_area = sorted(area, key=lambda item: item[1][0])
        # 构建该行内的层次结构
        line_hierarchy = find_lower_right_rects(sorted_area)
        # 将该行的层次结构加入总体层次结构中
        for key, higher_keys in line_hierarchy.items():
            overall_hierarchy[key] = higher_keys


    return overall_hierarchy


def search(fixed_area):
    logic = fixed_to_flexible(fixed_area)

    return logic
