def find_father(logic, element):
    """
    找到给定元素的所有上级节点（父节点）。

    :param logic: 逻辑链字典，格式为 {父节点: [子节点1, 子节点2, ...]}
    :param element: 需要查找父节点的元素
    :return: 包含所有父节点的列表
    """
    father_list = []
    visited = set()  # 用于避免循环引用

    def dfs(node):
        for key, values in logic.items():
            if node in values and key not in visited:
                father_list.append(key)
                visited.add(key)
                dfs(key)

    dfs(element)
    return father_list


def edge_logic_find(logic):
    """
    找到逻辑链中的“边缘”元素及其所有上级节点。

    :param logic: 逻辑链字典，格式为 {父节点: [子节点1, 子节点2, ...]}
    :return: 边缘元素及其所有上级节点的字典
    """
    edge = {}
    for element in logic.keys():
        if not logic[element]:
            edge[element] = find_father(logic, element)
            # 包括自身节点
            edge[element].insert(0, element)

    return edge


def find_nearest_fixed_areas(unfixed, fixed_area, unfixed_area):
    """
    对于给定的 unfixed_area，在其左侧和上侧找到 x 和 y 坐标相同的最近 fixed_area。

    :param unfixed: 单个非固定区域坐标 (x0, y0, x1, y1)
    :param fixed_area: 固定区域字典 {文本: (x0, y0, x1, y1), ...}
    :param unfixed_area: [(x0, y0, x1, y1),...]
    :return: (最接近的 x 方向 fixed_area, 最接近的 y 方向 fixed_area)
    """
    nearest_x = None
    nearest_y = None
    min_distance_x = float('inf')
    min_distance_y = float('inf')

    for text, position in fixed_area.items():
        fx0, fy0, fx1, fy1 = position

        # 检查是否在左侧 (fx1 <= unfixed[0]) 或 上侧 (fy1 <= unfixed[1])
        if fx1 <= unfixed[0] and (fy0 <= unfixed[1] and fy1 >= unfixed[3]):
            distance = abs(unfixed[0] - fx1)  # 使用左侧的距离
            if distance < min_distance_y:
                min_distance_y = distance
                nearest_y = text

        if fy1 <= unfixed[1] and (fx0 <= unfixed[0] and fx1 >= unfixed[2]):
            distance = abs(unfixed[1] - fy1)  # 使用左侧的距离
            if distance < min_distance_x:
                min_distance_x = distance
                nearest_x = text

    # 检查 x 方向上的 nearest_x 下方有没有相邻的 unfixed
    if nearest_x is not None:
        fx0, fy0, fx1, fy1 = fixed_area[nearest_x]
        for position in unfixed_area:
            if fx0 == position[0] == fx1 and fy1 == position[1]:  # 下方相邻
                nearest_x = None
                break

    # 检查 y 方向上的 nearest_y 右方有没有相邻的 unfixed
    if nearest_y is not None:
        fx0, fy0, fx1, fy1 = fixed_area[nearest_y]
        for position in unfixed_area:
            if fy0 == position[1] == fy1 and fx1 == position[0]:  # 右方相邻
                nearest_y = None
                break

    return nearest_x, nearest_y


def flexible_abstract(logic, fixed_area, unfixed_area):
    """
    根据逻辑链和固定区域，确定哪些 unfixed_area 单元格属于“灵活区域”，并找到与之最近的 fixed_area 的逻辑链。

    :param logic: 逻辑链字典 {父节点: [子节点1, 子节点2, ...]}
    :param fixed_area: 固定区域字典 {文本: (x0, y0, x1, y1), ...}
    :param unfixed_area: 非固定区域列表 [(x0, y0, x1, y1), ...]
    :return: 灵活区域字典 {灵活区域单元格坐标: [该节点+父节点列表], ...}
    """
    result = {}
    edge_logic = edge_logic_find(logic)  # 预先计算所有边缘元素及其上级节点

    for unfixed in unfixed_area:
        nearest_x, nearest_y = find_nearest_fixed_areas(unfixed, fixed_area, unfixed_area)

        # 如果两个方向的 fixed_area 相同，则只保留一个逻辑链
        if nearest_x == nearest_y and nearest_x is not None:
            result[unfixed] = edge_logic[nearest_x]
        else:
            # 获取 x 和 y 方向的逻辑链
            chain_x = edge_logic.get(nearest_x, []) if nearest_x is not None else []
            chain_y = edge_logic.get(nearest_y, []) if nearest_y is not None else []
            if chain_x in chain_y:
                result[unfixed] = chain_y
            elif chain_y in chain_x:
                result[unfixed] = chain_x
            else:
                result[unfixed] = (chain_x,chain_y)


    return result