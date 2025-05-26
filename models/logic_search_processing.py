from .layout_analysis import sort_fixed_area_into_lines, filter_fences, calculate_average_distance
from .table_structure import detect_grid_lines, partition_cells, identify_header_and_data, detect_merged_cells
from .relationship_inference import find_lower_right_rects


def fixed_to_flexible(fixed_area, format_info=None):
    """将固定区域转换为灵活的逻辑层次结构"""
    # 1. 行分组和围栏过滤
    lines_by_y0 = sort_fixed_area_into_lines(fixed_area)
    lines_by_y0 = filter_fences(lines_by_y0)

    if not lines_by_y0:
        return {"hierarchy": {}, "table_structure": {}}

    # 2. 表格结构分析
    vertical_lines, horizontal_lines = detect_grid_lines(fixed_area)
    cells = partition_cells(fixed_area, vertical_lines, horizontal_lines)

    if not cells:
        table_structure_result = {}
    else:
        headers, data = identify_header_and_data(cells, format_info or {})
        merged_cells = detect_merged_cells(cells)

        table_structure_result = {
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

    # 3. 添加格式信息
    rects_with_format = []
    for y0, rects in lines_by_y0.items():
        for key, rect in rects:
            fmt = format_info.get(key, {}) if format_info else {}
            rects_with_format.append((key, rect, fmt))

    # 4. 计算平均距离
    avg_distance = calculate_average_distance(rects_with_format)

    # 5. 构建层次结构（结合表格结构）
    overall_hierarchy = find_lower_right_rects(rects_with_format, avg_distance, table_structure_result)

    return overall_hierarchy,



def search(fixed_area, format_info=None):
    """搜索并返回固定区域的逻辑关系"""
    result = fixed_to_flexible(fixed_area, format_info)
    return result