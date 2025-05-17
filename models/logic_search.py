import jieba
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

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

# 计算文本相似度
def calculate_text_similarity(text1, text2):
    texts = [text1, text2]
    words_list = [" ".join(jieba.cut(text)) for text in texts]
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(words_list)
    similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])
    return similarity[0][0]

# 结合文本相似度和位置关系确定逻辑关系
def find_related_rects(rects, fixed_area, similarity_threshold=0.3):
    hierarchy = {}
    for i, (key_i, rect_i) in enumerate(rects):
        related_rects = []
        text_i = key_i.split(" 行：")[0]  # 提取文本
        for j, (key_j, rect_j) in enumerate(rects):
            if i != j:
                text_j = key_j.split(" 行：")[0]  # 提取文本
                similarity = calculate_text_similarity(text_i, text_j)
                # 考虑位置关系和文本相似度
                if (rect_j[3] < rect_i[3] or similarity >= similarity_threshold):
                    related_rects.append(key_j)
        hierarchy[key_i] = related_rects
    return hierarchy

def fixed_to_flexible(fixed_area):
    lines_by_y0 = sort_fixed_area_into_lines(fixed_area)
    overall_hierarchy = {}

    for y0, area in lines_by_y0.items():
        # 对每一行中的矩形按 x0 进行排序
        sorted_area = sorted(area, key=lambda item: item[1][0])
        # 构建该行内的层次结构
        line_hierarchy = find_related_rects(sorted_area, fixed_area)
        # 将该行的层次结构加入总体层次结构中
        for key, related_keys in line_hierarchy.items():
            if key in overall_hierarchy:
                overall_hierarchy[key].extend(related_keys)
            else:
                overall_hierarchy[key] = related_keys

    return overall_hierarchy

def search(fixed_area):
    logic = fixed_to_flexible(fixed_area)
    return logic




