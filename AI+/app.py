from flask import Flask, jsonify, request, send_from_directory
import json
import uuid
app = Flask(__name__, static_folder='.')


def transform_json(data, parent_key='', path=[]):
    nodes = []
    links = []

    if isinstance(data, dict):
        for key, value in data.items():
            current_path = path + [key]
            node_id = '_'.join(current_path)

            # 如果当前层级不是根节点，则创建一条从父节点到当前节点的边
            if parent_key:
                links.append({'source': parent_key, 'target': node_id})

            # 为当前节点添加信息
            nodes.append({
                'id': node_id,
                'label': key,
                'group': str(uuid.uuid4())  # 添加一个唯一组标识符，避免冲突
            })

            # 递归处理子节点
            sub_nodes, sub_links = transform_json(value, node_id, current_path)
            nodes.extend(sub_nodes)
            links.extend(sub_links)

    elif isinstance(data, list):
        for index, item in enumerate(data):
            current_path = path + [str(index)]
            node_id = '_'.join(current_path)

            # 如果列表项是一个复杂类型（字典或列表），则递归处理
            if isinstance(item, (dict, list)):
                sub_nodes, sub_links = transform_json(item, parent_key, current_path)
                nodes.extend(sub_nodes)
                links.extend(sub_links)

                # 如果父节点存在，则为每个子节点创建边
                if parent_key:
                    for sub_node in sub_nodes:
                        links.append({'source': parent_key, 'target': sub_node['id']})
            else:
                # 对于简单类型，直接作为叶子节点处理
                nodes.append({
                    'id': node_id,
                    'label': str(item),
                    'group': str(uuid.uuid4())
                })
                if parent_key:
                    links.append({'source': parent_key, 'target': node_id})

    return nodes, links

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' in request.files:
        file = request.files['file']
        data = json.load(file)
    else:
        try:
            data = request.get_json(force=True)
        except Exception as e:
            return jsonify({"error": str(e)}), 400

    nodes, links = transform_json(data)
    graph_data = {'nodes': nodes, 'links': links}
    return jsonify(graph_data)

@app.route('/')
def serve_homepage():
    return send_from_directory('.', 'index.html')


if __name__ == '__main__':
    app.run(debug=True)