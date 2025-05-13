from flask import Flask, request, jsonify, send_file, render_template
from flask_cors import CORS  # 处理跨域请求
import os
from models.demo import process1 as process1_function, process2 as process2_function, process3 as process3_function
from utils.file_processing import save_file, delete_file  # 文件处理工具
import json
# 创建 Flask 应用
app = Flask(__name__)

# 启用 CORS 支持
CORS(app, resources={r"/process*": {"origins": "*"}})

# 配置上传目录和输出文件路径
UPLOAD_DIR = 'uploads'
OUTPUT_FILE = 'output.json'

# 确保上传目录存在
os.makedirs(UPLOAD_DIR, exist_ok=True)

# 路由：渲染第一个前端页面（index.html）
@app.route('/')
def serve_index():
    return render_template('index.html')


# 路由：渲染第二个前端页面（visualize.html）
@app.route('/visualize')
def serve_visualize():
    return render_template('visualize.html')


def handle_process_request(req, process_func):
    """
    处理文件上传和逻辑处理的通用函数。
    """
    try:
        # 检查是否提供了必要的文件字段
        if 'scanFile' not in req.files or 'templateFile' not in req.files:
            return jsonify({"error": "Missing file part"}), 400

        scan_file = req.files['scanFile']
        template_file = req.files['templateFile']

        # 检查文件名是否为空
        if scan_file.filename == '' or template_file.filename == '':
            return jsonify({"error": "No selected file"}), 400

        # 检查文件类型是否为 PDF
        if not (scan_file.filename.endswith('.pdf') and template_file.filename.endswith('.pdf')):
            return jsonify({"error": "Invalid file type. Only PDF files are allowed."}), 400

        # 保存上传的文件
        scan_path = save_file(scan_file, UPLOAD_DIR)
        template_path = save_file(template_file, UPLOAD_DIR)

        # 调用核心处理逻辑
        result = process_func(template_path, scan_path)

        # 将结果保存为 JSON 文件
        with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
            import json
            json.dump(result, f, indent=4, ensure_ascii=False)

        # 删除临时文件
        delete_file(scan_path)
        delete_file(template_path)

        return jsonify(result)  # 返回处理结果

    except Exception as e:
        # 捕获异常并返回错误信息
        print(f"Error processing files: {e}")
        return jsonify({"error": str(e)}), 500


@app.route('/process1', methods=['POST'])
def process_files_1():
    return handle_process_request(request, process1_function)

@app.route('/process2', methods=['POST'])
def process_files_2():
    return handle_process_request(request, process2_function)

@app.route('/process3', methods=['POST'])
def process_files_3():
    def handle_process_request(req, process_func):
        # 只需要扫描文件
        if 'scanFile' not in req.files:
            return jsonify({"error": "Missing scan file part"}), 400

        scan_file = req.files['scanFile']

        if scan_file.filename == '':
            return jsonify({"error": "No selected scan file"}), 400

        if not scan_file.filename.endswith('.pdf'):
            return jsonify({"error": "Invalid scan file type. Only PDF files are allowed."}), 400

        scan_path = save_file(scan_file, UPLOAD_DIR)
        result = process_func(scan_path)
        delete_file(scan_path)

        return jsonify(result)

    return handle_process_request(request, process3_function)


# 处理JSON数据上传（用于第二个前端）
@app.route('/upload', methods=['POST'])
def upload_json():
    try:
        # 支持两种方式接收数据：JSON格式或表单文件
        if request.is_json:
            data = request.get_json()
        else:
            file = request.files['jsonData']
            if file:
                data = json.load(file)
            else:
                return jsonify({"error": "No JSON data provided"}), 400

        # 保存JSON数据到文件（可选）
        with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=4)

        return jsonify(data)

    except Exception as e:
        return jsonify({"error": str(e)}), 500


# 确保/download路由支持GET方法
@app.route('/download', methods=['GET'])
def download():
    if os.path.exists(OUTPUT_FILE):
        return send_file(OUTPUT_FILE, mimetype='application/json')
    else:
        return jsonify({"error": "No data available"}), 404


if __name__ == '__main__':
    app.run(debug=True)