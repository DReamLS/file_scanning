import json

def save_to_json(data, filename):
    try:
        with open(filename, 'w', encoding= "utf-8") as f:
            json.dump(data, f, indent=4)  # 保存 JSON 文件
    except Exception as e:
        print(f"Error saving JSON file: {e}")
        raise  # 将异常抛出以便捕获