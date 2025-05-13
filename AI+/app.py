from flask import Flask, jsonify, request, send_from_directory

app = Flask(__name__)

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' in request.files:
        file = request.files['file']
        data = file.read().decode('utf-8')
    else:
        try:
            data = request.get_data(as_text=True)
        except Exception as e:
            return jsonify({"error": str(e)}), 400

    return data, 200

@app.route('/')
def serve_homepage():
    return send_from_directory('.', '../templates/visualize.html')

if __name__ == '__main__':
    app.run(debug=True)