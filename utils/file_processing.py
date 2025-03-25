import os
from werkzeug.utils import secure_filename
from utils.pdf_utils import validate_pdf


def save_file(file, upload_dir):
    """
    保存上传的文件并验证是否为 PDF。
    确保目录存在且具有正确的权限。
    """
    # 创建上传目录并设置权限
    os.makedirs(upload_dir, exist_ok=True)
    os.chmod(upload_dir, 0o755)  # 授予读、写、执行权限

    # 构造文件路径
    filepath = os.path.join(upload_dir, secure_filename(file.filename))

    # 保存文件
    file.save(filepath)

    # 验证文件是否为有效的 PDF
    if not validate_pdf(filepath):
        os.remove(filepath)  # 删除无效文件
        raise ValueError("Invalid PDF file")

    # 确保保存的文件具有适当的权限
    os.chmod(filepath, 0o644)  # 授予读权限（所有者可写）

    return filepath


def delete_file(filepath):
    """
    删除文件，并处理可能的权限问题。
    """
    try:
        if os.path.exists(filepath):
            os.chmod(filepath, 0o644)  # 确保文件有删除权限
            os.remove(filepath)
    except Exception as e:
        print(f"Error deleting file: {e}")