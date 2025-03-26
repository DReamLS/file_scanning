import PyPDF2


def validate_pdf(file_path):
    """
    验证给定路径的文件是否为有效的 PDF 文件。

    参数:
        file_path (str): 要验证的文件路径。

    返回:
        bool: 如果文件是有效的 PDF，则返回 True；否则返回 False。

    异常:
        FileNotFoundError: 如果文件不存在。
        PermissionError: 如果没有权限访问文件。
    """
    try:
        with open(file_path, 'rb') as f:
            reader = PyPDF2.PdfReader(f)
            # 检查是否有页面存在作为有效性的一个指标
            if len(reader.pages) > 0:
                return True
            else:
                # 空PDF也可能被认为是有效的，根据业务需求调整此处逻辑
                return True
    except FileNotFoundError:
        print(f"文件 {file_path} 未找到")
        return False
    except PermissionError:
        print(f"没有权限访问文件 {file_path}")
        return False
    except PyPDF2.errors.PdfReadError:
        # PdfReadError 是 PyPDF2 中用于表示PDF读取过程中出现错误的异常类型
        print(f"文件 {file_path} 不是一个有效的PDF文件")
        return False