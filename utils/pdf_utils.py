# utils/pdf_utils.py
def validate_pdf(file_path):
    """验证文件是否为有效的 PDF 文件"""
    try:
        import PyPDF2
        with open(file_path, 'rb') as f:
            reader = PyPDF2.PdfReader(f)
            if len(reader.pages) > 0:
                return True
    except Exception:
        return False
    return False