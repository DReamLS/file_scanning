import base64
import urllib
import requests
import json
API_KEY = "Ja4NylqWXBOcqK0hoXLACqWF"
SECRET_KEY = "GA1cdLZC62FKRx6oeSFlisaQxRaR3Gd0"


def process(path,pdf_num,):
    url = "https://aip.baidubce.com/rest/2.0/ocr/v1/remove_handwriting?access_token=" + get_access_token()
    pdf_file = get_file_content_as_base64(path,True)
    payload = f"pdf_file={pdf_file}&pdf_file_num={pdf_num}&enable_detect=true"
    headers = {
        'Content-Type': 'application/x-www-form-urlencoded',
        'Accept': 'application/json'
    }

    response = requests.request("POST", url, headers=headers, data=payload.encode("utf-8"))
    print(response.text)
    return response.text


def get_file_content_as_base64(path, urlencoded=False):
    """
    获取文件base64编码
    :param path: 文件路径
    :param urlencoded: 是否对结果进行urlencoded
    :return: base64编码信息
    """
    with open(path, "rb") as f:
        content = base64.b64encode(f.read()).decode("utf8")
        if urlencoded:
            content = urllib.parse.quote_plus(content)
    return content


def get_access_token():
    """
    使用 AK，SK 生成鉴权签名（Access Token）
    :return: access_token，或是None(如果错误)
    """
    url = "https://aip.baidubce.com/oauth/2.0/token"
    params = {"grant_type": "client_credentials", "client_id": API_KEY, "client_secret": SECRET_KEY}
    return str(requests.post(url, params=params).json().get("access_token"))


def save_img(response_dict):

    base64_image_string = response_dict["image_processed"]  # 这里仅作为示例展示了一部分

    # 解码Base64字符串
    image_data = base64.b64decode(base64_image_string)

    # 指定输出文件路径和名称
    output_file_path = "output_image.jpg"  # 根据需要修改扩展名(.png, .jpg等)

    # 将解码后的数据写入文件
    with open(output_file_path, 'wb') as file:
        file.write(image_data)

    print(f"图像已成功保存为 {output_file_path}")

if __name__ == "__main__":
    response_text = process(r"D:\大一年度项目\AI+\backend\scanned-test_Page1.pdf",1)
    try:
        response_dict = json.loads(response_text)  # 将JSON字符串转换为字典
    except json.JSONDecodeError as e:
        print(f"JSON解析错误: {e}")
    else:
        save_img(response_dict)