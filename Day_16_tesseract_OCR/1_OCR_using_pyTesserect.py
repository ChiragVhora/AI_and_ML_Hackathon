'''
    download tesseract from here : https://github.com/UB-Mannheim/tesseract/wiki
    install pytesseract lib
'''
try:
    from PIL import Image
except ImportError:
    import Image

import pytesseract

# set up exe ##############33
tesseract_exe_path = r"C:\Users\chira\AppData\Local\Programs\Tesseract-OCR\tesseract.exe"
pytesseract.pytesseract.tesseract_cmd = tesseract_exe_path
# func #################3
def recognize_text(image_path):
    text_from_img = pytesseract.image_to_string(Image.open(image_path))
    return text_from_img

#######################
input_file_path = "text_p.jpg"

info_text = recognize_text(input_file_path)
print(info_text)
Da
with open("result.txt", "w") as f:
    f.write(info_text)
print("write successful!")