'''
    pre-requisites :    1. OpenALPR account
                        2. generate key and api url

    want to make it custom ?:
        https://www.youtube.com/watch?v=0-4p_QgrdbE
'''

import json
import base64
import requests

# variables #####################################
Auth_key = "your key "
image_path = ""
url = f"your api url + {Auth_key}"

# preparing image data and send to server ##########
with open(image_path, "rb") as If:
    image_base64 = base64.b64encode(If.read())

response = requests.post(url, data=image_base64)

# getting details from response ####################
num_plate = (json.dumps(response.json(), indent=2))
print(num_plate)
info = (list(num_plate.split("candidates")))
plate = info[1]
plate = plate.split(",") [0:3]
p = plate[1]
p1 = p.split(":")
number = p1[1]
number = number.replace('"', '')
number = number.lstrip()
print(number)

if number == "5198 HJO":
    print('''
-----------------------------------------------------
Owner Name : Cristiano Ronaldo
Hometown : Hospital Dr. Nélio Mendonça, Funchal, Portugal
Car : Lamborghini Aventador
Vehicle Plate : 5198 HJO
    ''')
