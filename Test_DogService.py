
#%%
import requests 
import os, json, base64
from io import BytesIO
import matplotlib.pyplot as plt
from PIL import Image
import urllib.request
import io

##Get random dog
def get_random_dog():
    r = requests.get(url ="https://dog.ceo/api/breeds/image/random")
    URL= r.json()['message']
    return URL

##Get Random Dog Image. Or replace the URL with a picture of your own dog
URL = get_random_dog()

with urllib.request.urlopen(URL) as url:
    test_img = io.BytesIO(url.read())

# ## If you downloaded the dataset, you can try this arbitrary image from the test dataset
# # test_img = os.path.join('breeds-10', 'val', 'n02085620-Chihuahua', 'n02085620_1271.jpg') 

plt.imshow(Image.open(test_img))

with urllib.request.urlopen(URL) as url:
    test_img = io.BytesIO(url.read())

plt.imshow(Image.open(test_img))


#%%
def imgToBase64(img):
    imgio = BytesIO()
    img.save(imgio, 'JPEG')
    img_str = base64.b64encode(imgio.getvalue())
    return img_str.decode('utf-8')

base64Img = imgToBase64(Image.open(test_img))

service_uri = "http://52.190.24.229:80/score"
input_data = json.dumps({'data': base64Img})
headers = {'Content-Type':'application/json'}
result = requests.post(service_uri, input_data, headers=headers).text

print(json.loads(result))
