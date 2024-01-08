import cv2
import numpy as np
import requests
import json
import base64
# from PIL import Image

# Used to test if TD runs correctly
# test_image = np.ones((200,200,3)) * 200
# cv2.imwrite(f'{parent().par.Imgoutdir}/test.png', test_image)

def img_to_base64(img):
    retval, buffer = cv2.imencode('.jpg', img)
    jpg_as_text = base64.b64encode(buffer)
    return str(jpg_as_text)

class Point():
    def __init__(self, X, Y):
        self.X = X
        self.Y = Y
    
class Rect():
    def __init__(self, center, shape):
        self.X0 = int(center[0] - shape[0] / 2)
        self.Y0 = int(center[1] - shape[1] / 2)
        self.W = int(shape[0])
        self.H = int(shape[1])
        self.X1 = int(center[0] + shape[0] / 2)
        self.Y1 = int(center[1] + shape[1] / 2)

class TDCanvas():
    def __init__(self, size):
        self.size = size
        self.images = []
        self.positions = []
        self.mask = np.zeros((size[0], size[1], 1))
        self.content = np.ones((size[0], size[1], 4)) * 255
        self.counter = 0
    
    def add_initial_image(self, image, position):
        self.images.append(image)
        self.positions.append(position)

        image_rect = Rect(position, image.shape)

        self.fill_image_to_rect(self.content, image_rect, image)
        self.fill_image_to_rect(self.mask, image_rect, image[:,:,3:])
        self.serialize_content()
        self.serialize_mask()
    
    def generate_new_tile(self, rect):
        input_image = self.content[rect.X0: rect.X1, rect.Y0:rect.Y1, :]
        input_mask = np.ones((rect.W, rect.H, 1)) * 255
        existing_mask = self.mask[rect.X0: rect.X1, rect.Y0:rect.Y1, :]
        complimentary_mask = input_mask - existing_mask

        input_image_base64_str = img_to_base64(input_image)
        input_mask_base64_str = img_to_base64(complimentary_mask)

        payload = json.dumps({'prompt': str(parent().par.Prompt),'image':input_image_base64_str, 'mask': input_mask_base64_str})
        response = requests.post('http://127.0.0.1:8001/outpainting', headers={'Accept': 'application/json'}, json = payload)

        if response.status_code == 200:
            res_payload_dict = response.json()
            print(res_payload_dict)
            response_image = base64.b64decode(res_payload_dict["image"])

            outpainting_result_path = f'image/output_image_{self.counter}.jpg'
            # Write to a file to show conversion worked
            with open(f'{outpainting_result_path}', 'wb') as f_output:
                f_output.write(response_image)
            
            infill_image = cv2.imread(outpainting_result_path)
            with open(f'{parent().par.Imgoutdir}/result.txt', 'w') as f:
                #f.write(str(response.json()))
                f.write(f'{self.content.shape},{rect.W}{rect.H}{infill_image.shape}')
            infill_image = cv2.resize(infill_image, (rect.H, rect.W))
            self.fill_image_to_rect(self.mask, rect, input_mask)
            self.serialize_mask()
            self.content[:,:,3:] = self.mask
            self.fill_image_to_rect(self.content[:,:,:3], rect, infill_image)
            self.serialize_content()
            self.counter += 1
            
    def serialize_content(self):
        cv2.imwrite(f'image/content_{self.counter}.png', self.content)

    def serialize_mask(self):
        cv2.imwrite(f'image/mask_{self.counter}.png', self.mask)
    
    @staticmethod
    def fill_image_to_rect(image, rect, infill):
        image[rect.X0:rect.X1, rect.Y0: rect.Y1, :] = infill[:,:,:]


#payload = json.dumps({'prompt': parent().par.Prompt})
img = op('image_a')
rgbA = img.numpyArray(delayed=False)*255
rgbA = np.flipud(rgbA)
rgbA = rgbA[:,:,:] #strip off alpha
rgbA = rgbA.astype(np.uint8)
rgbA = cv2.cvtColor(rgbA, cv2.COLOR_BGR2RGBA)

canvas = TDCanvas((int(parent().par.Canvassize1), int(parent().par.Canvassize2)))
canvas.add_initial_image(rgbA, (1024, 1024))

for i in range(200):
    canvas.generate_new_tile(Rect((np.random.randint(300, 1600, 1),np.random.randint(300, 2000, 1)),(300,300)))

#im = cv2.imencode('.jpg', cv2.cvtColor(rgbA, cv2.COLOR_BGR2RGB))[1].tobytes()


