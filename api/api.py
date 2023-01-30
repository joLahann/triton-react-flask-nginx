import time
from flask import Flask,request, make_response
import pandas as pd
import tritonclient.grpc as grpcclient
from tqdm import tqdm
from glob import glob
import numpy as np


from PIL import Image

def image_transform_onnx(path: str, size: int) -> np.ndarray:
    '''Image transform helper for onnx runtime inference.'''

    image = Image.open(path)
    image = image.resize((size,size))

    # now our image is represented by 3 layers - Red, Green, Blue
    # each layer has a 224 x 224 values representing
    image = np.array(image)

    # dummy input for the model at export - torch.randn(1, 3, 224, 224)
    image = image.transpose(2,0,1).astype(np.float32)

    # our image is currently represented by values ranging between 0-255
    # we need to convert these values to 0.0-1.0 - those are the values that are expected by our model
    image /= 255
    image = image[None, ...]
    return image

app = Flask(__name__, static_folder='../build', static_url_path='/')


@app.errorhandler(404)
def not_found(e):
    return app.send_static_file('index.html')


@app.route('/')
def index():
    return app.send_static_file('index.html')


@app.route('/api/time', methods=['Get'])
def get_current_time():
    return {'time': time.time()}

@app.route('/api/journal_entries', methods=['Get','Post'])
def get_journal_entries():
    args = request.args
    mandant_id = args.get('mandant_id')
    year = args.get('year')
    if None  in (mandant_id, year):
        return make_response({'message':"The parameters mandant_id and year has to be given." },400)

    data = {'Product': ['ABC','DDD','XYZ','AAA','CCC','PPP','NNN','RRR'],
              'Price': [630,790,250,370,880,1250,550,700],
           'Discount': ['No','Yes','No','Yes','Yes','No','No','Yes'],
           'Mandant':[mandant_id,0,0,0,0,0,0,0],
           'Year':[year,0,0,0,0,0,0,0]
            }

    df = pd.DataFrame(data, columns = ['Product','Price','Discount','Mandant','Year'])
    return df.to_dict() 



@app.route('/api/resnet', methods=['Get'])
def test_triton():
    TEST_DATA_PATH = glob("/app/*.jpg")
    INPUT_SHAPE = (224, 224)

    TRITON_IP = "triton_server"
    TRITON_PORT = 8001
    MODEL_NAME = "resnet"
    INPUTS = []
    OUTPUTS = []
    INPUT_LAYER_NAME = "input"
    OUTPUT_LAYER_NAME = "output"

    INPUTS.append(grpcclient.InferInput(INPUT_LAYER_NAME, [1, 3, INPUT_SHAPE[0], INPUT_SHAPE[1]], "FP32"))
    OUTPUTS.append(grpcclient.InferRequestedOutput(OUTPUT_LAYER_NAME))
    TRITON_CLIENT = grpcclient.InferenceServerClient(url=f"{TRITON_IP}:{TRITON_PORT}")


    labels = ['Dog', 'Cat']
    result ={}
    for test_path in tqdm(TEST_DATA_PATH):
        INPUTS[0].set_data_from_numpy(image_transform_onnx(test_path, 224))

        results = TRITON_CLIENT.infer(model_name=MODEL_NAME, inputs=INPUTS, outputs=OUTPUTS, headers={})
        output = np.squeeze(results.as_numpy(OUTPUT_LAYER_NAME))
        result["message"]= f"It's a {labels[np.argmax(output)]}, {output}"
    return result



