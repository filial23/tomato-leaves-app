import os
import numpy as np
import tensorflow as tf
from flask import Flask, request, redirect, render_template, flash
from werkzeug.utils import secure_filename
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.preprocessing import image

@tf.keras.utils.register_keras_serializable()
class CentralCropLayer(tf.keras.layers.Layer):
    def __init__(self, crop_size):
        super(CentralCropLayer, self).__init__()
        self.crop_size = crop_size

    def call(self, inputs):
        cropped_images = tf.image.central_crop(inputs, central_fraction=self.crop_size)
        return cropped_images

classes = ["0","1","2","3","4","5","6","7","8","9"]

classes = ['細菌性斑点 (Bacterial_spot)',
           '早期病害 (Early_blight)',
           '晩枯病 (Late_blight)',
           '葉カビ (Leaf_Mold)',
           'セプトリア葉斑病 (Septoria_leaf_spot)',
           'ハダニ類 二斑点性ハダニ (Spider_mites Two-spotted_spider_mites)',
           'ターゲットスポット (Target_Spot)',
           'トマト黄化葉巻病 (Tomoto_Yellow_Leaf_Curl_virus)',
           'トマトモザイクウイルス (Tomoto_mosaic_virus)',
           '健康 (healthy)',
           'うどんこ病 (powdery_mildew)']

image_size = 128

UPLOAD_FOLDER = "uploads"
ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg', 'gif'])

app = Flask(__name__)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

#学習済みモデルをロード
model = load_model('./model_1.0.0.keras', compile=False)

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        if 'file' not in request.files:
            flash('ファイルがありません')
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            flash('ファイルがありません')
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file.save(os.path.join(UPLOAD_FOLDER, filename))
            filepath = os.path.join(UPLOAD_FOLDER, filename)

            #受け取った画像を読み込み、np形式に変換
            img = image.load_img(filepath, target_size=(image_size, image_size, 3))
            img = image.img_to_array(img)
            data = np.array([img])
            #変換したデータをモデルに渡して予測する
            result = model.predict(data)[0]
            predicted = result.argmax()
            pred_answer = f"この葉は {result[predicted]*100:.2f} %で {classes[predicted]} です"

            return render_template("index.html",answer=pred_answer)

    return render_template("index.html",answer="")


if __name__ == "__main__":
    port = int(os.environ.get('PORT', 8080))
    app.run('0.0.0.0', port=port)