import tensorflow as tf
import numpy as np
from flask import Flask, render_template, request, jsonify
from flask_ngrok import run_with_ngrok
import cv2
import os
from fungsi import make_model

# =[Variabel Global]=============================
app = Flask(__name__, static_url_path='/static')
model = None

# =[Routing]=====================================

# [Routing untuk Halaman Utama atau Home]
@app.route("/")
def beranda():
    return render_template('index.html')

@app.route("/upload")
def upload_page():
    return render_template('upload.html')

@app.route("/inner-page")
def inner_page():
    return render_template('inner-page.html')

# [Routing untuk API]
@app.route("/api/deteksi", methods=['POST'])
def apiDeteksi():
    # Load model yang telah ditraining
    global model
    if model is None:
        model = make_model()
        model.load_weights('model_grape_tf.h5')

    if request.method == 'POST':
        # Menerima file gambar yang dikirim dari frontend
        file = request.files['file']

        # Simpan file gambar ke direktori temporary
        file_path = 'static/temp/temp.txt'
        file.save(file_path)

        # Membaca dan memproses gambar dengan OpenCV
        image = cv2.imread(file_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (224, 224))
        image = image / 255.0
        image = np.expand_dims(image, axis=0)

        # Melakukan prediksi menggunakan model
        prediksi = model.predict(image)
        class_index = np.argmax(prediksi)

        # Kamus kelas dan label yang lebih deskriptif
        class_mapping = {
            0: 'Black Rot',
            1: 'ESCA',
            2: 'Healty', 
            3: 'Leaf Blight'
        }

        # Mengambil label prediksi dari kamus
        hasil_prediksi = class_mapping[class_index]

        # Mengubah label prediksi menjadi teks yang lebih deskriptif
        if hasil_prediksi == 'class_0':
            hasil_prediksi = 'Black Rot'
            keterangan_penyakit = ' '
        elif hasil_prediksi == 'class_1':
            hasil_prediksi = 'ESCA'
            keterangan_penyakit = ' '
        elif hasil_prediksi == 'class_2':
            hasil_prediksi = 'Healty'
            keterangan_penyakit = ' '
        else:
            hasil_prediksi = 'Leaf Blight'
            keterangan_penyakit = ' '

        # Menghapus file gambar temporary
        os.remove(file_path)

        # Return hasil prediksi dengan format JSON
        return jsonify({
            "prediksi": hasil_prediksi
        })

# =[Main]========================================

if __name__ == '__main__':
    # Run Flask di localhost
    run_with_ngrok(app)
    app.run()
