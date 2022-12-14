from flask import Flask, render_template, request
import numpy as np
import pickle
import joblib
import os

modeldir = os.path.join(os.path.dirname(__file__), 'model/classifier_model.pkl')

model = pickle.load(open('model/classifier_model.pkl','rb'))

app = Flask(__name__, template_folder='template')

@app.route('/')
def index():
    return(render_template('index.html'))

@app.route('/predict',methods=['POST'])
def predict():
    features = [float(x) for x in request.form.values()]
    final_features = [np.array(features)]
    result = model.predict(final_features)

    output = {0:'tipe nasabah yang memiliki balance moderat, sangat jarang melakukan transaksi pembelian, lebih sering melakukan transaksi dengan uang tunai dimuka, hampir tidak pernah melakukan pembelian dengan metode mencicil. tipe user ini memiliki limit kartu kredit medium', 
    1:'Memiliki balance dan limit kartu kredit paling tinggi, lebih sering melakukan pembelian dengan metode sekali bayar(one off purchases), sering melakukan transaksi belanja, hampir tidak pernah melakukan pembelian dengan uang tunai dimuka ',
    2:'Memiliki balance paling rendah diantara tipe nasabah lain, frekuensi pembelian cukup tinggi dan sering melakukan pembelian dengan metode pembayaran mencicil, memiliki limit kartu kredit paling rendah'}
    return render_template('index.html',result_text='{}'.format(output[result[0]]))

if __name__=='__main__':
    app.run(debug=True)