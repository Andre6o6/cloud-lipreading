import flask
from predict import Predictor

app = flask.Flask(__name__)

WEIGHTS_PATH="weights/weights.hdf5"
DICT_PATH="dictionaries/english.txt"
model = Predictor(WEIGHTS_PATH, DICT_PATH)


@app.route('/', methods=['GET', 'POST'])
def main():
    if request.method == 'POST':
        pass
    return render_template('form.html')
    

@app.route("/predict", methods=["POST"])
def predict():
    data = {}
    try:
        #data = flask.request.get_json()['data']
        src = flask.request.get_json()['src']
    except Exception:
        return flask.jsonify(status_code='400', msg='Bad Request'), 400
    #data = base64.b64decode(data)
    #image = io.BytesIO(data)
    
    video_data = skvideo.io.vread(src)
    predictions = model.predict_subs(video_data)
    flask.current_app.logger.info('Predictions: %s', predictions)
    return flask.jsonify(predictions=predictions)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080, debug=True)