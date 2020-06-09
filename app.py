from __future__ import print_function
import repo
import elastic_service
from tagging import tag_message

from tensorflow.keras.models import load_model
from tensorflow.keras import backend
import tensorflow as tf
import numpy as np
from scipy.stats import zscore
import librosa

import os
import json
import datetime

from flask import Flask, jsonify, request, Response
from werkzeug.utils import secure_filename

app = Flask(__name__)
app.debug = True

# Load the Audio Emotion Detection Model
try:
    model = load_model('Models/[CNN-LSTM]Model.h5')
    model._make_predict_function()
except IOError:
    raise IOError("Could not find Voice Analysis model. Ensure model is present in: ./Models")

def mel_spectrogram(y, sr=16000, n_fft=512, win_length=256, hop_length=128, window='hamming', n_mels=128, fmax=4000):
    '''
    Mel-spectogram computation
    '''
    # Compute spectogram
    mel_spect = np.abs(librosa.stft(y, n_fft=n_fft, window=window, win_length=win_length, hop_length=hop_length)) ** 2

    # Compute mel spectrogram
    mel_spect = librosa.feature.melspectrogram(S=mel_spect, sr=sr, n_mels=n_mels, fmax=fmax)

    # Compute log-mel spectrogram
    mel_spect = librosa.power_to_db(mel_spect, ref=np.max)

    return np.asarray(mel_spect)

def frame(y, win_step=64, win_size=128):
    '''
    Audio framing
    '''
    # Number of frames
    nb_frames = 1 + int((y.shape[2] - win_size) / win_step)

    # Framming
    frames = np.zeros((y.shape[0], nb_frames, y.shape[1], win_size)).astype(np.float16)
    for t in range(nb_frames):
        frames[:,t,:,:] = np.copy(y[:,:,(t * win_step):(t * win_step + win_size)]).astype(np.float16)

    return frames

@app.route("/predict_recording", methods=["GET","POST"])
def predict_audio(chunk_step=16000, chunk_size=49100, predict_proba=True, sample_rate=16000):
    """
    This API is used to store audio recording and corresponding transcript,
    and the detected emotion and symptoms from the recording and transcript
    respectively.
    ---
    tags:
      - Detect Emotion & Symptoms API
    parameters:
      - name: file
        in: body
        type: MultiPart File
        required: true
        description: The audio recording submitted by user
      - name: text_message
        in: query
        type: string
        required: true
        description: the transcript for the audio recording
    responses:
      500:
        {
            "data": None,
            "error": error description
        }
      200:
        description: Detected emotion for the audio recording
        {
            "data": {
                "emotion_audio": detected emotion,
                "emotion_audio_score": prediction probability/confidence,
                "filepath": filename with which the recording is stored locally,
                "timestamp": timestamp
            },
            "error": None
        }

    """
    # Get the text message and tag symptoms, drugs and tests
    text = request.args.get("text_message")
    print(text)
    res = tag_message(text_msg=text)
    symptoms = []
    for entity in res['entities']:
        if entity['type'] == "problem":
            symptoms.append(entity['text'])
    symptoms = ','.join(symptoms)
    print("Detected Symptoms: ")
    print(symptoms)

    # Emotion encoding
    _emotion = {0:'Angry', 1:'Disgust', 2:'Fear', 3:'Happy', 4:'Neutral', 5:'Sad', 6:'Surprise'}
    label_dict_ravdess = {'02': 'NEU', '03':'HAP', '04':'SAD', '05':'ANG', '06':'FEA', '07':'DIS', '08':'SUR'}
    
    # Retrieve file from request
    f = request.files['file']
    filepath = "./Audios/"+secure_filename(f.filename)
    f.save(filepath)
    max_pad_len = 49100
    
    # Read audio file
    y, sr = librosa.core.load(filepath, sr=sample_rate, offset=0.5)

    # Z-normalization
    y = zscore(y)
    
    # Padding or truncated signal 
    if len(y) < max_pad_len:    
        y_padded = np.zeros(max_pad_len)
        y_padded[:len(y)] = y
        y = y_padded
    elif len(y) > max_pad_len:
        y = np.asarray(y[:max_pad_len])
        
    # Split audio signals into chunks
    chunks = frame(y.reshape(1, 1, -1), chunk_step, chunk_size)

    # Reshape chunks
    chunks = chunks.reshape(chunks.shape[1],chunks.shape[-1])

    # Z-normalization
    y = np.asarray(list(map(zscore, chunks)))

    # Compute mel spectrogram
    mel_spect = np.asarray(list(map(mel_spectrogram, y)))
    
    # Time distributed Framing
    mel_spect_ts = frame(mel_spect)

    # Build X for time distributed CNN
    X = mel_spect_ts.reshape(mel_spect_ts.shape[0],
                                mel_spect_ts.shape[1],
                                mel_spect_ts.shape[2],
                                mel_spect_ts.shape[3],
                                1)

    # Predict emotion
    if predict_proba is True:
        predict = model.predict(X)
    else:
        predict = np.argmax(model.predict(X), axis=1)
        predict = [_emotion.get(emotion) for emotion in predict]

    # Predict timestamp
    timestamp = np.concatenate([[chunk_size], np.ones((len(predict) - 1)) * chunk_step]).cumsum()
    timestamp = np.round(timestamp / sample_rate)
    result = [predict,timestamp]

    result_np = np.array(result[0][0])
    probability = result_np.max()
    emotion = _emotion.get(result_np.argmax())

    timestamp = datetime.datetime.now().strftime("%d/%m/%Y %H:%M:%S")
    emo_dict = {"emotion_audio": str(emotion), "emotion_audio_score": str(probability), "filepath":str(secure_filename(f.filename)), "timestamp": timestamp}
    print(emo_dict)
    response = create_json(emo_dict, None)

    # Save filepath along with emotion in the database
    tyler_repo = repo.TylerRepo()
    emo_dict['username'] = "Rahul"
    tyler_repo.insert_record(collection_name="user_history", 
        query={"timestamp":timestamp, "username":"Rahul", "filepath":secure_filename(f.filename)}, 
        insert_doc={"$set":emo_dict})
    
    es = elastic_service.ElasticService()
    es.insert_record({
        "username":"Rahul",
        "filepath": emo_dict['filepath'],
        "message": text,
        "symptoms": symptoms,
        "timestamp": timestamp
    })
    
    return response

@app.route("/save_audio", methods=["POST"])
def save_audio_with_emotion():
    """
    This API is used to update the stored emotion for
    the latest audio recording submitted by the user, if
    changes are necessary i.e. the user's mood differs
    from what the model has predicted.
    ---
    tags:
      - Update Emotion API
    parameters:
      - name: username
        in: query
        type: string
        required: true
        description: The username for identifying each user
      - name: filepath
        in: query
        type: string
        required: true
        description: The name of the file with which the audio
                     recording is stored.
      - name: emotion_audio
        in: query
        type: string
        required: true
        description: The actual emotion of the user right now, when
                     submitting the clip
    responses:
      500:
        {
            "data": None,
            "error": error description
        }
      200:
        description: Detected emotion for the audio recording
        {
            "data": "Inserted successfully",
            "error": None
        }

    """
    # Get the attributes from query parameters
    form={}
    form['username']=request.args.get("username")
    form['filepath']=request.args.get("filepath")
    if ".wav" not in str(form['filepath']):
        form['filepath'] = form['filepath'] + ".wav"
    form['emotion_audio']=request.args.get("emotion_audio")
    print(form)

    # Update previously stored record in the database
    try:
        tyler_repo = repo.TylerRepo()
        tyler_repo.insert_record(collection_name="user_history", 
            query={"username":form['username'], "filepath":form['filepath']}, 
            insert_doc={"$set":form})
        return create_json({"data":"Inserted successfully."}, None)
    except Exception as e:
        print(e)
        return create_json(None, {"error": "Could not save the data. Please try again later."})

# send CORS headers
@app.after_request
def after_request(response):
    response.headers.add('Access-Control-Allow-Origin', '*')
    if request.method == 'OPTIONS':
        response.headers['Access-Control-Allow-Methods'] = 'DELETE, GET, POST, PUT'
        headers = request.headers.get('Access-Control-Request-Headers')
        if headers:
            response.headers['Access-Control-Allow-Headers'] = headers
    return response

def create_json(data, error):
    """
    Utility method that creates a json response from the data returned by the service method.
    :param data:
    :return:
    """
    base_response_dto = {}
    if error is None:
        base_response_dto = {'data': data, "error": None}
    else:
        base_response_dto = {"data": None, "error": error}
    js = json.dumps(base_response_dto)
    resp = Response(js, status=200, mimetype='application/json')
    return resp

if __name__ == '__main__':
    app.run(debug=True, use_reloader=False, port=5000, threaded=False)
