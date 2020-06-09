'''
Python script to test the CNN-LSTM Audio Emotion
Detection Model.

Outputs the predicted emotion and the prediction
probability for "./Audios/test_audio.wav" file.
'''
from tensorflow.keras.models import load_model
import tensorflow as tf
import numpy as np
from scipy.stats import zscore
import librosa
import datetime

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

def predict_audio(chunk_step=16000, chunk_size=49100, predict_proba=True, sample_rate=16000):
    '''
    Method that loads a test audio file from the ./Audios directory
    and predicts emotion using the trained model.
    '''
    _emotion = {0:'Angry', 1:'Disgust', 2:'Fear', 3:'Happy', 4:'Neutral', 5:'Sad', 6:'Surprise'}
    label_dict_ravdess = {'02': 'NEU', '03':'HAP', '04':'SAD', '05':'ANG', '06':'FEA', '07':'DIS', '08':'SUR'}
    
    # Retrieve file from request
    filepath = "./Audios/test_audio.wav"
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
    emo_dict = {"emotion_audio": str(emotion), "prediction_probability": str(probability)}
    print(emo_dict)

if __name__ == "__main__":
    predict_audio()
