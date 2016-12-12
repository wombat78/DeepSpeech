import numpy as np
import scipy.io.wavfile as wav

from python_speech_features import mfcc

def audiofile_to_input_vector(audio_filename, numcep):
    # Load wav files
    fs, audio = wav.read(audio_filename)

    # Get mfcc coefficients
    inputs = mfcc(audio, samplerate=fs, numcep=numcep)

    # Whiten inputs (TODO: Should we whiten)
    inputs = (inputs - np.mean(inputs))/np.std(inputs)

    # Return results
    return inputs
