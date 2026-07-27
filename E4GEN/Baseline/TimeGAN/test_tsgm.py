import os
os.environ["KERAS_BACKEND"] = "torch"

import keras
import tsgm

print("backend =", keras.backend.backend())
print("tsgm imported successfully")