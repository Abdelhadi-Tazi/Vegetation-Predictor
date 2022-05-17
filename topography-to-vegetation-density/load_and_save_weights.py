import tensorflow as tf
model = tf.keras.models.load_model("models/model6")
model.save("models/h5/model6.h5")
