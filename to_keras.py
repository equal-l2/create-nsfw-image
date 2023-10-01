import tensorflow as tf

model = tf.keras.models.load_model("./model/mobilenet_v2_140_224")
model.save("model.keras")
