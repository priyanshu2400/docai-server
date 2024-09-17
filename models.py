import tensorflow as tf

# Dictionary to hold the model instances
models = {}

def load_model(disease_name: str, model_path: str):
    models[disease_name] = tf.keras.models.load_model(model_path)

def get_model(disease_name: str):
    return models.get(disease_name)
