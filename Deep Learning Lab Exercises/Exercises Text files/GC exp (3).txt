import tensorflow as tf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
titanic_file_path = tf.keras.utils.get_file("train.csv", 
                                            "https://storage.googleapis.com/tf-datasets/titanic/train.csv")
df = pd.read_csv(titanic_file_path)
df.head()
df.rename(columns ={"survived":"target"},inplace=True)
np.random.seed(5)
train, val, test = np.split(df.sample(frac=1),[int(0.8*len(df)),
int(0.9*len(df))])
train
def df_to_dataset(dataframe, shuffle=True, batch_size=32):
    df = dataframe.copy()
    labels = df.pop('target')
    df = {key: value.values[:, tf.newaxis] for key, value in dataframe.items()}
    ds = tf.data.Dataset.from_tensor_slices((dict(df), labels))
    
    if shuffle:
        ds = ds.shuffle(buffer_size=len(dataframe))
    
    ds = ds.batch(batch_size)
    ds = ds.prefetch(batch_size)
    return ds

batch_size = 10
train_ds = df_to_dataset(train, batch_size=batch_size)
val_ds = df_to_dataset(val, batch_size=batch_size)
test_ds = df_to_dataset(test, batch_size=batch_size)
def get_normalization_layer(name, dataset):
   
    normalizer = tf.keras.layers.Normalization(axis=None)
    
   
    feature_ds = dataset.map(lambda x, y: x[name])
    
    
    normalizer.adapt(feature_ds)
    
    return normalizer


def get_category_encoding_layer(name, dataset, dtype, max_tokens=None):
    
    if dtype == 'string':
        index = tf.keras.layers.StringLookup(max_tokens=max_tokens)
    
    else:
        index = tf.keras.layers.IntegerLookup(max_tokens=max_tokens)

    feature_ds = dataset.map(lambda x, y: x[name])
    
 
    index.adapt(feature_ds)
    
    
    encoder = tf.keras.layers.CategoryEncoding(num_tokens=index.vocabulary_size())
    
   
    return lambda feature: encoder(index(feature))


numerical_cols = ["age", "fare"]
numerical_categorical_cols = ["n_siblings_spouses", "parch"]
categorical_cols = ["sex", "class", "deck", "embark_town", "alone"]

all_inputs = []
encoded_features = []


for header in numerical_cols:
    numeric_col = tf.keras.Input(shape=(1,), name=header)
    normalization_layer = get_normalization_layer(header, train_ds)  # Normalization
    encoded_numeric_col = normalization_layer(numeric_col)
    all_inputs.append(numeric_col)
    encoded_features.append(encoded_numeric_col)


for header in numerical_categorical_cols:
    categorical_col = tf.keras.Input(shape=(1,), name=header, dtype='int64')
    encoding_layer = get_category_encoding_layer(name=header, dataset=train_ds, dtype='int64')  # Encoding
    encoded_categorical_col = encoding_layer(categorical_col)
    all_inputs.append(categorical_col)
    encoded_features.append(encoded_categorical_col)


for header in categorical_cols:
    categorical_col = tf.keras.Input(shape=(1,), name=header, dtype='string')
    encoding_layer = get_category_encoding_layer(name=header, dataset=train_ds, dtype='string', max_tokens=5)  # Encoding
    encoded_categorical_col = encoding_layer(categorical_col)
    all_inputs.append(categorical_col)
    encoded_features.append(encoded_categorical_col)

x = tf.keras.layers.concatenate(encoded_features)
x = tf.keras.layers.Dense(32, activation="relu")(x)
x = tf.keras.layers.Dense(8, activation="relu")(x)
x = tf.keras.layers.Dense(4, activation="relu")(x)
x = tf.keras.layers.Dense(2, activation="relu")(x)
outputs = tf.keras.layers.Dense(1, activation="sigmoid")(x)

model = tf.keras.Model(all_inputs, outputs)

model.compile(
    optimizer='adam',
    loss=tf.keras.losses.BinaryCrossentropy(from_logits=False),
    metrics=["accuracy"]
)

tf.keras.utils.plot_model(model)

history =model.fit(train_ds,validation_data=val_ds,epochs=50)
history = history.history

plt.figure(figsize=(15, 5))


plt.subplot(121)
plt.title("Accuracy")
plt.plot(history["accuracy"], label="train acc")
plt.plot(history["val_accuracy"], label="val acc")
plt.legend()


plt.subplot(122)
plt.title("Loss")
plt.plot(history["loss"], label="train loss")
plt.plot(history["val_loss"], label="val loss")
plt.legend()

plt.show()

loss ,accuracy =model.evaluate(test_ds)
print("test loss :",loss)
print("test accuracy:",accuracy)
print("Original")
display(df.head())

print("Predicted")
inference = df.head().drop("target", axis=1)

# Select only the relevant columns for inference
inference = inference[numerical_cols + numerical_categorical_cols + categorical_cols]

# Convert dataframe into dictionary format for TensorFlow model
inference_dict = {col: inference[col].values for col in inference.columns}

# Predict using the trained model
inference["target"] = model.predict(inference_dict)

# Display results
display(inference)
