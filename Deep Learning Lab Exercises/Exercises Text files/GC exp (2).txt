import tensorflow as tf
import pathlib
import os
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
t1 = tf.constant([
    [1, 2, 3],
    [4, 5, 6],
    [7, 8, 9],
], dtype=tf.float32)

ds1 = tf.data.Dataset.from_tensors(t1)  
ds2 = tf.data.Dataset.from_tensor_slices(t1) 


for element in ds1:
    print(element)

for element in ds2:
    print(element)
train, test = tf.keras.datasets.fashion_mnist.load_data()
images, labels = train
images = images/255
type(images),type(labels)

dataset = tf.data.Dataset.from_tensor_slices((images,labels))
dataset
directory_url = 'https://storage.googleapis.com/download.tensorflow.org/data/illiad/'

file_names = ['cowper.txt', 'derby.txt', 'butler.txt']


file_paths = [
    tf.keras.utils.get_file(file_name, directory_url + file_name)  
    for file_name in file_names
]

text_line_dataset = tf.data.TextLineDataset(file_paths)
for line in text_line_dataset.take(5):
    print(line.numpy())
data_url = 'https://storage.googleapis.com/download.tensorflow.org/data/stack_overflow_16k.tar.gz'

dataset_dir = tf.keras.utils.get_file(
    "stack_overflow_16k", 
    origin=data_url,
    untar=True
)

# Now point to extracted path
dataset_dir = pathlib.Path(dataset_dir)
train_dir = dataset_dir / 'train'

print(train_dir)  

# Now load the dataset
raw_train_ds = tf.keras.utils.text_dataset_from_directory(
    train_dir,
    batch_size=32,
    validation_split=0.2,
    subset='training',
    seed=42
)

titanic_file = tf.keras.utils.get_file("train.csv", "https://storage.googleapis.com/tf-datasets/titanic/train.csv")


df = pd.read_csv(titanic_file)


titanic_dataset = tf.data.Dataset.from_tensor_slices(dict(df))


for feature_batch in titanic_dataset.take(1):
    for key, value in feature_batch.items():
        print("{!r:20s}:{}".format(key, value))

titanic_batches = tf.data.experimental.make_csv_dataset(
    titanic_file,
    batch_size=4, 
    label_name="survived",  
    select_columns=['class', 'fare', 'survived']
)

for feature_batch, label_batch in titanic_batches.take(1):
    print(f"Survived: {label_batch}")
    for key, value in feature_batch.items():
        print(f"{key:20s}: {value}")

titanic_types = [tf.int32, tf.string, tf.float32,tf.int32,tf.int32,
tf.float32,tf.string,tf.string, tf.string, tf.string]
dataset = tf.data.experimental.CsvDataset(titanic_file, titanic_types ,
header=True)
for line in dataset.take(10):
     print([item.numpy()for item in line])

flowers_root = tf.keras.utils.get_file(
    'flower_photos',
    'https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz',
    untar=True
)


flowers_root = pathlib.Path(flowers_root)

for item in flowers_root.glob("*"):
    print(item)
file_path_ds = tf.data.Dataset.list_files(str(flowers_root / '*/*'))


def process_path(file_path):
    
    label = tf.strings.split(file_path, os.sep)[-2]
    return tf.io.read_file(file_path), label


labeled_ds = file_path_ds.map(process_path)


for image_raw, label in labeled_ds.take(1):
    print(image_raw, label, sep="\n")

inc_dataset = tf.data.Dataset.range(100)


dec_dataset = tf.data.Dataset.range(0, -100, -1)


dataset = tf.data.Dataset.zip((inc_dataset, dec_dataset))


batched_dataset = dataset.batch(4)  

for batch in batched_dataset.take(4):
    print([arr.numpy() for arr in batch])  

batched_dataset
dataset.batch(4,drop_remainder = True)


dataset = tf.data.Dataset.range(100)


dataset = dataset.map(lambda x: tf.fill([tf.cast(x, tf.int32)], x))

padded_batch_dataset = dataset.padded_batch(4, padded_shapes=(None,))


for batch in padded_batch_dataset.take(2):
    print(batch.numpy())
    print()

dataset = tf.data.TextLineDataset(titanic_file)
dataset.shuffle(buffer_size=10)

file_path_ds = tf.data.Dataset.list_files(str(flowers_root / '*/*'))


def parse_image(filename):
    label = tf.strings.split(filename, os.sep)[-2]  
    image = tf.io.read_file(filename)  
    image = tf.io.decode_jpeg(image)  
    image = tf.image.convert_image_dtype(image, tf.float32) 
    image = tf.image.resize(image, [128, 128]) 
    return image, label


image_ds = file_path_ds.map(parse_image)


def show(image, label):
    plt.imshow(image)  
    plt.title(label.numpy().decode("utf-8"))  
    plt.axis("off")  
    plt.show()


for image, label in image_ds.take(3):
    show(image, label)
(train, test) = tf.keras.datasets.fashion_mnist.load_data()


images, labels = train
images = images / 255.0  
labels = labels.astype(np.int32)  


fmnist_train_ds = tf.data.Dataset.from_tensor_slices((images, labels))
fmnist_train_ds = fmnist_train_ds.shuffle(5000).batch(32)  
model = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),  
    tf.keras.layers.Dense(10)  
])


model.compile(
    optimizer='adam',
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=['accuracy']
)


model.fit(fmnist_train_ds, epochs=2)
loss,accuracy = model.evaluate(fmnist_train_ds)
print("Loss:",loss)
print("Accuracy:",accuracy)
predict_ds = tf.data.Dataset.from_tensor_slices(images).batch(32)

result = model.predict(predict_ds,steps = 10)
print(result.shape)

