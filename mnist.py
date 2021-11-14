import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds

## データの作成
mnist_dataset , mnist_info = tfds.load(name='mnist' , with_info=True, as_supervised=True)

mnist_train , mnist_test= mnist_dataset['train'] , mnist_dataset['test']

num_validation_samples = 0.1 * mnist_info.splits['train'].num_examples
num_validation_samples = tf.cast(num_validation_samples , tf.int64)

num_test_samples = mnist_info.splits['train'].num_examples
num_test_samples = tf.cast(num_test_samples , tf.int64)

def scale(image,label): #imageの各変数は0から255までの値をとる
  image = tf.cast(image,tf.float32)
  image /= 255
  return image,label

scaled_train_and_validation_data = mnist_train.map(scale)#mnist_trainの全てのデータにscale関数を適用する
test_data = mnist_test.map(scale)

BUFFER_SIZE = 10000#メモリがオーバーしないように少しずつ処理を進めていくためのコード

shuffled_train_and_validation_data = scaled_train_and_validation_data.shuffle(BUFFER_SIZE)

validation_data = shuffled_train_and_validation_data.take(num_validation_samples)
train_data = shuffled_train_and_validation_data.skip(num_validation_samples)

BATCH_SIZE = 100

train_data = train_data.batch(BATCH_SIZE)
validation_data = validation_data.batch(num_validation_samples)
test_data = test_data.batch(num_test_samples)

validation_inputs, validation_targets = next(iter(validation_data))
#イテレータ:要素を一つづつ与えるリスト的なもの
#next():イテレータから要素を取り出す
#validation_dataはリストである

##モデルの作成
input_size = 784
output_size = 10
hidden_layer_size = 50

model = tf.keras.Sequential([
  tf.keras.layers.Flatten(input_shape=(28,28,1)),
  tf.keras.layers.Dense(hidden_layer_size , activation="relu"),
  tf.keras.layers.Dense(hidden_layer_size , activation="relu"),
  tf.keras.layers.Dense(output_size , activation="softmax")
])

## 最適化アルゴリズムと損失関数の決定
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

## 訓練
NUM_EPOCHS = 5
VALIDATION_STEPS = num_validation_samples

model.fit(train_data , epochs = NUM_EPOCHS , validation_data = (validation_inputs , validation_targets) , validation_steps= VALIDATION_STEPS , verbose=2)

## テスト
test_loss , test_accuracy = model.evaluate(test_data)