
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, classification_report
from transformers import AutoTokenizer, TFDistilBertForSequenceClassification

try:
    tpu = tf.distribute.cluster_resolver.TPUClusterResolver()
    print('Running on TPU:', tpu.master())
    tf.config.experimental_connect_to_cluster(tpu)
    tf.tpu.experimental.initialize_tpu_system(tpu)
    strategy = tf.distribute.TPUStrategy(tpu)
except ValueError:
    strategy = tf.distribute.get_strategy()
print("REPLICAS:", strategy.num_replicas_in_sync)

train = pd.read_csv('jigsaw-toxic-comment-train.csv')
validation = pd.read_csv('validation.csv')
test = pd.read_csv('test.csv')

train = train[['comment_text', 'toxic']]
validation = validation[['comment_text', 'toxic']]

train = train.iloc[:12000]

xtrain, xvalid, ytrain, yvalid = train_test_split(
    train.comment_text.values,
    train.toxic.values,
    stratify=train.toxic.values,
    test_size=0.2,
    random_state=42
)

tokenizer = AutoTokenizer.from_pretrained("distilbert-base-multilingual-cased")

def tokenize_texts(texts, max_length=256):
    return tokenizer(
        list(texts),
        padding='max_length',
        truncation=True,
        max_length=max_length,
        return_tensors='tf'
    )

max_length = 256
train_encodings = tokenize_texts(xtrain, max_length)
valid_encodings = tokenize_texts(xvalid, max_length)
val_labels = tf.convert_to_tensor(yvalid)

batch_size = 16 * strategy.num_replicas_in_sync

train_dataset = (
    tf.data.Dataset
    .from_tensor_slices((dict(train_encodings), ytrain))
    .shuffle(1024)
    .batch(batch_size)
    .prefetch(tf.data.AUTOTUNE)
)
valid_dataset = (
    tf.data.Dataset
    .from_tensor_slices((dict(valid_encodings), yvalid))
    .batch(batch_size)
    .prefetch(tf.data.AUTOTUNE)
)

with strategy.scope():
    model = TFDistilBertForSequenceClassification.from_pretrained(
        "distilbert-base-multilingual-cased",
        num_labels=2
    )
    class_weights = {0: 0.1, 1: 0.9}
    optimizer = tf.keras.optimizers.Adam(learning_rate=3e-5)
    loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    model.compile(optimizer=optimizer, loss=loss, metrics=['accuracy'])

history = model.fit(
    train_dataset,
    validation_data=valid_dataset,
    epochs=3,
    class_weight=class_weights,
    callbacks=[
        tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=2, restore_best_weights=True)
    ]
)

valid_logits = model.predict(valid_dataset).logits
valid_probs = tf.nn.softmax(valid_logits, axis=1)[:, 1].numpy()
roc_auc = roc_auc_score(yvalid, valid_probs)
print(f"Validation ROC-AUC: {roc_auc:.4f}")
print(classification_report(yvalid, valid_probs > 0.5, digits=4))

def predict_toxicity(text):
    inputs = tokenizer(text, return_tensors='tf', truncation=True, padding='max_length', max_length=max_length)
    logits = model(inputs).logits
    prob = tf.nn.softmax(logits, axis=1)[0, 1].numpy()
    return prob

example = "You are an idiot and you and your family should go back to your country"
print(f"Example toxicity score: {predict_toxicity(example):.4f}")

model.save_pretrained('distilbert-multilingual-toxic')
tokenizer.save_pretrained('distilbert-multilingual-toxic')
