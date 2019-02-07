from __future__ import absolute_import, division, print_function


import tensorflow as tf
tf.enable_eager_execution()

import numpy as np
import os
import time

#import argparse


#of the variables, I think only seq_length might be interesting to fiddle with
def processInput(path_to_file, seq_length=100, BATCH_SIZE=64, BUFFER_SIZE=1000):
    # Read in the text
    text = open(path_to_file, 'rb').read().decode(encoding='utf-8')

    # Get the unique characters in the file
    vocab = sorted(set(text))

    # Create a mapping from unique characters to indices
    char2idx = {u:i for i, u in enumerate(vocab)} # makes a dictionary mapping from character to integer
    idx2char = np.array(vocab) # makes the reverse, mapping from integer to associated character
    text_as_int = np.array([char2idx[c] for c in text]) # encodes the sample text according to the above mapping


    # Create training examples / targets
    char_dataset = tf.data.Dataset.from_tensor_slices(text_as_int) # seems to convert the numpy array into a "stream"
    sequences = char_dataset.batch(seq_length+1, drop_remainder=True) # creates a series of strings(?) of length seq_length+1

    def split_input_target(chunk):
        input_text = chunk[:-1]
        target_text = chunk[1:]
        return input_text, target_text

    dataset = sequences.map(split_input_target) # turns the strings of length seq_length+1 into
                                                # the input and target strings of the desired length

    # Creating Training Batches:
    examples_per_epoch = len(text)//seq_length
    steps_per_epoch = examples_per_epoch//BATCH_SIZE

    # Buffer size to shuffle the dataset
    # (TF data is designed to work with possibly infinite sequences,
    # so it doesn't attempt to shuffle the entire sequence in memory. Instead,
    # it maintains a buffer in which it shuffles elements).

    dataset = dataset.shuffle(BUFFER_SIZE).batch(BATCH_SIZE, drop_remainder=True)

    data = {"dataset" : dataset,
            "vocab" : vocab,
            "char2idx" : char2idx,
            "idx2char": idx2char,
            "steps_per_epoch": steps_per_epoch,
            "BATCH_SIZE": BATCH_SIZE}

    return data

def buildModel(data, embedding_dim=256, rnn_units=1024):
    vocab = data["vocab"]
    BATCH_SIZE = data["BATCH_SIZE"]

    #Build the Model:
    # Length of the vocabulary in chars
    vocab_size = len(vocab)

    if tf.test.is_gpu_available():
        rnn = tf.keras.layers.CuDNNGRU
    else:
        import functools
        rnn = functools.partial(
                tf.keras.layers.GRU, recurrent_activation='sigmoid')

        def build_model(vocab_size, embedding_dim, rnn_units, batch_size):
            model = tf.keras.Sequential([
                tf.keras.layers.Embedding(vocab_size, embedding_dim,
                                          batch_input_shape=[batch_size, None]),
                rnn(rnn_units,
                    return_sequences=True,
                    recurrent_initializer='glorot_uniform',
                    stateful=True),

                tf.keras.layers.Dense(vocab_size)
            ])
            return model

        model = build_model(
                  vocab_size = len(vocab),
                  embedding_dim=embedding_dim,
                  rnn_units=rnn_units,
                  batch_size=BATCH_SIZE)

    model.summary()

    return model

def trainModel(model, data, EPOCHS=7, checkpoint_dir='./training_checkpoints'):
    dataset = data["dataset"]
    steps_per_epoch = data["steps_per_epoch"]

    # define our loss function:
    def loss(labels, logits):
        return tf.keras.losses.sparse_categorical_crossentropy(labels, logits, from_logits=True)

    #configures the training procedure
    model.compile(
        optimizer = tf.train.AdamOptimizer(),
        loss = loss)

    # Name of the checkpoint files
    checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt_{epoch}")

    checkpoint_callback=tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_prefix,
        save_weights_only=True)

    history = model.fit(dataset.repeat(), epochs=EPOCHS, steps_per_epoch=steps_per_epoch, callbacks=[checkpoint_callback])


    # want just one output, so we need to rebuild the model a little from the last checkpoint
    # so that it will run on a batch_size of 1
    tf.train.latest_checkpoint(checkpoint_dir)
    model = build_model(vocab_size, embedding_dim, rnn_units, batch_size=1)
    model.load_weights(tf.train.latest_checkpoint(checkpoint_dir))
    model.build(tf.TensorShape([1, None]))


    model.summary()
    return model


# Evaluation step (generating text using the learned model)
# num_generate = Number of characters to generate
# Low temperatures results in more predictable text.
# Higher temperatures results in more surprising text.
# Experiment to find the best setting.
def generateText(model, data, start_string, num_generate=1000, temperature=1.0):
  char2idx= data["char2idx"]
  idx2char= data["idx2char"]

  # Converting our start string to numbers (vectorizing)
  input_eval = [char2idx[s] for s in start_string]
  input_eval = tf.expand_dims(input_eval, 0)

  # Empty string to store our results
  text_generated = []

  # Here batch size == 1
  model.reset_states()
  for i in range(num_generate):
      predictions = model(input_eval)
      # remove the batch dimension
      predictions = tf.squeeze(predictions, 0)

      # using a multinomial distribution to predict the word returned by the model
      predictions = predictions / temperature
      predicted_id = tf.multinomial(predictions, num_samples=1)[-1,0].numpy()

      # We pass the predicted word as the next input to the model
      # along with the previous hidden state
      input_eval = tf.expand_dims([predicted_id], 0)

      text_generated.append(idx2char[predicted_id])

  return (start_string + ''.join(text_generated))

if __name__== "__main__":

    inputFile = "Hamilton2.txt"
    start_string = u"[HAMILTON] "

    data = processInput(inputFile)
    model = buildModel(data["vocab"])
    model = trainModel(model, data)

    print(generateText(model, data, start_string))
