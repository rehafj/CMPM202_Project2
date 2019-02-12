from __future__ import absolute_import, division, print_function


import tensorflow as tf
tf.enable_eager_execution()

import numpy as np
import os
import time

import argparse

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

def build_model(vocab_size, embedding_dim, rnn_units, batch_size):
    if tf.test.is_gpu_available():
        rnn = tf.keras.layers.CuDNNGRU
    else:
        import functools
        rnn = functools.partial(
                tf.keras.layers.GRU, recurrent_activation='sigmoid')

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

def buildModel(data, embedding_dim=256, rnn_units=1024, BATCH_SIZE = None):
    vocab = data["vocab"]
    if BATCH_SIZE == None:
        BATCH_SIZE = data["BATCH_SIZE"]

    #Build the Model:
    # Length of the vocabulary in chars
    vocab_size = len(vocab)

    model = build_model(
              vocab_size = len(vocab), 
              embedding_dim=embedding_dim, 
              rnn_units=rnn_units, 
              batch_size=BATCH_SIZE)

    model.summary()

    return model

# define our loss function:
def loss(labels, logits):
    return tf.keras.losses.sparse_categorical_crossentropy(labels, logits, from_logits=True)

def trainModel(model, data, EPOCHS=3, checkpoint_dir='./training_checkpoints', embedding_dim=256,  rnn_units=1024):
    dataset = data["dataset"]
    steps_per_epoch = data["steps_per_epoch"]
    vocab_size = len(data["vocab"])

    
    print("steps_per_epoch: %s"%steps_per_epoch)
    print("vocab_size: %s"%vocab_size)
    
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

def trainAndGenerate(inputFile, model_path, start_string,
                seq_length=100, BATCH_SIZE=64, BUFFER_SIZE=1000,
                embedding_dim=256, rnn_units=1024,
                epochs=3, checkpoint_dir='./training_checkpoints',
                num_generate=1000, temperature=1.0 ):


    if not inputFile:
        inputFile = tf.keras.utils.get_file('shakespeare.txt', 'https://storage.googleapis.com/download.tensorflow.org/data/shakespeare.txt')

    data = processInput(inputFile, seq_length=seq_length, BATCH_SIZE=BATCH_SIZE, BUFFER_SIZE=BUFFER_SIZE)

    
    if model_path:
        model = buildModel(data, embedding_dim=embedding_dim, rnn_units=rnn_units, BATCH_SIZE=1)
        model.load_weights(tf.train.latest_checkpoint(model_path))
    else:
        model = buildModel(data, embedding_dim=embedding_dim, rnn_units=rnn_units)
        model = trainModel(model, data, EPOCHS=epochs, checkpoint_dir=checkpoint_dir, embedding_dim=embedding_dim, rnn_units=rnn_units)

        # if pickle_model:
        #     checkpoint = tf.train.Checkpoint(optimizer=optimizer, model=model)
        #     checkpoint.save(pickle_model)

    if start_string:
        return generateText(model, data, start_string, num_generate=num_generate, temperature=temperature)


if __name__== "__main__":
    parser = argparse.ArgumentParser(description='Builds and Trains an RNN built with TensorFlow, as well as generates new text based on the model.')

    parser.add_argument("--input_text_path", "-t", help="Location of the raw text.")
    parser.add_argument("--start_string", "-s", help="String to start generating on.")

    parser.add_argument("--pretrained_model_path", "-m", help="Location of pretrained_model.")
    parser.add_argument("--save_trained_model", default='./training_checkpoints', help="Location to save trained model.")

    # preprocessing arguments:
    parser.add_argument("--sequence_length", default=100, type=int, help="Length of the context used to train the model.")
    parser.add_argument("--batch_size", default=64, type=int)
    parser.add_argument("--buffer_size", default=1000, type=int)


    # model arguements:
    parser.add_argument("--epochs", "-e", default=3, type=int, help="Number of iterations to train over")
    #parser.add_argument("--checkpoint_dir", default='./training_checkpoints')
    parser.add_argument("--embedding_dim", default=256, type=int)
    parser.add_argument("--rnn_units", default=1024, type=int)

    # generation arguements:
    parser.add_argument("--output_length", "-o", default=1000, type=int)
    parser.add_argument("--temperature", default=1.0, type=float, help="Influences how 'predictable' the text output will be. Low temperatures results in more predictable text. Higher temperatures results in more surprising text. Experiment to find the best setting.")



    args = parser.parse_args()
    inputFile = args.input_text_path
    start_string = args.start_string
    
    model_path = args.pretrained_model_path

    seq_length = args.sequence_length
    batch_size = args.batch_size
    buffer_size = args.buffer_size

    embedding_dim = args.embedding_dim
    rnn_units = args.rnn_units
    epochs = args.epochs
    checkpoint_dir = args.save_trained_model

    output_length = args.output_length
    temp = args.temperature


    print(trainAndGenerate(inputFile, model_path, start_string,
                                seq_length=seq_length, BATCH_SIZE=batch_size, BUFFER_SIZE=buffer_size,
                                embedding_dim=embedding_dim, rnn_units=rnn_units,
                                epochs=epochs, checkpoint_dir=checkpoint_dir,
                                num_generate=output_length, temperature=temp ))

    