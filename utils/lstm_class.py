import numpy as np
from numpy import random
import nltk
#nltk.download_shell()
from nltk.tokenize import word_tokenize
import string
from unidecode import unidecode
#from lib.ops import *

import tensorflow as tf
from tensorflow.contrib import rnn

from tf_functions import *
from data_reader import *


class LSTM(object):
    """ Character-Level LSTM Implementation """

    def __init__(self):
        # Get the hyperparameters
        self.hparams = self.get_hparams()
        
        self.read_data = read_data()
        
        ALPHABET_SIZE = self.read_data.ALPHABET_SIZE
     
        # maximum length of each words
        max_word_length = self.hparams['max_word_length']
        
        # X is of shape ('b', 'sentence_length', 'max_word_length', 'alphabet_size']
        # Placeholder for the one-hot encoded sentences
        self.X = tf.placeholder('float32', 
                                shape=[None, None, max_word_length, ALPHABET_SIZE],
                                name='X')
        
        # Placeholder for the one-hot encoded sentiment
        self.Y = tf.placeholder('float32', shape=[None, 3], name='Y')
        
    def get_hparams(self):
        ''' Get Hyperparameters '''

        return {
            'BATCH_SIZE':       64,
            'EPOCHS':           500,
            'max_word_length':  16,
            'learning_rate':    0.0001,
            'patience':         10000,
        }



    def build(self,
              training=False,
              testing_batch_size=1000,
              kernels=[1, 2, 3, 4, 5, 6, 7],
              kernel_features=[25, 50, 75, 100, 125, 150, 175],
              rnn_size=650,
              dropout=0.0,
              size=700):
        
        """
        Build the computational graph
        
        :param training: 
            Boolean whether we are training (True) or testing (False)
        
        :param testing_batch_size: 
            Batch size to use during testing 
            
        :param kernels:
            Kernel width for each convolutional layer
         
        :param kernel_features: 
            Number of kernels for each convolutional layer
            
        :param rnn_size: 
            Size of the LSTM output
        
        :param dropout: 
            Retain probability when using dropout
        
        :param size: 
            Size of the Highway embeding
        """
        self.size = size
        self.hparams = self.get_hparams()
        self.max_word_length = self.hparams['max_word_length']
        self.train_samples = self.read_data.train_n_samples
        self.valid_samples = self.read_data.valid_n_samples
        
        # If we are training use the BATCH_SIZE from the hyperparameters
        # else use the testing batch size
        if training == True:
            BATCH_SIZE = self.hparams['BATCH_SIZE']
            self.BATCH_SIZE = BATCH_SIZE
        else:
            BATCH_SIZE = testing_batch_size
            self.BATCH_SIZE = BATCH_SIZE
        
        # Pass the sentences through the CharCNN network
        cnn = tdnn(self.X, kernels, kernel_features)

        # tdnn() returns a tensor of shape [batch_size * sentence_length x kernel_features]
        # highway() returns a tensor of shape [batch_size * sentence_length x size] to use
        # tensorflow dynamic_rnn module we need to reshape it to 
        # [batch_size x sentence_length x size]
        cnn = highway(cnn, self.size)
        cnn = tf.reshape(cnn, [BATCH_SIZE, -1, self.size])
        
        # Build the LSTM
        with tf.variable_scope('LSTM'):
        
            # The following is pretty straight-forward, create a cell and add dropout if
            # necessary. Note that I did not use dropout to get my results, but using it
            # will probably help
            def create_rnn_cell():
                cell = rnn.BasicLSTMCell(rnn_size, state_is_tuple=True,
                                         forget_bias=0.0, reuse=False)

                if dropout > 0.0:
                    cell = tf.contrib.rnn.DropoutWrapper(cell, output_keep_prob=1. - dropout)

                return cell

            cell = create_rnn_cell()
            
            # Initial state of the LSTM cell
            initial_rnn_state = cell.zero_state(BATCH_SIZE, dtype='float32')
            
            # This function returns the outputs at every steps of 
            # the LSTM (i.e. one output for every word)
            outputs, final_rnn_state = tf.nn.dynamic_rnn(cell, cnn,
                                                         initial_state=initial_rnn_state,
                                                         dtype=tf.float32)

            # In this implementation, we only care about the last outputs of the RNN
            # i.e. the output at the end of the sentence
            outputs = tf.transpose(outputs, [1, 0, 2])
            last = outputs[-1]

        self.prediction = softmax(last, 3)
        
    def train(self):
        PATH = '../saved_models/'
        TRAIN_SET = PATH + 'datasets/train_set.csv'
        TEST_SET = PATH + 'datasets/test_set.csv'
        VALID_SET = PATH + 'datasets/valid_set.csv'
        SAVE_PATH = '../saved_models/'
        LOGGING_PATH = '../model_logs/'

        BATCH_SIZE = self.hparams['BATCH_SIZE']
        EPOCHS = self.hparams['EPOCHS']
        max_word_length = self.hparams['max_word_length']
        learning_rate = self.hparams['learning_rate']

        # the probability for each sentiment (pos, neg)
        pred = self.prediction

        # Binary cross-entropy loss
        cost = - tf.reduce_sum(self.Y * tf.log(tf.clip_by_value(pred, 1e-10, 1.0)))

        # The number of "predictions" we got right, we assign a sentence with 
        # a positive connotation when the probability to be positive is greater then
        # the probability of being negative and vice-versa.
        predictions = tf.equal(tf.argmax(pred, 1), tf.argmax(self.Y, 1))

        # Accuracy: # predictions right / total number of predictions
        acc = tf.reduce_mean(tf.cast(predictions, 'float32'))
        tf.summary.scalar('accuracy',acc)
        
        # We use the Adam Optimizer
        global_step = tf.Variable(0, name='global_step',trainable=False)
        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost, global_step=global_step)

        n_batch = self.train_samples // BATCH_SIZE

        # wrtining predictions
        merged = tf.summary.merge_all()
        
        # parameters for saving and early stopping
        saver = tf.train.Saver()
        patience = self.hparams['patience']
        

        
        with tf.Session() as sess:
            train_writer = tf.summary.FileWriter(LOGGING_PATH+'train', sess.graph)
            test_writer = tf.summary.FileWriter(LOGGING_PATH + 'val_test')
            
            #sess.run(tf.initialize_all_variables())
            sess.run(tf.global_variables_initializer())
            
            best_acc = 0.0
            DONE = False
            epoch = 0
            
            #TRAINING
            while epoch <= EPOCHS and not DONE:
                loss = 0.0
                batch = 1
                epoch += 1

                # Load the file in a TextReader object until we've read it all
                # then create a new TextReader Object for how many epochs we 
                # want to loop on
                reader = self.read_data
                for minibatch in reader.iterate_minibatch(BATCH_SIZE, dataset='train'):

                    batch_x = minibatch[0]
                    batch_y = minibatch[1]

                    # Do backprop and compute the cost and accuracy on this minibatch
                    _, c, a, summary = sess.run([optimizer, cost, acc, merged],
                                       feed_dict={self.X: batch_x, self.Y: batch_y})
                    train_writer.add_summary(summary,global_step = global_step)
                    loss += c

                    if batch % 100 == 0:
                        # Compute Accuracy on the Training set and print some info
                        print('Epoch: %5d/%5d -- batch: %5d/%5d -- Loss: %.4f -- Train Accuracy: %.4f' %
                              (epoch, EPOCHS, batch, n_batch, loss/batch, a))

                        # Write loss and accuracy to some file
                        log = open(LOGGING_PATH, 'a')
                        log.write('%s, %6d, %.5f, %.5f \n' % ('train', epoch * batch, loss/batch, a))
                        log.close()

                        # --------------
                        # EARLY STOPPING
                        # --------------

                        # Compute Accuracy on the Validation set, check if validation has improved, save model, etc
                        if batch % 500 == 0:
                            accuracy = []

                            # Validation set is very large, so accuracy is computed on testing set
                            # instead of valid set, change TEST_SET to VALID_SET to compute accuracy on valid set
                            
                            #VALIDATION
                            valid_reader = self.read_data()
                            for mb in valid_reader.iterate_minibatch(BATCH_SIZE, dataset='validate'):
                                #valid_x, valid_y = mb
                                valid_x = mb[0]
                                valid_y = mb[1]
                                a, summary = sess.run([acc, merged], feed_dict={self.X: valid_x, self.Y: valid_y})
                                test_writer.add_summary(summary,global_step = global_step)
                                accuracy.append(a)
                            mean_acc = np.mean(accuracy)

                            # if accuracy has improved, save model and boost patience
                            if mean_acc > best_acc:
                                best_acc = mean_acc
                                save_path = saver.save(sess, SAVE_PATH)
                                patience = self.hparams['patience']
                                print('Model saved in file: %s' % save_path)

                            # else reduce patience and break loop if necessary
                            else:
                                patience -= 500
                                if patience <= 0:
                                    DONE = True
                                    break

                            print('Epoch: %5d/%5d -- batch: %5d/%5d -- Valid Accuracy: %.4f' %
                                 (epoch, EPOCHS, batch, n_batch, mean_acc))

                            # Write validation accuracy to log file
                            log = open(LOGGING_PATH, 'a')
                            log.write('%s, %6d, %.5f \n' % ('valid', epoch * batch, mean_acc))
                            log.close()

                        batch += 1
                        
    def evaluate_test_set(self):
        """
        Evaluate Test Set
        On a model that trained for around 5 epochs it achieved:
        # Valid loss: 23.50035 -- Valid Accuracy: 0.83613
        """
        BATCH_SIZE = self.hparams['BATCH_SIZE']
        max_word_length = self.hparams['max_word_length']

        pred = self.prediction
        
        cost = - tf.reduce_sum(self.Y * tf.log(tf.clip_by_value(pred, 1e-10, 1.0)))

        predictions = tf.equal(tf.argmax(pred, 1), tf.argmax(self.Y, 1))
        

        acc = tf.reduce_mean(tf.cast(predictions, 'float32'))
        tf.summary.scalar('accuracy',acc)
        
        # wrtining predictions
        merged = tf.summary.merge_all()
        
        # parameters for restoring variables
        saver = tf.train.Saver()

        with tf.Session() as sess:
            
            test_writer = tf.summary.FileWriter(LOGGING_PATH + 'test')
            
            print('Loading model %s...' % SAVE_PATH)
            saver.restore(sess, SAVE_PATH)
            print('Done!')
            loss = []
            accuracy = []

            with open(VALID_SET, 'r') as f:
                reader = self.read_data()
                for minibatch in reader.iterate_minibatch(BATCH_SIZE, dataset='train'):
                    batch_x = minibatch[0]
                    batch_y = minibatch[1]

                    c, a = sess.run([cost, acc], feed_dict={self.X: batch_x, self.Y: batch_y})
                    loss.append(c)
                    accuracy.append(a)

                loss = np.mean(loss)
                accuracy = np.mean(accuracy)
                print('Valid loss: %.5f -- Valid Accuracy: %.5f' % (loss, accuracy))
                return loss, accuracy