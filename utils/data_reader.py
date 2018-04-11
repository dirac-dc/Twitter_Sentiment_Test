import numpy as np
import tensorflow as tf
from numpy import random
import nltk
#nltk.download_shell()
from nltk.tokenize import word_tokenize
import string
from unidecode import unidecode

class read_data():
    
    def __init__(self,create=True):
        self.max_word_length = 16
        if create:
            self.create_data()
        self.batch_count = 0
        emb_alphabet = 'abcdefghijklmnopqrstuvwxyz0123456789-' \
           ',;.!?:\'"/\\|_@#$%^&*~`+-=<>()[]{} '
        self.ALPHABET_SIZE = len(emb_alphabet)
        self.DICT = {ch: ix for ix, ch in enumerate(emb_alphabet)}
        
    def create_data(self,samples = 3000, test_perc=0.2):
        print('CREATING DATA')
        self.samples = samples
        self.test_perc = test_perc
        DATA_LIST =[('./datasets/negative',-1), ('./datasets/neutral',0), ('./datasets/positive',1)]
        SAMPLES = self.samples
        lines =[]
        for dstr in DATA_LIST:
            with open(dstr[0], 'r') as f:
                    temp_lines = f.readlines()
                    temp_lines = random.choice(temp_lines,SAMPLES/3,replace=False)
                    temp_lines = [[x,dstr[1]] for x in temp_lines]
                    print('Number of {} samples used: {}'.format(dstr[0],len(temp_lines)))
                    lines += temp_lines

        random.shuffle(lines)
        lines_train = lines[:int(len(lines) * (1 -test_perc))]
        lines_test = lines[int(len(lines) * (1 - test_perc)):]
        self.train_data = lines_train
        self.test_data = lines_test
        
        self.test_n_samples = int( np.shape(self.test_data)[0] * 0.95 )
        self.train_n_samples = np.shape(self.train_data)[0]
        self.valid_n_samples = int( np.shape(self.test_data)[0] * 0.05 )
        
        self.current_data = lines_train # this act as a flag to change from training, validation and test sets
        return None
                              
    def oh_encoder(self, sentence):
    # Convert Sentences to np.array of Shape 
    # ('sent_length', 'word_length', 'emb_size')
        max_word_length = self.max_word_length
        sent = []

        # We need to keep track of the maximum length of the sentence in a minibatch
        # so that we can pad them with zeros, this is why we return the length of every
        # sentences after they are converted to one-hot tensors
        SENT_LENGTH = 0

        # Here, we remove any non-printable characters in a sentence (mostly
        # non-ASCII characters)
        printable = string.printable
        encoded_sentence = filter(lambda x: x in printable, sentence)

        # word_tokenize() splits a sentence into an array where each element is
        # a word in the sentence, for example, 
        # "My name is Charles" => ["My", "name", "is", Charles"]
        # Unidecode convert characters to utf-8
        for word in word_tokenize(unidecode(encoded_sentence)):

            # Encode one word as a matrix of shape [max_word_length x ALPHABET_SIZE]
            word_encoding = np.zeros(shape=(max_word_length, self.ALPHABET_SIZE))

            for i, char in enumerate(word):

                # If the character is not in the alphabet, ignore it    
                try:
                    char_encoding = DICT[char]
                    one_hot = np.zeros(ALPHABET_SIZE)
                    one_hot[char_encoding] = 1
                    word_encoding[i] = one_hot

                except Exception as e:
                    pass

            sent.append(np.array(word_encoding))
            SENT_LENGTH += 1

        return np.array(sent), SENT_LENGTH

    def make_minibatch(self, sentences):
    # Create a minibatch of sentences and convert sentiment
    # to a one-hot vector, also takes care of padding

        max_word_length = self.max_word_length
        minibatch_x = []
        minibatch_y = []
        max_length = 0

        for sentence in sentences:
            # Append the one-hot encoding of the sentiment to the minibatch of Y
            # -1: Negative 0: Neutral 1: Positive
            if sentence[1] == -1:
                minibatch_y.append(np.array([1,0,0]))
            elif sentence[1] == 0:
                minibatch_y.append(np.array([0,1,0]))
            elif sentence[1] == 1:
                minibatch_y.append(np.array([0,0,1]))

            # One-hot encoding of the sentence
            one_hot, length = self.oh_encoder(sentence[0])

            # Calculate maximum_sentence_length
            if length >= max_length:
                max_length = length

            # Append encoded sentence to the minibatch of X
            minibatch_x.append(one_hot)


        # data is a np.array of shape ('b', 's', 'w', 'e') we want to
        # pad it with np.zeros of shape ('e',) to get 
        # ('b', 'SENTENCE_MAX_LENGTH', 'WORD_MAX_LENGTH', 'e')
        def numpy_fillna(data):
            """ This is a very useful function that fill the holes in our tensor """

            # Get lengths of each row of data
            lens = np.array([len(i) for i in data])

            # Mask of valid places in each row
            mask = np.arange(lens.max()) < lens[:, None]

            # Setup output array and put elements from data into masked positions
            out = np.zeros(shape=(mask.shape + (max_word_length, self.ALPHABET_SIZE)),
                           dtype='float32')

            out[mask] = np.concatenate(data)
            return out

        # Padding...
        minibatch_x = numpy_fillna(minibatch_x)

        return minibatch_x, np.array(minibatch_y)
    
    def load_to_ram(self, batch_size):
        """ Load n Rows from File f to Ram """
        # Returns True if there are still lines in the buffer, 
        # otherwise returns false - the epoch is over
        
        self.data = []
        n_rows = batch_size
        temp_count = 0
        while n_rows > 0:
            self.data.append(self.current_data[self.batch_count+(batch_size-n_rows)])
            n_rows -= 1
        if n_rows == 0:
            self.batch_count += batch_size 
            return True
        else:
            return False
        
    def iterate_minibatch(self, batch_size, dataset='train'):
        """ Returns Next Batch """

        # I realize this could be more 
        if dataset == 'train':
            n_samples = self.train_n_samples
            self.current_data = self.train_data[:n_samples]
        elif dataset == 'validate':
            n_samples = self.valid_n_samples
            self.current_data = self.train_data[:n_samples:-1][::-1]
        elif dataset == 'test':
            n_samples = self.test_n_samples
            self.current_data = self.test_data
        # Number of batches / number of iterations per epoch
        n_batch = int(n_samples // batch_size)
        self.batch_count = 0
        
        # Creates a minibatch, loads it to RAM and feed it to the network
        # until the buffer is empty
        for i in range(n_batch):
            if self.load_to_ram(batch_size):
                inputs, targets = self.make_minibatch(self.data)
                return inputs, targets