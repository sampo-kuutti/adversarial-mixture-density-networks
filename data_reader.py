#data_reader loads training data for the neural network model for training/validation
import csv
import random
from numpy import array
NUM_INPUTS = 3
DATA_DIR = '/vol/research/safeav/Supervised Learning/data/'
SHUFFLE = True  # If true data is shuffled to a random order

class DataReader(object):
    """
    reads data from training_data.csv in DATA_DIR,
    appends data to input and label tensors,
    and loads training and/or validation batches to be used in train_sl.py
    """
    def __init__(self, data_dir=DATA_DIR, shuffle=SHUFFLE, file_name='training_data.csv'):
        self.shuffle = shuffle
        self.file_name = file_name
        self.data_dir = data_dir
        self.load()

    # load reader
    def load(self):
        xs = []  # input array
        ys = []  # label array

        # set batch pointers to 0
        self.train_batch_pointer = 0
        self.val_batch_pointer = 0

        total = 0

        # open data files and read inputs and labels
        with open(self.data_dir + self.file_name) as f:
            reader = csv.DictReader(f, delimiter=',')
            for row in reader:
                xs.append(row['DASensor.RadarL.relvTgt.NearPnt.dv'])  # rel velocity
                xs.append(float(row['Th']))  # time headway
                #xs.append(row['Car.ax'])  # host long acc
                xs.append(row['Car.v'])  # host vel
                ys.append(row['Gas-Brake'])  # gas - brake pedal value
                total += 1

        xs = array(xs).reshape(int(len(xs)/NUM_INPUTS), NUM_INPUTS)  # re-shape to a n x 4 matrix

        # shuffle
        if self.shuffle:
            c = list(zip(xs,ys))
            random.shuffle(c)
            xs, ys = zip(*c)

        # slice 80% of data to training data
        self.train_xs = xs[:int(len(xs) * 0.8)]
        self.train_ys = ys[:int(len(ys) * 0.8)]

        # 20% to validation data
        self.val_xs = xs[-int(len(xs) * 0.2):]
        self.val_ys = ys[-int(len(ys) * 0.2):]

        # example counts
        self.num_train_examples = int(len(self.train_ys))
        self.num_val_examples = int(len(self.val_ys))

    # load training data for one batch
    def load_train_batch(self, batch_size):
        x_out = []
        y_out = []
        for i in range(0, batch_size):
            x_out.append([self.train_xs[(self.train_batch_pointer + i) % self.num_train_examples]])
            y_out.append([self.train_ys[(self.train_batch_pointer + i) % self.num_train_examples]])
        self.train_batch_pointer += batch_size
        y_out = array(y_out).reshape(batch_size, 1)  # re-shape to a n x 1 matrix
        x_out = array(x_out).reshape(batch_size, NUM_INPUTS)  # re-shape to a batch_size x 4 matrix
        return x_out, y_out

    # load validation data for one batch
    def load_val_batch(self, batch_size):
        x_out = []
        y_out = []
        for i in range(0, batch_size):
            x_out.append([self.val_xs[(self.val_batch_pointer + i) % self.num_val_examples]])
            y_out.append([self.val_ys[(self.val_batch_pointer + i) % self.num_val_examples]])
        self.val_batch_pointer += batch_size
        y_out = array(y_out).reshape(batch_size, 1)  # re-shape to a n x 1 matrix
        x_out = array(x_out).reshape(batch_size, NUM_INPUTS)  # re-shape to a batch_size x 4 matrix
        return x_out, y_out

    # skip a specific number of training examples
    def skip(self, num):
        self.train_batch_pointer += num

