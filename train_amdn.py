import os
import tensorflow as tf
from tensorflow_probability import distributions as tfd
import argparse
import data_reader
import data_reader_coll
import numpy as np
import random
import datetime
import time
timestamp = datetime.datetime.fromtimestamp(time.time()).strftime('%H:%M:%S')
BATCH_SIZE = 100
DATA_DIR = './data'
LOG_DIR = '/vol/research/safeav/Supervised Learning/log_mdn/'
LOG_NAME = 'condor_amdn_v7_' + str(random.randint(0, 99)) + timestamp  # save location for logs
CHECKPOINT_EVERY = 1000
NUM_STEPS = int(1e6)
CKPT_FILE = 'model.ckpt'
LEARNING_RATE = 1e-4
LEARNING_RATE_A = 1e-5
LEARNING_RATE_B = 1e-8
VALIDATION_EVERY = 1000
RESTORE_FROM = None
# KEEP_PROB = 0.8
# MDN params
no_parameters = 3
components = 1
neurons = 50
LAMBDA_KD = 1e-5

def get_arguments():
    parser = argparse.ArgumentParser(description='SL training')
    parser.add_argument(
        '--lr',
        type=float,
        default=LEARNING_RATE,
        help='Initial learning rate'
    )
    parser.add_argument(
        '--lr_a',
        type=float,
        default=LEARNING_RATE_A,
        help='Initial learning rate for adv model'
    )
    parser.add_argument(
        '--lr_b',
        type=float,
        default=LEARNING_RATE_B,
        help='Initial learning rate for model on coll data'
    )
    parser.add_argument(
        '--neurons',
        type=int,
        default=neurons,
        help='no. of hidden neurons'
    )
    parser.add_argument(
        '--components',
        type=int,
        default=components,
        help='No. of components for gaussian mixture'
    )
    parser.add_argument(
        '--lambda_kd',
        type=float,
        default=LAMBDA_KD,
        help='Coefficient for KL Loss'
    )
    parser.add_argument(
        '--max_steps',
        type=int,
        default=NUM_STEPS,
        help='Number of steps to run trainer'
    )
    parser.add_argument(
        '--batch_size',
        type=int,
        default=BATCH_SIZE,
        help='Batch size. Must divide evenly into dataset sizes.'
    )
    parser.add_argument(
        '--checkpoint_every',
        type=int,
        default=CHECKPOINT_EVERY,
        help='Number of steps before checkpoint.'
    )
    parser.add_argument(
        '--validation_every',
        type=int,
        default=VALIDATION_EVERY,
        help='Number of steps after which the model is evaluated on validation data.'
    )
    parser.add_argument(
        '--input_data_dir',
        type=str,
        default=DATA_DIR,
        help='Directory to put the input data.'
    )
    parser.add_argument(
        '--log_dir',
        type=str,
        default=LOG_DIR,
        help='Directory to put the log data.'
    )
    parser.add_argument(
        '--log_name',
        type=str,
        default=LOG_NAME,
        help='Folder to put the log data.'
    )
    parser.add_argument(
        '--store_metadata',
        type=bool,
        default=False,
        help='Storing debug information for TensorBoard.'
    )
    parser.add_argument(
        '--restore_from',
        type=str,
        default=RESTORE_FROM,
        help='Checkpoint file to restore model weights from.'
    )
    return parser.parse_args()


def build_models(neurons=50, components=2, lambda_kd=LAMBDA_KD):
    inputs = tf.keras.layers.Input(shape=3, name='inputs')
    h1 = tf.keras.layers.Dense(neurons, activation='relu', name='h1')(inputs)
    h2 = tf.keras.layers.Dense(neurons, activation='relu', name='h2')(h1)
    h3 = tf.keras.layers.Dense(neurons, activation='relu', name='h3')(h2)

    # output for SL control action
    mus = tf.keras.layers.Dense(components, activation='tanh', name='mus')(h3)
    sigmas = tf.keras.layers.Dense(components, activation='nnelu', name='sigmas')(h3)

    # adversarial control action
    adv_mus = tf.keras.layers.Dense(components, activation='tanh', name='adv_mus')(h3)
    adv_sigmas = tf.keras.layers.Dense(components, activation='nnelu', name='adv_sigmas')(h3)

    labels = tf.keras.Input(shape=1, name='labels')
    adv_labels = tf.keras.Input(shape=1, name='adv_labels')
    labels_adv_mus = tf.keras.Input(shape=1, name='labels_adv_mus')
    labels_adv_sigmas = tf.keras.Input(shape=1, name='labels_adv_sigmas')

    # probability distributions
    gm = tfd.Normal(
            loc=mus,
            scale=sigmas,
    name='dist')
    adv_gm = tfd.Normal(
            loc=adv_mus,
            scale=adv_sigmas,
    name='dist_adv')
    adv2_gm = tfd.Normal(
        loc=labels_adv_mus,
        scale=labels_adv_sigmas,
        name='dist2_adv')

    # negative log likelihood loss
    nll_loss = -gm.log_prob(labels)

    # kl divergence
    kl = lambda_kd * tfd.kl_divergence(distribution_a=adv2_gm, distribution_b=gm)
    kl_loss = -kl

    # adversarial model's loss
    adv_nll_loss = -adv_gm.log_prob(adv_labels)

    # create models
    model = tf.keras.models.Model(inputs=[inputs, labels], outputs=[mus, sigmas])
    model2 = tf.keras.models.Model(inputs=[inputs, labels, labels_adv_mus, labels_adv_sigmas], outputs=[mus, sigmas])
    adv_model = tf.keras.models.Model(inputs=[inputs, adv_labels], outputs=[adv_mus, adv_sigmas])

    # add losses
    model.add_loss(nll_loss)
    model2.add_loss(kl_loss)
    adv_model.add_loss(adv_nll_loss)

    # add metrics
    model.add_metric(nll_loss, aggregation='mean', name='nll_loss')
    model2.add_metric(kl, aggregation='mean', name='kl_loss')

    return model, adv_model, model2


def nnelu(input):
    """ Computes the Non-Negative Exponential Linear Unit
    """
    return tf.add(tf.constant(1, dtype=tf.float32), tf.nn.elu(input))


def slice_parameter_vectors(parameter_vector):
    """ Returns an unpacked list of paramter vectors.
    """
    return [parameter_vector[:, i * components:(i + 1) * components] for i in range(no_parameters)]


def gnll_loss(y, parameter_vector):
    """ Computes the mean negative log-likelihood loss of y given the mixture parameters.
    """
    alpha, mu, sigma = slice_parameter_vectors(parameter_vector)  # Unpack parameter vectors

    gm = tfd.MixtureSameFamily(
        mixture_distribution=tfd.Categorical(probs=alpha),
        components_distribution=tfd.Normal(
            loc=mu,
            scale=sigma))

    log_likelihood = gm.log_prob(tf.transpose(y))  # Evaluate log-probability of y

    return -tf.reduce_mean(log_likelihood, axis=-1)



def main():
    args = get_arguments()
    sess = tf.Session()

    tf.keras.utils.get_custom_objects().update({'nnelu': tf.keras.layers.Activation(nnelu)})

    model, adv_model, model2 = build_models(neurons=neurons, components=components, lambda_kd=args.lambda_kd)
    opt = tf.keras.optimizers.Adam(learning_rate=args.lr)
    adv_opt = tf.keras.optimizers.Adam(learning_rate=args.lr_a)
    adv_model.compile(optimizer=adv_opt)
    model.compile(optimizer=opt)
    model2.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=args.lr_b))


    # tensorboard summary
    merged = tf.summary.merge_all()
    train_writer = tf.summary.FileWriter(args.log_dir + args.log_name + '/train', sess.graph)
    val_writer = tf.summary.FileWriter(args.log_dir + args.log_name + '/val')

    start_step = 0

    min_loss = 100
    min_loss_step = 0

    reader = data_reader.DataReader()
    adv_reader = data_reader_coll.DataReader(file_name='dataset_collisions_25.csv')

    # run training
    for i in range(start_step, start_step + args.max_steps):
        # run training on normal data
        xs, ys = reader.load_train_batch(args.batch_size)
        train_loss, nll_loss = model.train_on_batch([xs, ys])
        train_loss = np.mean(train_loss)

        # run adversarial training
        axs, ays = adv_reader.load_train_batch(args.batch_size)
        adv_train_loss = adv_model.train_on_batch([axs, ays])
        adv_train_loss = np.mean(adv_train_loss)
        # train model2 on kl loss
        adv_out = adv_model.predict_on_batch([axs, ays])

        # adv_mus = np.ravel(adv_out[0])
        adv_mus = np.array(adv_out[0])
        adv_mus = np.reshape(adv_mus, (len(adv_mus), 1))
        adv_sigmas = np.array(adv_out[1])
        adv_sigmas = np.reshape(adv_sigmas, (len(adv_sigmas), 1))
        kl2_loss, kl_loss = model2.train_on_batch([xs, ys, adv_mus, adv_sigmas])
        kl2_loss = np.mean(kl2_loss)

        if i % 100 == 0:  # print training loss every 100 steps and update tensorboard

            train_summary = tf.Summary(
                value=[tf.Summary.Value(tag='Loss', simple_value=float(train_loss))])
            train_writer.add_summary(train_summary, i)
            train_writer.flush()
            train_summary = tf.Summary(
                value=[tf.Summary.Value(tag='Nll_Loss', simple_value=float(nll_loss))])
            train_writer.add_summary(train_summary, i)
            train_writer.flush()
            train_summary = tf.Summary(
                value=[tf.Summary.Value(tag='Kl_Loss', simple_value=float(kl_loss))])
            train_writer.add_summary(train_summary, i)
            train_writer.flush()
            train_summary = tf.Summary(
                value=[tf.Summary.Value(tag='Kl2_Loss', simple_value=float(kl2_loss))])
            train_writer.add_summary(train_summary, i)
            train_writer.flush()
            train_summary = tf.Summary(
                value=[tf.Summary.Value(tag='Adv_Loss', simple_value=float(adv_train_loss))])
            train_writer.add_summary(train_summary, i)
            train_writer.flush()
            print("Step %d, train loss %g, adv loss %g" % (i, train_loss, adv_train_loss))

        # check validation error every 1000 steps
        if i % 1000 == 0:
            # normal data
            xs, ys = reader.load_val_batch(1000)

            val_loss, val_nll_loss = model.test_on_batch([xs, ys])
            val_loss = np.mean(val_loss)

            # adversarial data
            axs, ays = adv_reader.load_val_batch(1000)
            adv_val_loss = adv_model.test_on_batch([axs, ays])
            adv_val_loss = np.mean(adv_val_loss)
            # kl loss on model 2
            adv_out = adv_model.predict_on_batch([axs, ays])
            adv_mus = np.array(adv_out[0])
            adv_mus = np.reshape(adv_mus, (len(adv_mus), 1))
            adv_sigmas = np.array(adv_out[1])
            adv_sigmas = np.reshape(adv_sigmas, (len(adv_sigmas), 1))
            val_kl2_loss, val_kl_loss = model2.train_on_batch([axs, ays, adv_mus, adv_sigmas])
            val_kl2_loss = np.mean(val_kl2_loss)

            # tensorboard validation summary
            val_summary = tf.Summary(
                value=[tf.Summary.Value(tag='Loss', simple_value=float(val_loss))])
            val_writer.add_summary(val_summary, i)
            val_writer.flush()
            val_summary = tf.Summary(
                value=[tf.Summary.Value(tag='Nll_Loss', simple_value=float(val_nll_loss))])
            val_writer.add_summary(val_summary, i)
            val_writer.flush()
            val_summary = tf.Summary(
                value=[tf.Summary.Value(tag='Kl_Loss', simple_value=float(val_kl_loss))])
            val_writer.add_summary(val_summary, i)
            val_writer.flush()
            val_summary = tf.Summary(
                value=[tf.Summary.Value(tag='Kl2_Loss', simple_value=float(val_kl2_loss))])
            val_writer.add_summary(val_summary, i)
            val_writer.flush()
            val_summary = tf.Summary(
                value=[tf.Summary.Value(tag='Adv_Loss', simple_value=float(adv_val_loss))])
            val_writer.add_summary(val_summary, i)
            val_writer.flush()

            print("Step %d, val loss %g, adv val loss %g" % (i, val_loss, adv_val_loss))

            # save checkpoint
            if i > 10 and i % args.checkpoint_every == 0:
                if not os.path.exists(args.log_dir + args.log_name):
                    os.makedirs(args.log_dir + args.log_name)
                    checkpoint_path = os.path.join((args.log_dir + args.log_name), "model-step-%d-val-%g.h5" % (i, val_loss))
                    model.save_weights(checkpoint_path)
                    print("Model saved in file: %s" % checkpoint_path)
                elif val_loss < min_loss:
                    min_loss = val_loss
                    min_loss_step = i
                    if not os.path.exists(args.log_dir):
                        os.makedirs(args.log_dir)
                    checkpoint_path = os.path.join((args.log_dir + args.log_name), "model-step-%d-val-%g.h5" % (i, val_loss))
                    model.save_weights(checkpoint_path)
                    checkpoint_path = os.path.join((args.log_dir + args.log_name), "model-adv-step-%d-val-%g.h5" % (i, val_loss))
                    adv_model.save_weights(checkpoint_path)
                    checkpoint_path = os.path.join((args.log_dir + args.log_name), "model-2-step-%d-val-%g.h5" % (i, val_loss))
                    model2.save_weights(checkpoint_path)
                    print("Model saved in file: %s" % checkpoint_path)

    print('Minimum validation loss %g at step %d' % (min_loss, min_loss_step))


if __name__ == '__main__':
    main()
