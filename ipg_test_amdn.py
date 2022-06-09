# =====================================================================================================================
# Author: Sampo Kuutti (s.j.kuutti@surrey.ac.uk)
# Organisation: University of Surrey
#
# ipg_test_*.py tests the trained supervised learning agent in an IPG simulation,
# the selected trained neural network model is selected by the MODEL_FILE parameter in the LOG_DIR directory, the
# chosen model file should be a valid tensorflow trainer checkpoint
# =====================================================================================================================

import os
import tensorflow as tf
import argparse
import ctypes
import pythonapi
import csv
import time
from tensorflow_probability import distributions as tfd
import numpy as np

MODEL_FILE = 'model-step-507000-val--0.926923.h5'
DATA_DIR = './data/'
LOG_DIR = './log_mdn/c139742_p2'
# MDN params
no_parameters = 3
components = 1
neurons = 50


def get_arguments():
    parser = argparse.ArgumentParser(description='SL testing')
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
        '--store_metadata',
        type=bool,
        default=False,
        help='Storing debug information for TensorBoard.'
    )
    parser.add_argument(
        '--restore_from',
        type=str,
        default=MODEL_FILE,
        help='Checkpoint file to restore model weights from.'
    )
    return parser.parse_args()


class MdnAgent():
    def __init__(self, neurons=50, components=2):
        self.neurons = neurons
        self.components = components

        self.model = self.build_model(self.neurons, self.components)

    def build_model(self, neurons, components):
        inputs = tf.keras.layers.Input(shape=3, name='inputs')
        labels = tf.keras.layers.Input(shape=1, name='labels')
        h1 = tf.keras.layers.Dense(neurons, activation='relu', name='h1')(inputs)
        h2 = tf.keras.layers.Dense(neurons, activation='relu', name='h2')(h1)
        h3 = tf.keras.layers.Dense(neurons, activation='relu', name='h3')(h2)

        mus = tf.keras.layers.Dense(components, activation='tanh', name='mus')(h3)
        sigmas = tf.keras.layers.Dense(components, activation='nnelu', name='sigmas')(h3)

        gm = tfd.Normal(
            loc=mus,
            scale=sigmas,
            name='dist')
        nll_loss = -gm.log_prob(labels)

        model = tf.keras.models.Model(inputs=[inputs, labels], outputs=[mus, sigmas])
        model.add_loss(nll_loss)

        return model


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
    # set up tf session and model
    args = get_arguments()
    sess = tf.Session()

    tf.keras.utils.get_custom_objects().update({'nnelu': tf.keras.layers.Activation(nnelu)})

    with sess.as_default():
        agent = MdnAgent(neurons=neurons, components=components)
        model = agent.model
        opt = tf.train.AdamOptimizer(1e-3)
        model.compile(optimizer=opt)
        checkpoint_path = os.path.join(args.log_dir, args.restore_from)
        #saver = tf.train.Saver()
        #saver.restore(sess, checkpoint_path)
        model.load_weights(checkpoint_path, by_name=True)
    print('Restored model: %s' % args.restore_from)

    # initial threshold
    threshold = 0.86
    arr_scen = []

    for i in range(1, 121):
        # set up connection to carmaker
        pythonapi.api_setup()

        # empty arrays
        A = []  # acceleration array
        J = []  # jerk array
        T = []  # time array
        X = []  # x_rel array
        V = []  # velocity array
        DV = [] # relative velocity array
        T_h = [] # time headway array
        Y_0 = []    # original output
        Y_SC = []   # safety cage output
        SC = []     # safety cage number

        V_leader = []   # lead vehicle velocity
        A_leader = []   # lead vehicle acceleration

        print('test no. %d' % i)
        # load test run
        #scen = random.randint(1,25)
        #arr_scen.append(scen)
        with open('./traffic_data/' + 'scens.csv') as f:
            reader = csv.DictReader(f, delimiter=',')
            for row in reader:
                arr_scen.append(float(row['s']))  # test run id
        scen = int(arr_scen[i - 1])
        pythonapi.sim_loadrun2(scen)

        # subscribe quantities
        pythonapi.subscribe_quants()

        # while simulation is running feed states to nn and use outputs
        # set variables to 0
        b = 0
        v_rel = 0
        v = 0
        x_rel = 0
        a = 0
        t = 0

        # lead vehicle states
        #T_lead, X_lead, V_lead, A_lead = traffic.lead_vehicle()
        T_lead = []
        X_lead = []
        V_lead = []
        A_lead = []
        with open('./traffic_data/' + str(i) + '.csv') as f:
            reader = csv.DictReader(f, delimiter=',')
            for row in reader:
                T_lead.append(float(row['t']))      # time
                X_lead.append(float(row['x']))      # long. position
                V_lead.append(float(row['v']))      # velocity
                A_lead.append(float(row['a']))      # acceleration

        # initialise pedals at 0
        pythonapi.set_gas(ctypes.c_double(0))
        pythonapi.set_brake(ctypes.c_double(0))

        # start simulation
        pythonapi.sim_start()
        pythonapi.sim_waitready()

        while pythonapi.sim_isrunning() != 0:       # check if simulation is running
            pythonapi.ApoClnt_PollAndSleep()        # poll client
            t = pythonapi.get_time()
            if t > 0:

                b += 1

                # read host states
                v = pythonapi.get_hostvel()         # host velocity
                a = pythonapi.get_longacc()         # host longitudinal acceleration
                x = pythonapi.get_hostpos()         # host longitudinal position

                # lead vehicle states
                t_iter = int(t // 0.02)             # current time step
                v_rel = V_lead[t_iter] - v          # relative velocity
                x_rel = X_lead[t_iter] - x          # relative distance

                # enter variables into arrays
                A.append(a)
                T.append(t)
                X.append(x_rel)
                V.append(v)
                DV.append(v_rel)

                V_leader.append(V_lead[t_iter])
                A_leader.append(A_lead[t_iter])

                # calculate time headway
                if v != 0:
                    t_h = x_rel / v
                else:
                    t_h = x_rel

                T_h.append(t_h)

                # define inputs to neural network
                inputs = [v_rel, t_h, v]

                # evaluate neural network output
                with sess.as_default():
                    mu = model.predict(x=[np.reshape(inputs, (1, 3)), np.reshape(np.array(0.0), (1,1))], batch_size=1)[0]

                    y = mu

                Y_0.append(float(y))

                # safety cage implementation
                output = y
                sc = 0

                # clip outputs to [-1, 1]
                if output > 1:
                    output = 1
                elif output < -1:
                   output = -1

                Y_SC.append(float(output))
                SC.append(sc)

                # convert normalised output to gas and brake signals
                if output < 0:          # output brake command
                    gas = 0
                    brake = abs(output)
                elif output > 0:        # output gas command
                    gas = output
                    brake = 0
                elif output == 0:       # both outputs are zero
                    gas = 0
                    brake = 0
                else:   # something has gone wrong
                    gas = 0
                    brake = 0
                    print('invalid control signal, setting pedal values to 0')

                #  send commands to carmaker
                pythonapi.set_gas(ctypes.c_double(gas))
                pythonapi.set_brake(ctypes.c_double(brake))

                # stop simulation if a crash occurs
                if x_rel <= 0:
                    pythonapi.sim_stop()
                    print('crash occurred: simulation run stopped')

        print('mean th = %.4f, min th = %.4f' % (np.mean(T_h), np.min(T_h)))
        # calculate jerk array
        for k in range(0, 5):
            J.append(float(0))

        for k in range(5, len(T)):
            # calculate vehicle jerk
            if abs(T[k] - T[k-5]) != 0:
                J.append(((A[k]) - (A[k-5])) / (T[k] - T[k-5]))  # jerk
            else:
                J.append(0)

        # write results to file
        if not os.path.exists(args.log_dir + '/results_'):
            os.makedirs(args.log_dir + '/results_')
        headers = ['t', 'j', 'v', 'a', 'v_lead', 'a_lead', 'x_rel', 'v_rel', 'th', 'y_0', 'y_sc', 'sc']
        with open(args.log_dir + '/results_/' + str(i) + '.csv', 'w', newline='\n') as f:
            wr = csv.writer(f, delimiter=',')
            rows = zip(T, J, V, A, V_leader, A_leader, X, DV, T_h, Y_0, Y_SC, SC)
            wr.writerow(headers)
            wr.writerows(rows)

        # terminate api connection
        pythonapi.api_terminate()
        time.sleep(1)

    # print test stuff
    print('simulation run complete')

    # write scenarios used to file
    print('scenarios used:')
    print(arr_scen)
    #with open('./traffic_data/' + 'scens.csv', 'w', newline='\n') as f:
    #    wr = csv.writer(f, delimiter=',')
    #    rows = zip(arr_scen)
    #    wr.writerow('s')
    #    wr.writerows(rows)


if __name__ == '__main__':
    main()
