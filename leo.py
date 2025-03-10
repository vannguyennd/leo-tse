import tensorflow as tf
from tensorflow import keras
import numpy as np

from tensorflow.keras.layers import Dense, Input, Flatten, Add, Multiply, Lambda, Permute, \
    Embedding, Activation, Conv1D, GlobalMaxPooling1D, MaxPooling1D, Embedding, \
    Dropout, LSTM, Bidirectional, TimeDistributed, Reshape, Layer
from tensorflow.keras import backend as K
from tensorflow.keras import optimizers
from utils import get_eval_results_save_features_submit, supervised_nt_xent_loss

from sklearn.cluster import KMeans
import tensorflow_probability as tfp
import os
import tensorflow as tf
from readData import create_data_set_nd, split_data

tf.random.set_seed(10086)
np.random.seed(10086)

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    # Restrict TensorFlow to use specific GPUs, e.g., the first GPU
    try:
        tf.config.experimental.set_visible_devices(gpus[0], 'GPU')
        tf.config.experimental.set_memory_growth(gpus[0], True)
    except RuntimeError as e:
        # Visible devices must be set at program startup
        print(e)

class Embeddings(tf.keras.Model):
    def __init__(self, args, MAX_NUM_WORDS):
        super().__init__()
        self.output_dim = 250
        self.embedding_layer = Embedding(MAX_NUM_WORDS,
                                         args.emb,
                                         input_length=args.tlen,
                                         name='embedding',
                                         trainable=True)
        self.dropout = Dropout(0.2)
        ''''''
        self.conv1d = Conv1D(self.output_dim, 3,
                             padding='valid',
                             activation='relu',
                             strides=1)
        ''''''
        self.maxpooling1d = GlobalMaxPooling1D()

    def call(self, inputs):
        embedded_sequences = self.embedding_layer(inputs)
        net = self.dropout(embedded_sequences)
        net = self.conv1d(net)
        net = self.maxpooling1d(net)
        return net

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.output_dim)


class EmbeddingTimes(tf.keras.Model):
    def __init__(self, args, MAX_NUM_WORDS):
        super().__init__()
        self.embeddings = Embeddings(args, MAX_NUM_WORDS)
        self.encoding = TimeDistributed(self.embeddings)

        self.net_1_do = Dropout(0.20, name='dropout_1')
        self.x_net_1 = Dense(100, name='new_dense_1', activation='relu')

        self.net_2_do = Dropout(0.20, name='dropout_2')
        self.x_net_2 = Dense(100, name='new_dense_2', activation='relu')

    def call(self, inputs):
        embedded_sequences = self.encoding(inputs)
        embedded_sequences = self.net_1_do(embedded_sequences)
        embedded_sequences = self.x_net_1(embedded_sequences)

        embedded_sequences = self.net_2_do(embedded_sequences)
        embedded_sequences = self.x_net_2(embedded_sequences)

        return embedded_sequences


class Random_Bernoulli_Sampler(Layer):
    '''
    Layer to Sample r
    '''

    def __init__(self, **kwargs):
        super(Random_Bernoulli_Sampler, self).__init__(**kwargs)

    def call(self, logits):
        batch_size = tf.shape(logits)[0]
        d = tf.shape(logits)[1]

        u = tf.random.uniform(shape=(batch_size, d),
                              minval=np.finfo(tf.float32.as_numpy_dtype).tiny, maxval=1.0)

        r = K.cast(tf.stop_gradient(u > 0.5), tf.float32)
        r = tf.expand_dims(r, -1)

        return r
    
    def compute_output_shape(self, input_shape):
        return input_shape


class SampleConcrete(Layer):
    def __init__(self, args, **kwargs):
        super(SampleConcrete, self).__init__(**kwargs)

    def call(self, logits):
        lo_gits_ = K.permute_dimensions(logits, (0, 2, 1))

        uni_shape = K.shape(logits)[0]
        uniform_a = K.random_uniform(shape=(uni_shape, 1, args.slen),
                                     minval=np.finfo(tf.float32.as_numpy_dtype).tiny, maxval=1.0)
        uniform_b = K.random_uniform(shape=(uni_shape, 1, args.slen),
                                     minval=np.finfo(tf.float32.as_numpy_dtype).tiny, maxval=1.0)
        gumbel_a = -K.log(-K.log(uniform_a))
        gumbel_b = -K.log(-K.log(uniform_b))

        no_z_lo_gits = K.exp((gumbel_a + lo_gits_) / args.tau)
        de_z_lo_gits = no_z_lo_gits + K.exp((gumbel_b + (1.0 - lo_gits_)) / args.tau)

        samples = no_z_lo_gits / de_z_lo_gits

        logits = tf.reshape(lo_gits_, [-1, lo_gits_.shape[-1]])
        threshold = tf.expand_dims(tf.nn.top_k(logits, args.k, sorted=True)[0][:, -1], -1)
        discrete_logits = tf.cast(tf.greater_equal(logits, threshold), tf.float32)

        return K.in_train_phase(K.permute_dimensions(samples, (0, 2, 1)), tf.expand_dims(discrete_logits, -1))

    def compute_output_shape(self, input_shape):
        return input_shape


class Selector(tf.keras.Model):
    def __init__(self, args):
        super().__init__()

        self.sample_do = Dropout(0.20, name='dropout_3')
        self.sample_dense = Dense(1, name='new_dense_logits', activation='sigmoid')
        self.sc = SampleConcrete(args)
    
    def call(self, x_input):
        data_input = self.sample_do(x_input)
        data_input = self.sample_dense(data_input)
        embedded_sequences_T = self.sc(data_input)

        return data_input, embedded_sequences_T


class Predictor(tf.keras.Model):
    def __init__(self):
        super().__init__()

        self.multiply = Multiply()
        self.logits_T_lstm = LSTM(128)
        self.logits_T_dense = Dense(2, name='new_dense_softmax', activation='softmax')

        self.bernoulli_sampling = Random_Bernoulli_Sampler()

    def call(self, x_input, logits, T, joint=True):
        if joint is True:
            selected_encoding = self.multiply([x_input, T])
        else:
            T_bernoulli = self.bernoulli_sampling(logits)
            T_bernoulli = tf.squeeze(T_bernoulli)
            sc_T_bernoulli = tfp.distributions.RelaxedBernoulli(0.5, logits=T_bernoulli)
            sc_T_bernoulli = tf.expand_dims(sc_T_bernoulli.sample(), axis=-1)
            
            selected_encoding = self.multiply([x_input, sc_T_bernoulli])
            
        selected_encoding = self.logits_T_lstm(selected_encoding)
        selected_encoding = self.logits_T_dense(selected_encoding)

        return selected_encoding


class Mymodel(tf.keras.Model):
    def __init__(self, args, MAX_NUM_WORDS):
        super().__init__()

        self.embeddings = EmbeddingTimes(args, MAX_NUM_WORDS)
        self.selector = Selector(args)
        self.classifier = Predictor()

        self.joint = True

    def call(self, x_input):
        x_input_embeded = self.embeddings(x_input)
        logits, T = self.selector(x_input_embeded)
        logits_y = self.classifier(x_input_embeded, logits, T, self.joint)

        return logits_y, x_input_embeded, logits


def leo_train(train_data, hpcfs_path, args, MAX_NUM_WORDS):
    """
    LEO in the training phase.
    """
    # declare the optimizer and the loss function
    opt = optimizers.Adam(learning_rate=args.lr, clipnorm=1.0)
    loss_fn = keras.losses.SparseCategoricalCrossentropy()

    print('Creating model...')
    my_network = Mymodel(args, MAX_NUM_WORDS)

    # model directory
    saved_model_dir = hpcfs_path + 'saved_models' + '/' + \
        str(args.lr) + '_' + str(args.lam) + '_' + str(args.tau) + '_' + str(args.tem) + '_' + str(args.cls) + '/'
    if not os.path.exists(saved_model_dir):
        os.makedirs(saved_model_dir)

    results_file = open(hpcfs_path + 'train_logs.txt', 'a+')
    results_file.write('\np_lr: %f --- p_lambda: %f --- p_tau: %f --- p_temp: %f --- p_cluster: %d \n' %
                        (args.lr, args.lam, args.tau, args.tem, args.cls))

    # for training and saving the model
    for epoch in range(args.epo):
        results_file.write("epoch: %d \n" % epoch)
        print("epoch: %d" % epoch)
        for batch_no, (data, labels) in enumerate(train_data):
            with tf.GradientTape(persistent=True) as tape_ec:
                my_network.selector.trainable = False
                my_network.joint = False
                logits, x_input, data_input = my_network(data, training=True)

                loss_c = loss_fn(labels, logits)
                variables = my_network.trainable_variables
                gradients = tape_ec.gradient(loss_c, variables)
                opt.apply_gradients(zip(gradients, variables))

            with tf.GradientTape(persistent=True) as tape:
                my_network.selector.trainable = True
                my_network.joint = True
                logits, x_input, data_input = my_network(data, training=True)
                loss_c = loss_fn(labels, logits)

                sigma = 1e-1
                lre = 1e-3
                x_embed = K.square(K.mean(x_input, axis=-1))
                x_embed = tf.expand_dims(x_embed, axis=1)

                sigma_b = (1.0 / (2.0 * K.square(sigma)))
                w = tf.math.scalar_mul(sigma_b, data_input)
                w = tf.transpose(w, [0, 2, 1])

                loss_re = tf.reshape(tf.matmul(w, x_embed, transpose_b=True), [-1])
                loss_re = loss_re + args.slen * (K.log(sigma) + 0.5 * K.square(sigma))
                
                selected_encodes_sum = tf.math.reduce_sum(x_input, axis=1)
                selected_encodes_normalized = tf.math.l2_normalize(
                    selected_encodes_sum, axis=1)
                
                kmeans = KMeans(init="random", n_clusters=args.cls,
                                n_init=10, max_iter=300, random_state=42)
                kmeans.fit(selected_encodes_normalized)
                cluster_labels = kmeans.labels_ + 1
                cluster_labels = cluster_labels * labels

                cl_loss = supervised_nt_xent_loss(
                    selected_encodes_normalized, cluster_labels, temperature=args.tem)
                if tf.math.is_nan(cl_loss):
                    cl_loss = 0.0

                total_loss = cl_loss + args.lam*loss_c + lre*tf.reduce_mean(loss_re)

                variables = my_network.trainable_variables
                gradients = tape.gradient(total_loss, variables)
                opt.apply_gradients(zip(gradients, variables))

                # log every 10 batches.
                if batch_no % 10 == 0:
                    results_file.write("Training loss at step %d -- total_loss: %.4f -- cl_loss: %.4f \n" % (
                        batch_no, float(total_loss), float(cl_loss)))
                    results_file.write("Seen so far: %d samples \n" % ((batch_no + 1) * args.batch))
                    print("Training loss at step %d -- total_loss: %.4f -- cl_loss: %.4f \n" % (
                        batch_no, float(total_loss), float(cl_loss)))
                    print("Seen so far: %d samples" %((batch_no + 1) * args.batch))

    print('saving a model')
    my_network.save(saved_model_dir + 'my_network_model')
    results_file.close()


def leo_test(train_data, valid_data, nv_data_ood, v_data_ood, hpcfs_path, result_dir, args):
    """
    LEO in the testing phase.
    """

    history_file = open(result_dir + 'his_load_predictions_average.txt', 'w')
    history_file.write('p_lr: %f --- p_lambda: %f --- p_tau: %f --- p_tem: %f --- p_cluster: %d \n' %
                        (args.lr, args.lam, args.tau, args.tem, args.cls))
    dv_fpr95_select, dv_auroc_select, dv_aupr_select = 0.0, 0.0, 0.0
    
    for pp_cluster in [1, 3, 5, 7, 9]:
        # load the saved model
        saved_model_dir = hpcfs_path + 'saved_models' + '/' + \
        str(args.lr) + '_' + str(args.lam) + '_' + str(args.tau) + '_' + str(args.tem) + '_' + str(args.cls) + '/'
        if not os.path.exists(saved_model_dir):
            print('cannot find out the saved models')

        selection_network = tf.saved_model.load(saved_model_dir + 'my_network_model')

        """for the training set"""
        train_features = np.array([])
        train_labels = np.array([])
        for x_batch_train, _ in train_data:
            _, selected_encodes, _ = selection_network(x_batch_train, training=False)
            selected_rs = tf.math.reduce_mean(selected_encodes, axis=1)
            train_features = np.append(train_features, selected_rs)

        """for the validation set"""
        val_test_features = np.array([])
        for x_batch_val, _ in valid_data:
            _, selected_encodes, _ = selection_network(x_batch_val, training=False)
            selected_rs = tf.math.reduce_mean(selected_encodes, axis=1)
            val_test_features = np.append(val_test_features, selected_rs)

        """for the ood training set"""
        train_ood_features = np.array([])
        for x_batch_train_ood, _ in nv_data_ood:
            _, selected_encodes, _ = selection_network(x_batch_train_ood, training=False)
            selected_rs = tf.math.reduce_mean(selected_encodes, axis=1)
            train_ood_features = np.append(train_ood_features, selected_rs)

        """for the ood validation set"""
        val_ood_features = np.array([])
        for x_batch_val_ood, _ in v_data_ood:
            _, selected_encodes, _ = selection_network(x_batch_val_ood, training=False)
            selected_rs = tf.math.reduce_mean(selected_encodes, axis=1)
            val_ood_features = np.append(val_ood_features, selected_rs)

        train_features = np.reshape(train_features, [-1, 100])
        val_test_features = np.reshape(val_test_features, [-1, 100])
        train_ood_features = np.reshape(train_ood_features, [-1, 100])
        val_ood_features = np.reshape(val_ood_features, [-1, 100])

        results = get_eval_results_save_features_submit(train_features, val_test_features, train_ood_features,
                                                val_ood_features, train_labels, "SupCon", pp_cluster)
        
        if float(results['dv_auroc']) + float(results['dv_aupr']) > dv_auroc_select + dv_aupr_select:
            dv_fpr95_select = float(results['dv_fpr95'])
            dv_auroc_select = float(results['dv_auroc'])
            dv_aupr_select = float(results['dv_aupr'])

    history_file.write("sc Test vul (ood) average --- fpr95: %.4f --- dvauroc: %.4f --- dvaupr: %.4f \n" %
                    (dv_fpr95_select, dv_auroc_select, dv_aupr_select))
    print("sc Test vul (ood) average --- fpr95: %.4f --- auroc: %.4f --- dvaupr: %.4f" %
        (dv_fpr95_select, dv_auroc_select, dv_aupr_select))

    history_file.close()


if __name__ == '__main__':
    """
    (training), e.g., python leo.py --train --in_data=863 --lr=0.001 --lam=0.001 --tau=0.5 --tem=0.5 --cls=7
    (testing), e.g., python leo.py --in_data=863 --ood_data=287 --lr=0.001 --lam=0.001 --tau=0.5 --tem=0.5 --cls=7
    """
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--in_data', type=str, default='863', help='The in-distribution data used for training the model')
    parser.add_argument('--ood_data', type=str, default='287', help='The out-of-distribution data used for testing the model performance')
    parser.add_argument('--home', type=str, default='leo', help='The folder directory for saving the model')
    parser.add_argument('--train', action='store_true', help='store_true will default to True when the command-line argument is present and vice versa')
    parser.add_argument('--tau', type=float, default=0.5, help='The tau value used in the Gumbel softmax distribution')
    parser.add_argument('--tem', type=float, default=0.5, help='The temperature value used in the innovative cluster-contrastive learning')
    parser.add_argument('--k', type=int, default=10, help='The number of selected statements in each source code sample (function)')
    parser.add_argument('--cls', type=int, default=7, help='The number of clusters used in the innovative cluster-contrastive learning')
    parser.add_argument('--lr', type=float, default=0.001, help='The learning rate used in the training process')
    parser.add_argument("--epo", type=int, default=10, help="The number of epochs used to train the model")
    parser.add_argument("--batch", default=128, type=int, help="The number of data in each mini-batch used to train the mode.")
    parser.add_argument('--lam', type=float, default=0.001, help='The trade-off hyperparameter')
    parser.add_argument('--slen', type=int, default=100, help='The length of sentences in each source code data sample')
    parser.add_argument('--tlen', type=int, default=25, help='The length of tokens in each source code statement')
    parser.add_argument('--emb', type=int, default=150, help='The embedding dimension')
    args = parser.parse_args()

    hpcfs_path = args.home + "/" + args.in_data + "_" + args.ood_data + '/'
    if not os.path.exists(hpcfs_path):
        os.makedirs(hpcfs_path)

    """for in-distribution and out-of-distribution data"""
    cweid_in = 'CWE-' +  args.in_data
    cweid_out = 'CWE-' +  args.ood_data
    data_set = create_data_set_nd(args.slen, args.tlen, cweid_in, cweid_out)
    MAX_NUM_WORDS =  data_set['vocabulary_size']

    x_non_in, x_non_len_in, y_non_in = data_set['cwe_in_data_non_wi'], data_set['cwe_in_data_non_wi_len'], data_set['cwe_in_labels_non']
    x_vul_in, x_vul_len_in, y_vul_in = data_set['cwe_in_data_vul_wi'], data_set['cwe_in_data_vul_wi_len'], data_set['cwe_in_labels_vul']
    x_train, x_train_len, y_train, x_val, x_val_len, y_val = split_data(x_non_in, x_non_len_in, y_non_in, x_vul_in, x_vul_len_in, y_vul_in)
    y_train = np.float32(y_train)

    train_data = tf.data.Dataset.from_tensor_slices((x_train, y_train)).batch(args.batch).shuffle(buffer_size=1024)
    valid_data = tf.data.Dataset.from_tensor_slices((x_val, y_val)).batch(args.batch)

    x_train_ood, x_train_len_ood, y_train_ood = data_set['cwe_ood_data_non_wi'], data_set['cwe_ood_data_non_wi_len'], data_set['cwe_ood_labels_non']
    x_val_ood, x_val_len_ood, y_val_ood = data_set['cwe_ood_data_vul_wi'], data_set['cwe_ood_data_vul_wi_len'], data_set['cwe_ood_labels_vul']

    nv_data_ood = tf.data.Dataset.from_tensor_slices((x_train_ood, y_train_ood)).batch(args.batch)
    v_data_ood = tf.data.Dataset.from_tensor_slices((x_val_ood, y_val_ood)).batch(args.batch)
    
    if args.train:
        leo_train(train_data, hpcfs_path, args, MAX_NUM_WORDS)
    else:
        result_dir = hpcfs_path + args.ood_data + '_predictions_average' + '/'
        if not os.path.exists(result_dir):
            os.makedirs(result_dir)
        leo_test(train_data, valid_data, nv_data_ood, v_data_ood, hpcfs_path, result_dir, args)
