from time import time

import tensorflow as tf
from sklearn.base import BaseEstimator,TransformerMixin
#tf.enable_eager_execution()
import os

import numpy as np

embedding_size = 100

class RNN_Model(BaseEstimator,TransformerMixin):
    def __init__(self, vocab_size,
                 embedding,
                 embedding_dim=50,
                 hidden_size=30,
                 batch_size = 128,
                 is_training = True,
                 optimizer_type = 'adam',
                 learning_rate = 0.0001,
                 epoch = 50,
                 verbose = True,
                 log_dir = 'log',
                 max_length=20):
        super().__init__()
        self.vocab_size = vocab_size
        self.embedding = embedding
        self.embedding_dim = embedding_dim
        self.hidden_size = hidden_size
        self.batch_size = batch_size
        self.is_training = is_training
        self.optimizer_type = optimizer_type
        self.learning_rate = learning_rate
        self.epoch = epoch
        self.verbose = verbose
        self.log_dir = log_dir
        self.max_length = max_length
        self._init_graph()
        self.display_tensorboard()

    def _init_graph(self):
        self.graph = tf.Graph()
        with self.graph.as_default():

            self.ad_text = tf.placeholder(dtype=tf.int32, shape=[None,self.max_length])
            self.key_words = tf.placeholder(dtype=tf.int32, shape=[None,5])
            self.targets = tf.placeholder(dtype=tf.int32,shape=[None,self.max_length])

            targets_onehot = tf.one_hot(indices=self.targets,depth=self.vocab_size)

            self.weights = self._init_weigths()

            self.ad_inputs = tf.nn.embedding_lookup(self.weights['word_embeddings'],self.ad_text)
            self.keywords_inputs = tf.nn.embedding_lookup(self.weights['word_embeddings'],self.key_words)


            rnn_cell_left = tf.nn.rnn_cell.BasicRNNCell(num_units=self.hidden_size)
            self.initial_state_left = rnn_cell_left.zero_state(batch_size=self.batch_size,dtype=tf.float64)

            print('initial_state_left: ', tf.shape(self.initial_state_left))
            print('keywords_inputs: ', tf.shape(self.keywords_inputs))


            #If time_major == False (default), this will be a Tensor shaped: [batch_size, max_time, cell.output_size].
            self.outputs, self.state = tf.nn.dynamic_rnn(rnn_cell_left,self.keywords_inputs,
                                                         initial_state = self.initial_state_left,
                                                         dtype = tf.float64,
                                                         time_major = False,
                                                         scope='keyword_rnn')
            print('outputs: ', tf.shape(self.outputs))
            print('state: ', tf.shape(self.state))


            rnn_cell_right = tf.nn.rnn_cell.BasicRNNCell(self.hidden_size)

            #[batch_size,hidden_size]
            # initial_state_right = rnn_cell_right.zero_state(self.batch_size,dtype=tf.float32)
            initial_state_right = self.outputs[:,-1,:]

            self.y_outputs, self.y_state = tf.nn.dynamic_rnn(rnn_cell_right,self.ad_inputs,
                                                             initial_state = initial_state_right,
                                                             dtype = tf.float64,
                                                             time_major = False,
                                                             scope = 'ad_text_rnn')

            y_outputs = tf.reshape(self.y_outputs,shape=[-1, self.hidden_size])

            self.prediction = tf.layers.dense(inputs=y_outputs, units=self.vocab_size,activation=tf.identity if self.is_training else tf.nn.softmax)

            self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=self.prediction,labels=targets_onehot))


            if self.optimizer_type == "adam":
                self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate, beta1=0.9, beta2=0.999,
                                                        epsilon=1e-8).minimize(self.loss)
            elif self.optimizer_type == "adagrad":
                self.optimizer = tf.train.AdagradOptimizer(learning_rate=self.learning_rate,
                                                           initial_accumulator_value=1e-8).minimize(self.loss)
            elif self.optimizer_type == "gd":
                self.optimizer = tf.train.GradientDescentOptimizer(learning_rate=self.learning_rate).minimize(self.loss)
            elif self.optimizer_type == "momentum":
                self.optimizer = tf.train.MomentumOptimizer(learning_rate=self.learning_rate, momentum=0.95).minimize(
                    self.loss)

            #init
            self.saver = tf.train.Saver()
            init = tf.global_variables_initializer()
            self.sess = tf.Session()
            self.sess.run(init)

            # number of params
            total_parameters = 0
            for variable in self.weights.values():
                shape = variable.get_shape()
                variable_parameters = 1
                for dim in shape:
                    variable_parameters *= dim.value
                total_parameters += variable_parameters
            if self.verbose > 0:
                print("#params: %d" % total_parameters)

    def _init_weigths(self):
        weights = {}

        self.embedding = tf.Variable(initial_value=self.embedding, trainable=True)

        weights['word_embeddings'] = tf.Variable(initial_value=self.embedding, trainable=True,name='word_embeddings')

        return weights

    def get_batch(self,ad_text,key_words,targets,batch_size,index):
        start = index * batch_size
        end = (index + 1) * batch_size
        end = end if end < len(targets) else len(targets)
        return ad_text[start:end],key_words[start:end],targets[start:end]

    def fit_on_epoch(self,ad_text,key_words,targets):

        def fit_on_batch(ad_text,key_words,targets):
            feed_dict = {self.ad_text:ad_text,
                         self.key_words:key_words,
                         self.targets:targets}

            loss,opt = self.sess.run([self.loss,self.optimizer],feed_dict=feed_dict)
            return np.array(tf.reduce_sum(loss))

        self.shuffle_in_unison_scary(ad_text, key_words,targets)
        total_batch = int(len(targets) / self.batch_size)
        loss=[]
        for i in range(total_batch):
            Xi_batch, Xv_batch, y_batch = self.get_batch(ad_text, key_words, targets,self.batch_size, i)
            loss.append(fit_on_batch(Xi_batch, Xv_batch, y_batch))
        return sum(loss)

    def fit(self, ad_text,key_words,targets,ad_text_valid=None, key_words_valid=None,
            targets_valid=None, early_stopping=False, refit=False):

        has_valid = ad_text_valid is not None

        for epoch in range(self.epoch):
            t1 = time()
            self.fit_on_epoch(ad_text, key_words, targets)

            if has_valid:
                loss = self.fit_on_epoch(ad_text_valid, key_words_valid, targets_valid)
            t2 = time()
            print("epoch: ",epoch," time: ",t2-t1," validation loss: ",loss)


    def shuffle_in_unison_scary(self, *args):
        rng_state = np.random.get_state()
        for data in args:
            np.random.shuffle(data)
            np.random.set_state(rng_state)



    def attention(inputs, attention_size, time_major=False, return_alphas=False):


        if isinstance(inputs, tuple):
            # In case of Bi-RNN, concatenate the forward and the backward RNN outputs.
            inputs = tf.conkecat(inputs, 2)

        if time_major:
            # (T,B,D) => (B,T,D)
            inputs = tf.array_ops.transpose(inputs, [1, 0, 2])

        hidden_size = inputs.shape[2].value  # D value - hidden size of the RNN layer

        # Trainable parameters
        w_omega = tf.Variable(tf.random_normal([hidden_size, attention_size], stddev=0.1))
        b_omega = tf.Variable(tf.random_normal([attention_size], stddev=0.1))
        u_omega = tf.Variable(tf.random_normal([attention_size], stddev=0.1))

        with tf.name_scope('v'):
            # Applying fully connected layer with non-linear activation to each of the B*T timestamps;
            #  the shape of `v` is (B,T,D)*(D,A)=(B,T,A), where A=attention_size
            v = tf.tanh(tf.tensordot(inputs, w_omega, axes=1) + b_omega)

        # For each of the timestamps its vector of size A from `v` is reduced with `u` vector
        vu = tf.tensordot(v, u_omega, axes=1, name='vu')  # (B,T) shape
        alphas = tf.nn.softmax(vu, name='alphas')         # (B,T) shape

        # Output of (Bi-)RNN is reduced with attention vector; the result has (B,D) shape
        output = tf.reduce_sum(inputs * tf.expand_dims(alphas, -1), 1)

        if not return_alphas:
            return output
        else:
            return output, alphas



    def predict(self,ad_text,key_words,targets):
        """
        :param Xi: list of list of feature indices of each sample in the dataset
        :return: predicted probability of each sample
        """
        # dummy y
        feed_dict = {self.ad_text: ad_text,
                     self.targets: targets,
                     self.key_words:key_words}
        loss = self.sess.run([self.loss], feed_dict=feed_dict)

        return loss

    def display_tensorboard(self):
        # tf.summary.FileWriter is not compatible with eager execution. Use tf.contrib.summary instead
        writer = tf.summary.FileWriter(os.path.expanduser(self.log_dir), self.sess.graph)
        writer.close()


if __name__ == '__main__':
    embedding_weights = np.random.random(size=(1000,50))
    ad_text = np.random.randint(low=0,high=999,size=(1000,20))
    key_words = np.random.randint(low=0,high=999,size=(1000,5))
    y_predict = np.random.randint(low=0,high=999,size=(1000,20))


    ad_text_valid = np.random.randint(low=0,high=999,size=(500,20))
    key_words_valid = np.random.randint(low=0,high=999,size=(500,5))
    y_predict_valid = np.random.randint(low=0,high=999,size=(500,20))

    model = RNN_Model( vocab_size=1000,
                       embedding=embedding_weights,
                       embedding_dim=100,
                       hidden_size=30,
                       batch_size = 128,
                       is_training = True,
                       optimizer_type = 'adam',
                       learning_rate = 0.0001,
                       epoch = 50)
    model.fit(ad_text,key_words,targets=y_predict,ad_text_valid=ad_text_valid, key_words_valid=key_words_valid,
              targets_valid=y_predict_valid)


