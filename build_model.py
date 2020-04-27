"""
Reference : https://github.com/KrishnaswamyLab/SAUCIE/blob/master/README.md

"""


from helper import *
import sklearn.metrics
import os
import tensorflow as tf

def rename_tensor(tensor_op, name):
    """
    rename layer tensor, need for retrive layer information
    """
    tensor_op = tf.identity(tensor_op, name=name)
    return tensor_op


class SAUCIE(object):
    """The SAUCIE model."""

    def __init__(self, input_dim,
        lambda_c=0.1,
        layer_c=0,
        lambda_d=0,
        layers=[1024,512,256,2],
        activation=lrelu,
        learning_rate=0.00009,
        save_folder='',
        limit_gpu_fraction=.3,
        no_gpu=False):
        """
        The SAUCIE model.

        :param input_dim: the dimensionality of the data
        :param lambda_b: the coefficient for the MMD regularization
        :param lambda_c: the coefficient for the ID regularization
        :param layer_c: the index of layer_dimensions that ID regularization should be applied to (usually len(layer_dimensions)-2)
        :param lambda_d: the coefficient for the intracluster distance regularization
        :param activation: the nonlinearity to use in the hidden layers
        :param loss: the loss function to use, one of 'mse' or 'bce'
        :param learning_rate: the learning_rate to use while training
        :param restore_folder: string of the directory where a previous model is saved, if present will return a new Python object
                               with the old SAUCIE tensorflow graph
        :param save_folder: string of the directory to save SAUCIE to by default when save() is called
        """
        self.input_dim = input_dim
        self.lambda_c = lambda_c
        self.layer_c = layer_c
        self.lambda_d = lambda_d
        self.activation = activation
        self.learning_rate = learning_rate
        self.save_folder = save_folder
        self.iteration = 0
        self.layers = layers

        self.x = tf.placeholder(tf.float32, shape=[None, input_dim], name='x')
        self.y = tf.placeholder(tf.float32, shape=[None, input_dim], name='y')
        self.batches = tf.placeholder(tf.int32, shape=[None], name='batches')
        self.is_training = tf.placeholder(tf.bool, shape=[], name='is_training')
        self.learning_rate_tensor = tf.placeholder(tf.float32, shape=[], name='learning_rate_tensor')

        self._build_model()
        self.init_sessGPU(limit_gpu_fraction, no_gpu)

        self.graph_init(self.sess)

    def init_sessGPU(self, limit_gpu_fraction=.1, no_gpu=False):
        """
        Initialize a tensorflow session for SAUCIE.

        :param limit_gpu_fraction: float percentage of the avaiable gpu to use
        :param no_gpu: bool for whether or not to use the gpu if available
        """
        if no_gpu:
            os.environ['CUDA_VISIBLE_DEVICES'] = ''
            config = tf.ConfigProto(device_count={'GPU': 0})
            self.sess = tf.Session(config=config)
        elif limit_gpu_fraction:
            gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=limit_gpu_fraction)
            config = tf.ConfigProto(gpu_options=gpu_options)
            self.sess = tf.Session(config=config)
        else:
            self.sess = tf.Session()

    def _build_model(self):
        """Coordinate the building of each part of SAUCIE."""
        self._model_archtecture()

        self._all_losses()

        self._optimizater()

    def _model_archtecture(self):
        """Construct the layers of SAUCIE."""

        print('============================Building layers=============================================================')

        if self.lambda_c:
            h1 = tf.layers.dense(self.x, self.layers[0], activation=lrelu, name='encoder0', use_bias=True)
            h1 = rename_tensor(h1, 'encoderlayer0')

            h2 = tf.layers.dense(h1, self.layers[1], activation=lrelu, name='encoder1', use_bias=True)
            h2 = rename_tensor(h2, 'encoderlayer1')

            h3 = tf.layers.dense(h2, self.layers[2], activation=lrelu, name='encoder2', use_bias=True)
            h3 - rename_tensor(h3, 'encoderlayer2')


            self.embedded = tf.layers.dense(h3, self.layers[3], activation=tf.identity, name='embedding', use_bias=True)
            self.embedded = rename_tensor(self.embedded, 'embeddings')

            h5 = tf.layers.dense(self.embedded, self.layers[2], activation=lrelu, name='decoder0', use_bias=True)
            h5 = rename_tensor(h5, 'decoderlayer0')

            h6 = tf.layers.dense(h5, self.layers[1], activation=lrelu, name='decoder1', use_bias=True)
            h6 = rename_tensor(h6, 'decoderlayer1')

            h7 = tf.layers.dense(h6, self.layers[0], activation=tf.nn.relu, name='decoder2', use_bias=True)
            h7 = rename_tensor(h7, 'layer_c')

            print('++++++++++++++++++++++++++ input dimension : ', self.input_dim)
            self.reconstructed = tf.layers.dense(h7, self.input_dim, activation=tf.identity, name='recon', use_bias=True)
            self.reconstructed = rename_tensor(self.reconstructed, 'output1')

        else:
            h1 = tf.layers.dense(self.x, self.layers[0], activation=lrelu, name='encoder0')

            h2 = tf.layers.dense(h1, self.layers[1], activation=tf.nn.sigmoid, name='encoder1')

            h3 = tf.layers.dense(h2, self.layers[2], activation=lrelu, name='encoder2')

            self.embedded = tf.layers.dense(h3, self.layers[3], activation=tf.identity, name='embedding')
            self.embedded = rename_tensor(self.embedded, 'embeddings')

            h5 = tf.layers.dense(self.embedded, self.layers[2], activation=lrelu, name='decoder0')

            h6 = tf.layers.dense(h5, self.layers[1], activation=lrelu, name='decoder1')

            h7 = tf.layers.dense(h6, self.layers[0], activation=lrelu, name='decoder2')
            h7 = rename_tensor(h7, 'layer_c')

            # print('++++++++++++++++++++++++++ input dimension : ', self.input_dim)
            self.reconstructed = tf.layers.dense(h7, self.input_dim, activation=tf.identity, name='recon')
            self.reconstructed = rename_tensor(self.reconstructed, 'output1')

    def _all_losses(self):
        """Build all the loss ops for the network."""
        self.loss_recon = 0.

        with tf.variable_scope('reconstruction'):
            self._build_reconstruction_loss(self.reconstructed, self.x)

        if self.lambda_c:
            print('==============')
            with tf.variable_scope('clustering'):
                self.loss_c = 0

                act = tbn('layer_c:0')
                act = act / tf.reduce_max(act)
                print('act',act)
                self._build_reg_c(act)

        if self.lambda_d:
            with tf.variable_scope('intracluster_distances'):
                self._build_reg_d(act)

        self._build_total_loss()

    def _optimizater(self, norm_clip=5.):
        """Build all the optimization ops for the network."""
        opt = tf.train.AdamOptimizer(self.learning_rate)
        self.train_op = opt.minimize(self.loss, name='train_op')

    def _build_reconstruction_loss(self, reconstructed, y):
        """
        Build the reconstruction loss part of the network if batch correction isn't being performed.

        :param reconstructed: the tensorflow op that was output1 by the decoder
        :param y: the tensorflow op for the target
        """
        self.loss_recon = tf.reduce_mean((self.reconstructed - y)**2)

        self.loss_recon = rename_tensor(self.loss_recon, 'loss_recon')
        tf.add_to_collection('losses', self.loss_recon)

    def _build_reconstruction_loss_mmd(self, reconstructed, y):
        """
        Build the reconstruction loss part of the network if batch correction is being performed.

        :param reconstructed: the tensorflow op that was output1 by the decoder
        :param y: the tensorflow op for the target
        """
        refrecon = tf.boolean_mask(reconstructed, tf.equal(self.batches, 0))
        refy = tf.boolean_mask(y, tf.equal(self.batches, 0))
        l = (refy - refrecon)**2
        self.loss_recon += tf.reduce_mean(l)


        nonrefrecon = tf.boolean_mask(reconstructed, tf.equal(self.batches, 1))
        nonrefy = tf.boolean_mask(y, tf.equal(self.batches, 1))

        mean1, var1 = tf.nn.moments(nonrefrecon, 0)
        mean2, var2 = tf.nn.moments(nonrefy, 0)
        l = ( ((nonrefrecon - mean1) / (tf.sqrt(var1+1e-6)+1e-6)) - ((nonrefy - mean2) / (tf.sqrt(var2+1e-6)+1e-6)) )**2

        self.loss_recon += .01*tf.reduce_mean(l)

        self.loss_recon = rename_tensor(self.loss_recon, 'loss_recon')
        tf.add_to_collection('losses', self.loss_recon)

    def _build_reg_d(self, act):
        """
        Calculate the intracluster distances in the original data given binary-like codes.

        :param act: the codes that will be binarized and used to determine cluster assignment
        """
        out = self._pairwise_dists(act, act)
        same_cluster = self._gaussian_kernel_matrix(out)
        same_cluster = same_cluster - tf.reduce_min(same_cluster)
        same_cluster = same_cluster / tf.reduce_max(same_cluster)

        dists = self._pairwise_dists(self.x, self.x)
        dists = tf.sqrt(dists + 1e-3)

        intracluster_distances = dists * same_cluster
        intracluster_distances = tf.reduce_mean(intracluster_distances)

        self.loss_d = self.lambda_d * intracluster_distances
        self.loss_d = rename_tensor(self.loss_d, 'loss_d')
        tf.add_to_collection('losses', self.loss_d)

    def _build_reg_c(self, act):
        """Build the tensorflow graph for the ID regularization."""
        # sum down neurons
        p = tf.reduce_sum(act, axis=0, keep_dims=True)
        # normalize neuron sums
        normalized = p / tf.reduce_sum(p)

        self.loss_c += self.lambda_c * tf.reduce_sum(-normalized * tf.log(normalized + 1e-9))

        self.loss_c = rename_tensor(self.loss_c, 'loss_c')
        tf.add_to_collection('losses', self.loss_c)


    def _build_total_loss(self):
        """Collect all of the losses together."""
        self.loss = 0
        for l in tf.get_collection('losses'):
            self.loss += l
        self.loss = rename_tensor(self.loss, 'loss')
        print('build total loss ================: ',self.loss)

    def _gaussian_kernel_matrix(self, dist):
        """Multi-scale RBF kernel."""
        sigmas = [1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1, 5, 10, 15, 20, 25, 30, 35, 100, 1e3, 1e4, 1e5, 1e6]

        beta = 1. / (2. * (tf.expand_dims(sigmas, 1)))

        s = tf.matmul(beta, tf.reshape(dist, (1, -1)))

        return tf.reshape(tf.reduce_sum(tf.exp(-s), 0), tf.shape(dist)) / len(sigmas)

    def _pairwise_dists(self, x1, x2):
        """Helper function to calculate pairwise distances between tensors x1 and x2."""
        r1 = tf.reduce_sum(x1 * x1, 1, keep_dims=True)
        r2 = tf.reduce_sum(x2 * x2, 1, keep_dims=True)

        D = r1 - 2 * tf.matmul(x1, tf.transpose(x2)) + tf.transpose(r2)
        print('D==========',D)
        return D

    def graph_init(self, sess=None):
        """
        Initialize the tensorflow graph that's been created.

        :param sess: the session to use while initializing, if different from SAUCIE's sess member
        """
        if not sess: sess = self.sess

        self.saver = tf.train.Saver(tf.global_variables(), max_to_keep=1)

        sess.run(tf.global_variables_initializer())

    def save(self, iteration=None, saver=None, sess=None, folder=None):
        """
        Save the current state of SAUCIE.

        :param iteration: the number of training steps SAUCIE has taken, which distinguishes the saved states
                          throughout training
        :param saver: the saver instance to use
        :param sess: the session to save
        :param folder: the location to save SAUCIE's state to
        """
        if not iteration: iteration = self.iteration
        if not saver: saver = self.saver
        if not sess: sess = self.sess
        if not folder: folder = self.save_folder

        savefile = os.path.join(folder, 'SAUCIE')
        saver.save(sess, savefile, write_meta_graph=True)
        print("Model saved to {}".format(savefile))

        saver2 = tf.train.Saver(write_version=tf.train.SaverDef.V1)
        saver2.save(sess, "crfmodel.ckpt", global_step=0)

    def get_loss_names(self):
        """Return the strings of the loss names in the order they're printed during training."""
        losses = [tns.name[:-2].replace('loss_', '').split('/')[-1] for tns in tf.get_collection('losses')]
        return "Losses: {}".format(' '.join(losses))

    def train(self, load, steps, batch_size=32):
        """
        Train SAUCIE.

        :param load: the loader object to yield batches from
        :param steps: the number of steps to train for
        :param batch_size: the number of points to train on in each step
        """
        start = self.iteration
        losschange = 100000

        while (self.iteration - start) < steps:
            self.iteration += 1

            batch = load.next_batch(batch_size=batch_size)
            #print(batch)
            feed = {tbn('x:0'): batch[0],
                    tbn('y:0'): batch[0],
                     tbn('is_training:0'): True,
                    tbn('learning_rate_tensor:0'): self.learning_rate}

            if len(batch) == 2:
                feed[tbn('batches:0')] = batch[1]

            # if using batch-correction, must have labels
            if (self.lambda_b and len(batch) < 2):
                raise Exception("If using lambda_b (batch correction), you must provide each point's batch as a label")

            ops = [obn('train_op')]
            #print(feed)
            loss =  self.sess.run(ops, feed_dict=feed)
            print(loss)
            loss = self.get_loss(load)
            l = loss.split(' ')
            print('epoch : {}=========== loss : {}'.format(self.iteration,l[1]))
            if float(l[1]) < 0.08:
                break

    def get_loss(self, load, batch_size=32):
        """
        Get the current losses over the dataset.

        :param load: the loader object to iterate over
        """
        losses = None

        for i, batch in enumerate(load.iter_batches(batch_size=batch_size)):

            feed = {tbn('x:0'): batch[0],
                    tbn('y:0'): batch[0],
                    tbn('is_training:0'): False}
            if len(batch) == 2:
                feed[tbn('batches:0')] = batch[1]

            batch_losses = self.sess.run(tf.get_collection('losses'), feed_dict=feed)
            #print(batch_losses)
            if not losses:
                losses = batch_losses
            else:
                losses = [loss + batch_loss for loss, batch_loss in zip(losses, batch_losses)]

        losses = [loss / float(i + 1) for loss in losses]
        print(losses)
        lstring = ' '.join(['{:.3f}'.format(loss) for loss in losses])

        return lstring

    def get_layer(self, load, name):
        """
        Get the actual values in array_like form from an abstract tensor.

        :param load: the loader object to iterate over
        :param name: the name of the tensor to evaluate for each point
        """
        tensor_name = "{}:0".format(name)
        tensor = tbn(tensor_name)
        # print(',,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,',tensor)
        layer = []
        labels = []
        for batch in load.iter_batches():

            feed = {tbn('x:0'): batch[0],
                    tbn('y:0'): batch[0],
                    tbn('is_training:0'): False}
            if len(batch) == 2:
                feed[tbn('batches:0')] = batch[1]
                labels.append(batch[1])

            [act] = self.sess.run([tensor], feed_dict=feed)
            # print('act ===========================',act.shape)
            layer.append(act)

        layer = np.concatenate(layer, axis=0)

        if labels:
            labels = np.concatenate(labels, axis=0)
            return layer, labels
        else:
            return layer

    def get_cluster_merging(self, embedding, clusters):
        if len(np.unique(clusters))==1: return clusters

        clusters = clusters - clusters.min()
        clusts_to_use = np.unique(clusters)
        mmdclusts = np.zeros((len(clusts_to_use), len(clusts_to_use)))
        for i1, clust1 in enumerate(clusts_to_use):
            for i2, clust2 in enumerate(clusts_to_use[i1 + 1:]):
                ei = embedding[clusters == clust1]
                ej = embedding[clusters == clust2]
                ri = list(range(ei.shape[0])); np.random.shuffle(ri); ri = ri[:1000];
                rj = list(range(ej.shape[0])); np.random.shuffle(rj); rj = rj[:1000];
                ei = ei[ri, :]
                ej = ej[rj, :]

                k1 = sklearn.metrics.pairwise.pairwise_distances(ei, ei)
                k2 = sklearn.metrics.pairwise.pairwise_distances(ej, ej)
                k12 = sklearn.metrics.pairwise.pairwise_distances(ei, ej)

                mmd = 0
                for sigma in [.01, .1, 1., 10.]:
                    k1_ = np.exp(- k1 / (sigma**2))
                    k2_ = np.exp(- k2 / (sigma**2))
                    k12_ = np.exp(- k12 / (sigma**2))

                    mmd += calculate_mmd(k1_, k2_, k12_)
                mmdclusts[i1, i1 + i2 + 1] = mmd
                mmdclusts[i1 + i2 + 1, i1] = mmd

        clust_to = {}
        for i1 in range(mmdclusts.shape[0]):
            for i2 in range(mmdclusts.shape[1]):
                argmin1 = np.argsort(mmdclusts[i1, :])[1]
                argmin2 = np.argsort(mmdclusts[i2, :])[1]
                if argmin1 == (i1 + i2) and argmin2 == i1 and i2 > i1:
                    clust_to[i2] = i1


        for c in clust_to:
            mask = clusters == c
            clusters[mask.tolist()] = clust_to[c]

        clusts_to_use_map = [c for c in clusts_to_use.tolist() if c not in clust_to]
        clusts_to_use_map = {c:i for i,c in enumerate(clusts_to_use_map)}

        for c in clusts_to_use_map:
            mask = clusters==c
            clusters[mask.tolist()] = clusts_to_use_map[c]


        return clusters

    def get_clusters(self, load, binmin=100, max_clusters=1000, verbose=True):
        """
        Get cluster assignments from the ID regularization layer.

        :param load: the loader object to iterate over
        :param binmin: points in a cluster of less than this many points will be assigned the unclustered "-1" label
        :param max_clusters: going through the clusters can take a long time, so optionally abort any attempt to go
                             through more than a certain number of clusters
        :param verbose: whether or not to print the results of the clustering
        """
        acts = self.get_layer(load, 'layer_c')
        print(acts)
        print('acts shape',acts.shape)

        if isinstance(acts, list) or isinstance(acts, tuple):
            acts = acts[0]

        acts = acts / acts.max()

        binarized = np.where(acts > .000001, 1, 0)

        unique_rows, counts = np.unique(binarized, axis=0, return_counts=True)
        unique_rows = unique_rows[counts > binmin]

        num_clusters = unique_rows.shape[0]
        if num_clusters > max_clusters:
            print("Too many clusters ({}) to go through...".format(num_clusters))
            return num_clusters, np.zeros(acts.shape[0])

        num_clusters = 0
        rows_clustered = 0
        clusters = -1 * np.ones(acts.shape[0])
        for i, row in enumerate(unique_rows):
            rows_equal_to_this_code = np.where(np.all(binarized == row, axis=1))[0]

            clusters[rows_equal_to_this_code] = num_clusters
            num_clusters += 1
            rows_clustered += rows_equal_to_this_code.shape[0]

        embedding = self.get_embedding(load)
        clusters = self.get_cluster_merging(embedding, clusters)
        num_clusters = len(np.unique(clusters))

        if verbose:
            print("---- Num clusters: {} ---- Percent clustered: {:.3f} ----".format(num_clusters, 1. * rows_clustered / clusters.shape[0]))


        return num_clusters, clusters

    def get_embedding(self, load):
        """Return the embedding layer."""
        embedding = self.get_layer(load, 'embeddings')
        return embedding

    def get_reconstruction(self, load):
        """Return the reconstruction layer."""
        reconstruction = self.get_layer(load, 'output1')
        return reconstruction

    def get_allLayer(self,load):
        """"
        return all layers value
        """
        encoder0L = self.get_layer(load, 'encoderlayer0')
        encoder1L = self.get_layer(load, 'encoderlayer1')
        encoder2L = self.get_layer(load, 'encoderlayer2')
        embeddingsL = self.get_layer(load, 'embeddings')
        decoder0L = self.get_layer(load, 'decoderlayer0')
        decoder1L = self.get_layer(load, 'encoderlayer1')
        layer_cL = self.get_layer(load, 'layer_c')

        return encoder0L,encoder1L,encoder2L,embeddingsL,decoder0L,decoder1L,layer_cL

    def get_allweights(self):
        """

        :return: each layer weights
        """

        with tf.variable_scope("encoder0", reuse=True):
            encoder0W = tf.get_variable("kernel")

        with tf.variable_scope("encoder1", reuse=True):
            encoder1W = tf.get_variable("kernel")

        with tf.variable_scope("encoder2", reuse=True):
            encoder2W = tf.get_variable("kernel")

        with tf.variable_scope("embedding", reuse=True):
            embeddingW = tf.get_variable("kernel")

        with tf.variable_scope("decoder0", reuse=True):
            decoder0W = tf.get_variable("kernel")

        with tf.variable_scope("decoder1", reuse=True):
            decoder1W = tf.get_variable("kernel")

        with tf.variable_scope("decoder2", reuse=True):
            layer_cW = tf.get_variable("kernel")

        with tf.variable_scope("recon", reuse=True):
            output1W = tf.get_variable("kernel")

        return encoder0W,encoder1W,encoder2W,embeddingW,decoder0W,decoder1W,layer_cW,output1W



























