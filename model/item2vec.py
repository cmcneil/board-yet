import tensorflow as tf

flags = tf.app.flags

flags.DEFINE_integer("embedding_dimension", 100, "The dimension of the" +
                     "embedding space")

class Item2Vec(object):
    """This class encapsulates the definition, training and use of a skip-gram
    vector embedding model for recommendations. 
    See Barkan and Koenigstein, 2016, Item2Vec."""
    def __init__(self):
        self.embed_dim = FLAGS.embedding_dimension

        # The vocabulary size (number of games)
        # This should be determined by reading from the training set.
        self.vocab_size = 100
        self.batch_size = 100
        # The number of negatives to sample per batch for SGNS:
        self.num_negatives = 100 
        # The total counts of each item in the dataset. We keep track
        # of this so that we can negatively sample according to the
        # statistics of the training set.
        # This variable must be filled upon reading in training set.
        self.total_item_counts = []

    def init_graphs(self):
        # TODO(carson): Get variables representing batches and labels
        batch_logits, neg_logits = self.build_training_graph(batch, labels)
        self.loss = self.loss_function(batch_logits, neg_logits)
        self.train_node = self.build_optimize_graph(self.loss)

    def build_training_graph(self, batch, labels):
        """Takes in the graph nodes representing a training batch and
        associated labels, and builds the forward training graph, 
        including the embedding itself. Returns nodes representing
        the logits for the positive examples, as well as the logits
        for associated negatives for negative sampling."""
        # We do this because word2vec initializes the weights this way
        init_width = 0.5 / self.embed_dim
        # The actual embedding:
        # The shape of the tensor is weird because we are going to use the
        # "embedding_lookup" function instead of just multipling with a 1-hot.
        emb = tf.Variable(tf.random_uniform([self.vocab_size, self.embed_dim],
                                            -init_width, init_width),
                          name="embedding")
        self.emb = emb
        
        # For training, we actually need to train a softmax classifier.
        # This tensor can be thought of as a complete softmax unit for
        # every possible game. (Each row is a set of weights)
        softmax_w = tf.Variable(tf.zeros([self.vocab_size, self.embed_dim]),
                                name="softmax_weights")
        softmax_b = tf.Variables(tf.zeros([self.vocab_size]), 
                                 name="softmax_bias")

        # Negative sampling for SGNS. We make the assumption of sparsity.
        # On average, randomly sampled games will be negatives.
        labels_reformat = tf.reshape(tf.cast(labels, dtype=tf.int64),
                            [len(training_labels), 1])
        sampled_ids = tf.nn.fixed_unigram_candidate_sampler(
                true_classes=labels_reformat,
                num_true=1,
                num_sampled=self.num_negatives,
                unique=True,
                range_max=self.vocab_size,
                distortion=0.75,
                unigrams=self.total_item_counts)

        batch_embeds = tf.embedding_lookup(emb, batch)
        # Lookup the softmax classifiers for the training batch.
        # I don't particularly like the use of "embedding_lookup",
        # because softmax_w etc. aren't technically embedding
        # matrices. This is apparently the canonical way to do it in TF though.
        batch_sm_w = tf.embedding_lookup(softmax_w, labels)
        batch_sm_b = tf.embedding_lookup(softmax_b, labels)

        # Lookup the softmax classifers for the negative samples.
        neg_sm_w = tf.embedding_lookup(softmax_w, sampled_ids)
        neg_sm_b = tf.embedding_lookup(softmax_b, sampled_ids)

        # Produces a tensor that represents the logits (the arg of the
        # exponential numerator in softmax) for each of the examples
        # in the training batch.
        batch_logits = (tf.reduce_sum(tf.mul(batch_embeds, batch_sm_w), 1) 
                        + batch_sm_b)
        neg_logits = (tf.reduce_sum(tf.mul(batch_embeds, 
                                           neg_sm_w, transpose_b=True)) +
                      tf.reshape(neg_sm_b, [self.num_negatives]))

        return batch_logits, neg_logits

    def loss_function(self, batch_logits, neg_logits):
        batch_xent = tf.sigmoid_cross_entropy_with_logits(
                batch_logits, tf.ones_like(batch_logits))
        neg_xent = tf.sigmoid_cross_entropy_with_logits(
                neg_logits, tf.ones_like(neg_logits))
        nce_loss_tensor = (tf.reduce_sum(batch_logits) + 
                           tf.reduce_sum(neg_logits)) / self.batch_size
        return nce_loss_tensor

    def build_optimize_graph(self, loss):
        items_to_train = float(self.items_per_epoch * self.num_epochs)
        global_step = tf.Variable(0, name="global_step")
        self.global_step = global_step
        learning_rate = 0.001*self.learning_rate
        optimizer = tf.GradientDescentOptimizer(learning_rate)
        train = optimizer.minimize(loss, global_step=self.global_step,
                                   gate_gradients=optimizer.GATE_NONE)
        return train
