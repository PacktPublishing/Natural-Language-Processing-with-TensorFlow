import tensorflow as tf
import numpy as np
import math
sentence_cursors = None
tot_sentences = None
src_max_sent_length, tgt_max_sent_length = 0, 0
src_dictionary, tgt_dictionary = {}, {}
src_reverse_dictionary, tgt_reverse_dictionary = {},{}
train_inputs, train_outputs = None, None
embedding_size = None # Dimension of the embedding vector.
vocabulary_size = None
def define_data_and_hyperparameters(
        _tot_sentences, _src_max, _tgt_max, _src_dict, _tgt_dict,
        _src_rev_dict, _tgt_rev_dict, _tr_inp, _tr_out, _emb_size, _vocab_size):
    global tot_sentences, sentence_cursors
    global src_max_sent_length, tgt_max_sent_length
    global src_dictionary, tgt_dictionary
    global src_reverse_dictionary, tgt_reverse_dictionary
    global train_inputs, train_outputs
    global embedding_size, vocabulary_size

    embedding_size = _emb_size
    vocabulary_size = _vocab_size
    src_max_sent_length, tgt_max_sent_length = _src_max, _tgt_max

    src_dictionary = _src_dict
    tgt_dictionary = _tgt_dict

    src_reverse_dictionary = _src_rev_dict
    tgt_reverse_dictionary = _tgt_rev_dict

    train_inputs = _tr_inp
    train_outputs = _tr_out

    tot_sentences = _tot_sentences
    sentence_cursors = [0 for _ in range(tot_sentences)]


def generate_batch_for_word2vec(batch_size, window_size, is_source):
    # window_size is the amount of words we're looking at from each side of a given word
    # creates a single batch
    global sentence_cursors
    global src_dictionary, tgt_dictionary
    global train_inputs, train_outputs
    span = 2 * window_size + 1  # [ skip_window target skip_window ]

    batch = np.ndarray(shape=(batch_size, span - 1), dtype=np.int32)
    labels = np.ndarray(shape=(batch_size, 1), dtype=np.int32)
    # e.g if skip_window = 2 then span = 5
    # span is the length of the whole frame we are considering for a single word (left + word + right)
    # skip_window is the length of one side

    sentence_ids_for_batch = np.random.randint(0, tot_sentences, batch_size)

    for b_i in range(batch_size):
        sent_id = sentence_ids_for_batch[b_i]

        if is_source:
            buffer = train_inputs[sent_id, sentence_cursors[sent_id]:sentence_cursors[sent_id] + span]
        else:
            buffer = train_outputs[sent_id, sentence_cursors[sent_id]:sentence_cursors[sent_id] + span]
        assert buffer.size == span, 'Buffer length (%d), Current data index (%d), Span(%d)' % (
        buffer.size, sentence_cursors[sent_id], span)
        # If we only have EOS tokesn in the sampled text, we sample a new one
        if is_source:
            while np.all(buffer == src_dictionary['</s>']):
                # reset the sentence_cursors for that cap_id
                sentence_cursors[sent_id] = 0
                # sample a new cap_id
                sent_id = np.random.randint(0, tot_sentences)
                buffer = train_inputs[sent_id, sentence_cursors[sent_id]:sentence_cursors[sent_id] + span]
        else:
            while np.all(buffer == tgt_dictionary['</s>']):
                # reset the sentence_cursors for that cap_id
                sentence_cursors[sent_id] = 0
                # sample a new cap_id
                sent_id = np.random.randint(0, tot_sentences)
                buffer = train_outputs[sent_id, sentence_cursors[sent_id]:sentence_cursors[sent_id] + span]

        # fill left and right sides of batch
        batch[b_i, :window_size] = buffer[:window_size]
        batch[b_i, window_size:] = buffer[window_size + 1:]

        labels[b_i, 0] = buffer[window_size]

        # increase the corresponding index
        if is_source:
            sentence_cursors[sent_id] = (sentence_cursors[sent_id] + 1) % (src_max_sent_length - span)
        else:
            sentence_cursors[sent_id] = (sentence_cursors[sent_id] + 1) % (tgt_max_sent_length - span)

    assert batch.shape[0] == batch_size and batch.shape[1] == span - 1
    return batch, labels


def print_some_batches():
    global sentence_cursors, tot_sentences
    global src_reverse_dictionary

    for window_size in [1, 2]:
        sentence_cursors = [0 for _ in range(tot_sentences)]
        batch, labels = generate_batch_for_word2vec(batch_size=8, window_size=window_size, is_source=True)
        print('\nwith window_size = %d:' % (window_size))
        print('    batch:', [[src_reverse_dictionary[bii] for bii in bi] for bi in batch])
        print('    labels:', [src_reverse_dictionary[li] for li in labels.reshape(8)])

    sentence_cursors = [0 for _ in range(tot_sentences)]

batch_size, window_size = None, None
valid_size, valid_window, valid_examples = None, None, None
num_sampled = None

train_dataset, train_labels = None, None
valid_dataset = None

softmax_weights, softmax_biases = None, None

loss, optimizer, similarity, normalized_embeddings = None, None, None, None

def define_word2vec_tensorflow(batch_size):

    global embedding_size, window_size
    global	valid_size, valid_window, valid_examples
    global num_sampled
    global train_dataset, train_labels
    global valid_dataset
    global softmax_weights, softmax_biases
    global loss, optimizer, similarity
    global vocabulary_size, embedding_size
    global normalized_embeddings


    window_size = 2  # How many words to consider left and right.
    # We pick a random validation set to sample nearest neighbors. here we limit the
    # validation samples to the words that have a low numeric ID, which by
    # construction are also the most frequent.
    valid_size = 20  # Random set of words to evaluate similarity on.
    valid_window = 100  # Only pick dev samples in the head of the distribution.
    # pick 16 samples from 100
    valid_examples = np.array(np.random.randint(0, valid_window, valid_size // 2))
    valid_examples = np.append(valid_examples, np.random.randint(1000, 1000 + valid_window, valid_size // 2))
    num_sampled = 32  # Number of negative examples to sample.

    tf.reset_default_graph()

    # Input data.
    train_dataset = tf.placeholder(tf.int32, shape=[batch_size, 2 * window_size])
    train_labels = tf.placeholder(tf.int32, shape=[batch_size, 1])
    valid_dataset = tf.constant(valid_examples, dtype=tf.int32)

    # Variables.
    # embedding, vector for each word in the vocabulary
    embeddings = tf.Variable(tf.random_uniform([vocabulary_size, embedding_size], -1.0, 1.0, dtype=tf.float32))
    softmax_weights = tf.Variable(tf.truncated_normal([vocabulary_size, embedding_size],
                                                      stddev=1.0 / math.sqrt(embedding_size), dtype=tf.float32))
    softmax_biases = tf.Variable(tf.zeros([vocabulary_size], dtype=tf.float32))

    # Model.
    # Look up embeddings for inputs.
    # this might efficiently find the embeddings for given ids (traind dataset)
    # manually doing this might not be efficient given there are 50000 entries in embeddings
    stacked_embedings = None
    print('Defining %d embedding lookups representing each word in the context' % (2 * window_size))
    for i in range(2 * window_size):
        embedding_i = tf.nn.embedding_lookup(embeddings, train_dataset[:, i])
        x_size, y_size = embedding_i.get_shape().as_list()
        if stacked_embedings is None:
            stacked_embedings = tf.reshape(embedding_i, [x_size, y_size, 1])
        else:
            stacked_embedings = tf.concat(axis=2,
                                          values=[stacked_embedings, tf.reshape(embedding_i, [x_size, y_size, 1])])

    assert stacked_embedings.get_shape().as_list()[2] == 2 * window_size
    print("Stacked embedding size: %s" % stacked_embedings.get_shape().as_list())
    mean_embeddings = tf.reduce_mean(stacked_embedings, 2, keepdims=False)
    print("Reduced mean embedding size: %s" % mean_embeddings.get_shape().as_list())

    # Compute the softmax loss, using a sample of the negative labels each time.
    # inputs are embeddings of the train words
    # with this loss we optimize weights, biases, embeddings

    loss = tf.reduce_mean(
        tf.nn.sampled_softmax_loss(weights=softmax_weights, biases=softmax_biases, inputs=mean_embeddings,
                                   labels=train_labels, num_sampled=num_sampled, num_classes=vocabulary_size))

    # Optimizer.
    # Note: The optimizer will optimize the softmax_weights AND the embeddings.
    optimizer = tf.train.AdamOptimizer(0.001).minimize(loss)

    # Compute the similarity between minibatch examples and all embeddings.
    # We use the cosine distance:
    norm = tf.sqrt(tf.reduce_sum(tf.square(embeddings), 1, keepdims=True))
    normalized_embeddings = embeddings / norm
    valid_embeddings = tf.nn.embedding_lookup(normalized_embeddings, valid_dataset)
    similarity = tf.matmul(valid_embeddings, tf.transpose(normalized_embeddings))


def run_word2vec_source(batch_size):
    global embedding_size, window_size
    global valid_size, valid_window, valid_examples
    global num_sampled
    global train_dataset, train_labels
    global valid_dataset
    global softmax_weights, softmax_biases
    global loss, optimizer, similarity, normalized_embeddings
    global src_reverse_dictionary
    global vocabulary_size, embedding_size

    num_steps = 100001

    config=tf.ConfigProto(allow_soft_placement=True) 
    config.gpu_options.allow_growth = True	
    	
    with tf.Session(config=config) as session:
        tf.global_variables_initializer().run()
        print('Initialized')
        average_loss = 0
        for step in range(num_steps):

            batch_data, batch_labels = generate_batch_for_word2vec(batch_size, window_size, is_source=True)
            feed_dict = {train_dataset: batch_data, train_labels: batch_labels}
            _, l = session.run([optimizer, loss], feed_dict=feed_dict)
            average_loss += l
            if (step + 1) % 2000 == 0:
                if step > 0:
                    average_loss = average_loss / 2000
                    # The average loss is an estimate of the loss over the last 2000 batches.
                print('Average loss at step %d: %f' % (step + 1, average_loss))
                average_loss = 0
            # note that this is expensive (~20% slowdown if computed every 500 steps)
            if (step + 1) % 10000 == 0:
                sim = similarity.eval()
                for i in range(valid_size):
                    valid_word = src_reverse_dictionary[valid_examples[i]]
                    top_k = 8  # number of nearest neighbors
                    nearest = (-sim[i, :]).argsort()[1:top_k + 1]
                    log = 'Nearest to %s:' % valid_word
                    for k in range(top_k):
                        close_word = src_reverse_dictionary[nearest[k]]
                        log = '%s %s,' % (log, close_word)
                    print(log)
        cbow_final_embeddings = normalized_embeddings.eval()

    np.save('de-embeddings.npy', cbow_final_embeddings)

def run_word2vec_target(batch_size):
    global embedding_size, window_size
    global valid_size, valid_window, valid_examples
    global num_sampled
    global train_dataset, train_labels
    global valid_dataset
    global softmax_weights, softmax_biases
    global loss, optimizer, similarity, normalized_embeddings
    global tgt_reverse_dictionary
    global vocabulary_size, embedding_size

    num_steps = 100001
    
    config=tf.ConfigProto(allow_soft_placement=True) 
    config.gpu_options.allow_growth = True	
    with tf.Session(config=config) as session:
        tf.global_variables_initializer().run()
        print('Initialized')
        average_loss = 0
        for step in range(num_steps):

            batch_data, batch_labels = generate_batch_for_word2vec(batch_size, window_size, is_source=False)
            feed_dict = {train_dataset: batch_data, train_labels: batch_labels}
            _, l = session.run([optimizer, loss], feed_dict=feed_dict)
            average_loss += l
            if (step + 1) % 2000 == 0:
                if step > 0:
                    average_loss = average_loss / 2000
                    # The average loss is an estimate of the loss over the last 2000 batches.
                print('Average loss at step %d: %f' % (step + 1, average_loss))
                average_loss = 0
            # note that this is expensive (~20% slowdown if computed every 500 steps)
            if (step + 1) % 10000 == 0:
                sim = similarity.eval()
                for i in range(valid_size):
                    valid_word = tgt_reverse_dictionary[valid_examples[i]]
                    top_k = 8  # number of nearest neighbors
                    nearest = (-sim[i, :]).argsort()[1:top_k + 1]
                    log = 'Nearest to %s:' % valid_word
                    for k in range(top_k):
                        close_word = tgt_reverse_dictionary[nearest[k]]
                        log = '%s %s,' % (log, close_word)
                    print(log)
        cbow_final_embeddings = normalized_embeddings.eval()

    np.save('en-embeddings.npy', cbow_final_embeddings)