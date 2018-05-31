import numpy as np
import tensorflow as tf
import collections
import random
import math
import os

tot_captions, only_captions = None, None

data_indices = None
reverse_dictionary = None
embedding_size = None
vocabulary_size = None
max_caption_length = None

def define_data_and_hyperparameters(
        _tot_captions, _only_captions, _reverse_dictionary,
        _emb_size, _vocab_size, _max_cap_length):
    global data_indices, tot_captions, only_captions, reverse_dictionary
    global embedding_size, vocabulary_size, max_caption_length

    tot_captions = _tot_captions
    only_captions = _only_captions

    data_indices = [0 for _ in range(tot_captions)]
    reverse_dictionary = _reverse_dictionary
    embedding_size = _emb_size
    vocabulary_size = _vocab_size
    max_caption_length = _max_cap_length


def generate_batch_for_word2vec(batch_size, window_size):
    # window_size is the amount of words we're looking at from each side of a given word
    # creates a single batch
    global data_indices

    span = 2 * window_size + 1  # [ skip_window target skip_window ]

    batch = np.ndarray(shape=(batch_size, span - 1), dtype=np.int32)
    labels = np.ndarray(shape=(batch_size, 1), dtype=np.int32)
    # e.g if skip_window = 2 then span = 5
    # span is the length of the whole frame we are considering for a single word (left + word + right)
    # skip_window is the length of one side

    caption_ids_for_batch = np.random.randint(0, tot_captions, batch_size)

    for b_i in range(batch_size):
        cap_id = caption_ids_for_batch[b_i]

        buffer = only_captions[cap_id, data_indices[cap_id]:data_indices[cap_id] + span]
        assert buffer.size == span, 'Buffer length (%d), Current data index (%d), Span(%d)' % (
        buffer.size, data_indices[cap_id], span)
        # If we only have EOS tokesn in the sampled text, we sample a new one
        while np.all(buffer == 1):
            # reset the data_indices for that cap_id
            data_indices[cap_id] = 0
            # sample a new cap_id
            cap_id = np.random.randint(0, tot_captions)
            buffer = only_captions[cap_id, data_indices[cap_id]:data_indices[cap_id] + span]

        # fill left and right sides of batch
        batch[b_i, :window_size] = buffer[:window_size]
        batch[b_i, window_size:] = buffer[window_size + 1:]

        labels[b_i, 0] = buffer[window_size]

        # increase the corresponding index
        data_indices[cap_id] = (data_indices[cap_id] + 1) % (max_caption_length - span)

    assert batch.shape[0] == batch_size and batch.shape[1] == span - 1
    return batch, labels

def print_some_batches():
    global data_indices, reverse_dictionary

    for w_size in [1, 2]:
        data_indices = [0 for _ in range(tot_captions)]
        batch, labels = generate_batch_for_word2vec(batch_size=8, window_size=w_size)
        print('\nwith window_size = %d:' %w_size)
        print('    batch:', [[reverse_dictionary[bii] for bii in bi] for bi in batch])
        print('    labels:', [reverse_dictionary[li] for li in labels.reshape(8)])
		
batch_size, embedding_size, window_size = None, None, None
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
    
    # How many words to consider left and right.
    # Skip gram by design does not require to have all the context words in a given step
    # However, for CBOW that's a requirement, so we limit the window size
    window_size = 3 
    
    # We pick a random validation set to sample nearest neighbors
    valid_size = 16 # Random set of words to evaluate similarity on.
    # We sample valid datapoints randomly from a large window without always being deterministic
    valid_window = 50
    
    # When selecting valid examples, we select some of the most frequent words as well as
    # some moderately rare words as well
    valid_examples = np.array(random.sample(range(valid_window), valid_size))
    valid_examples = np.append(valid_examples,random.sample(range(1000, 1000+valid_window), valid_size),axis=0)

    num_sampled = 32 # Number of negative examples to sample.

    tf.reset_default_graph()

    # Training input data (target word IDs). Note that it has 2*window_size columns
    train_dataset = tf.placeholder(tf.int32, shape=[batch_size,2*window_size])
    # Training input label data (context word IDs)
    train_labels = tf.placeholder(tf.int32, shape=[batch_size, 1])
    # Validation input data, we don't need a placeholder
    # as we have already defined the IDs of the words selected
    # as validation data
    valid_dataset = tf.constant(valid_examples, dtype=tf.int32)

    # Variables.

    # Embedding layer, contains the word embeddings
    embeddings = tf.Variable(tf.random_uniform([vocabulary_size, embedding_size], -1.0, 1.0,dtype=tf.float32))

    # Softmax Weights and Biases
    softmax_weights = tf.Variable(tf.truncated_normal([vocabulary_size, embedding_size],
                 stddev=0.5 / math.sqrt(embedding_size),dtype=tf.float32))
    softmax_biases = tf.Variable(tf.random_uniform([vocabulary_size],0.0,0.01))
    
    # Model.
    # Look up embeddings for a batch of inputs.
    # Here we do embedding lookups for each column in the input placeholder
    # and then average them to produce an embedding_size word vector
    stacked_embedings = None
    print('Defining %d embedding lookups representing each word in the context'%(2*window_size))
    for i in range(2*window_size):
        embedding_i = tf.nn.embedding_lookup(embeddings, train_dataset[:,i])        
        x_size,y_size = embedding_i.get_shape().as_list()
        if stacked_embedings is None:
            stacked_embedings = tf.reshape(embedding_i,[x_size,y_size,1])
        else:
            stacked_embedings = tf.concat(axis=2,values=[stacked_embedings,tf.reshape(embedding_i,[x_size,y_size,1])])

    assert stacked_embedings.get_shape().as_list()[2]==2*window_size
    print("Stacked embedding size: %s"%stacked_embedings.get_shape().as_list())
    mean_embeddings =  tf.reduce_mean(stacked_embedings,2,keepdims=False)
    print("Reduced mean embedding size: %s"%mean_embeddings.get_shape().as_list())
    
    	
    # Compute the softmax loss, using a sample of the negative labels each time.
    # inputs are embeddings of the train words
    # with this loss we optimize weights, biases, embeddings
    loss = tf.reduce_mean(tf.nn.sampled_softmax_loss(weights=softmax_weights, biases=softmax_biases, inputs=mean_embeddings,
                           labels=train_labels, num_sampled=num_sampled, num_classes=vocabulary_size))
    # AdamOptimizer.
    optimizer = tf.train.AdamOptimizer(0.0005).minimize(loss)

    # Compute the similarity between minibatch examples and all embeddings.
    # We use the cosine distance:
    norm = tf.sqrt(tf.reduce_sum(tf.square(embeddings), 1, keepdims=True))
    normalized_embeddings = embeddings / norm
    valid_embeddings = tf.nn.embedding_lookup(normalized_embeddings, valid_dataset)
    similarity = tf.matmul(valid_embeddings, tf.transpose(normalized_embeddings))
	

def run_word2vec(batch_size):
    global embedding_size, window_size
    global valid_size, valid_window, valid_examples
    global num_sampled
    global train_dataset, train_labels
    global valid_dataset
    global softmax_weights, softmax_biases 
    global loss, optimizer, similarity, normalized_embeddings
    global data_list, num_files, reverse_dictionary
    global vocabulary_size, embedding_size

    work_dir = 'image_caption_data'
    num_steps = 100001

    session = tf.InteractiveSession()

    tf.global_variables_initializer().run()
    print('Initialized')
    average_loss = 0
    for step in range(num_steps):

        # Load a batch of data
        batch_data, batch_labels = generate_batch_for_word2vec(batch_size, window_size)

        # Populate the feed_dict and run the optimizer and get the loss out
        feed_dict = {train_dataset: batch_data, train_labels: batch_labels}
        _, l = session.run([optimizer, loss], feed_dict=feed_dict)

        average_loss += l

        if (step + 1) % 2000 == 0:
            if step > 0:
                # The average loss is an estimate of the loss over the last 2000 batches.
                average_loss = average_loss / 2000

            print('Average loss at step %d: %f' % (step + 1, average_loss))
            average_loss = 0 # Reset average loss

        if (step + 1) % 10000 == 0:
            sim = similarity.eval()
            # Calculate the most similar (top_k) words
            # to the previosly selected set of valid words
            # Note that this is an expensive step
            for i in range(valid_size):
                valid_word = reverse_dictionary[valid_examples[i]]
                top_k = 3  # number of nearest neighbors
                nearest = (-sim[i, :]).argsort()[1:top_k + 1]
                log = 'Nearest to %s:' % valid_word
                for k in range(top_k):
                    close_word = reverse_dictionary[nearest[k]]
                    log = '%s %s,' % (log, close_word)
                print(log)

    # Get the normalized embeddings we learnt
    cbow_final_embeddings = normalized_embeddings.eval()

    # Save the embeddings to the disk as 'caption_embeddings-tmp.npy'
    # If you want to use this embeddings in the next steps
    # please change the filename to 'caption-embeddings.npy'
    np.save(os.path.join(work_dir,'caption-embeddings-tmp'), cbow_final_embeddings)