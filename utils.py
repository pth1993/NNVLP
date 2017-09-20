import numpy as np
import theano
import theano.tensor as T
from labelencoder import LabelEncoder

MAX_CHAR_LENGTH = 45
word_end = "##WE##"


def get_max_length(word_sentences):
    max_len = 0
    for sentence in word_sentences:
        length = len(sentence)
        if length > max_len:
            max_len = length
    return max_len


def build_char_embedd_table(char_embedd_dim, char_alphabet):
    scale = np.sqrt(3.0 / char_embedd_dim)
    char_embedd_table = np.random.uniform(-scale, scale, [char_alphabet.size(), char_embedd_dim]).astype(
        theano.config.floatX)
    return char_embedd_table


def get_character_indexes(sentences, char_alphabet):
    index_sentences = []
    max_length = 0
    for words in sentences:
        index_words = []
        for word in words:
            index_chars = []
            if len(word) > max_length:
                max_length = len(word)
            for char in word[:MAX_CHAR_LENGTH]:
                char_id = char_alphabet.get_index(char)
                index_chars.append(char_id)
            index_words.append(index_chars)
        index_sentences.append(index_words)
    return index_sentences, max_length


def construct_tensor_char(index_sentences, max_sent_length, max_char_length, char_alphabet):
    C = np.empty([len(index_sentences), max_sent_length, max_char_length], dtype=np.int32)
    word_end_id = char_alphabet.get_index(word_end)
    for i in range(len(index_sentences)):
        words = index_sentences[i]
        sent_length = len(words)
        for j in range(sent_length):
            chars = words[j]
            char_length = len(chars)
            for k in range(char_length):
                cid = chars[k]
                C[i, j, k] = cid
            # fill index of word end after the end of word
            C[i, j, char_length:] = word_end_id
        # Zero out C after the end of the sentence
        C[i, sent_length:, :] = 0
    return C


def crf_prediction(energies):
    def inner_function(energies_one_step, prior_pi, prior_pointer):
        prior_pi_shuffled = prior_pi.dimshuffle(0, 1, 'x')
        pi_t = T.max(prior_pi_shuffled + energies_one_step, axis=1)
        pointer_t = T.argmax(prior_pi_shuffled + energies_one_step, axis=1)
        return [pi_t, pointer_t]

    def back_pointer(pointer, pointer_tp1):
        return pointer[T.arange(pointer.shape[0]), pointer_tp1]
    # Input should be provided as (n_batch, n_time_steps, num_labels, num_labels)
    # but scan requires the iterable dimension to be first
    # So, we need to dimshuffle to (n_time_steps, n_batch, num_labels, num_labels)
    energies_shuffled = energies.dimshuffle(1, 0, 2, 3)
    # pi at time 0 is the last rwo at time 0. but we need to remove the last column which is the pad symbol.
    pi_time0 = energies_shuffled[0, :, -1, :-1]
    # the last row and column is the tag for pad symbol. reduce these two dimensions by 1 to remove that.
    # now the shape of energies_shuffled is [n_time_steps, b_batch, t, t] where t = num_labels - 1.
    energies_shuffled = energies_shuffled[:, :, :-1, :-1]
    initials = [pi_time0, T.cast(T.fill(pi_time0, -1), 'int64')]
    [pis, pointers], _ = theano.scan(fn=inner_function, outputs_info=initials, sequences=[energies_shuffled[1:]])
    pi_n = pis[-1]
    pointer_n = T.argmax(pi_n, axis=1)
    back_pointers, _ = theano.scan(fn=back_pointer, outputs_info=pointer_n, sequences=[pointers], go_backwards=True)
    # prediction shape [batch_size, length]
    prediction_revered = T.concatenate([pointer_n.dimshuffle(0, 'x'), back_pointers.dimshuffle(1, 0)], axis=1)
    prediction = prediction_revered[:, T.arange(prediction_revered.shape[1] - 1, -1, -1)]
    return prediction


def output_predictions(predictions, targets, masks, filename, label_alphabet, is_flattened=True):
    batch_size, max_length = targets.shape
    with open(filename, 'a') as file:
        for i in range(batch_size):
            for j in range(max_length):
                if masks[i, j] > 0.:
                    prediction = predictions[i * max_length + j] + 1 if is_flattened else predictions[i, j] + 1
                    file.write('_ %s %s\n' % (label_alphabet.get_instance(targets[i, j] + 1),
                                              label_alphabet.get_instance(prediction)))
            file.write('\n')


def theano_logsumexp(x, axis=None):
    xmax = x.max(axis=axis, keepdims=True)
    xmax_ = x.max(axis=axis)
    return xmax_ + T.log(T.exp(x - xmax).sum(axis=axis))


def crf_loss(energies, targets, masks):
    assert energies.ndim == 4
    assert targets.ndim == 2
    assert masks.ndim == 2

    def inner_function(energies_one_step, targets_one_step, mask_one_step, prior_partition, prev_label, tg_energy):
        partition_shuffled = prior_partition.dimshuffle(0, 1, 'x')
        partition_t = T.switch(mask_one_step.dimshuffle(0, 'x'),
                               theano_logsumexp(energies_one_step + partition_shuffled, axis=1),
                               prior_partition)

        return [partition_t, targets_one_step,
                tg_energy + energies_one_step[T.arange(energies_one_step.shape[0]), prev_label, targets_one_step]]

    # Input should be provided as (n_batch, n_time_steps, num_labels, num_labels)
    # but scan requires the iterable dimension to be first
    # So, we need to dimshuffle to (n_time_steps, n_batch, num_labels, num_labels)
    energies_shuffled = energies.dimshuffle(1, 0, 2, 3)
    targets_shuffled = targets.dimshuffle(1, 0)
    masks_shuffled = masks.dimshuffle(1, 0)
    # initials should be energies_shuffles[0, :, -1, :]
    init_label = T.cast(T.fill(energies[:, 0, 0, 0], -1), 'int32')
    energy_time0 = energies_shuffled[0]
    target_time0 = targets_shuffled[0]
    initials = [energies_shuffled[0, :, -1, :], target_time0,
                energy_time0[T.arange(energy_time0.shape[0]), init_label, target_time0]]
    [partitions, _, target_energies], _ = theano.scan(fn=inner_function, outputs_info=initials,
                                                      sequences=[energies_shuffled[1:], targets_shuffled[1:],
                                                                 masks_shuffled[1:]])
    partition = partitions[-1]
    target_energy = target_energies[-1]
    loss = theano_logsumexp(partition, axis=1) - target_energy
    return loss


def crf_accuracy(energies, targets):
    assert energies.ndim == 4
    assert targets.ndim == 2

    def inner_function(energies_one_step, prior_pi, prior_pointer):
        prior_pi_shuffled = prior_pi.dimshuffle(0, 1, 'x')
        pi_t = T.max(prior_pi_shuffled + energies_one_step, axis=1)
        pointer_t = T.argmax(prior_pi_shuffled + energies_one_step, axis=1)
        return [pi_t, pointer_t]

    def back_pointer(pointer, pointer_tp1):
        return pointer[T.arange(pointer.shape[0]), pointer_tp1]

    # Input should be provided as (n_batch, n_time_steps, num_labels, num_labels)
    # but scan requires the iterable dimension to be first
    # So, we need to dimshuffle to (n_time_steps, n_batch, num_labels, num_labels)
    energies_shuffled = energies.dimshuffle(1, 0, 2, 3)
    # pi at time 0 is the last rwo at time 0. but we need to remove the last column which is the pad symbol.
    pi_time0 = energies_shuffled[0, :, -1, :-1]
    # the last row and column is the tag for pad symbol. reduce these two dimensions by 1 to remove that.
    # now the shape of energies_shuffled is [n_time_steps, b_batch, t, t] where t = num_labels - 1.
    energies_shuffled = energies_shuffled[:, :, :-1, :-1]
    initials = [pi_time0, T.cast(T.fill(pi_time0, -1), 'int64')]
    [pis, pointers], _ = theano.scan(fn=inner_function, outputs_info=initials, sequences=[energies_shuffled[1:]])
    pi_n = pis[-1]
    pointer_n = T.argmax(pi_n, axis=1)
    back_pointers, _ = theano.scan(fn=back_pointer, outputs_info=pointer_n, sequences=[pointers], go_backwards=True)
    # prediction shape [batch_size, length]
    prediction_revered = T.concatenate([pointer_n.dimshuffle(0, 'x'), back_pointers.dimshuffle(1, 0)], axis=1)
    prediction = prediction_revered[:, T.arange(prediction_revered.shape[1] - 1, -1, -1)]
    return prediction, T.eq(prediction, targets)


def map_string_2_id_open(string_list, name):
    string_id_list = []
    alphabet_string = LabelEncoder(name)
    for strings in string_list:
        ids = []
        for string in strings:
            id = alphabet_string.get_index(string)
            ids.append(id)
        string_id_list.append(ids)
    alphabet_string.close()
    return string_id_list, alphabet_string


def map_string_2_id_close(string_list, alphabet_string):
    string_id_list = []
    for strings in string_list:
        ids = []
        for string in strings:
            id = alphabet_string.get_index(string)
            ids.append(id)
        string_id_list.append(ids)
    return string_id_list


def construct_tensor_word(word_sentences, label_index_sentences, unknown_embedd, embedd_words, embedd_vectors,
                          embedd_dim, max_length):
    X = np.empty([len(word_sentences), max_length, embedd_dim], dtype=theano.config.floatX)
    Y = np.empty([len(word_sentences), max_length], dtype=np.int32)
    mask = np.zeros([len(word_sentences), max_length], dtype=theano.config.floatX)
    for i in range(len(word_sentences)):
        words = word_sentences[i]
        label_ids = label_index_sentences[i]
        length = len(words)
        for j in range(length):
            word = words[j].lower()
            label = label_ids[j]
            try:
                embedd = embedd_vectors[embedd_words.index(word)]
            except:
                embedd = unknown_embedd
            X[i, j, :] = embedd
            Y[i, j] = label - 1
        # Zero out X after the end of the sequence
        X[i, length:] = np.zeros([1, embedd_dim], dtype=theano.config.floatX)
        # Copy the last label after the end of the sequence
        Y[i, length:] = Y[i, length - 1]
        # Make the mask for this sample 1 within the range of length
        mask[i, :length] = 1
    return X, Y, mask


def construct_tensor_onehot(feature_sentences, max_length, dim):
    X = np.zeros([len(feature_sentences), max_length, dim], dtype=theano.config.floatX)
    for i in range(len(feature_sentences)):
        for j in range(len(feature_sentences[i])):
            if feature_sentences[i][j] > 0:
                X[i, j, feature_sentences[i][j]] = 1
    return X


def iterate_minibatches(inputs, targets, masks=None, char_inputs=None, batch_size=10, shuffle=False):
    assert len(inputs) == len(targets)
    assert len(inputs) == len(masks)
    assert len(inputs) == len(char_inputs)
    if shuffle:
        indices = np.arange(len(inputs))
        np.random.shuffle(indices)
    for start_idx in range(0, len(inputs), batch_size):
        if shuffle:
            excerpt = indices[start_idx:start_idx + batch_size]
        else:
            excerpt = slice(start_idx, start_idx + batch_size)
        yield inputs[excerpt], targets[excerpt], masks[excerpt], char_inputs[excerpt]
