import lasagne
import theano.tensor as T
import theano
from lasagne.layers import Gate
import lasagne.nonlinearities as nonlinearities
import utils
import time
import sys
import numpy as np
import subprocess
import shlex
from lasagne import init
from lasagne.layers import MergeLayer


class CRFLayer(MergeLayer):
    def __init__(self, incoming, num_labels, mask_input=None, W=init.GlorotUniform(), b=init.Constant(0.), **kwargs):
        self.input_shape = incoming.output_shape
        incomings = [incoming]
        self.mask_incoming_index = -1
        if mask_input is not None:
            incomings.append(mask_input)
            self.mask_incoming_index = 1
        super(CRFLayer, self).__init__(incomings, **kwargs)
        self.num_labels = num_labels + 1
        self.pad_label_index = num_labels
        num_inputs = self.input_shape[2]
        self.W = self.add_param(W, (num_inputs, self.num_labels, self.num_labels), name="W")
        if b is None:
            self.b = None
        else:
            self.b = self.add_param(b, (self.num_labels, self.num_labels), name="b", regularizable=False)

    def get_output_shape_for(self, input_shapes):
        input_shape = input_shapes[0]
        return input_shape[0], input_shape[1], self.num_labels, self.num_labels

    def get_output_for(self, inputs, **kwargs):
        input = inputs[0]
        mask = None
        if self.mask_incoming_index > 0:
            mask = inputs[self.mask_incoming_index]
        out = T.tensordot(input, self.W, axes=[[2], [0]])
        if self.b is not None:
            b_shuffled = self.b.dimshuffle('x', 'x', 0, 1)
            out = out + b_shuffled
        if mask is not None:
            mask_shuffled = mask.dimshuffle(0, 1, 'x', 'x')
            out = out * mask_shuffled
        return out


def build_model(embedd_dim, max_sent_length, max_char_length, char_alphabet_size, char_embedd_dim, num_labels, dropout,
                num_filters, num_units, grad_clipping, peepholes, char_embedd_table):
    # create target layer
    target_var = T.imatrix(name='targets')
    # create mask layer
    mask_var = T.matrix(name='masks', dtype=theano.config.floatX)
    layer_mask = lasagne.layers.InputLayer(shape=(None, max_sent_length), input_var=mask_var, name='mask')
    # create word input and char input layers
    input_var = T.tensor3(name='inputs', dtype=theano.config.floatX)
    char_input_var = T.itensor3(name='char-inputs')
    layer_embedding = lasagne.layers.InputLayer(shape=(None, max_sent_length, embedd_dim), input_var=input_var,
                                                name='input')
    layer_char_input = lasagne.layers.InputLayer(shape=(None, max_sent_length, max_char_length),
                                                 input_var=char_input_var, name='char-input')
    layer_char_input = lasagne.layers.reshape(layer_char_input, (-1, [2]))
    layer_char_embedding = lasagne.layers.EmbeddingLayer(layer_char_input, input_size=char_alphabet_size,
                                                         output_size=char_embedd_dim, name='char_embedding',
                                                         W=char_embedd_table)
    layer_char_input = lasagne.layers.DimshuffleLayer(layer_char_embedding, pattern=(0, 2, 1))
    # create window size
    conv_window = 3
    _, sent_length, _ = layer_embedding.output_shape
    if dropout:
        layer_char_input = lasagne.layers.DropoutLayer(layer_char_input, p=0.5)
    # construct convolution layer
    cnn_layer = lasagne.layers.Conv1DLayer(layer_char_input, num_filters=num_filters,
                                           filter_size=conv_window, pad='full',
                                           nonlinearity=lasagne.nonlinearities.tanh, name='cnn')
    # infer the pool size for pooling (pool size should go through all time step of cnn)
    _, _, pool_size = cnn_layer.output_shape
    # construct max pool layer
    pool_layer = lasagne.layers.MaxPool1DLayer(cnn_layer, pool_size=pool_size)
    # reshape the layer to match lstm incoming layer [batch * sent_length, num_filters, 1] --> [batch, sent_length,
    # num_filters]
    output_cnn_layer = lasagne.layers.reshape(pool_layer, (-1, sent_length, [1]))
    # finally, concatenate the two incoming layers together.
    incoming = lasagne.layers.concat([output_cnn_layer, layer_embedding], axis=2)
    # create bi-lstm
    if dropout:
        incoming = lasagne.layers.DropoutLayer(incoming, p=0.5)
    ingate_forward = Gate(W_in=lasagne.init.GlorotUniform(), W_hid=lasagne.init.GlorotUniform(),
                          W_cell=lasagne.init.Uniform(range=0.1))
    outgate_forward = Gate(W_in=lasagne.init.GlorotUniform(), W_hid=lasagne.init.GlorotUniform(),
                           W_cell=lasagne.init.Uniform(range=0.1))
    # according to Jozefowicz et al.(2015), init bias of forget gate to 1.
    forgetgate_forward = Gate(W_in=lasagne.init.GlorotUniform(), W_hid=lasagne.init.GlorotUniform(),
                              W_cell=lasagne.init.Uniform(range=0.1), b=lasagne.init.Constant(1.))
    # now use tanh for nonlinear function of cell, need to try pure linear cell
    cell_forward = Gate(W_in=lasagne.init.GlorotUniform(), W_hid=lasagne.init.GlorotUniform(), W_cell=None,
                        nonlinearity=nonlinearities.tanh)
    lstm_forward = lasagne.layers.LSTMLayer(incoming, num_units, mask_input=layer_mask,
                                            grad_clipping=grad_clipping, nonlinearity=nonlinearities.tanh,
                                            peepholes=peepholes, ingate=ingate_forward, outgate=outgate_forward,
                                            forgetgate=forgetgate_forward, cell=cell_forward, name='forward')
    ingate_backward = Gate(W_in=lasagne.init.GlorotUniform(), W_hid=lasagne.init.GlorotUniform(),
                           W_cell=lasagne.init.Uniform(range=0.1))
    outgate_backward = Gate(W_in=lasagne.init.GlorotUniform(), W_hid=lasagne.init.GlorotUniform(),
                            W_cell=lasagne.init.Uniform(range=0.1))
    # according to Jozefowicz et al.(2015), init bias of forget gate to 1.
    forgetgate_backward = Gate(W_in=lasagne.init.GlorotUniform(), W_hid=lasagne.init.GlorotUniform(),
                               W_cell=lasagne.init.Uniform(range=0.1), b=lasagne.init.Constant(1.))
    # now use tanh for nonlinear function of cell, need to try pure linear cell
    cell_backward = Gate(W_in=lasagne.init.GlorotUniform(), W_hid=lasagne.init.GlorotUniform(), W_cell=None,
                         nonlinearity=nonlinearities.tanh)
    lstm_backward = lasagne.layers.LSTMLayer(incoming, num_units, mask_input=layer_mask,
                                             grad_clipping=grad_clipping, nonlinearity=nonlinearities.tanh,
                                             peepholes=peepholes, backwards=True, ingate=ingate_backward,
                                             outgate=outgate_backward, forgetgate=forgetgate_backward,
                                             cell=cell_backward, name='backward')
    # concatenate the outputs of forward and backward RNNs to combine them.
    concat = lasagne.layers.concat([lstm_forward, lstm_backward], axis=2, name="bi-lstm")
    # dropout for output
    if dropout:
        concat = lasagne.layers.DropoutLayer(concat, p=0.5)
    # the shape of Bi-LSTM output (concat) is (batch_size, input_length, 2 * num_hidden_units)
    model = CRFLayer(concat, num_labels, mask_input=layer_mask)
    energies = lasagne.layers.get_output(model, deterministic=True)
    prediction = utils.crf_prediction(energies)
    prediction_fn = theano.function([input_var, mask_var, char_input_var], [prediction])
    return model, input_var, target_var, mask_var, char_input_var, prediction_fn


def train_model(num_data, batch_size, learning_rate, patience, decay_rate, X_train, Y_train, mask_train, C_train, X_dev,
                Y_dev, mask_dev, C_dev, X_test, Y_test, mask_test, C_test, input_var, target_var, mask_var,
                char_input_var, model, model_name, label_alphabet, output_dir):
    num_tokens = mask_var.sum(dtype=theano.config.floatX)
    energies_train = lasagne.layers.get_output(model)
    energies_eval = lasagne.layers.get_output(model, deterministic=True)
    loss_train = utils.crf_loss(energies_train, target_var, mask_var).mean()
    loss_eval = utils.crf_loss(energies_eval, target_var, mask_var).mean()
    _, corr_train = utils.crf_accuracy(energies_train, target_var)
    corr_train = (corr_train * mask_var).sum(dtype=theano.config.floatX)
    prediction_eval, corr_eval = utils.crf_accuracy(energies_eval, target_var)
    corr_eval = (corr_eval * mask_var).sum(dtype=theano.config.floatX)
    params = lasagne.layers.get_all_params(model, trainable=True)
    updates = lasagne.updates.momentum(loss_train, params=params, learning_rate=learning_rate, momentum=0.9)
    train_fn = theano.function([input_var, target_var, mask_var, char_input_var],
                               [loss_train, corr_train, num_tokens], updates=updates)
    eval_fn = theano.function([input_var, target_var, mask_var, char_input_var],
                              [loss_eval, corr_eval, num_tokens, prediction_eval])
    num_batches = num_data / batch_size
    num_epochs = 1000
    best_loss = 1e+12
    best_acc = 0.0
    best_epoch_loss = 0
    best_epoch_acc = 0
    best_loss_test_err = 0.
    best_loss_test_corr = 0.
    best_acc_test_err = 0.
    best_acc_test_corr = 0.
    stop_count = 0
    lr = learning_rate
    for epoch in range(1, num_epochs + 1):
        print 'Epoch %d (learning rate=%.4f, decay rate=%.4f): ' % (epoch, lr, decay_rate)
        train_err = 0.0
        train_corr = 0.0
        train_total = 0
        train_inst = 0
        start_time = time.time()
        num_back = 0
        train_batches = 0
        for batch in utils.iterate_minibatches(X_train, Y_train, masks=mask_train, char_inputs=C_train,
                                               batch_size=batch_size, shuffle=True):
            inputs, targets, masks, char_inputs = batch
            err, corr, num = train_fn(inputs, targets, masks, char_inputs)
            train_err += err * inputs.shape[0]
            train_corr += corr
            train_total += num
            train_inst += inputs.shape[0]
            train_batches += 1
            time_ave = (time.time() - start_time) / train_batches
            time_left = (num_batches - train_batches) * time_ave
            sys.stdout.write("\b" * num_back)
            log_info = 'train: %d/%d loss: %.4f, acc: %.2f%%, time left (estimated): %.2fs' % (
                min(train_batches * batch_size, num_data), num_data,
                train_err / train_inst, train_corr * 100 / train_total, time_left)
            sys.stdout.write(log_info)
            num_back = len(log_info)
        # update training log after each epoch
        assert train_inst == num_data
        sys.stdout.write("\b" * num_back)
        print 'train: %d/%d loss: %.4f, acc: %.2f%%, time: %.2fs' % (
            min(train_batches * batch_size, num_data), num_data,
            train_err / num_data, train_corr * 100 / train_total, time.time() - start_time)
        # evaluate performance on dev data
        dev_err = 0.0
        dev_corr = 0.0
        dev_total = 0
        dev_inst = 0
        for batch in utils.iterate_minibatches(X_dev, Y_dev, masks=mask_dev, char_inputs=C_dev,
                                               batch_size=batch_size):
            inputs, targets, masks, char_inputs = batch
            err, corr, num, predictions = eval_fn(inputs, targets, masks, char_inputs)
            dev_err += err * inputs.shape[0]
            dev_corr += corr
            dev_total += num
            dev_inst += inputs.shape[0]
            utils.output_predictions(predictions, targets, masks, output_dir + '/dev%d' % epoch, label_alphabet,
                                     is_flattened=False)
        print 'dev loss: %.4f, corr: %d, total: %d, acc: %.2f%%' % (
            dev_err / dev_inst, dev_corr, dev_total, dev_corr * 100 / dev_total)
        if model_name != 'pos':
            input = open(output_dir + '/dev%d' % epoch)
            p1 = subprocess.Popen(shlex.split("perl conlleval.pl"), stdin=input)
            p1.wait()
        if best_loss < dev_err and best_acc > dev_corr / dev_total:
            stop_count += 1
        else:
            update_loss = False
            update_acc = False
            stop_count = 0
            if best_loss > dev_err:
                update_loss = True
                best_loss = dev_err
                best_epoch_loss = epoch
            if best_acc < dev_corr / dev_total:
                update_acc = True
                best_acc = dev_corr / dev_total
                best_epoch_acc = epoch
            # evaluate on test data when better performance detected
            test_err = 0.0
            test_corr = 0.0
            test_total = 0
            test_inst = 0
            for batch in utils.iterate_minibatches(X_test, Y_test, masks=mask_test, char_inputs=C_test,
                                                   batch_size=batch_size):
                inputs, targets, masks, char_inputs = batch
                err, corr, num, predictions = eval_fn(inputs, targets, masks, char_inputs)
                test_err += err * inputs.shape[0]
                test_corr += corr
                test_total += num
                test_inst += inputs.shape[0]
                utils.output_predictions(predictions, targets, masks, output_dir + '/test%d' % epoch, label_alphabet,
                                         is_flattened=False)
            np.savez('pre-train-model/' + model_name + '/weights', *lasagne.layers.get_all_param_values(model))
            print 'test loss: %.4f, corr: %d, total: %d, acc: %.2f%%' % (
                test_err / test_inst, test_corr, test_total, test_corr * 100 / test_total)
            if model_name != 'pos':
                input = open(output_dir + '/test%d' % epoch)
                p1 = subprocess.Popen(shlex.split("perl conlleval.pl"), stdin=input)
                p1.wait()
            if update_loss:
                best_loss_test_err = test_err
                best_loss_test_corr = test_corr
            if update_acc:
                best_acc_test_err = test_err
                best_acc_test_corr = test_corr
        # stop if dev acc decrease patience time straightly.
        if stop_count == patience:
            break
        # re-compile a function with new learning rate for training
        lr = learning_rate / (1.0 + epoch * decay_rate)
        lasagne.updates.momentum(loss_train, params=params, learning_rate=lr, momentum=0.9)
        train_fn = theano.function([input_var, target_var, mask_var, char_input_var],
                                   [loss_train, corr_train, num_tokens],
                                   updates=updates)
    # print best performance on test data.
    print "final best loss test performance (at epoch %d)" % (best_epoch_loss)
    print 'test loss: %.4f, corr: %d, total: %d, acc: %.2f%%' % (
        best_loss_test_err / test_inst, best_loss_test_corr, test_total, best_loss_test_corr * 100 / test_total)
    print "final best acc test performance (at epoch %d)" % (best_epoch_acc)
    print 'test loss: %.4f, corr: %d, total: %d, acc: %.2f%%' % (
        best_acc_test_err / test_inst, best_acc_test_corr, test_total, best_acc_test_corr * 100 / test_total)