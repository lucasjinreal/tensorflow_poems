# -*- coding: utf-8 -*-
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import numpy as np


def _manual_rnn(cell_fn, inputs, initial_state, num_layers, sequence_length=None):
    """Run a multi-layer RNN manually, matching old checkpoint variable scopes.
    Variable scope structure: rnn/multi_rnn_cell/cell_0/basic_lstm_cell/kernel
    """
    # Transpose from [batch, time, features] to time-major [time, batch, features]
    inputs_time = tf.transpose(inputs, [1, 0, 2])
    time_steps = tf.shape(inputs_time)[0]

    c0, h0 = initial_state

    outputs_ta = tf.TensorArray(dtype=tf.float32, size=time_steps)

    def _time_step(time, c, h, outputs_ta):
        cell_inputs = inputs_time[time]

        for layer_idx in range(num_layers):
            with tf.compat.v1.variable_scope(
                    'rnn/multi_rnn_cell/cell_%d' % layer_idx):
                cell_output, (c, h) = cell_fn(cell_inputs, (c, h), layer_idx)
                cell_inputs = cell_output

        outputs_ta = outputs_ta.write(time, cell_output)
        return time + 1, c, h, outputs_ta

    _, fc, fh, outputs_ta = tf.while_loop(
        cond=lambda t, *_: t < time_steps,
        body=_time_step,
        loop_vars=(0, c0, h0, outputs_ta),
    )

    # Transpose back to [batch, time, features]
    outputs_time = outputs_ta.stack()  # [time, batch, features]
    outputs = tf.transpose(outputs_time, [1, 0, 2])  # [batch, time, features]

    return outputs, (fc, fh)


def _lstm_cell_fn(inputs, state_tuple, layer_idx):
    """Single LSTM step. Variable scope: basic_lstm_cell/ (kernel, bias)."""
    rnn_size = state_tuple[0].get_shape().as_list()[1]
    c, h = state_tuple

    input_dim = inputs.get_shape().as_list()[1]
    concat = tf.concat([inputs, h], 1)

    with tf.compat.v1.variable_scope('basic_lstm_cell'):
        kernel = tf.compat.v1.get_variable(
            'kernel', shape=[input_dim + rnn_size, 4 * rnn_size],
            dtype=tf.float32)
        bias = tf.compat.v1.get_variable(
            'bias', shape=[4 * rnn_size], dtype=tf.float32)

    gate_inputs = tf.nn.bias_add(tf.matmul(concat, kernel), bias)
    i, j, f, o = tf.split(value=gate_inputs, num_or_size_splits=4, axis=1)

    new_c = c * tf.sigmoid(f) + tf.sigmoid(i) * tf.tanh(j)
    new_h = tf.tanh(new_c) * tf.sigmoid(o)

    return new_h, (new_c, new_h)


def rnn_model(model, input_data, output_data, vocab_size, rnn_size=128, num_layers=2,
              batch_size=64, learning_rate=0.01):
    end_points = {}

    if model != 'lstm':
        raise NotImplementedError("Only 'lstm' model is supported")

    init_batch = batch_size if output_data is not None else 1

    initial_state = (
        tf.zeros([init_batch, rnn_size], tf.float32),
        tf.zeros([init_batch, rnn_size], tf.float32),
    )

    with tf.device("/cpu:0"):
        embedding = tf.compat.v1.get_variable('embedding', initializer=tf.random_uniform(
            [vocab_size + 1, rnn_size], -1.0, 1.0))
        inputs = tf.nn.embedding_lookup(embedding, input_data)

    outputs, last_state = _manual_rnn(
        _lstm_cell_fn, inputs, initial_state, num_layers)

    output = tf.reshape(outputs, [-1, rnn_size])

    weights = tf.Variable(tf.truncated_normal([rnn_size, vocab_size + 1]))
    bias = tf.Variable(tf.zeros(shape=[vocab_size + 1]))
    logits = tf.nn.bias_add(tf.matmul(output, weights), bias=bias)

    if output_data is not None:
        labels = tf.one_hot(tf.reshape(output_data, [-1]), depth=vocab_size + 1)
        loss = tf.nn.softmax_cross_entropy_with_logits(labels=labels, logits=logits)
        total_loss = tf.reduce_mean(loss)
        train_op = tf.train.AdamOptimizer(learning_rate).minimize(total_loss)

        end_points['initial_state'] = initial_state
        end_points['output'] = output
        end_points['train_op'] = train_op
        end_points['total_loss'] = total_loss
        end_points['loss'] = loss
        end_points['last_state'] = last_state
    else:
        prediction = tf.nn.softmax(logits)
        end_points['initial_state'] = initial_state
        end_points['last_state'] = last_state
        end_points['prediction'] = prediction

    return end_points
