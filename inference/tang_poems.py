# -*- coding: utf-8 -*-
# file: tang_poems.py
# author: JinTian
# time: 08/03/2017 7:45 PM
# Copyright 2017 JinTian. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ------------------------------------------------------------------------
import collections
import os
import sys
import numpy as np
import tensorflow as tf
from models.model import rnn_model
from dataset.poems import process_poems, generate_batch
import heapq

tf.app.flags.DEFINE_integer('batch_size', 64, 'batch size.')
tf.app.flags.DEFINE_float('learning_rate', 0.01, 'learning rate.')

# set this to 'main.py' relative path
tf.app.flags.DEFINE_string('checkpoints_dir', os.path.abspath('./checkpoints/poems/'), 'checkpoints save path.')
tf.app.flags.DEFINE_string('file_path', os.path.abspath('./dataset/data/poems.txt'), 'file name of poems.')


tf.app.flags.DEFINE_string('model_prefix', 'poems', 'model save prefix.')

tf.app.flags.DEFINE_integer('epochs', 50, 'train how many epochs.')

FLAGS = tf.app.flags.FLAGS

start_token = 'G'
end_token = 'E'


def run_training():
    if not os.path.exists(os.path.dirname(FLAGS.checkpoints_dir)):
        os.mkdir(os.path.dirname(FLAGS.checkpoints_dir))
    if not os.path.exists(FLAGS.checkpoints_dir):
        os.mkdir(FLAGS.checkpoints_dir)

    poems_vector, word_to_int, vocabularies = process_poems(FLAGS.file_path)
    batches_inputs, batches_outputs = generate_batch(FLAGS.batch_size, poems_vector, word_to_int)

    input_data = tf.placeholder(tf.int32, [FLAGS.batch_size, None])
    output_targets = tf.placeholder(tf.int32, [FLAGS.batch_size, None])

    end_points = rnn_model(model='lstm', input_data=input_data, output_data=output_targets, vocab_size=len(
        vocabularies), rnn_size=128, num_layers=2, batch_size=64, learning_rate=FLAGS.learning_rate)

    saver = tf.train.Saver(tf.global_variables())
    init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
    with tf.Session() as sess:
        # sess = tf_debug.LocalCLIDebugWrapperSession(sess=sess)
        # sess.add_tensor_filter("has_inf_or_nan", tf_debug.has_inf_or_nan)
        sess.run(init_op)

        start_epoch = 0
        checkpoint = tf.train.latest_checkpoint(FLAGS.checkpoints_dir)
        if checkpoint:
            saver.restore(sess, checkpoint)
            print("[INFO] restore from the checkpoint {0}".format(checkpoint))
            start_epoch += int(checkpoint.split('-')[-1])
        print('[INFO] start training...')
        try:
            for epoch in range(start_epoch, FLAGS.epochs):
                n = 0
                n_chunk = len(poems_vector) // FLAGS.batch_size
                for batch in range(n_chunk):
                    loss, _, _ = sess.run([
                        end_points['total_loss'],
                        end_points['last_state'],
                        end_points['train_op']
                    ], feed_dict={input_data: batches_inputs[n], output_targets: batches_outputs[n]})
                    n += 1
                    print('[INFO] Epoch: %d , batch: %d , training loss: %.6f' % (epoch, batch, loss))
                if epoch % 6 == 0:
                    saver.save(sess, os.path.join(FLAGS.checkpoints_dir, FLAGS.model_prefix), global_step=epoch)
        except KeyboardInterrupt:
            print('[INFO] Interrupt manually, try saving checkpoint for now...')
            saver.save(sess, os.path.join(FLAGS.checkpoints_dir, FLAGS.model_prefix), global_step=epoch)
            print('[INFO] Last epoch were saved, next time will start from epoch {}.'.format(epoch))


def to_word(predict, vocabs):
    t = np.cumsum(predict)
    s = np.sum(predict)
    sample = int(np.searchsorted(t, np.random.rand(1) * s))
    if sample > len(vocabs):
        sample = len(vocabs) - 1
    return vocabs[sample]


def gen_poem(begin_word):
    batch_size = 1
    print('[INFO] loading corpus from %s' % FLAGS.file_path)
    poems_vector, word_int_map, vocabularies = process_poems(FLAGS.file_path)

    input_data = tf.placeholder(tf.int32, [batch_size, None])

    end_points = rnn_model(model='lstm', input_data=input_data, output_data=None, vocab_size=len(
        vocabularies), rnn_size=128, num_layers=2, batch_size=64, learning_rate=FLAGS.learning_rate)

    saver = tf.train.Saver(tf.global_variables())
    init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
    with tf.Session() as sess:
        sess.run(init_op)

        checkpoint = tf.train.latest_checkpoint(FLAGS.checkpoints_dir)
        saver.restore(sess, checkpoint)

        x = np.array([list(map(word_int_map.get, start_token))])

        [predict, last_state] = sess.run([end_points['prediction'], end_points['last_state']],
                                         feed_dict={input_data: x})
        if begin_word:
            word = begin_word
        else:
            word = to_word(predict, vocabularies)
        poem = ''
        while word != end_token:
            poem += word
            x = np.zeros((1, 1))
            x[0, 0] = word_int_map[word]
            [predict, last_state] = sess.run([end_points['prediction'], end_points['last_state']],
                                             feed_dict={input_data: x, end_points['initial_state']: last_state})
            word = to_word(predict, vocabularies)
        # word = words[np.argmax(probs_)]
        return poem


def pretty_print_poem(poem):
    poem_sentences = poem.split('。')
    for s in poem_sentences:
        if s != '' and len(s) > 10:
            print(s + '。')


def main(is_train):
    if is_train:
        print('[INFO] train tang poem...')
        run_training()
    else:
        print('[INFO] write tang poem...')

        begin_word = input('全新藏头诗上线，输入起始字:')
        poem2 = gen_poem(begin_word)
        pretty_print_poem(poem2)


if __name__ == '__main__':
    tf.app.run()