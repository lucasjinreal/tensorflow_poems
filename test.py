# -*- coding: utf-8 -*-
# file: test.py
# author: JinTian
# time: 07/03/2017 6:44 PM
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
import numpy as np
import tensorflow as tf
from tensorflow.python import debug as tf_debug


def test():
    a = [[1, 2, 3],
         [2, 3, 4, 5, 6],
         [1, 4]]
    print(len(a))
    print(list(map(len, a)))

    with tf.Session() as sess:
        sess = tf_debug.LocalCLIDebugWrapperSession(sess)
        sess.add_tensor_filter('has_inf_or_nan', tensor_filter=tf_debug.has_inf_or_nan)
        a = tf.constant(np.array([1.0, 3.2, 4, 2.1, 1.4]))
        print(sess.run(tf.nn.softmax(a)))


def dynamic_rnn_test():
    with tf.Session() as sess:
        embedding_dict = tf.random_uniform([388, 30], -1.0, 1.0)
        print(embedding_dict.eval())
        train_inputs = tf.constant(
            [
                [1, 3, 56, 67, 4],
                [4, 6, 4, 56, 56],
                [4, 9, 34, 12, 4]
            ]
        )
        embed = tf.nn.embedding_lookup(params=embedding_dict, ids=train_inputs)
        print('embed shape: ', embed.get_shape())
        print(embed.eval())


def test2():
    a = [[2, 3],
         [1, 3],
         [3, 4],
         [1, 4]]
    b = [[2, 3, 4],
         [1, 6, 2]]
    print(np.matmul(a, b))

    weights = [[0.03, 0.124, 0.392, 0.00023, 0.0032]]
    t = np.cumsum(weights)
    print(t)
    s = np.sum(weights)
    print(s)
    sample = int(np.searchsorted(t, np.random.rand(1) * s))
    print(sample)

    a1 = [[1, 2, 3],
          [2, 3, 4, 5, 6],
          [1, 4]]
    test_map = [map(len(l), l) for l in a1]
    print(test_map[0])

    content = ['你好', 'hello', 'how', 'are', 'u']
    word_int_map = dict(zip(content, range(len(content))))
    print(word_int_map)
    print(word_int_map.get)

if __name__ == '__main__':
    test2()
    # dynamic_rnn_test()