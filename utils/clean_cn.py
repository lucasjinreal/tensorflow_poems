# -*- coding: utf-8 -*-
# file: clean_cn.py
# author: JinTian
# time: 08/03/2017 8:02 PM
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
"""
this script using for clean Chinese corpus.
you can set level for clean, i.e.:
level='all', will clean all character that not Chinese, include punctuations
level='normal', this will generate corpus like normal use, reserve alphabets and numbers
level='clean', this will remove all except Chinese and Chinese punctuations

besides, if you want remove complex Chinese characters, just set this to be true:
simple_only=True
"""
import numpy as np
import os
import string


cn_punctuation_set = ['，', '。', '！', '？', '"', '"', '、']
en_punctuation_set = [',', '.', '?', '!', '"', '"']


def clean_cn_corpus(file_name, clean_level='all', simple_only=True, is_save=True):
    """
    clean Chinese corpus.
    :param file_name:
    :param clean_level:
    :param simple_only:
    :param is_save:
    :return: clean corpus in list type.
    """
    if os.path.dirname(file_name):
        base_dir = os.path.dirname(file_name)
    else:
        print('not set dir. please check')

    save_file = os.path.join(base_dir, os.path.basename(file_name).split('.')[0] + '_cleaned.txt')
    with open(file_name, 'r+') as f:
        clean_content = []
        for l in f.readlines():
            l = l.strip()
            if l == '':
                pass
            else:
                l = list(l)
                should_remove_words = []
                for w in l:
                    if not should_reserve(w, clean_level):
                        should_remove_words.append(w)
                clean_line = [c for c in l if c not in should_remove_words]
                clean_line = ''.join(clean_line)
                if clean_line != '':
                    clean_content.append(clean_line)
    if is_save:
        with open(save_file, 'w+') as f:
            for l in clean_content:
                f.write(l + '\n')
        print('[INFO] cleaned file have been saved to %s.' % save_file)
    return clean_content


def should_reserve(w, clean_level):
    if w == ' ':
        return True
    else:
        if clean_level == 'all':
            # only reserve Chinese characters
            if w in cn_punctuation_set or w in string.punctuation or is_alphabet(w):
                return False
            else:
                return is_chinese(w)
        elif clean_level == 'normal':
            # reserve Chinese characters, English alphabet, number
            if is_chinese(w) or is_alphabet(w) or is_number(w):
                return True
            elif w in cn_punctuation_set or w in en_punctuation_set:
                return True
            else:
                return False
        elif clean_level == 'clean':
            if is_chinese(w):
                return True
            elif w in cn_punctuation_set:
                return True
            else:
                return False
        else:
            raise "clean_level not support %s, please set for all, normal, clean" % clean_level


def is_chinese(uchar):
    """is chinese"""
    if u'\u4e00' <= uchar <= u'\u9fa5':
        return True
    else:
        return False


def is_number(uchar):
    """is number"""
    if u'\u0030' <= uchar <= u'\u0039':
        return True
    else:
        return False


def is_alphabet(uchar):
    """is alphabet"""
    if (u'\u0041' <= uchar <= u'\u005a') or (u'\u0061' <= uchar <= u'\u007a'):
        return True
    else:
        return False


def semi_angle_to_sbc(uchar):
    """半角转全角"""
    inside_code = ord(uchar)
    if inside_code < 0x0020 or inside_code > 0x7e:
        return uchar
    if inside_code == 0x0020:
        inside_code = 0x3000
    else:
        inside_code += 0xfee0
    return chr(inside_code)


def sbc_to_semi_angle(uchar):
    """全角转半角"""
    inside_code = ord(uchar)
    if inside_code == 0x3000:
        inside_code = 0x0020
    else:
        inside_code -= 0xfee0
    if inside_code < 0x0020 or inside_code > 0x7e:
        return uchar
    return chr(inside_code)








