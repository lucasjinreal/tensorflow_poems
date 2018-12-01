# -*- coding: utf-8 -*-
#
#MIT License
#
#Copyright (c) 2018 damtharvey
#
#Permission is hereby granted, free of charge, to any person obtaining a copy
#of this software and associated documentation files (the "Software"), to deal
#in the Software without restriction, including without limitation the rights
#to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
#copies of the Software, and to permit persons to whom the Software is
#furnished to do so, subject to the following conditions:
#
#The above copyright notice and this permission notice shall be included in all
#copies or substantial portions of the Software.
#
#THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
#IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
#FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
#AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
#LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
#OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
#SOFTWARE.
"""
These functions can process data from
github.com/chinese-poetry/chinese-poetry/json to make regulated verse data
ready to use for training.

If you don't have jsons already, just
git clone https://github.com/chinese-poetry/chinese-poetry.git
to get it.

Regulated verse forms are expected to be tuples of (number of couplets,
characters per couplet). For reference:
wujue 五言絕句 = (2, 10)
qijue 七言絕句 = (2, 14)
wulv 五言律詩 = (4, 10)
qilv 七言律詩 = (4, 14)
"""

import glob
import os
import pandas as pd


def unregulated(paragraphs):
    """
    Return True if the df row describes unregulated verse.
    """
    if all(len(couplet) == len(paragraphs[0]) for couplet in paragraphs):
        return False
    else:
        return True

def get_poems_in_df(df, form):
    """
    Return a txt-friendly string of only poems in df of the specified form.
    """
    big_string = ""
    for row in range(len(df)):
        if len(df["strains"][row]) != form[0] or \
           len(df["strains"][row][0]) - 2 != form[1]:
            continue
        if "○" in str(df["paragraphs"][row]):
            continue
        if unregulated(df["paragraphs"][row]):
            continue
        big_string += df["title"][row] + ":"
        for couplet in df["paragraphs"][row]:
            big_string += couplet
        big_string += "\n"
    return big_string

def get_poems_in_dir(dir, form, save_dir):
    """
    Save to save_dir poems of form in dir in separate txt files by df.
    """
    files = [f for f in os.listdir(dir) if "poet" in f]
    for file in files: # Restart partway through if kernel dies.
        with open(os.path.join(save_dir, file[:-5] + ".txt"), "w") as data:
            print("Now reading " + file)
            df = pd.read_json(os.path.join(dir, file))
            poems = get_poems_in_df(df, form)
            data.write(poems)
            print(str(len(poems)) + " chars written to "
                  + save_dir + "/" + file[:-5] + ".txt")
    return 0

def combine_txt(txts_dir, save_file):
    """
    Combine .txt files in txts_dir and save to save_file.
    """
    read_files = glob.glob(os.path.join(txts_dir, "*.txt"))
    with open(save_file, "wb") as outfile:
        for f in read_files:
            with open(f, "rb") as infile:
                outfile.write(infile.read())
    return 0