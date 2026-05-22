# -*- coding: utf-8 -*-
"""
Generate a new poem imitating the structure of a template poem.
The model generates fresh content while preserving line lengths and punctuation
pattern of the template.
"""
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
from poems.model import rnn_model
from poems.poems import process_poems
import numpy as np

start_token = 'B'
end_token = 'E'
model_dir = './model/'
corpus_file = './data/poems.txt'

lr = 0.0002

PUNCTUATIONS = set('，。！？；、')


def to_word(predict, vocabs):
    predict = predict[0]
    predict /= np.sum(predict)
    sample = np.random.choice(np.arange(len(predict)), p=predict)
    if sample >= len(vocabs):
        return vocabs[-1]
    else:
        return vocabs[sample]


def parse_template(template):
    """Parse a template poem into its structural skeleton.
    Returns list of (segment_text, punctuation) tuples.

    Example:
        "春眠不觉晓，处处闻啼鸟。夜来风雨声，花落知多少。"
        -> [("春眠不觉晓", "，"), ("处处闻啼鸟", "。"),
            ("夜来风雨声", "，"), ("花落知多少", "。")]
    """
    segments = []
    current = []
    for ch in template:
        if ch in PUNCTUATIONS:
            if current:
                segments.append((''.join(current), ch))
                current = []
        else:
            current.append(ch)
    if current:
        segments.append((''.join(current), ''))
    return segments


def gen_poem_from_template(template, begin_word=None):
    """Generate a new poem following the structure of the template.

    Args:
        template: A string of the template poem, e.g. "春眠不觉晓，处处闻啼鸟。"
        begin_word: Optional first character. If None, the model picks randomly.

    Returns:
        The generated poem string.
    """
    structure = parse_template(template)
    if not structure:
        print('## Error: template has no content.')
        return ''

    total_segments = len(structure)
    print('## Template structure:')
    for seg_text, punct in structure:
        print('##   %d chars + [%s]' % (len(seg_text), punct or 'END'))

    tf.reset_default_graph()
    poems_vector, word_int_map, vocabularies = process_poems(corpus_file)

    input_data = tf.placeholder(tf.int32, [1, None])

    end_points = rnn_model(model='lstm', input_data=input_data, output_data=None,
                           vocab_size=len(vocabularies), rnn_size=128,
                           num_layers=1, batch_size=64, learning_rate=lr)

    saver = tf.train.Saver(tf.global_variables())
    init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())

    with tf.Session() as sess:
        sess.run(init_op)
        checkpoint = tf.train.latest_checkpoint(model_dir)
        saver.restore(sess, checkpoint)

        result = ''
        seg_idx = 0
        chars_in_seg = 0

        # Step 1: feed start_token to initialize RNN state
        x = np.array([list(map(word_int_map.get, start_token))])
        [predict, last_state] = sess.run(
            [end_points['prediction'], end_points['last_state']],
            feed_dict={input_data: x})

        # Step 2: determine the first word
        if begin_word and begin_word in word_int_map:
            word = begin_word
            # Feed begin_word through the model to get proper state
            wid = word_int_map[word]
            x = np.array([[wid]])
            [predict, last_state] = sess.run(
                [end_points['prediction'], end_points['last_state']],
                feed_dict={input_data: x, end_points['initial_state']: last_state})
        else:
            if begin_word:
                print('## Character "%s" not in vocabulary, using random start.' % begin_word)
            word = to_word(predict, vocabularies)

        # Step 3: generate segment by segment
        while seg_idx < total_segments:
            seg_text, punct = structure[seg_idx]
            target_len = len(seg_text)

            # Skip if the model outputs punctuation, spaces, end token, or bad chars
            _skip_chars = PUNCTUATIONS | {start_token, end_token, ' ', '\n', '\r', '□', '�'}
            while word in _skip_chars:
                word = to_word(predict, vocabularies)

            result += word
            chars_in_seg += 1

            # Reached end of this segment → add punctuation
            if chars_in_seg >= target_len:
                if punct:
                    result += punct
                    # Feed punctuation through model to keep state consistent
                    wid = word_int_map.get(punct)
                    if wid is not None:
                        x = np.array([[wid]])
                        [predict, last_state] = sess.run(
                            [end_points['prediction'], end_points['last_state']],
                            feed_dict={input_data: x,
                                       end_points['initial_state']: last_state})
                chars_in_seg = 0
                seg_idx += 1
                if seg_idx >= total_segments:
                    break
                word = to_word(predict, vocabularies)
                continue

            # Feed current word to get next prediction
            wid = word_int_map.get(word)
            if wid is None:
                break
            x = np.array([[wid]])
            [predict, last_state] = sess.run(
                [end_points['prediction'], end_points['last_state']],
                feed_dict={input_data: x,
                           end_points['initial_state']: last_state})
            word = to_word(predict, vocabularies)

        return result


def pretty_print_poem(poem_):
    """Print poem with line breaks at punctuation."""
    for ch in poem_:
        print(ch, end='')
        if ch in PUNCTUATIONS:
            print()
    print()


# ── Example templates ──────────────────────────────────────────────

TEMPLATES = {
    '1': '春眠不觉晓，处处闻啼鸟。夜来风雨声，花落知多少。',        # 五言绝句
    '2': '床前明月光，疑是地上霜。举头望明月，低头思故乡。',        # 五言绝句
    '3': '两个黄鹂鸣翠柳，一行白鹭上青天。窗含西岭千秋雪，门泊东吴万里船。',  # 七言绝句
    '4': '清明时节雨纷纷，路上行人欲断魂。借问酒家何处有，牧童遥指杏花村。',  # 七言绝句
    '5': '千山鸟飞绝，万径人踪灭。孤舟蓑笠翁，独钓寒江雪。',        # 五言绝句
}

if __name__ == '__main__':
    print('=' * 50)
    print('  模板仿写诗歌生成器')
    print('=' * 50)
    print()
    print('可选模板：')
    for key, t in TEMPLATES.items():
        print('  [%s] %s' % (key, t))
    print()
    print('你也可以粘贴自己的模板诗（直接输入诗句即可）')
    print()

    choice = input('请选择模板编号或直接粘贴模板诗: ').strip()

    if choice in TEMPLATES:
        template = TEMPLATES[choice]
    elif choice == 'quit':
        exit()
    elif len(choice) > 2:
        template = choice
    else:
        print('无效输入，使用默认模板。')
        template = TEMPLATES['1']

    print()
    print('模板诗: %s' % template)
    print()

    begin_char = input('请输入第一个字（直接回车则随机生成）: ').strip()
    begin_char = begin_char if begin_char else None

    if begin_char:
        _, word_int_map_check, _ = process_poems(corpus_file)
        if begin_char not in word_int_map_check:
            print('## 注意: "%s" 不在词汇表中，将随机生成首字。' % begin_char)
            begin_char = None

    print()
    print('正在生成...')
    print()

    poem = gen_poem_from_template(template, begin_word=begin_char)

    print('── 模板 ──')
    pretty_print_poem(template)
    print()
    print('── 仿写 ──')
    pretty_print_poem(poem)
