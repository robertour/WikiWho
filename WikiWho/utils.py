# -*- coding: utf-8 -*-
"""

:Authors:
    Maribel Acosta,
    Fabian Floeck,
    Kenan Erdogan
"""
from __future__ import division
from __future__ import unicode_literals
import hashlib
from collections import Counter
import re
import json
import os
import tempfile
import subprocess
from time import process_time
from difflib import SequenceMatcher as BaseSequenceMatcher

regex_dot = re.compile(r"([^\s\.=][^\s\.=][^\s\.=]\.) ")
regex_url = re.compile(r"(http.*?://.*?[ \|<>\n\r])")
# regex_url = re.compile(r"(http[s]?://.*?[ \|<>\n\r])")


class Timer():

    """Context manager that measures the time of a process, usage:

    with Timer():
        # the code that you want to meassure
        # and it will output the processing time
    
    Attributes:
        t1 (flost): initial time
    """
    
    def __enter__(self):
        self.t1 = process_time()

    def __exit__(self, *a):
        t2 = process_time()
        print(t2 - self.t1)

def browse_dict(_dict: dict, browser='firefox'):
    """Show a dictionary in a browser
    
    Args:
        _dict (dict): _dict object to visualize
        browser (str, optional): Description
    """
    with tempfile.NamedTemporaryFile('w', suffix='.json', delete=False) as fp:
        json.dump(_dict, fp)

    subprocess.call([browser, fp.name])
    os.remove(fp.name)


def calculate_hash(text):
    return hashlib.md5(text.encode('utf-8')).hexdigest()


def split_into_paragraphs(text):
    text = text.replace('\r\n', '\n').replace('\r', '\n')
    # html table syntax
    text = text.replace('<table>', '\n\n<table>').replace('</table>', '</table>\n\n')
    text = text.replace('<tr>', '\n\n<tr>').replace('</tr>', '</tr>\n\n')
    # wp table syntax
    text = text.replace('{|', '\n\n{|').replace('|}', '|}\n\n')
    text = text.replace('|-\n', '\n\n|-\n')
    return text.split('\n\n')


def split_into_sentences(text):
    text = text.replace('\n', '\n@@@@')
    # punctuation = ('. ', '\n', '; ', '? ', '! ', ': ', )
    # text = text.replace('. ', '.@@@@')
    text = regex_dot.sub(r'\1@@@@', text)
    text = text.replace('; ', ';@@@@')
    text = text.replace('? ', '?@@@@')
    text = text.replace('! ', '!@@@@')
    text = text.replace(': ', ':@@@@')
    text = text.replace('\t', '\t@@@@')
    # comments as sentence
    text = text.replace('<!--', '@@@@<!--')
    text = text.replace('-->', '-->@@@@')
    # references as sentence. ex: <ref name="...">{{ ... }}</ref>
    # text = text.replace('>{', '>@@@@{')
    # text = text.replace('}<', '}@@@@<')
    text = text.replace('<ref', '@@@@<ref')
    text = text.replace('/ref>', '/ref>@@@@')
    # urls as sentence
    text = regex_url.sub(r'@@@@\1@@@@', text)

    # text = text.replace('.{', '.||{')
    # text = text.replace('!{', '!||{')
    # text = text.replace('?{', '?||{')
    # text = text.replace('.[', '.||[')
    # text = text.replace('.]]', '.]]||')
    # text = text.replace('![', '!||[')
    # text = text.replace('?[', '?||[')

    while '@@@@@@@@' in text:
        text = text.replace('@@@@@@@@', '@@@@')
    return text.split('@@@@')


def split_into_tokens(text):
    text = text.replace('|', '||ææææ||')  # use | as delimiter

    text = text.replace('\n', '||').replace(' ', '||')

    symbols = ['.', ',', ';', ':', '?', '!', '-', '_', '/', '\\', '(', ')', '[', ']', '{', '}', '*', '#', '@',
               '&', '=', '+', '%', '~', '$', '^', '<', '>', '"', '\'', '´', '`', '¸', '˛', '’',
               '¤', '₳', '฿', '₵', '¢', '₡', '₢', '₫', '₯', '֏', '₠', '€', 'ƒ', '₣', '₲', '₴', '₭', '₺',
               '₾', 'ℳ', '₥', '₦', '₧', '₱', '₰', '£', '៛', '₽', '₹', '₨', '₪', '৳', '₸', '₮', '₩', '¥',
               '§', '‖', '¦', '⟨', '⟩', '–', '—', '¯', '»', '«', '”', '÷', '×', '′', '″', '‴', '¡',
               '¿', '©', '℗', '®', '℠', '™']
    # currency_symbols_long = '¢,£,¤,¥,֏,؋,৲,৳,৻,૱,௹,฿,៛,₠,₡,₢,₣,₤,₥,₦,₧,₨,₩,₪,₫,€,₭,₮,₯,₰,₱,₲,₳,₴,₵' \
    #                    ',₶,₷,₸,₹,₺,꠸,﷼,﹩,＄,￠,￡,￥,￦'.split(',')
    for c in symbols:
        text = text.replace(c, '||{}||'.format(c))

    # re-construct some special character groups as they are tokens
    text = text.replace('[||||[', '[[').replace(']||||]', ']]')
    text = text.replace('{||||{', '{{').replace('}||||}', '}}')
    # text = text.replace('||.||||.||||.||', '...')
    # text = text.replace('/||||>', '/>').replace('<||||/', '</')
    # text = text.replace('-||||-', '--')
    # text = text.replace('<||||!||||--||', '||<!--||').replace('||--||||>', '||-->||')
    text = text.replace('<||||!||||-||||-||', '||<!--||').replace('||-||||-||||>', '||-->||')

    while '||||' in text:
        text = text.replace('||||', '||')

    tokens = filter(lambda a: a != '', text.split('||'))  # filter empty strings
    tokens = ['|' if w == 'ææææ' else w for w in tokens]  # insert back the |s
    return tokens


def compute_avg_word_freq(token_list):
    c = Counter(token_list)  # compute count of each token in the list
    # remove some tokens
    remove_list = ('<', '>', 'tr', 'td', '[', ']', '"', '*', '==', '{', '}', '|', '-')  # '(', ')'
    for t in remove_list:
        if t in c:
            del c[t]

    return sum(c.values()) / len(c) if c else 0


def iter_rev_tokens(revision):
    """Yield tokens of the revision in order."""
    # from copy import deepcopy
    # ps_copy = deepcopy(revision.paragraphs)
    tmp = {'p': [], 's': []}
    for hash_paragraph in revision.ordered_paragraphs:
        # paragraph = ps_copy[hash_paragraph].pop(0)
        if len(revision.paragraphs[hash_paragraph]) > 1:
            tmp['p'].append(hash_paragraph)
            paragraph = revision.paragraphs[hash_paragraph][tmp['p'].count(hash_paragraph)-1]
        else:
            paragraph = revision.paragraphs[hash_paragraph][0]
        tmp['s'][:] = []
        for hash_sentence in paragraph.ordered_sentences:
            if len(paragraph.sentences[hash_sentence]) > 1:
                # tmp['s'].append('{}-{}'.format(hash_paragraph, hash_sentence))  # and dont do tmp['s'][:] = []
                tmp['s'].append(hash_sentence)
                sentence = paragraph.sentences[hash_sentence][tmp['s'].count(hash_sentence)-1]
            else:
                sentence = paragraph.sentences[hash_sentence][0]
            # sentence = paragraph.sentences[hash_sentence].pop(0)
            for word in sentence.words:
                yield word


# def iter_wikiwho_tokens(wikiwho):
#     """Yield tokens of the article in order."""
#     article_token_ids = set()
#     for rev_id in wikiwho.ordered_revisions:
#         for word in iter_rev_tokens(wikiwho.revisions[rev_id]):
#             if word.token_id not in article_token_ids:
#                 article_token_ids.add(word.token_id)
#                 yield word


class SequenceMatcher(BaseSequenceMatcher):

    def get_opcodes(self):
        """Return list of 5-tuples describing how to turn a into b.
        Each tuple is of the form (tag, i1, i2, j1, j2).  The first tuple
        has i1 == j1 == 0, and remaining tuples have i1 == the i2 from the
        tuple preceding it, and likewise for j1 == the previous j2.
        The tags are strings, with these meanings:
        'replace':  a[i1:i2] should be replaced by b[j1:j2]
        'delete':   a[i1:i2] should be deleted.
                    Note that j1==j2 in this case.
        'insert':   b[j1:j2] should be inserted at a[i1:i1].
                    Note that i1==i2 in this case.
        'equal':    a[i1:i2] == b[j1:j2]
        >>> a = "qabxcd"
        >>> b = "abycdf"
        >>> s = SequenceMatcher(None, a, b)
        >>> for tag, i1, i2, j1, j2 in s.get_opcodes():
        ...    print(("%7s a[%d:%d] (%s) b[%d:%d] (%s)" %
        ...           (tag, i1, i2, a[i1:i2], j1, j2, b[j1:j2])))
         delete a[0:1] (q) b[0:0] ()
          equal a[1:3] (ab) b[0:2] (ab)
        replace a[3:4] (x) b[2:3] (y)
          equal a[4:6] (cd) b[3:5] (cd)
         insert a[6:6] () b[5:6] (f)
        """

        if self.opcodes is not None:
            return self.opcodes
        i = j = 0
        self.opcodes = answer = []
        for ai, bj, size in self.get_matching_blocks():
            # invariant:  we've pumped out correct diffs to change
            # a[:i] into b[:j], and the next matching block is
            # a[ai:ai+size] == b[bj:bj+size].  So we need to pump
            # out a diff to change a[i:ai] into b[j:bj], pump out
            # the matching block, and move (i,j) beyond the match
            tag = ''
            if i < ai:
                for x in self.a[i:ai]:
                    answer.append( (-1, x) )
            if j < bj:
                for x in self.b[j:bj]:
                    answer.append( (1, x) )
            i, j = ai+size, bj+size
            # the list of matching blocks is terminated by a
            # sentinel with size 0
            if size:
                for x in self.a[ai:i]:
                    answer.append( (0, x) )

        return answer