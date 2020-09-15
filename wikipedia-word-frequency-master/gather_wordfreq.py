#!/usr/bin/env python3
from collections import defaultdict
import math
import pickle
import re
import subprocess
import sys
import time


MIN_ARTICLES = 3
line_trans = str.maketrans("–’", "-'")
words_split_re = re.compile(r"[^\w\-\']")
is_word_re = re.compile(r"^\w.*\w$")
not_is_word_re = re.compile(r".*\d.*")


if not len(sys.argv) > 1:
    sys.stderr.write("Usage: %s dumps/*.bz2\n" % sys.argv[0])
    sys.exit(1)


# collect data

word_freq = defaultdict(int)
word_docs = {}

start_time = time.time()

doc_no = 0
for fn in sys.argv[1:]:
    sys.stderr.write("Processing %s\n" % fn)
    with subprocess.Popen(
        "bzcat %s | wikiextractor/WikiExtractor.py --no_templates -o - -" % fn,
        stdout=subprocess.PIPE,
        shell=True,
    ) as proc:
        while True:
            line = proc.stdout.readline()
            if not line:
                break
            if line.startswith(b"<"):
                doc_no += 1
                continue
            line = line.decode("utf-8")
            line = line.translate(line_trans)
            line = line.lower()
            for word in filter(None, words_split_re.split(line)):
                if is_word_re.match(word) and not not_is_word_re.match(word):
                    word_freq[word] += 1
                    if not word in word_docs:
                        word_docs[word] = {doc_no}
                    else:
                        word_docs[word].add(doc_no)

exec_time = time.time() - start_time
log.info("Exec time:: " + str(exec_time / 60) + "s")

# # remove words only used once
#
# for word in list(word_uses.keys()):
# 	if len(word_docs[word]) < MIN_ARTICLES:
# 		del word_uses[word]

# save raw data

words = list(word_freq.keys())
words.sort(key=lambda w: word_freq[w], reverse=True)

with open("results/word_freq_it.csv", "w") as the_file:
    the_file.write("word, freq \n")
    for word in words:
        the_file.write('"%s", %d \n' % (word, word_freq[word]))

with open("results/word_docs_it.csv", "w") as the_file:
    the_file.write("word, docs \n")
    for word in words:
        docs = word_docs[word]
        the_file.write('"%s", %s \n' % (word, docs))

with open("results/docs_count.txt", "w") as the_file:
    the_file.write("%d \n" % doc_no)

# remove words only used once
import copy

word_freq2 = copy.deepcopy(word_freq)

for word in list(word_freq2.keys()):
    if len(word_docs[word]) < MIN_ARTICLES:
        del word_freq2[word]

# save raw data

words = list(word_freq2.keys())
words.sort(key=lambda w: word_freq2[w], reverse=True)

with open("results/word_freq_it_minfreq3.csv", "w") as the_file:
    the_file.write("word, freq \n")
    for word in words:
        the_file.write('"%s", %d \n' % (word, word_freq2[word]))

with open("results/word_docs_it_minfreq3.csv", "w") as the_file:
    the_file.write("word, docs \n")
    for word in words:
        docs = word_docs[word]
        the_file.write('"%s", %s \n' % (word, docs))

with open("results/docs_count_minfreq3.txt", "w") as the_file:
    the_file.write("%d \n" % doc_no)
