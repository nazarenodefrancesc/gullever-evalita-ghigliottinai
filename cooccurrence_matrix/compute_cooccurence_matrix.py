from collections import defaultdict

import numpy
from tqdm import tqdm

import pandas as pd
import numpy as np
import pickle

import logging.config
from logging_conf import LOGGING_CONF

log = logging.getLogger()
logging.config.dictConfig(LOGGING_CONF)


def compute_co_occurrence_matrix(sentences, freq_threshold, window_size):
    cooccurence_matrix = defaultdict(int)
    frequency_matrix = defaultdict(int)
    log.info("Building frequency and cooccurence matrix.")
    log.info(f"window size:: {window_size}")
    for text in tqdm(sentences):

        text = text.lower().strip().split()

        for i in range(len(text)):
            token = text[i].split("_")[0]
            frequency_matrix[token] += 1
        # log.info(f"Done. Len:: {len(frequency_matrix)}")

        for i in range(len(text)):
            token = text[i].split("_")[0]
            next_token = text[i + 1 : i + 1 + window_size]
            for t in next_token:
                key = tuple(sorted([t, token]))

                if frequency_matrix[key[0]] >= freq_threshold:
                    cooccurence_matrix[key] += 1
            # else:
            # 	log.warning(f"{key[0]} has freq: { frequency_matrix[key[0]]}")
    log.info(f"Done. Len cooccurence_matrix:: {len(cooccurence_matrix)}")
    log.info(f"Done. Len frequency_matrix:: {len(frequency_matrix)}")

    # formulate the dictionary into dataframe
    # vocab = sorted(vocab)  # sort vocab
    # df = pd.DataFrame(data=np.zeros((len(vocab), len(vocab)), dtype=np.int16),
    #                   index=vocab,
    #                   columns=vocab)
    # for key, value in cooccurence_matrix.items():
    # 	df.at[key[0], key[1]] = value
    # 	df.at[key[1], key[0]] = value
    return cooccurence_matrix, frequency_matrix


def get_pmi(evaluated_word, frequency_matrix, cooccurence_matrix):
    summ = 0
    for word_in_freq_mat in frequency_matrix:
        summ += cooccurence_matrix[tuple(sorted([evaluated_word, word_in_freq_mat]))]

    return summ


def compute_pmi_matrix(
    frequency_matrix, cooccurence_matrix, freq_threshold, reload=False
):
    log.info("Building pmi matrix.")
    log.info(f"freq_threshold:: {freq_threshold}")

    frequency_matrix = sorted(
        [
            word
            for word in frequency_matrix.keys()
            if frequency_matrix[word] >= freq_threshold
        ]
    )

    if reload:
        lines = open(
            "./cooccurrence_matrix/pmi_matrix.csv", "r", encoding="utf8"
        ).readlines()
        last_written_word = (
            lines[len(lines) - 1].split(",")[0] if len(lines) > 0 else None
        )
        log.info(f"Last written word is:: {last_written_word}")
        del lines

        last_written_word_index = (
            frequency_matrix.index(last_written_word)
            if last_written_word is not None
            else 0
        )
        frequency_list = frequency_matrix[last_written_word_index + 1 :]
    else:
        frequency_list = list(frequency_matrix)
    del frequency_matrix

    mode = "a" if reload else "w"
    with open(
        "./cooccurrence_matrix/pmi_matrix.csv", mode, encoding="utf8"
    ) as pmi_matrix_file:

        for word_in_freq_mat in tqdm(frequency_list):
            # if word_in_freq_mat not in pmi_matrix.keys():
            # pmi_matrix[word_in_freq_mat] = get_pmi(word_in_freq_mat, frequency_matrix, cooccurence_matrix, freq_threshold) #int per ridurre lo spazio occupato
            pmi_matrix_file.write(
                word_in_freq_mat
                + ","
                + str(get_pmi(word_in_freq_mat, frequency_list, cooccurence_matrix))
                + "\n"
            )

    # return pmi_matrix


if __name__ == "__main__":
    filebasepath = "/home/nazareno/CELI/repositories/python_projects/texmega_py"
    filename = "pattern_mining_corpus_2020_09_03.txt"
    reload_matrix = True
    freq_threshold = 1
    window_size = 4  # it is a forward window

    if not reload_matrix:
        sentences = open(f"{filebasepath}/{filename}", "r", encoding="utf8").readlines()
        cooccurence_matrix, frequency_matrix = compute_co_occurrence_matrix(
            sentences, freq_threshold=freq_threshold, window_size=window_size
        )
        del sentences
        pickle.dump(
            cooccurence_matrix,
            open(f"./cooccurrence_matrix/coocurrence_{filename}.pkl", "wb"),
        )
        pickle.dump(
            frequency_matrix,
            open(f"./cooccurrence_matrix/frequency_matrix_{filename}.pkl", "wb"),
        )
    else:
        cooccurence_matrix = pickle.load(
            open(f"./cooccurrence_matrix/coocurrence_{filename}.pkl", "rb")
        )
        frequency_matrix = pickle.load(
            open(f"./cooccurrence_matrix/frequency_matrix_{filename}.pkl", "rb")
        )

    compute_pmi_matrix(
        frequency_matrix,
        cooccurence_matrix,
        freq_threshold=freq_threshold,
        reload=reload_matrix,
    )

    # pickle.dump(pmi_matrix, open(f"./pmi_matrix_{filename}.pkl", "wb"))

    lines = open(
        f"./cooccurrence_matrix/pmi_matrix.csv", "r", encoding="utf8"
    ).readlines()
    pmi_dictionary = defaultdict(int)
    for line in lines:
        pmi_dictionary[line.split(",")[0]] = int(line.split(",")[1])

    log.info("Written ./cooccurrence_matrix/pmi_matrix.csv. All done.")

    pickle.dump(
        pmi_dictionary,
        open(f"./cooccurrence_matrix/pmi_dictionary_{filename}.pkl", "wb"),
    )
    #
    # solutions = ["straniero", "fronte"]
    # wordlist = ['paese', 'patria', 'film', 'piave', 'accento']
    # cooc_summ = 0
    # for solution in solutions:
    # 	for word in wordlist:
    # 		t = tuple(sorted([solution, word]))
    # 		numerator = cooccurence_matrix[t] if cooccurence_matrix[t] > 0 else 1
    # 		den_one = pmi_dictionary[t[0]]
    # 		den_two = pmi_dictionary[t[1]]
    # 		pmi = numpy.log(numerator / (den_one * den_two))
    #
    # 		# cooc = cooccurence_matrix[tuple(sorted([solution, word]))]
    # 		cooc_summ += pmi
    # 		log.info(f"{solution} - {word}:: {cooc_summ}")
    # 	log.info(f"cooc_summ for {solution}:: {cooc_summ}")
    # 	log.info(f"cooc_mean for {solution}:: {cooc_summ / len(wordlist)}")
    # 	cooc_summ = 0
