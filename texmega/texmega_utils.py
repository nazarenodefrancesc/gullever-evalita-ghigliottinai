import json
import math
import os
import pickle
import re
import time
from collections import Counter
from copy import copy

import nltk
from xml.etree import ElementTree as et

import numpy
import pandas as pd
import stanza
from gensim.models import Word2Vec, FastText

from matplotlib import pyplot
from py7zr import py7zr
from sklearn.decomposition import PCA
from spacy_stanza import StanzaLanguage
from stop_words import get_stop_words
from nltk.corpus import stopwords

import texmega.texmega as texmega

if not os.path.exists("../resources/stopwords.zip"):
    nltk.data.path.append("../resources")
    nltk.download("stopwords", download_dir="../resources")


import config

import logging.config
from logging_conf import LOGGING_CONF

log = logging.getLogger()
logging.config.dictConfig(LOGGING_CONF)


def load_xml(xml_file, df_cols):
    """Parse the input XML file and store the result in a pandas
	DataFrame with the given columns.

	The first element of df_cols is supposed to be the identifier
	variable, which is an attribute of each node element in the
	XML data; other features will be parsed from the text content
	of each sub-element.
	"""

    xtree = et.parse(xml_file)
    xroot = xtree.getroot()
    rows = []

    for node in xroot:
        res = []
        res.append(node.attrib.get(df_cols[0]))
        for el in df_cols[1:]:
            if node is not None and node.find(el) is not None:
                res.append(node.find(el).text)
            else:
                res.append(None)
        rows.append({df_cols[i]: res[i] for i, _ in enumerate(df_cols)})

    out_df = pd.DataFrame(rows, columns=df_cols)

    return out_df


def findMinMaxValueRange(model, vocab):
    max_value = None
    min_value = None
    for word in vocab.keys():
        max = numpy.amax(model.wv[word])
        min = numpy.amin(model.wv[word])
        if max_value is None or max > max_value:
            max_value = max
        if min_value is None or min < min_value:
            min_value = min
    # log.info(word, model.wv.vocab[word])  # frequency stats

    return min_value, max_value


def getWordArrayByWord(model=None, word=None):
    try:
        return model.wv[word[0][0]]
    except Exception as e:
        # log.info(e)
        return numpy.array(300, dtype=numpy.float32)


def isInWordList(word, wordlist, lemma_pos_cache, stemm_cache):

    # controllo che la parola, il suo lemma e il suo stemming non siano presenti nella wordlist e nella wordlist lemmatizzata

    word = word.lower()
    word_lemma = (
        lemma_pos_cache[word][0].lower()
        if (lemma_pos_cache is not None) and (word in lemma_pos_cache)
        else "NULL"
    )
    word_stemm = stemm_cache[word] if word in stemm_cache.keys() else word

    wordlist_lemmatized = [lemma_pos_cache[word][0].lower() for word in wordlist]
    wordlist_stemmized = [stemm_cache[word] for word in wordlist]

    if (
        (word in wordlist)
        or (word in wordlist_lemmatized)
        or (word in wordlist_stemmized)
    ):
        return True
    if (
        (word_lemma in wordlist)
        or (word_lemma in wordlist_lemmatized)
        or (word_lemma in wordlist_stemmized)
    ):
        return True
    if (
        (word_stemm in wordlist)
        or (word_stemm in wordlist_lemmatized)
        or (word_stemm in wordlist_stemmized)
    ):
        return True

    return False


def loadNormIdfs():

    start_time = time.time()
    word_norm_idf_dict_path = "resources/word_norm_idf/word_norm_idf_dict.pkl"
    word_norm_idf_dict_7zip_path = "resources/word_norm_idf/word_norm_idf_dict.7z"

    if not os.path.exists(word_norm_idf_dict_path) and os.path.exists(
        word_norm_idf_dict_7zip_path
    ):
        log.info(
            f"Unzipping word_norm_idf_dict cache {word_norm_idf_dict_7zip_path} to {word_norm_idf_dict_path}..."
        )
        archive = py7zr.SevenZipFile(word_norm_idf_dict_7zip_path, mode="r")
        archive.extractall(path="resources/")
        archive.close()
    elif not os.path.exists(word_norm_idf_dict_path) and not os.path.exists(
        word_norm_idf_dict_7zip_path
    ):

        log.info(f"Generating new cache {word_norm_idf_dict_path}")
        doc_count = int(
            open(
                "wikipedia-word-frequency-master/results/docs_count_minfreq3.txt", "r"
            ).read()
        )
        max_possible_idf = math.log(doc_count / 1)

        word_norm_idf_dict = {}
        for line in open(
            "wikipedia-word-frequency-master/results/word_docs_it_minfreq3.csv", "r"
        ).readlines()[1:]:
            word = line.split(",")[0].replace('"', "")
            # if word in wordlist and len(found) < len(wordlist):
            docs = line.split(", {")[1]
            num_docs = docs.count(",") + 1
            idf = math.log(doc_count / num_docs)
            norm_idf = idf / max_possible_idf
            word_norm_idf_dict[word] = norm_idf
        # del line

        pickle.dump(
            word_norm_idf_dict,
            open("resources/word_norm_idf/word_norm_idf_dict.pkl", "wb"),
        )
        exec_time = time.time() - start_time
        log.info("Exec time:: " + str(exec_time) + "s")

    log.info(f"Loading {word_norm_idf_dict_path}")
    word_norm_idf_dict = pickle.load(open(word_norm_idf_dict_path, "rb"))

    return word_norm_idf_dict


# se almeno una delle pos è nella whitelist delle pos
def has_at_least_one_permitted_pos(word_pos_list):
    for pos in word_pos_list:
        if pos in config.KEEP_POS:
            return True
    return False


def filter_by_pos(
    best_matches, nlp_it, whitewords, lemma_pos_cache, solution, keep_pos=None
):

    log.info(f"Lemmatizing {len(best_matches)} filtered best_matches...")

    if keep_pos is None:
        keep_pos = ["NOUN", "ADJ"]

    best_matches_filtered_by_pos = []

    for word_tuple in best_matches:
        word = word_tuple[0]
        score = word_tuple[1]

        word_pos_list = []
        if config.USE_LEMMA_POS_CACHE and word in lemma_pos_cache.keys():
            lemma = lemma_pos_cache[word][0]
            word_pos_list = lemma_pos_cache[word][1]
        else:
            for doc in nlp_it.pipe([word]):
                if doc.is_parsed:
                    for doc_word in doc:
                        lemma = doc_word.lemma_
                        word_pos_list.append(doc_word.pos_)
                        # break  # prendo solo il primo (funziona?)

        if solution is not None and word == solution:
            best_matches_filtered_by_pos.append((word, word_pos_list, score))
            if (
                not has_at_least_one_permitted_pos(word_pos_list)
                and "ALL" not in keep_pos
            ):
                log.warning(
                    f"Solution found '{word}' but its pos '{word_pos_list} is not in keep_pos:: {keep_pos}!!!'"
                )
                log.warning(f"Present in whitewords list:: {word in whitewords}")
                if not word in whitewords:
                    raise ValueError(f"Solution '{word}' non in whitelist!!!")
        elif "ALL" in keep_pos:
            best_matches_filtered_by_pos.append((word, word_pos_list, score))
        elif word in whitewords:
            best_matches_filtered_by_pos.append((word, word_pos_list, score))
        elif has_at_least_one_permitted_pos(word_pos_list):
            best_matches_filtered_by_pos.append((word, word_pos_list, score))

    return best_matches_filtered_by_pos


has_non_letters = re.compile("(?i)\b[/\\\(){}[\]'\"<>-]+\b")


def filter_words_and_lemmas_from_wordlist_and_by_filterlist(
    wordlist, solution, best_matches, filter_list, lemma_pos_cache, stemm_cache
):
    log.info(f"Filtering {len(best_matches)} best_matches by list...")

    filtered_list = []
    for word_tuple in best_matches:
        word = word_tuple[0]
        if solution is not None and word == solution:
            filtered_list.append(word_tuple)
        elif (word not in filter_list) and (
            not isInWordList(word, wordlist, lemma_pos_cache, stemm_cache)
        ):
            filtered_list.append(word_tuple)

    return filtered_list


def strip_lower_fix(best_matches):
    # tutto tu lower e strippato
    best_matches = [
        (tuple[0].strip().lower(), tuple[1], tuple[2]) for tuple in best_matches
    ]
    # fix "1treatro"
    best_matches = [
        (re.sub(r"^[0-9]*", "", tuple[0]), tuple[1], tuple[2]) for tuple in best_matches
    ]
    # rimuovo parole con <= 1 lettere
    best_matches = [best_match for best_match in best_matches if len(best_match[0]) > 1]

    return best_matches


def filter_vocab(
    all_word_in_vocab,
    wordlist,
    solution,
    filter_list,
    lemma_pos_cache,
    stemm_cache,
    whitewords,
    nlp_it,
    keep_pos,
    norm_idfs,
    is_vocab=False,
):

    if is_vocab:
        all_word_in_vocab = [(best_match, 0.0) for best_match in all_word_in_vocab]

    log.info(
        f"All_words not filtered:: {len(all_word_in_vocab)}, {all_word_in_vocab[:10]}"
    )

    # Elimino le parole indizio, filtro by filterlist usando anche il lemma
    all_words_filtered_by_filterlist = filter_words_and_lemmas_from_wordlist_and_by_filterlist(
        wordlist, solution, all_word_in_vocab, filter_list, lemma_pos_cache, stemm_cache
    )
    # log.info(f"all_words_filtered_by_filterlist:: {all_words_filtered_by_filterlist[:10]}")

    # Lemmatizzo e filtro i pos che non voglio
    all_words_filtered_by_pos = filter_by_pos(
        best_matches=all_words_filtered_by_filterlist,
        nlp_it=nlp_it,
        whitewords=whitewords,
        lemma_pos_cache=lemma_pos_cache,
        solution=solution,
        keep_pos=keep_pos,
    )
    # log.info(f"all_words_filtered_by_pos:: {all_words_filtered_by_pos[:10]}")

    # Riapplico la filter list sui lemmi
    all_words_filtered_by_filterlist = filter_words_and_lemmas_from_wordlist_and_by_filterlist(
        wordlist,
        solution,
        all_words_filtered_by_pos,
        filter_list,
        lemma_pos_cache,
        stemm_cache,
    )
    # log.info(f"all_words_filtered_by_filterlist:: {all_words_filtered_by_filterlist[:10]}")

    # Removing duplicates
    # log.info(
    #     f"De-duplicating {len(all_words_filtered_by_filterlist)} all_words_filtered_by_filterlist..."
    # )
    seen = set()
    all_words_deduplicated_filtered = [
        (word, pos, score)
        for word, pos, score in all_words_filtered_by_filterlist
        if not (word in seen or seen.add(word))
    ]
    log.info(
        f"all_words_deduplicated_filtered:: {all_words_deduplicated_filtered[:10]}"
    )

    log.info(
        f"Lemmatized, Filtered and Deduplicated best_matches list lenght:: {len(all_words_filtered_by_filterlist)}"
    )

    if config.SOLUTIONS:
        log.info(
            f"Vocab norm_idf min ::{numpy.min([norm_idfs[word[0]] for word in all_words_deduplicated_filtered if word[0] in norm_idfs.keys()])}"
        )
        log.info(
            f"Vocab norm_idf mean ::{numpy.mean([norm_idfs[word[0]] for word in all_words_deduplicated_filtered if word[0] in norm_idfs.keys()])}"
        )
        log.info(
            f"Vocab norm_idf max ::{numpy.max([norm_idfs[word[0]] for word in all_words_deduplicated_filtered if word[0] in norm_idfs.keys()])}"
        )

        log.info(
            f"Solutions norm_idf min ::{numpy.min([norm_idfs[solution] for solution in config.SOLUTIONS if solution in norm_idfs.keys()])}"
        )
        log.info(
            f"Solutions norm_idf mean ::{numpy.mean([norm_idfs[solution] for solution in config.SOLUTIONS if solution in norm_idfs.keys()])}"
        )
        log.info(
            f"Solutions norm_idf max ::{numpy.max([norm_idfs[solution] for solution in config.SOLUTIONS if solution in norm_idfs.keys()])}"
        )

    log.info(
        f"Norm-idf vocab cut:  config.NORM_IDF_UP_THRESHOLD:: {config.NORM_IDF_UP_THRESHOLD}, config.NORM_IDF_DOWN_THRESHOLD:: {config.NORM_IDF_DOWN_THRESHOLD}"
    )
    log.info(f"Vocab len before cut:: {len(all_words_deduplicated_filtered)}")
    all_words_deduplicated_filtered_idf_cutted = [
        word
        for word in all_words_deduplicated_filtered
        if word[0] in norm_idfs.keys()
        and (
            norm_idfs[word[0]] <= config.NORM_IDF_UP_THRESHOLD
            and norm_idfs[word[0]] >= config.NORM_IDF_DOWN_THRESHOLD
        )
    ]
    all_words_deduplicated_filtered = all_words_deduplicated_filtered_idf_cutted
    log.info(f"Vocab len after cut:: {len(all_words_deduplicated_filtered)}")

    return all_words_deduplicated_filtered


def search_solution(best_matches, solution, lemma_pos_cache):
    solution_position = 1
    found = False
    # se è stata data la soluzione
    if solution is not None:

        log.info(f"Solution provided:: {solution}")
        if solution in lemma_pos_cache.keys():
            solution_pos_list = lemma_pos_cache[solution][1]
            log.info(
                f"Solution pos:: {solution_pos_list}. KEEP_POS:: {config.KEEP_POS}"
            )

        for solution_match in best_matches:
            # log.info(best_match)
            if solution in solution_match[0].split("."):
                log.info(
                    f"FOUND '{solution_match}' W/Filter in position:: {solution_position}"
                )
                found = True
                break
            solution_position += 1

        # e non l'ho trovata
        if solution is not None and not found:
            log.info(
                f"Solution '{solution}' NOT FOUND in {len(best_matches)} results!'"
            )
    else:
        log.info("No solution provided.")

    return solution_position, found


def append_fitness_values(best_matches, model, wordlist, norm_idfs):
    best_matches = [
        (
            word_tuple[0],
            word_tuple[1],
            texmega.fitnessEvaluation(
                model,
                wordlist,
                norm_idfs,
                getWordArrayByWord(model=model, word=word_tuple[0]),
            ),
        )
        for word_tuple in best_matches
    ]
    log.info(f"{best_matches}")
    return best_matches


### TODO: SOLO PER GA!!!!
def filter_reorder(
    best_matches,
    model,
    wordlist,
    solution,
    filter_list,
    nlp_it,
    whitewords,
    lemma_pos_cache,
    norm_idfs,
    best_positions_not_sorted,
    best_positions_sorted,
    keep_pos=None,
):

    # # TODO: la filterlist e la lemmatizzazione può essere fatta una volta sola all'avvio di tutto, le wordlist invece le tolgo in fase di ricerca
    # best_matches = filter_best_matches(best_matches, wordlist, solution, filter_list, lemma_pos_cache, whitewords, nlp_it, keep_pos)

    # TODO:: NON SERVE TAGLIARE SE NON RIORDINO
    # TODO: prendi N solo su quelli già filtrati, se li prendo non filtrando prima rischio di non prendere la soluzione!!!
    # log.info(f"Taking only {config.TAKE_N_RESULTS} of {len(best_matches)}")  # TODO: speedup
    # best_matches = best_matches[:config.TAKE_N_RESULTS]
    # TODO: anziché prenderne N posso prendere fino ad una certa soglia di fitness e poi riordinare in qualche modo..

    # TODO: ce ne sono tanti che sono passati a 2, vedere che viene prima, magari possono andare in blacklist quelle parole

    solution_position, found = search_solution(best_matches, solution)

    best_positions_not_sorted.append(solution_position)

    best_matches = append_fitness_values(best_matches, model, wordlist, norm_idfs)

    #######################
    #######################
    #######################

    # TODO: controllare se conviene ancora riordinare per fitness...

    # riordino in base a norm_idf togliendo le parole di cui non conosco la norm_idf
    solution_match = None
    try:
        best_matches_minus_non_idf = [
            best_match
            for best_match in best_matches
            if best_match[0] in norm_idfs.keys()
        ]
        best_matches_idf_sorted = sorted(
            best_matches_minus_non_idf,
            reverse=True,
            key=lambda tuple: tuple[2][0] * norm_idfs[tuple[0]],
        )
        log.info(f"{best_matches_idf_sorted}")
        solution_position_idf = 1
        for solution_match in best_matches_idf_sorted:
            # log.info(best_match)
            if solution in solution_match[0].split("."):
                log.info(
                    f"FOUND '{solution_match}' With Filter and Reoder by norm_idf in position:: {solution_position_idf}"
                )
                break
            solution_position_idf += 1

        if solution_position_idf < solution_position:
            log.info(
                f"!!! norm_idf_migliore:: {solution_position_idf} < {solution_position}"
            )
        else:
            log.info(
                f"!!! norm_idf_peggiore_o_uguale:: {solution_position_idf} >= {solution_position}"
            )
    except Exception as e:
        log.error(e)
    # solution_position = solution_position_idf
    # best_matches_sorted_by_fitness = best_matches_idf_sorted

    # #salvo la posizione trovata
    # best_positions_sorted.append(solution_position)

    # TODO: controllare se conviene ancora riordinare per fitness...

    return best_matches, found, solution_position, solution_match


def save_log(
    wordlist,
    solution,
    best_matches_sorted,
    most_common_best_matches,
    best_fitnesses,
    found,
    solution_match,
    best_individual,
    position_freq_not_sorted,
    position_freq_sorted,
    best_positions_not_sorted,
    best_positions_sorted,
    exec_times,
    now_date,
):

    #
    # logs_file = F"logs/RUN_" + now_date + f"[{config.RUN_ID}]"
    # with open(logs_file, 'w+', encoding="utf-8") as log_file:
    #     log_file.write(F"best_individual_fitness: {best_individual.fitness}\n")
    #     log_file.write(
    #         F"Best_fitnesses -> min: {numpy.min(best_fitnesses)}, max: {numpy.max(best_fitnesses)}, mean: {numpy.mean(best_fitnesses)}, std: {numpy.std(best_fitnesses)}\n")
    #     log_file.write(F"Not_Sorted -> Frequency of positions: {position_freq_not_sorted}\n")
    #     log_file.write(F"Sorted -> Frequency of positions: {position_freq_sorted}\n")
    #     log_file.write(
    #         F"Not_Sorted -> min: {numpy.min(best_positions_not_sorted)}, max: {numpy.max(best_positions_not_sorted)}, mean: {numpy.mean(best_positions_not_sorted)}, std: {numpy.std(best_positions_not_sorted)}\n")
    #     log_file.write(
    #         F"Sorted -> min: {numpy.min(best_positions_sorted)}, max: {numpy.max(best_positions_sorted)}, mean: {numpy.mean(best_positions_sorted)}, std: {numpy.std(best_positions_sorted)}\n")
    #     log_file.write(F"exec_time_min: {numpy.min(exec_times)}\n")
    #     log_file.write(F"exec_time_mean: {numpy.mean(exec_times)}\n")
    #     log_file.write(F"exec_time_max: {numpy.max(exec_times)}\n")
    #     log_file.write(
    #         F"config:\nWORD_LIST {wordlist}\nSOLUTION {solution}\nN_GEN {config.N_GEN}\nN_POP {config.N_POP}\nCROSS_PROB {config.CROSS_PROB}\n"
    #         F"EMBEDDINGS_MODEL_PATH {config.EMBEDDINGS_MODEL_PATH}\nINDIVIDUAL_MAX_LENGTH {config.INDIVIDUAL_MAX_LENGTH}\nMUT_PROB {config.MUT_PROB}\n"
    #         F"MAX_RESULTS {config.MAX_RESULTS}\nRESTRICT_VOCAB {config.RESTRICT_VOCAB}\nSTAGNATION_STOP {config.STAGNATION_STOP}\nEVO_PATIENCE {config.EVO_PATIENCE}\n"
    #         F"TOURNAMENT_SIZE {config.TOURNAMENT_SIZE}\nWEIGHTS {config.WEIGHTS}\n")
    #     log.info(F"Saved {logs_file}")

    # MINI-REPORT

    report_file = f"logs/mini_report.csv"
    if not os.path.exists(report_file):
        with open(report_file, "w", encoding="utf-8") as report:
            headers = (
                "run_id, "
                "date, "
                "runs, "
                "word_list, "
                "found, "
                "solution, "
                "most_common_best_match, "
                "most_common_best_matches, "
                "min_sorted, "
                "max_sorted, "
                "mean_sorted, "
                "mean_exec, "
            )
            report.write(headers)

    with open(report_file, "a", encoding="utf-8") as report:
        line = "\n"
        line += f'"' + config.RUN_ID + '",'
        line += '"' + now_date + '",'
        line += str(config.RUNS_PER_WORDLIST) + ","
        line += '"' + str(wordlist) + '",'
        line += str(found) + ","
        line += '"' + solution + '",'

        most_common_best_match_field = (
            most_common_best_matches[0] if len(most_common_best_matches) > 0 else ""
        )
        line += '"' + str(most_common_best_match_field) + '",'

        line += '"' + str(most_common_best_matches) + '",'

        line += (
            str(numpy.min(best_positions_sorted)) + ","
            if len(best_positions_sorted) > 0
            else ""
        )
        line += (
            str(numpy.max(best_positions_sorted)) + ","
            if len(best_positions_sorted) > 0
            else ""
        )
        line += (
            str(numpy.mean(best_positions_sorted)) + ","
            if len(best_positions_sorted) > 0
            else ""
        )

        line += str(numpy.mean(exec_times)) + ","

        report.write(line)

        log.info(f"Saved {report_file}")

    # COMPLETE-REPORT

    report_file = f"logs/report.csv"
    if not os.path.exists(report_file):
        with open(report_file, "w", encoding="utf-8") as report:
            headers = (
                "run_id, "
                "date, "
                "word_list, "
                "solution, "
                "runs, "
                "found, "
                "debug_last_run, "
                "best_individual_fitness_mean, "
                "most_common_best_matches, "
                "most_common_best_match, "
                "most_common_best_match_fitness, "
                "solution_match, "
                "solution_match_fitness, "
                "min_sorted, "
                "max_sorted, "
                "mean_sorted, "
                "std_sorted, "
                "position_freq, "
                "mix_exec, "
                "max_exec, "
                "mean_exec, "
                "n_pop, "
                "n_gen, "
                "tourn_size, "
                "stagnation, "
                "evo_patience, "
                "mut_prob, "
                "cross_prob, "
                "weights, "
                "keep_pos, "
                "max_results, "
                "restrict_vocab"
            )
            report.write(headers)

    with open(report_file, "a", encoding="utf-8") as report:
        line = "\n"
        line += f'"' + config.RUN_ID + '",'
        line += '"' + now_date + '",'
        line += '"' + str(wordlist) + '",'
        line += '"' + solution + '",'

        most_common_best_match_field = (
            most_common_best_matches[0] if len(most_common_best_matches) > 0 else ""
        )
        most_common_best_match_fitness = (
            most_common_best_matches[0][0][2]
            if len(most_common_best_matches) > 0
            else ""
        )

        solution_match_field = solution_match[:2] if found else ""
        solution_match_fitness = solution_match[2][0] if found else ""

        line += str(config.RUNS_PER_WORDLIST) + ","
        line += str(found) + ","
        line += '"' + str(best_matches_sorted) + '",'

        line += str(numpy.mean(best_fitnesses)) + ","

        line += '"' + str(most_common_best_matches) + '",'
        line += '"' + str(most_common_best_match_field) + '",'
        line += str(most_common_best_match_fitness) + ","
        line += '"' + str(solution_match_field) + '",'
        line += str(solution_match_fitness) + ","

        line += (
            str(numpy.min(best_positions_sorted)) + ","
            if len(best_positions_sorted) > 0
            else ""
        )
        line += (
            str(numpy.max(best_positions_sorted)) + ","
            if len(best_positions_sorted) > 0
            else ""
        )
        line += (
            str(numpy.mean(best_positions_sorted)) + ","
            if len(best_positions_sorted) > 0
            else ""
        )
        line += (
            str(numpy.std(best_positions_sorted)) + ","
            if len(best_positions_sorted) > 0
            else ""
        )
        line += (
            f'"' + str(position_freq_sorted[:5]) + '",'
            if position_freq_sorted is not None
            else ""
        )
        line += str(numpy.min(exec_times)) + ","
        line += str(numpy.max(exec_times)) + ","
        line += str(numpy.mean(exec_times)) + ","

        line += str(config.N_POP) + ","
        line += str(config.N_GEN) + ","
        line += str(config.TOURNAMENT_SIZE) + ","
        line += str(config.STAGNATION_STOP) + ","
        line += str(config.EVO_PATIENCE) + ","
        line += str(config.MUT_PROB) + ","
        line += str(config.CROSS_PROB) + ","
        line += f'"' + str(config.WEIGHTS) + '",'

        line += f'"' + str(config.KEEP_POS) + '",'
        line += f'"' + str(config.MAX_RESULTS) + '",'
        line += f'"' + str(config.RESTRICT_VOCAB) + '",'

        report.write(line)

        log.info(f"Saved {report_file}")


def load_italian_stopwords():
    # italian_stop_words_list1 = get_stop_words("it")
    italian_stop_words_list2 = stopwords.words("italian")
    italian_stop_words_list3 = [
        line.rstrip()
        for line in open("resources/stopwords-it.txt", "r", encoding="utf8").readlines()
    ]  # TODO: ATTENZIONE, ci sono parole come BUONO!
    return (
        # italian_stop_words_list1 +
        italian_stop_words_list2
        + italian_stop_words_list3
    )


def load_filter_list():
    # Italian Stopwords
    italian_stop_words_list = load_italian_stopwords()
    log.info(f"Words in italian_stop_words_list:: {len(italian_stop_words_list)}")

    # Italian verbs list  + demauro verbs list (solutions are never verbs)
    italian_verbs_list = [
        line.rstrip().lower()
        for line in open(
            "resources/verbs_list_111.txt", "r", encoding="utf8"
        ).readlines()
    ]
    demauro_verbs_list = [
        line.rstrip().lower()
        for line in open(
            "resources/demauro_verb_list.txt", "r", encoding="utf8"
        ).readlines()
    ]
    italian_verbs_list = list(set(italian_verbs_list + demauro_verbs_list))
    log.info(f"Words in italian_verbs_list:: {len(italian_verbs_list)}")

    # Blacklist list
    blacklist = [
        line.rstrip().lower()
        for line in open("resources/blacklist.txt", "r", encoding="utf8").readlines()
    ]
    log.info(f"Words in blacklist:: {len(blacklist)}")
    #
    # # IT - EN intersection list
    # it_en_intesection = [
    #     line.rstrip().lower()
    #     for line in open("resources/it_en_intersection.txt", "r", encoding="utf8").readlines()
    # ]
    # log.info(f"Words in it_en_intesection:: {len(it_en_intesection)}")

    # Whitelist list
    whitewords = [
        line.rstrip().lower()
        for line in open("resources/whitelist.txt", "r", encoding="utf8").readlines()
    ]
    log.info(f"Words in whitelist:: {len(whitewords)}")

    filter_list = set(
        italian_stop_words_list
        + italian_verbs_list
        + blacklist
        # + it_en_intesection
    )

    whitewords_in_filter_list = filter_list.intersection(whitewords)
    if len(whitewords_in_filter_list) > 0:
        log.warning(
            f"Removing these {len(whitewords_in_filter_list)} words from filter_list because they are present in whitelist:: \n{whitewords_in_filter_list}"
        )
        filter_list = filter_list.difference(whitewords_in_filter_list)

    return filter_list, whitewords


def findBestMatch(
    model,
    wordlist,
    solution,
    filter_list,
    nlp_it,
    whitewords,
    lemma_pos_cache,
    norm_idfs,
    best_individual,
    best_positions_not_sorted,
    best_positions_sorted,
):

    search_index = 0
    found = False
    solution_position = 0
    solution_match = None
    position_freq_not_sorted = None
    position_freq_sorted = None
    best_matches = None

    # cerco la soluzione esatta

    while not found and search_index < len(config.MAX_RESULTS):
        # log.info(F"Looking for solution in the first {config.MAX_RESULTS[search_index]} results...")
        best_matches = model.wv.similar_by_vector(
            numpy.array(best_individual),
            topn=config.MAX_RESULTS[search_index],
            restrict_vocab=config.RESTRICT_VOCAB,
        )  # TODO: ma questo funziona bene? Dice che si può sortare per frequency

        best_matches_sorted, found, solution_position, solution_match = filter_reorder(
            best_matches,
            model,
            wordlist,
            solution,
            filter_list,
            nlp_it,
            whitewords,
            lemma_pos_cache,
            norm_idfs,
            best_positions_not_sorted,
            best_positions_sorted,
            config.KEEP_POS,
        )
        search_index += 1

    # se non la trovo cerco il lemma

    if not found:
        solution_lemma = lemma_pos_cache[solution][0]
        log.info(f"No solution found. Trying with lemma {solution_lemma}")

        search_index = 0
        found = False
        solution_position = 0
        solution_match = None
        position_freq_not_sorted = None
        position_freq_sorted = None
        best_matches = None

        while not found and search_index < len(config.MAX_RESULTS):
            # log.info(F"Looking for solution in the first {config.MAX_RESULTS[search_index]} results...")
            best_matches = model.wv.similar_by_vector(
                numpy.array(best_individual),
                topn=config.MAX_RESULTS[search_index],
                restrict_vocab=config.RESTRICT_VOCAB,
            )  # TODO: ma questo funziona bene? Dice che si può sortare per frequency

            (
                best_matches_sorted,
                found,
                solution_position,
                solution_match,
            ) = filter_reorder(
                best_matches,
                model,
                wordlist,
                solution,
                filter_list,
                nlp_it,
                whitewords,
                lemma_pos_cache,
                norm_idfs,
                best_positions_not_sorted,
                best_positions_sorted,
                config.KEEP_POS,
            )
            search_index += 1

    if not found:
        best_positions_not_sorted.append(0)
        best_positions_sorted.append(0)
        solution_position = 0

    unique_elements, counts_elements = numpy.unique(
        best_positions_not_sorted, return_counts=True
    )
    tuples = zip(unique_elements, counts_elements)
    position_freq_not_sorted = sorted(
        tuples, reverse=True, key=lambda tuples: tuples[1]
    )

    unique_elements, counts_elements = numpy.unique(
        best_positions_sorted, return_counts=True
    )
    tuples = zip(unique_elements, counts_elements)
    position_freq_sorted = sorted(tuples, reverse=True, key=lambda tuples: tuples[1])

    log.info(f"Not_Sorted -> Frequency of positions: {position_freq_not_sorted}")
    log.info(f"Sorted -> Frequency of positions: {position_freq_sorted}")
    log.info(
        f"Not_Sorted -> min: {numpy.min(best_positions_not_sorted)}, max: {numpy.max(best_positions_not_sorted)}, "
        f"mean: {numpy.mean(best_positions_not_sorted)}, std: {numpy.std(best_positions_not_sorted)}"
    )
    log.info(
        f"Sorted -> min: {numpy.min(best_positions_sorted)}, max: {numpy.max(best_positions_sorted)}, "
        f"mean: {numpy.mean(best_positions_sorted)}, std: {numpy.std(best_positions_sorted)}"
    )

    return (
        best_matches_sorted,
        found,
        solution_position,
        solution_match,
        position_freq_not_sorted,
        position_freq_sorted,
    )


def checkWordlist(model, lemma_pos_cache, wordlist):
    filtered_wordlist = []
    for word in wordlist:
        if word in model.wv.vocab.keys():
            filtered_wordlist.append(word)
        elif (
            word in lemma_pos_cache.keys()
            and lemma_pos_cache[word][0] in model.wv.vocab.keys()
        ):
            filtered_wordlist.append(lemma_pos_cache[word][0])
            log.warning(
                f" word '{word}' does not exist in the vocabulary, using lemma {lemma_pos_cache[word][0]} instead.\n"
            )
        else:
            log.warning(
                f" word '{word}' and its lemma does not exist in the vocabulary! It will be ignored.\n"
            )

    return filtered_wordlist


def load_test_json(test_path):

    wordlists = []
    solutions = []

    with open(test_path, mode="r", encoding="UTF-8") as json_file:
        test_json = json.load(json_file)

    for guillotine in test_json:
        guillotine_list = []
        guillotine_list.append(guillotine["w1"].strip().lower())
        guillotine_list.append(guillotine["w2"].strip().lower())
        guillotine_list.append(guillotine["w3"].strip().lower())
        guillotine_list.append(guillotine["w4"].strip().lower())
        guillotine_list.append(guillotine["w5"].strip().lower())
        wordlists.append(guillotine_list)
        solutions.append(guillotine["solution"].strip().lower())

    return wordlists, solutions


def buildLemmaPosCache(model, nlp_it, cache_name):

    if not config.USE_LEMMA_POS_CACHE:
        return None

    cache_name = cache_name.replace(".pkl", "")
    cache_path = f"resources/lemma_pos_cache/lemma_pos_cache_[{cache_name}].pkl"
    cache_7zip_path = f"resources/lemma_pos_cache/lemma_pos_cache_[{cache_name}].7z"

    if not os.path.exists(cache_path) and os.path.exists(cache_7zip_path):
        log.info(f"Unzipping Lemma-POS cache {cache_7zip_path} to {cache_path}...")
        archive = py7zr.SevenZipFile(cache_7zip_path, mode="r")
        archive.extractall(path="resources/")
        archive.close()
    elif config.DEFAULT_POS_CACHE_PATH and os.path.exists(
        config.DEFAULT_POS_CACHE_PATH
    ):
        log.info(f"Loading Default Lemma-POS cache {config.DEFAULT_POS_CACHE_PATH}...")
        cache_path = config.DEFAULT_POS_CACHE_PATH
    elif not os.path.exists(cache_path) and not os.path.exists(cache_7zip_path):
        log.info(f"Generating new Lemma-POS cache in {cache_path}.")
        log.info(f"This will take some time...")
        word_pos_dict = dict()
        num_words = len(model.wv.vocab.keys())
        count = 0
        start_time = time.time()

        for word in model.wv.vocab.keys():

            for doc in nlp_it.pipe([word]):
                if doc.is_parsed:
                    for doc_word in doc:
                        word_pos_dict[word] = [doc_word.lemma_, doc_word.pos_]
                        break  # prendo solo il primo lemma

            count += 1
            elapsed_time = time.time() - start_time
            if count == 1000:
                log.info(f"ETA:: {(elapsed_time * (num_words/1000))/60} m.")
                # break # remove me
            log.info(
                f"> {count}/{num_words}. \n elapsed_time:: {elapsed_time/60} m. "
            ) if count % 1000 == 0 else None

        pickle.dump(word_pos_dict, open(cache_path, "wb"))

    log.info(f"Reloading Lemma-POS cache {cache_path}...")
    word_pos_dict = pickle.load(open(cache_path, "rb"))

    return word_pos_dict


def buildStemmCache(model):

    ita_stemmer = nltk.stem.snowball.ItalianStemmer()

    cache_path = f"resources/lemma_pos_cache/stemm_cache.pkl"
    cache_7zip_path = f"resources/lemma_pos_cache/stemm_cache.7z"

    if not os.path.exists(cache_path) and os.path.exists(cache_7zip_path):
        log.info(f"Unzipping Stemm cache {cache_7zip_path} to {cache_path}...")
        archive = py7zr.SevenZipFile(cache_7zip_path, mode="r")
        archive.extractall(path="resources/")
        archive.close()
    elif os.path.exists(cache_path):
        log.info(f"Loading Default Stemm cache:: {cache_path}...")
    elif not os.path.exists(cache_path) and not os.path.exists(cache_7zip_path):
        log.info(f"Generating new Stemm cache in:: {cache_path}.")
        log.info(f"This will take some time...")
        word_stemm_dict = dict()
        num_words = len(model.wv.vocab.keys())
        count = 0
        start_time = time.time()

        for word in model.wv.vocab.keys():
            word_stemm_dict[word] = ita_stemmer.stem(word)

            count += 1
            elapsed_time = time.time() - start_time
            if count == 1000:
                log.info(f"ETA:: {(elapsed_time * (num_words / 1000)) / 60} m.")
                # break # remove me
            log.info(
                f"> {count}/{num_words}. \n elapsed_time:: {elapsed_time / 60} m. "
            ) if count % 1000 == 0 else None

        pickle.dump(word_stemm_dict, open(cache_path, "wb"))

    log.info(f"Reloading Stemm cache {cache_path}...")
    word_stemm_dict = pickle.load(open(cache_path, "rb"))

    return word_stemm_dict


def load_necessary_components():

    # Spacy NLP pipeline
    log.info(f"Loading Stanza-IT...")
    snlp = stanza.Pipeline(
        processors="tokenize,mwt,pos,lemma", lang="it", use_gpu=False
    )  # pos_batch_size=3000
    nlp_it = StanzaLanguage(snlp)

    # Filter list
    filter_list, whitewords = load_filter_list()

    # WORD2VEC
    log.info(f"Loading Vocabulary: {config.EMBEDDINGS_MODEL_PATH}")
    start_time = time.time()
    if config.MODEL == "fasttext":
        # model = FastText.load(config.EMBEDDINGS_MODEL_PATH)
        model = FastText.load_fasttext_format(config.EMBEDDINGS_MODEL_PATH)
    elif config.MODEL == "word2vec_glove":
        model = Word2Vec.load(config.EMBEDDINGS_MODEL_PATH)
    elif config.MODEL == "pkl":
        model = pickle.load(open(config.EMBEDDINGS_MODEL_PATH, "rb"))
    else:
        raise ValueError
    log.info(f"Time taken:: {(time.time() - start_time) / 60} m.")
    log.info(f"Words in vocab:: {len(model.wv.vocab.keys())}")

    # Build LEMMA_POS cache
    cache_name = config.EMBEDDINGS_MODEL_PATH.split("/")[
        len(config.EMBEDDINGS_MODEL_PATH.split("/")) - 1
    ]
    lemma_pos_cache = buildLemmaPosCache(model, nlp_it, cache_name=cache_name)
    log.info(
        f"Words in LEMMA-POS cache:: {len(lemma_pos_cache)}"
    ) if lemma_pos_cache is not None else None

    # Build Stemm cache
    stemm_cache = buildStemmCache(model)
    log.info(f"Words in STEMM cache:: {len(stemm_cache)}")

    # Retrieving Norm_Idfs
    norm_idfs = None
    if config.USE_NORM_IDF:
        norm_idfs = loadNormIdfs()
        log.info(f"Words in NormIdfs cache:: {len(norm_idfs)}")

    return (
        model,
        nlp_it,
        whitewords,
        lemma_pos_cache,
        stemm_cache,
        norm_idfs,
        filter_list,
    )


fig_count = 0


def visualize_pca(model, wordlist, best_match, solution):
    # Visualize PCA
    # fit a 2d PCA model to the vectors
    new_wordlist = wordlist + [best_match] + [solution]

    X = model[new_wordlist]
    pca = PCA(n_components=2)
    result = pca.fit_transform(X)

    # create a scatter plot of the projection
    global fig_count
    pyplot.figure(fig_count,)
    fig_count += 1
    pyplot.scatter(result[:, 0], result[:, 1])
    pyplot.title(
        f"words: '{' '.join(new_wordlist)}', best: '{best_match}', sol: '{solution}'"
    )
    words = list(new_wordlist)
    for i, word in enumerate(words):
        pyplot.annotate(word, xy=(result[i, 0], result[i, 1]))
    pyplot.show()


def check_if_words_are_in_filterlist(filter_list):

    solutions_to_whitelist = set(config.SOLUTIONS).intersection(filter_list)
    if len(solutions_to_whitelist) > 0:
        log.error(
            f"{len(solutions_to_whitelist)} solutions are in the filter_list! Remove them before starting the algorithm or put them in the whitelist! \n{solutions_to_whitelist}"
        )
        raise ValueError

    all_word_in_test = [word for l in config.WORD_LISTS for word in l]
    wordlist_solutions_to_whitelist = set(all_word_in_test).intersection(filter_list)
    if len(wordlist_solutions_to_whitelist) > 0:
        log.warning(
            f"{len(wordlist_solutions_to_whitelist)} clues are in the filter_list! Check them and move those that can be solutions in the whitelist! \n{wordlist_solutions_to_whitelist}"
        )


def check_solutions_pos(lemma_pos_cache, whitelist):
    if lemma_pos_cache is None:
        return
    for worlist, solution in zip(config.WORD_LISTS, config.SOLUTIONS):
        if solution in lemma_pos_cache.keys() and not has_at_least_one_permitted_pos(
            lemma_pos_cache[solution][1]
        ):
            log.warning(
                f"Solution '{solution}' for guillotine '{worlist}' has POSs '{lemma_pos_cache[solution][1]}' that is not included in config.KEEP_POS:: {config.KEEP_POS}"
            )
            if config.CHECK_SOLUTION_IN_WHITELIST and solution not in whitelist:
                log.error(f"Solution '{solution}' not in whitelist! ")
                # raise ValueError(f"Solution '{solution}' not in whitelist! ")
    # time.sleep(5)


def check_if_adj_in_wordlist_pos(lemma_pos_cache, wordlist):
    wordlist_pos = []
    for word in wordlist:
        word_pos_list = lemma_pos_cache[word][1]
        for pos in word_pos_list:
            wordlist_pos.append(pos)

    backup_keep_pos = copy(config.KEEP_POS)

    if "AJ" in config.KEEP_POS and "AJ" in wordlist_pos:
        config.KEEP_POS.remove("AJ")
        log.info(f"Found AJ in wordlist. KEEP_POS now is:: {config.KEEP_POS}")

    return backup_keep_pos


def get_most_similar(
    wordlist,
    model,
    filter_list,
    lemma_pos_cache,
    stemm_cache,
    whitewords,
    nlp_it,
    norm_idfs,
):

    if config.CHECK_IF_AJ_IN_WORDLIST:
        log.info(f"CHECK_IF_AJ_IN_WORDLIST is active.")
        backup_keep_pos = check_if_adj_in_wordlist_pos(lemma_pos_cache, wordlist)

    most_similars = set()
    for word in wordlist:
        word_most_similar_set = set(
            [
                word_tuple[0]
                for word_tuple in model.most_similar(
                    word, topn=config.MOST_SIMILAR_TOP_N
                )
                if word_tuple[1] >= config.SIMILARITY_THRESHOLD
            ]
        )
        most_similars.update(word_most_similar_set)
    all_word_in_vocab = filter_vocab(
        most_similars,
        wordlist,
        "SOLUTION",
        filter_list,
        lemma_pos_cache,
        stemm_cache,
        whitewords,
        nlp_it,
        config.KEEP_POS,
        norm_idfs,
        is_vocab=True,
    )
    # all_word_in_vocab = remove_non_italian_words(all_word_in_vocab)

    if config.CHECK_IF_AJ_IN_WORDLIST:
        config.KEEP_POS = backup_keep_pos

    return all_word_in_vocab


def print_cosins_of_bestmatches(best_matches, model, norm_idfs, solution, wordlist, n):
    try:
        for i in range(0, n):
            best_match = best_matches[i]
            log.info(f"Best match #{i}:: {best_match}")
            for word in wordlist:
                log.info(
                    f"\t cosin with clue {word}:: {texmega.compiledCosinSimilarity2(model.wv[word], model.wv[best_match[0]])}"
                )

        log.info(
            f"Solution:: {solution}, {texmega.fitnessEvaluation(model, wordlist, norm_idfs, model.wv[solution])}"
        )
        for word in wordlist:
            log.info(
                f"\t cosin with clue {solution}:: {texmega.compiledCosinSimilarity2(model.wv[word], model.wv[solution])}"
            )
    except Exception as e:
        log.error(e)


def final_stats(
    best_matches,
    model,
    norm_idfs,
    solution,
    wordlist,
    best_matches_sorted_all,
    solution_position,
    found,
    solution_positions,
    start_time,
    exec_times,
    guillotine_count,
    i
):
    print_cosins_of_bestmatches(best_matches, model, norm_idfs, solution, wordlist, n=3)

    if found:
        best_matches_sorted_all.append(
            (best_matches[0][0], best_matches[0][1], best_matches[0][2][0],)
        )

    solution_positions.append(solution_position)

    # getting the 10 most common best matches
    best_matches_sorted_all = [
        (best_match_sorted[0], "_".join(best_match_sorted[1]), best_match_sorted[2])
        for best_match_sorted in best_matches_sorted_all
    ]
    mode_counter = Counter(best_matches_sorted_all)
    most_common_best_matches = mode_counter.most_common(n=10)

    exec_time = time.time() - start_time
    exec_times.append(exec_time)
    log.info(f"Most common best matches:: {most_common_best_matches}")
    log.info("Exec time:: " + str(exec_time) + "s")
    log.info("Mean time:: " + str(numpy.mean(exec_times)) + "s")
    log.info("Total time taken:: " + str(numpy.sum(exec_times)) + "s")
    log.info(f"Total Solution positions mean:: {numpy.mean(solution_positions)}")
    log.info(f"Total Solution positions min:: {numpy.min(solution_positions)}")
    log.info(f"Total Solution positions max:: {numpy.max(solution_positions)}")
    log.info(
        f"Total Solution most common matches:: {sorted(Counter(solution_positions).most_common(n=10), reverse=True, key=lambda tuple: tuple[1])}"
    )

    # Percentage of success
    solution_positions_copy = solution_positions.copy()
    # solution_positions_copy = [sol for sol in solution_positions_copy if sol > 0] #rimuovo gli zero dalle statistiche
    if len(solution_positions_copy) > 0:
        zero_percentage = (
            len([sol for sol in solution_positions_copy if sol == 0])
            / len(solution_positions_copy)
        ) * 100
        first_position_percentage = (
            len([sol for sol in solution_positions_copy if sol == 1])
            / len(solution_positions_copy)
        ) * 100
        second_position_percentage = (
            len([sol for sol in solution_positions_copy if sol <= 2 and sol > 0])
            / len(solution_positions_copy)
        ) * 100
        third_position_percentage = (
            len([sol for sol in solution_positions_copy if sol <= 3 and sol > 0])
            / len(solution_positions_copy)
        ) * 100
        under_5_percentage = (
            len([sol for sol in solution_positions_copy if sol <= 5 and sol > 0])
            / len(solution_positions_copy)
        ) * 100
        under_10_percentage = (
            len([sol for sol in solution_positions_copy if sol <= 10 and sol > 0])
            / len(solution_positions_copy)
        ) * 100
        under_20_percentage = (
            len([sol for sol in solution_positions_copy if sol <= 20 and sol > 0])
            / len(solution_positions_copy)
        ) * 100
        under_30_percentage = (
            len([sol for sol in solution_positions_copy if sol <= 30 and sol > 0])
            / len(solution_positions_copy)
        ) * 100
        under_50_percentage = (
            len([sol for sol in solution_positions_copy if sol <= 50 and sol > 0])
            / len(solution_positions_copy)
        ) * 100
        under_100_percentage = (
            len([sol for sol in solution_positions_copy if sol <= 100 and sol > 0])
            / len(solution_positions_copy)
        ) * 100
        under_200_percentage = (
            len([sol for sol in solution_positions_copy if sol <= 200 and sol > 0])
            / len(solution_positions_copy)
        ) * 100
        log.info(
            f"#0: {zero_percentage:.2f}, #1:: {first_position_percentage:.2f}%, #2:: {second_position_percentage:.2f}%, #3:: {third_position_percentage:.2f}%, #5:: {under_5_percentage:.2f}%, #10:: {under_10_percentage:.2f}%, #20:: {under_20_percentage:.2f}%, #30:: {under_30_percentage:.2f}%, #50:: {under_50_percentage:.2f}%, #100:: {under_100_percentage:.2f}%, #200:: {under_200_percentage:.2f}%"
        )

    log.info(
        f"End of #{i + 1}/{config.RUNS_PER_WORDLIST} run of the "
        f"#{guillotine_count}/{len(config.WORD_LISTS)} guillotines. \n"
    )

    if config.VISUALIZE_PCA and best_matches:
        visualize_pca(model, wordlist, best_matches[0][0], solution)

    return most_common_best_matches