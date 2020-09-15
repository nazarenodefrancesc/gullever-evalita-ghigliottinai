import datetime

from texmega.texmega import *
from texmega.texmega_utils import *

import logging.config
from logging_conf import LOGGING_CONF

log = logging.getLogger()
logging.config.dictConfig(LOGGING_CONF)

# Requires: http://hlt.isti.cnr.it/wordembeddings/skipgram_wiki_window10_size300_neg-samples10.tar.gz
# Requires: http://hlt.isti.cnr.it/wordembeddings/glove_wiki_window10_size300_iteration50.tar.gz

# Presence of Neural Networks: GloVe does not use neural networks while word2vec does. In GloVe, the loss function is
# the difference between the product of word embeddings and the log of the probability of co-occurrence. We try to
# reduce that and use SGD but solve it as we would solve a linear regression. While in the case of word2vec,
# we either train the word on its context (skip-gram) or train the context on the word (continuous bag of words)
# using a 1-hidden layer neural network. Global information: word2vec does not have any explicit global information
# embedded in it by default. GloVe creates a global co-occurrence matrix by estimating the probability a given word
# will co-occur with other words. This presence of global information makes GloVe ideally work better. Although in a
# practical sense, they work almost similar and people have found similar performance with both.


if __name__ == "__main__":

    log.info("Started TexMEGA_py.\n")

    now_date = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M")

    if config.TEST_PATH is not None:
        log.info(f"Loading TEST_PATH:: {config.TEST_PATH}")
        config.WORD_LISTS, config.SOLUTIONS = load_test_json(config.TEST_PATH)
    else:
        log.info(f"No TEST_PATH defined. Using config guillotines.")

    log.info(
        f"Working on {len(config.WORD_LISTS)} guillotine. #{config.RUNS_PER_WORDLIST} run per guillotine.\n"
    )

    (
        model,
        nlp_it,
        whitewords,
        lemma_pos_cache,
        stemm_cache,
        norm_idfs,
        filter_list,
    ) = load_necessary_components()

    check_if_words_are_in_filterlist(filter_list)
    check_solutions_pos(lemma_pos_cache, whitewords)

    if not config.ONLY_MOST_SIMILAR:
        all_word_in_vocab = list(model.wv.vocab)
        all_word_in_vocab = filter_vocab(
            all_word_in_vocab,
            [],
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
        log.info(f"Vocab filtered size:: {len(all_word_in_vocab)}")

    guillotine_count = 0
    solution_positions = []
    exec_times = []
    for wordlist, solution in zip(config.WORD_LISTS, config.SOLUTIONS):

        guillotine_count += 1
        log.info(
            f"config:\nRUN_ID {config.RUN_ID}\nWORD_LIST {wordlist}\nSOLUTION {solution}\n"
            f"N_GEN {config.N_GEN}\nN_POP {config.N_POP}\nCROSS_PROB {config.CROSS_PROB}\n"
            f"EMBEDDINGS_MODEL_PATH {config.EMBEDDINGS_MODEL_PATH}\nINDIVIDUAL_MAX_LENGTH "
            f"{config.INDIVIDUAL_MAX_LENGTH}\nMUT_PROB {config.MUT_PROB}\n"
            f"MAX_RESULTS {config.MAX_RESULTS}\nRESTRICT_VOCAB {config.RESTRICT_VOCAB}"
            f"\nSTAGNATION_STOP {config.STAGNATION_STOP}\nEVO_PATIENCE {config.EVO_PATIENCE}\n"
            f"TOURNAMENT_SIZE {config.TOURNAMENT_SIZE}\nWEIGHTS {config.WEIGHTS}\nKEEP_POS {config.KEEP_POS}\n"
        )

        # Checking wordlist, removing not existing words
        log.info(f"Checking wordlist...")
        wordlist = checkWordlist(model, lemma_pos_cache, wordlist)

        if len(wordlist) < 2:
            log.error(
                f"Wordlist size is {len(wordlist)} < 2. Need at least two words to start the algorithm. Wordlist will be skipped."
            )
            continue

        if config.SEARCH_MODE == "genetic_algorithm":
            min, max = findMinMaxValueRange(model, model.wv.vocab)  # TODO: solo per ga
            log.info(
                f"In vocab, min feature: {min}, max feature: {max}"
            )  # TODO: come faccio ad avere valori normalizzati?

            # TEXMEGA
            toolbox = generateToolbox(
                model, wordlist, min, max, norm_idfs=norm_idfs
            )  # TODO: questo non fa funzionare la norm_idf se serve

        # Utility variables
        best_matches_sorted_all = []
        best_positions_sorted = []
        best_positions_not_sorted = []
        best_fitnesses = []
        most_common_best_matches = None
        position_freq_not_sorted = None
        position_freq_sorted = None
        best_matches = None
        solution_match = None
        found = False
        best_individual = None

        for i in range(config.RUNS_PER_WORDLIST):

            start_time = time.time()

            if config.SEARCH_MODE == "exhaustive_search":

                if config.ONLY_MOST_SIMILAR:
                    all_word_in_vocab = get_most_similar(
                        wordlist,
                        model,
                        filter_list,
                        lemma_pos_cache,
                        stemm_cache,
                        whitewords,
                        nlp_it,
                        norm_idfs,
                    )

                (
                    best_matches,
                    found,
                    solution_position,
                    solution_match,
                ) = exhaustive_search(
                    all_word_in_vocab,
                    model,
                    wordlist,
                    solution,
                    norm_idfs,
                    lemma_pos_cache,
                    stemm_cache,
                    best_positions_not_sorted,
                    best_positions_sorted,
                )

            elif config.SEARCH_MODE == "genetic_algorithm":
                (
                    best_matches,
                    found,
                    solution_position,
                    solution_match,
                    position_freq_not_sorted,
                    position_freq_sorted,
                ) = genetic_algorithm(
                    model,
                    toolbox,
                    wordlist,
                    solution,
                    norm_idfs,
                    filter_list,
                    nlp_it,
                    whitewords,
                    lemma_pos_cache,
                    best_fitnesses,
                    best_positions_not_sorted,
                    best_positions_sorted,
                )
            else:
                raise ValueError(config.SEARCH_MODE)

            most_common_best_matches = final_stats(
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
            )

        log.info(f"End of run:: {config.RUN_ID}")
        # save_log(
        #     wordlist,
        #     solution,
        #     best_matches,
        #     most_common_best_matches,
        #     best_fitnesses,
        #     found,
        #     solution_match,
        #     best_individual,
        #     position_freq_not_sorted,
        #     position_freq_sorted,
        #     best_positions_not_sorted,
        #     best_positions_sorted,
        #     exec_times,
        #     now_date,
        # )
