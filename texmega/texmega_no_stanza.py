import logging.config
import logging.config
import math
import random
import time

import numpy
from deap import creator, base, tools, algorithms
from numba import jit, numba
from scipy import spatial

import config
from logging_conf import LOGGING_CONF
from texmega import texmega_utils_no_stanza

log = logging.getLogger()
logging.config.dictConfig(LOGGING_CONF)


def cosinSimilarity(vector_one, vector_two):
    return 1 - spatial.distance.cosine(vector_one, vector_two)


@numba.jit(target="cpu", nopython=True, debug=False)
def compiled_euclidean_distance(vector_one, vector_two):
    summ = 0
    for a, b in zip(vector_one, vector_two):
        summ += math.pow(a - b, 2)
    # math.sqrt(numpy.sum([(a - b) ** 2 for a, b in zip(vector_one, vector_two)]))
    return 1 / math.sqrt(summ)


@jit(target="cpu", nopython=True, debug=False)
def compiledCosinSimilarity(vector_one, vector_two):

    # assert(vector_one.shape[0] == vector_two.shape[0])
    uv = 0
    uu = 0
    vv = 0
    for i in range(len(vector_one)):
        uv += vector_one[i] * vector_two[i]
        uu += vector_one[i] * vector_one[i]
        vv += vector_two[i] * vector_two[i]
    cos_theta = 1
    if uu != 0 and vv != 0:
        cos_theta = uv / numpy.sqrt(uu * vv)
    return cos_theta


@numba.jit(target="cpu", nopython=True)
def compiledCosinSimilarity2(vector_one, vector_two):

    udotv = 0
    u_norm = 0
    v_norm = 0
    for i in range(len(vector_one)):
        if (numpy.isnan(vector_one[i])) or (numpy.isnan(vector_two[i])):
            continue

        udotv += vector_one[i] * vector_two[i]
        u_norm += vector_one[i] * vector_one[i]
        v_norm += vector_two[i] * vector_two[i]

    u_norm = numpy.sqrt(u_norm)
    v_norm = numpy.sqrt(v_norm)

    if (u_norm == 0) or (v_norm == 0):
        ratio = 1.0
    else:
        ratio = udotv / (u_norm * v_norm)
    return ratio


def fitnessEvaluation(
    model, wordlist, norm_idfs, individual, suggested_solution=None
):  # shitty code

    cosins = []
    for word in wordlist:
        try:
            word_array = model.wv[word]

            try:
                # cosin = compiledCosinSimilarity(numpy.array(individual), word_array)
                cosin = compiledCosinSimilarity2(numpy.array(individual), word_array)
            except Exception as e:
                log.error(
                    f"compiledCosinSimilarity error, using non-compiled version. Wordlist:: {wordlist}"
                )
                log.error(e)
                cosin = cosinSimilarity(numpy.array(individual), word_array)

            if (
                cosin < config.COSIN_PENALITY_THRESHOLD
            ):  # se il coseno non supera una certa soglia, non lo considero nemmeno
                cosin = -1

            cosins.append(cosin)
        except Exception as e:
            log.info(e)

    summ = 0
    if norm_idfs is not None:  # media pesata
        for word, cosin in zip(wordlist, cosins):
            try:
                norm_idf = norm_idfs[word]  # if norm_idfs[word] > 0.6 else 0.00001
                summ += cosin * norm_idf
            except Exception as e:
                # log.info(e)
                # log.info(
                #     f"WARNING: NormIdf for word '{word}' does not exist! Setting summ to 0. TODO: check me!!!"
                # )
                summ = 0
                break
        mean = (summ / len(wordlist)) * config.MEAN_WEIGHT
    else:  # media standard
        mean = numpy.mean(cosins) * config.MEAN_WEIGHT

    std = numpy.std(cosins) * config.STD_WEIGHT

    fitness = mean / (1 + std)

    # cosin_major_than_thres = (numpy.array(cosins) > config.COSIN_THRESHOLD).sum()
    # fitness += cosin_major_than_thres

    return [fitness]  # TODO: fare una multi fitness con max_SUM, max_MEAN, min_STD


def mutByFunction(individual, indpb, mutate_function):
    for i in range(len(individual)):
        if random.random() < indpb:
            individual[i] = mutate_function()

    return [individual]


def eaSimpleWithStagnationStopCriteria(
    population,
    toolbox,
    cxpb,
    mutpb,
    ngen,
    stats=None,
    hall_of_fame=None,
    stagnation_stop=True,
    patience=10,
    verbose=__debug__,
):
    logbook = tools.Logbook()
    logbook.header = ["gen", "nevals"] + (stats.fields if stats else [])

    # Evaluate the individuals with an invalid fitness
    invalid_ind = [ind for ind in population if not ind.fitness.valid]
    fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
    for ind, fit in zip(invalid_ind, fitnesses):
        ind.fitness.values = fit

    if hall_of_fame is not None:
        hall_of_fame.update(population)

    record = stats.compile(population) if stats else {}
    logbook.record(gen=0, nevals=len(invalid_ind), **record)
    if verbose:
        log.info(logbook.stream)

    stagnation_generation = 0

    # Begin the generational process
    for gen in range(1, ngen + 1):

        # Select the next generation individuals
        offspring = toolbox.select(population, len(population))

        # Vary the pool of individuals
        offspring = algorithms.varAnd(offspring, toolbox, cxpb, mutpb)

        # Evaluate the individuals with an invalid fitness
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit

        # Update the hall of fame with the generated individuals
        if hall_of_fame is not None:
            old_best = hall_of_fame[0]

            hall_of_fame.update(offspring)

            maybe_new_best = hall_of_fame[0]
            if old_best.fitness != maybe_new_best.fitness:
                stagnation_generation = 0
            else:
                stagnation_generation += 1

        if verbose:
            log.info(
                f"*** Current best fitness at generation #{gen} is: {hall_of_fame[0].fitness}"
            )

        # Replace the current population by the offspring
        population[:] = offspring

        # Append the current generation statistics to the logbook
        record = stats.compile(population) if stats else {}
        logbook.record(gen=gen, nevals=len(invalid_ind), **record)
        if verbose:
            log.info(logbook.stream)

        if stagnation_stop and stagnation_generation > patience:
            log.info(
                f"Generation #{gen}/{ngen} -> Stagnation stopping criteria reached. {stagnation_generation} > {patience}"
            )
            break

    return population, logbook


def startGA(
    toolbox=None,
    ngen=100,
    n_pop=10,
    n_hof=2,
    cross_pb=0.5,
    mut_pb=0.1,
    stagnation_stop=True,
    patience=10,
):
    pop = toolbox.population(n_pop)
    hof = tools.HallOfFame(n_hof)  # it is not cloning

    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg-combined", numpy.mean)
    stats.register("std-combined", numpy.std)
    stats.register("max-combined", numpy.max)
    stats.register("min-combined", numpy.min)
    stats.register("avg-all", numpy.mean, axis=0)
    stats.register("std-all", numpy.std, axis=0)
    stats.register("max-all", numpy.max, axis=0)
    stats.register("min-all", numpy.min, axis=0)

    population, evolution_log = eaSimpleWithStagnationStopCriteria(
        pop,
        toolbox,
        cxpb=cross_pb,
        mutpb=mut_pb,
        ngen=ngen,
        stats=stats,
        hall_of_fame=hof,
        stagnation_stop=stagnation_stop,
        patience=patience,
        verbose=config.VERBOSE,
    )

    return hof, population, evolution_log


def initPopulation(model=None, individual_creator=None, wordlist=None, n_pop=100):
    population = []
    for i in range(
        n_pop
    ):  # TODO: ora sono tutti uguali alla wordlist, poi aggiungere parole vicine in una certa percentuale
        individual = individual_creator()
        random_from_wordlist = random.choice(wordlist)
        individual += model.wv[random_from_wordlist]
        population.append(individual)

    return population


def generateToolbox(model, wordlist, min, max, norm_idfs=None):

    # min = -1
    # max = 1

    # Creator
    creator.create("Fitness", base.Fitness, weights=config.WEIGHTS)
    creator.create("Individual", list, fitness=creator.Fitness)
    # TODO: Forse meglio un numpy per il coseno: https: // deap.readthedocs.io / en / master / examples / ga_onemax_numpy.html

    # Toolbox
    toolbox = base.Toolbox()

    # Attribute generator
    toolbox.register("attr_randomFloat", random.uniform, min, max)

    # Structure initializers
    toolbox.register(
        "individual",
        tools.initRepeat,
        creator.Individual,
        toolbox.attr_randomFloat,
        config.INDIVIDUAL_MAX_LENGTH,
    )
    toolbox.register("population", initPopulation, model, creator.Individual, wordlist)

    # Genetic Operators
    toolbox.register("mate", tools.cxTwoPoint)
    toolbox.register(
        "mutate",
        mutByFunction,
        indpb=config.MUT_PROB,
        mutate_function=toolbox.attr_randomFloat,
    )
    toolbox.register("select", tools.selTournament, tournsize=config.TOURNAMENT_SIZE)

    # Evaluation function
    toolbox.register("evaluate", fitnessEvaluation, model, wordlist, norm_idfs)

    return toolbox


def calculate_batch_cosin_similarity(word_batch, model, wordlist, norm_idfs):
    result_list = []
    for word in word_batch:
        try:
            result_list.append(
                (word, fitnessEvaluation(model, wordlist, norm_idfs, model[word]))
            )
        except Exception as e:
            log.error(f"Error with '{word}' in batch {word_batch}! {e}")
    return result_list


def sort_vocab_by_fitness(
    all_word_in_vocab, model, wordlist, norm_idfs, lemma_pos_cache, stemm_cache
):

    best_matches = []
    for word_tuple in all_word_in_vocab:
        word = word_tuple[0]
        pos = word_tuple[1]
        if not texmega_utils_no_stanza.isInWordList(word, wordlist, lemma_pos_cache, stemm_cache):
            fitness = fitnessEvaluation(model, wordlist, norm_idfs, model[word], word)[
                0
            ]
            # cooccurence_score = get_coocurrence_score(word, wordlist)
            best_matches.append((word, pos, [fitness]))

    best_matches = sorted(best_matches, reverse=True, key=lambda tuple: tuple[2])

    log.info(f"Bests matches:: {best_matches[:10]}")
    return best_matches


def getExpansionFitness(word, all_word_in_vocab, model, wordlist, norm_idfs):
    from nltk.corpus import wordnet as wn

    lemmas = wn.lemmas(word, "n", lang="ita")

    word_expansions = set()
    for lemma in lemmas:
        # print(lemma)
        syns = lemma.synset().lemmas(lang="ita")
        hyps = lemma.synset().hypernyms()
        hyponyms = lemma.synset().hyponyms()
        # print(syns)
        for syn in syns:
            word_expansions.add(syn._name)
        for hyp in hyps:
            for lemma in hyp.lemmas(lang="ita"):
                # print(lemma._name)
                word_expansions.add(lemma._name)
        for hypo in hyponyms:
            for lemma in hypo.lemmas(lang="ita"):
                # print(lemma._name)
                word_expansions.add(lemma._name)
    print(word_expansions)

    word_expansions_only_idf = [
        word for word in word_expansions if word in norm_idfs.keys()
    ]
    word_expansions = sorted(
        word_expansions_only_idf, reverse=False, key=lambda tuple: norm_idfs[tuple]
    )
    word_expansions = set(word_expansions[:3])
    word_expansions.add(word)
    print(word_expansions)

    fitness_expansion_mean = 0
    for expansion in word_expansions:
        if expansion in [tuple[0] for tuple in all_word_in_vocab]:
            fitness_expansion_mean += fitnessEvaluation(
                model, wordlist, norm_idfs, model[expansion]
            )[0]
    fitness_expansion_mean = fitness_expansion_mean / len(word_expansions)

    print(f"fitness_expansion_mean for {word} is:: {fitness_expansion_mean}")

    return fitness_expansion_mean


import pickle

log.info("Loading pmi_dictionary and coocurrence_matrix...")
cooccurence_matrix = pickle.load(open(config.COOCURRENCE_MATRIX_PATH, "rb"))
# frequency_matrix = pickle.load(open(f"./frequency_matrix_texmega_corpus.pkl", "rb"))
pmi_dictionary = pickle.load(open(config.PMI_DICTIONARY_PATH, "rb"))


def get_coocurrence_score(word_one, word_two):
    t = tuple(sorted([word_one, word_two]))
    numerator = cooccurence_matrix[t] if cooccurence_matrix[t] > 0 else 1
    den_one = pmi_dictionary[t[0]]
    den_two = pmi_dictionary[t[1]]

    pmi = (
        numpy.log(numerator / (den_one * den_two))
        if (numerator != 0 and den_one != 0 and den_two != 0)
        else -100000
    )

    return pmi


def get_coocurrence_score_with_wordlist(best_match_word, wordlist, verbose=True):
    cooc_summ = 0
    for word_in_wordlist in wordlist:
        try:
            cooc_summ += get_coocurrence_score(best_match_word, word_in_wordlist)
        except Exception as e:
            log.error(e)

        # cooc_summ += cooccurence_matrix[tuple(sorted([best_match, word_in_wordlist]))]
    log.info(f"cooc_summ for {best_match_word}:: {cooc_summ}") if verbose else None

    return cooc_summ


def reorderFirstResults(best_matches, wordlist):

    best_one_fscore = best_matches[0][2][0]
    best_two_fscore = best_matches[1][2][0]
    if (
        abs(best_one_fscore - best_two_fscore) >= config.REORDER_THRESHOLD
    ):  # se sono molto sicuro della prima soluzione, non riordino
        log.info(
            f"No need to reorder. Abs-Difference beetween first and second best_match is:: {abs(best_one_fscore - best_two_fscore)} >= {config.REORDER_THRESHOLD}"
        )
        for best_match in best_matches[:3]:
            log.info(
                f"best_match_not_reordered:: {best_match[0]}, "
                f"cooc_score:: {get_coocurrence_score_with_wordlist(best_match[0], wordlist)}, "
                f"mult_fit:: {(get_coocurrence_score_with_wordlist(best_match[0], wordlist) * config.COOCURRENCE_WEIGHT + 1000) * best_match[2][0]}"
            )
        return best_matches

    best_matches_reordered = sorted(
        best_matches[: config.REORDERED_RESULTS],
        reverse=True,
        key=lambda tuple: (
            get_coocurrence_score_with_wordlist(tuple[0], wordlist, verbose=False)
            * config.COOCURRENCE_WEIGHT
            + 1000
        )
        * tuple[2][0],
    )

    for best_match in best_matches_reordered[:3]:
        log.info(
            f"best_match_reorederd:: {best_match[0]}, "
            f"cooc_score:: {get_coocurrence_score_with_wordlist(best_match[0], wordlist)}, "
            f"mult_fit:: {(get_coocurrence_score_with_wordlist(best_match[0], wordlist) * config.COOCURRENCE_WEIGHT + 1000) * best_match[2][0]}"
        )

    best_matches = best_matches_reordered + best_matches[config.REORDERED_RESULTS :]
    log.info(f"First 20 best_matches reordered:: {best_matches[:20]}")

    return best_matches


def exhaustive_search(
    all_word_in_vocab,
    model,
    wordlist,
    solution,
    norm_idfs,
    lemma_pos_cache,
    stemm_cache,
    best_positions_not_sorted,
    best_positions_sorted,
):

    log.info(
        f"Beginning exhaustive search for guillotine:: {[(word, lemma_pos_cache[word])for word in wordlist]}..."
    )

    start_time = time.time()

    best_matches = sort_vocab_by_fitness(
        all_word_in_vocab, model, wordlist, norm_idfs, lemma_pos_cache, stemm_cache
    )
    best_matches = texmega_utils_no_stanza.strip_lower_fix(best_matches)
    solution_match = best_matches[0]

    if config.REORDERED_RESULTS is not None:
        best_matches = reorderFirstResults(best_matches, wordlist)
        solution_match = best_matches[0]

    end_time = time.time() - start_time
    log.info(f"Time taken in singleprocess:: {end_time}s")

    solution_position, found = texmega_utils_no_stanza.search_solution(best_matches, solution, lemma_pos_cache)

    if not found and solution is not None:
        all_words = [word[0] for word in all_word_in_vocab]
        log.warning(
            f"Solution {solution} not found. Is present in vocab? {solution in all_words}"
        )
        best_positions_not_sorted.append(0)
        best_positions_sorted.append(0)
        solution_position = 0

    best_positions_not_sorted.append(solution_position)

    return best_matches, found, solution_position, solution_match


def genetic_algorithm(
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
):

    # Starting Evolution
    log.info(f"Evolving words for guillotine:: {wordlist}...")

    hof, population, evolution_log = startGA(
        toolbox=toolbox,
        ngen=config.N_GEN,
        n_pop=config.N_POP,
        n_hof=config.N_HOF,
        cross_pb=config.CROSS_PROB,
        mut_pb=config.MUT_PROB,
        stagnation_stop=config.STAGNATION_STOP,
        patience=config.EVO_PATIENCE,
    )

    # Print Best & Log
    best_individual = hof[0]
    best_fitnesses.append(best_individual.fitness.getValues()[0])
    # log.info(F"{evolution_log}")
    log.info(f"Best Fitness:: {best_individual.fitness} -> {best_individual}")
    log.info(
        f"Best min value: {numpy.amin(best_individual)}, max value: {numpy.amax(best_individual)}"
    )

    (
        best_matches_sorted,
        found,
        solution_position,
        solution_match,
        position_freq_not_sorted,
        position_freq_sorted,
    ) = texmega_utils_no_stanza.findBestMatch(
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
    )

    return (
        best_matches_sorted,
        found,
        solution_position,
        solution_match,
        position_freq_not_sorted,
        position_freq_sorted,
    )