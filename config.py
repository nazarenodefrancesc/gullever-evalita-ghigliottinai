# TexMEGA_py Config file

# MODE = "genetic_algorithm"
SEARCH_MODE = "exhaustive_search"

# MODEL = "word2vec_glove"
# MODEL = "fasttext"
MODEL = "pkl"

EMBEDDINGS_MODEL_PATH = (
    "./resources/embeddings/glove_[min200_600_win10_ite50_xmax10]_songs_2020_08_29_texmega_py_[default].pkl"  # 81
)

DEFAULT_POS_CACHE_PATH = "resources/lemma_pos_cache/lemma_pos_cache_[glove_[min100_300_win10_ite50_xmax10_041]].pkl_fixed.pkl"
COOCURRENCE_MATRIX_PATH = (
    f"./cooccurrence_matrix/coocurrence_texmega_corpus_songs.pkl"  # def
)
# COOCURRENCE_MATRIX_PATH = f"./cooccurrence_matrix/coocurrence_pattern_mining_corpus_2020_09_03.txt.pkl"
PMI_DICTIONARY_PATH = (
    f"./cooccurrence_matrix/pmi_dictionary_texmega_corpus_songs.pkl"  # def
)
# PMI_DICTIONARY_PATH = f"./cooccurrence_matrix/pmi_dictionary_pattern_mining_corpus_2020_09_03.txt.pkl"

# TEST_PATH = None
TEST_PATH = "test_datasets/ghigliottinai_dataset.json"
# TEST_PATH = "test_datasets/Ghigliottin-AI_2020-gulliver.json"

dataset_id = TEST_PATH.replace("/", "_") if TEST_PATH is not None else "Custom"
note = "whitelist_noun_adj_filterfix_poscachefix_cleanwhitelist_mean*idf"
RUN_ID = f"{SEARCH_MODE}_{dataset_id}_{EMBEDDINGS_MODEL_PATH}_#{note}"

ONLY_MOST_SIMILAR = True
SIMILARITY_THRESHOLD = (
    -1
)  # prende la parola simile solo se supera la soglia. -1 prendi tutto.
MOST_SIMILAR_TOP_N = 375  # considera solo X parole simili per ogni indizio (ie, massimo 5000 per 5 indizi)
CHECK_IF_AJ_IN_WORDLIST = True  # se c'è un aggettivo nella lista indizi, la soluzione ADJ non viene considerata
COSIN_PENALITY_THRESHOLD = (
    -0.05
)  # se un coseno è <= di questa soglia, viene penalizzato a -1
REORDERED_RESULTS = (
    75  # 2,3,5,10,20 # riordina gli ultimi X risultati secondo la formula FITNESS * PMI
)
REORDER_THRESHOLD = 0.004  # il riordino viene effettuato solo se la differenza di fitness fra il primo e il secondo risultato supera la soglia impostata
COOCURRENCE_WEIGHT = 10  # 2,3,4,5,6,7,8,9,15 # peso del PMI nella formula di riordino
STD_WEIGHT = 5.5
MEAN_WEIGHT = 0.525
NORM_IDF_UP_THRESHOLD = 0.76  # 1
NORM_IDF_DOWN_THRESHOLD = 0.13  # -1

KEEP_POS = ["N", "AJ"]  # ['NOUN', 'ADJ', 'VERB', 'PRON', 'PROPN']
CHECK_SOLUTION_IN_WHITELIST = True  # def: True
VISUALIZE_PCA = False  # TEST_PATH is None
USE_LEMMA_POS_CACHE = True
USE_NORM_IDF = True  # TODO: sistemare bug: ValueError: The truth value of an array with more than one element is ambiguous. Use a.any() or a.all()
RUNS_PER_WORDLIST = 1

# 2020-09-09 12:13:45,053 - root.final_stats() - line: 72 - INFO - Total Solution most common matches:: [(1, 10), (2, 6), (6, 1), (3, 1), (51, 1), (4, 1), (8, 1), (1015, 1), (21, 1)]
# 2020-09-09 12:13:45,053 - root.final_stats() - line: 102 - INFO - #0: 0.00, #1:: 43.48%, #2:: 69.57%, #3:: 73.91%, #5:: 78.26%, #10:: 86.96%, #20:: 86.96%, #30:: 91.30%, #50:: 91.30%, #100:: 95.65%, #200:: 95.65%
# 2020-09-09 12:13:45,053 - root.final_stats() - line: 105 - INFO - End of #1/1 run of the #23/23 guillotines.
WORD_LISTS = [
    ["uccelli", "ex", "città", "acqua", "estiva"],
    ["terra", "incasso", "arco", "petrolio", "meravigliosa"],
    ["sotto", "misure", "solare", "rete", "civile"],
    ["conoscenza", "leggere", "fedele", "bella", "omaggio"],
    ["studio", "fisso", "internazionale", "famiglia", "autore"],
    ["fiume", "pesca", "medicina", "saggezza", "filo"],
    ["mettere", "dolci", "sotto", "ali", "argilla"],
    ["amor", "nero", "cadere", "passo", "marito"],
    ["essere", "comparsa", "ronaldo", "mondiale"],
    ["san", "peso", "funky", "canto", "francia"],
    ["nobile", "leone", "palco", "salto", "porto"],
    ["anna", "piazza", "corte", "campo", "scienza"],
    ["posto", "artificiale", "lavaggio", "allenare", "gallina"],
    ["gomma", "quotidiano", "denti", "cassetta", "albero"],
    ["fine", "vendere", "esclusiva", "parlamentare", "arresto"],
    ["fiducia", "medico", "consultare", "valore", "numero"],
    ["gigante", "difesa", "rosso", "terra", "scimmie"],
    ["buono", "volta", "emozione", "conto", "zucchero"],
    ["deserto", "capitan", "bicchiere", "secolo", "solare"],
    ["campionato", "breve", "senza", "area", "selvaggia"],
    ["ufficio", "baule", "muro", "cabina", "fondo"],
    ["sotto", "misure", "solare", "rete", "civile"],
    ["adottare", "codici", "disturbi", "civile", "sportivo"],
]
SOLUTIONS = [
    "colonia",
    "lampada",  # 2
    "protezione",  # 2
    "copia",  # 2
    "diritto",  # 2
    "saggio",  # 2
    "piedi",  # 2
    "cielo",  # 1
    "fenomeno",  # 2
    "gallo",  # [('grande', 'ADJ', [0.001750578277398638]), ('gallo', 'NOUN', [0.001750578277398638]),, grande ha idf minore ma poi viene superata...
    "cervo",  # 2
    "miracoli",  # ok, mean 20
    "cervello",  # ok 1
    "pane",  # ok 1
    "mandato",  # 2
    "legale",
    "pianeta",  # ok 1
    "poco",  # ok 1
    "tempesta",
    "sosta",
    "armadio",
    "protezione",
    "comportamento",
]


N_GEN = 200
N_POP = 400
N_HOF = 10
MUT_PROB = 0.1
CROSS_PROB = 0.5
TOURNAMENT_SIZE = 2
INDIVIDUAL_MAX_LENGTH = 300
WEIGHTS = (
    1.0,
)  # TODO: fare una multi fitness con max_SUM, max_MEAN, min_STD. Come pesarle?


STAGNATION_STOP = True
EVO_PATIENCE = 20
VERBOSE = False  # TODO:

MAX_RESULTS = [
    5,
    10,
    15,
    20,
    30,
    40,
    50,
    75,
    100,
    250,
    500,
    1000,
    1500,
    2000,
    2500,
    5000,
    10000,
]
RESTRICT_VOCAB = None  # 100000 #TODO: sembra che si possano avere risultati migliori restringendo, ma devo essere sicuro che sia ordinato per frequenza!!! COME?
# recuperare tutte le parole ed ordinarle in base alla frequenza estratta da wiki


# min, max (auto, auto) | summ*idf + mean / 1 + std |
# 200, 200 -> 100 iter, maxGen -> min: 9, max: 370, mean: 63.88

# min, max (auto, auto) | (summ*idf / N) / 1 + std |  #weighted mean
# 200, 200 -> 100 iter, maxGen -> min: 2, max: 189, mean: 57.9

# min, max (auto, auto) | mean / 1 + std |
# 200, 200 -> 100 iter, maxGen -> min: 3, max: 171, mean: 44.45
# 200, 200 -> 100 iter, stagne -> min: 1, max: 254, mean: 48.24

# 200, 200 -> 100 iter, st-fil -> min: 1, max: 191, mean: 34.4 (16s)
# 200, 500 -> 100 iter, st-fil -> min: 2, max: 350, mean: 32.67, std: 44.1101020175651 (30s)
# 200, 1000 -> 100 iter, st-fil -> min: 4, max: 137, mean: 32.22 (73s)

# 200, 200 -> 100 iter, st-fil-verbs -> min: 2, max: 152, mean: 30.65, std: 24.405890682374203 (11s)
# 200, 500 -> 100 iter, st-fil-verbs -> min: 4, max: 214, mean: 35.99, std: 35.73107750964138 (31s)
# 200, 1000 -> 71 iter, st-fil-verbs -> min: 2, max: 122, mean: 31.375, std: 25.142614057942875 (79s)

# 200, 200 -> 100 iter, st-fil-verbs-newfil -> min: 1, max: 293, mean: 37.64, std: 37.334038088586134
# 200, 200 -> 100 iter, st-fil-verbs-newfil -> min: 2, max: 151, mean: 27.81, std: 20.432178053257072 (10s)
# 200, 200 -> 100 iter, st-fil-verbs-newfil-lemma -> min: 1, max: 124, mean: 28.48, std: 22.718485865039508 (10s)
# 200, 200 -> 100 iter, st-fil-verbs-newfil-lemma-nodup -> min: 1, max: 128, mean: 28.75, std: 23.228592294842148 (10s)
# 200, 200 -> 100 iter, st-fil-verbs-newfil-lemma-nodup-fitsorted -> min: 2, max: 71, mean: 15.15, std: 12.981814202953299 (10s)
# 200, 200 -> 100 iter, 40patience, st-fil-verbs-newfil-lemma-nodup-fitsorted -> min: 2, max: 42, mean: 14.75, std: 9.265392598265871 (17s)
# 200, 200 -> 100 iter, 40patience, st-fil-verbs-newfil-lemma-nodup-fitsorted-stanza -> min: 3, max: 179, mean: 12.72, std: 17.926561298810213 (14s)
# 200, 200 -> 100 iter, 40patience, maxres100, st-fil-verbs-newfil-lemma-nodup-fitsorted-stanza -> min: -1, max: 22, mean: 9.39, std: 5.291304187060123 (non affidabile con i -1 che fanno media)
# 200, 200 -> 100 iter, 40patience, maxres1000, st-fil-verbs-newfil-lemma-nodup-fitsorted-stanza -> min: 3, max: 38, mean: 9.78, std: 6.287416003415076
# 200, 200 -> 100 iter, 40patience, maxres2500, st-fil-verbs-newfil-lemma-nodup-fitsorted-stanza -> min: 2, max: 53, mean: 12.42, std: 8.40021428298112
# 200, 200 -> 100 iter, 40patience, maxresDynamic[5,..,10000], st-fil-verbs-newfil-lemma-nodup-fitsorted-stanza -> min: 2, max: 48, mean: 14.1, std: 10.37063161046616
# 200, 200 -> 100 iter, 40patience, maxresDynamic[5,..,10000], st-fil-verbs-newfil-lemma-nodup-fitsorted-stanza -> min: 3, max: 46, mean: 10.47, std: 6.959101953556939
# 200, 200 -> 100 iter, 40patience, maxresDynamic[10000], st-fil-verbs-newfil-lemma-nodup-fitsorted-stanza -> min: 2, max: 31, mean: 10.67, std: 6.4078935696529795


# 200, 200 -> 100 iter, st-fil, tor 2 -> min: 2, max: 335, mean: 35.58
# 200, 200 -> 100 iter, st-fil, cross 0.8 -> min: 2, max: 335, mean: 35.58
# 200, 400 -> 100 iter, stagne -> min: 6, max: 186, mean: 47.95


# min, max (auto, auto) | mean, std |
#  200, 200 -> 30 iter, stagne -> min: 22, max: 85, mean: 44.0

# min, max (auto, auto) |  sum + mean / 1 + std |
# 200, 200 -> 100 iter -> min: 16, max: 155, mean: 45.07

# min, max (1, -1) | mean / 1 + std |
# 200, 200 -> 100 iter -> min: 2, max: 265, mean: 47.04

# min, max (10, -10) | mean / 1 + std |
# 200, 200 -> 100 iter -> min: 4, max: 290, mean: 58.76

# min, max (10, -10) | sum + mean / 1 + std |
# 200, 200 -> 100 iter -> min: 18, max: 118, mean: 46.18
