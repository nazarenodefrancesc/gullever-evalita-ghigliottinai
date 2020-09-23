import datetime
import logging.config
import os
import pickle
import platform
import re
import string
import time

import nltk
import stanza
from gensim.models import FastText
from gensim.models import Word2Vec
from matplotlib import pyplot
from sklearn.decomposition import PCA
from spacy_stanza import StanzaLanguage
from tqdm import tqdm

from logging_conf import LOGGING_CONF

log = logging.getLogger()
logging.config.dictConfig(LOGGING_CONF)

from embeddings_trainer.glove2word2Vec import doGlove2word2vec

ROOT_PATH = "."
GLOVE_FOLDER_PATH = "GloVe"
GLOVE_CORPUS_NO_RETURN_PATH = os.path.join(
    GLOVE_FOLDER_PATH, "texmega_corpus_no_return"
)
GLOVE_CORPUS_WITH_RETURN_PATH = os.path.join(
    GLOVE_FOLDER_PATH, "texmega_corpus_with_return"
)
GLOVE_VECTORS_PATH = os.path.join(GLOVE_FOLDER_PATH, "texmega_corpus_vectors.txt")
PUNCTUATION_TABLE = str.maketrans("", "", string.punctuation)


def load_necessary_components():
    # Spacy NLP pipeline
    log.info(f"Loading Stanza-IT...")
    snlp = stanza.Pipeline(
        processors="tokenize,mwt,pos,lemma", lang="it", use_gpu=False
    )  # pos_batch_size=3000
    nlp_it = StanzaLanguage(snlp)

    ita_stemmer = nltk.stem.snowball.ItalianStemmer()

    return nlp_it, ita_stemmer


def clean_from_punctuation(line):
    return re.sub("[!\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~]+", " ", line)


def analyze_and_filter_from_punct(
    sentences, nlp_it, ita_stemmer, word_pos_dict, tokenize=False
):

    start_time = time.time()

    processed_lines = []

    for line in tqdm(sentences):

        line = clean_from_punctuation(line)
        line = line.lower()
        line = line.strip()

        try:
            if len(line) <= 0 or line == "\n" or line == "":
                continue

            temp_line = []
            # temp_line_lemmatized = []

            for doc in nlp_it.pipe([line]):
                if doc.is_parsed:
                    for doc_word in doc:

                        if doc_word.pos_ in ["PUNCT"]:
                            continue

                        # doc_word.string = re.sub(r'^[0-9]*', '', doc_word.string) #fix 1teatro
                        # doc_word.lemma_ = re.sub(r'^[0-9]*', '', doc_word.lemma_)
                        doc_word_stemm = ita_stemmer.stem(doc_word.string.strip())

                        temp_line.append(
                            doc_word.string.strip()
                            + "_"
                            + doc_word.lemma_.strip()
                            + "_"
                            + doc_word_stemm.strip()
                            + "_"
                            + doc_word.pos_
                        )

                        # temp_line_lemmatized.append(doc_word.lemma_)
                        #
                        # if (
                        #     doc_word not in word_pos_dict.keys()
                        # ):  # caching pos and lemmas
                        #     word_pos_dict[doc_word.string] = [ #costa
                        #         doc_word.lemma_, #costare
                        #         doc_word.pos_, #noun
                        #     ]
                        #
                        #     for lemma_doc in nlp_it.pipe([doc_word.lemma_]):
                        #         if lemma_doc.is_parsed:
                        #             for lemma_doc_word in lemma_doc:
                        #                 word_pos_dict[doc_word.lemma_] = [ #costare
                        #                     lemma_doc_word.lemma_, #costare
                        #                     lemma_doc_word.pos_, #verb
                        #                 ]

            if len(temp_line) > 0:
                if tokenize:
                    processed_lines.append(temp_line)
                    # processed_lines.append(temp_line_lemmatized)
                else:
                    processed_lines.append(" ".join(temp_line))
                    # processed_lines.append(" ".join(temp_line_lemmatized))
        except Exception as e:
            log.info(f"{e}, line:: [{line}]")

    log.info(f"Time taken:: {(time.time() - start_time) / 60} m.")

    return processed_lines


def load_and_preprocess_sentences_from_file(filepaths, nlp_it, ita_stemmer, tokenize):

    all_sentences = []
    word_pos_dict = dict()
    for filepath in filepaths:
        log.info(f"Loading new sentences from file:: {filepath}")
        sentences = open(filepath, "r", encoding="utf8").readlines()

        log.info(
            f"Filtering {len(sentences)} sentences from punctuation. \nTokenize flag is {tokenize}"
        )
        sentences = analyze_and_filter_from_punct(
            sentences, nlp_it, ita_stemmer, word_pos_dict, tokenize=tokenize
        )
        all_sentences += sentences

    log.info("")

    log.info(f"Sample of corpus:: {all_sentences[:10]}")
    return all_sentences, word_pos_dict


num_fig = 0


def show_results(model, num_vectors=1000):
    global num_fig
    # summarize the loaded model
    log.info(model)
    # summarize vocabulary
    words = list(model.wv.vocab)
    log.info(words[:100])
    log.info(len(words))

    try:
        # Visualize PCA
        # fit a 2d PCA model to the vectors
        X = model[model.wv.vocab][:num_vectors]
        pca = PCA(n_components=2)
        result = pca.fit_transform(X)
        # create a scatter plot of the projection
        pyplot.figure(num_fig,)
        pyplot.scatter(result[:, 0], result[:, 1])
        pyplot.title(f"{model}")
        words = list(model.wv.vocab)
        for i, word in enumerate(words):
            pyplot.annotate(word, xy=(result[i, 0], result[i, 1]))
        pyplot.show()
        num_fig += 1
    except Exception as e:
        log.error(e)


def train_model(model_name, corpus, script_name):
    start_time = time.time()

    log.info(f"Training model:: {model_name} on {len(corpus)} sentences.")

    if model_name == "fasttext":
        model_name = FastText(size=300, window=10, min_count=1)
        model_name.build_vocab(sentences=corpus)
        model_name.train(sentences=corpus, total_examples=len(corpus), epochs=10)
    elif model_name == "word2vec":
        model_name = Word2Vec(
            corpus,
            size=300,
            iter=5,
            negative=10,
            window=10,
            sg=1,
            workers=4,
            min_count=1,
        )
    elif model_name == "glove":
        del corpus  # non serve per glove

        if platform.system() == "Linux":
            os.system(f"cd GloVe/ && sh {script_name}")
        elif platform.system() == "Windows":
            os.system(f"cd GloVe/ && texmega_py.bat")
        else:
            raise Exception(f"OS {platform.system()} not supported.")

        model_name = doGlove2word2vec(
            os.path.abspath(GLOVE_VECTORS_PATH), dump_model=True
        )
    else:
        raise ValueError

    log.info(f"Time taken:: {(time.time() - start_time) / 60} m.")

    return model_name


def dump_model(model, dumpname, script, path):
    dumpname = dumpname + "_" + script + ".pkl"
    log.info(f"Saving:: {dumpname}")
    pickle.dump(model, open(os.path.join(path, dumpname), "wb"))


def dump_corpus(corpus, path, return_char):
    with open(path, "w", encoding="utf8") as corpus_file:
        if isinstance(corpus, list):
            log.info(f"Corpus is a list. return_char:: {return_char}")
            for sentence in corpus:
                if return_char:
                    corpus_file.write(sentence + "\n")
                else:
                    corpus_file.write(sentence + " ")
        elif isinstance(corpus, str):
            log.info("Corpus is a str.")
            corpus_file.write(corpus)
    log.info(f"Saved corpus to:: {path}")


def reload_corpus_and_word_pos_dict(model_name, cache_path, generate_corpus):
    corpus, word_pos_dict = [], {}
    log.info(f"Reloading previous corpus:: {model_name}")

    corpus = open(GLOVE_CORPUS_NO_RETURN_PATH, "r", encoding="utf8").read()

    if generate_corpus:
        word_pos_dict = pickle.load(open(os.path.join(cache_path), "rb"))

    log.info(f"reloaded_word_pos_dict vocab size:: {len(word_pos_dict.keys())}")
    log.info(f"reloaded_corpus size:: {len(corpus.split())}")

    return corpus, word_pos_dict


def get_filepaths(raw_folder_path):
    file_paths = []
    for filename in os.listdir(raw_folder_path):
        if filename.endswith(".txt"):
            file_paths.append(os.path.join(raw_folder_path, filename))
    return file_paths


def preprocess_corpus_from_files(
    file_paths,
    reload_model_name,
    dumpname,
    cache_path,
    generate_corpus,
    reload_previous_corpus,
    tokenize,
):
    reload_previous_corpus = generate_corpus and reload_previous_corpus

    corpus = []
    if generate_corpus:

        log.info(f"Working on {len(file_paths)} files.")

        if reload_previous_corpus:
            reloaded_corpus, reloaded_word_pos_dict = reload_corpus_and_word_pos_dict(
                reload_model_name, cache_path, generate_corpus
            )

        nlp_it, ita_stemmer = load_necessary_components()
        corpus, word_pos_dict = load_and_preprocess_sentences_from_file(
            file_paths, nlp_it, ita_stemmer, tokenize=tokenize
        )

        if reload_previous_corpus:  # NB: non unire con l'if di sopra!!!
            word_pos_dict.update(reloaded_word_pos_dict)
            log.info(
                f"Merged word_pos_dict vocab size:: {len(reloaded_word_pos_dict.keys())}"
            )
            log.info(f"Corpus len:: {len(corpus)}")
            joined_corpus = " ".join(corpus)
            reloaded_corpus += " " + joined_corpus
            corpus = reloaded_corpus
            log.info(f"Merged corpus size:: {len(reloaded_corpus.split())}")

        # Dump cache and clean memory
        # cache_path = f"lemma_pos_cache_[{dumpname.replace('.pkl', '')}].pkl"
        pickle.dump(word_pos_dict, open(cache_path, "wb"))
        log.info(f"Saved cache to:: {cache_path}")
        del word_pos_dict
    else:
        log.info(
            f"Train-only mode active. No new corpus will be generated. Are you sure?"
        )

    return corpus


if __name__ == "__main__":

    scripts = ["texmega_with_return_py.sh"]

    VECTOR_SIZE = 600
    VOCAB_MIN_COUNT = 200
    MAX_ITER = 50
    WINDOW_SIZE = 15
    X_MAX = 10

    generate_corpus = False  # True se vuoi generare un nuovo texmega_corpus aggiungengo i file nella cartella raw/TO_DO. Con False non carica nulla e parte direttamente il train sul corpus esistente.
    reload_previous_corpus = False  # True se vuoi aggiungere i file in TO_DO ad un texmega_corpus già generato
    cache_path = (
        f"../resources/lemma_pos_cache_[glove_[min100_300_win10_ite50_xmax10_041]].pkl"
    )

    model_name = "glove"
    params = f"[min{VOCAB_MIN_COUNT}_{VECTOR_SIZE}_win{WINDOW_SIZE}_ite{MAX_ITER}_xmax{X_MAX}]"
    note = f"_songs"

    log.info(f"Working with this model: {model_name}")
    log.info(f"Generating new corpus? {generate_corpus}")

    raw_folder_path = "../resources/raw/TO_BE_USED"
    embeddings_folder_path = "../resources/embeddings"
    file_paths = get_filepaths(raw_folder_path)

    now_date = datetime.datetime.now().strftime("%Y_%m_%d")
    model_dumpname = f"{model_name}_{params}_{note}_{now_date}"

    corpus = preprocess_corpus_from_files(
        file_paths=file_paths,
        reload_model_name="all",
        dumpname=model_dumpname,
        cache_path=cache_path,
        generate_corpus=generate_corpus,
        reload_previous_corpus=reload_previous_corpus,
        tokenize=(model_name != "glove"),  # False per GloVe
    )

    if generate_corpus:
        dump_corpus(
            corpus, GLOVE_CORPUS_NO_RETURN_PATH, return_char=False
        )  # False per GloVe
        dump_corpus(corpus, GLOVE_CORPUS_WITH_RETURN_PATH, return_char=True)

    for script in scripts:
        log.info(f"Running script:: {script}")
        model = train_model(model_name, corpus, script_name=script)
        dump_model(model, model_dumpname, script, embeddings_folder_path)
        show_results(model, num_vectors=1000)
        log.info(f"End of script:: {script}")

    log.info("All done!")
