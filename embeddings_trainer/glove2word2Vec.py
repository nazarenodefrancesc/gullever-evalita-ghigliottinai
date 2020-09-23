import pickle
from gensim.test.utils import datapath, get_tmpfile
from gensim.models import KeyedVectors
from gensim.scripts.glove2word2vec import glove2word2vec

import logging.config
from logging_conf import LOGGING_CONF

log = logging.getLogger()
logging.config.dictConfig(LOGGING_CONF)


def doGlove2word2vec(vectors_path, dump_model=False):
    log.info(f"Converting {vectors_path} to word2vec format...")
    glove_file = datapath(vectors_path)
    tmp_file = get_tmpfile("vectors_glove.txt")

    _ = glove2word2vec(glove_file, tmp_file)
    model = KeyedVectors.load_word2vec_format(tmp_file)

    if dump_model:
        pickle.dump(model, open("vectors_glove.pkl", "wb"))
    log.info("Done.")
    return model


if __name__ == "__main__":
    path = "F:\PythonProjects\\texmega_py\embeddings_trainer\GloVe\\texmega_corpus_vectors.txt"
    model = doGlove2word2vec(path, dump_model=True)
