from collections import defaultdict

import datetime
import pickle
import stanza
import logging.config
from spacy_stanza import StanzaLanguage
from tqdm import tqdm
from spacy.matcher import Matcher
from logging_conf import LOGGING_CONF

log = logging.getLogger()
logging.config.dictConfig(LOGGING_CONF)


def create_matcher(nlp):
    matcher = Matcher(nlp.vocab)
    # matcher.add("noun-verb-adj", None, [{'POS': 'NOUN'}, {'POS': 'VERB'}, {'POS': 'ADJ'}]) # ciliegia tira altra (forse tira fuori roba troppo sporca)
    # matcher.add("verb-noun-noun", None, [{'POS': 'VERB'}, {'POS': 'NOUN'}, {'POS': 'NOUN'}]) # rompere l anima, scoprire l america
    # matcher.add("verb-noun-adj", None, [{'POS': 'VERB'}, {'POS': 'NOUN'}, {'POS': 'ADJ'}]) # fare orecchie da mercante, prendere precauzioni tardive (un po' sporco)
    # matcher.add("verb-noun", None, [{'POS': 'VERB'}, {'POS': 'NOUN'}]) # fissare limiti
    # matcher.add("verb-adj", None, [{'POS': 'VERB'}, {'POS': 'ADJ'}]) # parlare chiaro, torna comodo
    # matcher.add("adj-noun", None, [{'POS': 'ADJ'}, {'POS': 'NOUN'}]) # malo modo, buon senso
    # matcher.add("noun-adj", None, [{'POS': 'NOUN'}, {'POS': 'ADJ'}]) # modo umiliante, pompa magna
    # matcher.add("noun-prep-noun", None, [{'POS': 'NOUN'}, {'POS': 'ADP'}, {'POS': 'NOUN'}]) # colpo di coda, cognizione di causa, conti in tasca
    # matcher.add("noun-conj-noun", None, [{'POS': 'NOUN'}, {'POS': 'CCONJ'}, {'POS': 'NOUN'}]) # causa e effetto

    matcher.add(
        "noun-(adp-det-conj)-noun",  # arogmento in discussione, regole di comportamento, rotto della cuffia
        None,
        [
            {"POS": "NOUN", "IS_STOP": False, "LENGTH": {">=": 3}},
            {"POS": "ADP", "OP": "?"},
            {"POS": "DET", "OP": "?"},
            {"POS": "CCONJ", "OP": "?"},
            {"POS": "NOUN", "IS_STOP": False, "LENGTH": {">=": 3}},
        ],
    )

    # matcher.add(
    # 	"noun-noun-noun",
    # 	None,
    # 	[
    # 		{"POS": "NOUN", "IS_STOP": False, "LENGTH": {">=": 3}},
    # 		{"POS": "NOUN", "IS_STOP": False, "LENGTH": {">=": 3}},
    # 		{"POS": "NOUN", "IS_STOP": False, "LENGTH": {">=": 3}},
    # 	],
    # )

    matcher.add(
        "adj-noun",  # cattiva azione, buon senso
        None,
        [
            {"POS": "ADJ", "IS_STOP": False, "LENGTH": {">=": 3}},
            {"POS": "NOUN", "IS_STOP": False, "LENGTH": {">=": 3}},
        ],
    )

    matcher.add(
        "noun-adj",  # pompa magna, cose impossibili
        None,
        [
            {"POS": "NOUN", "IS_STOP": False, "LENGTH": {">=": 3}},
            {"POS": "ADJ", "IS_STOP": False, "LENGTH": {">=": 3}},
        ],
    )

    matcher.add(
        "verb-(adp-det)-noun",  # parlare al vento, fissare limiti
        None,
        [
            {"POS": "VERB", "IS_STOP": False},
            {"POS": "ADP", "OP": "?"},
            {"POS": "DET", "OP": "?"},
            {"POS": "NOUN", "IS_STOP": False, "LENGTH": {">=": 3}},
        ],
    )

    matcher.add(
        "verb-(adp-det)-adj",  # partire in quarta, parlare chiaro
        None,
        [
            {"POS": "VERB", "IS_STOP": False},
            {"POS": "ADP", "OP": "?"},
            # {"POS": "DET", "OP": "?"},
            {"POS": "ADJ", "IS_STOP": False, "LENGTH": {">=": 3}},
        ],
    )

    return matcher


def mine_pattern(phrases, matcher, limit=None):

    phrases = phrases[:limit] if limit is not None else phrases
    pattern_dict = defaultdict(int)

    for phrase in tqdm(phrases):

        doc = nlp(phrase)
        matches = matcher(doc)

        # for word in doc:
        # 	log.info(f"{word} - {word.pos_}")

        for match_id, start, end in matches:
            string_id = nlp.vocab.strings[match_id]
            span = doc[start:end]
            pattern = span.text.replace("'", " ").lower()
            pattern_dict[pattern] += 1
            log.info(
                f"{string_id} [{start}, {end}] -> '{pattern}' [{pattern_dict[pattern]}]"
            )

        if len(pattern_dict) % (len(phrases) / 100) == 0:  # loggo ogni 1%
            log.info(
                f"First 20 patterns:: {sorted([(word, pattern_dict[word]) for word in pattern_dict.keys()], reverse=True, key=lambda x: x[1])[:10]}"
            )
            log.info(
                f"Last 20 patterns:: {sorted([(word, pattern_dict[word]) for word in pattern_dict.keys()], reverse=False, key=lambda x: x[1])[:10]}"
            )
            log.info(f"Patterns mined:: {len(pattern_dict)}")

    return pattern_dict


if __name__ == "__main__":

    datetime = datetime.datetime.now().strftime("%Y_%m_%d")

    corpus_paths = [
        "/home/nazareno/CELI/repositories/python_projects/texmega_py/resources/raw/TO_BE_USED/polirematiche_all_cleaned.txt",
        "/home/nazareno/CELI/repositories/python_projects/texmega_py/resources/raw/TO_BE_USED/modi_di_dire_impariamoitaliano_text.txt",
        "/home/nazareno/CELI/repositories/python_projects/texmega_py/resources/raw/TO_BE_USED/modi_di_dire_pdf.txt",
        "/home/nazareno/CELI/repositories/python_projects/texmega_py/resources/raw/TO_BE_USED/proverbi_italiani.txt",
        "/home/nazareno/CELI/repositories/python_projects/texmega_py/resources/raw/TO_BE_USED/proverbi_italiani_dige.txt",
        "/home/nazareno/CELI/repositories/python_projects/texmega_py/resources/raw/TO_BE_USED/titoli_canzoni_italiane.txt",
        "/home/nazareno/CELI/repositories/python_projects/texmega_py/resources/raw/TO_BE_USED/Dizionario_collocazioni_cleaned_processed.txt",
    ]
    log.info(f"corpus_path:: {corpus_paths}")

    phrases = []
    for corpus_path in corpus_paths:
        phrases += open(corpus_path, "r", encoding="utf8").readlines()

    # Spacy NLP pipeline
    snlp = stanza.Pipeline(
        processors="tokenize,mwt,pos,lemma", lang="it", use_gpu=False
    )  # pos_batch_size=3000
    nlp = StanzaLanguage(snlp)

    matcher = create_matcher(nlp)

    log.info("Cleaning phrases...")
    phrases = [
        " ".join([word.split("_")[0] for word in phrase.split()])
        for phrase in tqdm(phrases)
    ]

    log.info("Mining patterns...")
    pattern_dict = mine_pattern(phrases, matcher, limit=None)

    log.info(
        f"First 20 patterns:: {sorted([(word, pattern_dict[word]) for word in pattern_dict.keys()], reverse=True, key=lambda x: x[1])[:10]}"
    )
    log.info(
        f"Last 20 patterns:: {sorted([(word, pattern_dict[word]) for word in pattern_dict.keys()], reverse=False, key=lambda x: x[1])[:10]}"
    )
    log.info(f"Patterns mined:: {len(pattern_dict)}")

    log.info("Writing pkl...")
    pickle.dump(
        pattern_dict, open(f"./pattern_mining/pattern_dict_{datetime}.pkl", "wb")
    )
    # pattern_dict = pickle.load(open(f"./pattern_mining/pattern_dict_2020_09_03.pkl", "rb"))

    log.info(f"Writing pattern_mining_corpus_{datetime}.txt, frequency_balanced...")
    with open(
        f"./pattern_mining/pattern_mining_corpus_frequency_balanced_{datetime}.txt",
        "w",
        encoding="utf8",
    ) as corpus_file:
        for pattern in pattern_dict.keys():
            for _ in range(pattern_dict[pattern]):
                corpus_file.write(pattern)
                corpus_file.write("\n")

    log.info(f"Writing pattern_mining_corpus_{datetime}.txt, not frequency_balanced...")
    with open(
        f"./pattern_mining/pattern_mining_corpus_{datetime}.txt", "w", encoding="utf8"
    ) as corpus_file:
        for pattern in pattern_dict.keys():
            corpus_file.write(pattern)
            corpus_file.write("\n")

    log.info("All done.")
