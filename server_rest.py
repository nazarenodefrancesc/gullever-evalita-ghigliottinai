#!/usr/bin/env python
# coding: utf-8

import os
os.chdir(os.path.dirname(os.path.abspath(__file__)))
print(f"Working dir:: {os.getcwd()}")

import socket
import requests
from collections import defaultdict

import flask
import config
from texmega import texmega_utils, texmega
from flask import Flask, request, jsonify, json

import logging.config
from logging_conf import LOGGING_CONF

log = logging.getLogger()
logging.config.dictConfig(LOGGING_CONF)

import stanza

for lang in ["it"]:
    try:
        stanza.Pipeline(lang)
    except:
        stanza.download(lang)

### VARIABLES

model = None
nlp_it = None
whitewords = None
lemma_pos_cache = None
stemm_cache = None
norm_idfs = None
filter_list = None

# FLASK
app = Flask(__name__)


@app.route("/")
def hello():
    return "Hello, I'm a Guillotine resolver made by CELI s.r.l \nauthor: Nazareno De Francesco \nemail: nazareno.defrancesco@celi.it"


@app.route("/solve_guillotine", methods=["GET", "POST"])
def solve_guillotine_api():
    resp = defaultdict(str)
    response = flask.Response(status=200)

    try:
        guillotine = []
        callback = ""
        game_id = None
        if request.method == "POST":
            data = (
                request.data
            )  # https://stackoverflow.com/questions/10434599/get-the-data-received-in-a-flask-request
            json_data = json.loads(data)
            guillotine = [
                json_data["w1"],
                json_data["w2"],
                json_data["w3"],
                json_data["w4"],
                json_data["w5"],
            ]
            callback = json_data["callback"]
            game_id = json_data["game_id"]
            log.info(data)
        elif request.method == "GET":
            guillotine = request.args.get("guillotine", "").split(",")

        guillotine = [clue.lower().strip() for clue in guillotine]

        solution, best_matches = solve_guillotine(guillotine)

        resp["guillotine"] = guillotine
        resp["first_10_solutions"] = best_matches[0:10]
        log.info(f"guillotine: {guillotine}")
        log.info(f"solution: {solution}")

        response.response = str(resp)
        do_callback_response(callback, game_id, solution)

    except Exception as e:
        log.error(e)
        response.status_code = 500
        response.response = str(e)

    if request.method == "GET":
        return jsonify(resp)
    elif request.method == "POST":
        return response


def do_callback_response(callback, game_id, solution):
    try:
        payload = {
            "game_id": game_id,
            "uuid": "1bd8c8a3-a41d-11ea-8ea7-4a535f44e15a",
            "solution": solution,
        }
        req = requests.post(
            callback,
            data=payload,
            headers={
                "Authorization": "7e5be28f35bf49f67bf7f623c5674be3b46b9282b8830970931515007db16e78"
            },
        )
        log.info(
            f"Calling back:: {callback} Status:: {req.status_code} Payload:: {payload}"
        )
    except Exception as e:
        log.error("Error calling back service with solution", e)


def loadModel():
    global model, nlp_it, whitewords, lemma_pos_cache, stemm_cache, norm_idfs, filter_list, all_word_in_vocab

    (
        model,
        nlp_it,
        whitewords,
        lemma_pos_cache,
        stemm_cache,
        norm_idfs,
        filter_list,
    ) = texmega_utils.load_necessary_components()

    texmega_utils.check_if_words_are_in_filterlist(filter_list)
    texmega_utils.check_solutions_pos(lemma_pos_cache, whitewords)

    if not config.ONLY_MOST_SIMILAR:
        all_word_in_vocab = list(model.wv.vocab)
        all_word_in_vocab = texmega_utils.filter_vocab(
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


def solve_guillotine(wordlist):
    global all_word_in_vocab

    # Checking wordlist, removing not existing words
    log.info(f"Checking wordlist...")
    wordlist = texmega_utils.checkWordlist(model, lemma_pos_cache, wordlist)

    if len(wordlist) < 2:
        log.error(
            f"Wordlist size is {len(wordlist)} < 2. Need at least two words to start the algorithm. Wordlist will be skipped."
        )

    if config.ONLY_MOST_SIMILAR:
        all_word_in_vocab = texmega_utils.get_most_similar(
            wordlist,
            model,
            filter_list,
            lemma_pos_cache,
            stemm_cache,
            whitewords,
            nlp_it,
            norm_idfs,
        )

    best_matches, found, solution_position, solution_match = texmega.exhaustive_search(
        all_word_in_vocab,
        model,
        wordlist,
        None,
        norm_idfs,
        lemma_pos_cache,
        stemm_cache,
        [],
        [],
    )

    solution = best_matches[0][0]
    texmega_utils.print_cosins_of_bestmatches(best_matches, model, norm_idfs, solution, wordlist, n=3)

    return solution, best_matches


# UTILITY
def get_ip():
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        # doesn't even have to be reachable
        s.connect(("10.255.255.255", 1))
        IP = s.getsockname()[0]
    except:
        IP = "127.0.0.1"
    finally:
        s.close()
    return IP


# MAIN
if __name__ == "__main__":

    # ip = get_ip()
    port = 5001

    print("Running on port:" + str(port))
    loadModel()
    app.run(port=port)

