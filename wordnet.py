from multiwordnet.db import compile

compile("italian")
compile("italian", "synset")
compile("common")

from multiwordnet.wordnet import WordNet

LWN = WordNet("italian")
for lemma in LWN.lemmas:  # all the lemmas currently in the WordNet
    if lemma is not None:
        print(lemma.lemma, lemma.pos)

LWN.get_lemma("visto").pos
word = LWN.get_lemma(
    "tratto"
)  # this gives you access to a single lemma abalieno.synonyms
word.get_derivates("n")

# all lemmas that share a synset with 'abalieno' abalieno.antonyms abalieno.derivates
# use .get_derivates('n') to restrict by POS abalieno.relatives
# use .get_relatives('n') to restrict by POS abalieno.synsets

LWN.get("sonoro", pos="v", strict=False)
# restrict the results to verbs

synset = LWN.get_synset("n#07462736")
# you can find a synset directly, if you know its offset ID

synset.lemmas

LWN.get_relations(
    source=synset
)  # all semantic relations where 'synset' is the source LWN.get_relations(source=synset, type='@')  # restrict to hyponymy relations
