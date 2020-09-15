#
# WORD_LISTS = [
#     ['mettere', 'dolci', 'sotto', 'ali', 'argilla'],
#     ['amor', 'nero', 'cadere', 'passo', 'marito'],
#     ['essere', 'comparsa', 'ronaldo', 'mondiale'],
#     ['san', 'peso', 'funky', 'canto', 'francia'],
#     ['nobile', 'leone', 'palco', 'salto', 'porto'],
#     ["anna", "piazza", "corte", "campo", "scienza"],
#     ["posto", "artificiale", "lavaggio", "allenare", "gallina"],
#     ["gomma", "quotidiano", "denti", "cassetta", "albero"],
#     ['fine', 'vendere', 'esclusiva', 'parlamentare', 'arresto'],
#     ['fiducia', 'medico', 'consultare', 'valore', 'numero'],
#     ['gigante', 'difesa', 'rosso', 'terra', 'scimmie'],
#     ['buono', 'volta', 'emozione', 'conto', 'zucchero'],
# ]
# SOLUTIONS = [
#     "piedi",
#     "cielo", #2 STEMM
#     "fenomeno", #2
#     "gallo", # [('grande', 'ADJ', [0.001750578277398638]), ('gallo', 'NOUN', [0.001750578277398638]),, grande ha idf minore ma poi viene superata...
#     "cervo", #2
#     "miracoli", #ok, mean 20
#     "cervello", #ok 1
#     "pane", #ok 1
#     "mandato", #2
#     "legale",
#     "pianeta", #ok 1
#     "poco", #ok 1
# ]
#
# count_true = 0
# count_false = 0
# for wordlist, solution in zip(WORD_LISTS, SOLUTIONS):
# 	most_similars = set()
# 	for word in wordlist:
# 		if word in [word[0] for word in all_word_in_vocab]:
# 			word_most_similar_set = set([word_tuple[0] for word_tuple in model.most_similar(word, topn=50)])
# 			most_similars.update(word_most_similar_set)
# 	# print(most_similars)
# 	print(len(most_similars))
# 	print(f"{wordlist} --> {solution}")
# 	print(f"solution:: {solution in most_similars}")
# 	# print(f"not_solution:: {not_solution in most_similars}")
#
# 	count_true += 1 if solution in most_similars else 0
# 	count_false += 1 if solution not in most_similars else 0
# 	print(f"true:: {count_true}, false:: {count_false}")
#

#################################
import re


def fix_partial_collocation(partial_collocation):
    partial_collocation = partial_collocation.replace("●", "")
    partial_collocation = partial_collocation.strip().lower()
    partial_collocation = partial_collocation.replace("avverbi", "")
    partial_collocation = partial_collocation.replace("aggettivi", "")
    partial_collocation = partial_collocation.replace("verbo+complemento", "")
    partial_collocation = partial_collocation.replace("soggetto+verbo", "")
    partial_collocation = partial_collocation.replace(" nm", " ")
    partial_collocation = partial_collocation.replace(" v", " ")
    return partial_collocation


def compute_collocations(sentence_list):
    complete_collocations = []
    master_line = sentence_list[0].strip()

    if " nm" in master_line:
        master_words = master_line.split(" nm")
        master_words.remove("nm") if "nm" in master_words else None
        master_words.remove("") if "" in master_words else None
    elif " v " in master_line:
        master_words = master_line.split(" v ")
        master_words.remove(" v ") if " v " in master_words else None
        master_words.remove("") if "" in master_words else None
    else:
        master_words = [master_line]

    for master_word in master_words:

        if len(master_word.split(",")) > 1:
            master_words_splitted = master_word.strip().split(",")
            master_words.remove(master_word)
            for master_word_splitted in master_words_splitted:
                master_word_splitted = re.sub(" v$", "", master_word_splitted).strip()
                master_word_splitted = re.sub(" nm$", "", master_word_splitted).strip()
                master_words.append(
                    fix_partial_collocation(master_word_splitted.strip())
                )

    master_words = [
        master_word.lower().strip()
        for master_word in master_words
        if len(master_word.strip()) > 2
    ]
    master_words = list(set(master_words))

    other_sentences = sentence_list[1:]

    for master_word in master_words:
        for other_sentence in other_sentences:
            partial_collocations = other_sentence.split(",")
            for partial_collocation in partial_collocations:
                partial_collocation = fix_partial_collocation(
                    partial_collocation
                ).strip()
                complete_collocation = master_word + " " + partial_collocation
                complete_collocations.append(complete_collocation.strip().lower())

    return complete_collocations


sentences = open(
    "resources/raw/TO_BE_USED/Dizionario_collocazioni_cleaned.txt", "r", encoding="utf8"
).readlines()

collocations = []
temp_sentence = []
for sentence_unprocessed in sentences:
    if not sentence_unprocessed.startswith("########") and sentence_unprocessed != "\n":
        temp_sentence.append(sentence_unprocessed)
    if sentence_unprocessed.startswith("########"):
        if len(temp_sentence) > 0:
            computed_collocations = compute_collocations(temp_sentence)
            for computed_collocation in computed_collocations:
                computed_collocation = re.sub(" v ", " ", computed_collocation).strip()
                computed_collocation = re.sub(" nm ", " ", computed_collocation).strip()
                computed_collocation = re.sub(" nf ", " ", computed_collocation).strip()
                computed_collocation = re.sub(" pl ", " ", computed_collocation).strip()
                collocations.append(computed_collocation)
        temp_sentence = []
print(collocations[:100])
print(len(collocations))

with open(
    "resources/raw/TO_BE_USED/Dizionario_collocazioni_cleaned_processed.txt",
    "w",
    encoding="utf8",
) as file:
    for collocation in collocations:
        file.write(collocation + "\n")

#####

from collections import OrderedDict
import numpy as np
import stanza
from spacy_stanza import StanzaLanguage
from spacy.lang.en.stop_words import STOP_WORDS

# Spacy NLP pipeline
snlp = stanza.Pipeline(
    processors="tokenize,mwt,pos,lemma", lang="it", use_gpu=False
)  # pos_batch_size=3000
nlp = StanzaLanguage(snlp)


class TextRank4Keyword:
    """Extract keywords from text"""

    def __init__(self):
        self.d = 0.85  # damping coefficient, usually is .85
        self.min_diff = 1e-5  # convergence threshold
        self.steps = 10  # iteration steps
        self.node_weight = None  # save keywords and its weight

    def set_stopwords(self, stopwords):
        """Set stop words"""
        for word in STOP_WORDS.union(set(stopwords)):
            lexeme = nlp.vocab[word]
            lexeme.is_stop = True

    def sentence_segment(self, doc, candidate_pos, lower):
        """Store those words only in cadidate_pos"""
        sentences = []
        for sent in doc.sents:
            selected_words = []
            for token in sent:
                # Store words only with cadidate POS tag
                if token.pos_ in candidate_pos and token.is_stop is False:
                    if lower is True:
                        selected_words.append(token.text.lower())
                    else:
                        selected_words.append(token.text)
            sentences.append(selected_words)
        return sentences

    def get_vocab(self, sentences):
        """Get all tokens"""
        vocab = OrderedDict()
        i = 0
        for sentence in sentences:
            for word in sentence:
                if word not in vocab:
                    vocab[word] = i
                    i += 1
        return vocab

    def get_token_pairs(self, window_size, sentences):
        """Build token_pairs from windows in sentences"""
        token_pairs = list()
        for sentence in sentences:
            for i, word in enumerate(sentence):
                for j in range(i + 1, i + window_size):
                    if j >= len(sentence):
                        break
                    pair = (word, sentence[j])
                    if pair not in token_pairs:
                        token_pairs.append(pair)
        return token_pairs

    def symmetrize(self, a):
        return a + a.T - np.diag(a.diagonal())

    def get_matrix(self, vocab, token_pairs):
        """Get normalized matrix"""
        # Build matrix
        vocab_size = len(vocab)
        g = np.zeros((vocab_size, vocab_size), dtype="float")
        for word1, word2 in token_pairs:
            i, j = vocab[word1], vocab[word2]
            g[i][j] = 1

        # Get Symmeric matrix
        g = self.symmetrize(g)

        # Normalize matrix by column
        norm = np.sum(g, axis=0)
        g_norm = np.divide(
            g, norm, where=norm != 0
        )  # this is ignore the 0 element in norm

        return g_norm

    def get_keywords(self, number=10):
        """Print top number keywords"""
        node_weight = OrderedDict(
            sorted(self.node_weight.items(), key=lambda t: t[1], reverse=True)
        )
        for i, (key, value) in enumerate(node_weight.items()):
            print(key + " - " + str(value))
            if i > number:
                break

    def analyze(
        self,
        text,
        candidate_pos=["NOUN", "PROPN"],
        window_size=4,
        lower=False,
        stopwords=list(),
    ):
        """Main function to analyze text"""

        # Set stop words
        self.set_stopwords(stopwords)

        # Pare text by spaCy
        doc = nlp(text)

        # Filter sentences
        sentences = self.sentence_segment(
            doc, candidate_pos, lower
        )  # list of list of words

        # Build vocabulary
        vocab = self.get_vocab(sentences)

        # Get token_pairs from windows
        token_pairs = self.get_token_pairs(window_size, sentences)

        # Get normalized matrix
        g = self.get_matrix(vocab, token_pairs)

        # Initionlization for weight(pagerank value)
        pr = np.array([1] * len(vocab))

        # Iteration
        previous_pr = 0
        for epoch in range(self.steps):
            pr = (1 - self.d) + self.d * np.dot(g, pr)
            if abs(previous_pr - sum(pr)) < self.min_diff:
                break
            else:
                previous_pr = sum(pr)

        # Get weight for each node
        node_weight = dict()
        for word, index in vocab.items():
            node_weight[word] = pr[index]

        self.node_weight = node_weight


text = """
A bizzeffe = in grande quantità
• A caval donato non si guarda in bocca = non si deve criticare ciò che ti è stato donato
• A ciascun il suo = a ciascuno secondo i suoi meriti
• A gogo = a volontà, in abbondanza
• A occhio e croce = circa, approssimativamente, a prima vista
• A spron battuto = a tutta velocità, in gran fretta
• A stecchetto (stare, tenere, essere) = con poco cibo, a dieta, con poco denaro
• A ufo = gratis, senza pagare nulla
• Abbaiare alla luna = gridare invano, parlare al deserto
• Abbassare la cresta = far atto di sottomissione, diventare umili
• L' abito non fa il monaco = non bastano i segni esteriori a garantire la sostanza interiore
• Acqua chieta = per persone che, sotto l' apparenza di semplicità e di mitezza nascondono astuzia
• Acqua in bocca = non rivelare un segreto, si invita a non parlare
• Ad ogni morte di papa = molto raramente, ad intervalli di tempo lunghissimi
• Adorare il vitello d' oro = essere schiavi del denaro
• Aiutati e Dio t' aiuta = per riuscire nell' impresa bisogna darsi da fare perchè è inutile pregare il
Cielo se poi manca la buona volontà
• Al contadino non far sapere quanto sia buono il cacio con le pere = quando una cosa è buona non
bisogna farla conoscere a chi potrebbe tenersela per sè e non darla più
• Al acqua di rose = debole, inconsistente
• Alle calende greche = rimandare o pagare in un tempo che non verrà mai
• Allevare una serpe in seno = beneficare chi poi si rivelerà ingrato
• Alto papavero = persona importante, persona che conta
• Ambasciator non porta pena = chi porta notixie o messaggi per conto d' altri, anche se sgraditi,
non è responsabile e quindi non può essere punito
• L' amico del giaguaro = chi tiene le parti dell' avversario del suo amico più che dell' amico proprio
• Amor non sente fatica (Cicerone) = per amore si fa tutto, per amore si superano tutte le difficoltà
2
• Andare a Buda = è andato via per non tornare più, è morto
• Andare a fagiolo = andare a genio, piacere, calzare
• Andare a gonfie vele = andare benissimo, ottenere ottimi risultati
• Andare a letto con le galline = andare a letto molto presto
• Andare a monte (mandare a monte) = fallire, non raggiungere lo scopo
• Andare a pennello = andare benissimo, perfettamente
• Andare in tilt = bloccarsi, arrestarsi di macchine
• Andare in visibilio = entusiasmarsi, andare in estasi
• Andare in vacca = finire in nulla, avere cattivo esito
• Andare per la maggiore = essere tra i primi
• L' appetito vien mangiando = più si ha, più si vorrebbe avere
• Apriti sesamo = (in tono ironoco) aiuto prodigioso, mezzo miracoloso per districarsi da una difficile
situazione
• Arrivano i nostri = l' arrivo di qualcuno che ci toglie dai guai (presa dai western)
• L' arte è lunga, la vita è breve (ars longa, vita brevis - Ipocrate attraverso Seneca) = la vita dell'
uomo è troppo corta per raggiungere la perfezione
• Attaccare (appendere) al chiodo = abbandonare, ritirarsi da una attività
• Attaccare un bottone = intrattenere a lungo uno costringendolo ad ascoltare cose che non gli
interessano o che l' annoiano
• Aver sette vite (anime, spiriti) come i gatti = essere dotati di una grande volontà, resistere a gravi
batoste (malattie, incidenti ecc.)
• Avere dei numeri = avere delle ottime qualità, buone capacità
• Avere la coda di paglia = non avere la coscienza tranqulla e sospettare sempre di tutto sapendo di
essere in colpa
• Avere la luna (storta) = essere di cattivo umore, scontroso, irritabile
• Avere l' asso nella manica = avere le migliori possibilità di successo
• Avere le mani bucate = essere spendaccione, spendere esageratamente
• Avere le mani in pasta = essere addentro a un affare, esservi coinvolto, avere esperienza
• Avere lo stomaco di struzzo = mangiare qualunque cosa e digerire tutto
• Avere paura della propria ombra = avere paura di tutto
• Avere sale in zucca = essere molto intelligente
• Avere una marcia in più = avere qualcosa di più degli altri, essere superiore agli altri
3
B
• Il bacio di Giuda = I' ipocrita manifestazione di affetto o amicizia prima o dopo un tradimento
• Bastian contrario = per chi per abitudine o per carattere fa il contrario di quello che fanno gli altri.
• (usare) il bastone e la carota = alternare la maniera dura a quella dolce.
• Battere il ferro finchè è caldo = bisogna intraprendere qualcosa quando la situazione è favorevole.
• La belezza dell' asino = per una persona giovane che non ha grandi doti di avvenenza, ma è
ugualmente piacevole perchè è fresca e graziosa.
• Bere grosso o berle grosse = essere un gran credulone, credere a tutto
• La bestia nera = una persona che si odia, di cui si ha paura
• La bocca della verità = una persona che dice sempre la verità.
• Brutto come il peccato = bruttissimo, orrendo
• Le bugie hanno le gambe corte = le bugie non si possono nascondere e presto o tardi la verità viene
a galla.
C
• Caccia alle streghe = persecuzione, mossa da pregiudizi, sospetti, tabù ecc. infondati e non
documentati.
• Cadere dalla padella nella brace = andare di male in peggio
• Cadere dalle nuvole = meravigliarsi, stupirsi.
• Cadere nelle braccia di Morfeo = cadere in un profondo sonno
• Il calcio dell' asino = disprezzo e rivalsa verso chi, prima potente, è caduto in basso
• Fumare il culumet della pace = fare pace
• Campa cavallo (che l' erba cresce) = un invito ironico a cercare di sopravvivere in attesa di
momenti favorevoli che però sono lontani e incerti.
• Can che abbaia non morde = chi fa molte minacce generalmente non passa ai fatti.
• Cane non mangia cane = un potente non si mette in lotta con un altro potente.
• Canta che ti passa = un invito a non preoccuparsi, a non aver paura.
• Cantare ai sordi (parlare al muro) = parlare invano.
• Il canto del cigno = l' ultima opera pregevole di un artista, politico ecc.
• Capro espiatorio = la persona su cui ricadono le colpe degli altri e che paga per tutti.
• Carne da cannone o carne da macello = la massa anonima di soldati esposti cinicamente alla
morte.
4
• Carta bianca (dare, avere)= pieni poteri, piena facoltà di agire
• Il caval di San Francesco = andare a piedi
• Cavalcare la tigre = cercare di controllare una situazione disperata, pericolosa, in cui si è coinvolti.
• Cavallo di battaglia = una attività, una prova, in cui uno si sente più sicuro, più preparato ed
esprime il meglio di se stesso.
• Cavallo di Troia = dono subdolo che danneggia chi lo riceve.
• Cavar sangue da una rapa = pretendere da qualcuno ciò che non può dare.
• Cercare il pelo nell' uovo = essere estremamente minuziosi e pignoli.
• Cercare rogne (cose fastidiose) = cercare dei guai.
• Chi ben comincia è a metà dell' opera = chi avvia bene un lavoro è come se ne avesse già fatto la
metà.
• Chi di spada ferisce, di spada perisce = chi usa la violenza sarà vittima della violenza.
• Chi fa da sè fa per tre = i propri interessi uno se li deve curare da solo
• Chi ha fatto trenta può fare trentuno = quando si è fatta una gran parte del lavoro conviene
finirlo, che si può arrivare oltre.
• Chi la fa l’ aspetti = chi danneggia gli altri sarà ricompensato con la stessa moneta
• Chi non ha cervello abbia gambe = chi si dimentica di qualche cosa, deve tornare indietro a
prenderla, e quindi deve fare doppia fatica.
• Chi semina vento raccoglie tempersta = chi con le parole o con i fatti provoca del male, spesso è
vittima dello stesso male.
• Chi tace acconsente = chi non risponde a un discorso o non manifesta il proprio dissenso, è d’
accordo.
• Chi troppo vuole nulla stringe = non bisogna eccedere nelle pretese e chi vuole troppo non ottiene
nulla e rischia di perdere anche quello che ha.
• Chiudersi in una torre d’ avorio = isolarsi dal mondo, ingorando tutto quello che è attorno a noi.
• Ci vedremo a Filippi = « non è finita qui, verrà il giorno della resa dei conti, della punizione.
• Cogliere (o prendere) in contropiede = cogliere di sorpresa.
• Colpo di fulmine = un avvenimento improvviso ed inaspettato / innamoramento a prima vista, la
cotta improvvisa.
• Coltivare il proprio orticello = dedicarsi esclusivamente alle proprie cose senza interessarsi degli
altri.
• Come il cacio sui maccheroni = molto opportuno, molto a proposito, proprio come si desiderava.
• Con la testa nel sacco = senza rendersi conto.
5
• Conoscere i propri polli = quando uno sa bene con chi ha a che fare.
• Contare le pecore = un metodo quasi infallibile per vincere l’ insonnia.
• Contento come una Pasqua = contentissimo, felicissimo.
• la Corsa dell’ asino = l’ affrettarsi all’ ultimo momento.
• Cortina di ferro = la linea di separazione ideologica, politica ed economica tra l’ Occidente e i
paesi comunisti europei, dalla fine della guerra alla caduta del muro di Berlino.
• Cosa fatta capo ha = una volta presa una decisione bisogna metterla in atto
• Costruire sulla sabbia = fara qualcosa di effimero, che dura poco, che non ha fondamenta.
• Credere che un asino voli = credere o far credere a cose impossibili.
D
• Da prendere con le molle = si dice di persona o cosa difficile da trattare.
• Dall’ a alla zeta = dal principio alla fine.
• Dare corda (o spago) = concedere la libertà di agire.
• Dare del filo da torcere = procurare difficoltà, ostacolare con ogni mezzo.
• Dare i numeri = parlare a vanvera, sembrare fuori di sè, essere impazzito.
• Dare il colpo di grazia = è il colpo mortale per abbreviare l’ agonia o la sofferenza di qualcuno, l’
ultimo atto di un combattimento.
• Dare la berta = prendere in giro, scherzare.
• Darla a bere = far credere a qualcuno una cosa non vera, far credere quello che non è.
• De gustibus (non est disputandum) = sui gusti non si discute, ognuno ha i suoi gusti.
• Deus ex machina = persona in grado di risolvere situazioni difficili e complesse.
• Di punto in bianco = all’ improvviso
• Il Diavolo insegna a fare le pentole ma non i coperchi = quando una cosa non riesce bene noi
diamo la colpa al diavolo
• Dietro le quinte = di nascosto, senza apparire.
• Dio li fa e poi li accoppia = di persone che stanno bene insieme e che hanno gli stessi difetti.
• Dire a nuora perchè suocera intenda = quando ci si rivolge ad una persona, con l’ intenzione che
il messaggio arrivi ad un’ altra che è la vera interessata a capirlo.
• Dire pane al pane e vino al vino = dire con molta chiarezza come stanno le cose, chiamare le cose
con il loro nome.
• Discutere dell’ ombra dell’ asino = discutere di cose inutili.
6
• Divide et impera (dividi e comanda) = se vuoi comandare devi mettere gli altri in discordia tra di
loro.
• Dormire tra due guanciali = non avere nessuna preoccupazione, vivere tranquillo.
E
• È il principio della fine = per una situazione che sta per precipitare
• Ecce homo (ecco l’ uomo – Pilato presentando Cristo al popolo) = per indicare una persona
fisicamente malridotta.
• Entrare da un orecchio ed uscire dall’ altro = si dice di cosa udita e cancellata subito dalla
memoria.
• L’ erba del vicino è sempre più verde = si riferisce particolarmente a coloro che soffrono d’invidia
e che non sono mai contenti della loro situazione.
• L’ esperienza insegna (experientia docet) = simile al proverbio la pratica val più della grammatica
• Essere al lumicino = essere in fin di vita - essere alla fine delle forze, delle risorse economiche ecc
• Essere al verde = non avere una lira, essere in miseria
• Essere come il diavolo e l’ acqua santa = essere in pieno disacordo, in contrasto, odiarsi a vicenda.
• Essere come il prezzemolo = intrufolarsi dappertutto, essere ovunque.
• Essere di manica larga = essere molto indulgente e tollerante, permissivo.
• Essere il gallo della Checca = essere molto ammirato dalle donne, avere successo con le donne.
• Essere il pomo della discordia = è ciò che causa discordia tra le persone.
• Essere in area di parcheggio = non avere lavoro, essere disoccupato.
• Essere in auge = aver raggiunto grande notorietà, fortuna, gloria.
• Essere in bolletta = vedi essere al verde
• Essere in rodaggio = aver da poco intrapreso un lavoro, essere in prova.
• Essere la ninfa egeria = essere l’ ispiratrice
• Essere la quintescenza = essere l’ intima natura, la verità profonda
• Essere l’ araba fenice = di persona o cosa molto perfetta e quasi impossibile da trovare, più unica
che rara.
• Essere lo zimbello (uccello da richiamo) = essere oggetto di burla e di scherzo
• Essere l’ ultima ruota del carro = non contare nulla in gruppo.
• Essere nato di domenica = non essere molto intelligente, avere poco giudizio
7
• Essere (o fare) il tirapiedi = essere il servitore pedissequo di una persona.
• Essere l’ uccello del malaugurio = di persona che porta cattive notizie.
• Essere (o sentirsi) in vena = essere ben disposto, essere di buon umore, nelle migliori condizioni.
• Essere sano come un pesce = essere sanissimo
• Essere su di giri = essere eccitato, esaltato, euforico
• Essere sulla cresta dell’ onda = essere all’ apice del successo
• Essere tra l’ incudine e il martello = trovarsi in una situazione difficile, essere di fronte ad una
alternativa scomoda e pericolosa.
• Essere un altro paio di maniche = essere un’ altra cosa, del tutto diversa.
• Essere un arpagone (Molière) = una persona molto avara.
• Essere un beota (Beozia) = essere stupido, tardo.
• Essere un calvario = una lunga serie di sofferenze, pene e dolori (da calvarium = teschio >luogo del
cranio >Golgota)
• Essere un capitan fracassa = essere uno sbruffone, persona vanitosa.
• Essere un colosso dai piedi d’ argilla = per chi dimostra di avere una grande forza ma in realtà non
ha solide basi.
• Essere un creso = essere molto ricco
• Essere (fare) il dongiovanni = essere un corteggiatore irresistibile, seduttore, donnaiolo
• Essere un epigono = chi continua ed elabora idee e forme dei suoi predecessori.
• Essere un gradasso (personaggio nell’ Orlando Furioso di Ariosto) = uno spaccone, un orgoglioso
• Essere un mentore (amico di Ulisse) = essere consigliere, guida, maestro, precettore, amico fidato.
• Essere un oracolo = essere una verità incontestabile, una verità assoluta.
• Essere un paria (casta bassa e povera dell’ India merid.) = essere un emarginato, un poveraccio.
• Essere un pigmalione = colui che ammaestra e indirizza qualcuno, specialmente una donna,
affinandone e sviluppandone le facoltà intellettuali e il comportamento.
• Essere un satiro = essere un uomo lussurioso, libidinoso.
• Essere un voltagabbana (mantello) = chi per utilità o comodo cambia facilmente opinione, partito,
alleanze ecc.
• Essere una babele = per un luogo di grande baccano e confusione.
• Essere una cassandra = persona che è solita fare previsioni catastrofiche, senza che nessuno le
presti fede.
• Essere una circe = sinonimo di seduttrice, ingannatrice.
8
• Essere una palla al piede = essere un grave impedimento, un ostacolo, un grave peso che rallenta o
impedisce di fare qualcosa.
• Essere una panacea = un rimedio adatto per risolvere ogni problema.
• Essere una piaga (in tono familiare e scherzoso) = essere una persona insopportabile, pesante,
noiosa
• Essere una sfinge = essere una persona enigmatica, di cui non si riescono a capire il pensiero, I
sentimenti, le intenzioni.
• Essere una sibilla = per una donna che fa profezie, che prevede il futuro.
• Est modus in rebus = in tutte le cose c’ è una misura (Orazio) – in medio stat virtus (Aristotele).
• Eureka = esclamazione che esprime gioia per aver trovato una soluzione ad un problema difficile
(Archimede).
• Ex professo = per professione, con competenza.
F
• Facile è criticare, difficile è l’ arte = per coloro che criticano tutto senza competenza.
• Far fiasco = non riuscire, non aver successo.
• Farci la birra = non fare nulla di una cosa, non avere alcun valore.
• Fare bancarotta = fallire, avere un insuccesso totale.
• Fare castelli in aria = fare progetti che non si possono realizzare.
• Fare cilecca (per armi da fuoco quando scattano a vuoto) = fallire, non riuscire.
• Fare della solfa (da le note sol e fa) = ripetere noiosamente, con monotonia, parole, discorsi,
prediche, atti ecc.
• Fare d’ ogni erba un fascio = mettere insieme confusamente cose disparate senza alcuna
distinzione, ragionare senza riflettere facendo confusione.
• Fare i conti senza l’ oste = decidere senza tener conto di chi è coinvolto nella vicenda, fare dei
progetti senza considerare gli imprevisti.
• Fare il portoghese = riuscire a entrare in un luogo di spettacolo senza pagare il biglietto.
• Fare la spola = andare avanti ed indietro da un luogo ad un altro
• Fare le cose alla carlona (da Carlo Magno « re Carlone ») = alla buona, grossolanamente, senza
cura.
• Fare le ore piccole = andare a letto molto tardi.
• Fare la gattamorta = fingere di essere ingenuo, di non accorgersi di nulla.
9
• Fare (o metterci) una croce sopra = chiudere un argomento
• Fare orecchie da mercante (fare il pesce in barile) = far finta di non sentire.
• Fare un buco nell’ acqua = fare una cosa inutile, non riuscire in un’ impresa.
• Fare un tiro mancino = fare un atto maligno o cattivo a qualcuno.
• Fare una sviolinata = lusingare sfacciatamente.
• Fare una vita da cane = fare una vita stentata, faticosa, piena di sacrifici.
• La Farina del diavolo va tutta in crusca = la ricchezza o i vantaggi ottenuti con mezzi illeciti si
perdono facilmente.
• Fidarsi è bene ma non fidarsi è meglio = nella vita ci vuole fiducia, occorre però non essere
ottimisti ad oltranza.
• Figlio prodigo = colui che si pente di ciò che ha fatto e torna sulle sue decisioni.
• Finchè c’è vita c’è speranza = non bisogna mai disperare.
• Il Fine giustifica i mezzi = per raggiungere uno scopo, si può ricorrere a qualsiasi mezzo.
• Fischiare le orecchie (mi fischiano le orecchie) = per quando si ha l’ impressione che in quel
momento qualcuno stia parlando di noi e non sempre benevolmente.
• Forzare la mano = esagerare, fare troppa pressione.
G
• Galeotto fu il libro e chi lo scrisse ( da Galeotto che favorì l’amore tra Ginevra, moglie di re Artù, e
il cavaliere Lancillotto – la storia di Paolo e Franesca , Divina Commedia) = intermediario in amore,
mezzano.
• La Gallina dalle uova d’ oro = indica una fonte sicura e facile di guadagno che non finisce mai.
• Gatta ci cova = qualcosa non va, c’ è un trucco, un inganno, un’ insidia
• Gettare le perle ai porci = dare cose preziose a chi non è in grado di valutarle.
• Gettare olio sul fuoco = attizzare l’ ira, rinfocolare risentimenti, odi ecc.
• Gratis et amore dei (gratis) –
• Gridare al lupo = dare un allarme per nulla
• Guastare la festa = comportarsi in modo da disturbare una festa, un incontro piacevole, una lieta
conversazione, un affare, ecc.
10
H
• Homo homini lupus = l’ uomo è un lupo per un altro uomo, nemico dei suoi simili
• Honni soit qui mal y pense (francese antico) = male incolga a chi pensa male, indirizzato a coloro
che si scandalizzano di tutto e che sono portati a vedere il male anche dove non c’ è.
I
• In barba a… (alla barba di…) = a dispetto di…, alla faccia di…, nonostante il divieto di…
• In bocca al lupo = l’ augurio che si fa a chi sta per affrontare una situazione difficile, un esame, un
affare, un pericolo. (dal gergo dei cacciatore – buona caccia)
• In costume adamitico = con le vesti di Adamo, nudo.
• In nero (al nero) = senza tutela legislative, e senza contributi assicurativi (riferita al lavoro).
• In un fiat (dalla Bibbia, Genesi: Fiat lux) = in pochissimo tempo, in un attimo.
• In vino veritas = nel vino la verità
• Indorare la pillola = usare parole meno dure per rendere più accettabile una cosa non gradita
• Indovinala grillo = nei momenti di incertezza, quando non si sa con precisione che cosa fare o
come andrà a finire.
• L’ Inferno è lastricato di buone intenzioni = i buoni propositi non bastano per raggiungere buoni
risultati.
• Ingoiare il rospo = far buon viso e cattivo gioco, accettare una cosa sgradita.
• Inter nos = tra noi, in confidenza, a quattr’ occhi.
• Ipse dixit = lo ha detto lui, cioè il maestro, e non si può quindi mettere in discussione.
L
• Lacrime di coccodrillo = pentimento tardivo o finto.
• Lambiccarsi il cervello = sforzarsi di capire o di trovare una soluzione ad un problema.
• Lapsus linguae - lapsus calami = errore di lingua, errore di penna, un passo falso, una svista, un
errore involontario che si fa scrivendo o parlando.
• Lasciar cuocere nel proprio brodo = lasciare che uno faccia quel che gli pare, disinteressarsi di
uno che vuole fare di sua testa.
• Lavare i panni sporchi in casa (i panni sporchi si lavano in famiglia) = le magagne, le liti, gli
errori, ecc. Non si devono mettere in pubblico, ma si devono risolvere nel proprio ambiente.
11
• Lavarsene le mani (Ponzio Pilato) = non volere la responsabilità di qualcosa, lasciare ad altri la
responsabilità di decidere.
• Legare l’ asino dove vuole il padrone = fare come vuole chi comanda per non avere noie.
• Legarsela al dito = ricordarsi bene di un torto ricevuto aspettando poi di vendicarsi.
• Lemme – lemme = adagio adagio, lentamente, pian piano.
• Il Letto di Procuste = una situazione di costrizione estremamente tormentosa.
• La Lingua batte dove il dente duole = quando uno ritorna continuamente su certi argomenti che gli
stanno a cuore, anche se vuole fare finta di passarci sopra.
• Lontani dagli occhi, lontani dal cuore = quando una persona si allontana da noi anche i nostri
affetti si attenuano, la lontananza fa dimenticare la persona amata.
• Luna di miele = il primo periodo di matrimonio, tutto dolcezza e felicità.
• Lupus in fabula = il lupo nella favola che appare per lo più all’ improvviso. Quando arriva una
persona che ci obbliga a troncare il discorso o si usa per dire « stiamo proprio parlando di te ! ».
M
• Maddalena (pentita) = una donna che si mostra umiliata e pentita con un po’ di ipocrisia.
• Mangiare la foglia = capire una cosa al volo.
• Una Mano lava l’ altra = l’ aiuto reciproco è sempre vantaggioso.
• Maramaldo (tu uccidi un uomo morto) = sinonimo di persona vile e codarda che inferisce sugli
inermi o tradisce facilmente.
• Meglio un asino vivo che un dottore morto = la salute vale più del sapere e non bisogna rovinarsi
la salute studiando troppo.
• Meglio un uovo oggi che una gallina domani = meglio accontentarsi del poco subito che del molto
domani.
• Menare il can per l’ aia = mandare per le lunghe senza concludere.
• Menare per il naso = prendere in giro qualcuno.
• Mens sana in corpore sano = mente sana in corpo sano.
• Metodi draconiani = severissimo, durissimo, spietato.
• Mettere al bando = escludere, allontanare.
• Mettere alla berlina = rendere qualcuno ridicolo al pubblico.
• Mettere i puntini sulle i = essere molto preciso.
• Mettere il becco = intromettersi, essere invadenti.
12
• Mettere il carro innanzi ai buoi = fare una cosa che dovrebbe essere fatta dopo un’ altra.
• Mettere la mano sul fuoco = un modo enfatico per garantire ciò che si afferma.
• Mettere la pulce nell’ orecchio = insinuare in qualcuno sospetti ed inquietudini.
• Mettere le carte in tavola = dire chiaramente quello che si pensa.
• Modus vivendi = modo di vivere.
• Moglie e buoi dei paesi tuoi = è bene sposare una donna che ci è familiare perchè nata e cresciuta
nello stesso ambiente.
• Mordere la polvere = restare sconfitto e umiliato.
• Mors tua vita mea = la tua morte è la mia vita.
• Mosca cocchiera = persona modesta che si illude di essere molto importante.
N
• Nascere con la camicia (nascere vestito) = essere fortunato
• Nessuno è profeta in patria (nemo est propheta in patria) = raramente un uomo di prestigio gode
del fascino nella terra in cui è nato.
• Nodo gordiano = una questione estremamente difficile o complessa da risolvere.
• Non avere nè arte nè parte = non conoscere alcun mestiere, non saper far nulla.
• Non avere nè capo nè coda = di cosa disordinata e inconcludente, che non ha nè principio nè fine.
• Non capire un’ acca = non capire niente.
• Non cavare un ragho dal buco = non riuscire a niente.
• Non c’ è rosa senza spine = ogni cosa bella o desiderabile ha necessariamente i suoi lati meno belli
o spiacevoli.
• Non è farina del tuo sacco = non è un’ idea tua, non è cosa scritta da te.
• Non è tutto oro quello che luce = non tutto ciò che splende esteriormente è in realtà prezioso.
• Non plus ultra = non più oltre, per una cosa ritenuta la migliore, la massima meta che si possa
raggiungere.
• Non sapere a che santo votarsi = non sapere a chi ricorrere trovandosi in uno stato di bisogno.
• Non si vive di solo pane = oltre allo stipendio occorrono altri riconoscimenti come la stima, la
gratitudine, l’ affetto, ecc.
• Non stare più nei panni = non riuscire a contenere un’ emozione.
13
O
• O mangiar questa minestra o saltar quella finestra = per dire che non c’è altra via d’ uscita, che
si deve scegliere il male minore.
• L’ occhio del padrone ingrassa il cavallo = l’ attenzione costante e amorevole del padrone
conserva e fa crescere i propri beni.
• L’ occhio vuole la sua parte = le cose presentate bene piacciono di più poichè soddisfano le
esigenze estetiche.
• L’ ospite è come il pesce : dopo tre giorni puzza = chi riceve ospitalità non deve approfittarne e
deve ridurre al massimo la sua permanenza in casa altrui.
P
• Paese che vai, usanza che trovi = bisogna adattarsi agli usi e alle persone del luogo.
• Il paese di Bengodi = per designare un luogo d’ abbondanza dove c’ è ogni ben di Dio.
• Pagare alla romana = pagare ciascuno per proprio conto (al ristorante, al bar, al cinema ecc.)
• Panem et circenses = per indicare le concessioni offerte al popolo per tenerselo buono.
• Parenti serpenti, fratelli coltelli = per gli odi tra parenti.
• Parlare di corda in casa dell’ impiccato = parlare in modo inopportuno, a sproposito, fare una
gaffe.
• Passare il Rubicone = prendere una risoluzione decisiva (sinonimica « di alea iacta est »).
• Passare (o fare) la notte in bianco = trascorrere l’ intera notte senza riuscire a dormire.
• Pecora nera = una persona che in un gruppo, comunità o ambiente si distingue dagli altri per doti
negative e che è vista con disprezzo.
• Per filo e per segno = in modo continuo e dettagliato, in ordine e con estrema precisione.
• Perdere il filo del discorso = perdere la continuità del discorso.
• Perdere le staffe = perdere il controllo di sè, arrabbiarsi violentemente.
• Pesce d’ aprile = una burla che si fa il primo giorno di aprile.
• Pestare l’ acqua nel mortaio = fare una fatica inutile senza ottenere alcun risultato.
• Piantare in asso = abbandonare improvvisamente uno quando meno se l’ aspetta (da: piantare in
Nasso > Teseo che abbandonò Arianna nell’ isola di Nasso).
• Piove, governo ladro! = frase con cui si scarica sul governo ogni contrarietà.
• Piovere sul bagnato = per dire che ad una disgrazia se ne sono aggiunte delle altre.
• Porgere l’ altra guancia = accettare con rassegnazione umiliazioni, ingiurie (risale al Vangelo)
14
• Portare scalogna = portare sfortuna, disgrazia.
• Prendere sottogamba = prendere con leggerezza, con troppa disinvoltura.
• Prendere due piccioni con una fava = raggiungere due scopi diversi con una sola operazione.
• Prendere (cogliere) in castagna = cogliere in fallo, sorprendere in flagrante.
• Prendere in giro = canzonare, burlare, beffare.
• Prendere una cantonata = fare un grosso errore.
• Principe azzurro = lo sposo ideale a lungo sognato dalle adolescenti.
• Promettere mari e monti = fare grandi promesse senza l’ intenzione di mantenerle.
Q
• La quadratura del cerchio (o del circolo) = problema insolubile.
• Qui pro quo = un banale errore, un malinteso (in grammatica, uno scambio di una lettera con un’
altra) – qui (pronome relativo nominativo) – quo (ablativo).
R
• Rara avis (uccello raro) = una persona o una cosa rara che ha dei pregi particolari.
• Il re è morto, viva il re ! = la morte di una persona non rappresenta la fine delle istituzioni, che
nessuno è indispensabile.
• Reggere il lume (o il moccolo) = favorire i rapporti fra due innamorati facilitandone l’ incontro.
• Restare con un palmo di naso = rimanere deluso, insoddisfatto, ingannato.
• Restare di sale (di sasso) = rimanere attonito, rimanere sbalordito.
• Restare lettera morta = di qualcosa che rimane senza validità, effetto o applicazione.
• Riposare sugli allori = restarsene inattivo, pago dei successi ottenuti.
• Rispondere a tono = rispondere a proposito.
• Rispondere per le rime = rispondere energicamente per passare poi dalla difesa all’ attacco.
• Rispondere picche = opporre un deciso rifiuto.
• Rompere il ghiaccio = superare le prime difficoltà in una conversazione.
• Rompere le uova nel paniere = compromettere, far fallire.
S
• Salire al settimo cielo = essere felicissimo.
• Saltare di palo in frasca = passare da un argomento all’ altro all’ improvviso e in modo sconnesso.
15
• Saperne una più del diavolo = essere furbissimo, astutissimo, conoscere tutti i trucchi.
• Sbarcare il lunario = riuscire a campare stentatamente, tirare avanti alla meglio, arrangiarsi.
• Lo scheletro nell’ armadio = un segreto imbarazzante.
• Se son rose fioriranno = esprime un dubbio, solo dopo aver visto gli effetti si potrà giudicare.
• Seminare zizzania = seminare discordia, malcontento.
• Sordo come una campana = essere completamente sordo, essere molto duro d’ orecchio.
• Sotto l’ egida = sotto la tutela (da egida = scudo di Giove e Minerva)
• Spada di Damocle = il pericolo sempre presente, la minaccia incombente, la precarietà del
benessere.
• Spezzare una lancia (in favore di qualcuno) = prendere le difese di uno contro i suoi avversari.
• Star fresco = essere in una situazione difficile e avere la prospettiva di essere punito.
• Stare sulle sue = darsi delle arie, trattare gli altri con distacco.
• Statu(s) quo = uno stato di cose bloccato nella condizione preesistente.
• Sudare sette camicie = fare una grande fatica, un grande sforzo.
T
• Tabula rasa = essere privo d’ idee, di nozioni, di cognizioni elementari.
• Tagliare la corda = andarsene in fretta, scappare.
• Il tallone d’ Achille = il punto vulnerabile, il punto debole di una persona.
• Tanto va la gatta al lardo che ci lascia lo zampino = chi commette azioni malvage presto o tardi
sarà scoperto.
• La tela di Penelope = per un lavoro che sembra non finire mai.
• Tenere il piede su due staffe = tenersi amici due partiti o due persone tra loro opposte per uscire
senza danno da una situazione.
• Tenere in scacco = mettere uno in condizioni, in modo che non si possa più muovere, bloccarlo.
• Tenere (o far stare) sulla corda = lasciare nell’ incertezza tormentosa.
• Terra di nessuno = per indicare un argomento che nessuno osa affrontare o un luogo abbandonato
da tutti, in cui tutti possono fare quello che vogliono.
• Tirare l’ acqua al proprio mulino = volgere una situazione a proprio favore, fare il proprio
interesse.
• Tirare le cuoia = morire (cuoia sta per pelle).
• Tizio, caio e sempronio = indicano persone indeterminate che non si vuole o non si può nominare.
16
• Toccare ferro = fare scongiuri.
• Tornare con le pive nel sacco = tornare senza aver concluso nulla.
• Tra Scilla e Cariddi = trovarsi tra due gravi pericoli.
• Trovare pane per i propri denti = trovarsi di fronte a un avversario molto duro o a situazioni che
faranno tribolare.
• La (tua) sinistra non sappia quello che fa la (tua) sestra = chi fa del bene non deve ostentarlo.
• Tutte le strade portano a Roma = c’ è sempre una qualche via, anche se lunga e tortuosa, che può
portarci a raggiungere uno scopo.
• Tutti i nodi vengono al pettine = le situazioni o le cose poco chiare, tenute nascoste, presto o tardi
verranno alla luce.
U
• Uccidere il vitello grasso = per indicare un lauto banchetto per festeggiare il ritorno di una persona
amata lontana da molto tempo.
• L’ ultimo a comparir fu Gambacorta = di chi è abituato ad arrivare sempre ultimo e a chi è più
lento degli altri.
• Una rondine non fa primavera = un solo segno lieto o positivo non deve farci credere che la
situazione volga veramente al meglio.
• Unire l’ utile al dilettevole = locuzione presa da Orazio (« Ottiene pieni voti chi unisce l’ utile al
piacevole »).
• Uomo di paglia = un prestanome, comparsa.
• Uovo di Colombo = di un espediente facilissimo o di una soluzione semplicissima.
• Urbi et orbi = in tono scherzoso significa « a tutti, dappertutto », sbandierare una notizia ai quattro
venti.
V
• Il vaso di Pandora = una cosa apparentemente stupenda che, tuttavia, può provocare catastrofi se
qualcuno cerca di modificarla.
• Vecchio bacucco = di persona vecchia e rimbecillita (vecchio rimbambito).
• Vedere le stelle = provare un fortissimo dolore fisico.
• Veni vidi vici (Giulio Cesare) = per manifestare soddisfazione per la rapida e felice riuscita di un’
impresa.
17
• La verità viene sempre a galla
• La vittoria di Pirro = una vittoria ottenuta a prezzo di danni cosi gravi da mettere sullo stesso piano
vincitore e vinto.
• La voce (o il richiamo) del sangue = l’ istinto che fa riconoscere e amare i propri parenti.
• Volere la botte piena e la moglie ubriaca = volere due cose che non si possono avere insieme,
volere due vantaggi di cui uno esclude l’ altro.
• Volo pindarico = un brusco passaggio da un argomento all’ altro.
• Vox populi, vox Dei = l’ opinione pubblica spesso corrisponde a verità.
• Vuotare il sacco = dire tutto quello che si sa o si pensa senza riserve o pudori.
Z
• Zoccolo duro = nel linguaggio politico una base elettorale tenacissima / fan delle soap opera, dei
serial televisivi.
"""
#
# tr4w = TextRank4Keyword()
# tr4w.analyze(text, candidate_pos = ['NOUN', 'PROPN'], window_size=4, lower=False)
# tr4w.get_token_pairs(4, [sentence.split() for sentence in text.split("\n")])
#

######
import pytextrank

phrases = text.split("\n")

# add PyTextRank to the spaCy pipeline
tr = pytextrank.TextRank()
nlp.add_pipe(tr.PipelineComponent, name="textrank", last=True)

doc = nlp(text)

# examine the top-ranked phrases in the document

for p in doc._.phrases:
    print("{:.4f} {:5d}  {}".format(p.rank, p.count, p.text))
    print(p.chunks)
###


############
############

# Pattern mining con spacy

from collections import OrderedDict
import numpy as np
import stanza
from spacy_stanza import StanzaLanguage
from spacy.lang.en.stop_words import STOP_WORDS

# Spacy NLP pipeline
snlp = stanza.Pipeline(
    processors="tokenize,mwt,pos,lemma", lang="it", use_gpu=False
)  # pos_batch_size=3000
nlp = StanzaLanguage(snlp)

from tqdm import tqdm
from spacy.matcher import Matcher

# /home/nazareno/CELI/repositories/python_projects/texmega_py/embeddings_trainer/GloVe/texmega_corpus_with_return

phrases = open(
    "/home/nazareno/CELI/repositories/python_projects/texmega_py/embeddings_trainer/GloVe/texmega_corpus_with_return",
    "r",
    encoding="utf8",
).readlines()
patterns = [
    [{"POS": "NOUN"}, {"POS": "VERB"}, {"POS": "NOUN"}],
    [{"POS": "VERB"}, {"POS": "NOUN"}],
]
matcher = Matcher(nlp.vocab)
matcher.add("noun-verb-noun", None, patterns[0])
matcher.add("verb-noun", None, patterns[1])
for phrase in tqdm(phrases):

    phrase = " ".join([word.split("_")[0] for word in phrase.split()])

    doc = nlp(phrase)
    matches = matcher(doc)
    for match_id, start, end in matches:
        string_id = nlp.vocab.strings[match_id]
        span = doc[start:end]
        print("\n", string_id, start, end, span.text)

### LOAD AND CLEAN CORPUS FROM TAGS AND NUMBERS

###
from tqdm import tqdm

corpus_paths = [
    "/home/nazareno/CELI/repositories/python_projects/texmega_py/embeddings_trainer/GloVe/texmega_corpus_with_return",
    "/home/nazareno/CELI/repositories/python_projects/texmega_py/pattern_mining/pattern_mining_corpus_frequency_balanced_2020_09_07.txt",
]

phrases = []
for corpus_path in corpus_paths:
    phrases += open(corpus_path, "r", encoding="utf8").readlines()

phrases = [
    " ".join([word.split("_")[0] for word in phrase.split()])
    for phrase in tqdm(phrases)
]
phrases = [re.sub("[0-9]+", "", phrase) for phrase in tqdm(phrases)]

with open(
    "/home/nazareno/CELI/repositories/python_projects/texmega_py/embeddings_trainer/GloVe/normal_plus_pattern_corpus_no_return.txt",
    "w",
    encoding="utf8",
) as file:
    for p in phrases:
        file.write(p)
        file.write(" ")

with open(
    "/home/nazareno/CELI/repositories/python_projects/texmega_py/embeddings_trainer/GloVe/normal_plus_pattern_corpus_with_return.txt",
    "w",
    encoding="utf8",
) as file:
    for p in phrases:
        file.write(p)
        file.write("\n")


### LOAD AND CLEAN CORPUS FROM SINGLE_WORD LINE

corpus_paths = [
    "/home/nazareno/CELI/repositories/python_projects/texmega_py/resources/raw/TO_BE_USED/Dizionario_collocazioni_cleaned_processed.txt",
    "/home/nazareno/CELI/repositories/python_projects/texmega_py/pattern_mining/pattern_mining_corpus_frequency_balanced_2020_09_07.txt",
]

phrases = []
for corpus_path in corpus_paths:
    phrases += open(corpus_path, "r", encoding="utf8").readlines()

phrases_with = [phras for phras in phrases if "/" in phras]
phrases_without = [phras for phras in phrases if "/" not in phras]

import re

phrases_with_new = []
for p in phrases_with:
    new_p = ""
    for np in p.split():
        if "/" not in np:
            new_p += np + " "
        else:
            new_p += np.split("/")[0] + " "
    print(new_p)
    phrases_with_new.append(new_p)

phrases = phrases_with_new + phrases_without

with open(
    "/home/nazareno/CELI/repositories/python_projects/texmega_py/resources/raw/TO_BE_USED/Dizionario_collocazioni_cleaned_processed_2.txt",
    "w",
    encoding="utf8",
) as file:
    for p in phrases:
        file.write(p)
        file.write("\n")

len("'! '".split())

lines_int = set(
    open("/home/nazareno/Desktop/lex_it.txt", "r", encoding="utf8").readlines()
)
lines_int = set([line.strip() for line in lines_int])
lines_en = set(
    open("/home/nazareno/Desktop/lex_en.txt", "r", encoding="utf8").readlines()
)
lines_en = set([line.strip() for line in lines_en])

intersection = lines_en.intersection(lines_int)


with open(
    "/home/nazareno/CELI/repositories/python_projects/texmega_py/resources/it_en_intersection.txt",
    "w",
    encoding="utf8",
) as file:
    for p in sorted(intersection):
        file.write(p)
        file.write("\n")

###
lines_int = set(
    open(
        "/home/nazareno/CELI/repositories/python_projects/texmega_py/resources/it_en_intersection.txt",
        "r",
        encoding="utf8",
    ).readlines()
)
lines_int = set([line.strip() for line in lines_int])
from langdetect import detect
from tqdm import tqdm

lines_int_2 = []
for line in tqdm(lines_int):
    if detect(line) == "it":
        lines_int_2.append(line)
    else:
        print(f"{line} is not italian")

lines_int_2 = set(lines_int_2)
lines_int -= lines_int_2
len(lines_int)

## FIX lemma_pos_cache con morpho celi
import pickle

lemma_pos_cache_name = "lemma_pos_cache_[glove_[min100_300_win10_ite50_xmax10_041]].pkl"
cache_path = f"/home/nazareno/CELI/repositories/python_projects/texmega_py/resources/lemma_pos_cache/{lemma_pos_cache_name}"
lemma_pos_cache_unfixed = pickle.load(open(cache_path, "rb"))
import pandas as pd
from tqdm import tqdm

df = pd.read_csv("/home/nazareno/Downloads/lex_it.txt", delimiter="\t")
df = df[df["status"] == "W"]
df = df[["token", "pos", "lemma"]]

# from collections import defaultdict
# lemma_pos_cache_fixed = defaultdict(list)
for word in tqdm(list(lemma_pos_cache_unfixed.keys())):
    pos_list = sorted(set(df.loc[df["token"] == word]["pos"]))
    lemma_pos_cache_unfixed[word][1] = pos_list

pickle.dump(lemma_pos_cache_unfixed, open(f"{lemma_pos_cache_name}_fixed.pkl", "wb"))

#####

words_in_pattern = set()
for p in open(
    "/home/nazareno/CELI/repositories/python_projects/texmega_py/pattern_mining_corpus_2020_09_03.txt",
    "r",
    encoding="utf8",
).readlines():
    for word in p.split():
        words_in_pattern.add(word)
