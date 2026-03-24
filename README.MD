# GU.L.LI.VER — GUillotine gLove ceLI resolVER

> **An AI system for solving the Italian word game "La Ghigliottina" from the TV show *L'Eredità* (RAI).**

This repository contains the code for the paper:

> **N. De Francesco, G. Moreno** — *GU.L.LI.VER @ EVALITA 2020: GUillotine solver leveraging gLoVe representations and cELI linguistic resources.*
> In: Proceedings of the 7th Evaluation Campaign of Natural Language Processing and Speech Tools for Italian (EVALITA 2020), CEUR-WS, Vol. 2765.
>
> 📄 **Paper:** [ceur-ws.org/Vol-2765/paper126.pdf](https://ceur-ws.org/Vol-2765/paper126.pdf)

---

## 📖 About

**La Ghigliottina** is the final game of the Italian TV quiz show *L'Eredità*. The contestant is given 5 clue words and must find a single word that is semantically related to all of them.

**GU.L.LI.VER** tackles this challenge by combining:
- **GloVe word embeddings** trained on a custom Italian corpus
- **Co-occurrence matrices** and **PMI (Pointwise Mutual Information)** for candidate re-ranking
- **POS tagging** and linguistic filtering (noun/adjective constraints)
- **IDF-based weighting** from Italian language resources

The system participated in the [Ghigliottin-AI shared task](https://ghigliottin-ai.github.io/) at **EVALITA 2020**.

---

## 🚀 Getting Started

### Prerequisites

- Python 3.7+
- [Git LFS](https://git-lfs.github.com/) (large binary files are stored via Git LFS)

### 1. Clone the repository

```bash
git lfs install
git clone https://github.com/nazarenodefrancesc/gullever-evalita-ghigliottinai.git
cd gullever-evalita-ghigliottinai
```

### 2. Decompress resource files

The pre-trained models and data files are stored as compressed `.pkl.zip` archives. **You must unzip them before running the system:**

```bash
# Unzip all .pkl.zip files in-place (each .zip contains the corresponding .pkl)
find . -name "*.pkl.zip" -exec sh -c 'unzip -o "$1" -d "$(dirname "$1")"' _ {} \;
```

This will extract the following resources:

| Compressed file | Description |
|---|---|
| `cooccurrence_matrix/coocurrence_texmega_corpus_songs.pkl` | Co-occurrence matrix |
| `cooccurrence_matrix/pmi_dictionary_texmega_corpus_songs.pkl` | PMI dictionary |
| `resources/embeddings/glove_[...].pkl` | GloVe embeddings model |
| `resources/lemma_pos_cache/lemma_pos_cache_[...].pkl` | Lemma/POS cache |
| `resources/lemma_pos_cache/stemm_cache.pkl` | Stemming cache |
| `resources/word_norm_idf/word_norm_idf_dict.pkl` | Normalized IDF dictionary |

### 3. Install dependencies

**On Windows**, install PyTorch first:
```bash
# CPU only
pip install torch==1.4.0+cpu torchvision==0.5.0+cpu -f https://download.pytorch.org/whl/torch_stable.html
# CUDA 10.1
pip install torch==1.5.0+cu101 torchvision==0.6.0+cu101 -f https://download.pytorch.org/whl/torch_stable.html
```

Then install the package:
```bash
pip install --user -e .
```

This will also download the required [Stanza](https://stanfordnlp.github.io/stanza/) Italian NLP models.

---

## 🎮 Usage

### Run locally

```bash
python main.py
```

Edit `config.py` to change the test dataset, search parameters, and guillotine clues.

### REST API Server

Start the server:
```bash
# With Stanza NLP
python server_rest.py

# Lite version (without Stanza)
python server_rest_lite.py
```

Query the solver:
```
GET http://localhost:5001/solve_guillotine?guillotine=gerusalemme,quattro,fuori,castello,lunghe
```

### Docker

```bash
# Build
docker build -t gulliver-ghigliottinai .

# Run
docker run --name gulliver -d -p 9000:9000 gulliver-ghigliottinai

# Query
curl "http://localhost:9000/solve_guillotine?guillotine=gerusalemme,quattro,fuori,castello,lunghe"
```

---

## 📁 Project Structure

```
├── main.py                    # Main entry point for batch evaluation
├── config.py                  # Configuration and hyperparameters
├── server_rest.py             # Flask REST API server (full)
├── server_rest_lite.py        # Flask REST API server (no Stanza)
├── setup.py                   # Package setup and post-install hooks
├── Dockerfile                 # Docker image definition
├── texmega/                   # Core solver: TexMEGA algorithm
├── pattern_mining/            # Pattern mining utilities
├── embeddings_trainer/        # GloVe/Word2Vec training scripts
│   └── GloVe/                 # GloVe C implementation
├── cooccurrence_matrix/       # Pre-computed co-occurrence & PMI data (.pkl.zip)
├── resources/
│   ├── embeddings/            # Pre-trained GloVe model (.pkl.zip)
│   ├── lemma_pos_cache/       # Cached lemma/POS lookups (.pkl.zip)
│   ├── word_norm_idf/         # IDF dictionary (.pkl.zip)
│   ├── blacklist.txt          # Words to exclude
│   ├── whitelist.txt          # Candidate solution words
│   └── stopwords-it.txt       # Italian stopwords
└── test_datasets/             # Evaluation datasets (JSON/XML/CSV)
```

---

## ⚙️ Configuration

Key parameters in `config.py`:

| Parameter | Description | Default |
|---|---|---|
| `SEARCH_MODE` | Search strategy | `"exhaustive_search"` |
| `MOST_SIMILAR_TOP_N` | Top-N similar words per clue | `375` |
| `REORDERED_RESULTS` | Candidates to re-rank with PMI | `75` |
| `COOCURRENCE_WEIGHT` | PMI weight in re-ranking | `10` |
| `KEEP_POS` | Allowed POS tags for solutions | `["N", "AJ"]` |
| `TEST_PATH` | Path to evaluation dataset | `ghigliottinai_dataset.json` |

---

## 📚 Optional Resources

Pre-trained Italian word embeddings from [HLT-ISTI CNR](http://hlt.isti.cnr.it/):
- [Skip-gram (Wikipedia)](http://hlt.isti.cnr.it/wordembeddings/skipgram_wiki_window10_size300_neg-samples10.tar.gz)
- [GloVe (Wikipedia)](http://hlt.isti.cnr.it/wordembeddings/glove_wiki_window10_size300_iteration50.tar.gz)

---

## 📝 Citation

If you use this code in your research, please cite:

```bibtex
@inproceedings{defrancesco2020gulliver,
  title     = {{GU.L.LI.VER @ EVALITA 2020: GUillotine solver leveraging gLoVe representations and cELI linguistic resources}},
  author    = {De Francesco, Nazareno and Moreno, Giovanni},
  booktitle = {Proceedings of the 7th Evaluation Campaign of Natural Language Processing and Speech Tools for Italian (EVALITA 2020)},
  series    = {CEUR Workshop Proceedings},
  volume    = {2765},
  year      = {2020},
  url       = {https://ceur-ws.org/Vol-2765/paper126.pdf}
}
```

---

## 📄 License

This project was developed at [CELI s.r.l.](https://www.celi.it/) — a company of the [H-FARM](https://www.h-farm.com/) group.
