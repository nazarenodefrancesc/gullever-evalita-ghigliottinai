# TEXMEGA_PY

## INSTALLAZIONE

- (SOLO PER WINDOWS) prima di avviare il setup.py, installare pytorch

    -   ONLY-CPU: pip install torch==1.4.0+cpu torchvision==0.5.0+cpu -f https://download.pytorch.org/whl/torch_stable.html
    -   CUDA10.1: pip install torch==1.5.0+cu101 torchvision==0.6.0+cu101 -f https://download.pytorch.org/whl/torch_stable.html --user

- Per installare il setup.py:

    -   pip install --user -e .

- Per scaricare i modelli di stanza in italiano:

    -   python download_stanza_models.py
    
- Per scaricare i modelli di Glove/Word2Vec basati su Wikipedia italiano:

    - http://hlt.isti.cnr.it/wordembeddings/skipgram_wiki_window10_size300_neg-samples10.tar.gz
    - http://hlt.isti.cnr.it/wordembeddings/glove_wiki_window10_size300_iteration50.tar.gz
   
- Per installare Glove-Trainer:
    - pip install https://github.com/JonathanRaiman/glove/archive/master.zip

- Build immagine docker:
    - docker build -t evalita-ghigliottinai:<data> .

- Run immagine docker:
    - docker run -p 5001:5001 --name evalita-ghigliottinai evalita-ghigliottinai:<data>