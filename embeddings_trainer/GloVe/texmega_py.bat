@echo on
chcp 65001

make

set CORPUS=texmega_corpus
set VOCAB_FILE=texmega_vocab.txt
set COOCCURRENCE_FILE=texmega_cooccurrence.bin
set COOCCURRENCE_SHUF_FILE=texmega_cooccurrence.shuf.bin
set BUILDDIR=build_win
set SAVE_FILE=texmega_corpus_vectors
set VERBOSE=2
set BINARY=2
set MEMORY=14.0
set NUM_THREADS=3

set VECTOR_SIZE=600
set VOCAB_MIN_COUNT=200
set MAX_ITER=50
set WINDOW_SIZE=10
set X_MAX=10

cd %BUILDDIR%
vocab_count -min-count %VOCAB_MIN_COUNT% -verbose %VERBOSE% < ../%CORPUS% > ../%VOCAB_FILE%
cooccur -memory %MEMORY% -vocab-file ../%VOCAB_FILE% -verbose %VERBOSE% -window-size %WINDOW_SIZE% < ../%CORPUS% > ../%COOCCURRENCE_FILE%
shuffle -memory %MEMORY% -verbose %VERBOSE% < ../%COOCCURRENCE_FILE% > ../%COOCCURRENCE_SHUF_FILE%
glove -save-file ../%SAVE_FILE% -threads %NUM_THREADS% -input-file ../%COOCCURRENCE_SHUF_FILE% -x-max %X_MAX% -iter %MAX_ITER% -vector-size %VECTOR_SIZE% -binary %BINARY% -vocab-file ../%VOCAB_FILE% -verbose %VERBOSE%

