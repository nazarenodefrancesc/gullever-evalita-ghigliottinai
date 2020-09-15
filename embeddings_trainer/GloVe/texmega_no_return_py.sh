#!/bin/bash

make

# Makes programs, downloads sample data, trains a GloVe model, and then evaluates it.
# One optional argument can specify the language used for eval script: matlab, octave or [default] python


CORPUS=texmega_corpus_no_return
VOCAB_FILE=texmega_vocab.txt
COOCCURRENCE_FILE=texmega_cooccurrence.bin
COOCCURRENCE_SHUF_FILE=texmega_cooccurrence.shuf.bin
BUILDDIR=build
SAVE_FILE=texmega_corpus_vectors
VERBOSE=2
ETA=0.05
ALPHA=0.75
MEMORY=13.0
VECTOR_SIZE=300
VOCAB_MIN_COUNT=200
MAX_ITER=100
WINDOW_SIZE=10
X_MAX=10
BINARY=2
NUM_THREADS=4



$BUILDDIR/vocab_count -min-count $VOCAB_MIN_COUNT -verbose $VERBOSE < $CORPUS > $VOCAB_FILE
$BUILDDIR/cooccur -memory $MEMORY -vocab-file $VOCAB_FILE -verbose $VERBOSE -window-size $WINDOW_SIZE < $CORPUS > $COOCCURRENCE_FILE
$BUILDDIR/shuffle -memory $MEMORY -verbose $VERBOSE < $COOCCURRENCE_FILE > $COOCCURRENCE_SHUF_FILE
$BUILDDIR/glove -alpha $ALPHA -eta $ETA -save-file $SAVE_FILE -threads $NUM_THREADS -input-file $COOCCURRENCE_SHUF_FILE -x-max $X_MAX -iter $MAX_ITER -vector-size $VECTOR_SIZE -binary $BINARY -vocab-file $VOCAB_FILE -verbose $VERBOSE

