#!/bin/bash

files="word_embeddings_tf.ipynb
machine_translation_and_attention_tf.ipynb
memory_networks_tf.ipynb
memn2n/memn2n_skeleton.py
"

if [ ! -f notebooks.pdf ]; then
    echo "Please follow the instructions in Piazza post @301 to export and concatenate all three notebooks as a single PDF."
    echo "Please name this PDF 'notebooks.pdf', and ensure that the component notebooks are concatenated in order (i.e., word embedding, MT, then MemNets)."
    exit 0
fi
for file in $files
do
    if [ ! -f $file ]; then
        echo "Required file $file not found."
        exit 0
    fi
done

rm -f assignment3.zip
zip -r assignment3.zip . -x "*.git" "images/*" "*.ipynb_checkpoints*" "*README.md" "*collect_submission.sh" "*requirements.txt" ".env/*" "*.pyc"
exit 1
