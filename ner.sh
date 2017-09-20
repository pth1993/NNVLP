#!/usr/bin/env bash
dir="embedding"
if [ ! -d "$dir" ]; then
    mkdir "embedding"
fi
file="embedding/vectors.npy"
if [ -f "$file" ]; then
	echo "$file found."
else
	url="https://drive.google.com/uc?export=download&id=0BytHkPDTyLo9WU93NEI1bGhmYmc"
    wget --load-cookie cookie.txt --save-cookie cookie.txt "${url}" -O tmp
    c=`grep -o "confirm=...." tmp`
    wget --load-cookie cookie.txt --save-cookie cookie.txt "${url}&$c" -O "${file}"
    rm cookie.txt tmp
fi
file="embedding/words.pl"
if [ -f "$file" ]; then
    echo "$file found."
else
    url="https://drive.google.com/uc?export=download&id=0BytHkPDTyLo9SC1mRXpkbWhfUDA"
    wget --load-cookie cookie.txt --save-cookie cookie.txt "${url}" -O tmp
    c=`grep -o "confirm=...." tmp`
    wget --load-cookie cookie.txt --save-cookie cookie.txt "${url}&$c" -O "${file}"
    rm cookie.txt tmp
fi
file="embedding/unknown.npy"
if [ -f "$file" ]; then
    echo "$file found."
else
    url="https://drive.google.com/uc?export=download&id=0BytHkPDTyLo9VVlld1VlVVVoSHM"
    wget --load-cookie cookie.txt --save-cookie cookie.txt "${url}" -O tmp
    c=`grep -o "confirm=...." tmp`
    wget --load-cookie cookie.txt --save-cookie cookie.txt "${url}&$c" -O "${file}"
    rm cookie.txt tmp
fi
python ner.py --train_dir "data/ner/ner_train.txt" --dev_dir "data/ner/ner_dev.txt" --test_dir "data/ner/ner_test.txt" --word_dir "embedding/vectors.npy" --vector_dir "embedding/words.pl" --char_embedd_dim 30 --num_units 300 --num_filters 30 --dropout --grad_clipping 5.0 --peepholes --batch_size 10 --learning_rate 0.01 --decay_rate 0.05 --patience 5