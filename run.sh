#!/bin/bash

sources=(3 9 11 12 13 14)

for test_source in "${sources[@]}"; do
    if [ -d "log/source_$test_source" ] && [ -f "log/source_$test_source/training_log.npy" ]; then
        echo "Skipping source $test_source as training_log.npy already exists."
        continue
    fi
    echo "./train.py --test_source $test_source"
    ./train.py --test_source $test_source
    echo "*"*100
done
