#!/bin/bash

# This script is used to build a mini dataset from a larger dataset.
# The mini dataset will contain the first n files from each subdirectory of the source directory.

if [ "$#" -ne 3 ] || [ "$1" == "-h" ] || [ "$1" == "--help" ]; then
    echo "Usage: $0 source_directory target_directory n"
    exit 0
fi

source_directory=$1
target_directory=$2
n=$3

for dir in $source_directory/*/; do
    subdir=$(basename "$dir")
    target_subdir="$target_directory/$subdir"
    mkdir -p "$target_subdir"
    find "$dir" -type f | head -n $n | xargs -I {} cp {} "$target_subdir/"
done