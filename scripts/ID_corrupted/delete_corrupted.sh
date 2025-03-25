#!/bin/bash

CORRUPTED_FILES_PATH="$1"
while IFS= read -r file; do
    rm -f "$file"
done < $CORRUPTED_FILES_PATH
