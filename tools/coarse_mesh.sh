#!/bin/sh
alias tetwild="/Users/zachary/Development/libraries/fTetWild/build/FloatTetwild_bin"

filename=$(basename -- "$1")
extension="${filename##*.}"
filename="${filename%.*}"

# tetwild -i "${1}" -o "${filename}.msh" -l 0.1 -e 0.003 --no-color
tetwild -i "${1}" -o "${filename}.msh" -l 0.1 -e 0.01 --no-color
