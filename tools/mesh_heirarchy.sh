#!/bin/sh
alias tetwild="/Users/zachary/Development/libraries/fTetWild/build/FloatTetwild_bin"

filename=$(basename -- "$1")
extension="${filename##*.}"
filename="${filename%.*}"

for i in $(seq 0 4); do
    tetwild -i "${1}" -o "${filename}-level${i}.msh" -l $(python -c "print(0.1 / 2**${i})") -e $(python -c "print(0.01 / 2**${i})") --no-color
done
