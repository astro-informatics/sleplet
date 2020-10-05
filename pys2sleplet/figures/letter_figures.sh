#!/bin/bash
L=128
types=(
    abs
    imag
    real
)

echo "figure 1"
for t in ${types[*]}; do
    echo $t;
    plotting harmonic_gaussian -L $L -m translate -t $t;
    plotting harmonic_gaussian -L $L -e 2 1 -m translate -t $t;
done

echo "figure 2"
plotting earth -L $L -t real

echo "figure 3"
plotting harmonic_gaussian -c earth -L $L;
plotting harmonic_gaussian -c earth -L $L -e 2 1;
