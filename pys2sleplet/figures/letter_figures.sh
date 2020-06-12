#!/bin/bash
L=256

echo "figure 1"
for t in real imag abs; do
    echo $t;
    plotting harmonic_gaussian -L $L -m translate -t $t;
    plotting harmonic_gaussian -L $L -e 3 1 -m translate -t $t;
done

echo "figure 2"
plotting earth -L $L -t real

echo "figure 3"
for t in real imag abs; do
    echo $t;
    plotting harmonic_gaussian -c earth -L $L -t $t;
    plotting harmonic_gaussian -c earth -L $L -e 3 1 -t $t;
done
