#!/bin/bash
echo "figure 1"
for t in real imag abs; do
    echo $t;
    plotting harmonic_gaussian -m translate -t $t;
    plotting harmonic_gaussian -e 3 1 -m translate -t $t;
done

echo "figure 2"
plotting earth -t real

echo "figure 3"
for t in real imag abs; do
    echo $t;
    plotting harmonic_gaussian -c earth -t $t;
    plotting harmonic_gaussian -c earth -e 3 1 -t $t;
done
