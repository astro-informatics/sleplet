#!/bin/bash
L=128

echo "figure 1"
plotting harmonic_gaussian -L $L -m translate -o;
plotting harmonic_gaussian -L $L -e 2 1 -m translate -o;

echo "figure 2"
plotting earth -L $L -t real

echo "figure 3"
plotting harmonic_gaussian -c earth -L $L;
plotting harmonic_gaussian -c earth -L $L -e 2 1;
