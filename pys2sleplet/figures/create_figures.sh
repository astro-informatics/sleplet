#!/bin/bash
L=128

echo "maps"
for t in real imag abs; do
    echo 'earth', $t;
    plotting earth -L $L -t $t;
done
for t in real imag abs; do
    echo 'wmap', $t;
    plotting wmap -L $L -t $t;
done

echo "convolutions"
for f in dirac_delta elongated_gaussian gaussian harmonic_gaussian squashed_gaussian; do
    for m in wmap earth; do
        for t in real imag abs; do
            echo $f, $m, $t;
            plotting $f -L $L -c $m -t $t;
        done;
    done;
done

echo "north pole"
for f in dirac_delta elongated_gaussian gaussian harmonic_gaussian squashed_gaussian; do
    for t in real imag abs; do
        echo $f, $t;
        plotting $f -L $L -t $t -m north;
    done;
done

echo "rotation/translation"
for f in dirac_delta elongated_gaussian gaussian harmonic_gaussian squashed_gaussian; do
    for r in rotate translate; do
        for t in real imag abs; do
            echo $f, $r, $t;
            plotting $f -L $L -t $t -m $r;
        done;
    done;
done

echo "harmonic Gaussian larger kernel"
for t in real imag abs; do
    plotting harmonic_gaussian -L $L -e 2 1 -t $t;
done
plotting harmonic_gaussian -c earth -L $L -e 2 1;

echo "presentation rotation demo"
echo Y_{43}
plotting spherical_harmonic -L $L -e 4 3 -t real
echo Y_{43} rot 0 0 0.25
plotting spherical_harmonic -L $L -e 4 3 -t real -m rotate -a 0 -b 0 -g 0.25
echo Y_{43} rot 0 0.25 0.25
plotting spherical_harmonic -L $L -e 4 3 -t real -m rotate -a 0 -b 0.25 -g 0.25
echo Y_{43} rot 0.25 0.25 0.25
plotting spherical_harmonic -L $L -e 4 3 -t real -m rotate -a 0.25 -b 0.25 -g 0.25
