#!/bin/bash
L=128
kernels=(
    dirac_delta
    elongated_gaussian
    gaussian
    harmonic_gaussian
    squashed_gaussian
)
maps=(
    earth
    wmap
)
routines=(
    rotate
    translate
)
types=(
    abs
    imag
    real
)

echo "maps"
for t in ${types[*]}; do
    echo 'earth', $t;
    plotting earth -L $L -t $t;
done
for t in ${types[*]}; do
    echo 'wmap', $t;
    plotting wmap -L $L -t $t;
done

echo "convolutions"
for k in ${kernels[*]}; do
    for m in ${maps[*]}; do
        for t in ${types[*]}; do
            echo $k, $m, $t;
            plotting $k -L $L -c $m -t $t;
        done;
    done;
done

echo "north pole"
for k in ${kernels[*]}; do
    for t in ${types[*]}; do
        echo $k, $t;
        plotting $k -L $L -t $t -m north;
    done;
done

echo "rotation/translation"
for k in ${kernels[*]}; do
    for r in ${routines[*]}; do
        for t in ${types[*]}; do
            echo $k, $r, $t;
            plotting $k -L $L -t $t -m $r -o;
        done;
    done;
done

echo "harmonic Gaussian larger kernel"
plotting harmonic_gaussian -L $L -e 2 1 -m translate -o;
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
