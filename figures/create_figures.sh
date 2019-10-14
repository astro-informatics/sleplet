#!/bin/bash
dir="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# maps
for t in real imag abs; do
    echo 'earth', $t;
    $dir/../plotting.py earth -t $t;
done
for t in real imag abs; do
    echo 'wmap', $t;
    $dir/../plotting.py wmap -t $t;
done

# convolutions
for f in dirac_delta elongated_gaussian gaussian harmonic_gaussian squashed_gaussian; do
    for m in wmap earth; do
        for t in real imag abs; do
            echo $f, $m, $t;
            $dir/../plotting.py $m -c $f -t $t;
        done;
    done;
done

# north
for f in dirac_delta elongated_gaussian gaussian harmonic_gaussian squashed_gaussian; do
    for t in real imag abs; do
        echo $f, $t;
        $dir/../plotting.py $f -t $t -r north;
    done;
done

# rotation/translation
for f in dirac_delta elongated_gaussian gaussian harmonic_gaussian squashed_gaussian; do
    for r in rotate translate; do
        for t in real imag abs; do
            echo $f, $r, $t;
            $dir/../plotting.py $f -t $t -r $r;
        done;
    done;
done

# harmonic Gaussian larger kernel
echo harmonic
for t in real imag abs; do
    $dir/../plotting.py harmonic_gaussian -e 3 1 -t $t;
    $dir/../plotting.py earth -c harmonic_gaussian -e 3 1 -t $t;
done

# presentation rotation demo
echo Y_{43}
$dir/../plotting.py spherical_harmonic -e 4 3 -t real
echo Y_{43} rot 0 0 0.25
$dir/../plotting.py spherical_harmonic -e 4 3 -t real -r rotate -a 0 -b 0 -g 0.25
echo Y_{43} rot 0 0.25 0.25
$dir/../plotting.py spherical_harmonic -e 4 3 -t real -r rotate -a 0 -b 0.25 -g 0.25
echo Y_{43} rot 0.25 0.25 0.25
$dir/../plotting.py spherical_harmonic -e 4 3 -t real -r rotate -a 0.25 -b 0.25 -g 0.25
