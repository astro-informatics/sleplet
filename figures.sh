#!/bin/bash
# maps
for t in real imag abs; do echo 'earth', $t; ./plotting.py earth -t $t; done
for t in real imag abs; do echo 'wmap', $t; ./plotting.py wmap -t $t; done

# convolutions
for f in dirac_delta elongated_gaussian gaussian harmonic_gaussian squashed_gaussian; do for m in wmap earth; do for t in real imag abs; do echo $f, $m, $t; ./plotting.py $m -c $f -t $t; done; done; done

# north
for f in dirac_delta elongated_gaussian gaussian harmonic_gaussian squashed_gaussian; do for t in real imag abs; do echo $f, $t; ./plotting.py $f -t $t -r north; done; done

# rotation/translation
for f in dirac_delta elongated_gaussian gaussian harmonic_gaussian squashed_gaussian; do for r in rotate translate; do for t in real imag abs; do echo $f, $r, $t; ./plotting.py $f -t $t -r $r; done; done; done

# presentation rotation demo
echo Y_{43}
./plotting.py spherical_harmonic -e 4 3 -t real
echo Y_{43} rot 0 0 0.25
./plotting.py spherical_harmonic -e 4 3 -t real -r rotate -a 0 -b 0 -g 0.25
echo Y_{43} rot 0 0.25 0.25
./plotting.py spherical_harmonic -e 4 3 -t real -r rotate -a 0 -b 0.25 -g 0.25
echo Y_{43} rot 0.25 0.25 0.25
./plotting.py spherical_harmonic -e 4 3 -t real -r rotate -a 0.25 -b 0.25 -g 0.25
