#!/bin/bash
# maps
echo 'earth'
./plotting.py earth
echo 'wmap'
./plotting.py wmap

# convolutions
for f in dirac_delta gaussian squashed_gaussian elongated_gaussian; do for m in wmap earth; do echo $f, $m; ./plotting.py $f -c $m; done; done

# north
for f in dirac_delta gaussian squashed_gaussian elongated_gaussian; do for t in real imag abs; do echo $f, $t; ./plotting.py $f -t $t -r north; done; done

# rotation/translation
for f in dirac_delta gaussian squashed_gaussian elongated_gaussian; do for r in rotate translate; do for t in real imag abs; do echo $f, $r, $t; ./plotting.py $f -t $t -r $r; done; done; done

# spherical harmonics
for l in {0..3}; do for m in $(seq -$l $l); do for t in real imag abs sum; do echo $l, $m, $t; ./plotting.py spherical_harmonic -l $l -m $m -t $t; done; done; done

# elongated gaussian
for t in $(seq -2 2); do for p in $(seq -2 2); do ./plotting.py elongated_gaussian -e $t $p; ./plotting.py elongated_gaussian -e $t $p -r translate; ./plotting.py elongated_gaussian -e $t $p -c earth; echo $t, $p; done; done

# presentation rotation demo
echo Y_{43}
./plotting.py spherical_harmonic -l 4 -m 3
echo Y_{43} rot 0 0 0.25
./plotting.py spherical_harmonic -l 4 -m 3 -r rotate -a 0 -b 0 -g 0.25
echo Y_{43} rot 0 0.25 0.25
./plotting.py spherical_harmonic -l 4 -m 3 -r rotate -a 0 -b 0.25 -g 0.25
echo Y_{43} rot 0.25 0.25 0.25
./plotting.py spherical_harmonic -l 4 -m 3 -r rotate -a 0.25 -b 0.25 -g 0.25
