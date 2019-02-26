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
for f in dirac_delta gaussian squashed_gaussian elongated_gaussian; do for r in rotate translate; do for t in real imag abs; do echo $f, $r, $t; ./plotting.py $f -t $t -r $r -a 0.75 -b 0.25; done; done; done

# spherical harmonics
for l in {0..3}; do for m in $(seq -$l $l); do for t in real imag abs sum; do echo $l, $m, $t; ./plotting.py spherical_harmonic -l $l -m $m -t $t; done; done; done

# presentation rotation demo
echo 'remember to create Y_{43} and rotate (alpha,beta,gamma)='
echo '(0,0,pi/4)'
echo '(0,pi/4,pi/4)'
echo '(pi/4,pi/4,pi/4)'
