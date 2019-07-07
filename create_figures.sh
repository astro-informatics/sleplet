#!/bin/bash
# maps
for t in real imag abs; do echo 'earth', $t; ./sifting.py earth -t $t; done
for t in real imag abs; do echo 'wmap', $t; ./sifting.py wmap -t $t; done

# convolutions
for f in dirac_delta gaussian squashed_gaussian elongated_gaussian; do for m in wmap earth; do for t in real imag abs; do echo $f, $m, $t; ./sifting.py $m -c $f -t $t; done; done; done

# north
for f in dirac_delta gaussian squashed_gaussian elongated_gaussian; do for t in real imag abs; do echo $f, $t; ./sifting.py $f -t $t -r north; done; done

# rotation/translation
for f in dirac_delta gaussian squashed_gaussian elongated_gaussian; do for r in rotate translate; do for t in real imag abs; do echo $f, $r, $t; ./sifting.py $f -t $t -r $r; done; done; done

# elongated gaussian
# arrows annotations
for the in $(seq -1 2); do for phi in $(seq -3 -2); do for t in real imag abs; do echo $the $phi $t; ./sifting.py elongated_gaussian -t $t -e $the $phi; ./sifting.py elongated_gaussian -t $t -e $the $phi -r translate; ./sifting.py earth -c elongated_gaussian -t $t -e $the $phi; done; done; done
# no arrow annotations
for the in $(seq -2 2); do for phi in $(seq -1 1); do for t in real imag abs; do echo $the $phi $t; ./sifting.py elongated_gaussian -t $t -e $the $phi -n; ./sifting.py elongated_gaussian -t $t -e $the $phi -r translate -n; ./sifting.py earth -c elongated_gaussian -t $t -e $the $phi -n; done; done; done
# top left corner
for phi in $(seq -3 2); do for t in real imag abs; do echo $the $phi $t; ./sifting.py elongated_gaussian -t $t -e -2 $phi -n; ./sifting.py elongated_gaussian -t $t -e -2 $phi -r translate -n; ./sifting.py earth -c elongated_gaussian -t $t -e -2 $phi -n; done; done

# presentation rotation demo
echo Y_{43}
./sifting.py spherical_harmonic -l 4 -m 3 -t real
echo Y_{43} rot 0 0 0.25
./sifting.py spherical_harmonic -l 4 -m 3 -t real -r rotate -a 0 -b 0 -g 0.25
echo Y_{43} rot 0 0.25 0.25
./sifting.py spherical_harmonic -l 4 -m 3 -t real -r rotate -a 0 -b 0.25 -g 0.25
echo Y_{43} rot 0.25 0.25 0.25
./sifting.py spherical_harmonic -l 4 -m 3 -t real -r rotate -a 0.25 -b 0.25 -g 0.25
