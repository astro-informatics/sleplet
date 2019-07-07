#!/bin/bash
# figure 1
echo "figure 1"
for t in real imag abs; do echo $t; ./plot_sifting.py elongated_gaussian -t $t -r translate; ./plot_sifting.py elongated_gaussian -t $t -e 2 -1 -r translate -n; done

# figure 2
echo "figure 2"
./plot_sifting.py earth -t real

# figure 3
echo "figure 3"
for t in real imag abs; do echo $t; ./plot_sifting.py earth -c elongated_gaussian -t $t; ./plot_sifting.py earth -c elongated_gaussian -t $t -e 2 -1; done
