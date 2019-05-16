#!/bin/bash
# figure 1
for t in real imag abs; do echo $t; ./plotting.py elongated_gaussian -t $t; ./plotting.py elongated_gaussian -t $t -e 2 -1; done

# figure 2
./plotting.py earth -t real

# figure 3
for t in real imag abs; do echo $t; ./plotting.py earth -c elongated_gaussian -t $t; ./plotting.py earth -c elongated_gaussian -t $t -e 2 -1; done
