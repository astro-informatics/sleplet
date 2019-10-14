#!/bin/bash
dir="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# figure 1
echo "figure 1"
for t in real imag abs; do
    echo $t;
    $dir/../plotting.py harmonic_gaussian -r translate -t $t;
    $dir/../plotting.py harmonic_gaussian -e 3 1 -r translate -t $t;
done

# figure 2
echo "figure 2"
$dir/../plotting.py earth -t real

# figure 3
echo "figure 3"
for t in real imag abs; do
    echo $t;
    $dir/../plotting.py earth -c harmonic_gaussian -t $t;
    $dir/../plotting.py earth -c harmonic_gaussian -e 3 1 -t $t;
done
