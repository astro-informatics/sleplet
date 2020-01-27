echo "figure 1"
for t in real imag abs; do
    echo $t;
    plotting.py harmonic_gaussian -r translate -t $t;
    plotting.py harmonic_gaussian -e 3 1 -r translate -t $t;
done

echo "figure 2"
plotting.py earth -t real

echo "figure 3"
for t in real imag abs; do
    echo $t;
    plotting.py earth -c harmonic_gaussian -t $t;
    plotting.py earth -c harmonic_gaussian -e 3 1 -t $t;
done
