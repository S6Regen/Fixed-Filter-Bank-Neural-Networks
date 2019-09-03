# Fixed-Filter-Bank-Neural-Networks
Adjust the nonlinear functions rather than the filter bank.

In a conventional artificial neural network a fixed nonlinear function is applied to the elements of a vector and then multiple filters (ie. weighed sums) are adjusted to produce the wanted responses, per layer. For a fully connected layer this takes n squared fused multiply accumulates.  A hefty computational burden and there are viewpoints you can take that suggest this is a very inefficient use of weight parameters.

An alternative:
You can use the fast Walsh Hadamard transform (WHT) or the FFT as a fixed filter bank that requires only n.log_base_2(n) operations but then you must make the nonlinear functions individually adjustable to produce the wanted responses.  One suitable parameterized nonlinear function with nice properties is a switch slope at zero function.  f(x)=a.x x>=0, f(x)=b.x x<0.

I presume you could use back-propagation (as the WHT is self-inverse) to update the function parameters.
However I used an evolutionary algorithm for training: www.cs.bham.ac.uk/~jer/papers/ctsgray.pdf

Condensed information about the WHT: https://github.com/FALCONN-LIB/FFHT/issues/26

Book and library containing some other transforms: https://www.jjj.de/fxt/

The WHT and the central limit theorem: https://archive.org/details/bitsavers_mitreESDTe69266ANewMethodofGeneratingGaussianRando_2706065
