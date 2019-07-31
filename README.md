# Fixed-Filter-Bank-Neural-Networks
Adjust the nonlinear functions rather than the filter bank.

In a conventional artificial neural network a fixed nonlinear function is applied to the elements of a vector and then multiple filters (ie. weighed sums) are adjusted to produce the wanted responses, per layer. For a fully connected layer this takes n squared fused multiply accumulates.  A hefty computational burden and there are viewpoints you take that suggest this is a very inefficient use of weight parameters.

An alternative:
You can use the fast Walsh Hadamard transform or the FFT as a fixed filter bank that requires only n.log_base_2(n) operations but then you must make the nonlinear functions adjustable to produce the wanted responses.  One suitable parameterized nonlinear function with nice properties is a switch slope at zero function.  f(x)=a.x x>=0, f(x)=b.x x<0.

