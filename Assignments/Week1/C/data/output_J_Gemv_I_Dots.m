% number of repeats:% 3
% enter first, last, inc:% 48 480 48 
data = [
%  n          reference      |         current implementation 
%        time       GFLOPS   |    time       GFLOPS     diff 
   480 1.8508e-02 1.1951e+01    1.3451e-01 1.6444e+00 3.4106e-13
   432 1.4263e-02 1.1305e+01    9.1499e-02 1.7622e+00 2.9843e-13
   384 9.3473e-03 1.2115e+01    8.3398e-02 1.3579e+00 2.2737e-13
   336 6.0188e-03 1.2605e+01    4.3307e-02 1.7518e+00 1.7053e-13
   288 3.7786e-03 1.2644e+01    2.6050e-02 1.8340e+00 1.1369e-13
   240 2.1473e-03 1.2876e+01    1.5497e-02 1.7841e+00 4.2633e-14
   192 1.0931e-03 1.2950e+01    7.4290e-03 1.9055e+00 2.8422e-14
   144 4.5909e-04 1.3008e+01    2.8123e-03 2.1235e+00 2.8422e-14
    96 1.7753e-04 9.9669e+00    7.6125e-04 2.3244e+00 1.0658e-14
    48 2.3344e-05 9.4750e+00    8.1759e-05 2.7053e+00 7.1054e-15
];

% Maximum difference between reference and your implementation: 3.410605e-13.
