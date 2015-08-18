# neural_network
Implementation of a Feed Forward Neural Network.

## Description
This code was developed as part of the Google Summer of Code 2015[1] for the
International Institute of Geogebra[2]. It is part of a bigger project which
you can find at https://github.com/gouzouni625/handwritten_equations_recognizer .

## Components
Besides the implementation of a Feed Forward Neural Network, in this
repository you can also find an Image Distorter and a Trainer. The Image
Distorter can be used to apply random affine transformation on images using
OpenCV library. That is a technique used often to virtually increase the size
of a training set when using it to train a machine learning algorithm like
a Feed Forward Neural Network. The Trainer provided is a guide line on how
to use this Feed Forward Neural Network.

## Building the Documentation
The documentation is written using Doxygen[3]. To generate the documentation
pages, you have to install Doxygen and download the code to your local file
system. After that, do:

```
cd neural_network/doxygen
doxygen Doxyfile
```

. This will create two directories inside
`neural_network/doxygen`, `html` and `latex`. To view the documentation pages,
open  `html/index.html` with a browser of your choice.

## External Libraries
The external libraries used by this project are:

1. OpenCV for image processing. For more information, please visit
   http://opencv.org/

2. JUnit for testing. For more information, please visit
   http://junit.org/

Each of these external libraries is subject to its own license.

## Datasets
During the developing of the code, the MNIST dataset was used extensively for
testing. For more information, please visit
http://yann.lecun.com/exdb/mnist/

## References

[1] https://www.google-melange.com/gsoc/homepage/google/gsoc2015

[2] http://wiki.geogebra.org/en/Comments:International_GeoGebra_Institute

[3] http://www.doxygen.org
