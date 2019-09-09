# Mixed Strategies

This Rust codebase contains a library crate,
`mixed-strategies`, for computing optimal mixed strategies
and game values given the payoff matrix for a classical
two-player zero-sum game.

A simple driver program is also provided that reads a payoff
matrix (as ASCII text) from standard input and displays the
game solution. The `examples/` directory in this
distribution contains a number of example payoff matrices.

The code in this distribution is an implementation of
algorithms described in the most excellent book

> *The Compleat Strategyst*  
> John D. Williams  
> RAND Commercial Books 1954  
> https://www.rand.org/pubs/commercial_books/CB113-1.html

It's fantastic that The RAND Corporation has chosen to
provide a free PDF of this book for download: my paper copy
is getting pretty dog-eared.  I highly recommend this book
as an introduction to game theory. All errors and bugs in
the implementation are, of course, my own.

Please see the rustdoc for usage of the library crate.

# Author

Bart Massey \<bart.massey@gmail.com\>

# License

This program is licensed under the "MIT License".  Please
see the file LICENSE in the source distribution of this
software for license terms.
