Overview
========
A collection of utilities for analysis of investment funds' historical data and
choosing an optimal portfolio.

There's a planned blog post for [my blog site](https://gregorias.github.io) that
will describe this project in more detail.

Running
=======

Download `mstfun` data from
[http://bossa.pl/notowania/metastock/](http://bossa.pl/notowania/metastock/) and
put it into `data/mstfun` directory.

Download requirements from `requirements.txt`:

    pip install -r requirements.txt

, or you may use `init_dev_env.sh` to initialize a development environment by
creating a virtual environment in `pyvenv/`

Run `main.py`:

    ./main.py

A successful run of the script should display a collection of plots showcasing
optimal portfolios.

Sources
=======
* [quadratic optimization book](http://web.stanford.edu/~boyd/cvxbook/bv_cvxbook.pdf)
* [cvxopt's qopt documentation](http://cvxopt.org/userguide/coneprog.html#quadratic-programming)
