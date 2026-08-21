# stochastic-methods-lab

Three coursework projects from Métodos Estocásticos, refactored from
exploratory notebooks into a tested, reusable Python package. Each
module keeps the original statistical method but generalizes the code
so it isn't tied to one specific dataset.

## Modules

### `src/poisson_process/nhpp.py`
Fits and simulates Poisson / Non-Homogeneous Poisson processes:
estimate a rate and its dispersion, test counts against a Poisson
distribution with a chi-square goodness-of-fit test, fit inter-event
times to an exponential distribution with a Kolmogorov-Smirnov test,
compute wait-time probabilities, estimate hour-of-day rates, test
whether the rate is homogeneous across hours, and simulate an NHPP by
thinning. Originally applied to network failure events; the functions
here take plain timestamp arrays so they apply to any event stream.

### `src/bst/tree.py`
A binary search tree (`BinarySearchTree`) storing arbitrary `(key,
record)` pairs: insert, search, delete (all three cases — leaf,
one child, two children via in-order successor), in/pre/post-order
traversal, count, and height. Originally built for a student registry
keyed on `matricula`; generalized to any numeric key.

### `src/queueing/mm1.py`
Simulates a single-server FIFO queue from arrival and service times,
estimates lambda/mu/rho from data, computes closed-form M/M/1 metrics
(L, Lq, W, Wq), verifies Little's Law against the simulation, and
compares multiple `(lambda, mu)` scenarios side by side. Originally
used to analyze a customer service queue's stability and bottlenecks.

## Notebooks

The `notebooks/` folder has the original exploratory analysis with real
data, plots, and interpretation — the source material these modules
were extracted from. The modules are the tested, reusable version; the
notebooks are the analysis and write-up.

## Running the tests

```bash
pip install -r requirements.txt
pytest -v
```

24 tests across the three modules, covering known closed-form results
(e.g., M/M/1 metrics against hand-calculated values), statistical
recovery on synthetic data (e.g., recovering a known Poisson rate or
exponential rate from simulated samples), and structural correctness
(BST in-order traversal stays sorted after every insert/delete case).

## Why this project

Binary search trees, Poisson processes, and queueing theory look like
three unrelated topics until you notice they're all "events happening
over time, need to reason about them statistically" — which is also
the core of a lot of quantitative finance (order arrivals, execution
queues, event-driven risk). See the `quant-backtester` repo for where
the Poisson and queueing ideas get applied directly to trading.
