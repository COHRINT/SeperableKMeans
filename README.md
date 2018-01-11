# SeperableKMeans
Exploring alternative metrics with which to cluster gaussians using K-means and Runnals method

## Data Storage

Data from experiments run via **testArena.py** are stored in a SQLite database file (whose filename is specified in **testArena.py**). Each row in a created database contains the following parameters (and database column names, sql datatype):
- Dimension of mixtures ('dim', integer)
- Starting number of mixands ('start_num', integer)
- Distance metric used for clustering via K-means ('dist', text)
- The number of clusters ('mid_num', integer)
- The final number of mixands per cluster ('fin_num', integer)
- The 'run number' or number of times a parameter set has been run ('run_num', integer)
- The ISD between the test mixture and the condensed result mixture ('isd', real)
- The time required to run the condensation ('time', real)
- The means, variances, and weights of the test mixture ('test_mix', blob)
- The means, variances, and weights of the result mixture ('result_mix', blob)
