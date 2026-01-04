# Configuration for Benchmarks

# Dimensions to test
LIST_DIMS = [20]

# Condition numbers
LIST_CONDNUM = [1.0, 1e4, 1e8]

# Noise parameters (Unified: serves as bound for Uniform or sigma for Gaussian)
# Original: [1.0, 1e-3, 0.0]
LIST_NOISE_PARAM = [1.0, 1e-3, 0.0]

# Noise types to sweep (OPTIONS: "uniform", "gaussian")
LIST_NOISE_TYPE = ["uniform"]

# Problems (OPTIONS: "least-squares", "lasso", "l1-log-reg", "l2-log-reg", "log-sum-exp")
LIST_PROBLEM = [
    'least-squares', 
    'lasso', 
    'l1-log-reg', 
    'l2-log-reg', 
    'log-sum-exp'
]

# Gradient estimators (OPTIONS: "ffd", "cfd", "gsg", "cgsg", "nmxfd", "sage")
LIST_GRAD_EST = [
    "ffd", "cfd", "gsg", "cgsg", "nmxfd"
]

# Benchmark Settings
BMK_MAXTRIALS = 100
MAX_EVALS_MULT = 50 # Multiplied by D
