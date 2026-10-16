"""Study constants. Values mirror PROTOCOL.md; change them only via its deviation log."""

import os

N_HOURS = 742          # hours 0..741 present in the Criteo attribution dataset
TRAIN_END = 504        # train hours [0, 504)   = days 0-20
VAL_END = 600          # val hours   [504, 600) = days 21-24
TEST_END = 720         # test hours  [600, 720) = days 25-29; hours 720-741 discarded

LOOKBACK = 168
HORIZON = 24

# 0.1/0.9 give the 80% interval; 0.5/0.7/0.9 are the hedging levels used by the policy
QUANTILES = (0.1, 0.5, 0.7, 0.9)
HEDGE_QUANTILES = (0.5, 0.7, 0.9)

RHOS = (0.3, 0.5, 0.7)          # daily budget = rho * mean daily training spend
MIN_ACTIVE_FRAC = 0.9           # campaign kept if spend > 0 in >= 90% of training hours
EARLY_DEPLETION_HOUR = 20       # exhausted before hour 20 of the day counts as early
DEPLETION_TOL = 0.99            # budget counts as exhausted at 99% spent
N_REPLAY_SEEDS = 10
N_NN_SEEDS = 3
N_BOOTSTRAP = 2000

PATHS = {
    "raw": "~/datasets/criteo_attribution/criteo_attribution_dataset.tsv.gz",
    "work": "~/pacing_v2/work",        # derived data
    # everything the paper reads; PACING_RESULTS redirects smoke tests elsewhere
    "results": os.environ.get("PACING_RESULTS", "~/pacing_v2/results"),
}

# smoke tests only: limit the number of campaigns simulated (0 = all)
MAX_CAMPAIGNS = int(os.environ.get("PACING_MAX_CAMPAIGNS", "0"))
