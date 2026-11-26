"""Quantum CyberShield — team submission
=======================================

BEFORE YOU SUBMIT, PLEASE READ:

1. Rename this file with the name of your team, e.g.  team_qubit_hunters.py.

2. Submit this code TOGETHER WITH:
   - your predictions for the two test sets, in two separate files written by this code (see point 3):
       submission_hidden_a.csv   predictions for data/test_hidden_a.csv (Hidden A)
       submission_hidden_b.csv   predictions for data/test_hidden_b.csv (Hidden B)
   - a presentation of your approach and your results.
   - Presentation: 3 minutes.
   - Questions and answers with the judges: 2 minutes.
   Put ALL of these files (code, submission_hidden_a.csv, submission_hidden_b.csv, presentation,
   requirements.txt) inside one folder named after your team, e.g.  team_qubit_hunters/,
   and submit that folder.

3. This file must run from start to finish and write both prediction files. Each file has exactly
   two columns:
       sample_id,predicted_label
   and one row for every ID of its own test set, with
   predicted_label = 0 (benign), 1 (loud attack) or 2 (stealth attack).

4. Reproducibility: fix your random seeds, list every package you use in requirements.txt,
   and train only on data/train.csv and data/validation.csv.

5. Environment: work in a virtual environment with the library versions from requirements.txt
   (Python 3.10 or newer), so the judges can rerun your code with the same versions.
   From the participant folder:
       python -m venv .venv
       .venv\\Scripts\\activate          (Windows)
       source .venv/bin/activate        (macOS / Linux)
       python -m pip install -r requirements.txt
   If you need another package, install it in the same environment and add it, with its
   version, to requirements.txt.

"""
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
DATA = HERE / 'data'
SEED = 42
TEST_SETS = {'hidden_a': 'test_hidden_a.csv', 'hidden_b': 'test_hidden_b.csv'}


def main():
    train = pd.read_csv(DATA / 'train.csv')
    validation = pd.read_csv(DATA / 'validation.csv')
    tests = {name: pd.read_csv(DATA / file) for name, file in TEST_SETS.items()}

    # ------------------------------------------------------------------
    # Your approach goes here: preprocessing, quantum / hybrid model,
    # training, and evaluation on the validation set.
    # It must produce one label (0, 1 or 2) per row of each test set.
    # ------------------------------------------------------------------
    predictions = {
        'hidden_a': ...,   # labels for tests['hidden_a'], in the same row order
        'hidden_b': ...,   # labels for tests['hidden_b'], in the same row order
    }

    for name, test in tests.items():
        out = pd.DataFrame({'sample_id': test.sample_id, 'predicted_label': predictions[name]})
        out.to_csv(HERE / f'submission_{name}.csv', index=False)
        print(f'Wrote submission_{name}.csv with {len(out)} rows')


if __name__ == '__main__':
    main()
