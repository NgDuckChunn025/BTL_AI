"""Command-line entry point; notebooks and app share src.experiments."""
import argparse
import pandas as pd
from src.data_engine import PROFILE_FILE, COURSE_FILE
from src.experiments import train, scores

def main():
    parser = argparse.ArgumentParser(description="Five models + Dummy; grouped CV; validation F2 threshold.")
    parser.add_argument("--folds", type=int, default=5, choices=range(2, 11))
    parser.add_argument("--jobs", type=int, default=1)
    args = parser.parse_args()
    result = train(pd.read_csv(PROFILE_FILE), pd.read_csv(COURSE_FILE), folds_count=args.folds, jobs=args.jobs)
    print(result[["Model", "CV_AP_mean", "Threshold", "Recall_NguyCo", "Precision_NguyCo", "F2_NguyCo", "Selected"]].round(4).to_string(index=False))

if __name__ == "__main__":
    main()
