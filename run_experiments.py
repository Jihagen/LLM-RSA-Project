from configs import DEFAULT_HPC_PIPELINE
from pipeline import run_wic_validation_suite


def main():
    run_wic_validation_suite(DEFAULT_HPC_PIPELINE)


if __name__ == "__main__":
    main()
