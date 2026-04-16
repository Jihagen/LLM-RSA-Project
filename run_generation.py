from configs import DEFAULT_HPC_PIPELINE
from pipeline.homonym_pipeline import generate_or_load_dataset


def main():
    generate_or_load_dataset(DEFAULT_HPC_PIPELINE)


if __name__ == "__main__":
    main()
