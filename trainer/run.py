from .work import Worker
import yaml
import os

def main():
    # Parse the arguments and load configuration
    config_path = os.path.join(os.path.abspath(os.path.dirname(__file__)), '..', 'config', 'config.yml')
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)

    worker = Worker(config)
    # worker.train()
    # worker.train_multi_model()
    worker.train_generation()
    # worker.test()


if __name__ == "__main__":
    main()