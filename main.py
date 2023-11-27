import hydra

import experiment
from experiment import Exp


@hydra.main(config_path="conf", config_name="main", version_base="1.2")
def main(config):
    exp: Exp = Exp(config)
    exp.run()


if __name__ == "__main__":
    main()
