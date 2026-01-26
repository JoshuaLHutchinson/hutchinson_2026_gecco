import hydra
from omegaconf import DictConfig


@hydra.main(version_base=None, config_path="configs/", config_name="config")
def main(config: DictConfig) -> None:
    if config.operator.name == "me_iso":
        import main_iso as main
    elif config.operator.name == "me_isoline":
        import main_isoline as main
    elif config.operator.name == "me_isocross":
        import main_isocross as main
    elif config.operator.name == "me_isolinecross":
        import main_isolinecross as main
    else:
        raise NotImplementedError

    main.main(config)


if __name__ == "__main__":
    main()
