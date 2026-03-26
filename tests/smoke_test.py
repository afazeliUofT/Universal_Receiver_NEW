from upair5g.config import load_config
from upair5g.evaluation import evaluate_model
from upair5g.plotting import make_all_plots
from upair5g.training import train_model


def main() -> None:
    cfg = load_config("configs/smoke_test.yaml")
    train_result = train_model(cfg)
    evaluate_model(cfg, checkpoint_path=train_result["checkpoint_path"])
    make_all_plots(cfg)
    print("Smoke test finished.")


if __name__ == "__main__":
    main()
