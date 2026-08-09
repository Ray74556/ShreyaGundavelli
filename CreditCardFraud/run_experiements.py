"""
Run experiments
"""

import gc
from fraud_model import load_dataset, NeuralNetwork
from experiement_log import clear_log, experiment_log
from visualize import (
    plot_arch,
    plot_class_dist,
    plot_metric_comparison,
    plot_confusion_matrix,
)

EXPERIMENTS = [
    {"epochs": 100, "learning_rate": 0.001},
    {"epochs": 500, "learning_rate": 0.005},
    {"epochs": 750, "learning_rate": 0.009},
]

def main():
    clear_log()
    df = load_dataset()
    plot_class_dist(df)
    results = []

    for i, params in enumerate(EXPERIMENTS, start=1):
        model = NeuralNetwork(df.copy())
        model.preprocess()
        model.train_test_split()
        model.check_split()
        model.initialize_weights()
        model.train(epochs=params["epochs"], learning_rate=params["learning_rate"],
                    batch_size=256, X_val=model.X_test, y_val=model.y_test)

        metrics = model.predict_and_eval(model.X_test, model.y_test, threshold=0.3)
        experiment_log(i, params["epochs"], params["learning_rate"], metrics)

        results.append(
            {"label": f"Exp {i} \n (e={params['epochs']}, lr={params['learning_rate']})", **metrics}
        )

        del model
        gc.collect()

    plot_arch()
    plot_metric_comparison(results)

    best = max(results, key=lambda r: r["F1"])
    plot_confusion_matrix(best["CM"], title=f"Confusion Matrix - Best Experiment (F1={best['F1']:.3f})")

    print("\nDone. Figures saved to project root, results logged to experiment_log.txt")

if __name__ == "__main__":
    main()