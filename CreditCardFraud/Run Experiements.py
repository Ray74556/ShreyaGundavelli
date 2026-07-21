"""
Run experiments
"""

import gc
from fraud_model import load_dataset, NeuralNetwork
from experiement_log import clear_log, experiment_log

EXPERIMENTS = [
    {"epochs": 100, "learning_rate": 0.001},
    {"epochs": 500, "learning_rate": 0.005},
    {"epochs": 750, "learning_rate": 0.009},
]

def main():
    clear_log()
    df = load_dataset()
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
            {"label": f"Exp {i} \n (e={params['epochs']}, lr={params['learning_rate']})" })

        del model
        gc.collect()

if __name__ == "__main__":
    main()