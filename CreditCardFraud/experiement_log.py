"""
Experiment Log

"""

def clear_log(log_path="experiment_log.txt"):
    with open(log_path, "w"): pass

def experiment_log(experiment_num, epochs, learning_rate, metrics, log_path="experiment_log.txt"):
    divder = "-" * 50
    entry = (
        f"{divder}\n"
        f"Experiment Num: {experiment_num}\n"
        f"{divder}\n"
        f"Parameters Chosen: \n"
        f"  Neural Network: \n"
        f"      Layers: 2\n"
        f"      Neurons: (30, 64, 1)\n"
        f"      Epochs: {epochs}\n"
        f"      Learning Rate: {learning_rate}\n"
        f"{divder}\n"
        f"Results:"
        f"  Accuracy: {metrics['accuracy']:.4f}\n"
        f"  Preciission: {metrics['precision']:.4f}\n"
        f"  Recall: {metrics['recall']:.4f}\n"
        f"  F1 Score: {metrics['F1']:.4f}\n"
        f"{divder}\n"
    )
    with open(log_path, "a") as f:
        f.write(entry)

