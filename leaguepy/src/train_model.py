import numpy as np
from pathlib import Path

from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier

# For training script https://scikit-learn.org/stable/modules/neural_networks_supervised.html


class TrainerEvaluator:
    def __init__(
        self,
        X: np.ndarray,
        y: np.ndarray,
        split: float = 0.2,
        size: int = 10,
        layers: int = 2,
    ) -> None:
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=split
        )
        self.split = split
        self.model = MLPClassifier(
            solver="lbfgs",
            alpha=1e-5,
            hidden_layer_sizes=(size, layers),
            random_state=1,
        )

    def train(self):
        self.model.fit(self.X_train, self.y_train)

    def evaluate(self):
        y_pred = self.model.predict(self.X_test)
        return accuracy_score(self.y_test, y_pred)

    # Would be interesting to explore - What snapshot the game was not wrong after


def main():
    pass
    # dir = Path(
    #     r"C:\Users\jonhuster\Desktop\General\Personal\Projects\Python\LeaguePredictor\data\matches"
    # )
    # matches = list(dir.glob("KR_*.pqt"))
    # X, y = format_data.make_outcome_data(matches, dir)

    # clf = MLPClassifier(
    #     solver="lbfgs", alpha=1e-5, hidden_layer_sizes=(22, 2), random_state=1
    # )
    # clf.fit(X[:-1], y[:-1])
    # clf.predict(X[-1], y[-1])


if __name__ == "__main__":
    main()
