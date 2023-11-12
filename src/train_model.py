from pathlib import Path

from sklearn.neural_network import MLPClassifier

import format_data

def main():
    dir = Path(r"C:\Users\jonhuster\Desktop\General\Personal\Projects\Python\LeaguePredictor\data\matches")
    matches = list(dir.glob("KR_*.pqt"))
    X, y = format_data.make_outcome_data(matches, dir)
    
    clf = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(22, 2), random_state=1)
    clf.fit(X[:-1], y[:-1])
    clf.predict(X[-1], y[-1])
    
main()
