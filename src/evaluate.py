from sklearn.metrics import accuracy_score, confusion_matrix

class Evaluator:
    @staticmethod
    def evaluate(y_true, y_pred):
        accuracy = accuracy_score(y_true, y_pred)
        cm = confusion_matrix(y_true, y_pred)
        return accuracy, cm
