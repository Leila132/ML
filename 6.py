import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
import numpy as np

class DecisionTree:
    def __init__(self):
        self.root_node = None

    def fit(self, data, target, limit_depth, limit_data_len, limit_entropy):
        self.root_node = self.create_tree(data, target, 0, limit_depth, limit_data_len, limit_entropy)

    def predict(self, vector):
        return self.decision_tree(vector, self.root_node)

    def decision_tree(self, vector, node):
        if not node['is_terminal']:
            if vector[node['coord_ind']] < node['threshold']:
                return self.decision_tree(vector, node['left'])
            else:
                return self.decision_tree(vector, node['right'])
        else: return node['t']

    def create_tree(self, data, target, depth, limit_depth, limit_data_len, limit_entropy):
        if not self.stop_criteria(data, target, depth, limit_depth, limit_data_len, limit_entropy):
            params = self.get_params(data, target)
            left_data, right_data = self.split_data(data, params)
            node = self.create_split_node(params)
            node['left'] = self.create_tree(data[left_data], target[left_data],
                                            depth + 1, limit_depth, limit_data_len, limit_entropy)

            node['right'] = self.create_tree(data[right_data], target[right_data],
                                             depth + 1, limit_depth, limit_data_len, limit_entropy)
        else:
            node = self.create_terminal_node(target)

        return node

    def stop_criteria(self, data, target, depth, limit_depth, limit_data_len, limit_entropy):
        if depth == limit_depth or len(data) < limit_data_len \
                or self.get_entropy(target) < limit_entropy:
            return True

        return False

    def get_params(self, data, target):

        i_best = coord_ind = threshold = 0

        for tau in range(16):
            for x_j in range(data.shape[1]):
                i = self.information_gain(data, target,
                                          data[:, x_j] < tau,
                                          data[:, x_j] >= tau)

                if i > i_best:
                    i_best, coord_ind, threshold = i, x_j, tau

        return {'coord_ind': coord_ind, 'threshold': threshold}

    @staticmethod
    def split_data(data, params):
        return data[:, params['coord_ind']] < params['threshold'], \
               data[:, params['coord_ind']] >= params['threshold']

    @staticmethod
    def create_split_node(params):
        return {'coord_ind': params['coord_ind'],
                'threshold': params['threshold'],
                'left': None,
                'right': None,
                'is_terminal': False}

    @staticmethod
    def create_terminal_node(target):
        return {'t': np.array([len(target[target == i]) / len(target) for i in range(10)]),
                'is_terminal': True}

    @staticmethod
    def get_entropy(target):

        h, n_i = 0, target.shape[0]
        n_k = np.unique(target, return_counts=True)[1]

        return np.sum([-i / n_i * np.log(i / n_i) for i in n_k])


    def information_gain(self, data, target, index_left, index_right):
        return self.get_entropy(target) \
               - len(data[index_left]) / len(data) * self.get_entropy(target[index_left]) \
               - len(data[index_right]) / len(data) * self.get_entropy(target[index_right])


def get_acc(mx, n): return np.sum(np.diagonal(mx)) / n


def show_graph(true_train, false_train,
               true_test, false_test):
    plt.figure("Гистограммы уверенности")
    plt.suptitle("Правильно и ошибочно классифицированные изображения")

    create_block(true_train, 1, "Уверенность", "Количество элементов",
                 '#8c2e67', "Правильные на Train")
    create_block(false_train, 2, "Уверенность","Количество элементов",
                 '#8c2e67', "Неправильные на Train")
    create_block(true_test, 3, "Уверенность", "Количество элементов",
                 '#2b85a1', "Правильные на Test")
    create_block(false_test, 4, "Уверенность", "Количество элементов",
                 '#2b85a1', "Неправильные на Test")

    plt.show()


def create_block(data, number, xlabel, ylabel, color, label):
    plt.subplot(2, 2, number)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.hist(data, color=color, label=label)
    plt.legend()


def get_metrics(data, target, tree):
    confusion_matrix = np.zeros((10, 10))
    true_prediction, false_prediction = [], []
    length = data.shape[0]

    for i in range(length):
        y = tree.predict(data[i])

        confusion_matrix[target[i], np.argmax(y)] += 1
        if np.argmax(y) == target[i]:
            true_prediction.append(np.max(y))
        else:
            false_prediction.append(np.max(y))

    return {'confusion_matrix': confusion_matrix, 'true_prediction': true_prediction,
            'false_prediction': false_prediction, 'accuracy': get_acc(confusion_matrix, length)}


limit_depth = 15
limit_data_len = 30
limit_entropy = 0.15

digits = load_digits()

ind = np.arange(digits.data.shape[0])
np.random.shuffle(ind)
ind_train = ind[:int(0.8 * digits.data.shape[0])]
ind_test = ind[int(0.8 * digits.data.shape[0]):]

d_train, t_train = digits.data[ind_train], digits.target[ind_train]
d_test, t_test = digits.data[ind_test], digits.target[ind_test]

dtree = DecisionTree()
dtree.fit(d_train, t_train, limit_depth, limit_data_len, limit_entropy)

metrics_train = get_metrics(d_train, t_train, dtree)
metrics_test = get_metrics(d_test, t_test, dtree)

print("Train:\nAccuracy = {0}\nConfusion matrix:\n{1}\n".
        format(metrics_train['accuracy'], metrics_train['confusion_matrix']))
print("Test:\nAccuracy = {0}\nConfusion matrix:\n{1}\n".
        format(metrics_test['accuracy'], metrics_test['confusion_matrix']))

show_graph(metrics_train['true_prediction'], metrics_train['false_prediction'],
            metrics_test['true_prediction'],  metrics_test['false_prediction'])
