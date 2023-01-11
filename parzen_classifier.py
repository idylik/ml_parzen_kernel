import numpy as np

banknote = np.genfromtxt('data_banknote_authentication.txt', delimiter=',')

######## DO NOT MODIFY THIS FUNCTION ########
def draw_rand_label(x, label_list):
    seed = abs(np.sum(x))
    while seed < 1:
        seed = 10 * seed
    seed = int(1000000 * seed)
    np.random.seed(seed)
    return np.random.choice(label_list)
#############################################


def minkowski_mat(x, Y, p=2):
    return (np.sum((np.abs(x - Y)) ** p, axis=1)) ** (1.0 / p)


class Q1:

    def feature_means(self, banknote):
        return np.mean(banknote[:,:-1], axis=0)

    def covariance_matrix(self, banknote):
        return np.cov(banknote[:,:-1], rowvar=False)

    def feature_means_class_1(self, banknote):
        return np.mean(banknote[banknote[:,-1]==1][:,:-1], axis=0)

    def covariance_matrix_class_1(self, banknote):
        return np.cov(banknote[banknote[:,-1]==1][:,:-1], rowvar=False)



class HardParzen:
    def __init__(self, h):
        self.h = h

    def train(self, train_inputs, train_labels):
        self.train_inputs = train_inputs
        self.train_labels = train_labels
        self.label_list = np.unique(self.train_labels)
        self.n_classes = len(self.label_list)

    def compute_predictions(self, test_data):
        classes_pred = np.empty(len(test_data))

        for i, ex in enumerate(test_data):
            Y = np.zeros(self.n_classes)
            distances = minkowski_mat(ex, self.train_inputs) #longueur n
            weights = np.array(distances<=self.h).astype('int') #longueur n

            for j in range(len(self.train_inputs)):
                one_hot = np.zeros(self.n_classes)
                one_hot[int(self.train_labels[j])] = 1
                Y += weights[j]*one_hot

            if (np.sum(weights) == 0):
                pred = int(draw_rand_label(ex, self.label_list))
            else:
                pred = int(np.argmax(Y))

            classes_pred[i] = pred

        return classes_pred




class SoftRBFParzen:
    def __init__(self, sigma):
        self.sigma  = sigma

    def train(self, train_inputs, train_labels):
        self.train_inputs = train_inputs
        self.train_labels = train_labels
        self.label_list = np.unique(self.train_labels)
        self.n_classes = len(self.label_list)
        self.n_features = train_inputs.shape[1]

    def compute_predictions(self, test_data):
        classes_pred = np.empty(len(test_data))

        for i, ex in enumerate(test_data):
            Y = np.zeros(self.n_classes) #Somme de tous les vecteurs k_j*y_j
            prob_exemples = self.gaussian_kernel(self.train_inputs, ex, self.sigma)

            for j in range(len(self.train_inputs)):
                one_hot = np.zeros(self.n_classes)
                one_hot[int(self.train_labels[j])] = 1
                Y += prob_exemples[j]*one_hot
            classes_pred[i]=int(np.argmax(Y))
        return classes_pred

    def gaussian_kernel(self, pos, mu, sigma):
        N = (2*np.pi)**(self.n_features/2)*sigma**self.n_features
        exponent = -0.5*(minkowski_mat(mu,pos)**2)/sigma**2
        return np.exp(exponent)/N



def split_dataset(banknote):
    train = np.array([banknote[i] for i in range(len(banknote)) if i%5==0 or i%5==1 or i%5==2])
    validation = np.array([banknote[i] for i in range(len(banknote)) if i%5==3])
    test = np.array([banknote[i] for i in range(len(banknote)) if i%5==4])

    return (train, validation, test)




class ErrorRate:
    def __init__(self, x_train, y_train, x_val, y_val):
        self.x_train = x_train
        self.y_train = y_train
        self.x_val = x_val
        self.y_val = y_val

    def hard_parzen(self, h):
        hp = HardParzen(h)
        hp.train(self.x_train, self.y_train)
        pred_val = hp.compute_predictions(self.x_val)
        return np.mean(pred_val != self.y_val)

    def soft_parzen(self, sigma):
        sp = SoftRBFParzen(sigma)
        sp.train(self.x_train, self.y_train)
        pred_val = sp.compute_predictions(self.x_val)
        return np.mean(pred_val != self.y_val)



def get_test_errors(banknote):
    #Séparer en ensembles: train, validation, test
    sets = split_dataset(banknote)
    #Trouver erreurs de validation sur parametres proposes
    param_vals = [0.01, 0.1, 0.2, 0.3, 0.4, 0.5, 1.0, 3.0, 10.0, 20.0]
    err_val = ErrorRate(sets[0][:, :-1], sets[0][:, -1], sets[1][:, :-1], sets[1][:, -1])
    error_val_rate_hp = []
    error_val_rate_sp = []

    for p in param_vals:
        error_val_rate_hp.append(err_val.hard_parzen(p))
        error_val_rate_sp.append(err_val.soft_parzen(p))

    optimal_h = param_vals[np.argmin(error_val_rate_hp)]
    optimal_sigma = param_vals[np.argmin(error_val_rate_sp)]

    err_test = ErrorRate(sets[0][:, :-1], sets[0][:, -1], sets[2][:, :-1], sets[2][:, -1])
    error_test_hp = err_test.hard_parzen(optimal_h)
    error_test_sp = err_test.soft_parzen(optimal_sigma)

    return [error_test_hp, error_test_sp]


def random_projections(X, A):
    return (1 / np.sqrt(2)) * np.dot(X, A)
