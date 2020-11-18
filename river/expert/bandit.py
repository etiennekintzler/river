import abc
import copy
import typing

import numpy as np

from river import base
from river import metrics
from river import preprocessing
from river import stats


__all__ = [
    "Bandit",
    "EpsilonGreedyRegressor",
    "EpsilonGreedyClassifier",
    'EpsilonGreedyBandit',
    'Exp3Bandit',
    'OracleBandit',
    'RandomBandit',
    'RandomBanditRegressor',
    'RandomBanditClassifier',
    'UCB',
    'UCBRegressor',
    'UCBClassifier',
]

# TODO :
# type annotation in __init__
# loss-based reward ('_compute_scaled_reward')
# tests for classification
# tests on real datasets
# In Exp3: NaN in distribution when np.exp(x) becomes too large
# weird behavior if learn_one without predict_one

# Ref not integrated : 
# [Python code from Toulouse univ](https://www.math.univ-toulouse.fr/~jlouedec/demoBandits.html)
# [Slivkins, A. Introduction to Multi-Armed Bandits](https://arxiv.org/pdf/1904.07272.pdf)

# Echange avec Max:
# Les classes filles EpsilonGreedy et EpsilonRegressor héritent de Bandit et BaseClassifier ou BaseRegressor

# Réflexions:
# Séparer l'effet selection du best arm de l'effet entrainement --> donner les modèles entraînés par l'oracle à un bandit
# Utiliser directement river.stats.mean.Mean()-> pas nécessaire ni souhaitable car je travail sur des vecteurs

class Bandit(base.EnsembleMixin):
    def __init__(self, models, metric: metrics.Metric, reward_scaler: base.Transformer, verbose=False, save_rewards=False):
        if len(models) <= 1:
            raise ValueError(f"You supply {len(models)} models. At least 2 models should be supplied.")
        # Check that the model and the metric are in accordance
        for model in models:
            if not metric.works_with(model):
                raise ValueError(f"{metric.__class__.__name__} metric can't be used to evaluate a " +
                                 f'{model.__class__.__name__}')
        super().__init__(models)
        self.reward_scaler = copy.deepcopy(reward_scaler)
        self.metric = copy.deepcopy(metric)
        self.verbose = verbose
        self.models = models
        self.save_rewards = save_rewards
        if save_rewards:
            self.rewards: typing.List = []

        self._n_arms = len(models)
        self._N = np.zeros(self._n_arms, dtype=np.int)
        self._average_reward = np.zeros(self._n_arms, dtype=np.float)

    @abc.abstractmethod
    def _pull_arm(self):
        pass

    @abc.abstractmethod
    def _update_arm(self, arm, reward):
        pass
    
    @abc.abstractmethod
    def _pred_func(self, model):
        pass

    @property
    def _best_model_idx(self):
        return np.argmax(self._average_reward) # average reward instead of cumulated (otherwise favors arms which are pulled often)

    @property
    def best_model(self):
        return self[self._best_model_idx]
        
    def _compute_scaled_reward(self, y_pred, y_true):
        metric_value = self.metric._eval(y_pred, y_true)
        metric_to_reward_dict = {"metric": metric_value if self.metric.bigger_is_better else (-1) * metric_value}
        self.reward_scaler.learn_one(metric_to_reward_dict)
        reward = self.reward_scaler.transform_one(metric_to_reward_dict)["metric"]
        return reward

    def get_percentage_pulled(self):
        percentages = self._N / sum(self._N)
        return percentages

    def predict_one(self, x):
        best_arm = self._pull_arm()
        y_pred = self._pred_func(self[best_arm])(x)
        #y_pred = self[best_arm].predict_one(x)
        return y_pred

    def learn_one(self, x, y):
        chosen_arm = self._pull_arm()
        chosen_model = self[chosen_arm]
        y_pred = chosen_model.predict_one(x)

        # Train chosen model
        chosen_model.learn_one(x=x, y=y)

        # Compute reward and update chosen arm
        reward = self._compute_scaled_reward(y_pred=y_pred, y_true=y)
        self._N[chosen_arm] += 1
        self._average_reward[chosen_arm] += (1.0 / self._N[chosen_arm]) * (reward - self._average_reward[chosen_arm])
        
        # Update arm based on the bandit class
        self._update_arm(chosen_arm, reward)

        if self.save_rewards:
            self.rewards += [reward]

        if self.verbose:
            print(f'best {self._best_model_idx}')
            print("y_pred:", y_pred, ", y_true:", y)

        if self.metric:
            self.metric.update(y_pred=y_pred, y_true=y)

        return self


class EpsilonGreedyBandit(Bandit):
    def __init__(self, epsilon=0.1, reduce_epsilon=0.99, **kwargs):
        super().__init__(**kwargs)
        self.epsilon = epsilon
        self.reduce_epsilon = reduce_epsilon
        if not self.reward_scaler:
            self.reward_scaler = preprocessing.StandardScaler()

    def _pull_arm(self):
        if np.random.rand() > self.epsilon:
            chosen_arm = np.argmax(self._average_reward)
        else:
            chosen_arm = np.random.choice(self._n_arms)

        return chosen_arm

    def _update_arm(self, arm, reward):
        # The arm is already updated in the learn_one phase of class Bandit
        if self.reduce_epsilon:
            self.epsilon *= self.reduce_epsilon


class EpsilonGreedyRegressor(EpsilonGreedyBandit, base.Regressor):
    """Epsilon-greedy bandit algorithm for regression.
    
    This bandit selects the best arm (defined as the one with the highest average reward) with probability (1 - epsilon)
    and draw a random arm with probability epsilon. It is also called Follow-The-Leader FTL algorithm.

    For this bandit, reward are supposed to be 1-subgaussian, hence the use of the StandardScaler and MaxAbsScaler as reward_scaler
    
    Parameters
    ----------
    models
        The models to compare.
    metric
        Metric used for comparing models with.
    epsilon
        Exploration parameter (default : 0.1).
    reduce_epsilon
        Factor applied to reduce epsilon at each time-step (default : 0.99).


    Examples
    --------

    >>> from river import linear_model
    >>> from river import expert
    >>> from river import preprocessing
    >>> from river import metrics

    
    TODO: finish ex

    References
    ----------
    [^1]: [Sutton, R. S., & Barto, A. G. (2018). Reinforcement learning: An introduction. MIT press.](http://incompleteideas.net/book/RLbook2020.pdf)
    [^2]: [Rivasplata, O. (2012). Subgaussian random variables: An expository note. Internet publication, PDF.]: (https://sites.ualberta.ca/~omarr/publications/subgaussians.pdf)
    [^3]: [Lattimore, T., & Szepesvári, C. (2020). Bandit algorithms. Cambridge University Press.](https://tor-lattimore.com/downloads/book/book.pdf)
    """
    def _pred_func(self, model):
        return model.predict_one

class EpsilonGreedyClassifier(EpsilonGreedyBandit, base.Classifier):
    """Epsilon-greedy bandit algorithm for classification.
    
    This bandit selects the best arm (defined as the one with the highest average reward) with probability (1 - epsilon)
    and draw a random arm with probability epsilon. It is also called Follow-The-Leader FTL algorithm.
    
    For this bandit, reward are supposed to be 1-subgaussian, hence the use of the StandardScaler and MaxAbsScaler as reward_scaler
    
    Parameters
    ----------
    models
        The models to compare.
    metric
        Metric used for comparing models with.
    epsilon
        Exploration parameter (default : 0.1).
    reduce_epsilon
        Factor applied to reduce epsilon at each time-step (default : 0.99).


    Examples
    --------

    >>> from river import linear_model
    >>> from river import expert
    >>> from river import preprocessing
    >>> from river import metrics

    
    TODO: finish ex

    References
    ----------
    [^1]: [Sutton, R. S., & Barto, A. G. (2018). Reinforcement learning: An introduction. MIT press.](http://incompleteideas.net/book/RLbook2020.pdf)
    [^2]: [Rivasplata, O. (2012). Subgaussian random variables: An expository note. Internet publication, PDF.]: (https://sites.ualberta.ca/~omarr/publications/subgaussians.pdf)
    [^3]: [Lattimore, T., & Szepesvári, C. (2020). Bandit algorithms. Cambridge University Press.](https://tor-lattimore.com/downloads/book/book.pdf)
    """
    def _pred_func(self, model):
        return model.predict_proba_one


class UCBBandit(Bandit):
    def __init__(self, delta=None, **kwargs):
        super().__init__(**kwargs)
        self._n_iter = 0
        self.delta = delta
        if not self.reward_scaler:
            self.reward_scaler = preprocessing.StandardScaler()

    def fun_delta(self):
        return 1 / self.delta
        #return 1 + self._n_iter*(np.log(self._n_iter)**2)

    def _pull_arm(self):
        explore_each = 5
        not_pulled_enough = self._N <= explore_each
        if any(not_pulled_enough): # Explore all arms first
            never_pulled_arm = np.where(not_pulled_enough)[0] #[0] because returned a tuple (array(),)
            chosen_arm = np.random.choice(never_pulled_arm)
        else:
            if self.delta:
                exploration_bonus = np.sqrt(2 * np.log(self.fun_delta()) / self._N)
            else:
                exploration_bonus = np.sqrt(2 * np.log(self._n_iter) / self._N)
            upper_bound = self._average_reward + exploration_bonus
            chosen_arm = np.argmax(upper_bound)

        return chosen_arm

    def _update_arm(self, arm, reward):
        # The arm is partially updated (average reward) in the learn_one phase of class `Bandit`.
        self._n_iter += 1


class UCBRegressor(UCBBandit, base.Regressor):
    """Upper Confidence Bound bandit for regression.
    
    For this bandit, rewards are supposed to be 1-subgaussian (see Lattimore and Szepesvári, chapter 6, p. 91) hence the use of the StandardScaler and MaxAbsScaler as reward_scaler

    References
    ----------
    [^1]: [Auer, P., Cesa-Bianchi, N., & Fischer, P. (2002). Finite-time analysis of the multiarmed bandit problem. Machine learning, 47(2-3), 235-256.](https://link.springer.com/content/pdf/10.1023/A:1013689704352.pdf)
    [^2]: [Rivasplata, O. (2012). Subgaussian random variables: An expository note. Internet publication, PDF.]: (https://sites.ualberta.ca/~omarr/publications/subgaussians.pdf)
    [^3]: [Lattimore, T., & Szepesvári, C. (2020). Bandit algorithms. Cambridge University Press.](https://tor-lattimore.com/downloads/book/book.pdf)
    """
    def _pred_func(self, model):
        return model.predict_one


class UCBClassifier(UCBBandit, base.Classifier):
    """Upper Confidence Bound bandit for classification.
    
    For this bandit, rewards are supposed to be 1-subgaussian (see Lattimore and Szepesvári, chapter 6, p. 91) hence the use of the StandardScaler and MaxAbsScaler as reward_scaler

    References
    ----------
    [^1]: [Auer, P., Cesa-Bianchi, N., & Fischer, P. (2002). Finite-time analysis of the multiarmed bandit problem. Machine learning, 47(2-3), 235-256.](https://link.springer.com/content/pdf/10.1023/A:1013689704352.pdf)
    [^2]: [Rivasplata, O. (2012). Subgaussian random variables: An expository note. Internet publication, PDF.]: (https://sites.ualberta.ca/~omarr/publications/subgaussians.pdf)
    [^3]: [Lattimore, T., & Szepesvári, C. (2020). Bandit algorithms. Cambridge University Press.](https://tor-lattimore.com/downloads/book/book.pdf)
    """
    def _pred_func(self, model):
        return model.predict_proba_one


class Exp3Bandit(Bandit):
    """Exp3 implementation from Lattimore and Szepesvári. 
    The algorithm makes the hypothesis that the reward is in [0, 1]. Thus the scaler MinMaxScaler should be used.
   
    Parameters
    ----------
    models
        The models to compare.
    metric
        Metric used for comparing models with.
    gamma
        Parameter for exploration.

    References
    ----------
    [^1]: [Lattimore, T., & Szepesvári, C. (2020). Bandit algorithms. Cambridge University Press.](https://tor-lattimore.com/downloads/book/book.pdf)
    """
    def __init__(self, gamma=0.5, **kwargs):
        super().__init__(**kwargs)
        self._p = np.ones(self._n_arms, dtype=np.float)
        self._sum_expected_reward = np.zeros(self._n_arms, dtype=np.float)
        self.gamma = gamma

    def _make_distr(self):
        s_hat = self._sum_expected_reward  - np.min(self._sum_expected_reward)
        numerator = np.exp(self.gamma * s_hat)
        #numerator =  np.exp(self.gamma * self._s)
        return numerator / np.sum(numerator)

    def _pull_arm(self):
        self._p = self._make_distr()
        chosen_arm = np.random.choice(a=range(self._n_arms), size=1, p=self._p)[0]
        return chosen_arm

    def _update_arm(self, arm, reward):
        self._sum_expected_reward += 1.0
        self._sum_expected_reward[arm] -= (1.0 - reward) / self._p[arm]


class RandomBandit(Bandit):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def _pull_arm(self):
        random_arm = np.random.choice(self._n_arms)
        return random_arm

    def _update_arm(self, arm, reward):
        pass


class RandomBanditRegressor(RandomBandit, base.Regressor):
    """Random Bandit Regressor.
    Does random selection and update of models. It is just created for testing purpose"""
    def _pred_func(self, model):
        return model.predict_one


class RandomBanditClassifier(RandomBandit, base.Classifier):
    """Random Bandit Classifier.
    Does random selection and update of models. It is just created for testing purpose"""
    def _pred_func(self, model):
        return model.predict_proba_one


class OracleBandit(Bandit):
    """The oracle bandit updates every model.
    The `predict_one` is the prediction that minimize the error for this time step.
    The `update_one` update all the models available."""

    def __init__(self, save_predictions=True, **kwargs):
        super().__init__(**kwargs)
        self.save_predictions = save_predictions
        if save_predictions:
            self.predictions = []

    @property
    def _best_model_idx(self):
        cum_rewards = np.max(np.array(self.rewards), axis=0)
        return np.argmax(cum_rewards)

    @property
    def max_rewards(self):
        return np.max(np.array(self.rewards), axis=1)

    def predict_one(self, x, y):
        best_arm = self._pull_arm(x, y)
        y_pred = self[best_arm].predict_one(x)
        return y_pred

    def _pull_arm(self, x, y):
        preds = [model.predict_one(x) for model in self]
        losses = [np.abs(y_pred - y) for y_pred in preds]
        chosen_arm = np.argmin(losses)
        return chosen_arm

    def _update_arm(self, arm, reward):
        pass

    def learn_one(self, x, y):
        rewards_timestep = []
        predictions_timestep = []

        for model in self:
            if self.save_predictions or self.save_rewards:
                y_pred = model.predict_one(x=x)

            model.learn_one(x=x, y=y)
            if self.save_predictions:
                predictions_timestep += [y_pred]

            if self.save_rewards:
                rewards_timestep += [self._compute_scaled_reward(y_pred=y_pred, y_true=y)]

        if self.save_predictions:
            self.predictions += [predictions_timestep]
        
        if self.save_rewards:
            self.rewards += [rewards_timestep]

        return self

