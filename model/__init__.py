from model import Model
from q_learning import QLearningModel
from dqn import DQNModel


def get_model(model_name, **kwargs):
    model_map = {
        'QLearning': QLearningModel,
        'DQN': DQNModel,
    }
    model_class = model_map.get(model_name, Model)
    model = model_class(**kwargs)
    return model
