from model import Model
from q_learning import QLearningModel


def get_model(model_name, **kwargs):
    model_map = {
        'QLearning': QLearningModel,
    }
    model_class = model_map.get(model_name, Model)
    model = model_class(**kwargs)
    return model
