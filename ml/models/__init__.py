from ml.models.base_model import BaseModel


def get_model(model_config):
    """
    Select the model according to the model config name and its parameters

    :param model_config: model_config namespace, containing the name and the params

    :return: model
    """

    if model_config.name == 'base_model':
        return BaseModel(model_config.params)
    else:
        raise ValueError(f'Wrong model name in model configs: {model_config.name}')
