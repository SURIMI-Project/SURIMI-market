from models.elsa import ElsaHandler
from models.arimax import ArimaxHandler
from models.market import MarketHandler
from models.catboost import CatboostHandler

MODELS = {
    "elsa": ElsaHandler.ElsaHandler,
    "arimax": ArimaxHandler.ArimaxHandler,
    "market": MarketHandler.MarketHandler,
    "catboost": CatboostHandler.CatboostHandler,
}

def get_model_handler(model_name: str):
    """
    Returns the corresponding handler to the model name.

    The handlers are inherited from the base class `Handler`.

    Arguments
    ---------
        model_name: the model name to get the handler by.
    
    Returns
    -------
    If a corresponding model handler is found, then it is returned. Otherwise,
    `None` is returned.    
    """
    return MODELS.get(model_name)