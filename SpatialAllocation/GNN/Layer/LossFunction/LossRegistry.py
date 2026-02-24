

class LossRegistry:
    """
    Loss function registry class for managing and registering different loss functions.
    """
    def __init__(self):
        self._losses = {}
        self._weights = {}
        self._descriptions = {}

    def register(self, name, default_weight=0.0, description=""):
        def wrapper(func):
            self._losses[name] = func
            self._weights[name] = default_weight
            self._descriptions[name] = description
            return func

        return wrapper

    def get_loss(self, name):
        return self._losses.get(name)

    def get_default_weights(self):
        return self._weights.copy()

    def get_available_losses(self):
        return {
            name:{
                'weight':self._weights[name],
                'description':self._descriptions.get(name, "")
            }
            for name in self._losses
        }

