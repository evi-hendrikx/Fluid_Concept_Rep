from sklearn.linear_model import *
from mapping_models.basic_mapper import BasicMapper

# The standard ridge regression model for mapping stimuli to scans.
class RegressionMapper(BasicMapper):
    def __init__(self, alpha = 1.0, model_fn=Ridge):
        super(RegressionMapper, self).__init__()
        self.alpha = alpha
        self.model_fn = model_fn
        self.model = None
    
    def build(self, is_train=True):
        """Create the model object using model_fn
        """
        self.model = self.model_fn(alpha=self.alpha)
        
    def map(self, inputs, targets=None):
        if self.model is None:
          self.build()
        predictions = self.model.predict(inputs)
         
        return predictions
        
        # TODO: skipped the loss for now

    def train(self, inputs, targets):
        if self.model is None:
          self.build()
        
        self.model.fit(inputs, targets)