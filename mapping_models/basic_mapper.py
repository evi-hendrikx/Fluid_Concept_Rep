# Abstract class for the mapping model.
class BasicMapper(object):
  def __init__(self):
    pass

  def build(self, is_train):
    raise NotImplementedError()

  def map(self, inputs, targets):
    raise NotImplementedError()

  def prepare_inputs(self, **kwargs):
    raise NotImplementedError()
