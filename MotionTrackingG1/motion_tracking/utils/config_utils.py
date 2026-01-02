from omegaconf import OmegaConf
import math


OmegaConf.register_new_resolver("eval", eval)
OmegaConf.register_new_resolver("if", lambda pred, a, b: a if pred else b)
OmegaConf.register_new_resolver("eq", lambda x, y: x.lower() == y.lower())
OmegaConf.register_new_resolver("sqrt", lambda x: math.sqrt(float(x)))
OmegaConf.register_new_resolver("sum", lambda x: sum(x))
OmegaConf.register_new_resolver("ceil", lambda x: math.ceil(x))
OmegaConf.register_new_resolver("len", lambda x: len(x))
