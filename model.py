from typing import Callable
from sb3_contrib import RecurrentPPO


default_model = lambda e: RecurrentPPO(
    "MlpLstmPolicy",
    e,
    verbose=1,
    learning_rate=0.01,
    tensorboard_log="./logs/",
    ent_coef=0.05,
)
