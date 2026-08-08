from gymnasium import Env
from sb3_contrib import RecurrentPPO


def default_model[T, U](e: Env[T, U]) -> RecurrentPPO:
    return RecurrentPPO(
        "MlpLstmPolicy",
        e,
        verbose=1,
        learning_rate=0.0003,
        target_kl=0.03,
        gamma=0.999,
        n_epochs=4,
        tensorboard_log="./logs/",
        ent_coef=0.03,
    )
