from opacus.accountants import RDPAccountant
from opacus.accountants.utils import get_noise_multiplier


def calculate_noise_multiplier(
    target_epsilon: float,
    target_delta: float,
    sample_rate: float,
    total_steps: int,
) -> float:

    noise_multiplier = get_noise_multiplier(
        target_epsilon=target_epsilon,
        target_delta=target_delta,
        sample_rate=sample_rate,
        steps=total_steps,
        accountant="rdp",
    )

    return noise_multiplier


def calculate_epsilon(
    noise_multiplier: float,
    sample_rate: float,
    total_steps: int,
    target_delta: float,
) -> float:

    accountant = RDPAccountant()

    for _ in range(total_steps):

        accountant.step(
            noise_multiplier=noise_multiplier,
            sample_rate=sample_rate,
        )

    epsilon = accountant.get_epsilon(
        delta=target_delta
    )

    return epsilon