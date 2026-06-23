from pydantic import BaseModel


class TeambuildConfig(BaseModel):
    """Configuration for the surrogate-assisted genetic algorithm teambuild policy.

    Args:
        population_size: Number of candidate teams in each generation.
        generations: Number of evolutionary generations to run.
        mutation_rate: Probability of a single-species mutation per team member.
        elite_fraction: Fraction of the population retained unchanged each generation.
    """

    population_size: int = 100
    generations: int = 50
    mutation_rate: float = 0.05
    elite_fraction: float = 0.10
