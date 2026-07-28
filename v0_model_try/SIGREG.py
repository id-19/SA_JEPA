# Implement basic normality loss
# Take a bunch of latents, project onto unit vectors of same dimension, then do normality test on the project distribution
import torch


def create_unit_vectors(dimension: int, num_unit_vecs: int) -> list[torch.Tensor]:
    unit_vectors = []
    for _ in range(num_unit_vecs):
        unit_vectors.append(torch.randn()) # TODO implement this
    return unit_vectors


def project_latents(latents: list[torch.Tensor]) -> torch.Tensor:
    latent_dimension = latents[0].shape[-1]
    unit_vectors = create_unit_vectors(latent_dimension, int(len(latents) / 10)) # 0.1 as many unit vectors as latents
    # TODO: Implement this
    return torch.randn()
