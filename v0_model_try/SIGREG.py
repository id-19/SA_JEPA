# Implement basic normality loss
# Take a bunch of latents, project onto unit vectors of same dimension, then do normality test on the project distribution
import torch


def create_unit_vectors(dimension: int, num_unit_vecs: int) -> list[torch.Tensor]:
    unit_vectors = []
    for _ in range(num_unit_vecs):
        # Create unit vectors of same dimension as latent
        unit_vectors.append(torch.randn(dimension)) # Broadcasting will automatically infer earlier dimensions(i.e. matches right to left so (Z, K) and (K) can be multiplied as (Z,K) and (1, K))
    return unit_vectors


def project_latents(latents: list[torch.Tensor]) -> torch.Tensor:
    latent_dimension = latents[0].shape[-1]
    unit_vectors = create_unit_vectors(latent_dimension, int(len(latents) / 10)) # 0.1 as many unit vectors as latents
    # TODO: Implement this
    return torch.randn()

if __name__ == '__main__':
    unit_vectors = create_unit_vectors(512, 10)
    print(unit_vectors[0].shape)
