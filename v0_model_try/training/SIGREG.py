# Implement basic normality loss
# Take a bunch of latents, project onto unit vectors of same dimension, then do normality test on the project distribution
import time

import torch
from collections.abc import Callable

def create_unit_vectors(dimension: int, num_unit_vecs: int) -> list[torch.Tensor]:
    unit_vectors = []
    for _ in range(num_unit_vecs):
        # Create unit vectors of same dimension as latent
        unit_vectors.append(torch.randn(dimension, requires_grad=False)) # Broadcasting will automatically infer earlier dimensions(i.e. matches right to left so (Z, K) and (K) can be multiplied as (Z,K) and (1, K))
    return unit_vectors

def calculate_empirical_char_fn(projections: torch.Tensor) -> Callable[[torch.Tensor], tuple[torch.Tensor, torch.Tensor]]:
    # Returns a function that gives value of empirical char fn. at a point
    # Get number of projections
    num_projections = projections.shape[0]
    # Real part
    real_calc = lambda t: (1 / num_projections) * torch.sum(torch.cos(t * projections))
    # img part
    img_calc = lambda t: (1 / num_projections) * torch.sum(torch.sin(t * projections))
    # Put it together
    return lambda t: (real_calc(t), img_calc(t))

def char_fn_normal_dist(t: int) -> tuple[torch.Tensor, torch.Tensor]:
    return (torch.tensor(0), torch.exp(-1/2 * torch.square(torch.tensor(t))))

def calculate_l2_dist(a_real: torch.Tensor, a_img: torch.Tensor, b_real: torch.Tensor, b_img: torch.Tensor) -> torch.Tensor:
    return torch.sqrt(torch.square(a_real - b_real) + torch.square(a_img - b_img))

def calculate_epps_pulley_score(projections: torch.Tensor) -> torch.Tensor:
    # 1. Calculate empirical char_fn
    empirical_char_fn = calculate_empirical_char_fn(projections)

    # 2. Sum ()^2 of diff of empiric vs normal characteristic function
    loss = torch.tensor(0.0, dtype=torch.float32)
    for i in range(-1000, 1000):
        a_real, a_img = empirical_char_fn(i)
        b_real, b_img = char_fn_normal_dist(i)
        loss += torch.square(a_real - b_real) + torch.square(a_img - b_img)
    return loss


def calculate_normality(latents: list[torch.Tensor], debug: bool = False) -> torch.Tensor:
    # All operations need backprop so they need grad
    latent_dimension = latents[0].shape[-1]

    # Flatten the latents from a P[(B, T, K)] shape to a (P * B * T, K) shape
    ## Use flatten because it doesn't break for non-contiguous storage in memory
    latents = [batch_latent.flatten(0, 1) for batch_latent in latents] # P[(B*T, K)]
    final_latents = torch.cat(latents, dim=0) # (P * B * T, K)

    # Get unit vectors
    unit_vectors = create_unit_vectors(latent_dimension, int(final_latents.size(0) / 10)) # 0.1 as many unit vectors as latents

    if debug:
        print(f"unit_vectors: {unit_vectors}")

    if debug:
        print(f"Shape of final latents: {final_latents.size()}")
        print(f"Sample latent: {final_latents[0]}")

    # Project all latents onto each unit vector, call normality test function on each set of projections(per unit vector)
    normality_scores_sum = torch.Tensor([0.0])
    num_scores = 0
    start_time = time.time()
    for unit_vec in unit_vectors:
        projections = final_latents @ unit_vec

        epps_pulley_score = calculate_epps_pulley_score(projections)

        if debug:
            print(f"projections: {projections}")
            print(f"epps_pulley_score: {epps_pulley_score}")

        normality_scores_sum += epps_pulley_score
        num_scores += 1

    end_time = time.time()
    if debug:
        print(f"normality_scores_sum: {normality_scores_sum}")
        print(f"num_scores: {num_scores}")
        print(f"Time taken: {end_time - start_time}")
        print(f"Time taken per normality score: {(end_time - start_time) / num_scores}")

    return normality_scores_sum / num_scores

if __name__ == '__main__':
    # latents = [torch.randn((10, 100, 512)) for _ in range(10)]
    latents = [torch.randn((5, 20, 512)) for _ in range(10)]
    normality = calculate_normality(latents, debug=True)
    print(normality)
