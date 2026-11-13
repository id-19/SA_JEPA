// The core sort of ideas of the paper

#INTRODUCTION
## What breaks in Linear attention + Memory bounds
1. Current Linear attention variants are seen as key-value memory: 
  - Each key is an address in memory i.e. some place to store stuff
  - Each input is stored as a 'memory' or 'value'(the v in attention)
  - The input is taken, the closest address to it is figured out(all other addresses are orthogonal roughly), and that address is written a clean version of the memory
  - These addresses are then limited as the number of 'key' vectors, which cannot be greater than d for a d-dim state(since the keys are then stored like so.)
  - Hence, these memories are limited, and after a point get noisy
2. The key equation: S.k_i[RECALL] = v_i . (k_i . k_i) [SIGNAL] + sum_j(v_j. (k_i . k_j)) [NOISE/CROSSTALK]
  - So recall only stays clear while your SIGNAL/NOISE ratio is within your bounds
3. Each key, query and value are the transformer ones, calculate from token embedding vector types
4. So, as time goes on, and the number of terms grows(N terms for N inputs), the share of the SIGNAL(one term) gets reduced(and is even decayed, since it's older) with time. At some point, the SIGNAL is drowned out by the noise
Solid, this is why memory systems can't have perfect recall. If you keep putting stuff in, you won't be able to get a particular one from before out.

## Mamba issues
1. Mamba applies a simple uniform forget gate across all key-value stores(all memories, note that these are not properly defined for a Mamba architecture). S_i = <alpha>. S_(i-1) + v_i * k_i^T
2. This doesn't differentiate what you need to forget(to make space for a newer memory), it just decays all channels(so all memories get uniformly decayed)

## Issues with existing DeltaNet
1. Since memory updates only happen one address at a time, there's no efficient memory clearing mechanism.
2. This is likely why it has only performed moderately till now.

## **Their core idea: Gated Delta Rule
Basically, in the equation: S_i = <alpha>. S_(i-1) + v_i * k_i^T. <alpha> is the thing that determines memory forgetting.
The idea: Make alpha dynamic, so alpha at:
1) <alpha> = 0: Wipe memory clean, just remember latest value
2) <alpha> = 1: Keep state as is, just write memory updates for that one token/value.
In easier words: Forget memories over time generally, but also forget memories specifically when newer ones need their space. The balance here is maintained by looking at your current input.
