
v0:
1) audio file
2) load waveform 
3) resample to 24k mono
4) cut/pad fixed clip 
5) log-mel spectrogram
6) normalize 
7) batch
8) training tensor

v1(changes to make): DO LATER
1) Batching module generates a bunch of clip sizes(in seconds) for each batch
2) These are used after re-sample to construct batches with different clip lengths internally
3) This helps the model learn on different clip lengths

v2(changes to make): DO LATER
Note: v0 uses simple fixed-length clipping for batching simplicity, so some long-range context may be lost at clip boundaries.
Later improvement: replace hard clipping with trim-leading/trailing-silence + random/overlapping crops to preserve more natural acoustic context
This matters because pauses, transitions, and events that cross the clip boundary can carry useful semantic information.
For now, fixed clipping is acceptable because the goal is a minimal working preprocessing pipeline, not optimal context preservation