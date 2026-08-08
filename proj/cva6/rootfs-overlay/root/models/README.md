# Persistent models

The SD-card image mounts its writable model partition here. It contains `LFM2.5-350M-Q4_0.gguf` and `Bonsai-1.7B.gguf` when those source models were available while building the image. The RAM-only boot leaves this directory unmounted and continues to use weights staged directly into reserved DDR.
