import constriction
import numpy as np
import scipy as sp
rg=65536

entropy_model = constriction.stream.model.QuantizedGaussian(-rg, rg, mean=0, std=1e-5)
coder = constriction.stream.queue.RangeEncoder()
theory=0
for i in range(2048):
    message=np.random.randint(-100, 100)
    coder.encode(message, entropy_model)
    compressed = coder.get_compressed()
    theory -= np.log2(sp.stats.norm.cdf(message+0.5, loc=0, scale=1e-5) - sp.stats.norm.cdf(message-0.5, loc=0, scale=1e-5))
    print(f'{theory} vs {len(compressed) * 32} bits (includes padding to a multiple of 32 bits).')
