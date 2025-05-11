#!/bin/bash

# Configure the hyperparameter grid for the experiment/benchmark sweep
args=(
  # Run each experiment configuration multiple times using different seeds to
  # allow for averaging out noise effects later
  -S seed="12,345,657"
  # Input shapes to "scale up" the benchmarked model. Only test two-dimensional
  # layouts NxC, these should already cover all interesting behavior
  -S shape="[256]"
  # Sweep some larger bit-widths at the input - in a real model we would get
  # these larger bit-widths out of some matrix multiplication
  -S model.input_bits="8,16,24"
  # Sweep smaller bit-widths actually testing the quantized activation functions
  -S model.bits="2,4,8"
  # FP or POWER_OF_TWO for both, quantizers and affine scales
  -S model.restrict_scaling_type="FP,POWER_OF_TWO"
  # There are probably no insights to gain from comparing narrow range vs.
  # non-narrow range
  -S model.narrow_range="false"
  # Test per-channel and per-tensor affine scales
  -S model.affine.per_channel="false,true"
  # Always test with streamlining for some cleaner comparisons: The key
  # comparison is standalone vs. thresholds
  -S prepare.streamline="true"
  # Test thresholding (with operator fusion) vs. standalone quantizers (with
  # elementwise operations potentially in float arithmetic)
  -S prepare.thresholds="true"
  # Always test with bit-width minimization for some cleaner comparisons: The
  # key comparison is standalone vs. thresholds
  -S build.minimize_bit_width="true"
)

# Fill the experiment queue spanning the whole grid configured above
dvc exp run --queue "${args[@]}" --force
