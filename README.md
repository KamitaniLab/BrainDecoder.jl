# BrainDecoder.jl

Brain decoder toolbox for Julia.

## Installation

``` julia
(@v1.5) pkg> add https://github.com/KamitaniLab/BrainDecoder.jl
```

## Examples

``` julia

using BrainDecoder

# BData

bdata = BData("/path/to/bdata_file.h5")

bdata.select("VoxelData")
bdata.select("ROI_V1")
bdata.select("Run")
bdata.get_labels("stimulus_name")

# Features

features = Features("/path/to/features_directory")

features.layers
features.labels
features.get_features("conv1")
```
