using BrainDecoder
using Test

@testset "BrainDecoder.jl" begin

    # BData
    voxel_data = [0.0   1.0   2.0   3.0   4.0   5.0   6.0   7.0   8.0   9.0
                  10.0  11.0  12.0  13.0  14.0  15.0  16.0  17.0  18.0  19.0
                  20.0  21.0  22.0  23.0  24.0  25.0  26.0  27.0  28.0  29.0
                  30.0  31.0  32.0  33.0  34.0  35.0  36.0  37.0  38.0  39.0
                  40.0  41.0  42.0  43.0  44.0  45.0  46.0  47.0  48.0  49.0
                  50.0  51.0  52.0  53.0  54.0  55.0  56.0  57.0  58.0  59.0]
    runs = reshape([1, 1, 1, 2, 2, 2], 6, 1)
    stimulus_labels = ["stimulus_02"; "stimulus_01"; "stimulus_03"; "stimulus_01"; "stimulus_03"; "stimulus_02"]
    
    bdata = BData("./data/test.h5")

    @test bdata.select("VoxelData") == voxel_data
    @test bdata.select("Run") == runs
    @test bdata.get_labels("stimulus_name") == stimulus_labels
    @test bdata.get_label("stimulus_name") == stimulus_labels

    # Features
    # TODO: add better tests
    feat_dir = "/home/nu/data/contents_shared/ImageNetTest/derivatives/features/caffe/bvlc_alexnet"
    features = Features(feat_dir)

    @test size(features.get_features("fc8")) == (50, 1000)
    @test size(features.get_features("conv5")) == (50, 256, 13, 13)
    #@test size(features.get_features("conv1")) == (50, 96, 55, 55)
    
end
