using BrainDecoder
using HDF5
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
    layers = ["conv1", "conv2", "conv3", "conv4", "conv5", "fc6", "fc7", "fc8", "norm1", "norm2", "pool1", "pool2", "pool5", "prob", "relu1", "relu2", "relu3", "relu4", "relu5", "relu6", "relu7"]
    labels = ["n01443537_22563", "n01621127_19020", "n01677366_18182", "n01846331_17038", "n01858441_11077", "n01943899_24131", "n01976957_13223", "n02071294_46212", "n02128385_20264", "n02139199_10398", "n02190790_15121", "n02274259_24319", "n02416519_12793", "n02437136_12836", "n02437971_5013", "n02690373_7713", "n02797295_15411", "n02824058_18729", "n02882301_14188", "n02916179_24850", "n02950256_22949", "n02951358_23759", "n03064758_38750", "n03122295_31279", "n03124170_13920", "n03237416_58334", "n03272010_11001", "n03345837_12501", "n03379051_8496", "n03452741_24622", "n03455488_28622", "n03482252_22530", "n03495258_9895", "n03584254_5040", "n03626115_19498", "n03710193_22225", "n03716966_28524", "n03761084_43533", "n03767745_109", "n03941684_21672", "n03954393_10038", "n04210120_9062", "n04252077_10859", "n04254777_16338", "n04297750_25624", "n04387400_16693", "n04507155_21299", "n04533802_19479", "n04554684_53399", "n04572121_3262"]

    features = Features(feat_dir)

    @test features.layers == layers
    @test features.labels == labels
    @test features.get_labels() == labels
    @test size(features.get_features("fc8")) == (50, 1000)
    @test size(features.get_features("conv5")) == (50, 256, 13, 13)
    #@test size(features.get_features("conv1")) == (50, 96, 55, 55)

    # save_array
    data = [1 2 3 4 5
            6 7 8 9 10]
    save_array("test_save_array_dense.mat", "data", data)

    data = [1 0 0 0
            2 2 0 0
            3 3 3 0]
    save_array("test_save_array_sparse_2d.mat", "data", data, sparse=true)

end
