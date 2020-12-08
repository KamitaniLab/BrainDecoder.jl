module BrainDecoder

export BData, Features

using HDF5
using MAT

# BData ######################################################################

struct BData
    select
    get_labels
    get_label  # For compatibility to bdpy
end

function BData(_file::String)

    # Load Bdata from the given file
    h5 = h5open(_file, "r")
    dataset = read(h5, "dataset")'
    metadata = MetaData(read(h5, "metadata/key"),
                        read(h5, "metadata/value")')
    vmap = read(h5, "vmap")

    close(h5)

    function select(query::String)
        # TODO: add vectorize option
        md_idx = Bool[ query == metadata.key[i] for i = 1:size(metadata.key, 1) ]
        ds_col_idx = Bool[ [1.0] == metadata.value[md_idx, i] for i = 1:size(metadata.value[md_idx, :], 2) ]  # ???
        return dataset[:, ds_col_idx]
    end

    function get_labels(key::String)
        return [vmap[key][string(v)] for v in vec(select(key))]
    end    

    function get_label(key::String)
        return get_labels(key)
    end
    
    return BData(select, get_labels, get_label)
end

struct MetaData
    key::Array{String,1}
    value::Array{AbstractFloat,2}
end


# Features ###################################################################

struct Features
    layers
    labels
    get_features
    get_labels
end

function Features(_dir::String)

    feature_key = "feat"
    
    # Get layers
    layers = [basename(d) for d in readdir(_dir, join=true, sort=true) if isdir(d)]

    # Get labels
    labels = [splitext(basename(f))[1] for f in readdir(joinpath(_dir, layers[1]), join=true, sort=true) if isfile(f)]

    function get_features(layer::String)
        features = zeros(0)
        for lb in labels
            features = cat(features, _load_features(joinpath(_dir, layer, lb * ".mat"), feature_key), dims=1)
        end
        return features
    end

    function get_labels()
        return labels
    end

    function _load_features(fpath::String, key::String)
        mat = matopen(fpath)
        feat = read(mat, key)
        close(mat)
        return feat
    end
    
    return Features(layers, labels, get_features, get_labels)
end

end # module BrainDecoder
