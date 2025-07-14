using Flux
using Images
using FileIO
using Glob
using MLUtils
using ImageTransformations
using Statistics
using Random
using ColorTypes
using BSON

# --- Dataset Loader  ---
function load_dataset(folder_path::String; max_per_class::Int=100)
    classes = sort(readdir(folder_path))
    label_map = Dict(class => i for (i, class) in enumerate(classes))

    X, y = [], Int[]
    for (class, label) in label_map
        img_paths = Glob.glob("*.jpg", joinpath(folder_path, class))[1:max_per_class]
        for img_file in img_paths
            try
                img = load(img_file)
                img = imresize(img, (100, 100))
                img = RGB.(img)
                arr = Float32.(permutedims(channelview(img), (2, 3, 1)))
                if size(arr) == (100, 100, 3)
                    push!(X, arr)
                    push!(y, label)
                else
                    @warn "Skipped: wrong shape $(size(arr)) in $img_file"
                end
            catch e
                @warn "Skipped: error loading $img_file — $e"
            end
        end
    end

    println("Loaded $(length(X)) valid images.")
    return X, y, label_map
end

# --- Preprocessing ---
function preprocess_data(X, y)
    # Filter valid image entries
    valid_data = [(x, y[i]) for (i, x) in enumerate(X) if size(x) == (100, 100, 3) && eltype(x) == Float32]
    X_clean = [copy(first(pair)) for pair in valid_data]  # ensure full Float32 arrays
    y_clean = [last(pair) for pair in valid_data]

    println("Preprocessing $(length(X_clean)) valid images...")

    # Preallocate a tensor
    H, W, C = 100, 100, 3
    N = length(X_clean)
    X_tensor = Array{Float32}(undef, H, W, C, N)
    for i in 1:N
        X_tensor[:, :, :, i] = X_clean[i]
    end

    y_tensor = Flux.onehotbatch(y_clean, 1:maximum(y_clean))
    return X_tensor, y_tensor
end


# --- Dynamic CNN (LeNet-style) for 100×100 input ---
function build_lenet_for_100x100(num_classes::Int)
    cnn = Chain(
        Conv((5, 5), 3=>6, relu),
        MaxPool((2, 2)),
        Conv((5, 5), 6=>16, relu),
        MaxPool((2, 2)),
        flatten
    )

    dummy = rand(Float32, 100, 100, 3, 1)
    flattened_size = size(cnn(dummy), 1)

    return Chain(
        cnn,
        Dense(flattened_size, 120, relu),
        Dense(120, 84, relu),
        Dense(84, num_classes),
        softmax
    )
end

# --- Training ---
function train_model(model, X, y; epochs=5, batch_size=16)
    data = Flux.DataLoader((X, y), batchsize=batch_size, shuffle=true)
    opt_state = Flux.setup(Flux.ADAM(0.001), model)

    for epoch in 1:epochs
        for (x_batch, y_batch) in data
            grads = Flux.gradient(model) do m
                Flux.crossentropy(m(x_batch), y_batch)
            end
            Flux.update!(opt_state, model, grads)
            # Flux.Optimise.update!(opt_state, model, grads)

        end
        println("Epoch $epoch complete")
    end
end

# --- Accuracy ---
function test_accuracy(model, X, y)
    ŷ = model(X)
    acc = mean(Flux.onecold(ŷ) .== Flux.onecold(y))
    println("Test Accuracy: $(round(acc * 100, digits=2))%")
end

# --- 10-Fold Cross-Validation ---
function cross_validate_10fold(X_tensor, y_tensor, num_classes::Int)
    N = size(X_tensor, 4)
    k = 10
    idx = collect(1:N)
    Random.shuffle!(idx)
    fold_size = cld(N, k)
    accs = Float64[]

    for i in 1:k
        test_idx = idx[((i-1)*fold_size + 1):min(i*fold_size, N)]
        train_idx = setdiff(idx, test_idx)

        X_train = X_tensor[:, :, :, train_idx]
        y_train = y_tensor[:, train_idx]
        X_test = X_tensor[:, :, :, test_idx]
        y_test = y_tensor[:, test_idx]

        model = build_lenet_for_100x100(num_classes)
        train_model(model, X_train, y_train, epochs=3)
        acc = mean(Flux.onecold(model(X_test)) .== Flux.onecold(y_test))
        println("Fold $i Accuracy: $(round(acc * 100, digits=2))%")
        push!(accs, acc)

        BSON.@save "lenet_fold_$i.bson" model
    end

    println("\nFold-wise Accuracies: ", round.(accs .* 100, digits=2))
    println("Average Accuracy over 10 folds: $(round(mean(accs) * 100, digits=2))%")
end

# --- Run 10-Fold Evaluation ---
folder_path = "fruits-360/fruits-360_100x100/fruits-360/Training"
X_raw, y_raw, label_map = load_dataset(folder_path; max_per_class=100)
X_tensor, y_tensor = preprocess_data(X_raw, y_raw)
cross_validate_10fold(X_tensor, y_tensor, length(label_map))
