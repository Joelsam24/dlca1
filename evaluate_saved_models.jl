using Flux
using Images, FileIO, Glob, MLUtils, ImageTransformations, Random, BSON, Statistics, ColorTypes

# --- Dataset Loader with validation ---
function load_dataset(folder_path::String; max_per_class::Int=50)
    classes = sort(readdir(folder_path))
    label_map = Dict(class => i for (i, class) in enumerate(classes))

    X, y = [], Int[]
    for (class, label) in label_map
        img_paths = Glob.glob("*.jpg", joinpath(folder_path, class))[1:min(max_per_class, end)]
        for img_file in img_paths
            try
                img = imresize(load(img_file), (100, 100))
                img = RGB.(img)
                arr = Float32.(permutedims(channelview(img), (2, 3, 1)))
                if size(arr) == (100, 100, 3)
                    push!(X, arr)
                    push!(y, label)
                end
            catch
                @warn "Skipped: error loading or processing $img_file"
            end
        end
    end
    return X, y, label_map
end

# --- Preprocessing ---
function preprocess_data(X, y)
    good = [i for i in 1:length(X) if size(X[i]) == (100, 100, 3)]
    X_clean = X[good]
    y_clean = y[good]

    H, W, C = 100, 100, 3
    N = length(X_clean)
    X_tensor = Array{Float32}(undef, H, W, C, N)
    for i in 1:N
        X_tensor[:, :, :, i] = copy(X_clean[i])
    end

    y_tensor = Flux.onehotbatch(y_clean, 1:maximum(y_clean))
    return X_tensor, y_tensor
end

# --- Accuracy Evaluation ---
function test_accuracy(model, X, y)
    ŷ = model(X)
    acc = mean(Flux.onecold(ŷ) .== Flux.onecold(y))
    return acc
end

# --- Load and Evaluate All Fold Models ---
function evaluate_all_saved_models(testX, testY; prefix="lenet_fold_", out="best_lenet_model.bson")
    best_acc = 0.0
    best_model = nothing
    accs = Float64[]

    for i in 1:10
        file = "$prefix$i.bson"
        try
            model = BSON.load(file)[:model]
            acc = test_accuracy(model, testX, testY)
            println("Fold $i Accuracy on test set: $(round(acc * 100, digits=2))%")
            push!(accs, acc)
            if acc > best_acc
                best_acc = acc
                best_model = model
            end
        catch e
            @warn "Failed to load or evaluate $file: $e"
        end
    end

    if best_model !== nothing
        BSON.@save out best_model
        println("\nBest model saved to \"$out\" with accuracy: $(round(best_acc * 100, digits=2))%")
    else
        println("No valid models found.")
    end

    println("Fold-wise Test Accuracies: ", round.(accs .* 100, digits=2))
end

# --- Main Execution ---
folder_path = "fruits-360/fruits-360_100x100/fruits-360/Test"
X_raw, y_raw, _ = load_dataset(folder_path; max_per_class=50)
X_test, y_test = preprocess_data(X_raw, y_raw)

evaluate_all_saved_models(X_test, y_test)
