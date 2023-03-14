PLOTS_DIRNAME = "plots"

function create_experiment_directory(filename, nested_dirs)
    dir_name = PLOTS_DIRNAME * "/" * filename

    for dir in nested_dirs
        dir_name *= "/$dir"
    end

    mkpath(dir_name)
    return dir_name
end