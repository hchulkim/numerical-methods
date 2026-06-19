
path_default_nix <- "."

library(rix)

rix(date = "2025-12-29",
    r_pkgs = c("languageserver", "quarto", "knitr", "rmarkdown", "pacman", "rix", "here", "data.table", "ggplot2", "fixest", "broom", "glue","kableExtra", "bit64", "readxl", "readr", "dplyr", "lubridate", "tidyr", "R.utils", "stringr"),
    system_pkgs = "quarto",
    tex_pkgs = c("amsmath", "amsfonts", "geometry", "setspace", "threeparttable", "tools", "ulem"),
    git_pkgs = NULL,
    jl_conf = list(
                   jl_version = "1.11",
                   jl_pkgs = c(
                               "BenchmarkTools",
                               "CSV",
                               "DataFrames",
                               "Distributions",
                               "ForwardDiff",
                               "Interpolations",
                               "LinearAlgebra",
                               "NLsolve",
                               "Optim",
                               "Plots",
                               "PrettyTables",
                               "QuantEcon",
                               "Random",
                               "Roots",
                               "Statistics",
                               "StatsBase",
                               "JuMP",
                               "Ipopt",
                               "QuartoNotebookRunner"
                   )
                   ),
    ide = "none",
    project_path = path_default_nix,
    overwrite = TRUE,
    print = TRUE,
    shell_hook = "
    # Make R.nvim's vendored nvimcom package available without writing
      # to a global R library. rix's .Rprofile strips R_LIBS_USER but
      # leaves R_LIBS_SITE alone.
        export R_LIBS_SITE=\"$PWD/.Rlib:$R_LIBS_SITE\"
        mkdir -p \"$PWD/.Rlib\"
        rm -f ~/.local/share/nvim/lazy/R.nvim/nvimcom/src/apps/rnvimserver
        R CMD INSTALL --library=\"$PWD/.Rlib\" ~/.local/share/nvim/lazy/R.nvim/nvimcom

    # Point Quarto's julia engine at the nix Julia project. The
    # QuartoNotebookRunner worker activates this project, so it sees the
    # nix-provided packages (CSV, DataFrames, ...). Lets `quarto render
    # computation.qmd` work without per-call --project flags.
        export QUARTO_JULIA_PROJECT=\"$(julia -e 'print(dirname(Base.active_project()))')\"
    ")
