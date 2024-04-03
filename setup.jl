using Pkg

# Install all julia packages here in local environment
Pkg.activate(".")
Pkg.instantiate()

# Ensure needed python/r packages are installed in local conda env
using CondaPkg

# Use local conda environment for IJulia
ENV["JUPYTER"] = ".CondaPkg/env/bin/jupyter"
Pkg.build("IJulia")

# Use local conda environment for RCall
ENV["R_HOME"] = "*"
Pkg.build("RCall")

