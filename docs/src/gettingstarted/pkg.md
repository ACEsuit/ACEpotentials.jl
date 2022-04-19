# Using the Julia Package Manager

This is a very brief introduction to the [Julia package manager](https://github.com/JuliaLang/Pkg.jl), intended for newcomers to Julia who are here primarily to use the `ACEsuit`. But it is not really ACE specific at all. 

The package manager provides functionality to organize reproducable Julia projects. A project is specified by a `Project.toml` where the user specifies which packages are required, and version bounds on those packages. The Package manager can then *resolve* these dependencies which results in a `Manifest.toml` where the full Julia environment is precisely specified. This can be used in a workflow as follows:

1. To start a new project that uses `ACE1.jl`, e.g. to develop a new interatomic potential for `TiAl` we first create a new folder where the project will live, e.g., `ace1_TiAl_project`. Change to that folder and start the Julia REPL. Type `]` to switch to the package manager, then *activate* a new project in the current directory via `activate .`

2. You now have an empty project. Start adding the packages you need, e.g., 
    ```
    add ACE1, JuLIP, IPFitting
    ```
Type `status` to see your required packages listed. (Note this is only a subset of the installed packages!). Exit the REPL and type `ls`; you will then see a new file `Project.toml` which lists the project requirements, and a `Manifest.toml` which lists the actually packages and the version that have been installed.

3. Specify version bounds: Open `Project.toml` in an editor and under the [compat] section you can now add version bounds, e.g. ACE1 = "0.9, 0.10". Please see the [Pkg.jl docs](https://pkgdocs.julialang.org/dev/compatibility/) for details on how to specify those bounds. Start a Julia REPL again, type `]` to switch to the package manager and then `up` to up- or down-grade all installed packages to the latest version compatible with your bounds.

#### Notes 

* The `Project.toml` should always be committed to your project git repo. Whether `Manifest.toml` is also committed is a matter of taste or context. Tracking the Manifest will (normally) ensure future compatibility since it allows you to reconstruct the precise Julia environemt that was used when the Manifest was created.
* If you are a user rather than developer it should almost never be required for you to check out a package (or, `dev` it in the package manager). When we (the developers) make changes to - say - `ACE1.jl` we almost always immediately tag another version and then you can adjust your version bounds in your project to update as well as enforce which version to use.

#### Links 

* https://pkgdocs.julialang.org/v1/
* https://pkgdocs.julialang.org/v1/compatibility/


