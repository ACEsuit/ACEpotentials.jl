# Using the Julia Package Manager

This is a very brief introduction to the [Julia package manager](https://github.com/JuliaLang/Pkg.jl), intended for newcomers to Julia who are here primarily to use the `ACEsuit`. But it is not really ACE specific at all. 

```@raw html
<!-- If you plan to use `ACEpotentials.jl` from Python or the command line, then you need not read this. -->
```

The package manager provides functionality to organize reproducable Julia projects. A project is specified by a `Project.toml` where the user specifies which packages are required, and version bounds on those packages. The Package manager can then *resolve* these dependencies which results in a `Manifest.toml` where the full Julia environment is precisely specified. This can be used in a workflow as follows:

1. To start a new project that uses `ACEpotentials.jl`, e.g. to develop a new interatomic potential for `TiAl` we first create a new folder where the project will live, e.g., `ace_TiAl_project`. Change to that folder and start the Julia REPL. Type `]` to switch to the package manager, then *activate* a new project in the current directory via `activate .`

2. You now have an empty project. Start adding the packages you need, often just 
    ```
    add ACEpotentials
    ```
    will suffice. 
    Type `status` to see your required packages listed. (Note this is only a subset of the installed packages!). Exit the REPL and type `ls`; you will then see a new file `Project.toml` which lists the project requirements, and a `Manifest.toml` which lists the actually packages and the version that have been installed.

3. Specify version bounds: Open `Project.toml` in an editor and under the [compat] section you can now add version bounds, e.g. `ACEpotentials = "0.6.1"` following semver. Please see the [Pkg.jl docs](https://pkgdocs.julialang.org/dev/compatibility/) for details on how to specify those bounds. Start a Julia REPL again, type `]` to switch to the package manager and then `up` to up- or down-grade all installed packages to the latest version compatible with your bounds.

#### Using a Development Branch (rarely required)

If you are a user rather than developer it should almost never be required for you to check out a package (or, `dev` it in the package manager). When developers make changes to - say - `ACEpotentials.jl` they will always immediately tag another version and then you can adjust your version bounds in your project to update as well as enforce which version to use. However a developer would frequently do this, and occasionally it might be required when iterating between a user and developer for testing. There are multiple ways to achieve this; the following is our recommended procedure: 

Suppose for example that a development branch `co/dev` of `ACE1.jl` is needed in a project `project`. Then one should perform the following steps: 
* Make sure `ACE1` has been added to `project/Project.toml` 
* In a separate folder, `/path/to/` , clone `ACE1.jl`
```
cd /path/to
git clone git@github.com:ACEsuit/ACE1.jl.git
git checkout co/dev
```
so that the repo will now live in `/path/to/ACE1.jl`
* Go to and activate `project`, then in a Julia REPL switch to the package manager `]` and execute
```
dev /path/to/ACE1.jl
```
This will replace the `ACE1` package in the Manifest with the version that lives in `/path/to/ACE1.jl` 

Later on, when you want to go back to the standad Pkg versin control you can simply `free ACE1`.

#### Further Notes 

* The `Project.toml` should always be committed to your project git repo. Whether `Manifest.toml` is also committed is a matter of taste or context. Tracking the Manifest will (normally) ensure future compatibility since it allows you to reconstruct the precise Julia environment that was used when the Manifest was created.

#### Links 

* https://pkgdocs.julialang.org/v1/
* https://pkgdocs.julialang.org/v1/compatibility/


