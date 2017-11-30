Please make sure that your pull request satisfies the following requirements:

#### If you have fixed a bug

  - [ ] Explain the fix in the pull request, and add a minimal code that reproduces the bug & the fix
  - [ ] Add the minimal code to the docstring tests if possible

#### If you have written a new method in one of the available modules

  - [ ] Document it with a docstring
  - [ ] Add a short, deterministic test in the docstring 
  - [ ] Explain briefly what the patch is about in the pull request

#### "Big" Pull requests 
  - [ ] Submit one commit per issue/change if possible. 
  - [ ] There can be many commits in a pull request, but please try to group them thematically (e.g., avoid `git commit -a`)
  - [ ] In case you have many small commits, you can always rebase them in your tree before submitting the pull request.
  - [ ] Pull requests that include merge commits won't be in general accepted
  - [ ] If you make a pull request from your master branch, you do it at your own risk (see also the point above)

#### If you have implemented a new module

  - [ ] All main methods should be documented
  - [ ] Tests should be added to the docstrings
  - [ ] The module works with as many different input file formats as possible
  - [ ] Interfacial atoms (if calculated) are marked as such, similarly to the other modules
  - [ ] The module does not change the configuration (positions, velocities,...) as loaded by MDAnalysis
  - [ ] `__init__()` calls `pytim.PatchTrajectory` 
  - [ ] you have prepared a section to be added to the online manual
  - [ ] you have prepared one or more sample jupyter notebooks
  - [ ] you have added one or more minimal example python scripts to be added to `pytim/examples/`
