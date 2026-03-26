# neurodesign-plus

`neurodesign-plus` is an extended and maintained fork of the original [neurodesign](https://github.com/neuropower/neurodesign) package for fMRI experimental design optimisation.

In addition to the base package workflow, `neurodesign-plus` adds support for:

1. fixed user-defined event orders via `order`,
2. variable stimulus durations via `stimuli_durations`,
3. transition-specific ITIs via `conditional_ITI`,
4. probabilistic template-based ordering via `order_keys`, `order_probabilities`, and `order_length`.

## Documentation

- Base package documentation: [neurodesign on Read the Docs](http://neurodesign.readthedocs.io/en/latest/)
- `neurodesign-plus` documentation: [neurodesign-plus on Read the Docs](https://neurodesign-plus.readthedocs.io/en/latest/)

## Repository Layout

```text
docs/        Source files for the Read the Docs documentation
manuals/     Longer-form markdown documentation included by the docs site
neurodesign/ Python package source code
tests/       Automated tests for package behavior
tutorials/   Jupyter tutorials for the main workflow, base functions, and new features
```

## Tutorials

The tutorial collection includes:

- three overview tutorials in `tutorials/`,
- four base-function tutorials in `tutorials/base_functions/`,
- four feature-focused tutorials in `tutorials/new_functions/`.

The Read the Docs tutorials page is the best place to browse the current notebook set and their intended learning progression.

## Credits

This project is a fork of the original **Neurodesign** package.

- Original author: [Neuropower Team](https://github.com/neuropower)
- Primary refactoring, extensions, and tutorials: Atharv Amar Umap (Social Learning and Decisions Lab, UMD) <aumap@terpmail.umd.edu>
- Supervision, design guidance, and tutorials: Valentin Guigon (Social Learning and Decisions Lab, UMD) <vguigon@umd.edu>
