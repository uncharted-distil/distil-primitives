# distil-primitives

Distil AutoML primitives

A repository of D3M compliant primitives used in baseline Distil AutoML pipelines.

To install for local development from the project root first run:

```console
pip install -r build_requirements.txt
```

Then to install the primitive source **without** GPU support run:

```
pip install -e .[cpu]
```

To install **with** GPU suport run:

```console
pip install -e .[gpu]
```

Note that if you are using zshell you will need to include a `\` before the brackets.
