# distil-primitives

## Distil AutoML primitives

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

## D3M Annotations

Annotations for each primitive must be generated and submitted to the [D3M primitives repository](https://gitlab.com/datadrivendiscovery.org.primitives) in order for them to be built into D3M primitive library image. Generation of the annotations is done by running the following:

```
python3 generate_annotations.py
```
The generated annotations will be written to the `annotations` directory by default.

