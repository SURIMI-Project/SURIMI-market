# Testing

We include a few tools for testing purpose. They are the following
| Test type | Name |
|:--|:--|
| Unit/integration test | [pytest](https://docs.pytest.org/en/stable/) |
| Static code analyser | [pyflakes](https://github.com/PyCQA/pyflakes) |
| Dynamic code analyser | [Coverage](https://coverage.readthedocs.io/en/7.14.1/) |

Unit tests primarily provide the following benefits to the Market model:
- Testing small units, ensuring that small building blocks that make larger systems work.
- Write code in a readable and cognitively comprehensive way.
- We have more confidence when refactoring code.

Pyflakes is a static analyser that help e.g. finding unused imports.

Coverage requires us to run the program and execute all possible branches. On the flip side, it provides more accurate analysis and comprehensive reports, enabling us to couple it with pytest.

### Manual Testing
Each of the tests can be executed manually with the command-line interface.

*Unit test:*
```
pytest
```

*Static analysis:*
```
pyflakes -sxvv ./src ./server
```

*Dynamic analysis:*
```
coverage run src/app.py
```

Note that with `coverage`, the `PYTHONPATH` environment variable must be configured. We can also run `coverage` with `pytest`:
```
coverage -m pytest
```

### Automated Testing
A script is provided in `scripts/test.py` that executes Coverage with PyTest. You can execute it as follows:
```
python ./scripts/test.py
```

### Test Types
We support both unit tests and integration tests. By default, the automated testing script `scripts/test.py` executes both tests. Because the two test types are separated in the filesystem, executing them separately is easy and as follows:

- Unit test:
```
pytest src/
```
- Integration test:
```
pytest -sxvv test/
```

The unit tests (found in various subdirectories under `src/`) will test targeted, critical pieces of code that as far as possible do one thing only. Meanwhile, the integration tests (found in `test/`) target the intended interactions between the end user (e.g. the Controller or the developer) and the Market service.

For integration tests, we suggest using the combined flags `-sxvv` as they give us better output and control. The behaviours are as follows:
- `-s`: print captured outputs (`print()`). Useful for print debugging.
- `-x`: stop the tests after the first failed test occurs. One test may depend on another (e.g. "update sales"-test depends on a successful initialisation).
- `-vv`: do not crop the output. This happens if the console window is too small.