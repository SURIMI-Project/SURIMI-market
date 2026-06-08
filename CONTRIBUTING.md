# Contributing to the SURIMI Market Model

This document serves as a guideline for contributing to the project. It is a living document, meaning you might find something from the past that don't align with the current guidelines.

## Submission Guidelines
### General Info About Commits
Commits should address specific problems (so-called atomic commits) and maintain good "commit hygiene". While one problem is preferable, this is not always possible. Techniques like breaking down issues to smaller chunks can help. As a rule of thumb, it should be easy to look at the commit at a glance and understand what were changed. This provides a few benefits, e.g.:

- Problematic commits can quickly be identified.
- Reverts are as safe as possible to revert. While technical debt is inevitable, cleaning up after the revert will be more manageable.
- The commit history becomes cleaner and easier to manage.

You can look read more about commit hygiene in the article of [*"Developer Git Commit Hygiene"*](https://dzone.com/articles/developer-git-commit-hygiene-1) by Shekhar and Moka (2024). We will bring to light the most important aspects to the project here.

Avoid refactoring existing code with e.g. extra whitespace and inconsistent indentations of function parameters. Insignificant changes should be avoided to finding bugs with e.g. "blame" less challenging and maintains a clean and significant git history.

### Commit Messages
As part of an ecosystem where interconnected components in constant development, we strive for code maintainability. In addition to atomic commits, git message formats help a great portion in reading commits at a glance. We use the following guidelines:

1. Leave a blank line between the subject and the body.
This helps git log and git rebase to work nicely.
2. Try to keep the subject line below 72 characters, ideally 50. The subject line acts as a summary of the commit.
3. Prefer to not capitalize the subject line.
4. Do not end the subject line with a period.
5. Use the imperative mood in the subject line (e.g., `fix: CLI args correctly display descriptions` and `feat: market model can read historical price data`).
6. Use the commit body to further describe what the commit addresses, reasons behind decisions, etc.. Wrap the commit body at 80 characters.
7. Use semantic line feeds to separate different ideas, improving readability.
8. Be concise but thorough: If significant alternative solutions were available, explain why they were discarded.
9. Reference related discussions or issues: If the commit refers to a topic discussed, an issue, fixes a regression test or similar, provide the link. If it fixes a compiler error, include a minimal verbatim message of the compiler error. If the commit closes an issue, use the [GitHub issue closing keywords](https://docs.github.com/en/issues/tracking-your-work-with-issues/using-issues/linking-a-pull-request-to-an-issue).

The title of a commit message should start with one of the following tags:
| Commit Type | Title                    | Description                                                                                                 |
|:-----------:|--------------------------|-------------------------------------------------------------------------------------------------------------|
|   `feat`    | Features                 | Introduce a new feature.                                                                                    |
|    `fix`    | Bug Fixes                | Address a problem, bug, etc..                                                                               |
|   `docs`    | Documentation            | Documentation only changes. This may also include comments/docstrings.                                      |
|   `style`   | Styles                   | Changes that do not affect the meaning of the code (white-space, formatting, missing semi-colons, etc).     |
| `refactor`  | Code Refactoring         | A code change that neither fixes a bug nor adds a feature.                                                  |
|   `perf`    | Performance Improvements | A code change that improves performance.                                                                    |
|   `test`    | Tests                    | Adding missing tests or correcting existing tests.                                                          |
|   `build`   | Builds                   | Changes that affect the build system or external dependencies (example scopes: pip, pipx).                  |
|    `ci`     | Continuous Integrations  | Changes to CI configuration files and scripts.                                                              |
|   `api`     | gRPC Client Changes      | Changes to files in e.g. [`grpc_requests`](./grpc_requests/) that are used by external clients, e.g. [Bruno](https://www.usebruno.com/). |
|  `revert`   | Reverts                  | Reverts a previous commit.                                                                                  |
|   `chore`   | Chores                   | Other changes that don't modify src or test files.                                                          |

Example commit message:

> We address a problem in a function should accept real numbers, but instead accepts integers only.

<!-- The following example uses C, as it examplifies the add() situation easier than using Python's type hints. -->

Original code:
```c
float add(int a, int b) {
    return a + b;
}
```

The above becomes the following:
```c
float add(float a, float b) {
    return a + b;
}
```

This change can be summarised with the below commit. Although unecessary in this specific situation, the commit body is used to provide an example:

```
fix: add() correctly accepts floats instead of ints

When two integers are summarised in C, the resulting data type is also an
integer. Changing to type float correctly returns type float.
```


## Coding Style
We follow the Python naming convention. The following applies:

| Keyword | Case | Example |
|:--|:--|:--|
| File names | `CamelCase` for class files, otherwise `snake_case` | `ValueError.py or utils.py` |
| Variables | `snake_case` | `my_list = []` |
| Constants | `majuscule` | `PI = 3.14` |
| Functions | `snake_case` | `def my_func` |
| Classes | `CamelCase` | `MarketModel` |
