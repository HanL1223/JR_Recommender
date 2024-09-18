# CONTRIBUTING

## How to Contribute

Note: It is recommended to create a JIRA ticket before starting to work on a new feature or bug fix. You can use the JIRA ticket number in your branch name.

1. Git clone the repository to your local machine.
2. Create a new branch: `git checkout -b new-branch-name`.
   1. Please use a meaningful name for your branch: for example, `feature/your-feature-name` or `bugfix/your-bug-name`. You can use your JIRA ticket number as well: for example, `feature/CA-1234`.
   2. Please avoid using `main` or `master` as your branch name.
3. Make your changes and commit them: `git commit -m 'Your commit message'`.
4. Push to the original branch: `git push origin project-name/your-branch-name`.
5. Create a pull request. Please make sure to provide a clear description of your changes and the reason for the changes.
6. Review the pull request and make sure all the checks are passed.
7. Merge the pull request.

## General Guidelines

- **What to Contribute**: You can contribute to the existing projects by **adding new features**, **fixing bugs**, or **improving the documentation**.
- **No Data Upload**: Please do not upload any actual data (such as notebooks with running results, or CSV files) to the repository. Such actions will put the **data security** at risk. As long as your code works with the data warehouse, other users can reproduce the results with the same code.
- **No Sensitive Information**: Please do not upload any sensitive information (such as passwords, API keys, or personal information) to the repository. Such actions will put the **data security** at risk.
- **Frequent and Small Updates**: Please make frequent and small updates to the repository in the form of commits and pull requests. Also please make sure to keep the pull requests small and focused on a single change, so that it is easier to review. For example, if you added a new line in your daily journal, you can commit and push it to the repository, and create a pull request. This will help to keep the repository up-to-date and make it easier to review the changes.

## Style Guide

1. Please follow the [Markdown Guide](https://www.markdownguide.org/) for writing documentation.
2. Please follow the [PEP 8](https://www.python.org/dev/peps/pep-0008/) and [Google Python Style Guide](https://google.github.io/styleguide/pyguide.html) for writing Python code.
3. Please follow the [Gitlab SQL Style Guide](https://handbook.gitlab.com/handbook/business-technology/data-team/platform/sql-style-guide/) for writing SQL code.

## Reporting Issues

1. Please use the [Issues](https://gitlab.com/centauri-alpha/data-training/-/issues) section of this repository to report any issues.
2. You are wellcome to create a pull request to fix the issue directly.

## Code of Conduct

- Please follow the [LICENSE](./LICENSE) of this repository.
