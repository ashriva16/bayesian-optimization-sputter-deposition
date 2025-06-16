# Bayesian optimization of sputter-deposition

<p align="left">
  <a href="https://opensource.org/licenses/MIT">
    <img alt="License: MIT" src="https://img.shields.io/badge/License-MIT-yellow.svg" />
  </a>
  <img alt="Python Version" src="https://img.shields.io/badge/python-3.8%2B-blue" />
  <img alt="Repo Size" src="https://img.shields.io/github/repo-size/ashriva16/{{ cookiecutter.project_name }}" />
  <img alt="Last Commit" src="https://img.shields.io/github/last-commit/ashriva16/{{ cookiecutter.project_name }}" />
  <a href="https://github.com/ashriva16/{{ cookiecutter.project_name }}/issues">
    <img alt="Issues" src="https://img.shields.io/github/issues/ashriva16/{{ cookiecutter.project_name }}" />
  </a>
  <a href="https://github.com/ashriva16/{{ cookiecutter.project_name }}/pulls">
    <img alt="Pull Requests" src="https://img.shields.io/github/issues-pr/ashriva16/{{ cookiecutter.project_name }}" />
  </a>
</p>

## 📌 Project Description

This repository presents a **Bayesian optimization framework** for guiding the **sputter deposition of molybdenum thin films**, targeting optimal **residual stress** and **sheet resistance**, while minimizing sensitivity to stochastic process variations.
Key deposition parameters — **power**, **pressure**, and **working distance** — influence these properties.
We apply **Bayesian optimization** to efficiently search the process space using a **custom objective function** that incorporates:

- Empirical stress and resistance data
- Prior knowledge about pressure-dependent variability

### ✅ Key Features

- Rapid identification of optimal deposition parameters
- Improved consistency and reproducibility of thin film properties
- Reduced experimental effort

Our results confirm that Bayesian optimization is a powerful tool for thin film process development, delivering high-performance films with controlled stress and resistance characteristics.

---

## 🧱 Project Structure

```text
.
├── CHANGELOG.md        # Chronologically tracks added, changed, fixed, or removed features
├── docs/               # Sphinx or MkDocs-based documentation (API, usage, design, papers, etc.)
├── environment.yml     # Conda environment specification for reproducibility
├── LICENSE             # Licensing information (e.g., MIT, Apache 2.0)
├── Makefile            # Automation commands (e.g., setup, test, lint, build)
├── playground/         # Prototyping area for experiments, quick tests, or notebooks (not production)
├── pyproject.toml      # Project metadata and build config (PEP 621, setuptools, linting tools)
├── README.md           # Project overview, usage, setup, and contribution guidelines
├── pvd_exp_demo/            # scripts to demonstrate bayesopt behavior
├── pvd_exp_demo/            # scripts used for during experiment design
├── utils/              # Shared utility functions and helper modules used across the project
└── VERSION             # Plain text file holding the current version of the project (e.g., 0.1.0)
```

---

## 🧩 Packaging

This project uses **PEP 621-compliant** configuration via `pyproject.toml` with setuptools.

Only `utils` and submodules under `utils/` are included as installable packages by default. To include more:

```toml
[tool.setuptools.packages.find]
where = ["."]
include = ["utils", "utils.*", "src", "src.*", "common", "common.*"]
```

---

Badge (once setup):

```markdown
[![CI](https://github.com/ashriva16/bayesian-optimization-sputter-deposition/actions/workflows/ci.yml/badge.svg)](https://github.com/ashriva16/bayesian-optimization-sputter-deposition/actions)
```

---

## 👤 Maintainer

**Ankit Shrivastava**
Feel free to open an issue or discussion for support.

---

## 📜 License

This project is licensed under the [MIT License](https://opensource.org/licenses/MIT). See the `LICENSE` file for full details.

---

## 📈 Project Status

> Status: 🚧 In Development — Not ready for use.

---

## 📘 References

- [Cookiecutter Docs](https://cookiecutter.readthedocs.io)
- [PEP 621](https://peps.python.org/pep-0621/)
- [GitHub Actions](https://docs.github.com/en/actions)
