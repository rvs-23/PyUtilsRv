# Python Utilities Project

A curated collection of reusable Python utility functions for common tasks in data science, data wrangling, machine learning, and visualization. This project aims to grow into a robust toolkit to streamline data-driven workflows.

## Overview

This project, named `python-utilities` for distribution and typically imported as `utilities`, provides a set of well-organized and tested helper functions. The goal is to build a personal library that can be easily integrated into various data science and software engineering projects, reducing boilerplate code and promoting consistency.

## Modules (Current & Planned)

Currently, the utilities are organized into the following modules (located under `src/utilities/`):

* **`dw_utils` (Data Wrangling Utilities):** Functions for data cleaning, transformation, information extraction (e.g., `get_df_info`, `find_problematic_cols_df`, `common_cols_by_name_bw_dfs`, `get_common_elements`).
* **`viz_utils` (Visualization Utilities):** Helpers for generating common plots (e.g., `hist_distribution`).
* **`ml_utils` (Machine Learning Utilities):** (Planned) Functions to assist with model training, evaluation, preprocessing specific to ML tasks.
* *(More modules can be added as the library grows)*

## Project Goals and Roadmap

* **Comprehensive:** Cover a wide range of common utility needs.
* **Well-Tested:** Ensure reliability through thorough unit testing with `pytest`.
* **Well-Documented:** Provide clear docstrings for all functions and a helpful README.
* **Modern Tooling:** Utilize modern Python development tools like `uv` for environment and package management, `ruff` for linting/formatting, and `pyproject.toml` for packaging.
* **Evolving:** Continuously add new utilities and improve existing ones based on practical needs.

## Getting Started

Follow these instructions to set up the project locally for development and use.

### Prerequisites

* Python (>=3.13, as specified in `pyproject.toml`)
* `uv` (recommended for package and environment management, can use `pip` as an alternative)

### Installation

1.  **Clone the Repository (if applicable):**
    If this project were hosted on Git:
    ```bash
    git clone <repository_url>
    cd PyUtilsRv
    ```
    For now, you are working in your local `PyUtilsRv` directory.

2.  **Create and Activate a Virtual Environment:**
    It's crucial to work within a virtual environment & install dependencies there. Fortunately, uv has made this task simple. From the root of the `PyUtilsRv` project directory:
    ```bash
    # Using uv
    uv sync
    ```

3.  **Install Project in Editable Mode with Test Dependencies:**
    This step installs the `utilities` package itself (making it importable) along with dependencies needed for running tests (like `pytest`).
    From the root of the `PyUtilsRv` project directory (where `pyproject.toml` is):
    ```bash
    uv pip install -e .[test]
    ```

## Development

### Project Structure

This project follows a standard `src`-layout:
.
├── build
│   ├── bdist.macosx-11.0-arm64
│   └── lib
│       └── utilities
│           ├── __init__.py
│           ├── dw_utils
│           │   ├── __init__.py
│           │   └── utils.py
│           ├── ml_utils
│           │   └── __init__.py
│           └── viz_utils
│               ├── __init__.py
│               └── distribution_plots.py
├── pyproject.toml
├── README.md
├── src
│   ├── python_utilities.egg-info
│   │   ├── dependency_links.txt
│   │   ├── PKG-INFO
│   │   ├── requires.txt
│   │   ├── SOURCES.txt
│   │   └── top_level.txt
│   └── utilities
│       ├── __init__.py
│       ├── dw_utils
│       │   ├── __init__.py
│       │   └── utils.py
│       ├── ml_utils
│       │   └── __init__.py
│       └── viz_utils
│           ├── __init__.py
│           └── distribution_plots.py
├── tests
│   ├── dw_utils
│   │   ├── __pycache__
│   │   │   └── test_utils.cpython-313-pytest-8.3.5.pyc
│   │   └── test_utils.py
│   ├── ml_utils
│   └── viz_utils
│       ├── __pycache__
│       │   └── test_distribution_plots.cpython-313-pytest-8.3.5.pyc
│       └── test_distribution_plots.py
└── uv.lock


### Dependency Management with `uv`

* Dependencies are defined in `pyproject.toml`.
* To add a new dependency: `uv add <package_name>`
* To install dependencies from `pyproject.toml`: `uv sync`
* For editable install of the current project: `uv pip install -e .`

### Linting and Formatting with Ruff

* **Ruff** is used for linting and formatting, configured in `pyproject.toml`.
* It's set up for Black-like formatting and minimal type hint warnings (by ignoring most `ANN` rules).
* **To check for linting issues:**
    ```bash
    ruff check .
    ```
* **To automatically fix linting issues (including formatting):**
    ```bash
    ruff check . --fix
    ```
* **To format code:**
    ```bash
    ruff format .
    ```
* It's recommended to integrate Ruff with your code editor (e.g., VS Code) for real-time feedback and format-on-save.

## Testing with Pytest

### Writing Tests

* Tests are located in the `tests/` directory, with a structure mirroring the `src/utilities/` directory.
* Test files are named `test_*.py` (e.g., `test_utils.py`).
* Test functions within these files are named `test_*()` (e.g., `def test_basic_scenario():`).
* The **Arrange-Act-Assert** pattern is followed:
    1.  **Arrange:** Set up input data and any necessary conditions.
    2.  **Act:** Call the function being tested.
    3.  **Assert:** Verify the output or behavior is as expected using `assert` statements.
* Use `pytest.raises` to test for expected exceptions.
* Use fixtures (`@pytest.fixture`) to create reusable setup code (e.g., sample DataFrames).

### Running Tests

Ensure your virtual environment is activated and you are in the root directory of the `python-utilities` project.

* **Run all tests:**
    ```bash
    pytest
    ```
* **Run tests in a specific file:**
    ```bash
    pytest tests/dw_utils/test_utils.py
    ```
* **Run a specific test function:**
    ```bash
    pytest tests/dw_utils/test_utils.py::test_basic_scenario
    ```
* **Verbose output:**
    ```bash
    pytest -v
    ```

## Key Learnings and Common Fixes Encountered

This section documents some of the setup and development hurdles overcome during the initial phase of this project.

### 1. Import Resolution (Editable Installs & Pylance)

* **Problem:** Pylance (VS Code) or `pytest` not resolving imports like `from utilities.dw_utils.utils import ...`.
* **Fix:**
    1.  **Editable Install:** The primary solution is to install the project in editable mode from its root directory (`cd python-utilities && uv pip install -e .`). This makes the `utilities` package (defined in `pyproject.toml` and sourced from `src/utilities`) available on the Python path for the active virtual environment.
    2.  **VS Code Interpreter:** Ensure VS Code is using the correct Python interpreter from the project's virtual environment.
    3.  **VS Code Workspace:**
        * Opening the `python-utilities` folder directly as the VS Code workspace root is often simplest.

### 2. Virtual Environment Path Issues

* **Problem:** `uv` (or `pip`) and other tools stop working if the project's root folder name is changed after the virtual environment (`.venv`) was created, because paths within the venv become invalid.
* **Fix:** The safest and quickest solution is to:
    1.  Delete the old `.venv` folder.
    2.  Recreate it: `uv venv`
    3.  Reinstall dependencies: `uv pip install -e .[test]` (or `uv sync` / `uv pip install -r requirements.txt`).

### 3. Build Errors with `uv pip install -e .` (Flat vs. Src Layout)

* **Problem:** Initial attempts to install with a "flat layout" (where utility modules like `data_wrangling_utils` were at the same level as `pyproject.toml`) resulted in a setuptools error: `Multiple top-level packages discovered...`.
* **Fix:** Adopted the `src`-layout:
    * Created `src/utilities/`.
    * Moved all utility modules (e.g., `dw_utils`, `viz_utils`) into `src/utilities/`.
    * Added an `__init__.py` to `src/utilities/`.
    * Configured `pyproject.toml` to find packages in `src/`:
        ```toml
        [tool.setuptools.packages.find]
        where = ["src"]
        include = ["utilities*"]
        ```
    This clearly defines `utilities` as the single top-level package to be installed from the `src` directory.

