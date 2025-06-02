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
    cd python-utilities
    ```
    For now, you are working in your local `python-utilities` directory.

2.  **Create and Activate a Virtual Environment:**
    It's crucial to work within a virtual environment. From the root of the `python-utilities` project directory:
    ```bash
    # Using uv
    uv venv
    source .venv/bin/activate  # On macOS/Linux
    # .venv\Scripts\Activate.ps1 # On Windows PowerShell
    # .venv\Scripts\activate.bat   # On Windows CMD
    ```

3.  **Install Project in Editable Mode with Test Dependencies:**
    This step installs the `utilities` package itself (making it importable) along with dependencies needed for running tests (like `pytest`).
    From the root of the `python-utilities` project directory (where `pyproject.toml` is):
    ```bash
    uv pip install -e .[test]
    ```
    *(If you don't have a `[test]` extra defined in `pyproject.toml` yet, you might install `pytest` separately: `uv pip install pytest` after `uv pip install -e .`)*

    Your `pyproject.toml` should define `pytest` as a dependency, ideally under `[project.optional-dependencies]`:
    ```toml
    # pyproject.toml example snippet
    [project]
    name = "python-utilities" # Distribution name
    # ... other project metadata ...

    [project.optional-dependencies]
    test = [
        "pytest>=7.0" # Or your desired version
    ]
    ```

## Development

### Project Structure

This project follows a standard `src`-layout: