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
