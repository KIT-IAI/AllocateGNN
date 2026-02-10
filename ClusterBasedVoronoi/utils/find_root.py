"""
Author: Xuanhao Mu
Email: xuanhao.mu@kit.edu
Description: This module contains functions for finding the project root directory.
"""


from pathlib import Path

def find_project_root(current_dir: Path, target_files=('requirements.txt', '.git', 'pyproject.toml')):

    for target_file in target_files:
        if (current_dir / target_file).exists():
            return current_dir
    if current_dir.parent == current_dir:

        raise FileNotFoundError("Project root not found.")
    return find_project_root(current_dir.parent, target_files)