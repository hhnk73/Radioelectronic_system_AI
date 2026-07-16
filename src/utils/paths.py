from pathlib import Path


def get_project_root():
    return Path(__file__).resolve().parents[2]


def resolve_project_path(relative_path):
    return get_project_root() / relative_path
