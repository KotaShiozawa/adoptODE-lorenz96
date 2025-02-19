import git

def git_dir(path: str = './') -> str:
    return git.Repo().git.rev_parse('--show-toplevel')