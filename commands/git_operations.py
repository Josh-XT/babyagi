import git
from Commands import Commands
from Config import Config

CFG = Config()

class clone_repository(Commands):
    def __init__(self):
        super().__init__()
        self.commands = {
            "Clone Repository": self.clone_repo
        }

    def clone_repo(self, repo_url: str, clone_path: str) -> str:
        split_url = repo_url.split("//")
        if CFG.GITHUB_USERNAME is not None and CFG.GITHUB_API_KEY is not None:
            auth_repo_url = f"//{CFG.GITHUB_USERNAME}:{CFG.GITHUB_API_KEY}@".join(split_url)
        else:
            auth_repo_url = "//".join(split_url)
        try:
            git.Repo.clone_from(auth_repo_url, clone_path)
            return f"""Cloned {repo_url} to {clone_path}"""
        except Exception as e:
            return f"Error: {str(e)}"