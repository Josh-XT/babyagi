import git
import os
from github import Github
from Commands import Commands
from Config import Config

CFG = Config()

class clone_repository(Commands):
    def __init__(self):
        super().__init__()
        if CFG.GITHUB_USERNAME is None and CFG.GITHUB_API_KEY is None:
            self.commands = {
                "Clone Repository": self.clone_repo,
                "Create Repository": self.create_repo
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

    def create_repo(self, repo_name: str, readme: str) -> str:
        g = Github(CFG.GITHUB_API_KEY)
        user = g.get_user(CFG.GITHUB_USERNAME)
        repo = user.create_repo(repo_name, private=True)
        repo_url = repo.clone_url
        repo_dir = f"./{repo_name}"
        repo = git.Repo.init(repo_dir)
        with open(f"{repo_dir}/README.md", 'w') as f:
            f.write(readme)
        repo.git.add(A=True)
        repo.git.commit(m="Added README")
        origin = repo.create_remote("origin", repo_url)
        repo.git.push("origin", "HEAD:main")
        return repo_url