#!python3

import requests
from icecream import ic
from pydantic import BaseModel

class RepoInfo(BaseModel):
    """Information about a GitHub repository"""
    url: str
    name: str

def get_latest_github_commit_url(repo: str, file_path: str) -> str:
    """Get the URL to the latest commit version of a file in a GitHub repo.
    
    Args:
        repo: Repository name in format 'owner/repo'
        file_path: Path to file within the repository
        
    Returns:
        URL to the latest commit version of the file, or main branch as fallback
    """
    try:
        api_url = f"https://api.github.com/repos/{repo}/commits?path={file_path}&page=1&per_page=1"
        response = requests.get(api_url)
        response.raise_for_status()

        commits = response.json()
        if not commits:  # Handle empty response
            ic(f"No commits found for {file_path}")
            return f"https://github.com/{repo}/blob/main/{file_path}"

        latest_commit = commits[0]
        commit_sha = latest_commit["sha"]
        return f"https://github.com/{repo}/blob/{commit_sha}/{file_path}"
    except Exception as e:
        ic(f"Failed to get latest GitHub commit URL for {file_path}:", e)
        return f"https://github.com/{repo}/blob/main/{file_path}"  # Fallback

def get_repo_info() -> RepoInfo:
    """Get the repository URL and name from git remote.
    
    Returns:
        RepoInfo containing the repo URL and name in format 'owner/repo'
    """
    import subprocess
    
    result = subprocess.run(
        ["git", "remote", "get-url", "origin"], capture_output=True, text=True
    )

    repo_url = result.stdout.strip()
    base_path = "Unknown"
    if repo_url.startswith("https"):
        base_path = repo_url.split("/")[-2] + "/" + repo_url.split("/")[-1]
    elif repo_url.startswith("git@"):
        base_path = repo_url.split(":")[1]
        base_path = base_path.replace(".git", "")
    return RepoInfo(url=repo_url, name=base_path)
