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

def get_repo_info(for_file_changes: bool = False) -> RepoInfo:
    """Get the repository URL and name from git remote.
    
    Args:
        for_file_changes: If True, always use current directory.
                         If False, try current directory first, then fall back to idvorkin/nlp
                         Used when getting URLs to source files like think.py or changes.py
    
    Returns:
        RepoInfo containing the repo URL and name in format 'owner/repo'
    """
    try:
        import subprocess
        import os
        
        # Use the current working directory
        cwd = os.getcwd()
        
        result = subprocess.run(
            ["git", "remote", "get-url", "origin"],
            capture_output=True,
            text=True,
            cwd=cwd
        )
        
        if result.returncode != 0:
            if for_file_changes:
                raise Exception("Not in a git repository")
            return RepoInfo(
                url="https://github.com/idvorkin/nlp",
                name="idvorkin/nlp"
            )

        repo_url = result.stdout.strip()
        
        if repo_url.startswith("https"):
            base_path = repo_url.split("/")[-2] + "/" + repo_url.split("/")[-1]
        elif repo_url.startswith("git@"):
            base_path = repo_url.split(":")[1]
            base_path = base_path.replace(".git", "")
        
        # Clean up the repo URL to always return https format
        if repo_url.startswith("git@"):
            repo_url = f"https://github.com/{base_path}"
            
        # If not for file changes and not in nlp repo, default to nlp repo
        if not for_file_changes and "idvorkin/nlp" not in base_path:
            return RepoInfo(
                url="https://github.com/idvorkin/nlp",
                name="idvorkin/nlp"
            )
            
        return RepoInfo(url=repo_url, name=base_path)
        
    except Exception as e:
        ic(f"Failed to get repo info: {e}")
        if for_file_changes:
            raise  # Re-raise for file changes as we need a valid repo
        # Default to idvorkin/nlp repo for source file URLs
        return RepoInfo(
            url="https://github.com/idvorkin/nlp",
            name="idvorkin/nlp"
        )
