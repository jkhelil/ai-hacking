#!/usr/bin/env python3
import argparse
import logging
import os
import sys
import json
import datetime
from typing import List, Dict, Optional, Any, Union
import requests

# Try to import OpenAI, with graceful fallback if not installed
try:
    import openai
    import httpx
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

# ========== Logging Setup ==========
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger("ReleaseNoteGen")

# ========== GitHub API Operations ==========
class GitHubAPI:
    """Class to interact with GitHub API."""
    
    def __init__(self, owner: str, repo: str, token: Optional[str] = None):
        """Initialize with GitHub repository details."""
        self.owner = owner
        self.repo = repo
        self.base_url = f"https://api.github.com/repos/{owner}/{repo}"
        self.headers = {
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": "2022-11-28"
        }
        if token:
            self.headers["Authorization"] = f"Bearer {token}"
    
    def get_commits_between(self, from_ref: str, to_ref: str) -> List[Dict[str, Any]]:
        """Get commits between two references."""
        logger.info(f"Fetching commits between {from_ref} and {to_ref}")
        
        # First, get the full commit SHA for both refs
        from_sha = self._get_sha_for_ref(from_ref)
        to_sha = self._get_sha_for_ref(to_ref)
        
        if not from_sha or not to_sha:
            raise ValueError(f"Could not resolve references: {from_ref}, {to_ref}")
        
        # Get comparison data
        url = f"{self.base_url}/compare/{from_sha}...{to_sha}"
        response = requests.get(url, headers=self.headers)
        response.raise_for_status()
        comparison = response.json()
        
        # Process commits
        commits = []
        for commit in comparison.get("commits", []):
            commit_data = {
                "sha": commit["sha"][:7],  # Short SHA
                "message": commit["commit"]["message"].split("\n")[0],  # First line only
                "author": commit["commit"]["author"]["name"],
                "date": commit["commit"]["author"]["date"],
                "html_url": commit["html_url"]
            }
            commits.append(commit_data)
        
        logger.info(f"Found {len(commits)} commits between {from_ref} and {to_ref}")
        return commits
    
    def _get_sha_for_ref(self, ref: str) -> Optional[str]:
        """Get the full SHA for a reference (branch, tag, commit)."""
        # First try as a branch
        url = f"{self.base_url}/branches/{ref}"
        response = requests.get(url, headers=self.headers)
        if response.status_code == 200:
            return response.json()["commit"]["sha"]
        
        # Then try as a tag
        url = f"{self.base_url}/git/refs/tags/{ref}"
        response = requests.get(url, headers=self.headers)
        if response.status_code == 200:
            return response.json()["object"]["sha"]
        
        # Finally, assume it's a commit SHA and validate
        if len(ref) >= 4:  # At least 4 chars of the SHA
            url = f"{self.base_url}/commits/{ref}"
            response = requests.get(url, headers=self.headers)
            if response.status_code == 200:
                return response.json()["sha"]
        
        return None
    
    def get_latest_commit(self, branch: str = "main") -> Optional[Dict[str, Any]]:
        """Get the latest commit for a branch."""
        url = f"{self.base_url}/commits/{branch}"
        response = requests.get(url, headers=self.headers)
        if response.status_code == 200:
            commit_data = response.json()
            return {
                "sha": commit_data["sha"],
                "message": commit_data["commit"]["message"].split("\n")[0],
                "author": commit_data["commit"]["author"]["name"],
                "date": commit_data["commit"]["committer"]["date"]
            }
        return None
    
    def get_commit_from_days_ago(self, days: int, branch: str = "main") -> Optional[Dict[str, Any]]:
        """Get a commit from X days ago on a branch."""
        # Calculate the date
        target_date = (datetime.datetime.now() - datetime.timedelta(days=days)).strftime("%Y-%m-%d")
        
        # Get commits until the target date
        url = f"{self.base_url}/commits"
        params = {
            "sha": branch,
            "until": target_date,
            "per_page": 1
        }
        response = requests.get(url, headers=self.headers, params=params)
        if response.status_code == 200 and response.json():
            commit_data = response.json()[0]
            return {
                "sha": commit_data["sha"],
                "message": commit_data["commit"]["message"].split("\n")[0],
                "author": commit_data["commit"]["author"]["name"],
                "date": commit_data["commit"]["committer"]["date"]
            }
        return None

# ========== AI Interaction ==========
def generate_release_notes(commits: List[Dict[str, Any]], model: str, api_key: str, 
                          api_base_url: str, org_name: Optional[str] = None) -> str:
    """Generate release notes using AI based on commit data."""
    if not OPENAI_AVAILABLE:
        logger.error("OpenAI libraries not installed. Please install with: pip install openai httpx")
        sys.exit(1)
    
    # Format commits for the prompt
    formatted_commits = "\n".join([
        f"- {commit['sha']}: {commit['message']} ({commit['author']})" 
        for commit in commits
    ])
    
    prompt = f"""
You are a release note assistant.
Generate a professional and concise release note from the following list of git commit messages.
Group related items into these categories:
- 🚀 New Features
- 🐛 Bug Fixes
- 📚 Documentation
- 🔧 Maintenance
- ⚙️ Performance
- 🧪 Testing

Format the output in Markdown with the following rules:
1. DO NOT include a title like "# Release Notes" or "# Changelog" - this will be added separately
2. Use the emoji bullet points exactly as shown above
3. For each item, include the commit hash in backticks like: `{{commit['sha']}}`: Commit message (Author name)
4. Skip any categories that have no entries
5. Keep the format consistent with the emoji followed by the category name and a colon

Commit log:
{formatted_commits}
"""
    try:
        logger.info(f"Using AI model: {model}")
        client_kwargs = {
            "base_url": api_base_url,
            "api_key": api_key,
            "http_client": httpx.Client(verify=False)
        }
        
        if org_name:
            client_kwargs["organization"] = org_name
            
        client = openai.OpenAI(**client_kwargs)
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are a helpful assistant that creates professional release notes."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3,
            max_tokens=1000,
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        logger.error(f"AI generation failed: {str(e)}")
        raise

def save_release_notes(release_notes: str, output_file: Optional[str] = None) -> None:
    """Save release notes to a file if output_file is provided."""
    if output_file:
        os.makedirs(os.path.dirname(os.path.abspath(output_file)), exist_ok=True)
        
        # Add a title and date to the release notes file
        today = datetime.datetime.now().strftime("%Y-%m-%d")
        titled_notes = f"# Release Notes {today}\n\n{release_notes}"
        
        with open(output_file, 'w') as f:
            f.write(titled_notes)
        logger.info(f"Release notes saved to {output_file}")

def parse_github_url(repo_url: str) -> tuple:
    """Parse GitHub URL to get owner and repo name."""
    # Handle different URL formats
    if repo_url.startswith("https://github.com/"):
        parts = repo_url.replace("https://github.com/", "").strip("/").split("/")
    elif repo_url.startswith("git@github.com:"):
        parts = repo_url.replace("git@github.com:", "").replace(".git", "").strip("/").split("/")
    else:
        raise ValueError(f"Invalid GitHub URL format: {repo_url}")
    
    if len(parts) < 2:
        raise ValueError(f"Could not extract owner and repo from URL: {repo_url}")
    
    owner = parts[0]
    repo = parts[1].replace(".git", "")
    return owner, repo

# ========== Main ==========
def main():
    parser = argparse.ArgumentParser(description="Generate release notes from GitHub commits using AI")
    parser.add_argument("--repo-url", required=True, help="GitHub repository URL")
    parser.add_argument("--from-ref", help="Starting git ref (tag, branch, or commit SHA)")
    parser.add_argument("--to-ref", default="main", help="Ending git ref (tag, branch, or commit SHA). Default: main")
    parser.add_argument("--days-ago", type=int, help="Days ago to use as start reference (alternative to --from-ref)")
    parser.add_argument("--github-token", help="GitHub Personal Access Token (defaults to GITHUB_TOKEN env var)")
    parser.add_argument("--model", default="gpt-4", help="OpenAI model to use")
    parser.add_argument("--api-model", help="Alternative name to specify the OpenAI model")
    parser.add_argument("--api-key", help="OpenAI API key (defaults to OPENAI_API_KEY env var)")
    parser.add_argument("--api-base-url", default="https://api.openai.com/v1", help="OpenAI API base URL")
    parser.add_argument("--org-id", help="OpenAI organization ID")
    parser.add_argument("--output-file", help="Path to save release notes (optional)")
    parser.add_argument("--state-file", default=".last-nightly-commit", help="File to store last processed commit for nightly builds")
    parser.add_argument("--nightly", action="store_true", help="Generate nightly changelog using state file")
    
    args = parser.parse_args()
    
    # Get GitHub token from args or environment
    github_token = args.github_token or os.environ.get("GITHUB_TOKEN")
    
    # Get OpenAI API key from args or environment
    api_key = args.api_key or os.environ.get("OPENAI_API_KEY")
    if not api_key and OPENAI_AVAILABLE:
        logger.error("OpenAI API key not provided. Use --api-key or set OPENAI_API_KEY environment variable.")
        return 1
    
    # Determine which model to use (api-model takes precedence over model)
    model = args.api_model or args.model
    
    # Parse GitHub repository URL
    try:
        owner, repo = parse_github_url(args.repo_url)
        logger.info(f"Working with repository: {owner}/{repo}")
    except ValueError as e:
        logger.error(str(e))
        return 1
    
    # Initialize GitHub API client
    github = GitHubAPI(owner, repo, github_token)
    
    # Determine the references to use
    to_ref = args.to_ref
    from_ref = args.from_ref
    
    # Handle nightly build scenario
    if args.nightly:
        logger.info("Running in nightly mode")
        latest_commit = github.get_latest_commit(to_ref)
        if not latest_commit:
            logger.error(f"Could not get latest commit for {to_ref}")
            return 1
        
        latest_sha = latest_commit["sha"]
        
        # Try to get previous commit from state file
        if os.path.exists(args.state_file):
            with open(args.state_file, 'r') as f:
                from_ref = f.read().strip()
            logger.info(f"Using commit from state file: {from_ref}")
        elif args.days_ago:
            # Fall back to days ago
            commit_days_ago = github.get_commit_from_days_ago(args.days_ago, to_ref)
            if commit_days_ago:
                from_ref = commit_days_ago["sha"]
                logger.info(f"Using commit from {args.days_ago} days ago: {from_ref}")
            else:
                logger.error(f"Could not get commit from {args.days_ago} days ago")
                return 1
        else:
            # Default to 1 day ago
            commit_days_ago = github.get_commit_from_days_ago(1, to_ref)
            if commit_days_ago:
                from_ref = commit_days_ago["sha"]
                logger.info(f"Using commit from 1 day ago: {from_ref}")
            else:
                logger.error("Could not get commit from 1 day ago")
                return 1
    
    # For non-nightly when days-ago is specified
    elif args.days_ago and not from_ref:
        commit_days_ago = github.get_commit_from_days_ago(args.days_ago, to_ref)
        if commit_days_ago:
            from_ref = commit_days_ago["sha"]
            logger.info(f"Using commit from {args.days_ago} days ago: {from_ref}")
        else:
            logger.error(f"Could not get commit from {args.days_ago} days ago")
            return 1
    
    # Check if we have valid references
    if not from_ref or not to_ref:
        logger.error("Both from and to references are required")
        return 1
    
    try:
        # Get commits between references
        commits = github.get_commits_between(from_ref, to_ref)
        
        if not commits:
            logger.warning("No commits found between given refs.")
            return 0
        
        # Generate release notes
        logger.info(f"Sending {len(commits)} commits to AI for release note generation...")
        release_notes = generate_release_notes(
            commits, 
            model, 
            api_key, 
            args.api_base_url,
            args.org_id
        )
        
        # Save release notes if output file specified
        save_release_notes(release_notes, args.output_file)
        
        # Print release notes
        print("\n========== 📢 Release Notes ==========\n")
        print(release_notes)
        print("\n======================================\n")
        
        # If in nightly mode, update the state file with current commit
        if args.nightly and latest_commit:
            with open(args.state_file, 'w') as f:
                f.write(latest_commit["sha"])
            logger.info(f"Updated state file with commit: {latest_commit['sha']}")
        
    except Exception as e:
        logger.error(f"Error: {str(e)}")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
