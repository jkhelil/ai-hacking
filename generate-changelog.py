#!/usr/bin/env python3
import argparse
import logging
import os
import sys
import json
import datetime
from typing import List, Dict, Optional, Any, Union
import requests
import re  # For PR number extraction from commit messages

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

    def _should_skip_commit(self, files_changed: List[Dict[str, Any]]) -> bool:
        """Determine if a commit should be skipped based on changed files."""
        # If there are no files, don't skip
        if not files_changed:
            return False
        
        # Check if all files are either in vendor folder or are go.mod/go.sum
        go_mod_files = {"go.mod", "go.sum"}
        owners_files = {"OWNERS", "OWNERS_ALIASES"}
        for file in files_changed:
            file_path = file.get("filename", "")
            # If any file is not in vendor and not go.mod/go.sum, don't skip the commit
            if not file_path.startswith("vendor/") and file_path not in go_mod_files and file_path not in owners_files:
                return False
        
        # All files are either in vendor folder or go.mod/go.sum
        return True
    
    def get_commits_between(self, from_ref: str, to_ref: str) -> List[Dict[str, Any]]:
        """Get commits between two references with filtering."""
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
            # Skip if commit is from Dependabot[bot]
            author_login = commit.get("author", {}).get("login", "")
            if author_login == "dependabot[bot]":
                logger.debug(f"Skipping Dependabot commit: {commit['sha']}")
                continue
            
            # Get files changed in this commit
            commit_detail_url = f"{self.base_url}/commits/{commit['sha']}"
            # Request the diff as well by specifying the 'Accept' header
            headers_with_diff = self.headers.copy()
            headers_with_diff["Accept"] = "application/vnd.github.v3.diff"
            
            # Get the diff
            diff_response = requests.get(commit_detail_url, headers=headers_with_diff)
            diff_response.raise_for_status()
            diff_content = diff_response.text
            
            # Get the commit details (for file information)
            commit_detail_response = requests.get(commit_detail_url, headers=self.headers)
            commit_detail_response.raise_for_status()
            commit_detail = commit_detail_response.json()
            files_changed = commit_detail.get("files", [])
            
            # Skip if all changes are in vendor folder or only in go.mod/go.sum
            if self._should_skip_commit(files_changed):
                logger.debug(f"Skipping commit with filtered changes: {commit['sha']}")
                continue
            
            # Get pull request details for this commit
            pull_request_data = self._get_pull_request_for_commit(commit["sha"])
            pr_number = None
            pr_url = None
            if pull_request_data:
                pr_number = pull_request_data.get("number")
                pr_url = pull_request_data.get("html_url")
            else:
                # Try to extract PR number from commit message (e.g., #1234 or PR #1234)
                msg = commit["commit"]["message"]
                match = re.search(r'#(\d+)', msg)
                if not match:
                    match = re.search(r'PR[\s#]*(\d+)', msg, re.IGNORECASE)
                if match:
                    pr_number = match.group(1)
                    pr_url = f"https://github.com/{self.owner}/{self.repo}/pull/{pr_number}"
            
            commit_data = {
                "sha": commit["sha"][:7],  # Short SHA
                "message": commit["commit"]["message"].split("\n")[0],  # First line only
                "author": commit["commit"]["author"]["name"],
                "date": commit["commit"]["author"]["date"],
                "html_url": commit["html_url"],
                "pull_request": pull_request_data,  # Add PR data to commit info
                "pr_number": pr_number,  # Always add pr_number
                "pr_url": pr_url,        # Always add pr_url
                "diff": diff_content,  # Add the diff to commit info
                "files_changed": files_changed  # Add files_changed info with stats
            }
 
            commits.append(commit_data)
        #print(f"Processing commits: {commits}")
        
        logger.info(f"Found {len(commits)} commits between {from_ref} and {to_ref} after filtering")
        return commits
    
    def _get_pull_request_for_commit(self, commit_sha: str) -> Dict[str, Any]:
        """Get pull request details for a commit."""
        # GitHub API endpoint to list pull requests associated with a commit
        url = f"{self.base_url}/commits/{commit_sha}/pulls"
        # Use the special 'Accept' header for the pulls API (this is a preview feature)
        headers = self.headers.copy()
        headers["Accept"] = "application/vnd.github.groot-preview+json"
        
        response = requests.get(url, headers=headers)
        
        # Handle API errors gracefully
        if response.status_code != 200:
            logger.warning(f"Failed to get PR details for commit {commit_sha}: {response.status_code}")
            return None
        
        pull_requests = response.json()
        
        # If no PRs found, return None
        if not pull_requests:
            return None
        
        # Use the first PR (typically there's only one PR per commit)
        pr = pull_requests[0]
       # Extract relevant PR information    
        return {
            "number": pr.get("number"),
            "title": pr.get("title"),
            "labels": pr.get("labels"),
            "html_url": pr.get("html_url"),
            "state": pr.get("state"),
            "merged": pr.get("merged"),
            "created_at": pr.get("created_at"),
            "merged_at": pr.get("merged_at"),
            "user": {
            "login": pr.get("user", {}).get("login"),
            "html_url": pr.get("user", {}).get("html_url")
            }
        }
    
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
        f"- {commit['sha']}: {commit['message']} ({commit['author']})" + (f" [PR #{commit['pr_number']}]({commit['pr_url']})" if commit.get('pr_number') and commit.get('pr_url') else "")
        for commit in commits
    ])
    
    template = """### 🚨 Breaking Changes
[BREAKING_CHANGES]

### ✨ New Features
[NEW_FEATURES]

### 🔧 Improvements
[IMPROVEMENTS]

### 🐛 Bug Fixes
[BUG_FIXES]

### 📝 Documentation Changes
[DOCUMENTATION_CHANGES]

### ⚙️ Configuration Changes
[CONFIGURATION_CHANGES]

### 🔄 Dependencies
[DEPENDENCIES]"""

    prompt = f"""
You are an expert release analysis assistant.

From the following list of git commits, you must fill in the template below by categorizing each commit into the appropriate section.

**CATEGORIZATION RULES:**

1. 🚨 **Breaking Changes**: Any code or behavior changes that may cause failures, remove or alter APIs, modify existing functionality, or introduce backward incompatibility.
2. ✨ **New Features**: Any new functionality, capabilities, or enhancements added to the system.
3. 🔧 **Improvements**: Enhancements to existing features that don't add entirely new functionality.
4. 🐛 **Bug Fixes**: Corrections to resolve issues, errors, or unexpected behavior.
5. 📝 **Documentation Changes**: Updates to documentation, examples, or comments.
6. ⚙️ **Configuration Changes**: Any changes that introduce, remove, or modify configuration files, environment variables, deployment manifests, feature flags, or system-level parameters.
7. 🔄 **Dependencies**: Updates to external dependencies, libraries, or modules.

**FORMATTING REQUIREMENTS:**
- For each commit, use this exact format: `commit-hash`: Short summary (Author) [PR #123](pr_link)
- If a section has no commits, write "No changes in this category."
- Do NOT omit any sections from the template
- Do NOT add any additional text outside the template structure
- Focus only on what is essential for teams to verify safety and compatibility after an upgrade

**ANALYSIS GUIDELINES:**
1. Analyze the diff content to identify potentially breaking changes or any change (addition, removal) of environment variables, or configuration modifications
2. Look for specific PR labels or commit message patterns:
   - "feature", "feat", "enhancement" → New Features
   - "fix", "bugfix", "resolve" → Bug Fixes
   - "improve", "optimize", "refactor" → Improvements
   - "breaking", "api-change" → Breaking Changes
   - "docs", "documentation" → Documentation Changes
   - "config", "deployment", "env" → Configuration Changes
   - "dependency", "upgrade", "update" → Dependencies
3. Check for configuration-related files or paths in the modified files list
4. Examine commit messages for keywords like "api", "CRD", "breaking", "deprecated", "removed", "config", "manifest"

**TEMPLATE TO FILL:**
{template}

**COMMIT LOG TO ANALYZE:**
{formatted_commits}

**INSTRUCTIONS:**
Replace each [SECTION_NAME] placeholder with the appropriate commits for that category. If no commits belong to a category, replace the placeholder with "No changes in this category."
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

def process_repository(repo_url: str, from_ref: str, to_ref: str, days_ago: int, nightly: bool,
                       state_file: str, model: str, api_key: str, api_base_url: str, org_id: str, 
                       github_token: str) -> Optional[Dict[str, Any]]:
    """Process a single GitHub repository and generate release notes."""
    try:
        # Parse GitHub repository URL
        owner, repo = parse_github_url(repo_url)
        logger.info(f"Working with repository: {owner}/{repo}")
        
        # Initialize GitHub API client
        github = GitHubAPI(owner, repo, github_token)
        
        # Determine the references to use
        local_to_ref = to_ref
        local_from_ref = from_ref
        
        # Handle nightly build scenario
        if nightly:
            logger.info("Running in nightly mode")
            latest_commit = github.get_latest_commit(local_to_ref)
            if not latest_commit:
                logger.error(f"Could not get latest commit for {local_to_ref}")
                return None
            
            latest_sha = latest_commit["sha"]
            
            # Try to get previous commit from state file
            repo_state_file = f"{state_file}_{owner}_{repo}"
            if os.path.exists(repo_state_file):
                with open(repo_state_file, 'r') as f:
                    local_from_ref = f.read().strip()
                logger.info(f"Using commit from state file: {local_from_ref}")
            elif days_ago:
                # Fall back to days ago
                commit_days_ago = github.get_commit_from_days_ago(days_ago, local_to_ref)
                if commit_days_ago:
                    local_from_ref = commit_days_ago["sha"]
                    logger.info(f"Using commit from {days_ago} days ago: {local_from_ref}")
                else:
                    logger.error(f"Could not get commit from {days_ago} days ago")
                    return None
            else:
                # Default to 1 day ago
                commit_days_ago = github.get_commit_from_days_ago(1, local_to_ref)
                if commit_days_ago:
                    local_from_ref = commit_days_ago["sha"]
                    logger.info(f"Using commit from 1 day ago: {local_from_ref}")
                else:
                    logger.error("Could not get commit from 1 day ago")
                    return None
        
        # For non-nightly when days-ago is specified
        elif days_ago and not local_from_ref:
            commit_days_ago = github.get_commit_from_days_ago(days_ago, local_to_ref)
            if commit_days_ago:
                local_from_ref = commit_days_ago["sha"]
                logger.info(f"Using commit from {days_ago} days ago: {local_from_ref}")
            else:
                logger.error(f"Could not get commit from {days_ago} days ago")
                return None
        
        # Check if we have valid references
        if not local_from_ref or not local_to_ref:
            logger.error("Both from and to references are required")
            return None
        
        # Get commits between references
        commits = github.get_commits_between(local_from_ref, local_to_ref)
        
        if not commits:
            logger.warning("No commits found between given refs.")
            return {
                "repo_url": repo_url,
                "owner": owner,
                "repo": repo,
                "release_notes": f"No commits found for {owner}/{repo} between {local_from_ref} and {local_to_ref}",
                "from_ref": local_from_ref,
                "to_ref": local_to_ref,
                "nightly": nightly,
                "latest_commit": latest_commit if nightly else None
            }
        
        # Generate release notes
        logger.info(f"Sending {len(commits)} commits to AI for release note generation...")
        release_notes = generate_release_notes(
            commits, 
            model, 
            api_key, 
            api_base_url,
            org_id
        )
        
        # If in nightly mode, update the state file with current commit
        if nightly and latest_commit:
            repo_state_file = f"{state_file}_{owner}_{repo}"
            with open(repo_state_file, 'w') as f:
                f.write(latest_commit["sha"])
            logger.info(f"Updated state file with commit: {latest_commit['sha']}")
        
        return {
            "repo_url": repo_url,
            "owner": owner,
            "repo": repo,
            "release_notes": release_notes,
            "from_ref": local_from_ref,
            "to_ref": local_to_ref,
            "nightly": nightly,
            "latest_commit": latest_commit if nightly else None
        }
        
    except Exception as e:
        logger.error(f"Error processing {repo_url}: {str(e)}")
        return None

def create_combined_release_notes(repo_results: List[Dict[str, Any]]) -> str:
    """Create a combined release notes document from multiple repositories."""
    today = datetime.datetime.now().strftime("%Y-%m-%d")
    
    # Create the header
    combined_notes = f"# Combined Release Notes ({today})\n\n"
    
    # Summary section
    combined_notes += "## Repositories Analyzed\n\n"
    for repo_data in repo_results:
        owner = repo_data["owner"]
        repo = repo_data["repo"]
        from_ref = repo_data["from_ref"]
        to_ref = repo_data["to_ref"]
        combined_notes += f"- **{owner}/{repo}**: Changes from `{from_ref}` to `{to_ref}`\n"
    combined_notes += "\n"
    
    # Add each repository's release notes
    for repo_data in repo_results:
        owner = repo_data["owner"]
        repo = repo_data["repo"]
        release_notes = repo_data["release_notes"]
        
        combined_notes += f"## {owner}/{repo}\n\n"
        combined_notes += release_notes + "\n\n"
        combined_notes += "---\n\n"
    
    return combined_notes

# ========== Main ==========
def main():
    parser = argparse.ArgumentParser(description="Generate release notes from GitHub commits using AI")
    # Repository options
    parser.add_argument("--repo-url", help="GitHub repository URL (deprecated, use --repos instead)")
    parser.add_argument("--repos", nargs='+', help="List of GitHub repository URLs to analyze")
    parser.add_argument("--combined", action="store_true", help="Generate combined release notes from all repositories")
    
    # Reference options
    parser.add_argument("--from-ref", help="Starting git ref (tag, branch, or commit SHA)")
    parser.add_argument("--to-ref", default="main", help="Ending git ref (tag, branch, or commit SHA). Default: main")
    parser.add_argument("--days-ago", type=int, help="Days ago to use as start reference (alternative to --from-ref)")
    
    # GitHub options
    parser.add_argument("--github-token", help="GitHub Personal Access Token (defaults to GITHUB_TOKEN env var)")
    
    # OpenAI options
    parser.add_argument("--model", default="gpt-4", help="OpenAI model to use")
    parser.add_argument("--api-model", help="Alternative name to specify the OpenAI model")
    parser.add_argument("--api-key", help="OpenAI API key (defaults to OPENAI_API_KEY env var)")
    parser.add_argument("--api-base-url", default="https://api.openai.com/v1", help="OpenAI API base URL")
    parser.add_argument("--org-id", help="OpenAI organization ID")
    
    # Output options
    parser.add_argument("--output-file", help="Path to save release notes (optional)")
    parser.add_argument("--state-file", default=".last-nightly-commit", help="File to store last processed commit for nightly builds")
    parser.add_argument("--nightly", action="store_true", help="Generate nightly changelog using state file")
    
    args = parser.parse_args()
    
    # Validate repository arguments
    if args.repos and args.repo_url:
        logger.warning("Both --repos and --repo-url provided. Using --repos and ignoring --repo-url.")
        repo_urls = args.repos
    elif args.repos:
        repo_urls = args.repos
    elif args.repo_url:
        repo_urls = [args.repo_url]
    else:
        logger.error("No repositories specified. Use --repos or --repo-url.")
        return 1
    
    # Get GitHub token from args or environment
    github_token = args.github_token or os.environ.get("GITHUB_TOKEN")
    
    # Get OpenAI API key from args or environment
    api_key = args.api_key or os.environ.get("OPENAI_API_KEY")
    if not api_key and OPENAI_AVAILABLE:
        logger.error("OpenAI API key not provided. Use --api-key or set OPENAI_API_KEY environment variable.")
        return 1
    
    # Determine which model to use (api-model takes precedence over model)
    model = args.api_model or args.model
    
    # Process each repository
    repo_results = []
    
    for repo_url in repo_urls:
        logger.info(f"\n=== Processing {repo_url} ===\n")
        result = process_repository(
            repo_url=repo_url,
            from_ref=args.from_ref,
            to_ref=args.to_ref,
            days_ago=args.days_ago,
            nightly=args.nightly,
            state_file=args.state_file,
            model=model,
            api_key=api_key,
            api_base_url=args.api_base_url,
            org_id=args.org_id,
            github_token=github_token
        )
        
        if result:
            repo_results.append(result)
            
            # Print individual repository release notes if not generating combined
            if not args.combined:
                print(f"\n========== 📢 Release Notes for {result['owner']}/{result['repo']} ==========\n")
                print(result["release_notes"])
                print("\n======================================\n")
                
                # Save individual repository release notes if output file specified
                if args.output_file and len(repo_urls) == 1:
                    save_release_notes(result["release_notes"], args.output_file)
    
    # Generate and save combined release notes if requested
    if args.combined and repo_results:
        combined_notes = create_combined_release_notes(repo_results)
        
        # Print combined release notes
        print("\n========== 📢 Combined Release Notes ==========\n")
        print(combined_notes)
        print("\n=============================================\n")
        
        # Save combined release notes if output file specified
        if args.output_file:
            save_release_notes(combined_notes, args.output_file)
    
    # Return success if we processed at least one repository
    return 0 if repo_results else 1

if __name__ == "__main__":
    sys.exit(main())
