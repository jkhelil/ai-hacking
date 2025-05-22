
# ChangeLog Generator
This script generates release notes for a given repository or multiple repositories.
It uses the GitHub API to fetch commit messages and the OpenAI API to generate release notes.


# Installation
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

# Usage:
```bash
export API_BASE_URL=""
export API_KEY="xx"
export GITHUB_TOKEN="xx"
export API_MODEL="/data/granite-3.1-8b-instruct"
```
## For a single repository:
```bash
/generate-changelog.py --repo-url https://github.com/tektoncd/operator.git \
      --from-ref "main@{30 day ago}" \
      --to-ref "main" \
      --api-base-url "$API_BASE_URL" \
      --api-key "$API_KEY" \
      --github-token "$GITHUB_TOKEN" \
      --api-model "$API_MODEL"
````
## for multiple repositories:
```bash
/generate-changelog.py  --repos https://github.com/tektoncd/operator.git https://github.com/tektoncd/pipeline.git \
  --from-ref "main@{30 day ago}" \
  --to-ref "main" \
  --api-base-url "$API_BASE_URL" \
  --api-key "$API_KEY" \
  --github-token "$GITHUB_TOKEN" \
  --api-model "$API_MODEL" \
  --combined \
  --output-file combined-release-notes.md
```
