"""Fetch all features and epics of PROJECT from Jira (read-only) and save the raw responses.

Usage: python jira_rohdaten.py  (standard library only)
"""

import getpass
import json
import os
import sys
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

JIRA_URL = os.environ.get("FSEA_JIRA_URL", "https://jira.example.com")
OUTPUT_DIR = Path.home() / "Desktop"
PAGE_SIZE = 200  # issues per request
PROJECT = os.environ.get("FSEA_JIRA_PROJECT", "DEMO")  # export only this project (data minimisation)

# (output file name, JQL query)
QUERIES = [
    ("features.json", f'project = {PROJECT} AND issuetype = "Feature" ORDER BY key ASC'),
    ("epics.json", f'project = {PROJECT} AND issuetype = "Epic" ORDER BY key ASC'),
]


def exit_with_message(message, code=1):
    """Print a message and keep the window open until Enter is pressed."""
    print()
    print(message)
    input("Press Enter to close ...")
    sys.exit(code)


def fetch_page(token, jql, start_at):
    """Run one Jira search request and return the response as raw bytes."""
    query = urllib.parse.urlencode(
        {"jql": jql, "fields": "*all", "startAt": start_at, "maxResults": PAGE_SIZE}
    )
    request = urllib.request.Request(
        f"{JIRA_URL}/rest/api/2/search?{query}",
        headers={"Authorization": f"Bearer {token}"},
    )
    try:
        with urllib.request.urlopen(request, timeout=120) as response:
            data = response.read()
    except urllib.error.HTTPError as error:
        exit_with_message(f"ERROR: Jira responded with {error.code} {error.reason}. Check the token.")
    except urllib.error.URLError as error:
        exit_with_message(f"ERROR: no connection ({error.reason}). Check the VPN.")

    if not data.lstrip().startswith(b"{"):
        exit_with_message(
            "ERROR: Jira did not return JSON, most likely the login page.\n"
            "Check the VPN / company network and the token, then run the script again."
        )
    return data


def main():
    token = getpass.getpass("Jira Personal Access Token (input is hidden): ").strip()
    if not token:
        exit_with_message("No token entered.")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    for file_name, jql in QUERIES:
        raw = fetch_page(token, jql, 0)
        (OUTPUT_DIR / file_name).write_bytes(raw)

        info = json.loads(raw)
        total = info.get("total", 0)
        fetched = len(info.get("issues", []))
        print(f"{file_name}: {fetched} of {total}")

        # further pages if the server limits the page size
        start_at, part = fetched, 2
        while fetched and start_at < total:
            raw = fetch_page(token, jql, start_at)
            count = len(json.loads(raw).get("issues", []))
            if count == 0:
                break
            path = Path(file_name)
            extra_file = OUTPUT_DIR / f"{path.stem}_part{part}{path.suffix}"
            extra_file.write_bytes(raw)
            print(f"  Server limits the page size, {count} more saved in {extra_file.name}")
            start_at += count
            part += 1

    exit_with_message(f"Done. The files are in: {OUTPUT_DIR}", code=0)


if __name__ == "__main__":
    main()
