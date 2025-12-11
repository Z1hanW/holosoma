from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta, timezone
from enum import Enum, auto

import wandb


class RunStatus(Enum):
    """Enum representing the possible states of a nightly test run."""

    SUCCEEDED = auto()
    CRASHED = auto()
    METRICS_REGRESSION = auto()
    FAILED = auto()
    UNKNOWN = auto()


# Mapping to help visually determine what happened via a slack message.
RunStatus2Emoji = {
    RunStatus.SUCCEEDED: "🟩",
    RunStatus.CRASHED: "🟥",
    RunStatus.METRICS_REGRESSION: "⚠️",
    RunStatus.FAILED: "🚫",
    RunStatus.UNKNOWN: "❓",
}


def _get_run_status(run: wandb.Run) -> RunStatus:
    """Gets `RunStatus` for a given wandb `run`."""

    # Work around mypy issues while still maintaining annotations
    run_state = getattr(run, "state", None)
    run_tags = getattr(run, "tags", []) or []

    if run_state == "finished" and "nightly_test_passed" in run_tags:
        status = RunStatus.SUCCEEDED
    elif run_state == "finished" and "nightly_test_failed" in run_tags:
        status = RunStatus.METRICS_REGRESSION
    elif run_state == "crashed":
        status = RunStatus.CRASHED
    elif run_state == "failed":
        status = RunStatus.FAILED
    else:
        # We don't cover some states (like {running, pending, killed}). These will fall back to UNKNOWN.
        status = RunStatus.UNKNOWN
    return status


def _fetch_project_runs(api: wandb.Api, project_name: str, since_iso: str) -> list[tuple[str, RunStatus]]:
    """Helper function to fetch runs for a single project.

    Returns a list of tuples containing (url, run_status)
    """
    run_data: list[tuple[str, RunStatus]] = []  # (url, run_status)
    try:
        runs = api.runs(
            path=f"far-wandb/{project_name}",
            filters={
                "tags": "hv-nightly",
                "created_at": {"$gte": since_iso},
            },
            order="-created_at",
        )
        # Determine run status based on run state and test results
        run_data.extend((run.url, _get_run_status(run)) for run in runs)
    except Exception as e:
        print(f"Error fetching runs for project {project_name}: {e}")
    return run_data


def print_last_nightly_urls():
    """Fetches & prints the url of all projects in FAR wandb with the 'hv-nightly' tag that have completed runs
    within the last 12 hours.
    """

    api = wandb.Api(timeout=60)

    # Fetch all projects for the FAR entity
    all_projects = list(api.projects("far-wandb"))

    # Get runs from the last 12 hours
    since_time = datetime.now(timezone.utc) - timedelta(hours=12)
    since_iso = since_time.isoformat()

    # Use parallel processing to speed up API calls
    # Default to a reasonable number of workers based on CPU count
    max_workers = min(32, (os.cpu_count() or 1) + 4)

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks
        future2project_name = {
            executor.submit(_fetch_project_runs, api, p.name, since_iso): p.name for p in all_projects
        }

        # Process completed tasks as they finish
        for future in as_completed(future2project_name):
            project_name = future2project_name[future]
            try:
                run_data = future.result()
                for url, status in run_data:
                    print(f"{RunStatus2Emoji.get(status)} {url}")
            except Exception as e:
                print(f"Error processing project {project_name}: {e}")


if __name__ == "__main__":
    print("\n")  # To make the message layout cleaner
    print_last_nightly_urls()
