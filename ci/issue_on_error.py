# issue_on_error_post.py
# Requires python 3.6+ and GitHub CLI in the environment.
#
# Creates or updates an issue on action failure related to a workflow.
#
# Looks though REPO for open issues with FLAG_LABEL. If none,
# creates a new issue. If there is an open issue with FLAG_LABEL,
# looks for WORKFLOW in body. If none, creates an issue. If an
# issue for WORKFLOW exists, makes adds a comment.
#
# Requires the environment provide the variables in the block below.
# Authentication is gh's own: it reads GITHUB_TOKEN (or GH_TOKEN) from the
# environment, which the calling workflow supplies. That token needs
# `issues: write`, which also covers the labels API used below.
#
# If called with an optional PR_NUMBER and PR_LINK the issue will
# include a link to the PR.
#
# This script only ever runs because something else already failed, which sets
# how its own failures are graded:
#
#   - A *partial* degradation -- the report was filed or commented, but some
#     step along the way had to fall back -- is a `::warning::` and exit 0.
#     The report exists; the annotation says how it was compromised.
#   - Reporting *nothing* is a `::error::` and a non-zero exit. A reporter that
#     cannot report but stays green is indistinguishable from a healthy one,
#     which is exactly the silent breakage this script is supposed to catch.
#     The trap runs as its own job alongside the failing one, so a red trap
#     hides nothing: both appear in the run.
#
# The required environment variables below are read at import, so a caller that
# omits one gets a traceback rather than an annotation. That is deliberate --
# it is a wiring bug in the calling workflow, not a runtime degradation.


from datetime import datetime
import os
import json
import subprocess
import sys

REPO_NAME = os.environ['REPO']
WORKFLOW = os.environ['WORKFLOW']
FLAG_LABEL = os.environ['FLAG_LABEL']
RUN_NUMBER = os.environ['RUN_NUMBER']
RUN_ID = os.environ['RUN_ID']
# optional variables
PR_NUMBER = os.getenv('PR_NUMBER')
PR_LINK = os.getenv('PR_LINK')

# Colour and description used when self-healing a missing flag label. Keep in
# sync with the label as it exists in the repository so `--force` (which edits
# an existing label) is a no-op rather than a per-run colour change.
FLAG_LABEL_COLOR = "B60205"
FLAG_LABEL_DESCRIPTION = "Auto-filed by CI failure traps"


def warn(message):
    print(f"::warning::issue_on_error: {message}")


def error(message):
    print(f"::error::issue_on_error: {message}")


def describe(args):
    # A workflow annotation renders only up to its first newline, and the issue
    # body is multi-line, so echoing the full argv would truncate the message
    # to something useless. Name the subcommand instead.
    return "gh " + " ".join(args[:2])


def run_gh(args):
    # Returns stdout on success, None on failure. gh writes its diagnostics to
    # stderr, so capture and re-emit them: a bare CalledProcessError says only
    # "exit status 1" and the reason for a trap that cannot file is the whole
    # thing worth knowing.
    try:
        result = subprocess.run(["gh"] + args, stdout=subprocess.PIPE,
                                stderr=subprocess.PIPE, encoding="utf-8")
    except OSError as err:
        warn(f"could not run `{describe(args)}`: {err}")
        return None
    if result.returncode != 0:
        warn(f"`{describe(args)}` failed ({result.returncode}): "
             f"{' '.join(result.stderr.split())}")
        return None
    return result.stdout


def ensure_label(repo_name, flag_label):
    # Only called after a labelled create has already failed, never up front.
    # Running it unconditionally would mean a transient 502 from the labels
    # endpoint downgrades a perfectly healthy filing to an unlabelled issue
    # -- which the next run's `--label` search cannot see, resurrecting the
    # duplicate accumulation this script now avoids. `--force` also edits an
    # existing label, so an unconditional call would silently revert a manual
    # recolour on every run.
    return run_gh(["label", "create", flag_label,
                   "--repo", repo_name,
                   "--color", FLAG_LABEL_COLOR,
                   "--description", FLAG_LABEL_DESCRIPTION,
                   "--force"]) is not None


def get_tagged_issues(repo_name, flag_label, workflow):
    # Returns None (not []) when the search itself fails, so the caller can
    # tell "nothing filed yet" from "cannot tell", and refuse to file blind.
    # --limit is explicit because `gh issue list` defaults to 30. Past 30 open
    # flagged issues the one this run should comment on sorts out of view, the
    # search reports "nothing filed yet", and every failing run files a fresh
    # duplicate -- which then makes the backlog worse.
    issues = run_gh(["issue", "list",
                     "--repo", repo_name,
                     "--state", "open",
                     "--label", flag_label,
                     "--limit", "100",
                     "--json", "title,number,body"])
    if issues is None:
        return None
    try:
        issues = json.loads(issues)
    except ValueError as err:
        warn(f"could not parse `gh issue list` output: {err}")
        return None
    tagged_issues =[]
    for issue in issues:
        if workflow in issue["body"]:
            tagged_issues.append(issue)
    return(tagged_issues)


def select_existing(tagged_issues, pr_number, pr_link):
    # Pick at most one issue to act on. Creating inside a loop over matches is
    # what made N open issues for a workflow produce N duplicates in one run.
    for issue in tagged_issues:
        if not pr_number:
            return(issue)
        # An empty PR_LINK must not match: `"" in body` is true for every
        # issue, which would comment on an unrelated one instead of filing.
        if pr_link and pr_link in issue["body"]:
            return(issue)
    return(None)


def create_issue(flag_label, workflow, run_number, run_id, repo_name, pr_number, pr_link):
    # Returns a (outcome, payload) pair:
    #   ("created", gh stdout)   a new issue was filed
    #   ("exists",  issue dict)  an issue for this failure is already there
    #   ("failed",  None)        nothing was filed
    #   ("unknown", None)        cannot determine whether anything was filed

    run_link = f"https://github.com/{repo_name}/actions/runs/{run_id}"
    body_string = ""
    title_string = f"{workflow} CI Run Failed"
    if pr_number:
        body_string = f"PR {pr_number} ({pr_link}) had a CI failure: \n"
        title_string = f"PR #{pr_number} CI Run Failed"
    body_string += f"{workflow} [run number {run_number}]({run_link}) failed. \n\n"
    body_string += "This issue has been automatically generated for "
    body_string += "notification purposes."

    # Title and body are passed as argv elements, never through a shell, so
    # neither needs quoting however the workflow name or PR link is spelled.
    create_args = ["issue", "create",
                   "--repo", repo_name,
                   "--title", title_string,
                   "--body", body_string]

    # Try the labelled create first. The happy path is then a single call and
    # touches the labels endpoint not at all.
    new_issue = run_gh(create_args + ["--label", flag_label])
    if new_issue is not None:
        return("created", new_issue)

    # Re-run the dedupe search before retrying anything. A failed create has
    # two quite different causes and this distinguishes them by observation
    # rather than by assumption:
    #
    #   - The label does not exist. `gh issue create --label` is fatal in that
    #     case, which is exactly how this trap spent its life failing instead
    #     of filing. No issue was created, the search finds nothing, and the
    #     retry below is safe.
    #   - The response was lost after the server had already created the issue
    #     (connection reset, 502 read timeout, gh killed mid-call). Retrying
    #     blind would file a duplicate. The search finds the new issue and
    #     turns this into the existing-issue path instead.
    #
    # Checking rather than reasoning also removes any dependence on *when* gh
    # resolves label names relative to POSTing the issue -- an ordering this
    # script cannot verify and should not have to rely on.
    warn("issue creation failed; re-checking whether an issue was filed anyway")
    tagged_issues = get_tagged_issues(repo_name, flag_label, workflow)

    if tagged_issues is None:
        # The re-check itself failed, so the two cases above are now
        # indistinguishable: the create may have landed and the confirmation
        # been lost. Retrying would risk the duplicate this re-check exists to
        # prevent, so stop here and let the caller report a non-zero exit. An
        # unreported failure is recoverable by reading the red job; a duplicate
        # issue filed on every run is the failure mode that buries it.
        return("unknown", None)

    if tagged_issues:
        existing = select_existing(tagged_issues, pr_number, pr_link)
        if existing is not None:
            return("exists", existing)

    warn(f"no issue was filed; ensuring label {flag_label!r} exists and retrying")
    if ensure_label(repo_name, flag_label):
        new_issue = run_gh(create_args + ["--label", flag_label])
        if new_issue is not None:
            return("created", new_issue)

    # Last resort: an unlabelled issue. Dedupe keys off the label, so a later
    # failure will not find this one and may file a second -- still better than
    # reporting nothing at all.
    warn(f"filing an unlabelled issue; dedupe by {flag_label!r} will not see it")
    new_issue = run_gh(create_args)
    if new_issue is not None:
        return("created", new_issue)
    return("failed", None)


def add_comment(issue_number, run_number, run_id, repo_name):
    dt_string = datetime.now().strftime("%d/%m/%Y %H:%M:%S")
    run_link = f"https://github.com/{repo_name}/actions/runs/{run_id}"
    msg_string = f"Error reoccurred: {dt_string}\n"
    msg_string += f"[Run number: {run_number}]({run_link})\n"
    return(run_gh(["issue", "comment", issue_number,
                   "--repo", repo_name,
                   "--body", msg_string]) is not None)

if __name__ == "__main__":
    # Set once the failure has actually been recorded somewhere a human will
    # see: a new issue, or a comment on an existing one. Anything short of
    # that is a trap that reported nothing, and must not exit green.
    reported = False

    tagged_issues = get_tagged_issues(REPO_NAME, FLAG_LABEL, WORKFLOW)

    if tagged_issues is None:
        # The dedupe search failed, so filing now could duplicate an issue that
        # already exists. Refusing to file blind is the right call, but it
        # leaves the failure unreported -- hence the non-zero exit below.
        error("cannot search existing issues; filed nothing (would risk duplicates)")
    else:
        existing = select_existing(tagged_issues, PR_NUMBER, PR_LINK)

        # The logic catches the case where an issue exists for the workflow
        # but we are testing against a PR and want a created issue to link to the PR.
        # Otherwise, we just add a comment to the existing issue.
        if existing is None:
            outcome, payload = create_issue(FLAG_LABEL, WORKFLOW, RUN_NUMBER,
                                            RUN_ID, REPO_NAME, PR_NUMBER,
                                            PR_LINK)
            if outcome == "created":
                reported = True
            elif outcome == "exists":
                # The create reported failure but an issue for this failure is
                # on the tracker -- almost always the one this invocation just
                # filed, whose response was lost in transit. The failure IS
                # recorded, so this is a partial degradation, not a silent
                # miss: comment for the extra run detail, but do not fail the
                # job if that comment does not land.
                reported = True
                print(payload["number"])
                if not add_comment(str(payload["number"]), RUN_NUMBER, RUN_ID,
                                   REPO_NAME):
                    warn(f"issue #{payload['number']} already records this "
                         "failure, but the follow-up comment did not post")
            elif outcome == "unknown":
                error("issue creation failed and the follow-up search could "
                      "not confirm whether it landed; stopped rather than "
                      "risk filing a duplicate")
            else:
                error("every issue-creation attempt failed; filed nothing")
        else:
            reported = add_comment(str(existing["number"]), RUN_NUMBER, RUN_ID,
                                   REPO_NAME)
            if reported:
                # Only announce the issue number once the comment is really on
                # it; printing unconditionally reads as success in the log.
                print(existing["number"])
            else:
                error(f"could not comment on issue #{existing['number']}; "
                      "the failure went unreported")

    if not reported:
        sys.exit(1)
