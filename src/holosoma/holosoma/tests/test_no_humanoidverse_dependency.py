import subprocess

from holosoma.utils.module_utils import get_holosoma_root


def test():
    """Check that no files under holosoma include the string 'humanoidverse' (excluding this test file)."""

    holosoma_dir = get_holosoma_root()

    # Use find to search for humanoidverse references, excluding this test file
    cmd = [
        "find",
        holosoma_dir,
        "-name",
        "*.py",
        "-not",
        "-name",
        "*test_no_humanoidverse_dependency*",
        "-exec",
        "grep",
        "-l",
        "humanoidverse",
        "{}",
        ";",
    ]

    result = subprocess.run(cmd, capture_output=True, text=True, check=False)

    if result.stdout.strip():
        files_with_refs = result.stdout.strip().split("\n")
        files_with_refs = [f for f in files_with_refs if f]  # Remove empty lines
        raise AssertionError("Found humanoidverse references in holosoma:\n" + "\n".join(files_with_refs))
