"""Build the policy index from documents in data/policies/."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.config import settings          # noqa: E402
from app.ingest import ingest_directory  # noqa: E402
from app.store import NumpyStore         # noqa: E402


def main() -> int:
    if not os.path.isdir(settings.policy_dir):
        print(f"No such directory: {settings.policy_dir}", file=sys.stderr)
        return 1

    store = NumpyStore(settings.index_dir)
    report = ingest_directory(settings.policy_dir, store)

    for filename, count in sorted(report.per_file.items()):
        print(f"  {count:5d} passages   {filename}")
    for filename in report.skipped_files:
        print(f"  skipped (unsupported type): {filename}")

    print(f"\nIndexed {report.passages_added} passages. Index now holds {store.count()}.")

    if report.empty_files:
        print(
            "\nWARNING - these files produced no text and are NOT searchable:",
            file=sys.stderr,
        )
        for filename in report.empty_files:
            print(f"  {filename}", file=sys.stderr)
        print(
            "  A PDF with no text layer (a scan) extracts nothing. Either OCR it or\n"
            "  replace it with a text version - otherwise every request that depends\n"
            "  on it will escalate with insufficient_policy_match.",
            file=sys.stderr,
        )
    if report.passages_added == 0:
        print("\nNothing was indexed. Add policy documents to data/policies/.", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
