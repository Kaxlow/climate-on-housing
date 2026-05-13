from __future__ import annotations

import argparse
import json

from housing_climate_risk.page_data.registry import PAGE_HTML_FILES, build_all_pages, build_page


def _summary(result: dict[str, object]) -> dict[str, object]:
    return {
        "page": result["page"],
        "html_file": result["html_file"],
        "written_files": len(result.get("written_paths", [])),
        "url": result.get("url"),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a registered visualization page.")
    parser.add_argument("page", choices=[*PAGE_HTML_FILES, "all"], help="Page to build.")
    parser.add_argument("--serve", action="store_true", help="Serve the matching HTML page after building.")
    parser.add_argument("--host", default="127.0.0.1", help="Host for --serve.")
    parser.add_argument("--port", type=int, default=None, help="Port for --serve. Defaults to the first open port from 8000-8100.")
    args = parser.parse_args()

    if args.page == "all":
        if args.serve:
            raise SystemExit("--serve can only be used with a single page.")
        results = build_all_pages(progress=True)
        print(json.dumps({"pages": [_summary(result) for result in results]}, indent=2))
        return

    result = build_page(args.page, serve=args.serve, host=args.host, port=args.port)
    print(json.dumps(_summary(result), indent=2))


if __name__ == "__main__":
    main()
