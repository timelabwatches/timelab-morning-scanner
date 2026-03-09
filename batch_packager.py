import json
from pathlib import Path

from listing_packager import fetch_listing_details


URLS = [
    "https://www.cashconverters.es/es/es/segunda-mano/CC016_E730733_0.html",
]

OUTPUT_DIR = Path("data/packaged")


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    results = []

    for url in URLS:
        try:
            data = fetch_listing_details(url)
            results.append(data)

            listing_id = data.get("listing_id") or "unknown_listing"
            output_path = OUTPUT_DIR / f"{listing_id}.json"

            with output_path.open("w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)

            print(f"Saved: {output_path}")

        except Exception as e:
            print(f"Error with {url}: {type(e).__name__}: {e}")

    summary_path = OUTPUT_DIR / "summary.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"Saved summary: {summary_path}")


if __name__ == "__main__":
    main()