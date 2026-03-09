import json
from pathlib import Path

from listing_packager import fetch_listing_details


URL = "https://www.cashconverters.es/es/es/segunda-mano/CC016_E730733_0.html"
OUTPUT_DIR = Path("data/packaged")


def main() -> None:
    data = fetch_listing_details(URL)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    listing_id = data.get("listing_id") or "unknown_listing"
    output_path = OUTPUT_DIR / f"{listing_id}.json"

    with output_path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    print(f"Saved: {output_path}")


if __name__ == "__main__":
    main()