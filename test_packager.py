import json

from listing_packager import fetch_listing_details


url = "https://www.cashconverters.es/es/es/segunda-mano/CC016_E730733_0.html"

data = fetch_listing_details(url)

print(json.dumps(data, indent=2, ensure_ascii=False))