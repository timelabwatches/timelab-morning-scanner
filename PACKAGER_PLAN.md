# TIMELAB Listing Packager v1

## Goal
Convert individual CashConverters listings into structured JSON records ready for later AI analysis.

## Current status
- Added `fetch_listing_details(url)` to `cashconverters_scanner.py`
- Output target fields:
  - source
  - listing_id
  - url
  - title
  - price_eur
  - shipping_eur
  - description
  - brand_hint
  - main_image_url
  - image_urls
  - collected_at
  - status

## Next steps
1. Create a minimal test runner for one CashConverters URL
2. Improve image extraction to avoid icons / irrelevant images
3. Improve price extraction if needed
4. Save packaged listings into JSON files
5. Process multiple listings automatically