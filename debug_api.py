"""
debug_api.py - Run this to test if the Energi Data Service API works on your machine.

Usage: python debug_api.py

This script tests 3 different URL formats to find one that works.
Copy the FULL output and paste it back to Claude if none work.
"""

import requests
import json

print("=" * 60)
print("🔍 ENERGI DATA SERVICE API DEBUG TOOL")
print("=" * 60)

base_url = "https://api.energidataservice.dk/dataset/Elspotprices"

# ── Test 1: Simplest possible call (no filter, just 5 records) ────
print("\n📋 TEST 1: Simple call - no filter, last 5 records")
url1 = f"{base_url}?limit=5"
print(f"   URL: {url1}")
try:
    r = requests.get(url1, timeout=30)
    print(f"   Status: {r.status_code}")
    data = r.json()
    print(f"   Response keys: {list(data.keys())}")
    print(f"   Total records available: {data.get('total', '?')}")
    print(f"   Records returned: {len(data.get('records', []))}")
    if data.get("records"):
        rec = data["records"][0]
        print(f"   First record keys: {list(rec.keys())}")
        print(f"   First record: {json.dumps(rec, indent=2, default=str)}")
        print("   ✅ TEST 1 PASSED")
    else:
        print(f"   ⚠️  No records. Full response: {json.dumps(data, indent=2)[:500]}")
except Exception as e:
    print(f"   ❌ FAILED: {e}")

# ── Test 2: With filter using params dict ─────────────────────────
print("\n📋 TEST 2: Filter via requests params dict")
url2 = base_url
params2 = {
    "limit": 5,
    "filter": '{"PriceArea":["DK1"]}',
    "sort": "HourUTC asc",
}
print(f"   URL: {url2}")
print(f"   Params: {params2}")
try:
    r = requests.get(url2, params=params2, timeout=30)
    print(f"   Status: {r.status_code}")
    print(f"   Actual URL sent: {r.url}")
    data = r.json()
    print(f"   Total: {data.get('total', '?')}, Records: {len(data.get('records', []))}")
    if data.get("records"):
        print(f"   First record PriceArea: {data['records'][0].get('PriceArea', '?')}")
        print("   ✅ TEST 2 PASSED")
    else:
        print(f"   ⚠️  No records. Response: {json.dumps(data, indent=2)[:500]}")
except Exception as e:
    print(f"   ❌ FAILED: {e}")

# ── Test 3: With filter baked into URL string ─────────────────────
print("\n📋 TEST 3: Filter baked directly into URL string")
url3 = f'{base_url}?limit=5&filter={{"PriceArea":["DK1"]}}&sort=HourUTC%20asc'
print(f"   URL: {url3}")
try:
    r = requests.get(url3, timeout=30)
    print(f"   Status: {r.status_code}")
    data = r.json()
    print(f"   Total: {data.get('total', '?')}, Records: {len(data.get('records', []))}")
    if data.get("records"):
        print(f"   First record PriceArea: {data['records'][0].get('PriceArea', '?')}")
        print("   ✅ TEST 3 PASSED")
    else:
        print(f"   ⚠️  No records. Response: {json.dumps(data, indent=2)[:500]}")
except Exception as e:
    print(f"   ❌ FAILED: {e}")

# ── Test 4: With date range ───────────────────────────────────────
print("\n📋 TEST 4: With date range (last 7 days)")
from datetime import datetime, timedelta
start = (datetime.utcnow() - timedelta(days=7)).strftime("%Y-%m-%d")
end = datetime.utcnow().strftime("%Y-%m-%d")
url4 = f'{base_url}?start={start}T00:00&end={end}T00:00&filter={{"PriceArea":["DK1"]}}&limit=5'
print(f"   URL: {url4}")
try:
    r = requests.get(url4, timeout=30)
    print(f"   Status: {r.status_code}")
    data = r.json()
    print(f"   Total: {data.get('total', '?')}, Records: {len(data.get('records', []))}")
    if data.get("records"):
        print("   ✅ TEST 4 PASSED")
    else:
        print(f"   ⚠️  No records. Response: {json.dumps(data, indent=2)[:500]}")
except Exception as e:
    print(f"   ❌ FAILED: {e}")

# ── Test 5: Old-style filter (non-array) ──────────────────────────
print("\n📋 TEST 5: Old-style filter (non-array value)")
url5 = f'{base_url}?limit=5&filter={{"PriceArea":"DK1"}}'
print(f"   URL: {url5}")
try:
    r = requests.get(url5, timeout=30)
    print(f"   Status: {r.status_code}")
    data = r.json()
    print(f"   Total: {data.get('total', '?')}, Records: {len(data.get('records', []))}")
    if data.get("records"):
        print("   ✅ TEST 5 PASSED")
    else:
        print(f"   ⚠️  No records. Response: {json.dumps(data, indent=2)[:500]}")
except Exception as e:
    print(f"   ❌ FAILED: {e}")

print("\n" + "=" * 60)
print("Done! Copy ALL output above and paste it to Claude.")
print("=" * 60)
