import requests
import json
import openpyxl
from openpyxl.styles import Font

# Base URLs for NEON Data API
SITE_URL = "https://data.neonscience.org/api/v0/sites"
PRODUCT_URL = "https://data.neonscience.org/api/v0/products"

def get_all_sites(release=None):
    """
    Fetches information about all NEON field sites.
    """
    params = {}
    if release:
        params['release'] = release

    response = requests.get(SITE_URL, params=params)
    if response.status_code == 200:
        return response.json().get("data", [])
    else:
        print(f"Error {response.status_code}: {response.text}")
        return []

def get_all_products():
    """
    Fetches all NEON data products.
    """
    response = requests.get(PRODUCT_URL)
    if response.status_code == 200:
        return response.json().get("data", [])
    else:
        print(f"Error fetching products: {response.status_code}")
        return []

def build_site_product_map(products):
    """
    Builds a mapping of siteCode → list of product descriptions.
    """
    site_product_map = {}
    for product in products:
        code = product.get("productCode")
        name = product.get("productName")
        site_list = product.get("siteCodes") or []  # ← FIX: ensures it's always iterable
        for site in site_list:
            site_code = site.get("siteCode")
            if site_code:
                if site_code not in site_product_map:
                    site_product_map[site_code] = []
                site_product_map[site_code].append(f"{code} - {name}")
    return site_product_map

import textwrap

def print_site_summary(sites, site_product_map=None, to_screen=True, to_xlsx=False, xlsx_filename="site_summary.xlsx"):
    """
    Prints a summary of site information to screen and/or XLSX file.
    Each data product is placed in its own column in the spreadsheet.
    """
    if to_xlsx:
        wb = openpyxl.Workbook()
        ws = wb.active
        ws.title = "NEON Sites"

        # Determine max number of products across all sites
        max_products = max(len(site_product_map.get(site.get('siteCode'), [])) for site in sites)

        # Build headers
        base_headers = ["Site Code", "Site Name", "Site Type", "State", "Domain", "Latitude", "Longitude"]
        product_headers = [f"Product {i+1}" for i in range(max_products)]
        headers = base_headers + product_headers
        ws.append(headers)

        for cell in ws[1]:
            cell.font = Font(bold=True)

    for site in sites:
        site_code = site.get('siteCode')
        site_name = site.get('siteName')
        site_type = site.get('siteType')
        state = f"{site.get('stateName')} ({site.get('stateCode')})"
        domain = f"{site.get('domainCode')} - {site.get('domainName')}"
        latitude = site.get('siteLatitude')
        longitude = site.get('siteLongitude')
        products = site_product_map.get(site_code, [])

        if to_screen:
            print(f"\n{'='*80}")
            print(f"📍 Site Code: {site_code}")
            print(f"🏞️ Name     : {site_name}")
            print(f"🔧 Type     : {site_type}")
            print(f"🗺️ Location : {state}")
            print(f"🌐 Domain   : {domain}")
            print(f"📌 Lat/Lon  : {latitude}, {longitude}")
            print(f"📦 Data Products:")
            for product in products:
                print(f"   • {product}")
            if not products:
                print("   None listed")
            print(f"{'='*80}")

        if to_xlsx:
            row = [site_code, site_name, site_type, state, domain, latitude, longitude] + products
            # Pad row to match header length
            row += [""] * (len(headers) - len(row))
            ws.append(row)

    if to_xlsx:
        wb.save(xlsx_filename)
        print(f"\n✅ Site summary saved to '{xlsx_filename}'")

if __name__ == "__main__":
    sites = get_all_sites()
    products = get_all_products()
    site_product_map = build_site_product_map(products)
    print(f"Retrieved {len(sites)} sites and {len(products)} products.\n")
    print_site_summary(sites, site_product_map, to_screen=True, to_xlsx=True)
