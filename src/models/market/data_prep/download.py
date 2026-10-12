"""Downloads the current EMODnet First Sales dataset snapshot."""
import re
import zipfile
from pathlib import Path

import requests

METADATA_RECORD_URL = (
    "https://emodnet.ec.europa.eu/geonetwork/srv/api/records/"
    "ff5f738f-94ee-4704-94c2-7c951c931973/formatters/xml"
)
ATTACHMENT_BASE_URL = (
    "https://ows.emodnet-humanactivities.eu/geonetwork/srv/api/records/"
    "ff5f738f-94ee-4704-94c2-7c951c931973/attachments/"
)


def get_current_zip_filename(metadata_xml: str) -> str:
    """Extract the current dated ZIP filename (e.g.
    "EMODnet_HA_Fisheries_FirstSales_20250319.zip") from the record's XML
    metadata -- the filename changes with each twice-yearly update, so it
    is never hardcoded."""
    match = re.search(r"EMODnet_HA_Fisheries_FirstSales_\d{8}\.zip", metadata_xml)
    if match is None:
        raise ValueError(
            "Could not find a dated EMODnet_HA_Fisheries_FirstSales_*.zip "
            "filename in the metadata record."
        )
    return match.group(0)


def get_current_zip_url(session=None) -> str:
    """Query the EMODnet metadata record and return the current ZIP download URL."""
    session = session or requests
    response = session.get(METADATA_RECORD_URL, timeout=30)
    response.raise_for_status()
    filename = get_current_zip_filename(response.text)
    return ATTACHMENT_BASE_URL + filename


def download_and_extract(zip_url: str, dest_dir, session=None) -> Path:
    """Download the EMODnet ZIP to dest_dir and extract it. Returns the
    path to the extracted .gdb directory."""
    session = session or requests
    dest_dir = Path(dest_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)

    zip_path = dest_dir / "emodnet_first_sales.zip"
    response = session.get(zip_url, timeout=300, stream=True)
    response.raise_for_status()
    with open(zip_path, "wb") as f:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)

    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(dest_dir)

    gdb_paths = list(dest_dir.glob("*.gdb"))
    if len(gdb_paths) != 1:
        raise ValueError(
            f"Expected exactly one .gdb directory after extraction, found {len(gdb_paths)}."
        )
    return gdb_paths[0]
