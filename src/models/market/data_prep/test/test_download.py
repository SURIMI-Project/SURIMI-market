from pathlib import Path
from unittest.mock import MagicMock
import zipfile

import pytest

from src.models.market.data_prep.download import get_current_zip_filename, get_current_zip_url, download_and_extract


def test_get_current_zip_filename_extracts_dated_filename():
    xml = (
        "<record><citationIdentifier>EMODnet_HA_Fisheries_FirstSales_20250319"
        "</citationIdentifier><link>.../EMODnet_HA_Fisheries_FirstSales_20250319.zip"
        "</link></record>"
    )
    assert get_current_zip_filename(xml) == "EMODnet_HA_Fisheries_FirstSales_20250319.zip"


def test_get_current_zip_filename_raises_when_not_found():
    with pytest.raises(ValueError):
        get_current_zip_filename("<record>no filename here</record>")


def test_get_current_zip_url_builds_full_url():
    mock_session = MagicMock()
    mock_session.get.return_value.text = (
        "<record>EMODnet_HA_Fisheries_FirstSales_20250920.zip</record>"
    )
    mock_session.get.return_value.raise_for_status.return_value = None

    url = get_current_zip_url(session=mock_session)

    assert url == (
        "https://ows.emodnet-humanactivities.eu/geonetwork/srv/api/records/"
        "ff5f738f-94ee-4704-94c2-7c951c931973/attachments/"
        "EMODnet_HA_Fisheries_FirstSales_20250920.zip"
    )


def test_download_and_extract_finds_single_gdb(tmp_path):
    source_dir = tmp_path / "source"
    gdb_dir = source_dir / "Fake_20250920.gdb"
    gdb_dir.mkdir(parents=True)
    (gdb_dir / "placeholder.txt").write_text("fake gdb contents")

    zip_path = tmp_path / "fake.zip"
    with zipfile.ZipFile(zip_path, "w") as zf:
        for file in gdb_dir.rglob("*"):
            zf.write(file, file.relative_to(source_dir))

    mock_session = MagicMock()
    with open(zip_path, "rb") as f:
        zip_bytes = f.read()
    mock_session.get.return_value.iter_content.return_value = [zip_bytes]
    mock_session.get.return_value.raise_for_status.return_value = None

    dest_dir = tmp_path / "dest"
    result = download_and_extract("http://fake-url/fake.zip", dest_dir, session=mock_session)

    assert result == dest_dir / "Fake_20250920.gdb"
    assert (result / "placeholder.txt").exists()
