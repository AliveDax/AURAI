import unicodedata

from artrec.data_loader import artist_title, match_annotations, normalize_rel_path


def test_normalize_rel_path():
    nfd = unicodedata.normalize("NFD", "Images\\Sōsaku hanga\\x.jpg")
    assert normalize_rel_path(nfd) == unicodedata.normalize("NFC", "Images/Sōsaku hanga/x.jpg")


def test_artist_title_from_filename():
    assert artist_title("0006080_YuriZlotnikov-Signalseries.jpg") == ("Yuri Zlotnikov", "Signalseries")
    assert artist_title("FS-6883_01.jpg") == ("", "FS-6883_01")


def test_matching_methods_in_order(tmp_path):
    disk = {k: tmp_path / k for k in [
        "Images/A/0001_X-One.jpg",           # exact
        "Images/A/0002_Y-Two.jpg",           # case-insensitive
        "Images/B/0003_Z-Renamed.jpg",       # via extract log
        "Images/B/0004_W-Short.jpg",         # via (folder, numeric id)
    ]}
    entries = [{"image_path": p} for p in [
        "Images\\A\\0001_X-One.jpg", "Images\\A\\0002_y-two.JPG",
        "Images\\B\\0003_Z-Re:named.jpg", "Images\\B\\0004_W-ShortButLongerInTheAnnotation.jpg",
        "Images\\C\\9999_missing.jpg",
    ]]
    log = {"Images/B/0003_Z-Re:named.jpg": "Images/B/0003_Z-Renamed.jpg"}
    matched, unmatched, methods = match_annotations(entries, disk, log, ["image_path"])
    assert [m for _, _, m in matched] == ["exact", "case_insensitive", "extract_log", "numeric_id"]
    assert unmatched == ["Images/C/9999_missing.jpg"]
