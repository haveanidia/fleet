from fleet.release import ALLOWED_PUBLIC_LICENSES


def test_public_license_allowlist():
    assert "apache-2.0" in ALLOWED_PUBLIC_LICENSES
    assert "mit" in ALLOWED_PUBLIC_LICENSES
    assert None not in ALLOWED_PUBLIC_LICENSES


def test_unknown_license_is_not_public():
    value = None
    normalized = str(value or "").lower()
    assert normalized not in ALLOWED_PUBLIC_LICENSES
