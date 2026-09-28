def test_blank_expert_name_is_not_valid():
    expert = str("" or "").strip()
    assert not expert


def test_colab_python_variable_expansion_contract():
    # Jupyter ! commands interpolate Python values with {NAME}, not $NAME.
    template = '--expert "{EXPERT}" --output "dist/{EXPERT}.fleet"'
    assert "$EXPERT" not in template
    assert "{EXPERT}" in template
