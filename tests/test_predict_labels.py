from Predict import parse_label_names


def test_parse_label_names_json_list():
    assert parse_label_names('["ok","toxic"]') == ["ok", "toxic"]


def test_parse_label_names_csv_fallback():
    assert parse_label_names("ok,toxic") == ["ok", "toxic"]
    assert parse_label_names("[ok,toxic]") == ["ok", "toxic"]


def test_parse_label_names_strips_quotes_and_spaces():
    assert parse_label_names(" 'ok' , \"toxic\" ") == ["ok", "toxic"]
