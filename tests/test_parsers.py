from dep_tools.parsers import datetime_parser, bool_parser


def test_datetime_parser():
    years = datetime_parser("1990_1995")
    assert years == [str(y) for y in [1990, 1991, 1992, 1993, 1994, 1995]]

def test_datetime_parser_single_year():
    assert datetime_parser("1795") == "1795"

def test_datetime_parser_other_splitter():
    assert datetime_parser("2001/2003") == ["2001", "2002", "2003"]

def test_bool_parser_true():
    assert bool_parser("True")

def test_bool_parser_false():
    assert not bool_parser("False")
