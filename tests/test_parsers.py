from dep_tools.parsers import datetime_parser, bool_parser


def test_datetime_parser():
    years = datetime_parser("1990_1995")
    assert years == [str(y) for y in [1990, 1991, 1992, 1993, 1994, 1995]]


def test_bool_parser_true():
    assert bool_parser("True")


def test_bool_parser_false():
    assert not bool_parser("False")
