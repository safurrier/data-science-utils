from src.utils.text.pii import (
    remove_pii_ssn,
    remove_pii_phone,
    remove_pii_name,
    remove_pii_email
)


def test_remove_pii_ssn():
    text_with_ssn = "My social security number is 123-45-6789"
    assert remove_pii_ssn(text_with_ssn) == \
        "My social security number is --SSN--", 'test failed'


def test_remove_pii_phone():
    text_with_phone = "Give me a call at 109-876-5432"
    assert remove_pii_phone(text_with_phone) == \
        "Give me a call at --PHONE--", 'test failed'


def test_remove_pii_email():
    text_with_email = "I can be reached at human@me.com"
    assert remove_pii_email(text_with_email) == \
        "I can be reached at --EMAIL--", 'test failed'


def test_remove_pii_name():
    text_with_name = "I was working the forklift and then a" \
        " bowling ball fell from the sky and hit " \
        "Oscar in the head."
    assert remove_pii_name(text_with_name) == \
        "I was working the forklift and then a " \
        "bowling ball fell from the sky and hit" \
        " --NAME-- in the head.", 'test failed'
