import spacy
import re
nlp = spacy.load("en_core_web_sm")


def remove_pii_name(text: str, replacement: str = '--NAME--') -> str:
    """
    Pass a string and return the text string with names removed

    Parameters
    ----------
    text : str
        The text to replace names in
    replacement : str
        The text to replace names with

    Returns
    -------
    replacement_text: str
        A text string with the name removed and replaced with specified text

    Example
    -------
    text_with_name = "I was working the forklift and then a" \
    " bowling ball fell from the sky and hit " \
    "Oscar in the head."

    print(remove_pii_name(text_with_name))

    > I was working the forklift and then a bowling ball fell
    > from the sky and hit --NAME-- in the head.
    """
    # Spacy Doc form text
    doc = nlp(text)
    # Extract named person entities
    filtered_entities = [entity for entity
                         in doc.ents
                         if entity.label_.lower() in ["person"]
                         ]
    # Get copy of text
    replacement_text = doc.text

    # Iterate through entities and replace
    for entity in filtered_entities:
        replacement_text = replacement_text.replace(entity.text, replacement)

    return replacement_text


def remove_pii_phone(text: str, replacement: str = '--PHONE--') -> str:
    """
    Pass a string and return the text string with phone numbers removed

    Parameters
    ----------
    text : str
        The text to replace phone numbers in
    replacement : str
        The text to replace phone numbers with

    Returns
    -------
    replacement_text: str
        A text string with the phone numbers removed and
        replaced with specified text

    Example
    -------
    text_with_phone = "Give me a call at 109-876-5432"

    print(remove_pii_phone(text_with_phone))

    > Give me a call at --PHONE--
    """
    phone_regex_pattern = re.compile(
        # core components of a phone number
        r"(?:^|(?<=[^\w)]))(\+?1[ .-]?)?(\(?\d{3}\)?[ .-]?)?(\d{3}[ .-]?\d{4})"
        # extensions, etc.
        r"(\s?(?:ext\.?|[#x-])\s?\d{2,6})?(?:$|(?=\W))",
        flags=re.UNICODE | re.IGNORECASE)

    replacement_text = re.sub(phone_regex_pattern, replacement, text)
    return replacement_text


def remove_pii_email(text: str, replacement: str = '--EMAIL--') -> str:
    """
    Pass a string and return the text string with email numbers removed

    Parameters
    ----------
    text : str
        The text to replace email numbers in
    replacement : str
        The text to replace email numbers with

    Returns
    -------
    replacement_text: str
        A text string with the email numbers removed and
        replaced with specified text

    Example
    -------
    text_with_email = "I can be reached at human_bean@me.com"

    print(remove_pii_email(text_with_email))

    > I can be reached at --EMAIL--
    """
    email_regex_pattern = re.compile(
        r"(?:mailto:)?"
        r"(?:^|(?<=[^\w@.)]))([\w+-](\.(?!\.))?)*?[\w+-]@(?:\w-?)*?\w+(\.([a-z]{2,})){1,3}"
        r"(?:$|(?=\b))",
        flags=re.UNICODE | re.IGNORECASE)

    replacement_text = re.sub(email_regex_pattern, replacement, text)
    return replacement_text


def remove_pii_ssn(text: str, replacement: str = '--SSN--') -> str:
    """
    Pass a string and return the text string with SSN numbers removed

    Parameters
    ----------
    text : str
        The text to replace SSN numbers in
    replacement : str
        The text to replace SSN numbers with

    Returns
    -------
    replacement_text: str
        A text string with the SSN numbers removed and
        replaced with specified text

    Example
    -------
    text_with_ssn = "My social security number is 123-45-6789"

    print(remove_pii_ssn(text_with_ssn))

    > My social security number is --SSN--
    """
    ssn_regex_pattern = re.compile(
        r'(?P<area>[0-9]{3})-?(?P<group>[0-9]{2})-?(?P<serial>[0-9]{4})'
    )

    replacement_text = re.sub(ssn_regex_pattern, replacement, text)
    return replacement_text


text_with_ssn = "My social security number is 123-45-6789"
print(remove_pii_ssn(text_with_ssn))

text_with_phone = "Give me a call at 109-876-5432"
print(remove_pii_phone(text_with_phone))

text_with_name = "I was working the forklift and then a" \
    " bowling ball fell from the sky and hit " \
    "Oscar in the head."
print(remove_pii_name(text_with_name))

text_with_email = "I can be reached at human_bean@me.com"
print(remove_pii_email(text_with_email))
