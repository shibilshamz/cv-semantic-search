"""Pull contact details back out of an indexed CV.

The index stores chunk text and four metadata fields; a phone number is not one
of them. It is in the text -- CVs put it in the contact header, on the same line
as the email -- so it can be recovered without reindexing anything, which is the
only reason this module exists rather than a migration.

Two rules keep it from inventing numbers:

  * only the first few lines of the *header* chunk are searched. A CV body is
    full of digits that are not phone numbers -- "handled 900 requests/sec",
    "cut cost 12%", "Nov 2019 - Present" -- and a regex loose enough to catch
    every real format will catch those too if you let it see them.
  * no match means an empty string, never a guess. A recruiter dialling a
    number scraped out of a bullet point is worse than one seeing a blank.
"""

import re

# International first (+971 50 118 7742, 00971501187742), then a local mobile
# (050 118 7742). Both require enough digits that a year or a percentage cannot
# satisfy them.
PHONE_RE = re.compile(
    r"(?:\+|\b00)\s?\d[\d\s().\-]{6,18}\d"
    r"|\b0\d{1,2}[\s.\-]?\d{3}[\s.\-]?\d{4}\b"
)

HEADER_LINES = 6


def phone_from(header_text: str) -> str:
    """First plausible phone number in a CV's contact header, or ""."""
    if not header_text:
        return ""
    for line in header_text.split("\n")[:HEADER_LINES]:
        found = PHONE_RE.search(line)
        if found:
            return re.sub(r"\s+", " ", found.group(0)).strip(" .-")
    return ""


if __name__ == "__main__":
    # The test suite, in the style of the sibling modules: run the file.
    cases = [
        ("Shibil Shamsudheen\n+971 50 118 7742 | a@b.com | Dubai, UAE", "+971 50 118 7742"),
        ("Grace Okonkwo\ngrace@example.com | +971 50 118 7742 | Sharjah", "+971 50 118 7742"),
        ("Name\n050 118 7742 | x@y.com", "050 118 7742"),
        ("Name\n00971501187742 | x@y.com", "00971501187742"),
        ("Name\nno phone here | x@y.com", ""),
        # The cases that matter: body text that looks numeric but is not a number
        # anyone should be dialling.
        ("Name\nNov 2019 - Present, managed 25 staff", ""),
        ("Name\nCut raw material cost 12% across 2024 and 2025", ""),
        ("Name\nHandled roughly 900 requests per second at peak", ""),
        ("", ""),
    ]
    for text, want in cases:
        got = phone_from(text)
        assert got == want, f"{text!r} -> {got!r}, expected {want!r}"

    # A number below the header must not be found, even if it is a real one.
    buried = "Name\nemail@x.com\n\n\n\n\n\n+971 50 118 7742"
    assert phone_from(buried) == "", "searched past the header"

    print(f"contact: all assertions passed ({len(cases) + 1} cases)")
