"""Split a CV into section-level chunks.

This is the one real design decision in the project; everything else is wiring.

An embedding compresses a passage into a single point. Feed it a whole CV and the
point lands in the average of everything the document says -- a three-page CV
covering React Native, warehouse logistics and a chemistry degree embeds to
somewhere near none of them. That is dilution, and it is why chunking exists.

Three options were considered (docs/chunking.md has the long version):

  one chunk per CV        rejected -- a CV is a multi-topic document, so a query
                          about one narrow skill competes against every unrelated
                          paragraph sharing its vector
  fixed 500-token windows rejected -- fine for prose, but on a CV it cuts
                          mid-sentence and splits a job title from the bullets
                          underneath it, ignoring structure the document already has
  one chunk per section   CHOSEN -- split on the headings a CV already carries,
                          and split Experience further, one chunk per role

Each chunk then represents one coherent claim. The consequence is that one
candidate produces six or seven vectors, so a top-20 search can return the same
person five times; search.py handles that by collapsing on candidate_id.
"""

import re
from typing import List, NamedTuple


class Chunk(NamedTuple):
    section: str  # human-readable label, shown to the recruiter as the match reason
    text: str


MIN_CHUNK_CHARS = 60

# Heading vocabulary. CVs are not consistent about these -- "Employment History",
# "Career Summary" and "Technical Competencies" all show up in the wild -- so the
# list is broad rather than canonical.
_HEADING_WORDS = [
    "summary", "professional summary", "career summary", "profile",
    "professional profile", "personal profile", "objective", "career objective",
    "about me", "about",
    "experience", "work experience", "professional experience", "employment",
    "employment history", "work history", "career history",
    "skills", "technical skills", "key skills", "core skills", "core competencies",
    "technical competencies", "competencies", "expertise", "areas of expertise",
    "education", "academic background", "academic qualifications", "qualifications",
    "projects", "key projects", "selected projects",
    "certifications", "certificates", "training", "courses",
    "achievements", "accomplishments", "awards",
    "languages", "references", "personal details", "personal information",
    "interests", "hobbies", "publications", "volunteering",
]

# A heading occupies its own line. Requiring that -- rather than merely matching
# at the start of a line -- is what stops a sentence like "Experience in payment
# systems includes..." from being treated as a section break. That false split is
# easy to miss because it produces plausible-looking chunks.
_HEADING = re.compile(
    r"^[ \t]*(?:[-*•#]\s*)?(?:"
    + "|".join(sorted(_HEADING_WORDS, key=len, reverse=True))
    + r")[ \t]*[:–—-]*[ \t]*$",
    re.IGNORECASE | re.MULTILINE,
)

# Sections whose body is a list of roles, and should be split one chunk per role.
_ROLE_SECTIONS = re.compile(
    r"\b(experience|employment|work history|career history)\b", re.IGNORECASE
)

# A date range is what actually marks the start of a new role. Matches
# "Jan 2021 - Present", "2019 to 2021", "March 2020 - Dec 2022", "2020-Now".
_MONTH = r"(?:jan|feb|mar|apr|may|jun|jul|aug|sep|sept|oct|nov|dec)[a-z]*\.?"
_DATE_RANGE = re.compile(
    r"(?:" + _MONTH + r"[ \t]*)?(?:19|20)\d{2}"
    r"[ \t]*(?:[-–—]|\bto\b|\buntil\b)[ \t]*"
    r"(?:present|current|now|date|(?:" + _MONTH + r"[ \t]*)?(?:19|20)\d{2})",
    re.IGNORECASE,
)


def _clean(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    return re.sub(r"\n{3,}", "\n\n", text).strip()


def _label(block: str, fallback: str) -> str:
    """First non-empty line of a block -- used as the chunk's section label."""
    for line in block.split("\n"):
        line = line.strip(" \t:-–—")
        if line:
            return line[:60]
    return fallback


def split_roles(body: str) -> List[str]:
    """Split an experience section into one block per role, on date ranges.

    The date range is what reliably marks a role, but CVs put it in one of two
    places and the difference matters:

        Senior Engineer, Emirates NBD        title on its own line, date below
        Jan 2021 - Present                   -> the boundary is the line ABOVE

        Sterling Perfumes LLC Sep 2024 - Present    employer and date together
        Lab Assistant, Perfumery Department          -> the boundary is THIS line

    Backtracking unconditionally breaks the second layout: the line above is the
    previous role's last bullet, so that bullet gets torn off its own role,
    prepended to the next one, and becomes its label. Nothing errors -- the
    chunks just quietly describe the wrong job. Found on a real CV; the synthetic
    corpus only had the first layout.

    So: strip the date out of the line and see what is left. Residual text means
    the dated line carries its own employer or title and is already the header.
    """
    lines = body.split("\n")
    dated = [i for i, line in enumerate(lines) if _DATE_RANGE.search(line)]
    if len(dated) < 2:
        return [body]

    bounds = []
    for i in dated:
        match = _DATE_RANGE.search(lines[i])
        rest = (lines[i][:match.start()] + lines[i][match.end():]).strip(" \t|,;:-–—·•()[]")
        is_bare_date = len(rest) < 3
        if (is_bare_date and i > 0 and lines[i - 1].strip()
                and not _DATE_RANGE.search(lines[i - 1])):
            i -= 1
        bounds.append(i)

    # Anything above the first boundary is the section heading and any preamble.
    # It is left as its own block; the caller drops it if it is too short to be
    # worth a vector, which for a bare heading it always is.
    cuts = [0] + sorted({b for b in bounds if b > 0}) + [len(lines)]
    blocks = ["\n".join(lines[a:b]).strip() for a, b in zip(cuts, cuts[1:])]
    return [b for b in blocks if b] or [body]


def split_sections(text: str) -> List[Chunk]:
    """Split CV text into section-level chunks.

    Falls back to a single chunk when no headings are found -- some CVs are one
    unstructured block, and that is a normal case, not an error.
    """
    text = _clean(text)
    if not text:
        return []

    cuts = [m.start() for m in _HEADING.finditer(text)]
    if not cuts:
        return [Chunk(section="Full CV", text=text)]

    # Everything before the first heading is the name / contact header.
    bounds = ([0] if cuts[0] > 0 else []) + cuts + [len(text)]
    chunks: List[Chunk] = []

    for a, b in zip(bounds, bounds[1:]):
        block = text[a:b].strip()
        if not block:
            continue
        label = _label(block, "Section")

        if _ROLE_SECTIONS.search(label):
            for role in split_roles(block):
                if len(role) >= MIN_CHUNK_CHARS:
                    # Label a role by its own first line (title / company) rather
                    # than the section heading -- that line is what a recruiter reads.
                    chunks.append(Chunk(section=_label(role, label), text=role))
        elif len(block) >= MIN_CHUNK_CHARS:
            # Below the floor is almost always a heading that matched with nothing
            # under it. Cheap guard, saves a lot of useless vectors.
            chunks.append(Chunk(section=label, text=block))

    # A CV of nothing but short sections would otherwise index as nothing at all.
    return chunks or [Chunk(section="Full CV", text=text)]


if __name__ == "__main__":
    # The test suite, in the style of the sibling repos: run the file.
    standard = """Ahmed Al-Rashid
ahmed.rashid@example.com | +971 50 111 2233 | Dubai, UAE

Professional Summary
Backend engineer with eight years building transaction processing systems for
regional banks. Comfortable owning a service from schema to on-call rotation.

Work Experience

Senior Backend Engineer, Emirates NBD
Jan 2021 - Present
Built the settlement reconciliation service that matches inbound card
transactions against issuer statements overnight. Cut manual investigation from
four hours a day to twenty minutes.

Backend Engineer, Network International
Mar 2017 - Dec 2020
Maintained the authorisation gateway handling roughly 900 requests per second at
peak. Owned the retry and idempotency logic.

Technical Skills
Python, Go, PostgreSQL, Kafka, Kubernetes, Terraform

Education
BSc Computer Science, American University of Sharjah, 2016
"""

    chunks = split_sections(standard)
    labels = [c.section for c in chunks]
    assert len(chunks) >= 5, f"expected the CV to split, got {len(chunks)}: {labels}"
    assert any("Summary" in l for l in labels), labels
    assert any("Emirates NBD" in l for l in labels), labels
    assert any("Network International" in l for l in labels), labels

    # The two roles must land in separate chunks, or a query about one competes
    # against the text of the other -- the whole point of role-level splitting.
    enbd = next(c for c in chunks if "Emirates NBD" in c.section)
    assert "Network International" not in enbd.text, "roles were not split"
    assert "settlement reconciliation" in enbd.text

    # The other real-world layout: employer and dates share a line, with the job
    # title underneath. Regression test for a bug found on a live CV, where the
    # boundary backtracked into the previous role and carried its last bullet
    # across -- so a chunk about one job was labelled with a line from another.
    inline_dates = """Priya Raman
priya.raman@example.com | Sharjah, UAE

WORK EXPERIENCE
Sterling Perfumes Industries LLC Sep 2024 - Present
Lab Assistant, Perfumery Department
Optimised five fine fragrance formulations, cutting raw material cost 12%.
Execute 25+ sample compounding cycles per month plus daily QC checks.
Faan Al Ibdaa Perfumes Jun 2023 - Jul 2024
Chemist and QC, Perfumery Department
Developed novel formulations within 1% of target cost deviation.
Conducted QC testing on every finished batch and prepared documentation.
"""
    roles = [c for c in split_sections(inline_dates) if "Perfumes" in c.section]
    assert len(roles) == 2, [c.section for c in split_sections(inline_dates)]
    sterling = next(c for c in roles if "Sterling" in c.section)
    faan = next(c for c in roles if "Faan" in c.section)
    # The bullet belongs to Sterling and must not lead the Faan chunk.
    assert "compounding cycles" in sterling.text, sterling.text
    assert "compounding cycles" not in faan.text, faan.text
    assert "Faan" not in sterling.text, sterling.text

    # Fallback: an unstructured CV is one chunk, not zero.
    blob = ("Maria Santos, warehouse supervisor in Jebel Ali with eleven years "
            "moving palletised freight. Managed a team of fourteen. Reduced pick "
            "errors by a third over two years. Contact mariasantos@example.com.")
    fb = split_sections(blob)
    assert len(fb) == 1 and fb[0].section == "Full CV", fb

    # A heading with nothing under it must not become a vector.
    sparse = standard + "\n\nReferences\n\nInterests\n"
    assert not any(c.section == "References" for c in split_sections(sparse)), \
        "empty section was indexed"

    # The false-split guard: a sentence that begins with a heading word is prose.
    prose = """Priya Nair
priya.nair@example.com

Profile
Experience in payment systems is the core of my background, spanning nine years.
Skills in distributed tracing were picked up along the way and are now central.
"""
    ps = split_sections(prose)
    # One chunk: the short contact header falls under the floor, and the two
    # sentences stay inside Profile. Three chunks would mean "Experience in..."
    # and "Skills in..." had been read as section headings.
    assert [c.section for c in ps] == ["Profile"], \
        f"prose mistaken for headings: {[c.section for c in ps]}"
    assert "distributed tracing" in ps[0].text and "payment systems" in ps[0].text

    assert split_sections("") == []
    assert all(len(c.text) >= MIN_CHUNK_CHARS for c in chunks)

    print(f"chunker: all assertions passed ({len(chunks)} chunks from the sample CV)")
    for c in chunks:
        print(f"  [{len(c.text):4d} chars] {c.section}")
