from collections import defaultdict
import random

import pytest

import WikiWho.wikiwho as wikiwho_module
from WikiWho.structures import Paragraph, Revision, Sentence, Word
from WikiWho.utils import (
    TOKEN_SYMBOLS,
    split_into_paragraphs,
    split_into_tokens,
)


native = pytest.importorskip("WikiWho._structural_native")
native.configure_token_symbols(TOKEN_SYMBOLS)


@pytest.mark.parametrize("text", [
    "",
    "| || ææææ xææææ",
    "[[[link]]] {{{template}}}",
    "<!--comment-->",
    "'''''Yeah Yeah Yeahs'''''",
    "日本語 ＡＢＣ a日本b",
    "<ref name='x'>{{cite web|title=One}}</ref>",
])
def test_native_tokenizer_matches_python_contract(text):
    assert native.split_into_tokens(text) == split_into_tokens(text)


@pytest.mark.parametrize("text", [
    "",
    "a\r\nb\rc",
    "x<table>y</table>z",
    "<tr>|-\n</tr>",
    "{|\n|-\n  |}\n",
    "{{#if:1\n|yes\n|}}\nafter",
    "|-<tr>",
    "\n\n|-\n",
])
def test_native_paragraph_splitter_matches_python_contract(text):
    assert native.split_into_paragraphs(text) == split_into_paragraphs(text)


def test_native_splitters_match_randomized_markup_contract():
    random_source = random.Random(20260807)
    atoms = [
        "a", "b", " ", "\t", "\n", "\r", "\r\n", "|", "}", "-",
        "<table>", "</table>", "<tr>", "</tr>", "{|", "|}", "|-\n",
        "[[", "]]", "{{", "}}", "<!--", "-->", "日本語", "🙂",
    ]
    for _ in range(2000):
        text = "".join(
            random_source.choice(atoms)
            for _ in range(random_source.randrange(0, 40))
        )
        assert native.split_into_paragraphs(text) == split_into_paragraphs(text)
        assert native.split_into_tokens(text) == split_into_tokens(text)


def _subsequence_count(tokens, needle):
    if not needle:
        return 0
    count = 0
    for start in range(len(tokens) - len(needle) + 1):
        if tuple(tokens[start:start + len(needle)]) == needle:
            count += 1
            if count == 2:
                break
    return count


def test_native_subsequence_counter_matches_exact_randomized_contract():
    random_source = random.Random(20260807)
    alphabet = ["a", "b", "c", "|", "{{", "}}", "日本", "🙂"]
    for _ in range(2000):
        tokens = [
            random_source.choice(alphabet)
            for _ in range(random_source.randrange(0, 100))
        ]
        needle = tuple(
            random_source.choice(alphabet)
            for _ in range(random_source.randrange(0, 16))
        )
        positions = defaultdict(list)
        for index in range(len(tokens) - 1):
            positions[(tokens[index], tokens[index + 1])].append(index)
        if len(needle) >= 2:
            first_positions = positions.get((needle[0], needle[1]), ())
            last_positions = positions.get((needle[-2], needle[-1]), ())
        else:
            first_positions = last_positions = ()
        assert native.count_subsequence_at_positions(
            tokens, needle, first_positions, last_positions,
        ) == _subsequence_count(tokens, needle)


def _make_document(paragraphs):
    values = []
    ranges = {}
    for paragraph_index, paragraph in enumerate(paragraphs):
        start = len(values)
        values.extend(paragraph)
        ranges[paragraph_index] = (start, len(values))
    return wikiwho_module._StructuralDocument(values, ranges, {})


def test_native_structural_kernels_match_exact_python_fallback(monkeypatch):
    anchor = ["anchor{}".format(index) for index in range(12)]
    duplicate = ["shared", "duplicate", "lineage", "window"]
    previous_paragraphs = [
        ["old"] + anchor + ["tail"],
        duplicate + ["separator"] + duplicate,
    ]
    current_paragraphs = [
        duplicate + ["changed"] + duplicate,
        ["new"] + anchor + ["tail"],
    ]

    native_previous = _make_document(previous_paragraphs)
    native_current = _make_document(current_paragraphs)
    native_previous.ensure_index()
    native_current.ensure_index()

    duplicate_candidates = {tuple(duplicate), tuple(anchor[:4])}
    native_duplicates = (
        wikiwho_module._duplicated_candidate_windows_in_document(
            native_previous, duplicate_candidates,
        )
    )
    native_chains = wikiwho_module._compact_structural_anchor_chains(
        native_previous, native_current, {0, 1}, {0, 1},
    )

    residual_by_article = {}
    residual_flags = bytearray(len(native_previous.values))
    for paragraph_index, (start, end) in (
            native_previous.paragraph_ranges.items()):
        for article_index in range(start, end):
            if article_index == 2:
                continue
            residual_by_article[article_index] = (
                article_index, (paragraph_index, 0, article_index - start),
            )
            residual_flags[article_index] = 1
    used_previous = {1: 0}
    native_available = wikiwho_module._compact_available_residual_windows(
        native_previous, residual_by_article,
        lambda index: index not in used_previous,
        native_availability=used_previous,
        native_previous_mode=True,
    )
    paragraph_length = native_previous.paragraph_length(0)
    native_residual = (
        wikiwho_module._compact_residual_structural_window_keys(
            native_previous, 0, 0, paragraph_length,
            residual_by_article, residual_flags=residual_flags,
        )
    )

    previous_lcs = ["left", "said", "said", "middle", "right"]
    current_lcs = ["left", "said", "middle", "right"]
    native_lcs = wikiwho_module._lcs_token_pairs(
        previous_lcs, current_lcs, previous_lcs, current_lcs,
    )

    ledger = wikiwho_module._MatchCandidateLedger(
        len(native_previous.values), len(native_current.values),
    )
    ledger.propose(0, 0, 100, "test")
    native_unresolved = wikiwho_module._unresolved_residual_windows(
        ledger, native_previous.values, native_current.values,
    )

    monkeypatch.setattr(wikiwho_module, "_structural_native", None)
    python_previous = _make_document(previous_paragraphs)
    python_current = _make_document(current_paragraphs)
    python_previous.ensure_index()
    python_current.ensure_index()

    assert native_previous.keys == python_previous.keys
    assert list(native_previous.informative_prefix) == (
        python_previous.informative_prefix
    )
    assert native_duplicates == (
        wikiwho_module._duplicated_candidate_windows_in_document(
            python_previous, duplicate_candidates,
        )
    ) == {tuple(duplicate)}
    assert native_chains == wikiwho_module._compact_structural_anchor_chains(
        python_previous, python_current, {0, 1}, {0, 1},
    )
    assert native_chains
    assert native_available == (
        wikiwho_module._compact_available_residual_windows(
            python_previous, residual_by_article,
            lambda index: index not in used_previous,
            native_availability=used_previous,
            native_previous_mode=True,
        )
    )
    assert native_residual == (
        wikiwho_module._compact_residual_structural_window_keys(
            python_previous, 0, 0, paragraph_length,
            residual_by_article, residual_flags=residual_flags,
        )
    )
    assert native_lcs == wikiwho_module._lcs_token_pairs(
        previous_lcs, current_lcs, previous_lcs, current_lcs,
    )
    assert native_unresolved == wikiwho_module._unresolved_residual_windows(
        ledger, python_previous.values, python_current.values,
    )


def _make_paragraph(label, sentence_values, persistent):
    paragraph = Paragraph()
    paragraph.hash_value = "paragraph-{}".format(label)
    sentence = Sentence()
    sentence.hash_value = "sentence-{}".format(label)
    if persistent:
        for value in sentence_values:
            word = Word()
            word.value = value
            sentence.words.append(word)
    else:
        sentence.splitted.extend(sentence_values)
    paragraph.sentences = {sentence.hash_value: [sentence]}
    paragraph.ordered_sentences = [sentence.hash_value]
    return paragraph, sentence


def _make_revision(paragraphs):
    revision = Revision()
    revision.paragraphs = {}
    revision.ordered_paragraphs = []
    for paragraph in paragraphs:
        revision.paragraphs.setdefault(paragraph.hash_value, []).append(
            paragraph
        )
        revision.ordered_paragraphs.append(paragraph.hash_value)
    return revision


def _document_snapshot(document):
    if document is None:
        return None
    return (
        document.values,
        document.paragraph_ranges,
        document.sentence_ranges,
    )


def test_native_document_pair_matches_fail_closed_python_builder(monkeypatch):
    shared, shared_sentence = _make_paragraph(
        "shared", ["stable", "shared", "paragraph"], True,
    )
    previous_only, previous_sentence = _make_paragraph(
        "previous", ["old", "duplicate", "lineage"], True,
    )
    current_only, current_sentence = _make_paragraph(
        "current", ["new", "duplicate", "lineage"], False,
    )
    previous = _make_revision([previous_only, shared])
    current = _make_revision([shared, current_only])
    targets = ({previous_sentence}, {current_sentence})

    native_all = wikiwho_module._revision_structural_document_pair(
        previous, current,
    )
    native_targeted = wikiwho_module._revision_structural_document_pair(
        previous, current, *targets,
    )

    monkeypatch.setattr(wikiwho_module, "_structural_native", None)
    python_all = wikiwho_module._revision_structural_document_pair(
        previous, current,
    )
    python_targeted = wikiwho_module._revision_structural_document_pair(
        previous, current, *targets,
    )

    assert tuple(map(_document_snapshot, native_all)) == tuple(
        map(_document_snapshot, python_all)
    )
    assert tuple(map(_document_snapshot, native_targeted)) == tuple(
        map(_document_snapshot, python_targeted)
    )
    assert shared_sentence in native_all[0].sentence_ranges
    assert shared_sentence not in native_targeted[0].sentence_ranges
