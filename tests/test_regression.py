import json
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from WikiWho.structures import Word
from WikiWho.wikiwho import (
    Wikiwho,
    _can_partially_restore_historical_sentence,
    _has_template_name_spacing_change,
    _match_word_sequences,
    _pipe_key_changed_only_by_template_spacing,
    _recover_unique_template_field_words,
    _word_match_keys,
)
from WikiWho.utils import iter_rev_tokens, split_into_paragraphs, split_into_tokens

FIXTURES_DIR = os.path.join(os.path.dirname(__file__), "fixtures")

ARTICLES = [
    "Adam Himebauch",
    "Luis Donaldo Colosio Riojas",
    "One Thing Leads 2 Another",
    "Splatoon 3",
    "The Tell-Tale Brain",
]


def slug(title):
    return title.lower().replace(" ", "_")


def load_fixture(title):
    rev_path = os.path.join(FIXTURES_DIR, f"{slug(title)}_revisions.json")
    golden_path = os.path.join(FIXTURES_DIR, f"{slug(title)}_golden.json")
    if not os.path.exists(rev_path) or not os.path.exists(golden_path):
        pytest.skip(
            f"Fixtures missing for '{title}' — run 'python -m tests.generate_fixtures'"
        )
    with open(rev_path, encoding="utf-8") as f:
        revisions = json.load(f)
    with open(golden_path, encoding="utf-8") as f:
        golden = json.load(f)
    return revisions, golden


@pytest.mark.parametrize("title", ARTICLES)
def test_token_authorship_matches_golden(title):
    revisions, golden = load_fixture(title)

    ww = Wikiwho(title)
    ww.analyse_article(revisions)

    tokens = [
        {
            "str": t.value,
            "token_id": t.token_id,
            "o_rev_id": t.origin_rev_id,
            "in": list(t.inbound),
            "out": list(t.outbound),
        }
        for t in ww.tokens
    ]

    assert len(tokens) == len(golden), (
        f"Token count mismatch: got {len(tokens)}, expected {len(golden)}"
    )
    for i, (got, want) in enumerate(zip(tokens, golden)):
        assert got == want, (
            f"token[{i}] ({got['str']!r}) mismatch:\n  got:  {got}\n  want: {want}"
        )


def test_three_word_link_run_preserves_adam_authorship():
    revisions, _ = load_fixture("Adam Himebauch")
    previous_revision_id = 748557058
    current_revision_id = 754869525
    current_index = next(
        index for index, revision in enumerate(revisions)
        if revision["revid"] == current_revision_id
    )
    wikiwho = Wikiwho("Adam Himebauch")
    wikiwho.analyse_article(revisions[:current_index + 1])
    phrase = (
        "a", "building", "in", "[[", "little", "italy", ",",
        "manhattan", "|", "little", "italy", "]]",
    )

    def phrase_words(revision_id):
        words = list(iter_rev_tokens(wikiwho.revisions[revision_id]))
        values = tuple(word.value for word in words)
        starts = [
            start for start in range(len(values) - len(phrase) + 1)
            if values[start:start + len(phrase)] == phrase
        ]
        assert len(starts) == 1
        return words[starts[0]:starts[0] + len(phrase)]

    previous_words = phrase_words(previous_revision_id)
    current_words = phrase_words(current_revision_id)

    assert [word.token_id for word in current_words] == [
        word.token_id for word in previous_words
    ]
    assert {word.origin_rev_id for word in current_words} == {678462685}


def test_moved_words_before_link_are_recovered():
    previous = (
        [f"p{index}" for index in range(12)]
        + ["lived", "in", "[[", "victoria", ",", "british", "columbia", "]]", "from", "1900"]
        + [f"q{index}" for index in range(12)]
    )
    current = (
        [f"q{index}" for index in range(12)]
        + ["lived", "in", "[[", "victoria", ",", "british", "columbia", "]]", "from", "1900"]
        + [f"p{index}" for index in range(12)]
    )

    mapping, _ = _match_word_sequences(
        previous,
        current,
        full_text_prev=previous,
        full_text_curr=current,
    )

    assert mapping[current.index("lived")] == previous.index("lived")


def test_validated_move_recovers_tokens_across_punctuation():
    moved = [
        "|", "title", "=", "colosio", ",", "vizcaíno", ",", "marina", "y",
        "cerqueda", ",", "las", "nuevas", "caras", "de", "la", "política", "en", "méxico",
    ]
    left = [f"left{index}" for index in range(16)]
    right = [f"right{index}" for index in range(24)]
    previous = left + moved + right
    current = right + moved + left

    mapping, _ = _match_word_sequences(
        previous,
        current,
        full_text_prev=previous,
        full_text_curr=current,
    )

    assert mapping[current.index("marina")] == previous.index("marina")


def test_reference_markup_does_not_strengthen_a_weak_move_anchor():
    moved = ["across", "the", "country", ".", "<", "ref", "name", "=", ":", "0", ">"]
    left = [f"left{index}" for index in range(16)]
    right = [f"right{index}" for index in range(24)]
    previous = left + moved + right
    current = right + moved + left

    mapping, _ = _match_word_sequences(
        previous,
        current,
        full_text_prev=previous,
        full_text_curr=current,
    )

    assert mapping[current.index("across")] is None


def test_link_pipe_keys_do_not_trigger_template_spacing_recovery():
    previous_link = [
        token.lower()
        for token in split_into_tokens("[[File:Foo bar.jpg|thumb|caption]]")
    ]
    current_link = [
        token.lower()
        for token in split_into_tokens("[[File:Foobar.jpg|thumb|caption]]")
    ]
    previous_keys = _word_match_keys(previous_link)
    current_keys = _word_match_keys(current_link)
    previous_pipes = [
        key for token, key in zip(previous_link, previous_keys) if token == "|"
    ]
    current_pipes = [
        key for token, key in zip(current_link, current_keys) if token == "|"
    ]

    assert len(previous_pipes) == len(current_pipes) == 2
    assert not _has_template_name_spacing_change(previous_keys, current_keys)
    assert all(
        not _pipe_key_changed_only_by_template_spacing(previous_key, current_key)
        for previous_key, current_key in zip(previous_pipes, current_pipes)
    )

    previous = (
        [f"previous{index}" for index in range(8)]
        + previous_link
        + [f"suffix{index}" for index in range(8)]
    )
    current = (
        [f"current{index}" for index in range(8)]
        + current_link
        + [f"ending{index}" for index in range(8)]
    )
    mapping = [None] * len(current)
    match_confidence = [0] * len(current)
    _recover_unique_template_field_words(
        previous,
        current,
        _word_match_keys(previous),
        _word_match_keys(current),
        mapping,
        match_confidence,
        {},
        full_text_prev=previous,
        full_text_curr=current,
        count_state={"counts": {}},
    )

    assert mapping[current.index("thumb")] is None


def test_template_pipe_keys_still_detect_template_name_spacing_change():
    previous = [
        token.lower()
        for token in split_into_tokens("{{single chart|country=switzerland|62}}")
    ]
    current = [
        token.lower()
        for token in split_into_tokens("{{singlechart|country=switzerland|62}}")
    ]
    previous_keys = _word_match_keys(previous)
    current_keys = _word_match_keys(current)
    previous_pipes = [
        key for token, key in zip(previous, previous_keys) if token == "|"
    ]
    current_pipes = [
        key for token, key in zip(current, current_keys) if token == "|"
    ]

    assert len(previous_pipes) == len(current_pipes) == 2
    assert {key[2] for key in previous_pipes} == {
        "template-field",
        "template-arg",
    }
    assert _has_template_name_spacing_change(previous_keys, current_keys)
    assert all(
        _pipe_key_changed_only_by_template_spacing(previous_key, current_key)
        for previous_key, current_key in zip(previous_pipes, current_pipes)
    )


def test_delayed_sentence_reinsertion_can_restore_available_token_identities():
    available = []
    for index in range(33):
        word = Word()
        word.value = f"available{index}"
        word.outbound.append(20)
        available.append(word)
    occupied = []
    for value in (",", "of"):
        word = Word()
        word.value = value
        word.matched = True
        occupied.append(word)

    words = available + occupied

    assert _can_partially_restore_historical_sentence(words, previous_revision_id=21)
    assert not _can_partially_restore_historical_sentence(words, previous_revision_id=20)


def test_partially_restored_sentence_is_registered_for_future_reuse():
    revisions, _ = load_fixture("The Tell-Tale Brain")
    target_revid = 466775504
    target_index = next(
        index for index, revision in enumerate(revisions)
        if revision["revid"] == target_revid
    )
    wikiwho = Wikiwho("The Tell-Tale Brain")
    wikiwho.analyse_article(revisions[:target_index + 1])

    prefix = ["\"", "when", "vs", "ramachandran", ",", "one", "of", "the"]
    matching_sentences = []
    revision = wikiwho.revisions[target_revid]
    for paragraphs in revision.paragraphs.values():
        for paragraph in paragraphs:
            for sentences in paragraph.sentences.values():
                for sentence in sentences:
                    if [word.value for word in sentence.words[:len(prefix)]] == prefix:
                        matching_sentences.append(sentence)

    assert len(matching_sentences) == 1
    restored = matching_sentences[0]
    assert any(
        candidate is restored
        for candidate in wikiwho.sentences_ht[restored.hash_value]
    )
    assert wikiwho.sentences_ht[restored.hash_value][0] is restored
    assert restored.value == ""
    assert restored.splitted is None


def test_inline_template_end_is_not_split_as_table_markup():
    text = "{{linktext|偉大|}}한 {{linktext|遺産|}}"
    multiline_template = "{{#if:1\n|yes\n|}}\nafter"
    parameter = "{{{name\n|default\n|}}}\nafter"
    template_in_table = "{|\n|-\n| {{#if:1\n|yes\n|}}\n|}\nafter"

    assert split_into_paragraphs(text) == [text]
    assert split_into_paragraphs(multiline_template) == [multiline_template]
    assert split_into_paragraphs(parameter) == [parameter]
    assert any(
        "{{#if:1\n|yes\n|}}" in paragraph
        for paragraph in split_into_paragraphs(template_in_table)
    )
    assert "{|\n| cell\n|}" in split_into_paragraphs("{|\n| cell\n|}")
