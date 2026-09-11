import json
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import WikiWho.wikiwho as wikiwho_module
from WikiWho.structures import Paragraph, Revision, Sentence, Word
from WikiWho.wikiwho import (
    _MatchCandidateLedger,
    Wikiwho,
    _StructuralDocument,
    _StructuralIndex,
    _TokenSlot,
    _can_partially_restore_historical_sentence,
    _compact_structural_anchor_chains,
    _compact_targeted_anchor_occurrences,
    _compact_targeted_anchor_occurrences_all,
    _current_revision_token_slots,
    _duplicated_candidate_windows,
    _duplicated_candidate_windows_in_document,
    _duplicated_structural_candidate_windows,
    _has_template_name_spacing_change,
    _matched_structural_anchor_occurrences,
    _match_word_sequences,
    _pipe_key_changed_only_by_template_spacing,
    _propose_structural_word_matches,
    _propose_structural_word_matches_document,
    _propose_structural_word_matches_slots,
    _recover_unique_template_field_words,
    _residual_structural_window_keys,
    _revision_structural_document,
    _structural_anchor_chains,
    _targeted_structural_anchor_occurrences,
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


def test_structural_context_is_not_built_without_an_available_duplicate():
    previous = ["stable", "content", "has", "no", "residual", "duplicate"]
    current = list(previous)
    context_calls = []

    def get_structural_context():
        context_calls.append(True)
        raise AssertionError("structural context should remain lazy")

    mapping, deleted = _match_word_sequences(
        previous,
        current,
        get_structural_context=get_structural_context,
    )

    assert mapping == list(range(len(previous)))
    assert deleted == []
    assert context_calls == []


def test_informative_token_fast_path_preserves_the_original_predicate():
    tokens = (
        None, "", ".", "{{", "alpha", "123", "—東京—", "__a__",
        "!!!a!!!", "…", "é", "\u0301", "Ⅻ",
    )

    for token in tokens:
        expected = (
            isinstance(token, str) and
            token not in wikiwho_module.WORD_MATCH_MOVE_STRUCTURAL_TOKENS and
            any(character.isalnum() for character in token)
        )
        assert wikiwho_module._is_informative_move_token(token) == expected


def test_residual_triplet_preflight_is_a_sound_negative_gate(monkeypatch):
    ledger = _MatchCandidateLedger(4, 4)

    def unexpected_full_window_scan(*args, **kwargs):
        raise AssertionError("negative triplet gate should avoid full windows")

    monkeypatch.setattr(
        wikiwho_module, "_available_residual_windows",
        unexpected_full_window_scan,
    )
    assert wikiwho_module._unresolved_residual_windows(
        ledger,
        ["alpha", "beta", "gamma", "delta"],
        ["alpha", "beta", "changed", "delta"],
    ) == set()


def test_residual_triplet_preflight_preserves_the_exact_positive_gate():
    previous = ["left", "alpha", "beta", "gamma", "right"]
    current = ["other", "alpha", "beta", "gamma", "tail"]
    ledger = _MatchCandidateLedger(len(previous), len(current))
    expected = wikiwho_module._available_residual_windows(
        previous, lambda index: index not in ledger.prev_used_by,
    ).intersection(wikiwho_module._available_residual_windows(
        current, lambda index: ledger.prev_for_curr[index] is None,
    ))

    assert wikiwho_module._unresolved_residual_windows(
        ledger, previous, current,
    ) == expected
    assert ("alpha", "beta", "gamma") in expected


def test_current_structural_slots_reuse_the_parsed_revision_hierarchy():
    revision = Revision()
    paragraph = Paragraph()
    paragraph.hash_value = "paragraph"

    reused = Sentence()
    reused.hash_value = "reused"
    for value in ("already", "parsed"):
        word = Word()
        word.value = value
        reused.words.append(word)

    unmatched = Sentence()
    unmatched.hash_value = "unmatched"
    unmatched.splitted.extend(("new", ",", "tokens"))

    paragraph.sentences = {
        reused.hash_value: [reused],
        unmatched.hash_value: [unmatched],
    }
    paragraph.ordered_sentences = [reused.hash_value, unmatched.hash_value]
    revision.paragraphs = {paragraph.hash_value: [paragraph]}
    revision.ordered_paragraphs = [paragraph.hash_value]

    slots = _current_revision_token_slots(revision)

    assert [
        (slot.value, slot.article_index, slot.path) for slot in slots
    ] == [
        ("already", 0, (0, 0, 0)),
        ("parsed", 1, (0, 0, 1)),
        ("new", 2, (0, 1, 0)),
        (",", 3, (0, 1, 1)),
        ("tokens", 4, (0, 1, 2)),
    ]
    assert slots[0].word is reused.words[0]
    assert slots[1].word is reused.words[1]
    assert all(slot.word is None for slot in slots[2:])

    document = _revision_structural_document(revision)
    assert document.values == [
        "already", "parsed", "new", ",", "tokens",
    ]
    assert document.paragraph_ranges == {0: (0, 5)}
    assert document.sentence_ranges[reused] == (0, 0, 0, 2)
    assert document.sentence_ranges[unmatched] == (0, 1, 2, 3)


def test_current_structural_slots_fail_closed_on_inconsistent_values():
    revision = Revision()
    paragraph = Paragraph()
    paragraph.hash_value = "paragraph"
    sentence = Sentence()
    sentence.hash_value = "sentence"
    word = Word()
    word.value = "persistent"
    sentence.words.append(word)
    sentence.splitted.append("different")
    paragraph.sentences = {sentence.hash_value: [sentence]}
    paragraph.ordered_sentences = [sentence.hash_value]
    revision.paragraphs = {paragraph.hash_value: [paragraph]}
    revision.ordered_paragraphs = [paragraph.hash_value]

    assert _current_revision_token_slots(revision) is None
    assert _revision_structural_document(revision) is None


def test_structural_document_fails_closed_on_aliased_occurrence_objects():
    sentence = Sentence()
    sentence.hash_value = "sentence"
    for value in ("same", "sentence", "object"):
        word = Word()
        word.value = value
        sentence.words.append(word)

    paragraph = Paragraph()
    paragraph.hash_value = "paragraph"
    paragraph.sentences = {sentence.hash_value: [sentence, sentence]}
    paragraph.ordered_sentences = [sentence.hash_value, sentence.hash_value]

    revision = Revision()
    revision.paragraphs = {paragraph.hash_value: [paragraph]}
    revision.ordered_paragraphs = [paragraph.hash_value]
    assert _revision_structural_document(revision) is None

    first = Sentence()
    first.hash_value = "first"
    shared_word = Word()
    shared_word.value = "shared"
    first.words.append(shared_word)
    second = Sentence()
    second.hash_value = "second"
    second.words.append(shared_word)
    paragraph.sentences = {first.hash_value: [first], second.hash_value: [second]}
    paragraph.ordered_sentences = [first.hash_value, second.hash_value]
    assert _revision_structural_document(revision) is None


def test_compact_anchor_automaton_preserves_exact_occurrence_states():
    def make_document(paragraphs):
        values = []
        ranges = {}
        for paragraph_index, paragraph in enumerate(paragraphs):
            start = len(values)
            values.extend(paragraph)
            ranges[paragraph_index] = (start, len(values))
        document = _StructuralDocument(values, ranges, {})
        document.ensure_index()
        return document

    shared = ["shared{}".format(index) for index in range(20)]
    repeated = [
        "duplicate", "window", "has", "context", "separator",
        "duplicate", "window", "has", "context",
    ]
    previous = make_document((
        shared,
        repeated,
        ["previous{}".format(index) for index in range(12)],
    ))
    current = make_document((
        repeated + ["changed"],
        shared,
        ["current{}".format(index) for index in range(12)],
    ))

    automaton = _compact_targeted_anchor_occurrences_all(
        previous, current, {0, 1}, {0, 1},
    )

    def normalized(states):
        return {
            key: (value if isinstance(value, tuple) or value is None
                  else "unseen")
            for key, value in states.items()
        }

    for size in (10, 8, 6, 4):
        expected = _compact_targeted_anchor_occurrences(
            previous, current, size, {0, 1}, {0, 1},
        )
        actual = {
            key: value for key, value in automaton.items()
            if len(key) == size
        }
        assert normalized(actual) == normalized(expected)


def test_compact_duplicate_scan_branches_are_exact(monkeypatch):
    values = [
        "dup", "one", "two", "alpha",
        "beta", "gamma", "dup", "one", "two",
    ]
    document = _StructuralDocument(
        values, {0: (0, 4), 1: (4, len(values))}, {},
    )
    candidates = {
        ("dup", "one", "two"),
        ("alpha", "beta", "gamma"),
        ("missing", "candidate", "window"),
    }

    monkeypatch.setattr(
        wikiwho_module,
        "WORD_MATCH_STRUCTURAL_DUPLICATE_AUTOMATON_MIN_TOKENS",
        0,
    )
    automaton = _duplicated_candidate_windows_in_document(
        document, candidates,
    )
    monkeypatch.setattr(
        wikiwho_module,
        "WORD_MATCH_STRUCTURAL_DUPLICATE_AUTOMATON_MAX_PATTERN_SYMBOLS",
        0,
    )
    budget_fallback = _duplicated_candidate_windows_in_document(
        document, candidates,
    )
    monkeypatch.setattr(
        wikiwho_module,
        "WORD_MATCH_STRUCTURAL_DUPLICATE_AUTOMATON_MIN_TOKENS",
        10 ** 9,
    )
    tuple_scan = _duplicated_candidate_windows_in_document(
        document, candidates,
    )

    assert automaton == budget_fallback == tuple_scan == {
        ("dup", "one", "two"),
    }


def _make_structural_representations(paragraphs, residual_paragraphs=None):
    if residual_paragraphs is None:
        residual_paragraphs = set(range(len(paragraphs)))
    else:
        residual_paragraphs = set(residual_paragraphs)

    values = []
    paragraph_ranges = {}
    full_slots = []
    residual_values = []
    residual_slots = []
    residual_by_article = {}
    for paragraph_index, paragraph in enumerate(paragraphs):
        paragraph_start = len(values)
        for word_index, value in enumerate(paragraph):
            article_index = len(values)
            residual_index = None
            if paragraph_index in residual_paragraphs:
                residual_index = len(residual_values)
                residual_values.append(value)
            slot = _TokenSlot(
                value, article_index, paragraph_index, 0, word_index,
                residual_index=residual_index,
            )
            values.append(value)
            full_slots.append(slot)
            if residual_index is not None:
                residual_slots.append(slot)
                residual_by_article[article_index] = (
                    residual_index, (paragraph_index, 0, word_index),
                )
        paragraph_ranges[paragraph_index] = (
            paragraph_start, len(values),
        )
    return {
        "document": _StructuralDocument(values, paragraph_ranges, {}),
        "full_slots": full_slots,
        "residual_by_article": residual_by_article,
        "residual_slots": residual_slots,
        "residual_values": residual_values,
    }


def _candidate_evidence(ledger):
    return [
        (
            candidate.pairs, candidate.confidence, candidate.source,
            candidate.support, candidate.displacement, candidate.paths,
        )
        for candidate in ledger.candidates
    ]


def test_compact_anchor_chains_equal_slot_chains_for_every_routing_branch(
        monkeypatch):
    paragraphs = [
        ["paragraph{}-token{}".format(paragraph, token)
         for token in range(20)]
        for paragraph in range(3)
    ]
    previous = _make_structural_representations(paragraphs)
    current = _make_structural_representations((
        paragraphs[1], paragraphs[0], paragraphs[2],
    ))
    previous["document"].ensure_index()
    current["document"].ensure_index()

    # Below half selects targeted discovery; exactly half and above half use
    # the complete scan.  Both automaton policies must preserve the slot
    # implementation's globally unique anchor universe and chain scores.
    target_cases = (
        ({0}, {1}),
        ({0}, {0, 1}),
        ({0, 1, 2}, {0, 1, 2}),
    )
    for threshold, symbol_budget in (
            (0, 10 ** 9), (0, 0), (10 ** 9, 10 ** 9)):
        monkeypatch.setattr(
            wikiwho_module,
            "WORD_MATCH_STRUCTURAL_ANCHOR_AUTOMATON_MIN_TOKENS",
            threshold,
        )
        monkeypatch.setattr(
            wikiwho_module,
            "WORD_MATCH_STRUCTURAL_ANCHOR_AUTOMATON_MAX_PATTERN_SYMBOLS",
            symbol_budget,
        )
        for prev_targets, curr_targets in target_cases:
            expected = _structural_anchor_chains(
                _StructuralIndex(
                    previous["full_slots"], build_occurrences=False,
                ),
                _StructuralIndex(
                    current["full_slots"], build_occurrences=False,
                ),
                prev_targets, curr_targets,
            )[2]
            actual = _compact_structural_anchor_chains(
                previous["document"], current["document"],
                prev_targets, curr_targets,
            )
            assert actual == expected


def _duplicate_reorder_representations():
    duplicate = ["shared", "duplicate", "lineage", "phrase"]
    anchor_a = ["anchor-a-{}".format(index) for index in range(24)]
    anchor_b = ["anchor-b-{}".format(index) for index in range(24)]
    fillers = [
        ["filler-{}-{}".format(paragraph, index) for index in range(24)]
        for paragraph in range(4)
    ]
    previous_paragraphs = [
        ["old-a"] + duplicate + ["before-a"] + anchor_a,
        ["old-b"] + duplicate + ["before-b"] + anchor_b,
    ] + fillers
    current_paragraphs = [
        ["new-b"] + duplicate + ["after-b"] + anchor_b,
        ["new-a"] + duplicate + ["after-a"] + anchor_a,
    ] + fillers
    return (
        _make_structural_representations(previous_paragraphs, {0, 1}),
        _make_structural_representations(current_paragraphs, {0, 1}),
    )


def test_compact_proposer_equals_slot_proposer_across_scan_policies(
        monkeypatch):
    previous, current = _duplicate_reorder_representations()

    expected = _MatchCandidateLedger(
        len(previous["residual_values"]),
        len(current["residual_values"]),
    )
    _propose_structural_word_matches_slots(
        expected, previous["residual_values"], current["residual_values"],
        prev_slots=previous["residual_slots"],
        curr_slots=current["residual_slots"],
        full_prev_slots=previous["full_slots"],
        full_curr_slots=current["full_slots"],
    )
    assert expected.candidates

    def duplicated_candidates(candidates):
        duplicated_previous = _duplicated_candidate_windows_in_document(
            previous["document"], candidates,
        )
        return _duplicated_candidate_windows_in_document(
            current["document"], duplicated_previous,
        )

    context = (
        previous["document"], current["document"],
        previous["residual_by_article"],
        current["residual_by_article"],
    )
    for anchor_threshold, duplicate_threshold in (
            (0, 0), (0, 10 ** 9), (10 ** 9, 10 ** 9)):
        monkeypatch.setattr(
            wikiwho_module,
            "WORD_MATCH_STRUCTURAL_ANCHOR_AUTOMATON_MIN_TOKENS",
            anchor_threshold,
        )
        monkeypatch.setattr(
            wikiwho_module,
            "WORD_MATCH_STRUCTURAL_DUPLICATE_AUTOMATON_MIN_TOKENS",
            duplicate_threshold,
        )
        actual = _MatchCandidateLedger(
            len(previous["residual_values"]),
            len(current["residual_values"]),
        )
        assert _propose_structural_word_matches_document(
            actual,
            previous["residual_values"], current["residual_values"],
            lambda: context, duplicated_candidates,
        )
        assert _candidate_evidence(actual) == _candidate_evidence(expected)
        assert actual.resolve() == expected.resolve()


@pytest.mark.parametrize(
    "fallback_reason", ("duplicate-state-unavailable", "document-unavailable"),
)
def test_compact_proposer_routes_failures_to_the_slot_oracle(
        fallback_reason):
    previous, current = _duplicate_reorder_representations()
    expected = _MatchCandidateLedger(
        len(previous["residual_values"]),
        len(current["residual_values"]),
    )
    _propose_structural_word_matches_slots(
        expected, previous["residual_values"], current["residual_values"],
        prev_slots=previous["residual_slots"],
        curr_slots=current["residual_slots"],
        full_prev_slots=previous["full_slots"],
        full_curr_slots=current["full_slots"],
    )

    context_calls = []
    document_calls = []

    def get_structural_context():
        context_calls.append(True)
        return (
            previous["residual_slots"], current["residual_slots"],
            previous["full_slots"], current["full_slots"],
        )

    def get_structural_documents():
        document_calls.append(True)
        return None

    def get_structural_duplicates(candidates):
        if fallback_reason == "duplicate-state-unavailable":
            return None
        duplicated_previous = _duplicated_candidate_windows_in_document(
            previous["document"], candidates,
        )
        return _duplicated_candidate_windows_in_document(
            current["document"], duplicated_previous,
        )

    actual = _MatchCandidateLedger(
        len(previous["residual_values"]),
        len(current["residual_values"]),
    )
    _propose_structural_word_matches(
        actual, previous["residual_values"], current["residual_values"],
        get_structural_context=get_structural_context,
        get_structural_duplicates=get_structural_duplicates,
        get_structural_documents=get_structural_documents,
    )

    assert context_calls == [True]
    assert document_calls == (
        [] if fallback_reason == "duplicate-state-unavailable" else [True]
    )
    assert _candidate_evidence(actual) == _candidate_evidence(expected)
    assert actual.resolve() == expected.resolve()


def test_duplicate_window_counts_respect_paragraph_boundaries():
    values = (
        (("dup", "one", "two", "alpha"), 0),
        (("beta", "gamma", "dup", "one", "two"), 1),
    )
    slots = []
    article_index = 0
    for paragraph_values, paragraph_index in values:
        for word_index, value in enumerate(paragraph_values):
            slots.append(_TokenSlot(
                value, article_index, paragraph_index, 0, word_index,
            ))
            article_index += 1

    duplicated = _duplicated_candidate_windows(slots, {
        ("dup", "one", "two"),
        ("alpha", "beta", "gamma"),
        ("missing", "candidate", "window"),
    })

    assert duplicated == {("dup", "one", "two")}


def test_structural_index_caps_occurrences_and_merges_anchor_windows():
    duplicated_values = [
        "alpha", "beta", "gamma", "delta", "separator",
        "alpha", "beta", "gamma", "delta",
    ]
    duplicated_slots = [
        _TokenSlot(value, index, 0, 0, index)
        for index, value in enumerate(duplicated_values)
    ]
    duplicated_index = _StructuralIndex(duplicated_slots)

    states = duplicated_index.occurrence_states(4, 4)
    assert states[("alpha", "beta", "gamma", "delta")] is None
    assert states[("beta", "gamma", "delta", "separator")] == (0, 1)
    assert duplicated_index.informative_count(
        duplicated_slots, 0, 4,
    ) == 4

    unique_values = ["token{}".format(index) for index in range(20)]
    prev_slots = [
        _TokenSlot(value, index, 0, 0, index)
        for index, value in enumerate(unique_values)
    ]
    curr_slots = [
        _TokenSlot(value, index, 0, 0, index)
        for index, value in enumerate(unique_values)
    ]
    _, _, chains = _structural_anchor_chains(
        _StructuralIndex(prev_slots), _StructuralIndex(curr_slots),
    )

    assert chains[(0, 0)] == ([(0, 20, 0, 20, 20)], (20, 20))


def test_structural_lcs_preflight_requires_a_residual_only_ambiguous_window():
    values = ["alpha", "beta", "gamma", "delta", "epsilon"]
    prev_slots = [
        _TokenSlot(value, index, 0, 0, index)
        for index, value in enumerate(values)
    ]
    curr_slots = [
        _TokenSlot(value, index, 0, 0, index)
        for index, value in enumerate(values)
    ]
    prev_index = _StructuralIndex(prev_slots, build_occurrences=False)
    curr_index = _StructuralIndex(curr_slots, build_occurrences=False)
    ambiguous = {("beta", "gamma", "delta")}
    complete_residual = {
        (0, 0, index): index for index in range(len(values))
    }

    prev_windows = _residual_structural_window_keys(
        prev_slots, prev_index, complete_residual, ambiguous,
    )
    assert prev_windows == ambiguous
    assert _residual_structural_window_keys(
        curr_slots, curr_index, complete_residual, prev_windows,
    ) == ambiguous

    interrupted_current = dict(complete_residual)
    del interrupted_current[(0, 0, 2)]
    assert not _residual_structural_window_keys(
        curr_slots, curr_index, interrupted_current, prev_windows,
    )


def test_single_map_structural_evidence_matches_independent_global_states():
    shared_values = ["shared{}".format(index) for index in range(20)]
    repeated_values = [
        "duplicate", "window", "has", "context", "separator",
        "duplicate", "window", "has", "context",
    ]
    untouched_values = [
        "untouched{}".format(index) for index in range(12)
    ]

    def make_slots(paragraphs):
        slots = []
        article_index = 0
        for paragraph_index, values in enumerate(paragraphs):
            for word_index, value in enumerate(values):
                slots.append(_TokenSlot(
                    value, article_index, paragraph_index, 0, word_index,
                ))
                article_index += 1
        return slots

    prev_slots = make_slots((
        shared_values,
        repeated_values,
        untouched_values,
    ))
    curr_slots = make_slots((
        repeated_values + ["changed"],
        shared_values,
        untouched_values,
    ))

    independent_prev = _StructuralIndex(prev_slots)
    independent_curr = _StructuralIndex(curr_slots)
    streamed_prev = _StructuralIndex(prev_slots, build_occurrences=False)
    streamed_curr = _StructuralIndex(curr_slots, build_occurrences=False)

    for size in (10, 8, 6, 4):
        expected_prev = independent_prev.occurrence_states(size, 4)
        expected_curr = independent_curr.occurrence_states(size, 4)
        expected_matches = {
            key: prev_position + expected_curr[key]
            for key, prev_position in expected_prev.items()
            if (prev_position is not None and
                expected_curr.get(key) is not None)
        }
        actual = _matched_structural_anchor_occurrences(
            streamed_prev, streamed_curr, size,
        )
        assert {
            key: occurrence
            for key, occurrence in actual.items()
            if occurrence is not None and len(occurrence) == 4
        } == expected_matches

        targeted = _targeted_structural_anchor_occurrences(
            streamed_prev, streamed_curr, size, {0}, {0},
        )
        assert {
            key: occurrence
            for key, occurrence in targeted.items()
            if isinstance(occurrence, tuple) and len(occurrence) == 4
        } == {
            key: occurrence for key, occurrence in expected_matches.items()
            if occurrence[0] == 0 or occurrence[2] == 0
        }

        for key, prev_position in expected_prev.items():
            if prev_position is None:
                assert actual[key] is None
            elif key not in expected_curr:
                assert actual[key] == prev_position
            elif expected_curr[key] is None:
                assert actual[key] is None

    ambiguity_sizes = (6, 5, 4, 3)
    expected_duplicates = set()
    candidate_windows = set()
    for size in ambiguity_sizes:
        expected_prev = independent_prev.occurrence_states(size, 3)
        expected_curr = independent_curr.occurrence_states(size, 3)
        candidate_windows.update(
            set(expected_prev).intersection(expected_curr)
        )
        prev_duplicates = set(
            key for key, position in expected_prev.items()
            if position is None
        )
        curr_duplicates = set(
            key for key, position in expected_curr.items()
            if position is None
        )
        expected_duplicates.update(
            prev_duplicates.intersection(curr_duplicates)
        )
    duplicated_prev = _duplicated_structural_candidate_windows(
        streamed_prev, candidate_windows,
    )
    assert _duplicated_structural_candidate_windows(
        streamed_curr, duplicated_prev,
    ) == expected_duplicates


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
    ledger = _MatchCandidateLedger(len(previous), len(current))
    _recover_unique_template_field_words(
        previous,
        current,
        _word_match_keys(previous),
        _word_match_keys(current),
        ledger,
        full_text_prev=previous,
        full_text_curr=current,
        count_state={"counts": {}},
    )
    mapping, _ = ledger.resolve()

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
