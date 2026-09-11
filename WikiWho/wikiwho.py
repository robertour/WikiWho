# -*- coding: utf-8 -*-
"""

:Authors:
    Maribel Acosta,
    Fabian Floeck,
    Andriy Rodchenko,
    Kenan Erdogan
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from bisect import bisect_left
from collections import Counter, defaultdict
from difflib import SequenceMatcher
from heapq import merge

from .structures import Word, Sentence, Paragraph, Revision
from .utils import calculate_hash, split_into_paragraphs, split_into_sentences, split_into_tokens, \
    compute_avg_word_freq, iter_rev_tokens, TOKEN_SYMBOLS

try:
    from . import _structural_native
except ImportError:
    _structural_native = None

if _structural_native is not None:
    _structural_native.configure_token_symbols(TOKEN_SYMBOLS)
    split_into_tokens = _structural_native.split_into_tokens
    split_into_paragraphs = _structural_native.split_into_paragraphs


# Spam detection variables.
CHANGE_PERCENTAGE = -0.40
PREVIOUS_LENGTH = 1000
CURR_LENGTH = 1000
FLAG = "move"
UNMATCHED_PARAGRAPH = 0.0
TOKEN_DENSITY_LIMIT = 20
TOKEN_LEN = 100

# Caps estimated identical-token prev/current pairs for the SequenceMatcher opcode pass. Above this, the whole unmatched middle span falls back to bounded nearest-neighbor matching.
WORD_MATCH_MAX_SEQUENCE_PAIRS = 200000

# Caps nearest-neighbor recovery inside one SequenceMatcher replace opcode region. Higher values preserve more matches in broad edits but can reintroduce expensive local scans.
WORD_MATCH_MAX_LOCAL_PAIRS = 10000

# Minimum positional drift allowed for nearest-neighbor reuse of a previous Word object.
WORD_MATCH_MAX_DRIFT_MIN = 50

# Ratio-based nearest-neighbor drift allowed, computed against the larger unmatched side. Higher drift preserves more heuristic matches; lower drift bounds cost and cross-section matches.
WORD_MATCH_MAX_DRIFT_RATIO = 0.10

# Match confidence is a precedence ladder for competing claims to a current token. Cheap local reuse is weakest, exact edge matches are strongest, and structural fixes sit below moved-run recovery so a verified moved copy can still win.
WORD_MATCH_CONF_LOCAL = 20
WORD_MATCH_CONF_SEQUENCE_EQUAL = 90
WORD_MATCH_CONF_STRUCTURAL_BOUNDARY = 92
# Structurally anchored local matches must outrank the generic article-wide
# SequenceMatcher alignment, while remaining below full-revision-unique moved
# runs.  This is deliberately a separate tier rather than another local score.
WORD_MATCH_CONF_STRUCTURAL_GAP = 94
WORD_MATCH_CONF_MOVED_RUN = 95
WORD_MATCH_CONF_EDGE = 100

# Structural correspondence is established by globally unique informative
# windows and alignment is restricted to bounded gaps between those anchors.
WORD_MATCH_STRUCTURAL_ANCHOR_SIZES = (10, 8, 6, 4)
WORD_MATCH_STRUCTURAL_MIN_ANCHOR_INFO = 4
WORD_MATCH_STRUCTURAL_AMBIGUITY_SIZES = (6, 5, 4, 3)
WORD_MATCH_STRUCTURAL_INDEX_SIZES = (10, 8, 6, 5, 4, 3)
# A paragraph pair must carry more than one short coincidental anchor's worth
# of lexical evidence before its gaps can confer structural lineage.  One long
# merged unchanged block also satisfies this requirement.
WORD_MATCH_STRUCTURAL_MIN_PAIR_INFO = 2 * max(WORD_MATCH_STRUCTURAL_ANCHOR_SIZES)
WORD_MATCH_STRUCTURAL_MIN_RUN_INFO = 3
WORD_MATCH_STRUCTURAL_MAX_GAP_CELLS = 50000
# Below these measured crossover points, short tuple windows are cheaper than
# constructing a failure automaton.  Both paths perform exact tuple equality
# and saturated occurrence counting; these constants affect cost, not matching
# semantics.
WORD_MATCH_STRUCTURAL_ANCHOR_AUTOMATON_MIN_TOKENS = 50000
WORD_MATCH_STRUCTURAL_DUPLICATE_AUTOMATON_MIN_TOKENS = 25000
# The automata allocate Python dictionaries and lists per trie state.  Bound
# the total input symbols so an unusually broad residual edit cannot exchange
# several linear tuple scans for an unbounded transient object graph.
WORD_MATCH_STRUCTURAL_ANCHOR_AUTOMATON_MAX_PATTERN_SYMBOLS = 250000
WORD_MATCH_STRUCTURAL_DUPLICATE_AUTOMATON_MAX_PATTERN_SYMBOLS = 250000
# Structural evidence is deliberately local.  If structural and generic runs
# form a conflict component larger than this many token endpoints, the
# occurrence assignment is too broad to resolve safely and retains the legacy
# mapping.  This also bounds resolver work on mass template/reference edits.
WORD_MATCH_STRUCTURAL_MAX_CONFLICT_TOKENS = 512

# Moved-run recovery looks for unique informative n-grams in unmatched diff regions. The sizes/caps below bound how much extra indexing we do per word diff while still finding copied or moved runs that SequenceMatcher misses.
WORD_MATCH_MOVE_NGRAM_SIZES = (10, 8, 6, 4, 3)
# Run admission and per-token coverage use different evidence floors. An
# unstructured content core needs four informative tokens to admit a moved run
# by itself. After separate structural evidence admits the run, unique windows
# with three informative tokens can provide per-token coverage.
WORD_MATCH_MOVE_MIN_INFO_TOKENS = 3
WORD_MATCH_MOVE_MIN_ANCHOR_INFO_TOKENS = 4
WORD_MATCH_MOVE_MIN_RECOVERABLE_TOKENS = 24
WORD_MATCH_MOVE_MIN_LINK_WINDOW = 6
WORD_MATCH_MOVE_MAX_WINDOWS = 300000

# Partial historical-sentence restoration needs enough unchanged context to make the old sentence identity stronger than the few token identities already occupied elsewhere.
WORD_MATCH_HISTORICAL_MIN_AVAILABLE_TOKENS = 24
WORD_MATCH_HISTORICAL_MAX_OCCUPIED_TOKENS = 2
WORD_MATCH_HISTORICAL_MIN_EVIDENCE_RATIO = 4

# A large pure deletion immediately before an unchanged suffix can otherwise keep very old glue words alive by edge matching. Limit that correction to the first suffix window so normal suffix preservation remains cheap.
WORD_MATCH_EDGE_STALE_REWRITE_MIN_TOKENS = 24
WORD_MATCH_EDGE_STALE_WINDOW = 64

# Structural punctuation is allowed to ride along with a verified moved run, but it cannot seed one by itself.
WORD_MATCH_MOVE_STRUCTURAL_TOKENS = frozenset((
    '.', ',', ';', ':', '?', '!', '-', '_', '/', '\\', '(', ')', '[', ']', '{', '}', '*', '#', '@',
    '&', '=', '+', '%', '~', '$', '^', '<', '>', '"', "'", '|', '{{', '}}', '[[', ']]',
))

# Wikitext constructs whose boundary tokens are too generic to claim through a cheap common-prefix/common-suffix match. If an edge match stops inside one of these constructs, the edge is rolled back so contextual matching sees the whole template/link/comment.
WIKITEXT_CONSTRUCT_PAIRS = (
    ('{{', '}}'),
    ('[[', ']]'),
    ('<!--', '-->'),
)
WIKITEXT_OPEN_TO_CLOSE = dict(WIKITEXT_CONSTRUCT_PAIRS)
WIKITEXT_CLOSE_TO_OPEN = dict((close, open_) for open_, close in WIKITEXT_CONSTRUCT_PAIRS)

class _TokenSlot(object):
    """One token occurrence with its revision-local structural path."""

    __slots__ = (
        'value', 'word', 'article_index', 'paragraph_index',
        'sentence_index', 'word_index', 'residual_index',
    )

    def __init__(self, value, article_index, paragraph_index, sentence_index,
                 word_index, word=None, residual_index=None):
        self.value = value
        self.word = word
        self.article_index = article_index
        self.paragraph_index = paragraph_index
        self.sentence_index = sentence_index
        self.word_index = word_index
        self.residual_index = residual_index

    @property
    def path(self):
        return (self.paragraph_index, self.sentence_index, self.word_index)


class _MatchCandidate(object):
    """A run-level proposal; Word identities are assigned only after resolve."""

    __slots__ = (
        'pairs', 'confidence', 'source', 'support', 'displacement', 'order',
        'paths',
    )

    def __init__(self, pairs, confidence, source, support, displacement, order,
                 paths=None):
        self.pairs = tuple(pairs)
        self.confidence = confidence
        self.source = source
        self.support = support
        self.displacement = displacement
        self.order = order
        self.paths = tuple(paths) if paths is not None else None


def _common_prefix_len(left, right):
    limit = min(len(left), len(right))
    index = 0
    while index < limit and left[index] == right[index]:
        index += 1
    return index


def _common_suffix_len(left, right, prefix_len):
    limit = min(len(left), len(right)) - prefix_len
    suffix_len = 0
    while suffix_len < limit and left[len(left) - suffix_len - 1] == right[len(right) - suffix_len - 1]:
        suffix_len += 1
    return suffix_len


def _construct_stack_at(tokens, end):
    stack = []
    for index in range(end):
        token = tokens[index]
        if token in WIKITEXT_OPEN_TO_CLOSE:
            stack.append((token, index))
        elif token in WIKITEXT_CLOSE_TO_OPEN:
            open_token = WIKITEXT_CLOSE_TO_OPEN[token]
            for stack_index in range(len(stack) - 1, -1, -1):
                if stack[stack_index][0] == open_token:
                    del stack[stack_index:]
                    break
    return stack


def _construct_end_after_boundary(tokens, start, open_token):
    close_token = WIKITEXT_OPEN_TO_CLOSE[open_token]
    depth = 1
    for index in range(start, len(tokens)):
        token = tokens[index]
        if token == open_token:
            depth += 1
        elif token == close_token:
            depth -= 1
            if depth == 0:
                return index + 1
    return None


def _rollback_prefix_construct_boundary(tokens, prefix_len):
    rollback = prefix_len
    while rollback:
        stack = _construct_stack_at(tokens, rollback)
        if not stack:
            return rollback
        open_token, open_index = stack[-1]
        if _construct_end_after_boundary(tokens, rollback, open_token) is None:
            return rollback
        rollback = open_index
    return rollback


def _suffix_construct_boundary_drop(tokens, suffix_start):
    drop = 0
    while suffix_start + drop < len(tokens):
        boundary = suffix_start + drop
        stack = _construct_stack_at(tokens, boundary)
        if not stack:
            return drop
        open_token, _ = stack[-1]
        construct_end = _construct_end_after_boundary(tokens, boundary, open_token)
        if construct_end is None:
            return drop
        drop = construct_end - suffix_start
    return drop


def _rollback_common_construct_edges(left, right, left_keys, right_keys, prefix_len):
    # Do not let cheap edge matches claim generic tokens from inside templates, links, or comments before the contextual matcher sees the whole construct.
    prefix_len = min(_rollback_prefix_construct_boundary(left, prefix_len),
                     _rollback_prefix_construct_boundary(right, prefix_len))
    suffix_len = _common_suffix_len(left_keys, right_keys, prefix_len)
    if suffix_len:
        left_drop = _suffix_construct_boundary_drop(left, len(left) - suffix_len)
        right_drop = _suffix_construct_boundary_drop(right, len(right) - suffix_len)
        suffix_len -= min(suffix_len, max(left_drop, right_drop))
    return prefix_len, suffix_len


def _tokens_until(tokens, start, stops):
    collected = []
    index = start
    while index < len(tokens) and tokens[index] not in stops:
        collected.append(tokens[index])
        index += 1
    return tuple(collected), index


def _template_name_after(tokens, start):
    name, _ = _tokens_until(tokens, start, ('|', '}}'))
    return name


def _link_target_after(tokens, start):
    target, _ = _tokens_until(tokens, start, ('|', ']]'))
    return target


def _template_field_after(tokens, start):
    field, index = _tokens_until(tokens, start, ('=', '|', '}}'))
    if index < len(tokens) and tokens[index] == '=' and field:
        return field
    return None


def _template_field_before(tokens, equals_index):
    field = []
    index = equals_index - 1
    while index >= 0 and tokens[index] not in ('{{', '|', '}}'):
        field.append(tokens[index])
        index -= 1
    if index >= 0 and tokens[index] == '|' and field:
        field.reverse()
        return tuple(field)
    return None


def _link_option_after(tokens, start):
    option, _ = _tokens_until(tokens, start, ('|', ']]'))
    return option


def _pop_construct(stack, construct_type):
    for stack_index in range(len(stack) - 1, -1, -1):
        if stack[stack_index]['type'] == construct_type:
            frame = stack[stack_index]
            del stack[stack_index:]
            return frame
    return None


# Normal tokens still match by value. Low-information wikitext tokens match by local syntax context so, for example, a link option "|" does not match an infobox field "|" and a "{{cite web}}" opener does not match "{{for-multi}}".
def _word_match_keys(tokens):
    keys = list(tokens)
    stack = []
    for index, token in enumerate(tokens):
        if token == '{{':
            name = _template_name_after(tokens, index + 1)
            keys[index] = ('wikitext', '{{', 'template', name) if name else token
            stack.append({'type': 'template', 'name': name, 'arg_index': 0})
        elif token == '}}':
            frame = _pop_construct(stack, 'template')
            name = frame['name'] if frame else None
            keys[index] = ('wikitext', '}}', 'template', name) if name else token
        elif token == '[[':
            target = _link_target_after(tokens, index + 1)
            keys[index] = ('wikitext', '[[', 'link', target) if target else token
            stack.append({'type': 'link', 'target': target, 'option_index': 0})
        elif token == ']]':
            frame = _pop_construct(stack, 'link')
            target = frame['target'] if frame else None
            keys[index] = ('wikitext', ']]', 'link', target) if target else token
        elif token == '<!--':
            keys[index] = ('wikitext', '<!--', 'comment')
            stack.append({'type': 'comment'})
        elif token == '-->':
            _pop_construct(stack, 'comment')
            keys[index] = ('wikitext', '-->', 'comment')
        elif token == '|' and stack:
            frame = stack[-1]
            if frame['type'] == 'link':
                option = _link_option_after(tokens, index + 1)
                keys[index] = ('wikitext', '|', 'link', frame['target'],
                               frame['option_index'], option)
                frame['option_index'] += 1
            elif frame['type'] == 'template':
                field = _template_field_after(tokens, index + 1)
                if field:
                    keys[index] = ('wikitext', '|', 'template-field',
                                   frame['name'], field)
                else:
                    keys[index] = ('wikitext', '|', 'template-arg',
                                   frame['name'], frame['arg_index'])
                frame['arg_index'] += 1
        elif token == '=' and stack and stack[-1]['type'] == 'template':
            field = _template_field_before(tokens, index)
            if field:
                keys[index] = ('wikitext', '=', 'template-field',
                               stack[-1]['name'], field)
    return keys


def _ordered_paragraph_occurrences(revision):
    counts = defaultdict(int)
    for paragraph_index, paragraph_hash in enumerate(revision.ordered_paragraphs):
        occurrence = counts[paragraph_hash]
        counts[paragraph_hash] += 1
        yield paragraph_index, revision.paragraphs[paragraph_hash][occurrence]


def _ordered_sentence_occurrences(paragraph):
    counts = defaultdict(int)
    for sentence_index, sentence_hash in enumerate(paragraph.ordered_sentences):
        occurrence = counts[sentence_hash]
        counts[sentence_hash] += 1
        yield sentence_index, paragraph.sentences[sentence_hash][occurrence]


def _revision_token_slots(revision):
    """Return persistent revision words with explicit occurrence paths."""
    slots = []
    article_index = 0
    for paragraph_index, paragraph in _ordered_paragraph_occurrences(revision):
        for sentence_index, sentence in _ordered_sentence_occurrences(paragraph):
            for word_index, word in enumerate(sentence.words):
                slots.append(_TokenSlot(
                    word.value, article_index, paragraph_index, sentence_index,
                    word_index, word=word,
                ))
                article_index += 1
    return slots


def _text_token_slots(text):
    """Tokenize current wikitext with the hierarchy used to build Revision."""
    slots = []
    article_index = 0
    paragraph_index = 0
    for raw_paragraph in split_into_paragraphs(text):
        paragraph = raw_paragraph.strip()
        if not paragraph:
            continue
        sentence_index = 0
        for raw_sentence in split_into_sentences(paragraph):
            sentence = raw_sentence.strip()
            if not sentence:
                continue
            for word_index, value in enumerate(split_into_tokens(sentence)):
                slots.append(_TokenSlot(
                    value, article_index, paragraph_index, sentence_index,
                    word_index,
                ))
                article_index += 1
            sentence_index += 1
        paragraph_index += 1
    return slots


def _current_revision_token_slots(revision):
    """Build current-side slots from the hierarchy already parsed this edit.

    Exact or historically reused sentences already contain ``Word`` objects.
    Newly unmatched sentences have their normalized values in ``splitted`` by
    the time word matching can request structural context.  ``None`` signals
    an incomplete or inconsistent hierarchy so the caller can use the original
    tokenizer as a correctness fallback.
    """
    slots = []
    article_index = 0
    for paragraph_index, paragraph in _ordered_paragraph_occurrences(revision):
        for sentence_index, sentence in _ordered_sentence_occurrences(paragraph):
            if sentence.words:
                persistent_words = sentence.words
                values = [word.value for word in persistent_words]
                if sentence.splitted and list(sentence.splitted) != values:
                    return None
            elif sentence.splitted:
                persistent_words = None
                values = sentence.splitted
            else:
                return None
            for word_index, value in enumerate(values):
                slots.append(_TokenSlot(
                    value, article_index, paragraph_index, sentence_index,
                    word_index,
                    word=(persistent_words[word_index]
                          if persistent_words is not None else None),
                ))
                article_index += 1
    return slots


class _StructuralDocument(object):
    """A compact hierarchy view used only by structural disambiguation.

    Full token values and paragraph/sentence offsets are retained from the
    hierarchy scan already needed by the duplicate gate.  Context-sensitive
    matching keys and informative prefix sums remain lazy until that gate has
    proved that structural matching may contribute.
    """

    __slots__ = (
        'values', 'paragraph_ranges', 'sentence_ranges', 'keys',
        'informative_prefix',
    )

    def __init__(self, values, paragraph_ranges, sentence_ranges):
        self.values = values
        self.paragraph_ranges = paragraph_ranges
        self.sentence_ranges = sentence_ranges
        self.keys = None
        self.informative_prefix = None

    def ensure_index(self):
        if self.keys is not None:
            return
        if _structural_native is not None:
            self.keys, native_prefix = (
                _structural_native.document_index(
                    self.values, WORD_MATCH_MOVE_STRUCTURAL_TOKENS,
                )
            )
            self.informative_prefix = memoryview(native_prefix).cast('Q')
        else:
            self.keys = _word_match_keys(self.values)
            self.informative_prefix = _informative_move_token_prefix(
                self.values,
            )

    def paragraph_range(self, paragraph_index):
        return self.paragraph_ranges[paragraph_index]

    def paragraph_length(self, paragraph_index):
        start, end = self.paragraph_ranges[paragraph_index]
        return end - start

    def informative_count(self, paragraph_index, start, end):
        paragraph_start, _ = self.paragraph_ranges[paragraph_index]
        article_start = paragraph_start + start
        return (
            self.informative_prefix[paragraph_start + end] -
            self.informative_prefix[article_start]
        )

    def window_key(self, paragraph_index, start, size):
        paragraph_start, _ = self.paragraph_ranges[paragraph_index]
        article_start = paragraph_start + start
        return tuple(self.keys[article_start:article_start + size])


def _revision_structural_document(revision):
    """Build a compact view from an already parsed revision hierarchy.

    ``None`` retains the existing tokenizer/slot fallback for an incomplete or
    inconsistent hierarchy.
    """
    values = []
    paragraph_ranges = {}
    sentence_ranges = {}
    seen_paragraphs = set()
    seen_sentences = set()
    seen_words = set()
    for paragraph_index, paragraph in _ordered_paragraph_occurrences(revision):
        if paragraph in seen_paragraphs:
            return None
        seen_paragraphs.add(paragraph)
        paragraph_start = len(values)
        for sentence_index, sentence in _ordered_sentence_occurrences(paragraph):
            if sentence in seen_sentences:
                return None
            seen_sentences.add(sentence)
            if sentence.words:
                word_identities = set(sentence.words)
                if (len(word_identities) != len(sentence.words) or
                        not seen_words.isdisjoint(word_identities)):
                    return None
                seen_words.update(word_identities)
                sentence_values = [word.value for word in sentence.words]
                if (sentence.splitted and
                        list(sentence.splitted) != sentence_values):
                    return None
            elif sentence.splitted:
                sentence_values = sentence.splitted
            else:
                return None
            sentence_start = len(values)
            values.extend(sentence_values)
            sentence_ranges[sentence] = (
                paragraph_index, sentence_index, sentence_start,
                len(sentence_values),
            )
        paragraph_ranges[paragraph_index] = (paragraph_start, len(values))
    return _StructuralDocument(values, paragraph_ranges, sentence_ranges)


def _revision_structural_document_pair(
        previous, current, previous_target_sentences=None,
        current_target_sentences=None):
    """Build adjacent compact documents while scanning shared paragraphs once.

    Sentence offsets are needed only to align residual sentences back to the
    complete document.  When both target collections are supplied, retain
    offsets only for those sentences while continuing to validate every
    sentence and word in both revisions.
    """
    paragraph_cache = {}
    invalid = object()
    retain_all_sentences = (
        previous_target_sentences is None or
        current_target_sentences is None
    )
    if retain_all_sentences:
        previous_targets = None
        current_targets = None
        all_targets = None
    else:
        previous_targets = set(previous_target_sentences)
        current_targets = set(current_target_sentences)
        all_targets = previous_targets.union(current_targets)

    if _structural_native is not None:
        native_documents = _structural_native.document_pair(
            previous, current, previous_targets, current_targets,
        )
        if native_documents is None:
            return None, None
        (prev_values, prev_paragraph_ranges, prev_sentence_ranges,
         curr_values, curr_paragraph_ranges,
         curr_sentence_ranges) = native_documents
        return (
            _StructuralDocument(
                prev_values, prev_paragraph_ranges, prev_sentence_ranges,
            ),
            _StructuralDocument(
                curr_values, curr_paragraph_ranges, curr_sentence_ranges,
            ),
        )

    def paragraph_snapshot(paragraph):
        cached = paragraph_cache.get(paragraph)
        if cached is invalid:
            return None
        if cached is not None:
            return cached

        values = []
        sentence_entries = []
        sentence_identities = set()
        word_identities = set()
        for sentence_index, sentence in _ordered_sentence_occurrences(
                paragraph):
            if sentence in sentence_identities:
                paragraph_cache[paragraph] = invalid
                return None
            sentence_identities.add(sentence)
            if sentence.words:
                current_word_identities = set(sentence.words)
                if (len(current_word_identities) != len(sentence.words) or
                        not word_identities.isdisjoint(
                            current_word_identities)):
                    paragraph_cache[paragraph] = invalid
                    return None
                word_identities.update(current_word_identities)
                sentence_values = [word.value for word in sentence.words]
                if (sentence.splitted and
                        list(sentence.splitted) != sentence_values):
                    paragraph_cache[paragraph] = invalid
                    return None
            elif sentence.splitted:
                sentence_values = list(sentence.splitted)
            else:
                paragraph_cache[paragraph] = invalid
                return None
            if all_targets is None or sentence in all_targets:
                sentence_entries.append((
                    sentence, sentence_index, len(values),
                    len(sentence_values),
                ))
            values.extend(sentence_values)
        snapshot = (
            values, sentence_entries, sentence_identities, word_identities,
        )
        paragraph_cache[paragraph] = snapshot
        return snapshot

    def build(revision, targets):
        values = []
        paragraph_ranges = {}
        sentence_ranges = {}
        seen_paragraphs = set()
        seen_sentences = set()
        seen_words = set()
        for paragraph_index, paragraph in _ordered_paragraph_occurrences(
                revision):
            if paragraph in seen_paragraphs:
                return None
            seen_paragraphs.add(paragraph)
            snapshot = paragraph_snapshot(paragraph)
            if snapshot is None:
                return None
            (paragraph_values, sentence_entries,
             sentence_identities, word_identities) = snapshot
            if (not seen_sentences.isdisjoint(sentence_identities) or
                    not seen_words.isdisjoint(word_identities)):
                return None
            seen_sentences.update(sentence_identities)
            seen_words.update(word_identities)
            paragraph_start = len(values)
            values.extend(paragraph_values)
            for (sentence, sentence_index, sentence_start,
                 sentence_length) in sentence_entries:
                if targets is not None and sentence not in targets:
                    continue
                sentence_ranges[sentence] = (
                    paragraph_index, sentence_index,
                    paragraph_start + sentence_start, sentence_length,
                )
            paragraph_ranges[paragraph_index] = (
                paragraph_start, len(values),
            )
        return _StructuralDocument(
            values, paragraph_ranges, sentence_ranges,
        )

    return (
        build(previous, previous_targets),
        build(current, current_targets),
    )


def _sentence_occurrence_paths(revision):
    paths = {}
    for paragraph_index, paragraph in _ordered_paragraph_occurrences(revision):
        for sentence_index, sentence in _ordered_sentence_occurrences(paragraph):
            paths[id(sentence)] = (paragraph_index, sentence_index)
    return paths


def _link_spans(tokens):
    # Capture only the target portion of each internal link. Boundary recovery is intentionally disabled for piped links because target/display edits need stricter handling than plain target extension.
    spans = []
    stack = []
    for index, token in enumerate(tokens):
        if token == '[[':
            target, target_end = _tokens_until(tokens, index + 1, ('|', ']]'))
            stack.append({
                'open': index,
                'target_start': index + 1,
                'target_end': target_end,
                'target': target,
                'has_option': target_end < len(tokens) and tokens[target_end] == '|',
            })
        elif token == ']]' and stack:
            frame = stack.pop()
            frame['close'] = index
            spans.append(frame)
    return spans


def _find_contiguous_subsequence(tokens, needle):
    if not needle or len(needle) > len(tokens):
        return None
    limit = len(tokens) - len(needle) + 1
    for start in range(limit):
        if tokens[start:start + len(needle)] == needle:
            return start
    return None


def _link_target_reused(prev_link, curr_link, prev_for_curr):
    if prev_link['has_option'] or curr_link['has_option']:
        return False

    prev_target = prev_link['target']
    curr_target = curr_link['target']
    if not prev_target or not curr_target:
        return False

    target_offset = _find_contiguous_subsequence(curr_target, prev_target)
    if target_offset is None:
        return False
    # Require the old target to remain a substantial part of the new target. This handles target extensions without letting short shared prefixes make unrelated links inherit each other's boundary tokens.
    if len(prev_target) * 2 < len(curr_target):
        return False

    for offset in range(len(prev_target)):
        curr_index = curr_link['target_start'] + target_offset + offset
        prev_index = prev_link['target_start'] + offset
        if curr_index >= len(prev_for_curr) or prev_for_curr[curr_index] != prev_index:
            return False
    return True


def _recover_edited_link_boundaries(text_prev, text_curr, ledger):
    # Link delimiters are keyed by full target in _word_match_keys, so an edited target can leave [[ and ]] unmatched even when the target body was reused. Recover those delimiters after the body tokens have already matched.
    curr_links_by_first = defaultdict(list)
    for curr_link in _link_spans(text_curr):
        if curr_link['target']:
            curr_links_by_first[curr_link['target'][0]].append(curr_link)

    for prev_link in _link_spans(text_prev):
        if not prev_link['target']:
            continue
        for curr_link in curr_links_by_first.get(prev_link['target'][0], ()):
            if not _link_target_reused(prev_link, curr_link, ledger.prev_for_curr):
                continue
            ledger.propose_pairs((
                (curr_link['open'], prev_link['open']),
                (curr_link['close'], prev_link['close']),
            ), WORD_MATCH_CONF_STRUCTURAL_BOUNDARY, 'edited-link-boundary')
            break


def _word_match_drift_limit(prev_len, curr_len):
    return max(WORD_MATCH_MAX_DRIFT_MIN,
               int(WORD_MATCH_MAX_DRIFT_RATIO * max(prev_len, curr_len)))


def _word_match_pair_estimate(prev_tokens, curr_tokens):
    prev_counts = Counter(prev_tokens)
    curr_counts = Counter(curr_tokens)
    total = 0
    for token, prev_count in prev_counts.items():
        total += prev_count * curr_counts.get(token, 0)
    return total


def _nearest_word_matches(prev_tokens, curr_tokens, prev_offset, curr_offset, max_drift):
    positions_by_token = defaultdict(list)
    for prev_index, token in enumerate(prev_tokens):
        positions_by_token[token].append(prev_index)

    curr_to_prev = {}
    used_prev = set()
    for curr_index, token in enumerate(curr_tokens):
        positions = positions_by_token.get(token)
        if not positions:
            continue

        expected_prev_index = curr_offset + curr_index - prev_offset
        right = bisect_left(positions, expected_prev_index)
        left = right - 1
        best_prev = None
        best_distance = None
        curr_abs_index = curr_offset + curr_index

        while left >= 0 or right < len(positions):
            if left >= 0:
                prev_index = positions[left]
                distance = abs((prev_offset + prev_index) - curr_abs_index)
                if distance > max_drift or (best_distance is not None and distance > best_distance):
                    left = -1
                else:
                    if prev_index not in used_prev and (
                            best_distance is None or distance < best_distance or
                            (distance == best_distance and prev_index < best_prev)):
                        best_prev = prev_index
                        best_distance = distance
                    left -= 1
            if right < len(positions):
                prev_index = positions[right]
                distance = abs((prev_offset + prev_index) - curr_abs_index)
                if distance > max_drift or (best_distance is not None and distance > best_distance):
                    right = len(positions)
                else:
                    if prev_index not in used_prev and (
                            best_distance is None or distance < best_distance or
                            (distance == best_distance and prev_index < best_prev)):
                        best_prev = prev_index
                        best_distance = distance
                    right += 1

        if best_prev is not None:
            curr_to_prev[curr_index] = best_prev
            used_prev.add(best_prev)
    return curr_to_prev


def _assign_word_match(prev_for_curr, match_conf, prev_used_by, curr_index, prev_index, confidence):
    old_prev = prev_for_curr[curr_index]
    if old_prev is not None:
        if match_conf[curr_index] >= confidence:
            return False
        if prev_used_by.get(old_prev) == curr_index:
            del prev_used_by[old_prev]

    old_curr = prev_used_by.get(prev_index)
    if old_curr is not None:
        if match_conf[old_curr] >= confidence:
            return False
        prev_for_curr[old_curr] = None
        match_conf[old_curr] = 0

    prev_for_curr[curr_index] = prev_index
    match_conf[curr_index] = confidence
    prev_used_by[prev_index] = curr_index
    return True


def _unassign_word_match(prev_for_curr, match_conf, prev_used_by, curr_index):
    prev_index = prev_for_curr[curr_index]
    if prev_index is None:
        return False
    if prev_used_by.get(prev_index) == curr_index:
        del prev_used_by[prev_index]
    prev_for_curr[curr_index] = None
    match_conf[curr_index] = 0
    return True


class _MatchCandidateLedger(object):
    """Collect competing proposals and resolve Word reuse in one place.

    The provisional arrays reproduce the old stage-by-stage eligibility view
    for candidate generators which still need to inspect earlier evidence.
    They never mutate Word objects.  ``resolve`` disregards that provisional
    history and derives the final one-to-one mapping from all candidates.
    """

    def __init__(self, prev_length, curr_length):
        self.prev_length = prev_length
        self.prev_for_curr = [None] * curr_length
        self.match_conf = [0] * curr_length
        self.prev_used_by = {}
        self.candidates = []
        self.blocked_curr = set()
        self._next_order = 0

    def propose_pairs(self, pairs, confidence, source, support=0,
                      displacement=None, paths=None):
        pairs = tuple(pairs)
        if not pairs:
            return None
        if paths is not None:
            paths = tuple(paths)
            if len(paths) != len(pairs):
                raise ValueError("candidate paths must align with candidate pairs")
        if displacement is None:
            displacement = min(abs(prev_index - curr_index)
                               for curr_index, prev_index in pairs)
        candidate = _MatchCandidate(
            pairs, confidence, source, support, displacement, self._next_order,
            paths=paths,
        )
        self._next_order += 1
        self.candidates.append(candidate)

        # Keep the legacy confidence view solely for later candidate discovery.
        for curr_index, prev_index in pairs:
            if curr_index in self.blocked_curr:
                continue
            _assign_word_match(
                self.prev_for_curr, self.match_conf, self.prev_used_by,
                curr_index, prev_index, confidence,
            )
        return candidate

    def propose(self, curr_index, prev_index, confidence, source, support=0):
        return self.propose_pairs(
            ((curr_index, prev_index),), confidence, source, support=support,
            displacement=abs(prev_index - curr_index),
        )

    def block_current(self, curr_index):
        # Stale-edge rejection intentionally leaves the token unmatched rather
        # than reviving evidence which the legacy matcher had already displaced.
        self.blocked_curr.add(curr_index)
        _unassign_word_match(
            self.prev_for_curr, self.match_conf, self.prev_used_by, curr_index,
        )

    def resolve(self):
        """Resolve all proposals by categorical evidence and stable tie rules."""
        higher_candidates = []
        lower_candidates = []
        structural_candidates = []
        for candidate in self.candidates:
            is_structural = candidate.source.startswith('structural-')
            if is_structural:
                structural_candidates.append(candidate)
            elif candidate.confidence > WORD_MATCH_CONF_STRUCTURAL_GAP:
                higher_candidates.append(candidate)
            else:
                lower_candidates.append(candidate)

        def candidate_edges(candidates):
            edges = []
            for candidate in candidates:
                for pair_order, (curr_index, prev_index) in enumerate(candidate.pairs):
                    if curr_index in self.blocked_curr:
                        continue
                    edges.append((
                        -candidate.confidence,
                        candidate.order,
                        pair_order,
                        curr_index,
                        prev_index,
                    ))
            return edges

        legacy_prev_for_curr = [None] * len(self.prev_for_curr)
        legacy_prev_used_by = {}
        for _, _, _, curr_index, prev_index in sorted(candidate_edges(
                higher_candidates + lower_candidates)):
            if (legacy_prev_for_curr[curr_index] is not None or
                    prev_index in legacy_prev_used_by):
                continue
            legacy_prev_for_curr[curr_index] = prev_index
            legacy_prev_used_by[prev_index] = curr_index

        prev_for_curr = [None] * len(self.prev_for_curr)
        prev_used_by = {}

        def assign_edge(curr_index, prev_index):
            if (prev_for_curr[curr_index] is not None or
                    prev_index in prev_used_by):
                return False
            prev_for_curr[curr_index] = prev_index
            prev_used_by[prev_index] = curr_index
            return True

        # Lock evidence above the structural-gap tier first.  These are exact
        # edges and full-revision-unique moved runs, which must not be displaced
        # by a merely local structural correspondence.
        for _, _, _, curr_index, prev_index in sorted(
                candidate_edges(higher_candidates)):
            assign_edge(curr_index, prev_index)

        # Build bounded conflict components before choosing structural runs.
        # A structural and a lower-tier candidate conflict when either claims
        # the same current slot or reuses the same previous slot.  Components
        # are expanded through both candidate families so a large duplicate
        # permutation cannot masquerade as many independent local decisions.
        structural_by_curr = defaultdict(set)
        structural_by_prev = defaultdict(set)
        lower_by_curr = defaultdict(set)
        lower_by_prev = defaultdict(set)
        for index, candidate in enumerate(structural_candidates):
            for curr_index, prev_index in candidate.pairs:
                if curr_index not in self.blocked_curr:
                    structural_by_curr[curr_index].add(index)
                    structural_by_prev[prev_index].add(index)
        for index, candidate in enumerate(lower_candidates):
            for curr_index, prev_index in candidate.pairs:
                if curr_index not in self.blocked_curr:
                    lower_by_curr[curr_index].add(index)
                    lower_by_prev[prev_index].add(index)

        eligible_structural = set()
        visited_structural = set()
        for initial_index in range(len(structural_candidates)):
            if initial_index in visited_structural:
                continue
            component_structural = set()
            component_lower = set()
            pending_structural = [initial_index]
            pending_lower = []
            while pending_structural or pending_lower:
                while pending_structural:
                    candidate_index = pending_structural.pop()
                    if candidate_index in component_structural:
                        continue
                    component_structural.add(candidate_index)
                    candidate = structural_candidates[candidate_index]
                    for curr_index, prev_index in candidate.pairs:
                        pending_structural.extend(
                            structural_by_curr.get(curr_index, ()))
                        pending_structural.extend(
                            structural_by_prev.get(prev_index, ()))
                        pending_lower.extend(lower_by_curr.get(curr_index, ()))
                        pending_lower.extend(lower_by_prev.get(prev_index, ()))
                while pending_lower:
                    candidate_index = pending_lower.pop()
                    if candidate_index in component_lower:
                        continue
                    component_lower.add(candidate_index)
                    candidate = lower_candidates[candidate_index]
                    for curr_index, prev_index in candidate.pairs:
                        pending_structural.extend(
                            structural_by_curr.get(curr_index, ()))
                        pending_structural.extend(
                            structural_by_prev.get(prev_index, ()))
                        pending_lower.extend(
                            lower_by_curr.get(curr_index, ()))
                        pending_lower.extend(
                            lower_by_prev.get(prev_index, ()))

            visited_structural.update(component_structural)
            component_curr = set()
            component_prev = set()
            for candidate_index in component_structural:
                for curr_index, prev_index in structural_candidates[candidate_index].pairs:
                    component_curr.add(curr_index)
                    component_prev.add(prev_index)
            for candidate_index in component_lower:
                for curr_index, prev_index in lower_candidates[candidate_index].pairs:
                    component_curr.add(curr_index)
                    component_prev.add(prev_index)
            if (len(component_curr) + len(component_prev) <=
                    WORD_MATCH_STRUCTURAL_MAX_CONFLICT_TOKENS):
                eligible_structural.update(component_structural)

        # Structural evidence belongs to the complete run.  Greedily taking
        # individual edges from overlapping runs can synthesize an assignment
        # which no candidate generator proposed.  Accept a run atomically, or
        # reject it if any edge conflicts with a higher or already-selected
        # structural mapping.  Identical locked edges may be clipped because
        # they independently prove the same correspondence.
        for candidate in sorted(
                (structural_candidates[index]
                 for index in eligible_structural),
                key=lambda item: (
                    -item.confidence, -item.support, item.displacement,
                    item.order,
                )):
            pairs = tuple(
                pair for pair in candidate.pairs
                if pair[0] not in self.blocked_curr
            )
            if not pairs:
                continue
            conflicts = False
            for curr_index, prev_index in pairs:
                assigned_prev = prev_for_curr[curr_index]
                assigned_curr = prev_used_by.get(prev_index)
                if ((assigned_prev is not None and assigned_prev != prev_index) or
                        (assigned_curr is not None and assigned_curr != curr_index)):
                    conflicts = True
                    break
            if conflicts:
                continue
            for curr_index, prev_index in pairs:
                if prev_for_curr[curr_index] is None:
                    assign_edge(curr_index, prev_index)

        # Preserve the established ordering of the remaining matcher evidence.
        for _, _, _, curr_index, prev_index in sorted(
                candidate_edges(lower_candidates)):
            assign_edge(curr_index, prev_index)

        deleted_prev = [
            index for index in range(self.prev_length)
            if index not in prev_used_by
        ]
        cardinality_gain = len(prev_used_by) - len(legacy_prev_used_by)
        # A structural reassignment may exchange identities at equal
        # cardinality or recover a complete informative run.  It must never
        # discard more established matches than it adds, nor promote an
        # isolated one- or two-token recovery.
        if (cardinality_gain < 0 or
                0 < cardinality_gain < WORD_MATCH_STRUCTURAL_MIN_RUN_INFO):
            legacy_deleted_prev = [
                index for index in range(self.prev_length)
                if index not in legacy_prev_used_by
            ]
            return legacy_prev_for_curr, legacy_deleted_prev
        return prev_for_curr, deleted_prev


def _is_low_authorship_edge_token(token):
    return isinstance(token, str) and len(token) <= 2 and token.isalpha()


def _demote_stale_suffix_edge_matches(text_prev, text_curr, prev_words,
                                      ledger, prefix_len, suffix_len):
    # Edge suffixes are usually reliable, but after a large pure deletion they can over-preserve old "glue" tokens at the start of a mature rewritten suffix. Demote only those low-authorship tokens and leave content words and replacement edits alone.
    if not prev_words or not suffix_len:
        return

    prev_rewritten_tokens = len(text_prev) - prefix_len - suffix_len
    curr_rewritten_tokens = len(text_curr) - prefix_len - suffix_len
    if curr_rewritten_tokens != 0:
        return
    if prev_rewritten_tokens < WORD_MATCH_EDGE_STALE_REWRITE_MIN_TOKENS:
        return

    suffix_curr_start = len(text_curr) - suffix_len
    limit = min(suffix_len, WORD_MATCH_EDGE_STALE_WINDOW)
    for offset in range(limit):
        curr_index = suffix_curr_start + offset
        prev_index = ledger.prev_for_curr[curr_index]
        if prev_index is None or ledger.match_conf[curr_index] != WORD_MATCH_CONF_EDGE:
            continue
        if prev_index >= len(prev_words):
            continue
        word_prev = prev_words[prev_index]
        if word_prev.origin_rev_id == word_prev.last_rev_id:
            continue
        if _is_low_authorship_edge_token(text_curr[curr_index]):
            ledger.block_current(curr_index)


def _contiguous_spans(indices):
    spans = []
    if not indices:
        return spans
    start = indices[0]
    previous = start
    for index in indices[1:]:
        if index == previous + 1:
            previous = index
        else:
            spans.append((start, previous + 1))
            start = index
            previous = index
    spans.append((start, previous + 1))
    return spans


def _is_informative_move_token(token):
    if (not isinstance(token, str) or not token or
            token in WORD_MATCH_MOVE_STRUCTURAL_TOKENS):
        return False
    # Most word tokens start or end in an alphanumeric character.  Preserve
    # the exact predicate while avoiding a generator allocation and a full
    # Unicode scan for that common case.
    return (token[0].isalnum() or token[-1].isalnum() or
            any(char.isalnum() for char in token[1:-1]))


def _informative_move_token_prefix(tokens):
    prefix = [0]
    total = 0
    for token in tokens:
        if _is_informative_move_token(token):
            total += 1
        prefix.append(total)
    return prefix


def _index_move_ngrams(keys, spans, ngram_size, informative_prefix):
    index = defaultdict(list)
    for start, end in spans:
        for position in range(start, end - ngram_size + 1):
            info_count = informative_prefix[position + ngram_size] - informative_prefix[position]
            if info_count >= WORD_MATCH_MOVE_MIN_INFO_TOKENS:
                key = tuple(keys[position:position + ngram_size])
                index[key].append(position)
    return index


def _content_runs(tokens):
    run = []
    for token in tokens:
        if _is_informative_move_token(token):
            run.append(token)
        elif run:
            yield run
            run = []
    if run:
        yield run


def _longest_content_core(tokens):
    runs = list(_content_runs(tokens))
    if not runs:
        return ()
    return tuple(max(runs, key=len))


def _count_subsequence(tokens, needle):
    if not needle:
        return 0
    count = 0
    needle_len = len(needle)
    for index in range(len(tokens) - needle_len + 1):
        if tuple(tokens[index:index + needle_len]) == needle:
            count += 1
    return count


def _tokens_equal_at(tokens, start, needle):
    for offset, token in enumerate(needle):
        if tokens[start + offset] != token:
            return False
    return True


def _pair_positions(tokens, count_state):
    # Moved-run safety only needs to distinguish unique from repeated runs. Indexing pair positions lets longer subsequence counts probe a small candidate list instead of scanning the whole article text each time.
    pair_indexes = count_state.setdefault('pair_indexes', {})
    index_key = (id(tokens), len(tokens), 'pair_positions')
    positions = pair_indexes.get(index_key)
    if positions is None:
        positions = defaultdict(list)
        for index in range(len(tokens) - 1):
            positions[(tokens[index], tokens[index + 1])].append(index)
        pair_indexes[index_key] = positions
    return positions


def _count_subsequence_cached(tokens, needle, count_state):
    if not needle:
        return 0
    needle = tuple(needle)
    needle_len = len(needle)
    if count_state is None:
        return _count_subsequence(tokens, needle)

    counts = count_state['counts']
    count_key = (id(tokens), len(tokens), needle)
    count = counts.get(count_key)
    if count is not None:
        return count

    if needle_len == 1:
        count = 0
        for token in tokens:
            if token == needle[0]:
                count += 1
                if count > 1:
                    break
        counts[count_key] = count
        return count

    positions = _pair_positions(tokens, count_state)
    first_positions = positions.get((needle[0], needle[1]), ())
    last_positions = positions.get((needle[-2], needle[-1]), ())
    if _structural_native is not None:
        count = _structural_native.count_subsequence_at_positions(
            tokens, needle, first_positions, last_positions,
        )
        counts[count_key] = count
        return count
    max_start = len(tokens) - needle_len
    count = 0
    if len(last_positions) < len(first_positions):
        for pair_index in last_positions:
            start = pair_index - needle_len + 2
            if start >= 0 and start <= max_start and _tokens_equal_at(tokens, start, needle):
                count += 1
                if count > 1:
                    break
    else:
        for start in first_positions:
            if start <= max_start and _tokens_equal_at(tokens, start, needle):
                count += 1
                if count > 1:
                    break
    counts[count_key] = count
    return count


def _link_anchor_bounds(tokens):
    try:
        link_open = tokens.index('[[')
        link_close = tokens.index(']]', link_open + 1)
    except ValueError:
        return None
    if link_open < 2 or not all(
            _is_informative_move_token(token) for token in tokens[link_open - 2:link_open]):
        return None
    if sum(_is_informative_move_token(token) for token in tokens[link_open + 1:link_close]) < 2:
        return None
    return link_open, link_close


def _template_field_anchor(run, content_core):
    """Return whether a content core follows a raw ``| name =`` opener."""
    core_length = len(content_core)
    for start in range(len(run) - core_length + 1):
        if tuple(run[start:start + core_length]) != content_core:
            continue
        if start < 3 or run[start - 1] != '=':
            continue
        index = start - 2
        while index >= 0 and _is_informative_move_token(run[index]):
            index -= 1
        if index >= 0 and index < start - 2 and run[index] == '|':
            return True
    return False


def _copy_safe_moved_run(count_text_prev, count_text_curr, text_curr, curr_start, length, count_state):
    run = text_curr[curr_start:curr_start + length]
    content_core = _longest_content_core(run)
    # Four content tokens are normally required to establish a moved run. Allow
    # an exact unique three-token value only when the run itself contains its
    # template-field opener. This does not admit weak prose anchors such as
    # "across the country". A bare digit-bearing exception is deliberately not
    # used here: repeated citation dates can otherwise swap token lineages.
    if (len(content_core) >= WORD_MATCH_MOVE_MIN_ANCHOR_INFO_TOKENS or
            (len(content_core) >= WORD_MATCH_MOVE_MIN_INFO_TOKENS and
             _template_field_anchor(run, content_core))):
        return (_count_subsequence_cached(count_text_prev, content_core, count_state) == 1 and
                _count_subsequence_cached(count_text_curr, content_core, count_state) == 1)
    if _link_anchor_bounds(run) is None:
        return False
    return (_count_subsequence_cached(count_text_prev, run, count_state) == 1 and
            _count_subsequence_cached(count_text_curr, run, count_state) == 1)


def _has_unique_link_move_window(count_text_prev, count_text_curr, text_curr,
                                 curr_index, run_start, run_end, count_state):
    max_window = min(run_end - run_start, max(WORD_MATCH_MOVE_NGRAM_SIZES))
    for ngram_size in range(max_window, WORD_MATCH_MOVE_MIN_LINK_WINDOW - 1, -1):
        earliest = max(run_start, curr_index - ngram_size + 1)
        latest = min(curr_index, run_end - ngram_size)
        for start in range(earliest, latest + 1):
            needle = tuple(text_curr[start:start + ngram_size])
            anchor = _link_anchor_bounds(needle)
            if anchor is None:
                continue
            link_open, _ = anchor
            if not start + link_open - 2 <= curr_index < start + link_open:
                continue
            if (_count_subsequence_cached(count_text_prev, needle, count_state) == 1 and
                    _count_subsequence_cached(count_text_curr, needle, count_state) == 1):
                return True
    return False


def _unique_moved_run_coverage(count_text_prev, count_text_curr, text_curr,
                               run_start, run_end, count_state,
                               minimum_informative=WORD_MATCH_MOVE_MIN_INFO_TOKENS):
    run_length = run_end - run_start
    max_window = min(run_length, max(WORD_MATCH_MOVE_NGRAM_SIZES))
    if max_window < minimum_informative:
        return set()

    informative_prefix = [0]
    informative_count = 0
    for token in text_curr[run_start:run_end]:
        if _is_informative_move_token(token):
            informative_count += 1
        informative_prefix.append(informative_count)

    covered = set()
    for window_size in range(minimum_informative, max_window + 1):
        for relative_start in range(run_length - window_size + 1):
            relative_end = relative_start + window_size
            if (informative_prefix[relative_end] - informative_prefix[relative_start] <
                    minimum_informative):
                continue
            start = run_start + relative_start
            needle = tuple(text_curr[start:start + window_size])
            if (_count_subsequence_cached(count_text_prev, needle, count_state) == 1 and
                    _count_subsequence_cached(count_text_curr, needle, count_state) == 1):
                covered.update(range(start, start + window_size))
    return covered


def _can_assign_moved_match(ledger, curr_index, prev_index):
    if ledger.match_conf[curr_index] >= WORD_MATCH_CONF_MOVED_RUN:
        return False
    old_curr = ledger.prev_used_by.get(prev_index)
    return old_curr is None or ledger.match_conf[old_curr] < WORD_MATCH_CONF_MOVED_RUN


def _moved_run_confidence(length, seed_length):
    bonus = min(WORD_MATCH_CONF_EDGE - WORD_MATCH_CONF_MOVED_RUN - 1,
                max(0, length - seed_length))
    return WORD_MATCH_CONF_MOVED_RUN + bonus


def _extend_moved_run(prev_keys, curr_keys, ledger, prev_start, curr_start, length):
    left = 0
    while curr_start - left - 1 >= 0 and prev_start - left - 1 >= 0:
        curr_index = curr_start - left - 1
        prev_index = prev_start - left - 1
        if prev_keys[prev_index] != curr_keys[curr_index]:
            break
        if not _can_assign_moved_match(ledger, curr_index, prev_index):
            break
        left += 1

    right = 0
    while curr_start + length + right < len(curr_keys) and prev_start + length + right < len(prev_keys):
        curr_index = curr_start + length + right
        prev_index = prev_start + length + right
        if prev_keys[prev_index] != curr_keys[curr_index]:
            break
        if not _can_assign_moved_match(ledger, curr_index, prev_index):
            break
        right += 1

    return prev_start - left, curr_start - left, length + left + right


def _move_ngram_sizes(recoverable_count):
    sizes = list(WORD_MATCH_MOVE_NGRAM_SIZES)
    while len(sizes) > 2 and recoverable_count * len(sizes) > WORD_MATCH_MOVE_MAX_WINDOWS:
        sizes.pop()
    return sizes


def _raw_context_ngram_sizes(prev_len, curr_len):
    sizes = []
    indexed_windows = 0
    for ngram_size in WORD_MATCH_MOVE_NGRAM_SIZES:
        window_count = (max(0, prev_len - ngram_size + 1) +
                        max(0, curr_len - ngram_size + 1))
        if not window_count:
            continue
        if indexed_windows + window_count > WORD_MATCH_MOVE_MAX_WINDOWS:
            break
        sizes.append(ngram_size)
        indexed_windows += window_count
    return sizes


_TEMPLATE_PIPE_KEY_KINDS = frozenset(('template-field', 'template-arg'))


def _is_template_pipe_key(key):
    return (isinstance(key, tuple) and len(key) == 5 and
            key[:2] == ('wikitext', '|') and
            key[2] in _TEMPLATE_PIPE_KEY_KINDS and
            isinstance(key[3], tuple))


def _pipe_key_changed_only_by_template_spacing(prev_key, curr_key):
    if not (_is_template_pipe_key(prev_key) and
            _is_template_pipe_key(curr_key)):
        return False
    if prev_key[:3] != curr_key[:3] or prev_key[4:] != curr_key[4:]:
        return False
    prev_name = prev_key[3]
    curr_name = curr_key[3]
    return prev_name != curr_name and ''.join(prev_name) == ''.join(curr_name)


def _has_template_name_spacing_change(prev_keys, curr_keys):
    forms = []
    for keys in (prev_keys, curr_keys):
        by_compact_name = defaultdict(set)
        for key in keys:
            if _is_template_pipe_key(key):
                by_compact_name[''.join(key[3])].add(key[3])
        forms.append(by_compact_name)

    prev_forms, curr_forms = forms
    for compact_name in set(prev_forms).intersection(curr_forms):
        if any(
                prev_name != curr_name
                for prev_name in prev_forms[compact_name]
                for curr_name in curr_forms[compact_name]):
            return True
    return False


def _recover_unique_template_field_words(text_prev, text_curr,
                                         prev_keys, curr_keys,
                                         ledger,
                                         full_text_prev=None, full_text_curr=None,
                                         get_full_texts=None, count_state=None):
    # Template renames change the contextual keys of their separators. Preserve
    # still-unmatched field content when it remains bracketed by the same exact
    # raw context and both separator keys changed together.
    recoverable_count = len(text_prev) + len(text_curr)
    if recoverable_count < WORD_MATCH_MOVE_MIN_RECOVERABLE_TOKENS:
        return
    if not _has_template_name_spacing_change(prev_keys, curr_keys):
        return

    if not any(
            _is_informative_move_token(token) and index not in ledger.prev_used_by
            for index, token in enumerate(text_prev)):
        return
    if not any(
            _is_informative_move_token(token) and ledger.prev_for_curr[index] is None
            for index, token in enumerate(text_curr)):
        return

    ngram_sizes = _raw_context_ngram_sizes(len(text_prev), len(text_curr))
    if not ngram_sizes:
        return

    count_texts = [full_text_prev, full_text_curr]

    def ensure_count_texts():
        if count_texts[0] is None or count_texts[1] is None:
            if get_full_texts is not None:
                count_texts[0], count_texts[1] = get_full_texts()
            else:
                count_texts[0] = text_prev
                count_texts[1] = text_curr
        return count_texts[0], count_texts[1]

    if count_state is None:
        count_state = {'counts': {}}
    prev_info = _informative_move_token_prefix(text_prev)
    curr_info = _informative_move_token_prefix(text_curr)
    prev_spans = [(0, len(text_prev))]
    curr_spans = [(0, len(text_curr))]

    for ngram_size in ngram_sizes:
        prev_contexts = _index_move_ngrams(
            text_prev, prev_spans, ngram_size, prev_info,
        )
        curr_contexts = _index_move_ngrams(
            text_curr, curr_spans, ngram_size, curr_info,
        )
        candidates = []
        for key, prev_positions in prev_contexts.items():
            curr_positions = curr_contexts.get(key)
            if (len(prev_positions) != 1 or not curr_positions or
                    len(curr_positions) != 1):
                continue
            prev_start = prev_positions[0]
            curr_start = curr_positions[0]
            changed_pipes = tuple(
                offset for offset in range(ngram_size)
                if text_curr[curr_start + offset] == '|' and
                _pipe_key_changed_only_by_template_spacing(
                    prev_keys[prev_start + offset],
                    curr_keys[curr_start + offset],
                )
            )
            if len(changed_pipes) >= 2:
                candidates.append((abs(prev_start - curr_start),
                                   prev_start, curr_start, key, changed_pipes))

        for _, prev_start, curr_start, key, changed_pipes in sorted(candidates, reverse=True):
            count_text_prev, count_text_curr = ensure_count_texts()
            if (_count_subsequence_cached(count_text_prev, key, count_state) != 1 or
                    _count_subsequence_cached(count_text_curr, key, count_state) != 1):
                continue
            for offset in range(ngram_size):
                prev_index = prev_start + offset
                curr_index = curr_start + offset
                if not _is_informative_move_token(text_curr[curr_index]):
                    continue
                if not (any(pipe < offset for pipe in changed_pipes) and
                        any(pipe > offset for pipe in changed_pipes)):
                    continue
                if (ledger.prev_for_curr[curr_index] is not None or
                        prev_index in ledger.prev_used_by):
                    continue
                ledger.propose(curr_index, prev_index,
                               WORD_MATCH_CONF_MOVED_RUN,
                               'template-field')


def _recoverable_indices_from_spans(spans, start_allowed):
    indices = []
    for span_start, span_end in spans:
        for index in range(span_start, span_end):
            if start_allowed(index):
                indices.append(index)
    return indices


def _template_field_before_content(tokens, content_start):
    if content_start < 3 or tokens[content_start - 1] != '=':
        return None
    field_end = content_start - 1
    index = field_end - 1
    while index >= 0 and tokens[index] != '|':
        if not (_is_informative_move_token(tokens[index]) or tokens[index] == '_'):
            return None
        index -= 1
    if index < 0 or index == field_end - 1:
        return None
    raw_field = tuple(tokens[index + 1:field_end])
    normalized = ''.join(
        token for token in raw_field if _is_informative_move_token(token)
    ).rstrip('0123456789')
    if not normalized:
        return None
    return raw_field, normalized


def _recover_unique_short_numeric_template_fields(
        text_prev, text_curr, ledger,
        full_text_prev=None, full_text_curr=None,
        get_full_texts=None,
        prev_candidate_spans=None,
        curr_candidate_spans=None,
        count_state=None):
    """Repair a renamed numeric template field split by one weaker match.

    A SequenceMatcher equal opcode can claim the year of an otherwise intact
    date for an unrelated newly added date. Moved-run recovery then cannot seed
    the old three-token run because one previous position lies outside its
    replace spans. Reconsider only maximal, digit-bearing three-token values
    whose template field changed to an equivalent compact name (for example,
    ``term_start`` to ``termstart2``), whose current positions are all in
    replace spans, and whose previous occurrence retains at least two positions
    there. Exact full-revision uniqueness and the confidence ladder still
    decide the assignment. Same-name citation date fields are excluded because
    their repeated values can otherwise exchange token lineages.
    """
    run_length = WORD_MATCH_MOVE_MIN_INFO_TOKENS
    if run_length != 3:
        return

    if curr_candidate_spans is None:
        candidate_ranges = []
        index = 0
        while index < len(text_curr):
            if not _is_informative_move_token(text_curr[index]):
                index += 1
                continue
            run_start = index
            index += 1
            while index < len(text_curr) and _is_informative_move_token(text_curr[index]):
                index += 1
            if index - run_start == run_length:
                candidate_ranges.append((run_start, index))
    else:
        # The conflict this pass repairs leaves the complete current value as
        # one three-token replace span. Inspecting only those spans makes the
        # common path proportional to the opcode count, without expanding span
        # sets or scanning the complete unmatched text.
        candidate_ranges = [
            (start, end) for start, end in curr_candidate_spans
            if end - start == run_length
        ]

    candidates = []
    for run_start, run_end in candidate_ranges:
        if ((run_start > 0 and _is_informative_move_token(text_curr[run_start - 1])) or
                (run_end < len(text_curr) and _is_informative_move_token(text_curr[run_end]))):
            continue
        if not all(_is_informative_move_token(text_curr[position])
                   for position in range(run_start, run_end)):
            continue
        needle = tuple(text_curr[run_start:run_end])
        if not any(char.isdigit() for token in needle for char in token):
            continue
        curr_field = _template_field_before_content(text_curr, run_start)
        if curr_field is None:
            continue
        if any(ledger.match_conf[position] >= WORD_MATCH_CONF_MOVED_RUN
               for position in range(run_start, run_end)):
            continue
        candidates.append((run_start, needle, curr_field))

    if not candidates:
        return

    candidate_needles = set(needle for _, needle, _ in candidates)
    prev_positions = defaultdict(list)
    for prev_start in range(len(text_prev) - run_length + 1):
        prev_end = prev_start + run_length
        needle = tuple(text_prev[prev_start:prev_end])
        if needle not in candidate_needles:
            continue
        if ((prev_start > 0 and _is_informative_move_token(text_prev[prev_start - 1])) or
                (prev_end < len(text_prev) and _is_informative_move_token(text_prev[prev_end]))):
            continue
        prev_positions[needle].append(prev_start)

    if count_state is None:
        count_state = {'counts': {}}
    count_texts = [full_text_prev, full_text_curr]

    def ensure_count_texts():
        if count_texts[0] is None or count_texts[1] is None:
            if get_full_texts is not None:
                count_texts[0], count_texts[1] = get_full_texts()
            else:
                count_texts[0] = text_prev
                count_texts[1] = text_curr
        return count_texts[0], count_texts[1]

    for curr_start, needle, curr_field in candidates:
        positions = prev_positions.get(needle, ())
        if len(positions) != 1:
            continue
        prev_start = positions[0]
        prev_field = _template_field_before_content(text_prev, prev_start)
        if (prev_field is None or prev_field[1] != curr_field[1] or
                prev_field[0] == curr_field[0]):
            continue
        previous_indices = range(prev_start, prev_start + run_length)
        if prev_candidate_spans is not None:
            candidate_count = 0
            for span_start, span_end in prev_candidate_spans:
                if span_end <= prev_start:
                    continue
                if span_start >= prev_start + run_length:
                    break
                candidate_count += max(
                    0,
                    min(span_end, prev_start + run_length) - max(span_start, prev_start),
                )
            if candidate_count < run_length - 1:
                continue
        if any(
                ledger.prev_used_by.get(position) is not None and
                ledger.match_conf[ledger.prev_used_by[position]] >=
                WORD_MATCH_CONF_MOVED_RUN
                for position in previous_indices):
            continue

        count_text_prev, count_text_curr = ensure_count_texts()
        if (_count_subsequence_cached(count_text_prev, needle, count_state) != 1 or
                _count_subsequence_cached(count_text_curr, needle, count_state) != 1):
            continue
        for offset in range(run_length):
            ledger.propose(
                curr_start + offset, prev_start + offset,
                WORD_MATCH_CONF_MOVED_RUN, 'template-numeric-field',
            )


def _recover_moved_word_runs(text_prev, text_curr, prev_keys, curr_keys,
                             ledger,
                             full_text_prev=None, full_text_curr=None,
                             get_full_texts=None, prev_candidate_spans=None,
                             curr_candidate_spans=None, count_state=None):
    count_texts = [full_text_prev, full_text_curr]
    if prev_candidate_spans is not None and curr_candidate_spans is not None:
        if not prev_candidate_spans or not curr_candidate_spans:
            return

    recoverable_count = len(text_prev) + len(text_curr)
    if recoverable_count < WORD_MATCH_MOVE_MIN_RECOVERABLE_TOKENS:
        return

    def ensure_count_texts():
        if count_texts[0] is None or count_texts[1] is None:
            if get_full_texts is not None:
                count_texts[0], count_texts[1] = get_full_texts()
            else:
                count_texts[0] = text_prev
                count_texts[1] = text_curr
        return count_texts[0], count_texts[1]

    prev_info_prefix = _informative_move_token_prefix(prev_keys)
    curr_info_prefix = _informative_move_token_prefix(curr_keys)
    if count_state is None:
        count_state = {'counts': {}}
    checked_runs = set()
    for ngram_size in _move_ngram_sizes(recoverable_count):
        protected_prev = set(
            prev_index for curr_index, prev_index in enumerate(ledger.prev_for_curr)
            if (prev_index is not None and
                ledger.match_conf[curr_index] >= WORD_MATCH_CONF_MOVED_RUN)
        )
        if prev_candidate_spans is None:
            recoverable_prev = [
                index for index in range(len(text_prev))
                if index not in protected_prev
            ]
        else:
            recoverable_prev = _recoverable_indices_from_spans(
                prev_candidate_spans,
                lambda index: index not in protected_prev,
            )
        if curr_candidate_spans is None:
            recoverable_curr = [
                index for index, confidence in enumerate(ledger.match_conf)
                if confidence < WORD_MATCH_CONF_MOVED_RUN
            ]
        else:
            recoverable_curr = _recoverable_indices_from_spans(
                curr_candidate_spans,
                lambda index: ledger.match_conf[index] < WORD_MATCH_CONF_MOVED_RUN,
            )

        prev_spans = _contiguous_spans(recoverable_prev)
        curr_spans = _contiguous_spans(recoverable_curr)
        prev_ngrams = _index_move_ngrams(prev_keys, prev_spans, ngram_size, prev_info_prefix)
        curr_ngrams = _index_move_ngrams(curr_keys, curr_spans, ngram_size, curr_info_prefix)

        candidates = []
        for key, prev_positions in prev_ngrams.items():
            curr_positions = curr_ngrams.get(key)
            if curr_positions and len(prev_positions) == 1 and len(curr_positions) == 1:
                candidates.append((abs(prev_positions[0] - curr_positions[0]),
                                   prev_positions[0], curr_positions[0]))

        for _, prev_start, curr_start in sorted(candidates, reverse=True):
            if any(
                not _can_assign_moved_match(
                    ledger, curr_start + offset, prev_start + offset,
                )
                for offset in range(ngram_size)
            ):
                continue

            prev_start, curr_start, length = _extend_moved_run(
                prev_keys, curr_keys, ledger, prev_start, curr_start, ngram_size,
            )
            run_key = (prev_start, curr_start, length)
            if run_key in checked_runs:
                continue
            checked_runs.add(run_key)
            count_text_prev, count_text_curr = ensure_count_texts()
            if not _copy_safe_moved_run(count_text_prev, count_text_curr, text_curr,
                                        curr_start, length, count_state):
                continue

            confidence = _moved_run_confidence(length, ngram_size)
            requires_complete_link_window = (
                len(_longest_content_core(text_curr[curr_start:curr_start + length])) <
                WORD_MATCH_MOVE_MIN_INFO_TOKENS
            )
            covered = None
            if not requires_complete_link_window:
                covered = _unique_moved_run_coverage(
                    count_text_prev, count_text_curr, text_curr,
                    curr_start, curr_start + length, count_state,
                )
            for offset in range(length):
                curr_index = curr_start + offset
                if requires_complete_link_window:
                    if not _has_unique_link_move_window(
                            count_text_prev, count_text_curr, text_curr,
                            curr_index, curr_start, curr_start + length,
                            count_state,
                    ):
                        continue
                elif curr_index not in covered:
                    continue
                ledger.propose(curr_index, prev_start + offset,
                               confidence, 'moved-run')


def _slots_grouped_by_paragraph(slots):
    grouped = defaultdict(list)
    for slot in slots:
        grouped[slot.paragraph_index].append(slot)
    return grouped


class _StructuralIndex(object):
    """Ephemeral, revision-local data shared by structural matching passes."""

    __slots__ = (
        'slots', 'paragraphs', 'keys', 'informative_prefix',
        '_occurrence_states',
    )

    def __init__(self, slots, build_occurrences=True):
        self.slots = slots
        self.paragraphs = _slots_grouped_by_paragraph(slots)
        values = [slot.value for slot in slots]
        self.keys = _word_match_keys(values)
        self.informative_prefix = [0]
        informative_total = 0
        for value in values:
            informative_total += int(_is_informative_move_token(value))
            self.informative_prefix.append(
                informative_total
            )
        self._occurrence_states = {}
        if build_occurrences:
            self._occurrence_states = self._build_occurrence_states()

    @staticmethod
    def _record_occurrence(states, key, position):
        if key in states:
            states[key] = None
        else:
            states[key] = position

    def _build_occurrence_states(self):
        """Build every structural occurrence tier in one shared index pass.

        Four- and six-token keys serve both ambiguity detection (three
        informative tokens) and anchor uniqueness (four informative tokens).
        The two occurrence states remain independent, but the normalized key
        is allocated only once per position.
        """
        states = {}
        for size in WORD_MATCH_STRUCTURAL_ANCHOR_SIZES:
            states[(size, WORD_MATCH_STRUCTURAL_MIN_ANCHOR_INFO)] = {}
        for size in WORD_MATCH_STRUCTURAL_AMBIGUITY_SIZES:
            states[(size, 3)] = {}

        keys = self.keys
        informative_prefix = self.informative_prefix
        for paragraph_index, slots in self.paragraphs.items():
            paragraph_length = len(slots)
            if not paragraph_length:
                continue
            paragraph_start = slots[0].article_index
            for size in WORD_MATCH_STRUCTURAL_INDEX_SIZES:
                if size > paragraph_length:
                    continue
                ambiguity_state = states.get((size, 3))
                anchor_state = states.get((
                    size, WORD_MATCH_STRUCTURAL_MIN_ANCHOR_INFO,
                ))
                minimum_informative = (
                    3 if ambiguity_state is not None
                    else WORD_MATCH_STRUCTURAL_MIN_ANCHOR_INFO
                )
                for start in range(paragraph_length - size + 1):
                    article_start = paragraph_start + start
                    informative = (
                        informative_prefix[article_start + size] -
                        informative_prefix[article_start]
                    )
                    if informative < minimum_informative:
                        continue
                    key = self._article_window_key(article_start, size)
                    position = None
                    if ambiguity_state is not None:
                        if key in ambiguity_state:
                            ambiguity_state[key] = None
                        else:
                            position = (paragraph_index, start)
                            ambiguity_state[key] = position
                    if (anchor_state is not None and
                            informative >=
                            WORD_MATCH_STRUCTURAL_MIN_ANCHOR_INFO):
                        if position is None:
                            position = (paragraph_index, start)
                        if key in anchor_state:
                            anchor_state[key] = None
                        else:
                            anchor_state[key] = position
        return states

    @staticmethod
    def _article_span(slots, start, end):
        if start >= end:
            return None
        article_start = slots[start].article_index
        article_end = slots[end - 1].article_index + 1
        if article_end - article_start != end - start:
            return None
        return article_start, article_end

    def informative_count(self, slots, start, end):
        span = self._article_span(slots, start, end)
        if span is not None:
            return (
                self.informative_prefix[span[1]] -
                self.informative_prefix[span[0]]
            )
        return sum(
            _is_informative_move_token(slots[index].value)
            for index in range(start, end)
        )

    def _article_window_key(self, article_start, size):
        return tuple(self.keys[article_start:article_start + size])

    def window_key(self, slots, start, end):
        span = self._article_span(slots, start, end)
        if span is not None:
            return self._article_window_key(span[0], span[1] - span[0])
        return tuple(
            self.keys[slots[index].article_index]
            for index in range(start, end)
        )

    def occurrence_states(self, size, min_informative):
        """Map a qualifying key to its sole position, or ``None`` if repeated."""
        cache_key = (size, min_informative)
        cached = self._occurrence_states.get(cache_key)
        if cached is not None:
            return cached

        states = {}
        for paragraph_index, slots in self.paragraphs.items():
            if size > len(slots):
                continue
            for start in range(len(slots) - size + 1):
                end = start + size
                if self.informative_count(slots, start, end) < min_informative:
                    continue
                key = self.window_key(slots, start, end)
                self._record_occurrence(
                    states, key, (paragraph_index, start),
                )
        self._occurrence_states[cache_key] = states
        return states


def _record_structural_anchor_occurrence(state, key, paragraph_index,
                                         start):
    if key in state:
        state[key] = None
    else:
        state[key] = (paragraph_index, start)


def _index_structural_anchor_size(structural_index, paragraph_index,
                                  size, state):
    slots = structural_index.paragraphs[paragraph_index]
    paragraph_length = len(slots)
    if size > paragraph_length:
        return
    paragraph_start = slots[0].article_index
    informative_prefix = structural_index.informative_prefix
    for start in range(paragraph_length - size + 1):
        article_start = paragraph_start + start
        informative = (
            informative_prefix[article_start + size] -
            informative_prefix[article_start]
        )
        if informative < WORD_MATCH_STRUCTURAL_MIN_ANCHOR_INFO:
            continue
        key = structural_index._article_window_key(article_start, size)
        _record_structural_anchor_occurrence(
            state, key, paragraph_index, start,
        )


def _targeted_structural_anchor_occurrences(
        prev_index, curr_index, size, prev_target_paragraphs,
        curr_target_paragraphs):
    """Count globally unique anchors incident to a residual paragraph.

    Chains whose two endpoints both lack residual words cannot produce a
    structural candidate or compete with a potentially productive pair in
    :func:`_unique_best_structural_pairs`.  Generate the union of keys found in
    residual-bearing paragraphs, then count only those exact keys across both
    complete revisions.  This retains every productive pair and every chain
    sharing either endpoint with one.
    """
    unseen = object()
    occurrences = {}

    def add_target_keys(structural_index, paragraph_indexes):
        for paragraph_index in paragraph_indexes:
            slots = structural_index.paragraphs.get(paragraph_index)
            if not slots or size > len(slots):
                continue
            paragraph_start = slots[0].article_index
            informative_prefix = structural_index.informative_prefix
            for start in range(len(slots) - size + 1):
                article_start = paragraph_start + start
                informative = (
                    informative_prefix[article_start + size] -
                    informative_prefix[article_start]
                )
                if informative < WORD_MATCH_STRUCTURAL_MIN_ANCHOR_INFO:
                    continue
                key = structural_index._article_window_key(
                    article_start, size,
                )
                occurrences.setdefault(key, unseen)

    add_target_keys(prev_index, prev_target_paragraphs)
    add_target_keys(curr_index, curr_target_paragraphs)
    if not occurrences:
        return occurrences

    missing = object()
    for paragraph_index, slots in prev_index.paragraphs.items():
        if size > len(slots):
            continue
        paragraph_start = slots[0].article_index
        informative_prefix = prev_index.informative_prefix
        for start in range(len(slots) - size + 1):
            article_start = paragraph_start + start
            informative = (
                informative_prefix[article_start + size] -
                informative_prefix[article_start]
            )
            if informative < WORD_MATCH_STRUCTURAL_MIN_ANCHOR_INFO:
                continue
            key = prev_index._article_window_key(article_start, size)
            previous = occurrences.get(key, missing)
            if previous is missing or previous is None:
                continue
            if previous is unseen:
                occurrences[key] = (paragraph_index, start)
            else:
                occurrences[key] = None

    for paragraph_index, slots in curr_index.paragraphs.items():
        if size > len(slots):
            continue
        paragraph_start = slots[0].article_index
        informative_prefix = curr_index.informative_prefix
        for start in range(len(slots) - size + 1):
            article_start = paragraph_start + start
            informative = (
                informative_prefix[article_start + size] -
                informative_prefix[article_start]
            )
            if informative < WORD_MATCH_STRUCTURAL_MIN_ANCHOR_INFO:
                continue
            key = curr_index._article_window_key(article_start, size)
            previous = occurrences.get(key)
            if not isinstance(previous, tuple):
                continue
            if len(previous) == 2:
                occurrences[key] = previous + (paragraph_index, start)
            else:
                occurrences[key] = None
    return occurrences


def _matched_structural_anchor_occurrences(
        prev_index, curr_index, size, prev_target_paragraphs=None,
        curr_target_paragraphs=None):
    """Join exact windows while retaining one revision-wide state map.

    The previous scan records a window's sole position, or ``None`` after a
    duplicate.  The current scan expands that position on its first occurrence
    and invalidates it on a second.  Four-item values are therefore exactly the
    windows unique in both complete revisions; current-only keys are never
    retained.
    """
    if (prev_target_paragraphs is not None and
            curr_target_paragraphs is not None):
        target_tokens = sum(
            len(prev_index.paragraphs[index])
            for index in prev_target_paragraphs
        ) + sum(
            len(curr_index.paragraphs[index])
            for index in curr_target_paragraphs
        )
        # Candidate generation adds one pass over target paragraphs.  Use it
        # only when that pass covers less than half of the complete input; the
        # full single-map join has a smaller constant when nearly every
        # paragraph is already in scope.
        if target_tokens * 2 < len(prev_index.slots) + len(curr_index.slots):
            return _targeted_structural_anchor_occurrences(
                prev_index, curr_index, size, prev_target_paragraphs,
                curr_target_paragraphs,
            )

    occurrences = {}
    for paragraph_index in prev_index.paragraphs:
        _index_structural_anchor_size(
            prev_index, paragraph_index, size, occurrences,
        )

    informative_prefix = curr_index.informative_prefix
    for paragraph_index, slots in curr_index.paragraphs.items():
        paragraph_length = len(slots)
        if size > paragraph_length:
            continue
        paragraph_start = slots[0].article_index
        for start in range(paragraph_length - size + 1):
            article_start = paragraph_start + start
            informative = (
                informative_prefix[article_start + size] -
                informative_prefix[article_start]
            )
            if informative < WORD_MATCH_STRUCTURAL_MIN_ANCHOR_INFO:
                continue
            key = curr_index._article_window_key(article_start, size)
            previous = occurrences.get(key)
            if previous is None:
                continue
            if len(previous) == 2:
                occurrences[key] = previous + (paragraph_index, start)
            else:
                occurrences[key] = None
    return occurrences


def _append_merged_structural_anchor(bucket, raw_anchor):
    if (bucket and
            raw_anchor[0] <= bucket[-1][1] and
            raw_anchor[2] <= bucket[-1][3]):
        bucket[-1] = (
            bucket[-1][0], max(bucket[-1][1], raw_anchor[1]),
            bucket[-1][2], max(bucket[-1][3], raw_anchor[3]),
        )
    else:
        bucket.append(raw_anchor)


def _structural_anchor_chains(
        prev_index, curr_index, prev_target_paragraphs=None,
        curr_target_paragraphs=None):
    prev_paragraphs = prev_index.paragraphs
    curr_paragraphs = curr_index.paragraphs

    # Global uniqueness is measured over the complete revisions.  Index and
    # consume one width at a time, joining the current scan into the previous
    # state so only one revision-wide occurrence map is live.
    segments_by_diagonal = defaultdict(dict)
    for size in WORD_MATCH_STRUCTURAL_ANCHOR_SIZES:
        occurrences = _matched_structural_anchor_occurrences(
            prev_index, curr_index, size, prev_target_paragraphs,
            curr_target_paragraphs,
        )

        size_segments = defaultdict(list)
        for occurrence in occurrences.values():
            if not isinstance(occurrence, tuple) or len(occurrence) != 4:
                continue
            (prev_paragraph_index, prev_start,
             curr_paragraph_index, curr_start) = occurrence
            pair = (prev_paragraph_index, curr_paragraph_index)
            diagonal = prev_start - curr_start
            _append_merged_structural_anchor(
                size_segments[(pair, diagonal)],
                (
                    prev_start, prev_start + size,
                    curr_start, curr_start + size,
                ),
            )
        for diagonal_key, segments in size_segments.items():
            segments_by_diagonal[diagonal_key][size] = segments

    merged_by_diagonal = {}
    for diagonal_key, segments_by_size in segments_by_diagonal.items():
        bucket = []
        streams = [
            segments_by_size[size]
            for size in WORD_MATCH_STRUCTURAL_ANCHOR_SIZES
            if size in segments_by_size
        ]
        for raw_anchor in merge(*streams):
            _append_merged_structural_anchor(bucket, raw_anchor)
        merged_by_diagonal[diagonal_key] = bucket

    anchors_by_pair = defaultdict(list)
    for (pair, _), merged_anchors in merged_by_diagonal.items():
        anchors_by_pair[pair].extend(merged_anchors)

    chains = {}
    for pair, merged in anchors_by_pair.items():
        anchors = []
        for prev_start, prev_end, curr_start, curr_end in merged:
            info = prev_index.informative_count(
                prev_paragraphs[pair[0]], prev_start, prev_end,
            )
            anchors.append((prev_start, prev_end, curr_start, curr_end, info))
        anchors.sort()
        best_scores = []
        previous = []
        for index, anchor in enumerate(anchors):
            length = anchor[1] - anchor[0]
            best_score = (anchor[4], length)
            best_previous = None
            for prior_index in range(index):
                prior = anchors[prior_index]
                if prior[1] > anchor[0] or prior[3] > anchor[2]:
                    continue
                candidate_score = (
                    best_scores[prior_index][0] + anchor[4],
                    best_scores[prior_index][1] + length,
                )
                if candidate_score > best_score:
                    best_score = candidate_score
                    best_previous = prior_index
            best_scores.append(best_score)
            previous.append(best_previous)

        if not anchors:
            continue
        end_index = max(
            range(len(anchors)),
            key=lambda index: (best_scores[index], -anchors[index][0], -anchors[index][2]),
        )
        chain = []
        while end_index is not None:
            chain.append(anchors[end_index])
            end_index = previous[end_index]
        chain.reverse()
        chains[pair] = (chain, best_scores[max(
            range(len(anchors)),
            key=lambda index: (best_scores[index], -anchors[index][0], -anchors[index][2]),
        )])
    return prev_paragraphs, curr_paragraphs, chains


def _compact_index_anchor_size(document, paragraph_index, size, state):
    paragraph_start, paragraph_end = document.paragraph_range(paragraph_index)
    paragraph_length = paragraph_end - paragraph_start
    if size > paragraph_length:
        return
    informative_prefix = document.informative_prefix
    keys = document.keys
    for start in range(paragraph_length - size + 1):
        article_start = paragraph_start + start
        if (informative_prefix[article_start + size] -
                informative_prefix[article_start]) < (
                    WORD_MATCH_STRUCTURAL_MIN_ANCHOR_INFO):
            continue
        key = tuple(keys[article_start:article_start + size])
        _record_structural_anchor_occurrence(
            state, key, paragraph_index, start,
        )


def _compact_targeted_anchor_occurrences(
        prev_document, curr_document, size, prev_target_paragraphs,
        curr_target_paragraphs):
    unseen = object()
    occurrences = {}

    def add_target_keys(document, paragraph_indexes):
        for paragraph_index in paragraph_indexes:
            paragraph_range = document.paragraph_ranges.get(paragraph_index)
            if paragraph_range is None:
                continue
            paragraph_start, paragraph_end = paragraph_range
            paragraph_length = paragraph_end - paragraph_start
            if size > paragraph_length:
                continue
            informative_prefix = document.informative_prefix
            keys = document.keys
            for start in range(paragraph_length - size + 1):
                article_start = paragraph_start + start
                if (informative_prefix[article_start + size] -
                        informative_prefix[article_start]) < (
                            WORD_MATCH_STRUCTURAL_MIN_ANCHOR_INFO):
                    continue
                occurrences.setdefault(
                    tuple(keys[article_start:article_start + size]), unseen,
                )

    add_target_keys(prev_document, prev_target_paragraphs)
    add_target_keys(curr_document, curr_target_paragraphs)
    if not occurrences:
        return occurrences

    missing = object()
    informative_prefix = prev_document.informative_prefix
    keys = prev_document.keys
    for paragraph_index, paragraph_range in (
            prev_document.paragraph_ranges.items()):
        paragraph_start, paragraph_end = paragraph_range
        paragraph_length = paragraph_end - paragraph_start
        if size > paragraph_length:
            continue
        for start in range(paragraph_length - size + 1):
            article_start = paragraph_start + start
            if (informative_prefix[article_start + size] -
                    informative_prefix[article_start]) < (
                        WORD_MATCH_STRUCTURAL_MIN_ANCHOR_INFO):
                continue
            key = tuple(keys[article_start:article_start + size])
            previous = occurrences.get(key, missing)
            if previous is missing or previous is None:
                continue
            if previous is unseen:
                occurrences[key] = (paragraph_index, start)
            else:
                occurrences[key] = None

    informative_prefix = curr_document.informative_prefix
    keys = curr_document.keys
    for paragraph_index, paragraph_range in (
            curr_document.paragraph_ranges.items()):
        paragraph_start, paragraph_end = paragraph_range
        paragraph_length = paragraph_end - paragraph_start
        if size > paragraph_length:
            continue
        for start in range(paragraph_length - size + 1):
            article_start = paragraph_start + start
            if (informative_prefix[article_start + size] -
                    informative_prefix[article_start]) < (
                        WORD_MATCH_STRUCTURAL_MIN_ANCHOR_INFO):
                continue
            key = tuple(keys[article_start:article_start + size])
            previous = occurrences.get(key)
            if not isinstance(previous, tuple):
                continue
            if len(previous) == 2:
                occurrences[key] = previous + (paragraph_index, start)
            else:
                occurrences[key] = None
    return occurrences


def _compact_matched_anchor_occurrences(
        prev_document, curr_document, size, prev_target_paragraphs,
        curr_target_paragraphs):
    target_tokens = sum(
        prev_document.paragraph_length(index)
        for index in prev_target_paragraphs
    ) + sum(
        curr_document.paragraph_length(index)
        for index in curr_target_paragraphs
    )
    if target_tokens * 2 < (
            len(prev_document.values) + len(curr_document.values)):
        return _compact_targeted_anchor_occurrences(
            prev_document, curr_document, size, prev_target_paragraphs,
            curr_target_paragraphs,
        )

    occurrences = {}
    for paragraph_index in prev_document.paragraph_ranges:
        _compact_index_anchor_size(
            prev_document, paragraph_index, size, occurrences,
        )

    informative_prefix = curr_document.informative_prefix
    keys = curr_document.keys
    for paragraph_index, paragraph_range in (
            curr_document.paragraph_ranges.items()):
        paragraph_start, paragraph_end = paragraph_range
        paragraph_length = paragraph_end - paragraph_start
        if size > paragraph_length:
            continue
        for start in range(paragraph_length - size + 1):
            article_start = paragraph_start + start
            if (informative_prefix[article_start + size] -
                    informative_prefix[article_start]) < (
                        WORD_MATCH_STRUCTURAL_MIN_ANCHOR_INFO):
                continue
            key = tuple(keys[article_start:article_start + size])
            previous = occurrences.get(key)
            if previous is None:
                continue
            if len(previous) == 2:
                occurrences[key] = previous + (paragraph_index, start)
            else:
                occurrences[key] = None
    return occurrences


def _compact_target_anchor_candidates(
        prev_document, curr_document, prev_target_paragraphs,
        curr_target_paragraphs):
    candidates = {}
    unseen = object()

    def add_document(document, paragraph_indexes):
        keys = document.keys
        informative_prefix = document.informative_prefix
        for paragraph_index in paragraph_indexes:
            paragraph_range = document.paragraph_ranges.get(paragraph_index)
            if paragraph_range is None:
                continue
            paragraph_start, paragraph_end = paragraph_range
            paragraph_length = paragraph_end - paragraph_start
            for size in WORD_MATCH_STRUCTURAL_ANCHOR_SIZES:
                if size > paragraph_length:
                    continue
                for start in range(paragraph_length - size + 1):
                    article_start = paragraph_start + start
                    if (informative_prefix[article_start + size] -
                            informative_prefix[article_start]) < (
                                WORD_MATCH_STRUCTURAL_MIN_ANCHOR_INFO):
                        continue
                    candidates.setdefault(tuple(
                        keys[article_start:article_start + size]
                    ), unseen)

    add_document(prev_document, prev_target_paragraphs)
    add_document(curr_document, curr_target_paragraphs)
    return candidates, unseen


def _compact_anchor_pattern_symbol_bound(document, paragraph_indexes):
    """Return a no-allocation upper bound on candidate trie symbols."""
    return sum(
        size * max(0, document.paragraph_length(paragraph_index) - size + 1)
        for paragraph_index in paragraph_indexes
        for size in WORD_MATCH_STRUCTURAL_ANCHOR_SIZES
    )


def _build_structural_pattern_automaton(patterns):
    patterns = list(patterns)
    transitions = [{}]
    failures = [0]
    outputs = [[]]
    for pattern_index, pattern in enumerate(patterns):
        state = 0
        for token in pattern:
            next_state = transitions[state].get(token)
            if next_state is None:
                next_state = len(transitions)
                transitions[state][token] = next_state
                transitions.append({})
                failures.append(0)
                outputs.append([])
            state = next_state
        outputs[state].append(pattern_index)

    queue = list(transitions[0].values())
    queue_index = 0
    while queue_index < len(queue):
        state = queue[queue_index]
        queue_index += 1
        for token, next_state in transitions[state].items():
            queue.append(next_state)
            failure = failures[state]
            while failure and token not in transitions[failure]:
                failure = failures[failure]
            failures[next_state] = transitions[failure].get(token, 0)
            failure_outputs = outputs[failures[next_state]]
            if failure_outputs:
                outputs[next_state].extend(failure_outputs)
    return patterns, transitions, failures, outputs


def _scan_structural_pattern_automaton(document, automaton, visit):
    patterns, transitions, failures, outputs = automaton
    keys = document.keys
    for paragraph_index, paragraph_range in (
            document.paragraph_ranges.items()):
        paragraph_start, paragraph_end = paragraph_range
        state = 0
        for article_index in range(paragraph_start, paragraph_end):
            token = keys[article_index]
            while state and token not in transitions[state]:
                state = failures[state]
            state = transitions[state].get(token, 0)
            if not outputs[state]:
                continue
            local_end = article_index - paragraph_start + 1
            for pattern_index in outputs[state]:
                pattern = patterns[pattern_index]
                visit(
                    pattern, paragraph_index, local_end - len(pattern),
                )


def _compact_targeted_anchor_occurrences_all(
        prev_document, curr_document, prev_target_paragraphs,
        curr_target_paragraphs):
    occurrences, unseen = _compact_target_anchor_candidates(
        prev_document, curr_document, prev_target_paragraphs,
        curr_target_paragraphs,
    )
    if not occurrences:
        return occurrences

    def visit_previous(pattern, paragraph_index, start):
        previous = occurrences[pattern]
        if previous is unseen:
            occurrences[pattern] = (paragraph_index, start)
        elif previous is not None:
            occurrences[pattern] = None

    _scan_structural_pattern_automaton(
        prev_document,
        _build_structural_pattern_automaton(occurrences),
        visit_previous,
    )
    unique_previous = [
        pattern for pattern, occurrence in occurrences.items()
        if isinstance(occurrence, tuple) and len(occurrence) == 2
    ]
    if not unique_previous:
        return occurrences

    def visit_current(pattern, paragraph_index, start):
        previous = occurrences[pattern]
        if not isinstance(previous, tuple):
            return
        if len(previous) == 2:
            occurrences[pattern] = previous + (paragraph_index, start)
        else:
            occurrences[pattern] = None

    _scan_structural_pattern_automaton(
        curr_document,
        _build_structural_pattern_automaton(unique_previous),
        visit_current,
    )
    return occurrences


def _compact_structural_anchor_chains(
        prev_document, curr_document, prev_target_paragraphs,
        curr_target_paragraphs):
    segments_by_diagonal = defaultdict(dict)

    def add_occurrences(size, occurrences):
        size_segments = defaultdict(list)
        for occurrence in occurrences.values():
            if not isinstance(occurrence, tuple) or len(occurrence) != 4:
                continue
            (prev_paragraph_index, prev_start,
             curr_paragraph_index, curr_start) = occurrence
            pair = (prev_paragraph_index, curr_paragraph_index)
            diagonal = prev_start - curr_start
            _append_merged_structural_anchor(
                size_segments[(pair, diagonal)],
                (
                    prev_start, prev_start + size,
                    curr_start, curr_start + size,
                ),
            )
        for diagonal_key, segments in size_segments.items():
            segments_by_diagonal[diagonal_key][size] = segments

    target_tokens = sum(
        prev_document.paragraph_length(index)
        for index in prev_target_paragraphs
    ) + sum(
        curr_document.paragraph_length(index)
        for index in curr_target_paragraphs
    )
    complete_tokens = (
        len(prev_document.values) + len(curr_document.values)
    )
    use_targeted = target_tokens * 2 < complete_tokens
    if _structural_native is not None:
        native_occurrences = _structural_native.unique_anchor_occurrences(
            prev_document.keys, curr_document.keys,
            prev_document.paragraph_ranges, curr_document.paragraph_ranges,
            prev_document.informative_prefix,
            curr_document.informative_prefix,
            prev_target_paragraphs, curr_target_paragraphs,
            WORD_MATCH_STRUCTURAL_ANCHOR_SIZES,
            WORD_MATCH_STRUCTURAL_MIN_ANCHOR_INFO,
            not use_targeted,
        )
        merged_size = WORD_MATCH_STRUCTURAL_ANCHOR_SIZES[0]
        for (prev_paragraph_index, prev_start, prev_end,
             curr_paragraph_index, curr_start,
             curr_end) in native_occurrences:
            pair = (prev_paragraph_index, curr_paragraph_index)
            diagonal = prev_start - curr_start
            segments_by_diagonal[(pair, diagonal)].setdefault(
                merged_size, [],
            ).append((prev_start, prev_end, curr_start, curr_end))

    use_automaton = (
        use_targeted and complete_tokens >=
        WORD_MATCH_STRUCTURAL_ANCHOR_AUTOMATON_MIN_TOKENS and
        _compact_anchor_pattern_symbol_bound(
            prev_document, prev_target_paragraphs,
        ) + _compact_anchor_pattern_symbol_bound(
            curr_document, curr_target_paragraphs,
        ) <= WORD_MATCH_STRUCTURAL_ANCHOR_AUTOMATON_MAX_PATTERN_SYMBOLS
    )
    if _structural_native is not None:
        pass
    elif use_automaton:
        for pattern, occurrence in (
                _compact_targeted_anchor_occurrences_all(
                    prev_document, curr_document,
                    prev_target_paragraphs, curr_target_paragraphs,
                ).items()):
            if not isinstance(occurrence, tuple) or len(occurrence) != 4:
                continue
            size = len(pattern)
            (prev_paragraph_index, prev_start,
             curr_paragraph_index, curr_start) = occurrence
            pair = (prev_paragraph_index, curr_paragraph_index)
            diagonal = prev_start - curr_start
            bucket = segments_by_diagonal[(pair, diagonal)].setdefault(
                size, [],
            )
            _append_merged_structural_anchor(
                bucket,
                (
                    prev_start, prev_start + size,
                    curr_start, curr_start + size,
                ),
            )
    else:
        # Retain only one revision-wide occurrence map at a time.  Keeping all
        # four maps alive defeats a material part of the compact-document
        # memory saving on histories below the automaton crossover.
        for size in WORD_MATCH_STRUCTURAL_ANCHOR_SIZES:
            add_occurrences(
                size, _compact_matched_anchor_occurrences(
                    prev_document, curr_document, size,
                    prev_target_paragraphs, curr_target_paragraphs,
                ),
            )

    merged_by_diagonal = {}
    for diagonal_key, segments_by_size in segments_by_diagonal.items():
        bucket = []
        streams = [
            segments_by_size[size]
            for size in WORD_MATCH_STRUCTURAL_ANCHOR_SIZES
            if size in segments_by_size
        ]
        for raw_anchor in merge(*streams):
            _append_merged_structural_anchor(bucket, raw_anchor)
        merged_by_diagonal[diagonal_key] = bucket

    anchors_by_pair = defaultdict(list)
    for (pair, _), merged_anchors in merged_by_diagonal.items():
        anchors_by_pair[pair].extend(merged_anchors)

    chains = {}
    for pair, merged_anchors in anchors_by_pair.items():
        anchors = []
        for prev_start, prev_end, curr_start, curr_end in merged_anchors:
            info = prev_document.informative_count(
                pair[0], prev_start, prev_end,
            )
            anchors.append((
                prev_start, prev_end, curr_start, curr_end, info,
            ))
        anchors.sort()
        best_scores = []
        previous = []
        for index, anchor in enumerate(anchors):
            length = anchor[1] - anchor[0]
            best_score = (anchor[4], length)
            best_previous = None
            for prior_index in range(index):
                prior = anchors[prior_index]
                if prior[1] > anchor[0] or prior[3] > anchor[2]:
                    continue
                candidate_score = (
                    best_scores[prior_index][0] + anchor[4],
                    best_scores[prior_index][1] + length,
                )
                if candidate_score > best_score:
                    best_score = candidate_score
                    best_previous = prior_index
            best_scores.append(best_score)
            previous.append(best_previous)

        if not anchors:
            continue
        end_index = max(
            range(len(anchors)),
            key=lambda index: (
                best_scores[index], -anchors[index][0], -anchors[index][2],
            ),
        )
        winning_index = end_index
        chain = []
        while end_index is not None:
            chain.append(anchors[end_index])
            end_index = previous[end_index]
        chain.reverse()
        chains[pair] = (chain, best_scores[winning_index])
    return chains


def _unique_best_structural_pairs(chains):
    by_prev = defaultdict(list)
    by_curr = defaultdict(list)
    for pair, (_, score) in chains.items():
        by_prev[pair[0]].append((score, pair))
        by_curr[pair[1]].append((score, pair))

    def unique_best(entries):
        ordered = sorted(entries, reverse=True)
        if not ordered:
            return None
        if len(ordered) > 1 and ordered[0][0] == ordered[1][0]:
            return None
        # Any independently certifiable secondary chain means this paragraph
        # participates in a split/merge.  Relative dominance is insufficient:
        # a large merged paragraph can contain one source region with twice the
        # support of another while both correspondences are still real.  Such
        # a paragraph must not receive virtual outer anchors.  Its explicit
        # unique intervals remain available to the ordinary moved matcher.
        if (len(ordered) > 1 and
                ordered[1][0][0] >= WORD_MATCH_STRUCTURAL_MIN_PAIR_INFO):
            return None
        return ordered[0][1]

    best_for_prev = dict((index, unique_best(entries))
                         for index, entries in by_prev.items())
    best_for_curr = dict((index, unique_best(entries))
                         for index, entries in by_curr.items())
    return set(
        pair for pair in chains
        if best_for_prev.get(pair[0]) == pair and best_for_curr.get(pair[1]) == pair
    )


def _structural_window_keys(slots, structural_index):
    windows = set()
    for size in WORD_MATCH_STRUCTURAL_AMBIGUITY_SIZES:
        if size > len(slots):
            continue
        for start in range(len(slots) - size + 1):
            end = start + size
            if structural_index.informative_count(slots, start, end) < 3:
                continue
            windows.add(structural_index.window_key(slots, start, end))
    return windows


def _ambiguous_windows_in_both(prev_slots, curr_slots, prev_index, curr_index,
                               ambiguous_windows):
    prev_windows = _structural_window_keys(prev_slots, prev_index)
    curr_windows = _structural_window_keys(curr_slots, curr_index)
    return bool(prev_windows.intersection(curr_windows, ambiguous_windows))


def _residual_structural_window_keys(
        slots, structural_index, residual_by_path, allowed_windows=None):
    """Return qualifying windows wholly inside maximal residual-only spans."""
    windows = set()
    run = []

    def add_run():
        if not run:
            return
        candidates = _structural_window_keys(run, structural_index)
        if allowed_windows is not None:
            candidates.intersection_update(allowed_windows)
        windows.update(candidates)

    for slot in slots:
        if slot.path in residual_by_path:
            run.append(slot)
        else:
            add_run()
            run = []
    add_run()
    return windows


def _duplicated_structural_candidate_windows(structural_index, candidates):
    """Count only structural windows that a certified gap can consume.

    Anchor discovery must retain a complete revision-wide index.  Ambiguity
    evidence has a narrower consumer: only a key present in both sides of a
    certified gap can admit an LCS run.  Scan the complete revision for those
    exact keys, saturating each count at two, instead of retaining occurrence
    state for every three- through six-token window in the revision.
    """
    remaining_by_size = {}
    signatures_by_size = {}
    for candidate in candidates:
        size = len(candidate)
        remaining_by_size.setdefault(size, set()).add(candidate)
        signatures = signatures_by_size.setdefault(size, {})
        by_middle = signatures.setdefault(candidate[0], {})
        by_last = by_middle.setdefault(candidate[size // 2], set())
        by_last.add(candidate[-1])
    candidate_count = sum(len(values) for values in remaining_by_size.values())
    if not candidate_count:
        return set()

    keys = structural_index.keys
    seen_once = set()
    duplicated = set()
    for slots in structural_index.paragraphs.values():
        paragraph_length = len(slots)
        if not paragraph_length:
            continue
        paragraph_start = slots[0].article_index
        for size in WORD_MATCH_STRUCTURAL_AMBIGUITY_SIZES:
            remaining = remaining_by_size.get(size)
            if not remaining or size > paragraph_length:
                continue
            signatures = signatures_by_size[size]
            for offset in range(paragraph_length - size + 1):
                start = paragraph_start + offset
                by_middle = signatures.get(keys[start])
                if by_middle is None:
                    continue
                by_last = by_middle.get(keys[start + size // 2])
                if by_last is None or keys[start + size - 1] not in by_last:
                    continue
                key = structural_index._article_window_key(start, size)
                if key not in remaining:
                    continue
                if key in seen_once:
                    duplicated.add(key)
                    remaining.remove(key)
                    if len(duplicated) == candidate_count:
                        return duplicated
                else:
                    seen_once.add(key)
    return duplicated


def _structural_gap_ranges(chain, prev_length, curr_length):
    gaps = [(0, chain[0][0], 0, chain[0][2])]
    for left, right in zip(chain, chain[1:]):
        gaps.append((left[1], right[0], left[3], right[2]))
    gaps.append((chain[-1][1], prev_length,
                 chain[-1][3], curr_length))
    return gaps


def _compact_available_residual_windows(
        document, residual_by_article, is_available,
        native_availability=None, native_previous_mode=False):
    """Return exact raw-value windows available inside paragraph boundaries."""
    if _structural_native is not None and native_availability is not None:
        return _structural_native.available_residual_windows(
            document.values, document.paragraph_ranges,
            residual_by_article, native_availability,
            native_previous_mode, WORD_MATCH_MOVE_STRUCTURAL_TOKENS,
            WORD_MATCH_STRUCTURAL_AMBIGUITY_SIZES, 3,
        )
    windows = set()

    def add_run(start, end):
        length = end - start
        for size in WORD_MATCH_STRUCTURAL_AMBIGUITY_SIZES:
            if size > length:
                continue
            for article_start in range(start, end - size + 1):
                values = document.values[article_start:article_start + size]
                if sum(_is_informative_move_token(value)
                       for value in values) >= 3:
                    windows.add(tuple(values))

    for paragraph_start, paragraph_end in document.paragraph_ranges.values():
        run_start = None
        for article_index in range(paragraph_start, paragraph_end):
            residual = residual_by_article.get(article_index)
            available = (
                residual is not None and is_available(residual[0])
            )
            if available:
                if run_start is None:
                    run_start = article_index
            elif run_start is not None:
                add_run(run_start, article_index)
                run_start = None
        if run_start is not None:
            add_run(run_start, paragraph_end)
    return windows


def _compact_residual_structural_window_keys(
        document, paragraph_index, start, end, residual_by_article,
        allowed_windows=None, residual_flags=None):
    windows = set()
    paragraph_start, _ = document.paragraph_range(paragraph_index)
    if _structural_native is not None and residual_flags is not None:
        return _structural_native.residual_structural_windows(
            document.keys, document.informative_prefix, residual_flags,
            paragraph_start + start, paragraph_start + end,
            allowed_windows, WORD_MATCH_STRUCTURAL_AMBIGUITY_SIZES, 3,
        )
    run_start = None

    def add_run(article_start, article_end):
        length = article_end - article_start
        for size in WORD_MATCH_STRUCTURAL_AMBIGUITY_SIZES:
            if size > length:
                continue
            for window_start in range(
                    article_start, article_end - size + 1):
                if (document.informative_prefix[window_start + size] -
                        document.informative_prefix[window_start]) < 3:
                    continue
                key = tuple(
                    document.keys[window_start:window_start + size]
                )
                if allowed_windows is None or key in allowed_windows:
                    windows.add(key)

    for article_index in range(paragraph_start + start,
                               paragraph_start + end):
        if article_index in residual_by_article:
            if run_start is None:
                run_start = article_index
        elif run_start is not None:
            add_run(run_start, article_index)
            run_start = None
    if run_start is not None:
        add_run(run_start, paragraph_start + end)
    return windows


def _compact_duplicated_structural_candidate_windows(document, candidates):
    if _structural_native is not None:
        return _structural_native.duplicated_candidate_windows(
            document.keys, document.paragraph_ranges, set(candidates),
        )
    remaining_by_size = {}
    signatures_by_size = {}
    for candidate in candidates:
        size = len(candidate)
        remaining_by_size.setdefault(size, set()).add(candidate)
        signatures = signatures_by_size.setdefault(size, {})
        by_middle = signatures.setdefault(candidate[0], {})
        by_last = by_middle.setdefault(candidate[size // 2], set())
        by_last.add(candidate[-1])
    candidate_count = sum(len(values) for values in remaining_by_size.values())
    if not candidate_count:
        return set()

    keys = document.keys
    seen_once = set()
    duplicated = set()
    for paragraph_start, paragraph_end in document.paragraph_ranges.values():
        paragraph_length = paragraph_end - paragraph_start
        for size in WORD_MATCH_STRUCTURAL_AMBIGUITY_SIZES:
            remaining = remaining_by_size.get(size)
            if not remaining or size > paragraph_length:
                continue
            signatures = signatures_by_size[size]
            for start in range(paragraph_start, paragraph_end - size + 1):
                by_middle = signatures.get(keys[start])
                if by_middle is None:
                    continue
                by_last = by_middle.get(keys[start + size // 2])
                if by_last is None or keys[start + size - 1] not in by_last:
                    continue
                key = tuple(keys[start:start + size])
                if key not in remaining:
                    continue
                if key in seen_once:
                    duplicated.add(key)
                    remaining.remove(key)
                    if len(duplicated) == candidate_count:
                        return duplicated
                else:
                    seen_once.add(key)
    return duplicated


def _compact_structural_run_has_boundary_context(
        prev_gap_values, curr_gap_values, run):
    informative_offsets = [
        offset for offset, (prev_index, curr_index) in enumerate(run)
        if (_is_informative_move_token(prev_gap_values[prev_index]) and
            _is_informative_move_token(curr_gap_values[curr_index]))
    ]
    if len(informative_offsets) < WORD_MATCH_STRUCTURAL_MIN_RUN_INFO:
        return False
    has_informative_support = (
        len(informative_offsets) >= WORD_MATCH_STRUCTURAL_MIN_ANCHOR_INFO
    )
    if not has_informative_support:
        first = informative_offsets[0]
        last = informative_offsets[-1]
        has_informative_support = first > 0 and last < len(run) - 1
    if not has_informative_support:
        return False

    prev_values = [prev_gap_values[prev_index] for prev_index, _ in run]
    values = [curr_gap_values[curr_index] for _, curr_index in run]

    def cuts_template_field_tail(gap, indices, run_values):
        if (not run_values or run_values[0] != '|' or
                '=' not in run_values[1:]):
            return False
        last_index = indices[-1]
        if last_index + 1 >= len(gap):
            return False
        return gap[last_index + 1] not in ('|', '}}')

    if (cuts_template_field_tail(
            prev_gap_values, [prev_index for prev_index, _ in run],
            prev_values) or
            cuts_template_field_tail(
                curr_gap_values, [curr_index for _, curr_index in run],
                values)):
        return False

    leading_equals = 0
    for value in values:
        if value != '=':
            break
        leading_equals += 1
    trailing_equals = 0
    for value in reversed(values):
        if value != '=':
            break
        trailing_equals += 1
    has_open_heading = leading_equals >= 2
    has_close_heading = trailing_equals >= 2
    if (has_open_heading != has_close_heading or
            (has_open_heading and leading_equals != trailing_equals)):
        return False

    construct_pairs = {'{{': '}}', '[[': ']]', '<': '>'}
    closing_tokens = dict(
        (close, open_) for open_, close in construct_pairs.items()
    )
    stack = []
    for value in values:
        if value in construct_pairs:
            stack.append(value)
        elif value in closing_tokens:
            if not stack or stack[-1] != closing_tokens[value]:
                return False
            stack.pop()
    return not stack


def _compact_run_has_ambiguous_window(
        prev_document, curr_document, prev_article_start, curr_article_start,
        run, ambiguous_windows):
    prev_windows = set()
    for size in WORD_MATCH_STRUCTURAL_AMBIGUITY_SIZES:
        if size > len(run):
            continue
        for offset in range(len(run) - size + 1):
            prev_start = prev_article_start + run[offset][0]
            prev_end = prev_start + size
            if (prev_document.informative_prefix[prev_end] -
                    prev_document.informative_prefix[prev_start]) < 3:
                continue
            prev_windows.add(tuple(
                prev_document.keys[prev_start:prev_end]
            ))
    if not prev_windows:
        return False
    candidates = prev_windows.intersection(ambiguous_windows)
    if not candidates:
        return False
    for size in WORD_MATCH_STRUCTURAL_AMBIGUITY_SIZES:
        if size > len(run):
            continue
        for offset in range(len(run) - size + 1):
            curr_start = curr_article_start + run[offset][1]
            curr_end = curr_start + size
            if (curr_document.informative_prefix[curr_end] -
                    curr_document.informative_prefix[curr_start]) < 3:
                continue
            if tuple(curr_document.keys[curr_start:curr_end]) in candidates:
                return True
    return False


def _lcs_token_pairs(prev_keys, curr_keys, prev_values=None, curr_values=None):
    """Return the lexicographically best order-preserving alignment.

    The score follows the structural-matching contract, in order: maximize
    matches, minimize disconnected runs, maximize informative matches, and
    minimize normalized displacement.  Exact score ties retain the earlier
    alignment, which implements the declared left-to-right duplicate rule.
    """
    prev_len = len(prev_keys)
    curr_len = len(curr_keys)
    if not prev_len or not curr_len:
        return []
    if prev_len * curr_len > WORD_MATCH_STRUCTURAL_MAX_GAP_CELLS:
        return []
    if prev_values is None:
        prev_values = prev_keys
    if curr_values is None:
        curr_values = curr_keys
    if (len(prev_values) != prev_len or len(curr_values) != curr_len):
        raise ValueError("alignment values must match alignment keys")
    if (_structural_native is not None and
            isinstance(prev_keys, list) and isinstance(curr_keys, list) and
            isinstance(prev_values, list) and isinstance(curr_values, list)):
        return _structural_native.lcs_token_pairs(
            prev_keys, curr_keys, prev_values, curr_values,
            WORD_MATCH_MOVE_STRUCTURAL_TOKENS,
            WORD_MATCH_STRUCTURAL_MAX_GAP_CELLS,
        )

    # ``matched`` contains alignments whose final operation paired the last
    # tokens.  ``gapped`` contains alignments whose final operation skipped at
    # least one token.  Keeping those states distinct lets a contiguous match
    # extend the current run without paying for a new disconnected run.
    matched = [[None] * (curr_len + 1) for _ in range(prev_len + 1)]
    gapped = [[None] * (curr_len + 1) for _ in range(prev_len + 1)]
    matched_back = [[None] * (curr_len + 1) for _ in range(prev_len + 1)]
    gapped_back = [[None] * (curr_len + 1) for _ in range(prev_len + 1)]
    gapped[0][0] = (0, 0, 0, 0)

    def best_state(prev_index, curr_index):
        matched_score = matched[prev_index][curr_index]
        gapped_score = gapped[prev_index][curr_index]
        if matched_score is None:
            return gapped_score, 'gapped'
        if gapped_score is None or matched_score > gapped_score:
            return matched_score, 'matched'
        # On an exact tie the gapped state already contains an earlier match,
        # whereas the matched state ends at the latest possible occurrence.
        return gapped_score, 'gapped'

    for prev_count in range(prev_len + 1):
        for curr_count in range(curr_len + 1):
            if prev_count or curr_count:
                skip_candidates = []
                if prev_count:
                    score, state = best_state(prev_count - 1, curr_count)
                    if score is not None:
                        # Prefer this top transition on a score tie: it keeps
                        # an already-selected earlier previous occurrence.
                        skip_candidates.append((score, 1, 'prev', state))
                if curr_count:
                    score, state = best_state(prev_count, curr_count - 1)
                    if score is not None:
                        skip_candidates.append((score, 0, 'curr', state))
                if skip_candidates:
                    score, _, direction, state = max(skip_candidates)
                    gapped[prev_count][curr_count] = score
                    gapped_back[prev_count][curr_count] = (direction, state)

            if (not prev_count or not curr_count or
                    prev_keys[prev_count - 1] != curr_keys[curr_count - 1]):
                continue
            prev_index = prev_count - 1
            curr_index = curr_count - 1
            informative = int(
                _is_informative_move_token(prev_values[prev_index]) and
                _is_informative_move_token(curr_values[curr_index])
            )
            # Compare relative positions without floating point.  The common
            # denominator is irrelevant because every candidate in this gap
            # uses the same two lengths.
            displacement = abs(
                prev_index * max(curr_len - 1, 1) -
                curr_index * max(prev_len - 1, 1)
            )
            match_candidates = []
            prior = matched[prev_count - 1][curr_count - 1]
            if prior is not None:
                match_candidates.append((
                    (prior[0] + 1, prior[1], prior[2] + informative,
                     prior[3] - displacement),
                    1,
                    'matched',
                ))
            prior = gapped[prev_count - 1][curr_count - 1]
            if prior is not None:
                match_candidates.append((
                    (prior[0] + 1, prior[1] - 1,
                     prior[2] + informative, prior[3] - displacement),
                    0,
                    'gapped',
                ))
            if match_candidates:
                score, _, state = max(match_candidates)
                matched[prev_count][curr_count] = score
                matched_back[prev_count][curr_count] = state

    _, state = best_state(prev_len, curr_len)
    pairs = []
    prev_count = prev_len
    curr_count = curr_len
    while prev_count or curr_count:
        if state == 'matched':
            pairs.append((prev_count - 1, curr_count - 1))
            state = matched_back[prev_count][curr_count]
            prev_count -= 1
            curr_count -= 1
            continue
        back = gapped_back[prev_count][curr_count]
        if back is None:
            break
        direction, state = back
        if direction == 'prev':
            prev_count -= 1
        else:
            curr_count -= 1
    pairs.reverse()
    return pairs


def _contiguous_pair_runs(pairs):
    if not pairs:
        return []
    runs = []
    run = [pairs[0]]
    for pair in pairs[1:]:
        if pair[0] == run[-1][0] + 1 and pair[1] == run[-1][1] + 1:
            run.append(pair)
        else:
            runs.append(run)
            run = [pair]
    runs.append(run)
    return runs


def _structural_run_has_boundary_context(prev_gap, curr_gap, run):
    """Require context around the minimum-size informative core.

    Three informative tokens alone are common enough to be a prefix of an
    edited phrase.  They are accepted only when the same contiguous run also
    includes matched low-information boundary tokens on both sides.  Runs
    with four or more informative tokens already meet the ordinary anchor
    standard.
    """
    informative_offsets = [
        offset for offset, (prev_index, curr_index) in enumerate(run)
        if (_is_informative_move_token(prev_gap[prev_index].value) and
            _is_informative_move_token(curr_gap[curr_index].value))
    ]
    if len(informative_offsets) < WORD_MATCH_STRUCTURAL_MIN_RUN_INFO:
        return False
    has_informative_support = (
        len(informative_offsets) >= WORD_MATCH_STRUCTURAL_MIN_ANCHOR_INFO
    )
    if not has_informative_support:
        first = informative_offsets[0]
        last = informative_offsets[-1]
        has_informative_support = first > 0 and last < len(run) - 1
    if not has_informative_support:
        return False

    # Do not promote a fragment which cuts through a markup construct.  Its
    # informative words may repeat, but the candidate does not represent a
    # complete syntactic occurrence and therefore cannot carry run-level
    # structural evidence.
    prev_values = [prev_gap[prev_index].value for prev_index, _ in run]
    values = [curr_gap[curr_index].value for _, curr_index in run]

    def cuts_template_field_tail(gap, indices, run_values):
        # A run beginning with ``| field =`` claims structural evidence for a
        # template-field occurrence.  If matching stops in the middle of its
        # value, the run is merely a common prefix of an edited field (for
        # example, a date or URL) and must remain lower-tier evidence.
        if (not run_values or run_values[0] != '|' or
                '=' not in run_values[1:]):
            return False
        last_index = indices[-1]
        if last_index + 1 >= len(gap):
            return False
        return gap[last_index + 1].value not in ('|', '}}')

    if (cuts_template_field_tail(
            prev_gap, [prev_index for prev_index, _ in run], prev_values) or
            cuts_template_field_tail(
                curr_gap, [curr_index for _, curr_index in run], values)):
        return False

    leading_equals = 0
    for value in values:
        if value != '=':
            break
        leading_equals += 1
    trailing_equals = 0
    for value in reversed(values):
        if value != '=':
            break
        trailing_equals += 1
    has_open_heading = leading_equals >= 2
    has_close_heading = trailing_equals >= 2
    if (has_open_heading != has_close_heading or
            (has_open_heading and leading_equals != trailing_equals)):
        return False

    construct_pairs = {'{{': '}}', '[[': ']]', '<': '>'}
    closing_tokens = dict((close, open_) for open_, close in construct_pairs.items())
    stack = []
    for value in values:
        if value in construct_pairs:
            stack.append(value)
        elif value in closing_tokens:
            if not stack or stack[-1] != closing_tokens[value]:
                return False
            stack.pop()
    return not stack


def _available_structural_windows(slots, is_available):
    windows = set()
    run = []
    for slot in slots:
        if (is_available(slot) and
                (not run or
                 (slot.paragraph_index == run[-1].paragraph_index and
                  slot.article_index == run[-1].article_index + 1))):
            run.append(slot)
        else:
            if run:
                for size in WORD_MATCH_STRUCTURAL_AMBIGUITY_SIZES:
                    for start in range(max(0, len(run) - size + 1)):
                        window = run[start:start + size]
                        if sum(_is_informative_move_token(item.value)
                               for item in window) >= 3:
                            windows.add(tuple(item.value for item in window))
            run = [slot] if is_available(slot) else []
    if run:
        for size in WORD_MATCH_STRUCTURAL_AMBIGUITY_SIZES:
            for start in range(max(0, len(run) - size + 1)):
                window = run[start:start + size]
                if sum(_is_informative_move_token(item.value)
                       for item in window) >= 3:
                    windows.add(tuple(item.value for item in window))
    return windows


def _available_residual_windows(values, is_available):
    """Return a conservative superset of available structural windows.

    Residual word arrays preserve article order but omit exact sentence and
    paragraph matches.  Consequently every paragraph-local structural window
    is present here, while a residual window may additionally cross an omitted
    structural boundary.  The latter only causes lazy context construction; it
    can never suppress structural matching.
    """
    windows = set()
    run = []

    def add_run_windows():
        for size in WORD_MATCH_STRUCTURAL_AMBIGUITY_SIZES:
            if size > len(run):
                continue
            for start in range(len(run) - size + 1):
                window = run[start:start + size]
                if sum(_is_informative_move_token(value)
                       for value in window) >= 3:
                    windows.add(tuple(window))

    for index, value in enumerate(values):
        if is_available(index):
            run.append(value)
        else:
            if run:
                add_run_windows()
            run = []
    if run:
        add_run_windows()
    return windows


def _has_shared_available_triplet(
        text_prev, text_curr, prev_is_available, curr_is_available):
    """Return whether the available residuals share any exact triplet.

    Every structural ambiguity window is at least three tokens long.  Its
    first three tokens therefore form a shared available triplet.  A negative
    result is a sound early exit; positives still use the full exact gate.
    """
    previous_triplets = set()
    run_length = 0
    for index, value in enumerate(text_prev):
        if prev_is_available(index):
            run_length += 1
            if run_length >= 3:
                previous_triplets.add((
                    text_prev[index - 2], text_prev[index - 1], value,
                ))
        else:
            run_length = 0
    if not previous_triplets:
        return False

    run_length = 0
    for index, value in enumerate(text_curr):
        if curr_is_available(index):
            run_length += 1
            if (run_length >= 3 and
                    (text_curr[index - 2], text_curr[index - 1], value)
                    in previous_triplets):
                return True
        else:
            run_length = 0
    return False


def _unresolved_residual_windows(ledger, text_prev, text_curr):
    """Return the conservative residual intersection used by the lazy gate."""
    if _structural_native is not None:
        return _structural_native.unresolved_residual_windows(
            text_prev, text_curr, ledger.prev_used_by,
            ledger.prev_for_curr, WORD_MATCH_MOVE_STRUCTURAL_TOKENS,
            WORD_MATCH_STRUCTURAL_AMBIGUITY_SIZES, 3,
        )
    prev_is_available = lambda index: index not in ledger.prev_used_by
    curr_is_available = lambda index: ledger.prev_for_curr[index] is None
    if not _has_shared_available_triplet(
            text_prev, text_curr, prev_is_available, curr_is_available):
        return set()
    prev_windows = _available_residual_windows(
        text_prev, prev_is_available,
    )
    if not prev_windows:
        return set()
    curr_windows = _available_residual_windows(
        text_curr, curr_is_available,
    )
    return prev_windows.intersection(curr_windows)


def _candidate_windows_by_size(candidates):
    remaining_by_size = {}
    for candidate in candidates:
        size = len(candidate)
        if (size not in WORD_MATCH_STRUCTURAL_AMBIGUITY_SIZES or
                sum(_is_informative_move_token(value)
                    for value in candidate) < 3):
            continue
        remaining_by_size.setdefault(size, set()).add(candidate)
    return remaining_by_size


def _candidate_window_signatures(remaining_by_size):
    signatures_by_size = {}
    for size, candidates in remaining_by_size.items():
        signatures = signatures_by_size.setdefault(size, {})
        for candidate in candidates:
            by_middle = signatures.setdefault(candidate[0], {})
            by_last = by_middle.setdefault(candidate[size // 2], set())
            by_last.add(candidate[-1])
    return signatures_by_size


def _candidate_window_search_state(candidates):
    remaining_by_size = _candidate_windows_by_size(candidates)
    signatures_by_size = _candidate_window_signatures(remaining_by_size)
    candidate_count = sum(
        len(values) for values in remaining_by_size.values()
    )
    return remaining_by_size, signatures_by_size, candidate_count


def _duplicated_candidate_windows(slots, candidates):
    """Return candidate windows occurring at least twice within paragraphs.

    ``candidates`` is normally the small residual intersection found by the
    exact structural gate.  Occurrence state is capped at two because callers
    only distinguish unique from duplicated windows.  Revision-order slots
    make paragraph boundaries contiguous, so no grouping lists or counters for
    unrelated windows are needed.
    """
    (remaining_by_size, signatures_by_size,
     candidate_count) = _candidate_window_search_state(candidates)
    if not candidate_count:
        return set()

    seen_once = set()
    duplicated = set()
    paragraph_start = 0
    while paragraph_start < len(slots):
        paragraph_index = slots[paragraph_start].paragraph_index
        paragraph_end = paragraph_start + 1
        while (paragraph_end < len(slots) and
               slots[paragraph_end].paragraph_index == paragraph_index):
            paragraph_end += 1

        paragraph_length = paragraph_end - paragraph_start
        for size in sorted(remaining_by_size, reverse=True):
            remaining = remaining_by_size[size]
            if not remaining or size > paragraph_length:
                continue
            for start in range(paragraph_start, paragraph_end - size + 1):
                signatures = signatures_by_size[size]
                by_middle = signatures.get(slots[start].value)
                if by_middle is None:
                    continue
                by_last = by_middle.get(slots[start + size // 2].value)
                if (by_last is None or
                        slots[start + size - 1].value not in by_last):
                    continue
                window = tuple(
                    slots[index].value for index in range(start, start + size)
                )
                if window not in remaining:
                    continue
                if window in seen_once:
                    duplicated.add(window)
                    remaining.remove(window)
                    if len(duplicated) == candidate_count:
                        return duplicated
                else:
                    seen_once.add(window)
        paragraph_start = paragraph_end
    return duplicated


def _duplicated_candidate_windows_in_revision(revision, candidates):
    """Count candidate windows directly in an existing revision hierarchy.

    ``None`` means that the hierarchy cannot reproduce a complete token stream
    and the caller must retain the slot/tokenizer fallback.  Otherwise the
    result has the same paragraph-local, count-saturated semantics as
    :func:`_duplicated_candidate_windows` without allocating one slot object
    per full-revision token.
    """
    (remaining_by_size, signatures_by_size,
     candidate_count) = _candidate_window_search_state(candidates)
    if not candidate_count:
        return set()

    seen_once = set()
    duplicated = set()
    for _, paragraph in _ordered_paragraph_occurrences(revision):
        values = []
        for _, sentence in _ordered_sentence_occurrences(paragraph):
            if sentence.words:
                sentence_values = [word.value for word in sentence.words]
                if (sentence.splitted and
                        list(sentence.splitted) != sentence_values):
                    return None
            elif sentence.splitted:
                sentence_values = sentence.splitted
            else:
                return None
            values.extend(sentence_values)

        paragraph_length = len(values)
        for size in sorted(remaining_by_size, reverse=True):
            remaining = remaining_by_size[size]
            if not remaining or size > paragraph_length:
                continue
            signatures = signatures_by_size[size]
            for start in range(paragraph_length - size + 1):
                by_middle = signatures.get(values[start])
                if by_middle is None:
                    continue
                by_last = by_middle.get(values[start + size // 2])
                if (by_last is None or
                        values[start + size - 1] not in by_last):
                    continue
                window = tuple(values[start:start + size])
                if window not in remaining:
                    continue
                if window in seen_once:
                    duplicated.add(window)
                    remaining.remove(window)
                    if len(duplicated) == candidate_count:
                        return duplicated
                else:
                    seen_once.add(window)
    return duplicated


def _duplicated_candidate_windows_in_document(document, candidates):
    """Count exact candidate occurrences with one multi-width scan."""
    if _structural_native is not None:
        return _structural_native.duplicated_candidate_windows(
            document.values, document.paragraph_ranges, set(candidates),
        )
    remaining_by_size = _candidate_windows_by_size(candidates)
    candidate_count = sum(
        len(values) for values in remaining_by_size.values()
    )
    if not candidate_count:
        return set()

    values = document.values
    pattern_symbols = sum(
        size * len(patterns)
        for size, patterns in remaining_by_size.items()
    )
    use_automaton = (
        len(remaining_by_size) > 1 and
        len(values) >=
        WORD_MATCH_STRUCTURAL_DUPLICATE_AUTOMATON_MIN_TOKENS and
        pattern_symbols <=
        WORD_MATCH_STRUCTURAL_DUPLICATE_AUTOMATON_MAX_PATTERN_SYMBOLS
    )
    if not use_automaton:
        signatures_by_size = _candidate_window_signatures(
            remaining_by_size,
        )
        seen_once = set()
        duplicated = set()
        for paragraph_start, paragraph_end in (
                document.paragraph_ranges.values()):
            paragraph_length = paragraph_end - paragraph_start
            for size in sorted(remaining_by_size, reverse=True):
                remaining = remaining_by_size[size]
                if not remaining or size > paragraph_length:
                    continue
                signatures = signatures_by_size[size]
                for start in range(
                        paragraph_start, paragraph_end - size + 1):
                    by_middle = signatures.get(values[start])
                    if by_middle is None:
                        continue
                    by_last = by_middle.get(values[start + size // 2])
                    if (by_last is None or
                            values[start + size - 1] not in by_last):
                        continue
                    window = tuple(values[start:start + size])
                    if window not in remaining:
                        continue
                    if window in seen_once:
                        duplicated.add(window)
                        remaining.remove(window)
                        if len(duplicated) == candidate_count:
                            return duplicated
                    else:
                        seen_once.add(window)
        return duplicated

    patterns = [
        candidate
        for size in WORD_MATCH_STRUCTURAL_AMBIGUITY_SIZES
        for candidate in remaining_by_size.get(size, ())
    ]
    automaton = _build_structural_pattern_automaton(patterns)
    patterns, transitions, failures, outputs = automaton
    seen_once = set()
    duplicated = set()
    for paragraph_start, paragraph_end in document.paragraph_ranges.values():
        state = 0
        for article_index in range(paragraph_start, paragraph_end):
            token = values[article_index]
            while state and token not in transitions[state]:
                state = failures[state]
            state = transitions[state].get(token, 0)
            for pattern_index in outputs[state]:
                pattern = patterns[pattern_index]
                if pattern in duplicated:
                    continue
                if pattern in seen_once:
                    duplicated.add(pattern)
                    if len(duplicated) == candidate_count:
                        return duplicated
                else:
                    seen_once.add(pattern)
    return duplicated


def _has_unresolved_duplicate_window(ledger, prev_slots, curr_slots,
                                     full_prev_slots, full_curr_slots,
                                     duplicated_candidates=None):
    prev_windows = _available_structural_windows(
        prev_slots,
        lambda slot: slot.residual_index not in ledger.prev_used_by,
    )
    if not prev_windows:
        return False
    curr_windows = _available_structural_windows(
        curr_slots,
        lambda slot: ledger.prev_for_curr[slot.residual_index] is None,
    )
    unresolved_windows = prev_windows.intersection(curr_windows)
    if not unresolved_windows:
        return False
    if duplicated_candidates is not None:
        duplicated_both = unresolved_windows.intersection(
            duplicated_candidates,
        )
        if not duplicated_both:
            return None
        return True
    # Structural disambiguation is needed only while multiple old occurrences
    # are competing with multiple current occurrences.  A one-sided duplicate
    # is a copy/deletion problem, for which paragraph context alone is not
    # sufficient evidence to override the existing copy and reinsertion rules.
    duplicated_prev = _duplicated_candidate_windows(
        full_prev_slots, unresolved_windows,
    )
    if not duplicated_prev:
        return None
    duplicated_both = _duplicated_candidate_windows(
        full_curr_slots, duplicated_prev,
    )
    if not duplicated_both:
        return None
    return True


def _propose_structural_word_matches_slots(
        ledger, text_prev, text_curr, prev_slots=None, curr_slots=None,
        full_prev_slots=None, full_curr_slots=None,
        get_structural_context=None, get_structural_duplicates=None):
    unresolved_candidates = _unresolved_residual_windows(
        ledger, text_prev, text_curr,
    )
    if not unresolved_candidates:
        return
    duplicated_candidates = None
    if get_structural_duplicates is not None:
        hierarchy_duplicates = get_structural_duplicates(
            unresolved_candidates,
        )
        if hierarchy_duplicates is not None:
            if not hierarchy_duplicates:
                return
            duplicated_candidates = hierarchy_duplicates
    if get_structural_context is not None:
        (prev_slots, curr_slots,
         full_prev_slots, full_curr_slots) = get_structural_context()
    if not (prev_slots and curr_slots and full_prev_slots and full_curr_slots):
        return
    trigger = _has_unresolved_duplicate_window(
        ledger, prev_slots, curr_slots, full_prev_slots, full_curr_slots,
        duplicated_candidates=duplicated_candidates,
    )
    if not trigger:
        return

    prev_residual_by_path = dict((slot.path, slot.residual_index) for slot in prev_slots)
    curr_residual_by_path = dict((slot.path, slot.residual_index) for slot in curr_slots)
    if (len(prev_residual_by_path) != len(prev_slots) or
            len(curr_residual_by_path) != len(curr_slots)):
        return

    # Once a genuine duplicate competition is established, preserve the
    # complete structural evidence universe across every certified pair.  The
    # later residual-completeness rule means only windows wholly contained in
    # residual spans can contribute a candidate, so global ambiguity counting
    # can be restricted exactly to that complete necessary set.
    prev_index = _StructuralIndex(
        full_prev_slots, build_occurrences=False,
    )
    curr_index = _StructuralIndex(
        full_curr_slots, build_occurrences=False,
    )
    prev_keys = prev_index.keys
    curr_keys = curr_index.keys
    prev_target_paragraphs = set(
        path[0] for path in prev_residual_by_path
    )
    curr_target_paragraphs = set(
        path[0] for path in curr_residual_by_path
    )
    prev_paragraphs, curr_paragraphs, chains = _structural_anchor_chains(
        prev_index, curr_index, prev_target_paragraphs,
        curr_target_paragraphs,
    )
    certified_pairs = _unique_best_structural_pairs(chains)

    ambiguity_candidates = set()
    candidate_gaps = defaultdict(list)
    for pair in sorted(certified_pairs):
        chain, pair_score = chains[pair]
        if pair_score[0] < WORD_MATCH_STRUCTURAL_MIN_PAIR_INFO:
            continue
        prev_paragraph = prev_paragraphs[pair[0]]
        curr_paragraph = curr_paragraphs[pair[1]]
        for prev_start, prev_end, curr_start, curr_end in (
                _structural_gap_ranges(
                    chain, len(prev_paragraph), len(curr_paragraph))):
            prev_gap = prev_paragraph[prev_start:prev_end]
            curr_gap = curr_paragraph[curr_start:curr_end]
            prev_windows = _residual_structural_window_keys(
                prev_gap, prev_index, prev_residual_by_path,
            )
            if not prev_windows:
                continue
            shared_windows = _residual_structural_window_keys(
                curr_gap, curr_index, curr_residual_by_path, prev_windows,
            )
            if not shared_windows:
                continue
            gap = (prev_start, prev_end, curr_start, curr_end)
            candidate_gaps[pair].append((gap, shared_windows))
            ambiguity_candidates.update(shared_windows)
    if not ambiguity_candidates:
        return

    duplicated_prev = _duplicated_structural_candidate_windows(
        prev_index, ambiguity_candidates,
    )
    ambiguous_windows = _duplicated_structural_candidate_windows(
        curr_index, duplicated_prev,
    )
    if not ambiguous_windows:
        return

    for pair in sorted(candidate_gaps):
        chain, pair_score = chains[pair]
        prev_paragraph = prev_paragraphs[pair[0]]
        curr_paragraph = curr_paragraphs[pair[1]]

        for gap, gap_candidates in candidate_gaps[pair]:
            if not gap_candidates.intersection(ambiguous_windows):
                continue
            prev_start, prev_end, curr_start, curr_end = gap
            prev_gap = prev_paragraph[prev_start:prev_end]
            curr_gap = curr_paragraph[curr_start:curr_end]
            gap_pairs = _lcs_token_pairs(
                [prev_keys[slot.article_index] for slot in prev_gap],
                [curr_keys[slot.article_index] for slot in curr_gap],
                [slot.value for slot in prev_gap],
                [slot.value for slot in curr_gap],
            )
            for run in _contiguous_pair_runs(gap_pairs):
                informative = sum(
                    _is_informative_move_token(
                        prev_gap[prev_offset].value
                    )
                    for prev_offset, _ in run
                )
                if informative < WORD_MATCH_STRUCTURAL_MIN_RUN_INFO:
                    continue
                if not _structural_run_has_boundary_context(
                        prev_gap, curr_gap, run):
                    continue
                run_prev_slots = [prev_gap[prev_offset] for prev_offset, _ in run]
                run_curr_slots = [curr_gap[curr_offset] for _, curr_offset in run]
                if not _ambiguous_windows_in_both(
                        run_prev_slots, run_curr_slots,
                        prev_index, curr_index, ambiguous_windows):
                    continue
                residual_pairs = []
                residual_paths = []
                for prev_offset, curr_offset in run:
                    prev_slot = prev_gap[prev_offset]
                    curr_slot = curr_gap[curr_offset]
                    prev_residual = prev_residual_by_path.get(prev_slot.path)
                    curr_residual = curr_residual_by_path.get(curr_slot.path)
                    if prev_residual is not None and curr_residual is not None:
                        residual_pairs.append((curr_residual, prev_residual))
                        residual_paths.append((curr_slot.path, prev_slot.path))
                # Evidence belongs to the complete aligned run.  If an exact
                # paragraph/sentence reuse has already removed part of it from
                # the residual word problem, the remaining fragment cannot
                # inherit the evidence tier of the original run.
                if len(residual_pairs) != len(run):
                    continue
                ledger.propose_pairs(
                    residual_pairs, WORD_MATCH_CONF_STRUCTURAL_GAP,
                    'structural-gap', support=pair_score[0] + informative,
                    paths=residual_paths,
                )


def _propose_structural_word_matches_document(
        ledger, text_prev, text_curr, get_structural_documents,
        get_structural_duplicates):
    """Propose the same structural runs over a compact offset document.

    ``True`` means the compact path handled the decision, including a proven
    no-op.  ``False`` requests the original slot/tokenizer fallback.
    """
    unresolved_candidates = _unresolved_residual_windows(
        ledger, text_prev, text_curr,
    )
    if not unresolved_candidates:
        return True

    duplicated_candidates = get_structural_duplicates(
        unresolved_candidates,
    )
    if duplicated_candidates is None:
        return False
    if not duplicated_candidates:
        return True

    context = get_structural_documents()
    if context is None:
        return False
    if len(context) == 4:
        (prev_document, curr_document,
         prev_residual_by_article, curr_residual_by_article) = context
        prev_residual_flags = None
        curr_residual_flags = None
    else:
        (prev_document, curr_document,
         prev_residual_by_article, curr_residual_by_article,
         prev_residual_flags, curr_residual_flags) = context
    if not (prev_residual_by_article and curr_residual_by_article):
        return True

    prev_windows = _compact_available_residual_windows(
        prev_document, prev_residual_by_article,
        lambda index: index not in ledger.prev_used_by,
        native_availability=ledger.prev_used_by,
        native_previous_mode=True,
    )
    if not prev_windows:
        return True
    curr_windows = _compact_available_residual_windows(
        curr_document, curr_residual_by_article,
        lambda index: ledger.prev_for_curr[index] is None,
        native_availability=ledger.prev_for_curr,
    )
    if not prev_windows.intersection(
            curr_windows, duplicated_candidates):
        return True

    prev_document.ensure_index()
    curr_document.ensure_index()
    prev_target_paragraphs = set(
        residual[1][0] for residual in prev_residual_by_article.values()
    )
    curr_target_paragraphs = set(
        residual[1][0] for residual in curr_residual_by_article.values()
    )
    chains = _compact_structural_anchor_chains(
        prev_document, curr_document, prev_target_paragraphs,
        curr_target_paragraphs,
    )
    certified_pairs = _unique_best_structural_pairs(chains)

    ambiguity_candidates = set()
    candidate_gaps = defaultdict(list)
    for pair in sorted(certified_pairs):
        chain, pair_score = chains[pair]
        if pair_score[0] < WORD_MATCH_STRUCTURAL_MIN_PAIR_INFO:
            continue
        prev_length = prev_document.paragraph_length(pair[0])
        curr_length = curr_document.paragraph_length(pair[1])
        for prev_start, prev_end, curr_start, curr_end in (
                _structural_gap_ranges(chain, prev_length, curr_length)):
            prev_windows = _compact_residual_structural_window_keys(
                prev_document, pair[0], prev_start, prev_end,
                prev_residual_by_article,
                residual_flags=prev_residual_flags,
            )
            if not prev_windows:
                continue
            shared_windows = _compact_residual_structural_window_keys(
                curr_document, pair[1], curr_start, curr_end,
                curr_residual_by_article, prev_windows,
                residual_flags=curr_residual_flags,
            )
            if not shared_windows:
                continue
            gap = (prev_start, prev_end, curr_start, curr_end)
            candidate_gaps[pair].append((gap, shared_windows))
            ambiguity_candidates.update(shared_windows)
    if not ambiguity_candidates:
        return True

    duplicated_prev = _compact_duplicated_structural_candidate_windows(
        prev_document, ambiguity_candidates,
    )
    ambiguous_windows = _compact_duplicated_structural_candidate_windows(
        curr_document, duplicated_prev,
    )
    if not ambiguous_windows:
        return True

    for pair in sorted(candidate_gaps):
        _, pair_score = chains[pair]
        prev_paragraph_start, _ = prev_document.paragraph_range(pair[0])
        curr_paragraph_start, _ = curr_document.paragraph_range(pair[1])
        for gap, gap_candidates in candidate_gaps[pair]:
            if not gap_candidates.intersection(ambiguous_windows):
                continue
            prev_start, prev_end, curr_start, curr_end = gap
            prev_article_start = prev_paragraph_start + prev_start
            prev_article_end = prev_paragraph_start + prev_end
            curr_article_start = curr_paragraph_start + curr_start
            curr_article_end = curr_paragraph_start + curr_end
            prev_gap_values = prev_document.values[
                prev_article_start:prev_article_end
            ]
            curr_gap_values = curr_document.values[
                curr_article_start:curr_article_end
            ]
            gap_pairs = _lcs_token_pairs(
                prev_document.keys[prev_article_start:prev_article_end],
                curr_document.keys[curr_article_start:curr_article_end],
                prev_gap_values, curr_gap_values,
            )
            for run in _contiguous_pair_runs(gap_pairs):
                informative = sum(
                    _is_informative_move_token(
                        prev_gap_values[prev_offset]
                    )
                    for prev_offset, _ in run
                )
                if informative < WORD_MATCH_STRUCTURAL_MIN_RUN_INFO:
                    continue
                if not _compact_structural_run_has_boundary_context(
                        prev_gap_values, curr_gap_values, run):
                    continue
                if not _compact_run_has_ambiguous_window(
                        prev_document, curr_document,
                        prev_article_start, curr_article_start, run,
                        ambiguous_windows):
                    continue

                residual_pairs = []
                residual_paths = []
                for prev_offset, curr_offset in run:
                    prev_residual = prev_residual_by_article.get(
                        prev_article_start + prev_offset
                    )
                    curr_residual = curr_residual_by_article.get(
                        curr_article_start + curr_offset
                    )
                    if prev_residual is not None and curr_residual is not None:
                        residual_pairs.append((
                            curr_residual[0], prev_residual[0],
                        ))
                        residual_paths.append((
                            curr_residual[1], prev_residual[1],
                        ))
                if len(residual_pairs) != len(run):
                    continue
                ledger.propose_pairs(
                    residual_pairs, WORD_MATCH_CONF_STRUCTURAL_GAP,
                    'structural-gap', support=pair_score[0] + informative,
                    paths=residual_paths,
                )
    return True


def _propose_structural_word_matches(
        ledger, text_prev, text_curr, prev_slots=None, curr_slots=None,
        full_prev_slots=None, full_curr_slots=None,
        get_structural_context=None, get_structural_duplicates=None,
        get_structural_documents=None):
    if (get_structural_documents is not None and
            get_structural_duplicates is not None):
        handled = _propose_structural_word_matches_document(
            ledger, text_prev, text_curr, get_structural_documents,
            get_structural_duplicates,
        )
        if handled:
            return
    _propose_structural_word_matches_slots(
        ledger, text_prev, text_curr,
        prev_slots=prev_slots, curr_slots=curr_slots,
        full_prev_slots=full_prev_slots, full_curr_slots=full_curr_slots,
        get_structural_context=get_structural_context,
        get_structural_duplicates=get_structural_duplicates,
    )


def _match_word_sequences(text_prev, text_curr, full_text_prev=None, full_text_curr=None,
                          get_full_texts=None, prev_words=None,
                          prev_slots=None, curr_slots=None,
                          full_prev_slots=None, full_curr_slots=None,
                          get_structural_context=None,
                          get_structural_duplicates=None,
                          get_structural_documents=None):
    ledger = _MatchCandidateLedger(len(text_prev), len(text_curr))
    prev_keys = _word_match_keys(text_prev)
    curr_keys = _word_match_keys(text_curr)

    prefix_len = _common_prefix_len(prev_keys, curr_keys)
    prefix_len, suffix_len = _rollback_common_construct_edges(text_prev, text_curr,
                                                              prev_keys, curr_keys,
                                                              prefix_len)
    ledger.propose_pairs(
        ((index, index) for index in range(prefix_len)),
        WORD_MATCH_CONF_EDGE, 'common-prefix',
    )
    ledger.propose_pairs((
        (len(text_curr) - suffix_len + index,
         len(text_prev) - suffix_len + index)
        for index in range(suffix_len)
    ), WORD_MATCH_CONF_EDGE, 'common-suffix')

    prev_mid_start = prefix_len
    prev_mid_end = len(text_prev) - suffix_len
    curr_mid_start = prefix_len
    curr_mid_end = len(text_curr) - suffix_len
    prev_mid = text_prev[prev_mid_start:prev_mid_end]
    curr_mid = text_curr[curr_mid_start:curr_mid_end]
    prev_mid_keys = prev_keys[prev_mid_start:prev_mid_end]
    curr_mid_keys = curr_keys[curr_mid_start:curr_mid_end]
    move_prev_spans = []
    move_curr_spans = []

    if prev_mid and curr_mid:
        max_drift = _word_match_drift_limit(len(prev_mid), len(curr_mid))
        if _word_match_pair_estimate(prev_mid_keys, curr_mid_keys) <= WORD_MATCH_MAX_SEQUENCE_PAIRS:
            matcher = SequenceMatcher(None, prev_mid_keys, curr_mid_keys, autojunk=False)
            for tag, i1, i2, j1, j2 in matcher.get_opcodes():
                if tag == 'equal':
                    ledger.propose_pairs((
                        (curr_mid_start + curr_index,
                         prev_mid_start + prev_index)
                        for prev_index, curr_index in zip(
                            range(i1, i2), range(j1, j2),
                        )
                    ), WORD_MATCH_CONF_SEQUENCE_EQUAL, 'sequence-equal')
                else:
                    if tag in ('replace', 'delete') and i1 < i2:
                        move_prev_spans.append((prev_mid_start + i1, prev_mid_start + i2))
                    if tag in ('replace', 'insert') and j1 < j2:
                        move_curr_spans.append((curr_mid_start + j1, curr_mid_start + j2))
                    if tag == 'replace' and (i2 - i1) * (j2 - j1) <= WORD_MATCH_MAX_LOCAL_PAIRS:
                        local_matches = _nearest_word_matches(prev_mid_keys[i1:i2],
                                                              curr_mid_keys[j1:j2],
                                                              prev_mid_start + i1,
                                                              curr_mid_start + j1,
                                                              max_drift)
                        ledger.propose_pairs((
                            (curr_mid_start + j1 + curr_index,
                             prev_mid_start + i1 + prev_index)
                            for curr_index, prev_index in sorted(local_matches.items())
                        ), WORD_MATCH_CONF_LOCAL, 'local-nearest')
        else:
            move_prev_spans.append((prev_mid_start, prev_mid_end))
            move_curr_spans.append((curr_mid_start, curr_mid_end))
            local_matches = _nearest_word_matches(prev_mid_keys, curr_mid_keys,
                                                  prev_mid_start, curr_mid_start,
                                                  max_drift)
            ledger.propose_pairs((
                (curr_mid_start + curr_index, prev_mid_start + prev_index)
                for curr_index, prev_index in sorted(local_matches.items())
            ), WORD_MATCH_CONF_LOCAL, 'local-nearest')

    recovery_count_state = {'counts': {}}
    _recover_moved_word_runs(text_prev, text_curr, prev_keys, curr_keys,
                             ledger,
                             full_text_prev=full_text_prev,
                             full_text_curr=full_text_curr,
                             get_full_texts=get_full_texts,
                             prev_candidate_spans=move_prev_spans,
                             curr_candidate_spans=move_curr_spans,
                             count_state=recovery_count_state)
    _recover_unique_short_numeric_template_fields(
        text_prev, text_curr, ledger,
        full_text_prev=full_text_prev,
        full_text_curr=full_text_curr,
        get_full_texts=get_full_texts,
        prev_candidate_spans=move_prev_spans,
        curr_candidate_spans=move_curr_spans,
        count_state=recovery_count_state,
    )
    _recover_unique_template_field_words(
        text_prev, text_curr, prev_keys, curr_keys,
        ledger,
        full_text_prev=full_text_prev,
        full_text_curr=full_text_curr,
        get_full_texts=get_full_texts,
        count_state=recovery_count_state,
    )
    _recover_edited_link_boundaries(text_prev, text_curr, ledger)
    _demote_stale_suffix_edge_matches(text_prev, text_curr, prev_words,
                                      ledger, prefix_len, suffix_len)
    _propose_structural_word_matches(
        ledger, text_prev, text_curr,
        prev_slots=prev_slots, curr_slots=curr_slots,
        full_prev_slots=full_prev_slots, full_curr_slots=full_curr_slots,
        get_structural_context=get_structural_context,
        get_structural_duplicates=get_structural_duplicates,
        get_structural_documents=get_structural_documents,
    )

    return ledger.resolve()


def _can_partially_restore_historical_sentence(words, previous_revision_id):
    available = [word for word in words if not word.matched]
    occupied_count = len(words) - len(available)
    if (occupied_count == 0 or
            occupied_count > WORD_MATCH_HISTORICAL_MAX_OCCUPIED_TOKENS or
            len(available) < WORD_MATCH_HISTORICAL_MIN_AVAILABLE_TOKENS or
            len(available) < WORD_MATCH_HISTORICAL_MIN_EVIDENCE_RATIO * occupied_count):
        return False
    removal_revisions = {
        word.outbound[-1]
        for word in available
        if word.outbound
    }
    return (len(removal_revisions) == 1 and
            previous_revision_id not in removal_revisions and
            all(word.outbound for word in available))


class Wikiwho:
    def __init__(self, article_title):
        # Hash tables.
        self.paragraphs_ht = {}
        self.sentences_ht = {}

        self.spam_ids = []
        # Mutable public list kept for compatibility. Changes to spam_hashes
        # must go through _add_spam_revision() exclusively so spam_hashes_set
        # stays in sync for membership checks.
        self.spam_hashes = []
        self.tokens = []  # [word_obj, ..] ordered, unique list of tokens of this article
        self.revisions = {}  # {rev_id : rev_obj, ...}
        self.ordered_revisions = []  # [rev_id, ...]
        self.rvcontinue = '0'
        self.title = article_title
        self.page_id = None  # article id
        self.token_id = 0  # sequential id for tokens in article. unique per token per article.
        # Revisions to compare.
        self.revision_curr = Revision()
        self.revision_prev = Revision()

        self.text_curr = ''
        self.temp = []

        # Keep the public list shape while using a set for membership checks.
        self.spam_hashes_set = set()

    def _add_spam_revision(self, rev_id, rev_hash):
        """Record a spam revision while keeping the public list and set synced."""
        self.spam_ids.append(rev_id)
        self.spam_hashes.append(rev_hash)
        self.spam_hashes_set.add(rev_hash)

    def clean_attributes(self):
        """
        Empty attributes that are usually not needed after analyzing an article.
        """
        self.revision_prev = None
        self.text_curr = ''
        self.temp = []

    def analyse_article_from_xml_dump(self, page):
        """
        Analyse page from XML Dump Iterator.
        :param page: Page meta data and a Revision iterator. Each revision contains metadata and text.
        """
        # Iterate over revisions of the article.
        for revision in page:
            text = revision.text or ''
            if not text and (revision.deleted.text or revision.deleted.restricted):
                # equivalent of "'texthidden' in revision or 'textmissing' in revision" in analyse_article
                continue

            vandalism = False
            # Update the information about the previous revision.
            self.revision_prev = self.revision_curr

            rev_id = revision.id
            rev_hash = revision.sha1 or calculate_hash(text)
            if rev_hash in self.spam_hashes_set:
                vandalism = True

            # TODO: spam detection: DELETION
            text_len = len(text)
            if not vandalism and not(revision.comment and revision.minor):
                # if content is not moved (flag) to different article in good faith, check for vandalism
                # if revisions have reached a certain size
                if self.revision_prev.length > PREVIOUS_LENGTH and \
                   text_len < CURR_LENGTH and \
                   ((text_len-self.revision_prev.length) / self.revision_prev.length) <= CHANGE_PERCENTAGE:
                    # VANDALISM: CHANGE PERCENTAGE - DELETION
                    vandalism = True

            if vandalism:
                # print("---------------------------- FLAG 1")
                self.revision_curr = self.revision_prev
                self._add_spam_revision(rev_id, rev_hash)
            else:
                # Information about the current revision.
                self.revision_curr = Revision()
                self.revision_curr.id = rev_id
                self.revision_curr.length = text_len
                self.revision_curr.timestamp = revision.timestamp.long_format()

                # Get editor information
                if revision.user:
                    user_text = revision.user.text
                    contributor_name = '' if not user_text or user_text == 'None' else user_text
                    if revision.user.id is None and contributor_name or revision.user.id == 0:
                        contributor_id = 0
                    else:
                        contributor_id = revision.user.id or ''
                else:
                    # Some revisions don't have contributor.
                    contributor_name = ''
                    contributor_id = ''
                editor = contributor_id
                editor = str(editor) if editor != 0 else '0|{}'.format(contributor_name)
                self.revision_curr.editor = editor

                # Content within the revision.
                self.text_curr = text.lower()

                # Perform comparison.
                vandalism = self.determine_authorship()

                if vandalism:
                    # print "---------------------------- FLAG 2"
                    self.revision_curr = self.revision_prev  # skip revision with vandalism in history
                    self._add_spam_revision(rev_id, rev_hash)
                else:
                    # Add the current revision with all the information.
                    self.revisions.update({self.revision_curr.id: self.revision_curr})
                    self.ordered_revisions.append(self.revision_curr.id)
            self.temp = []

    def analyse_article(self, page):
        """
        Analyse page in json form.
        :param page: List of revisions. Each revision is a dict and contains metadata and text.
        """
        # Iterate over revisions of the article.
        for revision in page:
            if 'texthidden' in revision or 'textmissing' in revision:
                continue

            vandalism = False
            # Update the information about the previous revision.
            self.revision_prev = self.revision_curr

            text = revision.get('*', '')
            rev_id = int(revision['revid'])
            rev_hash = revision.get('sha1')
            if not rev_hash:
                rev_hash = calculate_hash(text)
            if rev_hash in self.spam_hashes_set:
                vandalism = True

            # TODO: spam detection: DELETION
            text_len = len(text)
            if not vandalism and not(revision.get('comment') and 'minor' in revision):
                # if content is not moved (flag) to different article in good faith, check for vandalism
                # if revisions have reached a certain size
                if self.revision_prev.length > PREVIOUS_LENGTH and \
                   text_len < CURR_LENGTH and \
                   ((text_len-self.revision_prev.length) / self.revision_prev.length) <= CHANGE_PERCENTAGE:
                    # VANDALISM: CHANGE PERCENTAGE - DELETION
                    vandalism = True

            if vandalism:
                # print("---------------------------- FLAG 1")
                self.revision_curr = self.revision_prev
                self._add_spam_revision(rev_id, rev_hash)
            else:
                # Information about the current revision.
                self.revision_curr = Revision()
                self.revision_curr.id = rev_id
                self.revision_curr.length = text_len
                self.revision_curr.timestamp = revision['timestamp']

                # Get editor information.
                # Some revisions don't have editor.
                contributor_id = revision.get('userid', '')
                contributor_name = revision.get('user', '')
                editor = contributor_id
                editor = str(editor) if editor != 0 else '0|{}'.format(contributor_name)
                self.revision_curr.editor = editor

                # Content within the revision.
                self.text_curr = text.lower()

                # Perform comparison.
                vandalism = self.determine_authorship()

                if vandalism:
                    # print "---------------------------- FLAG 2"
                    self.revision_curr = self.revision_prev  # skip revision with vandalism in history
                    self._add_spam_revision(rev_id, rev_hash)
                else:
                    # Add the current revision with all the information.
                    self.revisions.update({self.revision_curr.id: self.revision_curr})
                    self.ordered_revisions.append(self.revision_curr.id)
            self.temp = []

    def determine_authorship(self):
        # Containers for unmatched paragraphs and sentences in both revisions.
        unmatched_sentences_curr = []
        restored_sentences_curr = []
        unmatched_sentences_prev = []
        matched_paragraphs_prev = []
        matched_sentences_prev = []
        matched_words_prev = []
        possible_vandalism = False
        vandalism = False

        try:
            # Analysis of the paragraphs in the current revision.
            unmatched_paragraphs_curr, unmatched_paragraphs_prev, matched_paragraphs_prev = \
                self.analyse_paragraphs_in_revision()

            # Analysis of the sentences in the unmatched paragraphs of the current revision.
            if unmatched_paragraphs_curr:
                unmatched_sentences_curr, restored_sentences_curr, unmatched_sentences_prev, \
                    matched_sentences_prev, total_sentences = \
                    self.analyse_sentences_in_paragraphs(unmatched_paragraphs_curr, unmatched_paragraphs_prev)

                # TODO: spam detection
                if len(unmatched_paragraphs_curr) / len(self.revision_curr.ordered_paragraphs) > UNMATCHED_PARAGRAPH:
                    # will be used to detect copy-paste vandalism - token density
                    possible_vandalism = True

                # Analysis of words in unmatched sentences (diff of both texts).
                if unmatched_sentences_curr:
                    matched_words_prev, vandalism = self.analyse_words_in_sentences(unmatched_sentences_curr,
                                                                                    unmatched_sentences_prev,
                                                                                    possible_vandalism)
        except Exception:
            # Error occurred during analysing the current revision
            # Hold the last successfully processed revision.
            self.revision_curr = self.revision_prev
            # Reset matched structures from old revisions.
            for matched_paragraph in matched_paragraphs_prev:
                matched_paragraph.matched = False
                for sentence_hash in matched_paragraph.sentences:
                    for sentence in matched_paragraph.sentences[sentence_hash]:
                        sentence.matched = False
                        for word_prev in sentence.words:
                            word_prev.matched = False
            for matched_sentence in matched_sentences_prev:
                matched_sentence.matched = False
                for word_prev in matched_sentence.words:
                    word_prev.matched = False
            for matched_word in matched_words_prev:
                matched_word.matched = False
            raise

        if not vandalism:
            # Add the information of 'deletion' to words
            for unmatched_sentence in unmatched_sentences_prev:
                for word_prev in unmatched_sentence.words:
                    if not word_prev.matched:
                        word_prev.outbound.append(self.revision_curr.id)
            if not unmatched_sentences_prev:
                # if all current paragraphs are matched
                for unmatched_paragraph in unmatched_paragraphs_prev:
                    for sentence_hash in unmatched_paragraph.sentences:
                        for sentence in unmatched_paragraph.sentences[sentence_hash]:
                            for word_prev in sentence.words:
                                if not word_prev.matched:
                                    word_prev.outbound.append(self.revision_curr.id)

        # Reset matched structures from old revisions. And update inbound and last used info of matched words.
        for matched_paragraph in matched_paragraphs_prev:
            matched_paragraph.matched = False
            for sentence_hash in matched_paragraph.sentences:
                for sentence in matched_paragraph.sentences[sentence_hash]:
                    sentence.matched = False
                    for word_prev in sentence.words:
                        # first update inbound and last used info of matched words of all previous revisions
                        if not vandalism and word_prev.matched and \
                                (not word_prev.outbound or word_prev.outbound[-1] != self.revision_curr.id):
                            if word_prev.last_rev_id != self.revision_prev.id:
                                word_prev.inbound.append(self.revision_curr.id)
                            word_prev.last_rev_id = self.revision_curr.id
                        # reset
                        word_prev.matched = False
        for matched_sentence in matched_sentences_prev:
            matched_sentence.matched = False
            for word_prev in matched_sentence.words:
                # first update inbound and last used info of matched words of all previous revisions
                if not vandalism and word_prev.matched and \
                        (not word_prev.outbound or word_prev.outbound[-1] != self.revision_curr.id):
                    if word_prev.last_rev_id != self.revision_prev.id:
                        word_prev.inbound.append(self.revision_curr.id)
                    word_prev.last_rev_id = self.revision_curr.id
                # reset
                word_prev.matched = False
        for matched_word in matched_words_prev:
            # first update last used info of matched prev words
            # there is no inbound chance because we only diff with words of previous revision
            if not vandalism and word_prev.matched:
                if not word_prev.outbound or word_prev.outbound[-1] != self.revision_curr.id:
                    word_prev.last_rev_id = self.revision_curr.id
            # reset
            matched_word.matched = False

        if not vandalism:
            # Add the new paragraphs to hash table of paragraphs.
            for unmatched_paragraph in unmatched_paragraphs_curr:
                if unmatched_paragraph.hash_value in self.paragraphs_ht:
                    self.paragraphs_ht[unmatched_paragraph.hash_value].append(unmatched_paragraph)
                else:
                    self.paragraphs_ht.update({unmatched_paragraph.hash_value: [unmatched_paragraph]})
                unmatched_paragraph.value = ''  # hash value is not used for next rev analysis

            # Add the new sentences to hash table of sentences.
            for sentence_curr in unmatched_sentences_curr:
                if sentence_curr.hash_value in self.sentences_ht:
                    self.sentences_ht[sentence_curr.hash_value].append(sentence_curr)
                else:
                    self.sentences_ht.update({sentence_curr.hash_value: [sentence_curr]})
                sentence_curr.value = ''  # hash value is not used for next rev analysis
                sentence_curr.splitted = None  # splitted word values are not used for next rev analysis

            # A partially restored sentence supersedes the stale historical
            # representation whose occupied words forced reconstruction. Put
            # it first so a later delete-and-reinsert tries the current token
            # identities before that stale representation.
            for sentence_curr in restored_sentences_curr:
                if sentence_curr.hash_value in self.sentences_ht:
                    self.sentences_ht[sentence_curr.hash_value].insert(0, sentence_curr)
                else:
                    self.sentences_ht.update({sentence_curr.hash_value: [sentence_curr]})
                sentence_curr.value = ''
                sentence_curr.splitted = None

        return vandalism

    def analyse_paragraphs_in_revision(self):
        # Containers for unmatched and matched paragraphs.
        unmatched_paragraphs_curr = []
        unmatched_paragraphs_prev = []
        matched_paragraphs_prev = []

        # Split the text of the current into paragraphs.
        paragraphs = split_into_paragraphs(self.text_curr)

        # Iterate over the paragraphs of the current version.
        for paragraph in paragraphs:
            # Build Paragraph structure and calculate hash value.
            paragraph = paragraph.strip()
            if not paragraph:
                # dont track empty lines
                continue
            # TODO should we clean whitespaces in paragraph level?
            # paragraph = ' '.join(split_into_tokens(paragraph))
            hash_curr = calculate_hash(paragraph)
            matched_curr = False

            # If the paragraph is in the previous revision,
            # update the authorship information and mark both paragraphs as matched (also in HT).
            for paragraph_prev in self.revision_prev.paragraphs.get(hash_curr, []):
                if not paragraph_prev.matched:
                    matched_one = False
                    matched_all = True
                    for h in paragraph_prev.sentences:
                        for s_prev in paragraph_prev.sentences[h]:
                            for w_prev in s_prev.words:
                                if w_prev.matched:
                                    matched_one = True
                                else:
                                    matched_all = False

                    if not matched_one:
                        # if there is not any already matched prev word, so set them all as matched
                        matched_curr = True
                        paragraph_prev.matched = True
                        matched_paragraphs_prev.append(paragraph_prev)

                        # Set all sentences and words of this paragraph as matched
                        for hash_sentence_prev in paragraph_prev.sentences:
                            for sentence_prev in paragraph_prev.sentences[hash_sentence_prev]:
                                sentence_prev.matched = True
                                for word_prev in sentence_prev.words:
                                    word_prev.matched = True

                        # Add paragraph to current revision.
                        if hash_curr in self.revision_curr.paragraphs:
                            self.revision_curr.paragraphs[hash_curr].append(paragraph_prev)
                        else:
                            self.revision_curr.paragraphs.update({paragraph_prev.hash_value: [paragraph_prev]})
                        self.revision_curr.ordered_paragraphs.append(paragraph_prev.hash_value)
                        break
                    elif matched_all:
                        # if all prev words in this paragraph are already matched
                        paragraph_prev.matched = True
                        # for hash_sentence_prev in paragraph_prev.sentences:
                        #     for sentence_prev in paragraph_prev.sentences[hash_sentence_prev]:
                        #         sentence_prev.matched = True
                        matched_paragraphs_prev.append(paragraph_prev)

            # If the paragraph is not in the previous revision, but it is in an older revision
            # update the authorship information and mark both paragraphs as matched.
            if not matched_curr:
                for paragraph_prev in self.paragraphs_ht.get(hash_curr, []):
                    if not paragraph_prev.matched:
                        matched_one = False
                        matched_all = True
                        for h in paragraph_prev.sentences:
                            for s_prev in paragraph_prev.sentences[h]:
                                for w_prev in s_prev.words:
                                    if w_prev.matched:
                                        matched_one = True
                                    else:
                                        matched_all = False

                        if not matched_one:
                            # if there is not any already matched prev word, so set them all as matched
                            matched_curr = True
                            paragraph_prev.matched = True
                            matched_paragraphs_prev.append(paragraph_prev)

                            # Set all sentences and words of this paragraph as matched
                            for hash_sentence_prev in paragraph_prev.sentences:
                                for sentence_prev in paragraph_prev.sentences[hash_sentence_prev]:
                                    sentence_prev.matched = True
                                    for word_prev in sentence_prev.words:
                                        word_prev.matched = True

                            # Add paragraph to current revision.
                            if hash_curr in self.revision_curr.paragraphs:
                                self.revision_curr.paragraphs[hash_curr].append(paragraph_prev)
                            else:
                                self.revision_curr.paragraphs.update({paragraph_prev.hash_value: [paragraph_prev]})
                            self.revision_curr.ordered_paragraphs.append(paragraph_prev.hash_value)
                            break
                        elif matched_all:
                            # if all prev words in this paragraph are already matched
                            paragraph_prev.matched = True
                            # for hash_sentence_prev in paragraph_prev.sentences:
                            #     for sentence_prev in paragraph_prev.sentences[hash_sentence_prev]:
                            #         sentence_prev.matched = True
                            matched_paragraphs_prev.append(paragraph_prev)

            # If the paragraph did not match with previous revisions,
            # add to container of unmatched paragraphs for further analysis.
            if not matched_curr:
                paragraph_curr = Paragraph()
                paragraph_curr.hash_value = hash_curr
                paragraph_curr.value = paragraph

                if hash_curr in self.revision_curr.paragraphs:
                    self.revision_curr.paragraphs[hash_curr].append(paragraph_curr)
                else:
                    self.revision_curr.paragraphs.update({paragraph_curr.hash_value: [paragraph_curr]})
                self.revision_curr.ordered_paragraphs.append(paragraph_curr.hash_value)
                unmatched_paragraphs_curr.append(paragraph_curr)

        # Identify unmatched paragraphs in previous revision for further analysis.
        paragraph_duplicate_counts = {}
        for paragraph_prev_hash in self.revision_prev.ordered_paragraphs:
            if len(self.revision_prev.paragraphs[paragraph_prev_hash]) > 1:
                count = paragraph_duplicate_counts.get(paragraph_prev_hash, 0) + 1
                paragraph_duplicate_counts[paragraph_prev_hash] = count
                paragraph_prev = self.revision_prev.paragraphs[paragraph_prev_hash][count - 1]
            else:
                paragraph_prev = self.revision_prev.paragraphs[paragraph_prev_hash][0]
            if not paragraph_prev.matched:
                unmatched_paragraphs_prev.append(paragraph_prev)

        return unmatched_paragraphs_curr, unmatched_paragraphs_prev, matched_paragraphs_prev

    def analyse_sentences_in_paragraphs(self, unmatched_paragraphs_curr, unmatched_paragraphs_prev):
        # Containers for unmatched and matched sentences.
        unmatched_sentences_curr = []
        restored_sentences_curr = []
        unmatched_sentences_prev = []
        matched_sentences_prev = []
        total_sentences = 0

        # Iterate over the unmatched paragraphs of the current revision.
        for paragraph_curr in unmatched_paragraphs_curr:
            # Split the current paragraph into sentences.
            sentences = split_into_sentences(paragraph_curr.value)
            # Iterate over the sentences of the current paragraph
            for sentence in sentences:
                # Create the Sentence structure.
                sentence = sentence.strip()
                if not sentence:
                    # dont track empty lines
                    continue
                sentence = ' '.join(split_into_tokens(sentence))  # here whitespaces in the sentence are cleaned
                hash_curr = calculate_hash(sentence)  # then hash values is calculated
                matched_curr = False
                total_sentences += 1

                # Iterate over the unmatched paragraphs from the previous revision.
                for paragraph_prev in unmatched_paragraphs_prev:
                    for sentence_prev in paragraph_prev.sentences.get(hash_curr, []):
                        if not sentence_prev.matched:
                            matched_one = False
                            matched_all = True
                            for word_prev in sentence_prev.words:
                                if word_prev.matched:
                                    matched_one = True
                                else:
                                    matched_all = False

                            if not matched_one:
                                # if there is not any already matched prev word, so set them all as matched
                                sentence_prev.matched = True
                                matched_curr = True
                                matched_sentences_prev.append(sentence_prev)

                                for word_prev in sentence_prev.words:
                                    word_prev.matched = True

                                # Add the sentence information to the paragraph.
                                if hash_curr in paragraph_curr.sentences:
                                    paragraph_curr.sentences[hash_curr].append(sentence_prev)
                                else:
                                    paragraph_curr.sentences.update({sentence_prev.hash_value: [sentence_prev]})
                                paragraph_curr.ordered_sentences.append(sentence_prev.hash_value)
                                break
                            elif matched_all:
                                # if all prev words in this sentence are already matched
                                sentence_prev.matched = True
                                matched_sentences_prev.append(sentence_prev)
                    if matched_curr:
                        break

                # Iterate over the hash table of sentences from old revisions.
                if not matched_curr:
                    for sentence_prev in self.sentences_ht.get(hash_curr, []):
                        if not sentence_prev.matched:
                            matched_one = False
                            matched_all = True
                            for word_prev in sentence_prev.words:
                                if word_prev.matched:
                                    matched_one = True
                                else:
                                    matched_all = False

                            if not matched_one:
                                # if there is not any already matched prev word, so set them all as matched
                                sentence_prev.matched = True
                                matched_curr = True
                                matched_sentences_prev.append(sentence_prev)

                                for word_prev in sentence_prev.words:
                                    word_prev.matched = True

                                # Add the sentence information to the paragraph.
                                if hash_curr in paragraph_curr.sentences:
                                    paragraph_curr.sentences[hash_curr].append(sentence_prev)
                                else:
                                    paragraph_curr.sentences.update({sentence_prev.hash_value: [sentence_prev]})
                                paragraph_curr.ordered_sentences.append(sentence_prev.hash_value)
                                break
                            elif matched_all:
                                # if all prev words in this sentence are already matched
                                sentence_prev.matched = True
                                matched_sentences_prev.append(sentence_prev)
                            elif _can_partially_restore_historical_sentence(
                                    sentence_prev.words, self.revision_prev.id):
                                # An exact historical sentence can return after a few of its
                                # generic token objects have been reused elsewhere. Restore the
                                # available identities and create new tokens only for occupied
                                # positions instead of rejecting the entire sentence.
                                sentence_reused = Sentence()
                                sentence_reused.hash_value = hash_curr
                                sentence_reused.value = sentence
                                for word_prev in sentence_prev.words:
                                    if word_prev.matched:
                                        word_curr = Word()
                                        word_curr.value = word_prev.value
                                        word_curr.token_id = self.token_id
                                        word_curr.origin_rev_id = self.revision_curr.id
                                        word_curr.last_rev_id = self.revision_curr.id
                                        sentence_reused.words.append(word_curr)
                                        self.token_id += 1
                                        self.revision_curr.original_adds += 1
                                        self.tokens.append(word_curr)
                                    else:
                                        word_prev.matched = True
                                        sentence_reused.words.append(word_prev)

                                sentence_prev.matched = True
                                matched_curr = True
                                matched_sentences_prev.append(sentence_prev)
                                if hash_curr in paragraph_curr.sentences:
                                    paragraph_curr.sentences[hash_curr].append(sentence_reused)
                                else:
                                    paragraph_curr.sentences.update({hash_curr: [sentence_reused]})
                                paragraph_curr.ordered_sentences.append(hash_curr)
                                restored_sentences_curr.append(sentence_reused)
                                break

                # If the sentence did not match,
                # then include in the container of unmatched sentences for further analysis.
                if not matched_curr:
                    sentence_curr = Sentence()
                    sentence_curr.value = sentence
                    sentence_curr.hash_value = hash_curr

                    if hash_curr in paragraph_curr.sentences:
                        paragraph_curr.sentences[hash_curr].append(sentence_curr)
                    else:
                        paragraph_curr.sentences.update({sentence_curr.hash_value: [sentence_curr]})
                    paragraph_curr.ordered_sentences.append(sentence_curr.hash_value)
                    unmatched_sentences_curr.append(sentence_curr)

        # Identify the unmatched sentences in the previous paragraph revision.
        sentence_duplicate_counts = {}
        for paragraph_prev in unmatched_paragraphs_prev:
            for sentence_prev_hash in paragraph_prev.ordered_sentences:
                if len(paragraph_prev.sentences[sentence_prev_hash]) > 1:
                    key = (id(paragraph_prev), sentence_prev_hash)
                    count = sentence_duplicate_counts.get(key, 0) + 1
                    sentence_duplicate_counts[key] = count
                    sentence_prev = paragraph_prev.sentences[sentence_prev_hash][count - 1]
                else:
                    sentence_prev = paragraph_prev.sentences[sentence_prev_hash][0]
                if not sentence_prev.matched:
                    unmatched_sentences_prev.append(sentence_prev)
                    # to reset 'matched words in analyse_words_in_sentences' of unmatched paragraphs and sentences
                    sentence_prev.matched = True
                    matched_sentences_prev.append(sentence_prev)

        return (unmatched_sentences_curr, restored_sentences_curr,
                unmatched_sentences_prev, matched_sentences_prev, total_sentences)

    def analyse_words_in_sentences(self, unmatched_sentences_curr, unmatched_sentences_prev, possible_vandalism):
        matched_words_prev = []
        unmatched_words_prev = []

        # Split sentences into words.
        text_prev = []
        for sentence_prev in unmatched_sentences_prev:
            for word_prev in sentence_prev.words:
                if not word_prev.matched:
                    text_prev.append(word_prev.value)
                    unmatched_words_prev.append(word_prev)

        # Build flat (sentence, token) slots so we can assign words during the
        # diff pass without re-scanning sentences or the diff list.
        curr_slots = []  # list of (sentence_curr, word_value)
        text_curr = []
        for sentence_curr in unmatched_sentences_curr:
            # split_into_tokens is already done in analyse_sentences_in_paragraphs
            words = sentence_curr.value.split(' ')
            text_curr.extend(words)
            sentence_curr.splitted.extend(words)
            for word in words:
                curr_slots.append((sentence_curr, word))

        # Edit consists of removing sentences, not adding new content.
        if not text_curr:
            return matched_words_prev, False

        # spam detection.
        if possible_vandalism:
            token_density = compute_avg_word_freq(text_curr)
            if token_density > TOKEN_DENSITY_LIMIT:
                return matched_words_prev, possible_vandalism
            else:
                possible_vandalism = False

        # Edit consists of adding new content, not changing/removing content
        if not text_prev:
            for sentence_curr, word in curr_slots:
                word_curr = Word()
                word_curr.value = word
                word_curr.token_id = self.token_id
                word_curr.origin_rev_id = self.revision_curr.id
                word_curr.last_rev_id = self.revision_curr.id
                sentence_curr.words.append(word_curr)
                self.token_id += 1
                self.revision_curr.original_adds += 1
                self.tokens.append(word_curr)
            return matched_words_prev, possible_vandalism

        structural_context = []
        structural_documents = []
        aligned_structural_documents = []

        def ensure_structural_documents():
            if not structural_documents:
                structural_documents.append(
                    _revision_structural_document_pair(
                        self.revision_prev, self.revision_curr,
                        unmatched_sentences_prev,
                        unmatched_sentences_curr,
                    )
                )
            return structural_documents[0]

        def get_structural_duplicates(candidates):
            prev_document, curr_document = ensure_structural_documents()
            if prev_document is None or curr_document is None:
                return None
            duplicated_prev = _duplicated_candidate_windows_in_document(
                prev_document, candidates,
            )
            if not duplicated_prev:
                return set()
            duplicated_curr = _duplicated_candidate_windows_in_document(
                curr_document, duplicated_prev,
            )
            return duplicated_curr

        def get_structural_documents():
            if aligned_structural_documents:
                return aligned_structural_documents[0]
            prev_document, curr_document = ensure_structural_documents()
            if prev_document is None or curr_document is None:
                return None

            prev_residual_by_article = {}
            prev_residual_flags = bytearray(len(prev_document.values))
            prev_residual_index = 0
            for sentence_prev in unmatched_sentences_prev:
                metadata = prev_document.sentence_ranges.get(
                    sentence_prev
                )
                if (metadata is None or
                        metadata[3] != len(sentence_prev.words)):
                    return None
                paragraph_index, sentence_index, sentence_start, _ = metadata
                for word_index, word_prev in enumerate(sentence_prev.words):
                    if word_prev.matched:
                        continue
                    article_index = sentence_start + word_index
                    if prev_document.values[article_index] != word_prev.value:
                        return None
                    prev_residual_by_article[article_index] = (
                        prev_residual_index,
                        (paragraph_index, sentence_index, word_index),
                    )
                    prev_residual_flags[article_index] = 1
                    prev_residual_index += 1
            if (prev_residual_index != len(text_prev) or
                    len(prev_residual_by_article) != len(text_prev)):
                return None

            curr_residual_by_article = {}
            curr_residual_flags = bytearray(len(curr_document.values))
            curr_residual_index = 0
            for sentence_curr in unmatched_sentences_curr:
                metadata = curr_document.sentence_ranges.get(
                    sentence_curr
                )
                if (metadata is None or
                        metadata[3] != len(sentence_curr.splitted)):
                    return None
                paragraph_index, sentence_index, sentence_start, _ = metadata
                for word_index, word in enumerate(sentence_curr.splitted):
                    article_index = sentence_start + word_index
                    if curr_document.values[article_index] != word:
                        return None
                    curr_residual_by_article[article_index] = (
                        curr_residual_index,
                        (paragraph_index, sentence_index, word_index),
                    )
                    curr_residual_flags[article_index] = 1
                    curr_residual_index += 1
            if (curr_residual_index != len(text_curr) or
                    len(curr_residual_by_article) != len(text_curr)):
                return None

            context = (
                prev_document, curr_document,
                prev_residual_by_article, curr_residual_by_article,
                prev_residual_flags, curr_residual_flags,
            )
            aligned_structural_documents.append(context)
            return context

        def get_structural_context():
            if structural_context:
                return structural_context[0]

            # Preserve the revision-local owner of every residual word only
            # after the residual gate has shown that structural matching may
            # contribute.  ``id`` is a lookup key for an ordinal path; it is
            # never itself matching evidence.
            full_prev_slots = _revision_token_slots(self.revision_prev)
            prev_slot_by_word = dict(
                (id(slot.word), slot) for slot in full_prev_slots
            )
            prev_sentence_paths = _sentence_occurrence_paths(
                self.revision_prev
            )
            curr_sentence_paths = _sentence_occurrence_paths(
                self.revision_curr
            )
            prev_match_slots = []
            prev_slots_valid = True

            prev_residual_index = 0
            for sentence_prev in unmatched_sentences_prev:
                sentence_path = prev_sentence_paths.get(id(sentence_prev))
                if sentence_path is None:
                    prev_slots_valid = False
                for word_index, word_prev in enumerate(sentence_prev.words):
                    if word_prev.matched:
                        continue
                    slot = prev_slot_by_word.get(id(word_prev))
                    if (slot is None or sentence_path is None or
                            slot.path != sentence_path + (word_index,)):
                        prev_slots_valid = False
                    else:
                        slot.residual_index = prev_residual_index
                        prev_match_slots.append(slot)
                    prev_residual_index += 1

            if prev_residual_index != len(text_prev):
                prev_slots_valid = False

            def align_current_slots(full_curr_slots):
                if not prev_slots_valid or full_curr_slots is None:
                    return None
                curr_slot_by_path = dict(
                    (slot.path, slot) for slot in full_curr_slots
                )
                if len(curr_slot_by_path) != len(full_curr_slots):
                    return None

                curr_match_slots = []
                curr_residual_index = 0
                for sentence_curr in unmatched_sentences_curr:
                    sentence_path = curr_sentence_paths.get(id(sentence_curr))
                    if sentence_path is None:
                        return None
                    for word_index, word in enumerate(sentence_curr.splitted):
                        slot = curr_slot_by_path.get(
                            sentence_path + (word_index,)
                        )
                        if slot is None or slot.value != word:
                            return None
                        slot.residual_index = curr_residual_index
                        curr_match_slots.append(slot)
                        curr_residual_index += 1

                if curr_residual_index != len(text_curr):
                    return None
                return (
                    prev_match_slots, curr_match_slots,
                    full_prev_slots, full_curr_slots,
                )

            # The current hierarchy is the normalized parse already produced
            # for this edit, so using it avoids parsing and tokenizing the full
            # wikitext a second time.  Retain the original tokenizer as a
            # correctness fallback for incomplete or inconsistent hierarchy
            # state rather than silently disabling structural matching.
            context = align_current_slots(
                _current_revision_token_slots(self.revision_curr)
            )
            if context is None:
                context = align_current_slots(_text_token_slots(self.text_curr))
            if context is None:
                context = (None, None, None, None)
            structural_context.append(context)
            return context

        full_texts = []

        def get_full_texts():
            if not full_texts:
                full_texts.append((
                    [word.value for word in iter_rev_tokens(self.revision_prev)],
                    split_into_tokens(self.text_curr),
                ))
            return full_texts[0]

        prev_for_curr, deleted_prev_indices = _match_word_sequences(
            text_prev,
            text_curr,
            get_full_texts=get_full_texts,
            prev_words=unmatched_words_prev,
            get_structural_context=get_structural_context,
            get_structural_duplicates=get_structural_duplicates,
            get_structural_documents=get_structural_documents,
        )
        for curr_index, prev_index in enumerate(prev_for_curr):
            sentence_curr, word = curr_slots[curr_index]
            if prev_index is None:
                word_curr = Word()
                word_curr.value = word
                word_curr.token_id = self.token_id
                word_curr.origin_rev_id = self.revision_curr.id
                word_curr.last_rev_id = self.revision_curr.id
                sentence_curr.words.append(word_curr)
                self.token_id += 1
                self.revision_curr.original_adds += 1
                self.tokens.append(word_curr)
            else:
                word_prev = unmatched_words_prev[prev_index]
                word_prev.matched = True
                sentence_curr.words.append(word_prev)
                matched_words_prev.append(word_prev)

        for prev_index in deleted_prev_indices:
            word_prev = unmatched_words_prev[prev_index]
            word_prev.matched = True
            word_prev.outbound.append(self.revision_curr.id)
            matched_words_prev.append(word_prev)

        return matched_words_prev, possible_vandalism
