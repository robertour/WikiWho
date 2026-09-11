#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

typedef struct {
    Py_ssize_t paragraph;
    Py_ssize_t start;
    Py_ssize_t end;
    int target;
} Range;

typedef struct {
    uint64_t hash;
    PyObject *pattern_values;
    Py_ssize_t pattern_start;
    Py_ssize_t prev_start;
    Py_ssize_t prev_paragraph;
    Py_ssize_t prev_local;
    Py_ssize_t curr_paragraph;
    Py_ssize_t curr_local;
    unsigned char prev_count;
    unsigned char curr_count;
    unsigned char prev_target;
    unsigned char curr_target;
    unsigned char occupied;
} Entry;

typedef struct {
    Entry *entries;
    size_t capacity;
    size_t used;
} Table;

typedef struct {
    uint64_t hash;
    PyObject *pattern;
    unsigned char count;
    unsigned char occupied;
} CandidateEntry;

typedef struct {
    CandidateEntry *entries;
    size_t capacity;
    size_t used;
} CandidateTable;

typedef struct {
    Py_ssize_t matches;
    Py_ssize_t runs;
    Py_ssize_t informative;
    Py_ssize_t displacement;
    unsigned char valid;
} AlignmentScore;

typedef struct {
    int type;
    PyObject *context;
    Py_ssize_t argument_index;
} ConstructFrame;

typedef struct {
    Py_ssize_t prev_paragraph;
    Py_ssize_t prev_start;
    Py_ssize_t prev_end;
    Py_ssize_t curr_paragraph;
    Py_ssize_t curr_start;
    Py_ssize_t curr_end;
} AnchorSegment;

typedef struct {
    uintptr_t *entries;
    size_t capacity;
    size_t used;
} PointerSet;

/* Call-scoped raw snapshots.  Token values remain owned by the previous
 * document list; only hierarchy identities and compact offsets are cached. */
typedef struct {
    PyObject *sentence;
    Py_ssize_t value_length;
    Py_ssize_t word_count;
} CachedSentence;

typedef struct {
    uintptr_t paragraph;
    Py_ssize_t value_start;
    Py_ssize_t sentence_start;
    Py_ssize_t sentence_count;
    Py_ssize_t word_start;
} CachedParagraph;

typedef struct {
    CachedParagraph *paragraphs;
    size_t paragraph_capacity;
    CachedSentence *sentences;
    size_t sentence_count;
    size_t sentence_capacity;
    PyObject **words;
    size_t word_count;
    size_t word_capacity;
    PyObject *values;
    PyObject *current_paragraphs;
} ParagraphCache;

/* Shared immutable labels used in contextual token keys.  Py_BuildValue's
 * ``s`` format creates a fresh Unicode object on every call; these labels are
 * constants, so allocating them per structural token adds traced allocations
 * without changing key equality. */
static PyObject *context_wikitext = NULL;
static PyObject *context_template = NULL;
static PyObject *context_link = NULL;
static PyObject *context_comment = NULL;
static PyObject *context_template_field = NULL;
static PyObject *context_template_arg = NULL;
static PyObject *attr_ordered_paragraphs = NULL;
static PyObject *attr_paragraphs = NULL;
static PyObject *attr_ordered_sentences = NULL;
static PyObject *attr_sentences = NULL;
static PyObject *attr_words = NULL;
static PyObject *attr_splitted = NULL;
static PyObject *attr_value = NULL;
static PyObject *token_symbols_text = NULL;
static PyObject *pipe_token = NULL;

static int
compare_ssize(Py_ssize_t left, Py_ssize_t right)
{
    return left < right ? -1 : (left > right ? 1 : 0);
}

static int
compare_anchor_segments(const void *left_pointer, const void *right_pointer)
{
    const AnchorSegment *left = (const AnchorSegment *)left_pointer;
    const AnchorSegment *right = (const AnchorSegment *)right_pointer;
    Py_ssize_t left_diagonal = left->prev_start - left->curr_start;
    Py_ssize_t right_diagonal = right->prev_start - right->curr_start;
    int result;
    result = compare_ssize(left->prev_paragraph, right->prev_paragraph);
    if (result) return result;
    result = compare_ssize(left->curr_paragraph, right->curr_paragraph);
    if (result) return result;
    result = compare_ssize(left_diagonal, right_diagonal);
    if (result) return result;
    result = compare_ssize(left->prev_start, right->prev_start);
    if (result) return result;
    result = compare_ssize(left->prev_end, right->prev_end);
    if (result) return result;
    result = compare_ssize(left->curr_start, right->curr_start);
    if (result) return result;
    return compare_ssize(left->curr_end, right->curr_end);
}

static int
append_anchor_segment(PyObject *output, const AnchorSegment *segment)
{
    PyObject *record = Py_BuildValue(
        "(nnnnnn)", segment->prev_paragraph, segment->prev_start,
        segment->prev_end, segment->curr_paragraph, segment->curr_start,
        segment->curr_end
    );
    int status;
    if (record == NULL) {
        return -1;
    }
    status = PyList_Append(output, record);
    Py_DECREF(record);
    return status;
}

#define STRUCTURAL_ROLLING_BASE UINT64_C(0x9e3779b185ebca87)

static uint64_t
structural_token_hash(Py_hash_t hash)
{
    uint64_t token = (uint64_t)hash;
    token ^= token >> 30;
    token *= UINT64_C(0xbf58476d1ce4e5b9);
    token ^= token >> 27;
    token *= UINT64_C(0x94d049bb133111eb);
    token ^= token >> 31;
    return token + UINT64_C(0x517cc1b727220a95);
}

static int
copy_rolling_hash_prefix(PyObject *keys, uint64_t **result)
{
    Py_ssize_t length = PyList_GET_SIZE(keys);
    uint64_t *prefix = (uint64_t *)malloc(
        (size_t)(length + 1) * sizeof(uint64_t)
    );
    Py_ssize_t index;
    if (prefix == NULL) {
        PyErr_NoMemory();
        return -1;
    }
    prefix[0] = 0;
    for (index = 0; index < length; index++) {
        Py_hash_t hash = PyObject_Hash(PyList_GET_ITEM(keys, index));
        if (hash == -1 && PyErr_Occurred()) {
            free(prefix);
            return -1;
        }
        prefix[index + 1] =
            prefix[index] * STRUCTURAL_ROLLING_BASE +
            structural_token_hash(hash);
    }
    *result = prefix;
    return 0;
}

static uint64_t
rolling_window_hash(const uint64_t *prefix, Py_ssize_t start,
                    Py_ssize_t size, uint64_t base_power)
{
    return prefix[start + size] - prefix[start] * base_power;
}

static int
windows_equal(PyObject *left, Py_ssize_t left_start,
              PyObject *right, Py_ssize_t right_start, Py_ssize_t size)
{
    Py_ssize_t offset;
    if (left == right && left_start == right_start) {
        return 1;
    }
    for (offset = 0; offset < size; offset++) {
        PyObject *left_item = PyList_GET_ITEM(left, left_start + offset);
        PyObject *right_item = PyList_GET_ITEM(right, right_start + offset);
        if (left_item == right_item) {
            continue;
        }
        int equal = PyObject_RichCompareBool(
            left_item, right_item, Py_EQ
        );
        if (equal <= 0) {
            return equal;
        }
    }
    return 1;
}

static int
table_init(Table *table, size_t expected)
{
    size_t capacity = 16;
    while (capacity < expected * 2 && capacity <= SIZE_MAX / 2) {
        capacity *= 2;
    }
    table->entries = (Entry *)calloc(capacity, sizeof(Entry));
    if (table->entries == NULL) {
        PyErr_NoMemory();
        return -1;
    }
    table->capacity = capacity;
    table->used = 0;
    return 0;
}

static void
table_clear(Table *table)
{
    free(table->entries);
    table->entries = NULL;
    table->capacity = 0;
    table->used = 0;
}

static size_t
pointer_hash(uintptr_t pointer)
{
    uint64_t value = (uint64_t)pointer;
    value ^= value >> 30;
    value *= UINT64_C(0xbf58476d1ce4e5b9);
    value ^= value >> 27;
    value *= UINT64_C(0x94d049bb133111eb);
    value ^= value >> 31;
    return (size_t)value;
}

static int
pointer_set_init(PointerSet *set, size_t expected)
{
    size_t capacity = 16;
    while (capacity < expected * 2 && capacity <= SIZE_MAX / 2) {
        capacity *= 2;
    }
    set->entries = (uintptr_t *)calloc(capacity, sizeof(uintptr_t));
    if (set->entries == NULL) {
        PyErr_NoMemory();
        return -1;
    }
    set->capacity = capacity;
    set->used = 0;
    return 0;
}

static void
pointer_set_clear(PointerSet *set)
{
    free(set->entries);
    set->entries = NULL;
    set->capacity = 0;
    set->used = 0;
}

static int
pointer_set_resize(PointerSet *set)
{
    size_t old_capacity = set->capacity;
    uintptr_t *old_entries = set->entries;
    size_t new_capacity;
    uintptr_t *new_entries;
    size_t index;
    if (old_capacity > SIZE_MAX / 2) {
        PyErr_NoMemory();
        return -1;
    }
    new_capacity = old_capacity * 2;
    new_entries = (uintptr_t *)calloc(
        new_capacity, sizeof(uintptr_t)
    );
    if (new_entries == NULL) {
        PyErr_NoMemory();
        return -1;
    }
    for (index = 0; index < old_capacity; index++) {
        uintptr_t pointer = old_entries[index];
        size_t slot;
        if (!pointer) {
            continue;
        }
        slot = pointer_hash(pointer) & (new_capacity - 1);
        while (new_entries[slot]) {
            slot = (slot + 1) & (new_capacity - 1);
        }
        new_entries[slot] = pointer;
    }
    free(old_entries);
    set->entries = new_entries;
    set->capacity = new_capacity;
    return 0;
}

/* Return 1 for a new identity, 0 for a duplicate, and -1 on failure. */
static int
pointer_set_add(PointerSet *set, PyObject *object)
{
    uintptr_t pointer = (uintptr_t)object;
    size_t slot;
    if ((set->used + 1) * 10 >= set->capacity * 7) {
        if (pointer_set_resize(set) < 0) {
            return -1;
        }
    }
    slot = pointer_hash(pointer) & (set->capacity - 1);
    while (set->entries[slot]) {
        if (set->entries[slot] == pointer) {
            return 0;
        }
        slot = (slot + 1) & (set->capacity - 1);
    }
    set->entries[slot] = pointer;
    set->used++;
    return 1;
}

static int
paragraph_cache_init(ParagraphCache *cache, PyObject *current,
                     size_t expected_paragraphs)
{
    size_t capacity = 16;
    size_t target;
    if (expected_paragraphs > (SIZE_MAX - 1) / 2) {
        PyErr_NoMemory();
        return -1;
    }
    target = expected_paragraphs + expected_paragraphs / 2 + 1;
    while (capacity < target) {
        if (capacity > SIZE_MAX / 2) {
            PyErr_NoMemory();
            return -1;
        }
        capacity *= 2;
    }
    cache->paragraphs = (CachedParagraph *)calloc(
        capacity, sizeof(CachedParagraph)
    );
    if (cache->paragraphs == NULL) {
        PyErr_NoMemory();
        return -1;
    }
    cache->paragraph_capacity = capacity;
    cache->current_paragraphs = PyObject_GetAttr(
        current, attr_paragraphs
    );
    if (cache->current_paragraphs == NULL) {
        return -1;
    }
    return 0;
}

static void
paragraph_cache_clear(ParagraphCache *cache)
{
    size_t index;
    for (index = 0; index < cache->sentence_count; index++) {
        Py_DECREF(cache->sentences[index].sentence);
    }
    for (index = 0; index < cache->word_count; index++) {
        Py_DECREF(cache->words[index]);
    }
    free(cache->paragraphs);
    free(cache->sentences);
    free(cache->words);
    Py_XDECREF(cache->current_paragraphs);
    memset(cache, 0, sizeof(*cache));
}

static int
paragraph_cache_may_reuse(ParagraphCache *cache, PyObject *paragraph_hash)
{
    if (!PyDict_Check(cache->current_paragraphs)) {
        return 0;
    }
    return PyDict_Contains(cache->current_paragraphs, paragraph_hash);
}

static int
paragraph_cache_reserve_sentences(ParagraphCache *cache, size_t needed)
{
    size_t capacity = cache->sentence_capacity ?
        cache->sentence_capacity : 256;
    CachedSentence *sentences;
    while (capacity < needed) {
        if (capacity > SIZE_MAX / 2) {
            PyErr_NoMemory();
            return -1;
        }
        capacity *= 2;
    }
    if (capacity == cache->sentence_capacity) {
        return 0;
    }
    if (capacity > SIZE_MAX / sizeof(CachedSentence)) {
        PyErr_NoMemory();
        return -1;
    }
    sentences = (CachedSentence *)realloc(
        cache->sentences, capacity * sizeof(CachedSentence)
    );
    if (sentences == NULL) {
        PyErr_NoMemory();
        return -1;
    }
    cache->sentences = sentences;
    cache->sentence_capacity = capacity;
    return 0;
}

static int
paragraph_cache_reserve_words(ParagraphCache *cache, size_t needed)
{
    size_t capacity = cache->word_capacity ? cache->word_capacity : 1024;
    PyObject **words;
    while (capacity < needed) {
        if (capacity > SIZE_MAX / 2) {
            PyErr_NoMemory();
            return -1;
        }
        capacity *= 2;
    }
    if (capacity == cache->word_capacity) {
        return 0;
    }
    if (capacity > SIZE_MAX / sizeof(PyObject *)) {
        PyErr_NoMemory();
        return -1;
    }
    words = (PyObject **)realloc(
        cache->words, capacity * sizeof(PyObject *)
    );
    if (words == NULL) {
        PyErr_NoMemory();
        return -1;
    }
    cache->words = words;
    cache->word_capacity = capacity;
    return 0;
}

static int
paragraph_cache_append_word(ParagraphCache *cache, PyObject *word)
{
    if (paragraph_cache_reserve_words(
            cache, cache->word_count + 1) < 0) {
        return -1;
    }
    Py_INCREF(word);
    cache->words[cache->word_count++] = word;
    return 0;
}

static int
paragraph_cache_append_sentence(
        ParagraphCache *cache, PyObject *sentence,
        Py_ssize_t value_length, Py_ssize_t word_count)
{
    CachedSentence *cached;
    if (paragraph_cache_reserve_sentences(
            cache, cache->sentence_count + 1) < 0) {
        return -1;
    }
    cached = &cache->sentences[cache->sentence_count++];
    Py_INCREF(sentence);
    cached->sentence = sentence;
    cached->value_length = value_length;
    cached->word_count = word_count;
    return 0;
}

static int
paragraph_cache_insert(
        ParagraphCache *cache, PyObject *paragraph,
        Py_ssize_t value_start, Py_ssize_t sentence_start,
        Py_ssize_t sentence_count, Py_ssize_t word_start)
{
    uintptr_t pointer = (uintptr_t)paragraph;
    size_t slot = pointer_hash(pointer) &
        (cache->paragraph_capacity - 1);
    CachedParagraph *cached;
    while (cache->paragraphs[slot].paragraph) {
        if (cache->paragraphs[slot].paragraph == pointer) {
            return 0;
        }
        slot = (slot + 1) & (cache->paragraph_capacity - 1);
    }
    cached = &cache->paragraphs[slot];
    cached->paragraph = pointer;
    cached->value_start = value_start;
    cached->sentence_start = sentence_start;
    cached->sentence_count = sentence_count;
    cached->word_start = word_start;
    return 1;
}

static CachedParagraph *
paragraph_cache_lookup(ParagraphCache *cache, PyObject *paragraph)
{
    uintptr_t pointer = (uintptr_t)paragraph;
    size_t slot = pointer_hash(pointer) &
        (cache->paragraph_capacity - 1);
    while (cache->paragraphs[slot].paragraph) {
        if (cache->paragraphs[slot].paragraph == pointer) {
            return &cache->paragraphs[slot];
        }
        slot = (slot + 1) & (cache->paragraph_capacity - 1);
    }
    return NULL;
}

static int
table_resize(Table *table)
{
    size_t old_capacity = table->capacity;
    Entry *old_entries = table->entries;
    size_t new_capacity;
    Entry *new_entries;
    size_t index;

    if (old_capacity > SIZE_MAX / 2) {
        PyErr_NoMemory();
        return -1;
    }
    new_capacity = old_capacity * 2;
    new_entries = (Entry *)calloc(new_capacity, sizeof(Entry));
    if (new_entries == NULL) {
        PyErr_NoMemory();
        return -1;
    }
    for (index = 0; index < old_capacity; index++) {
        Entry entry = old_entries[index];
        size_t slot;
        if (!entry.occupied) {
            continue;
        }
        slot = (size_t)entry.hash & (new_capacity - 1);
        while (new_entries[slot].occupied) {
            slot = (slot + 1) & (new_capacity - 1);
        }
        new_entries[slot] = entry;
    }
    free(old_entries);
    table->entries = new_entries;
    table->capacity = new_capacity;
    return 0;
}

static int
table_add_previous(Table *table, PyObject *prev_keys,
                   Py_ssize_t start, Py_ssize_t size,
                   Py_ssize_t paragraph, Py_ssize_t local, int target,
                   uint64_t hash)
{
    size_t slot;
    if ((table->used + 1) * 10 >= table->capacity * 7) {
        if (table_resize(table) < 0) {
            return -1;
        }
    }
    slot = (size_t)hash & (table->capacity - 1);
    while (table->entries[slot].occupied) {
        Entry *entry = &table->entries[slot];
        if (entry->hash == hash) {
            int equal = windows_equal(
                entry->pattern_values, entry->pattern_start,
                prev_keys, start, size
            );
            if (equal < 0) {
                return -1;
            }
            if (equal) {
                if (entry->prev_count < 2) {
                    entry->prev_count++;
                }
                if (target) {
                    entry->prev_target = 1;
                }
                return 0;
            }
        }
        slot = (slot + 1) & (table->capacity - 1);
    }
    table->entries[slot].occupied = 1;
    table->entries[slot].hash = hash;
    table->entries[slot].pattern_values = prev_keys;
    table->entries[slot].pattern_start = start;
    table->entries[slot].prev_start = start;
    table->entries[slot].prev_paragraph = paragraph;
    table->entries[slot].prev_local = local;
    table->entries[slot].prev_count = 1;
    table->entries[slot].prev_target = target ? 1 : 0;
    table->used++;
    return 0;
}

static int
table_add_current(Table *table, PyObject *prev_keys, PyObject *curr_keys,
                  Py_ssize_t start, Py_ssize_t size,
                  Py_ssize_t paragraph, Py_ssize_t local, int target,
                  uint64_t hash)
{
    size_t slot = (size_t)hash & (table->capacity - 1);
    while (table->entries[slot].occupied) {
        Entry *entry = &table->entries[slot];
        if (entry->hash == hash) {
            int equal = windows_equal(
                entry->pattern_values, entry->pattern_start,
                curr_keys, start, size
            );
            if (equal < 0) {
                return -1;
            }
            if (equal) {
                if (entry->prev_count == 1) {
                    if (entry->curr_count == 0) {
                        entry->curr_paragraph = paragraph;
                        entry->curr_local = local;
                    }
                    if (entry->curr_count < 2) {
                        entry->curr_count++;
                    }
                    if (target) {
                        entry->curr_target = 1;
                    }
                }
                return 0;
            }
        }
        slot = (slot + 1) & (table->capacity - 1);
    }
    return 0;
}

static int
table_add_target_candidate(Table *table, PyObject *keys,
                           Py_ssize_t start, Py_ssize_t size,
                           uint64_t hash)
{
    size_t slot;
    if ((table->used + 1) * 10 >= table->capacity * 7) {
        if (table_resize(table) < 0) {
            return -1;
        }
    }
    slot = (size_t)hash & (table->capacity - 1);
    while (table->entries[slot].occupied) {
        Entry *entry = &table->entries[slot];
        if (entry->hash == hash) {
            int equal = windows_equal(
                entry->pattern_values, entry->pattern_start,
                keys, start, size
            );
            if (equal < 0) {
                return -1;
            }
            if (equal) {
                return 0;
            }
        }
        slot = (slot + 1) & (table->capacity - 1);
    }
    table->entries[slot].occupied = 1;
    table->entries[slot].hash = hash;
    table->entries[slot].pattern_values = keys;
    table->entries[slot].pattern_start = start;
    table->used++;
    return 0;
}

static int
table_count_target_previous(Table *table, PyObject *keys,
                            Py_ssize_t start, Py_ssize_t size,
                            Py_ssize_t paragraph, Py_ssize_t local,
                            uint64_t hash)
{
    size_t slot = (size_t)hash & (table->capacity - 1);
    while (table->entries[slot].occupied) {
        Entry *entry = &table->entries[slot];
        if (entry->hash == hash) {
            int equal = windows_equal(
                entry->pattern_values, entry->pattern_start,
                keys, start, size
            );
            if (equal < 0) {
                return -1;
            }
            if (equal) {
                if (entry->prev_count == 0) {
                    entry->prev_start = start;
                    entry->prev_paragraph = paragraph;
                    entry->prev_local = local;
                }
                if (entry->prev_count < 2) {
                    entry->prev_count++;
                }
                return 0;
            }
        }
        slot = (slot + 1) & (table->capacity - 1);
    }
    return 0;
}

static int
table_count_target_current(Table *table, PyObject *keys,
                           Py_ssize_t start, Py_ssize_t size,
                           Py_ssize_t paragraph, Py_ssize_t local,
                           uint64_t hash)
{
    size_t slot = (size_t)hash & (table->capacity - 1);
    while (table->entries[slot].occupied) {
        Entry *entry = &table->entries[slot];
        if (entry->hash == hash) {
            int equal = windows_equal(
                entry->pattern_values, entry->pattern_start,
                keys, start, size
            );
            if (equal < 0) {
                return -1;
            }
            if (equal) {
                if (entry->curr_count == 0) {
                    entry->curr_paragraph = paragraph;
                    entry->curr_local = local;
                }
                if (entry->curr_count < 2) {
                    entry->curr_count++;
                }
                return 0;
            }
        }
        slot = (slot + 1) & (table->capacity - 1);
    }
    return 0;
}

static int
copy_prefix(PyObject *prefix, Py_ssize_t expected, Py_ssize_t **result)
{
    Py_ssize_t *values;
    Py_ssize_t index;
    values = (Py_ssize_t *)malloc(
        (size_t)(expected + 1) * sizeof(Py_ssize_t)
    );
    if (values == NULL) {
        PyErr_NoMemory();
        return -1;
    }

    if (PyList_Check(prefix)) {
        if (PyList_GET_SIZE(prefix) != expected + 1) {
            free(values);
            PyErr_SetString(PyExc_ValueError, "invalid informative prefix");
            return -1;
        }
        for (index = 0; index <= expected; index++) {
            values[index] = PyLong_AsSsize_t(
                PyList_GET_ITEM(prefix, index)
            );
            if (values[index] == -1 && PyErr_Occurred()) {
                free(values);
                return -1;
            }
        }
    } else {
        Py_buffer view;
        if (PyObject_GetBuffer(prefix, &view, PyBUF_CONTIG_RO) < 0) {
            free(values);
            return -1;
        }
        if (view.len != (expected + 1) * (Py_ssize_t)sizeof(uint64_t)) {
            PyBuffer_Release(&view);
            free(values);
            PyErr_SetString(PyExc_ValueError, "invalid informative prefix");
            return -1;
        }
        for (index = 0; index <= expected; index++) {
            uint64_t item;
            memcpy(
                &item,
                (const unsigned char *)view.buf +
                    (size_t)index * sizeof(uint64_t),
                sizeof(uint64_t)
            );
            if (item > (uint64_t)PY_SSIZE_T_MAX) {
                PyBuffer_Release(&view);
                free(values);
                PyErr_SetString(PyExc_OverflowError,
                                "informative prefix is too large");
                return -1;
            }
            values[index] = (Py_ssize_t)item;
        }
        PyBuffer_Release(&view);
    }
    *result = values;
    return 0;
}

static int
copy_ranges(PyObject *ranges, PyObject *targets, Range **result,
            Py_ssize_t *result_count)
{
    Py_ssize_t count;
    Range *copied;
    Py_ssize_t position = 0;
    Py_ssize_t iteration = 0;
    PyObject *key;
    PyObject *value;

    if (!PyDict_Check(ranges) || !PySet_Check(targets)) {
        PyErr_SetString(PyExc_TypeError, "ranges must be dict and targets set");
        return -1;
    }
    count = PyDict_Size(ranges);
    copied = (Range *)malloc(
        (size_t)(count > 0 ? count : 1) * sizeof(Range)
    );
    if (copied == NULL) {
        PyErr_NoMemory();
        return -1;
    }
    while (PyDict_Next(ranges, &iteration, &key, &value)) {
        int target;
        if (!PyTuple_Check(value) || PyTuple_GET_SIZE(value) != 2) {
            free(copied);
            PyErr_SetString(PyExc_ValueError, "invalid paragraph range");
            return -1;
        }
        copied[position].paragraph = PyLong_AsSsize_t(key);
        copied[position].start = PyLong_AsSsize_t(PyTuple_GET_ITEM(value, 0));
        copied[position].end = PyLong_AsSsize_t(PyTuple_GET_ITEM(value, 1));
        if (PyErr_Occurred()) {
            free(copied);
            return -1;
        }
        target = PySet_Contains(targets, key);
        if (target < 0) {
            free(copied);
            return -1;
        }
        copied[position].target = target;
        position++;
    }
    *result = copied;
    *result_count = count;
    return 0;
}

static void
candidate_table_clear(CandidateTable *table)
{
    size_t index;
    if (table->entries != NULL) {
        for (index = 0; index < table->capacity; index++) {
            if (table->entries[index].occupied) {
                Py_DECREF(table->entries[index].pattern);
            }
        }
    }
    free(table->entries);
    table->entries = NULL;
    table->capacity = 0;
    table->used = 0;
}

static int
candidate_pattern_equal(PyObject *pattern, PyObject *values,
                        Py_ssize_t start)
{
    Py_ssize_t offset;
    Py_ssize_t size = PyTuple_GET_SIZE(pattern);
    for (offset = 0; offset < size; offset++) {
        int equal = PyObject_RichCompareBool(
            PyTuple_GET_ITEM(pattern, offset),
            PyList_GET_ITEM(values, start + offset),
            Py_EQ
        );
        if (equal <= 0) {
            return equal;
        }
    }
    return 1;
}

static PyObject *
duplicated_candidate_windows(PyObject *self, PyObject *args)
{
    PyObject *values;
    PyObject *ranges_obj;
    PyObject *candidates;
    PyObject *iterator = NULL;
    PyObject *pattern;
    PyObject *output = NULL;
    uint64_t *hash_prefix = NULL;
    uint64_t base_powers[64];
    CandidateTable table = {NULL, 0, 0};
    unsigned char sizes[64] = {0};
    Py_ssize_t candidate_count;
    size_t capacity = 16;
    Py_ssize_t iteration = 0;
    PyObject *paragraph_key;
    PyObject *range_obj;

    if (!PyArg_ParseTuple(
            args, "OOO:duplicated_candidate_windows",
            &values, &ranges_obj, &candidates)) {
        return NULL;
    }
    if (!PyList_Check(values) || !PyDict_Check(ranges_obj) ||
            !PySet_Check(candidates)) {
        PyErr_SetString(
            PyExc_TypeError, "values list, ranges dict, and candidates set required"
        );
        return NULL;
    }
    candidate_count = PySet_Size(candidates);
    if (candidate_count == 0) {
        return PySet_New(NULL);
    }
    while (capacity < (size_t)candidate_count * 2 &&
            capacity <= SIZE_MAX / 2) {
        capacity *= 2;
    }
    table.entries = (CandidateEntry *)calloc(
        capacity, sizeof(CandidateEntry)
    );
    if (table.entries == NULL) {
        PyErr_NoMemory();
        goto error;
    }
    table.capacity = capacity;
    base_powers[0] = 1;
    for (iteration = 1; iteration < 64; iteration++) {
        base_powers[iteration] =
            base_powers[iteration - 1] * STRUCTURAL_ROLLING_BASE;
    }
    iteration = 0;

    iterator = PyObject_GetIter(candidates);
    if (iterator == NULL) {
        goto error;
    }
    while ((pattern = PyIter_Next(iterator)) != NULL) {
        Py_ssize_t size;
        Py_ssize_t offset;
        uint64_t hash;
        size_t slot;
        if (!PyTuple_Check(pattern)) {
            Py_DECREF(pattern);
            PyErr_SetString(PyExc_TypeError, "candidate must be a tuple");
            goto error;
        }
        size = PyTuple_GET_SIZE(pattern);
        if (size <= 0 || size >= (Py_ssize_t)sizeof(sizes)) {
            Py_DECREF(pattern);
            PyErr_SetString(PyExc_ValueError, "candidate width out of range");
            goto error;
        }
        sizes[size] = 1;
        hash = 0;
        for (offset = 0; offset < size; offset++) {
            Py_hash_t token_hash = PyObject_Hash(
                PyTuple_GET_ITEM(pattern, offset)
            );
            if (token_hash == -1 && PyErr_Occurred()) {
                Py_DECREF(pattern);
                goto error;
            }
            hash = hash * STRUCTURAL_ROLLING_BASE +
                   structural_token_hash(token_hash);
        }
        slot = (size_t)hash & (capacity - 1);
        while (table.entries[slot].occupied) {
            slot = (slot + 1) & (capacity - 1);
        }
        table.entries[slot].occupied = 1;
        table.entries[slot].hash = hash;
        table.entries[slot].pattern = pattern;
        table.used++;
    }
    if (PyErr_Occurred()) {
        goto error;
    }
    Py_CLEAR(iterator);
    if (copy_rolling_hash_prefix(values, &hash_prefix) < 0) {
        goto error;
    }

    while (PyDict_Next(
            ranges_obj, &iteration, &paragraph_key, &range_obj)) {
        Py_ssize_t paragraph_start;
        Py_ssize_t paragraph_end;
        Py_ssize_t size;
        if (!PyTuple_Check(range_obj) || PyTuple_GET_SIZE(range_obj) != 2) {
            PyErr_SetString(PyExc_ValueError, "invalid paragraph range");
            goto error;
        }
        paragraph_start = PyLong_AsSsize_t(PyTuple_GET_ITEM(range_obj, 0));
        paragraph_end = PyLong_AsSsize_t(PyTuple_GET_ITEM(range_obj, 1));
        if (PyErr_Occurred()) {
            goto error;
        }
        for (size = 1; size < (Py_ssize_t)sizeof(sizes); size++) {
            Py_ssize_t start;
            if (!sizes[size] || size > paragraph_end - paragraph_start) {
                continue;
            }
            for (start = paragraph_start;
                    start + size <= paragraph_end; start++) {
                uint64_t hash = rolling_window_hash(
                    hash_prefix, start, size, base_powers[size]
                );
                size_t slot = (size_t)hash & (capacity - 1);
                while (table.entries[slot].occupied) {
                    CandidateEntry *entry = &table.entries[slot];
                    if (entry->hash == hash && entry->count < 2) {
                        int equal = candidate_pattern_equal(
                            entry->pattern, values, start
                        );
                        if (equal < 0) {
                            goto error;
                        }
                        if (equal) {
                            entry->count++;
                            break;
                        }
                    }
                    slot = (slot + 1) & (capacity - 1);
                }
            }
        }
    }

    output = PySet_New(NULL);
    if (output == NULL) {
        goto error;
    }
    {
        size_t index;
        for (index = 0; index < table.capacity; index++) {
            CandidateEntry *entry = &table.entries[index];
            if (entry->occupied && entry->count >= 2 &&
                    PySet_Add(output, entry->pattern) < 0) {
                goto error;
            }
        }
    }
    free(hash_prefix);
    candidate_table_clear(&table);
    return output;

error:
    Py_XDECREF(iterator);
    Py_XDECREF(output);
    free(hash_prefix);
    candidate_table_clear(&table);
    return NULL;
}

static int
score_compare(const AlignmentScore *left, const AlignmentScore *right)
{
    if (left->matches != right->matches) {
        return left->matches > right->matches ? 1 : -1;
    }
    if (left->runs != right->runs) {
        return left->runs > right->runs ? 1 : -1;
    }
    if (left->informative != right->informative) {
        return left->informative > right->informative ? 1 : -1;
    }
    if (left->displacement != right->displacement) {
        return left->displacement > right->displacement ? 1 : -1;
    }
    return 0;
}

static const AlignmentScore *
best_alignment_state(const AlignmentScore *matched,
                     const AlignmentScore *gapped,
                     unsigned char *state)
{
    if (!matched->valid) {
        *state = 2;
        return gapped->valid ? gapped : NULL;
    }
    if (!gapped->valid || score_compare(matched, gapped) > 0) {
        *state = 1;
        return matched;
    }
    *state = 2;
    return gapped;
}

static int
native_informative_token(PyObject *token, PyObject *structural_tokens)
{
    Py_ssize_t length;
    Py_ssize_t index;
    int structural;
    if (!PyUnicode_Check(token)) {
        return 0;
    }
    length = PyUnicode_GET_LENGTH(token);
    if (length == 0) {
        return 0;
    }
    structural = PySet_Contains(structural_tokens, token);
    if (structural < 0) {
        return -1;
    }
    if (structural) {
        return 0;
    }
    for (index = 0; index < length; index++) {
        if (Py_UNICODE_ISALNUM(PyUnicode_READ_CHAR(token, index))) {
            return 1;
        }
    }
    return 0;
}

static PyObject *
lcs_token_pairs(PyObject *self, PyObject *args)
{
    PyObject *prev_keys;
    PyObject *curr_keys;
    PyObject *prev_values;
    PyObject *curr_values;
    PyObject *structural_tokens;
    Py_ssize_t max_cells;
    Py_ssize_t prev_len;
    Py_ssize_t curr_len;
    Py_ssize_t columns;
    size_t cell_count;
    AlignmentScore *matched = NULL;
    AlignmentScore *gapped = NULL;
    unsigned char *matched_back = NULL;
    unsigned char *gapped_back = NULL;
    PyObject *pairs = NULL;
    Py_ssize_t prev_count;
    Py_ssize_t curr_count;
    unsigned char state;

    if (!PyArg_ParseTuple(
            args, "OOOOOn:lcs_token_pairs", &prev_keys, &curr_keys,
            &prev_values, &curr_values, &structural_tokens, &max_cells)) {
        return NULL;
    }
    if (!PyList_Check(prev_keys) || !PyList_Check(curr_keys) ||
            !PyList_Check(prev_values) || !PyList_Check(curr_values) ||
            !PyAnySet_Check(structural_tokens)) {
        PyErr_SetString(PyExc_TypeError, "alignment inputs must be lists and set");
        return NULL;
    }
    prev_len = PyList_GET_SIZE(prev_keys);
    curr_len = PyList_GET_SIZE(curr_keys);
    if (PyList_GET_SIZE(prev_values) != prev_len ||
            PyList_GET_SIZE(curr_values) != curr_len) {
        PyErr_SetString(PyExc_ValueError, "alignment values must match keys");
        return NULL;
    }
    if (prev_len == 0 || curr_len == 0 ||
            (prev_len > 0 && curr_len > max_cells / prev_len)) {
        return PyList_New(0);
    }
    columns = curr_len + 1;
    cell_count = (size_t)(prev_len + 1) * (size_t)columns;
    matched = (AlignmentScore *)calloc(cell_count, sizeof(AlignmentScore));
    gapped = (AlignmentScore *)calloc(cell_count, sizeof(AlignmentScore));
    matched_back = (unsigned char *)calloc(cell_count, 1);
    gapped_back = (unsigned char *)calloc(cell_count, 1);
    if (matched == NULL || gapped == NULL || matched_back == NULL ||
            gapped_back == NULL) {
        PyErr_NoMemory();
        goto error;
    }
    gapped[0].valid = 1;

    for (prev_count = 0; prev_count <= prev_len; prev_count++) {
        for (curr_count = 0; curr_count <= curr_len; curr_count++) {
            size_t cell = (size_t)prev_count * (size_t)columns +
                          (size_t)curr_count;
            if (prev_count || curr_count) {
                const AlignmentScore *chosen = NULL;
                unsigned char chosen_state = 0;
                unsigned char chosen_back = 0;
                if (prev_count) {
                    size_t prior_cell = cell - (size_t)columns;
                    const AlignmentScore *score = best_alignment_state(
                        &matched[prior_cell], &gapped[prior_cell],
                        &chosen_state
                    );
                    if (score != NULL) {
                        chosen = score;
                        chosen_back = chosen_state == 1 ? 1 : 2;
                    }
                }
                if (curr_count) {
                    size_t prior_cell = cell - 1;
                    unsigned char candidate_state;
                    const AlignmentScore *score = best_alignment_state(
                        &matched[prior_cell], &gapped[prior_cell],
                        &candidate_state
                    );
                    if (score != NULL &&
                            (chosen == NULL || score_compare(score, chosen) > 0)) {
                        chosen = score;
                        chosen_back = candidate_state == 1 ? 3 : 4;
                    }
                }
                if (chosen != NULL) {
                    gapped[cell] = *chosen;
                    gapped_back[cell] = chosen_back;
                }
            }

            if (prev_count && curr_count) {
                int equal = PyObject_RichCompareBool(
                    PyList_GET_ITEM(prev_keys, prev_count - 1),
                    PyList_GET_ITEM(curr_keys, curr_count - 1),
                    Py_EQ
                );
                if (equal < 0) {
                    goto error;
                }
                if (equal) {
                    size_t prior_cell = cell - (size_t)columns - 1;
                    const AlignmentScore *chosen = NULL;
                    unsigned char chosen_state = 0;
                    int prev_info = native_informative_token(
                        PyList_GET_ITEM(prev_values, prev_count - 1),
                        structural_tokens
                    );
                    int curr_info = native_informative_token(
                        PyList_GET_ITEM(curr_values, curr_count - 1),
                        structural_tokens
                    );
                    Py_ssize_t prev_index = prev_count - 1;
                    Py_ssize_t curr_index = curr_count - 1;
                    Py_ssize_t prev_scale = curr_len > 1 ? curr_len - 1 : 1;
                    Py_ssize_t curr_scale = prev_len > 1 ? prev_len - 1 : 1;
                    Py_ssize_t delta;
                    Py_ssize_t displacement;
                    AlignmentScore candidate;
                    if (prev_info < 0 || curr_info < 0) {
                        goto error;
                    }
                    delta = prev_index * prev_scale - curr_index * curr_scale;
                    displacement = delta < 0 ? -delta : delta;
                    if (matched[prior_cell].valid) {
                        candidate = matched[prior_cell];
                        candidate.matches++;
                        candidate.informative += prev_info && curr_info;
                        candidate.displacement -= displacement;
                        matched[cell] = candidate;
                        matched[cell].valid = 1;
                        chosen = &matched[cell];
                        chosen_state = 1;
                    }
                    if (gapped[prior_cell].valid) {
                        candidate = gapped[prior_cell];
                        candidate.matches++;
                        candidate.runs--;
                        candidate.informative += prev_info && curr_info;
                        candidate.displacement -= displacement;
                        if (chosen == NULL ||
                                score_compare(&candidate, chosen) > 0) {
                            matched[cell] = candidate;
                            matched[cell].valid = 1;
                            chosen_state = 2;
                        }
                    }
                    if (chosen_state) {
                        matched_back[cell] = chosen_state;
                    }
                }
            }
        }
    }

    {
        size_t final_cell = (size_t)prev_len * (size_t)columns +
                            (size_t)curr_len;
        if (best_alignment_state(
                &matched[final_cell], &gapped[final_cell], &state) == NULL) {
            state = 0;
        }
    }
    pairs = PyList_New(0);
    if (pairs == NULL) {
        goto error;
    }
    prev_count = prev_len;
    curr_count = curr_len;
    while ((prev_count || curr_count) && state) {
        size_t cell = (size_t)prev_count * (size_t)columns +
                      (size_t)curr_count;
        if (state == 1) {
            PyObject *pair = Py_BuildValue(
                "(nn)", prev_count - 1, curr_count - 1
            );
            if (pair == NULL || PyList_Append(pairs, pair) < 0) {
                Py_XDECREF(pair);
                goto error;
            }
            Py_DECREF(pair);
            state = matched_back[cell];
            prev_count--;
            curr_count--;
        } else {
            unsigned char back = gapped_back[cell];
            if (!back) {
                break;
            }
            if (back <= 2) {
                prev_count--;
                state = back == 1 ? 1 : 2;
            } else {
                curr_count--;
                state = back == 3 ? 1 : 2;
            }
        }
    }
    if (PyList_Reverse(pairs) < 0) {
        goto error;
    }
    free(matched);
    free(gapped);
    free(matched_back);
    free(gapped_back);
    return pairs;

error:
    Py_XDECREF(pairs);
    free(matched);
    free(gapped);
    free(matched_back);
    free(gapped_back);
    return NULL;
}

static PyObject *
available_residual_windows(PyObject *self, PyObject *args)
{
    PyObject *values;
    PyObject *ranges;
    PyObject *residual_by_article;
    PyObject *availability;
    PyObject *structural_tokens;
    PyObject *sizes;
    int previous_mode;
    int minimum_info;
    unsigned char *available = NULL;
    unsigned char *informative = NULL;
    PyObject *output = NULL;
    Py_ssize_t value_count;
    Py_ssize_t iteration = 0;
    PyObject *article_key;
    PyObject *residual;

    if (!PyArg_ParseTuple(
            args, "OOOOpOOi:available_residual_windows",
            &values, &ranges, &residual_by_article, &availability,
            &previous_mode, &structural_tokens, &sizes, &minimum_info)) {
        return NULL;
    }
    if (!PyList_Check(values) || !PyDict_Check(ranges) ||
            !PyDict_Check(residual_by_article) ||
            !PyAnySet_Check(structural_tokens) || !PyTuple_Check(sizes) ||
            (previous_mode && !PyDict_Check(availability)) ||
            (!previous_mode && !PyList_Check(availability))) {
        PyErr_SetString(PyExc_TypeError, "invalid residual-window inputs");
        return NULL;
    }
    value_count = PyList_GET_SIZE(values);
    available = (unsigned char *)calloc(
        (size_t)(value_count > 0 ? value_count : 1), 1
    );
    informative = (unsigned char *)calloc(
        (size_t)(value_count > 0 ? value_count : 1), 1
    );
    if (available == NULL || informative == NULL) {
        PyErr_NoMemory();
        goto error;
    }
    {
        Py_ssize_t index;
        for (index = 0; index < value_count; index++) {
            int info = native_informative_token(
                PyList_GET_ITEM(values, index), structural_tokens
            );
            if (info < 0) {
                goto error;
            }
            informative[index] = info ? 1 : 0;
        }
    }
    while (PyDict_Next(
            residual_by_article, &iteration, &article_key, &residual)) {
        Py_ssize_t article_index = PyLong_AsSsize_t(article_key);
        PyObject *residual_index_obj;
        Py_ssize_t residual_index;
        int is_available;
        if (article_index == -1 && PyErr_Occurred()) {
            goto error;
        }
        if (!PyTuple_Check(residual) || PyTuple_GET_SIZE(residual) < 1) {
            PyErr_SetString(PyExc_ValueError, "invalid residual mapping");
            goto error;
        }
        residual_index_obj = PyTuple_GET_ITEM(residual, 0);
        residual_index = PyLong_AsSsize_t(residual_index_obj);
        if (residual_index == -1 && PyErr_Occurred()) {
            goto error;
        }
        if (previous_mode) {
            int used = PyDict_Contains(availability, residual_index_obj);
            if (used < 0) {
                goto error;
            }
            is_available = !used;
        } else {
            if (residual_index < 0 ||
                    residual_index >= PyList_GET_SIZE(availability)) {
                PyErr_SetString(PyExc_IndexError, "residual index out of range");
                goto error;
            }
            is_available = (
                PyList_GET_ITEM(availability, residual_index) == Py_None
            );
        }
        if (is_available && article_index >= 0 && article_index < value_count) {
            available[article_index] = 1;
        }
    }

    output = PySet_New(NULL);
    if (output == NULL) {
        goto error;
    }
    iteration = 0;
    {
        PyObject *paragraph_key;
        PyObject *range_obj;
        while (PyDict_Next(ranges, &iteration, &paragraph_key, &range_obj)) {
            Py_ssize_t paragraph_start;
            Py_ssize_t paragraph_end;
            Py_ssize_t run_start = -1;
            Py_ssize_t index;
            if (!PyTuple_Check(range_obj) || PyTuple_GET_SIZE(range_obj) != 2) {
                PyErr_SetString(PyExc_ValueError, "invalid paragraph range");
                goto error;
            }
            paragraph_start = PyLong_AsSsize_t(PyTuple_GET_ITEM(range_obj, 0));
            paragraph_end = PyLong_AsSsize_t(PyTuple_GET_ITEM(range_obj, 1));
            if (PyErr_Occurred()) {
                goto error;
            }
            for (index = paragraph_start; index <= paragraph_end; index++) {
                int at_end = index == paragraph_end;
                if (!at_end && available[index]) {
                    if (run_start < 0) {
                        run_start = index;
                    }
                    continue;
                }
                if (run_start >= 0) {
                    Py_ssize_t size_index;
                    Py_ssize_t run_end = index;
                    for (size_index = 0;
                            size_index < PyTuple_GET_SIZE(sizes); size_index++) {
                        Py_ssize_t size = PyLong_AsSsize_t(
                            PyTuple_GET_ITEM(sizes, size_index)
                        );
                        Py_ssize_t start;
                        if (size <= 0 || (size == -1 && PyErr_Occurred())) {
                            PyErr_SetString(PyExc_ValueError, "invalid window size");
                            goto error;
                        }
                        for (start = run_start; start + size <= run_end; start++) {
                            Py_ssize_t info = 0;
                            Py_ssize_t offset;
                            PyObject *window;
                            for (offset = 0; offset < size; offset++) {
                                info += informative[start + offset];
                            }
                            if (info < minimum_info) {
                                continue;
                            }
                            window = PyTuple_New(size);
                            if (window == NULL) {
                                goto error;
                            }
                            for (offset = 0; offset < size; offset++) {
                                PyObject *item = PyList_GET_ITEM(
                                    values, start + offset
                                );
                                Py_INCREF(item);
                                PyTuple_SET_ITEM(window, offset, item);
                            }
                            if (PySet_Add(output, window) < 0) {
                                Py_DECREF(window);
                                goto error;
                            }
                            Py_DECREF(window);
                        }
                    }
                    run_start = -1;
                }
            }
        }
    }
    free(available);
    free(informative);
    return output;

error:
    Py_XDECREF(output);
    free(available);
    free(informative);
    return NULL;
}

static int
token_equals_ascii(PyObject *token, const char *text)
{
    return PyUnicode_Check(token) &&
           PyUnicode_CompareWithASCIIString(token, text) == 0;
}

static PyObject *
configure_token_symbols(PyObject *self, PyObject *symbols)
{
    PyObject *separator;
    PyObject *joined;
    if (!PyTuple_Check(symbols)) {
        PyErr_SetString(PyExc_TypeError, "token symbols must be a tuple");
        return NULL;
    }
    separator = PyUnicode_FromString("");
    if (separator == NULL) {
        return NULL;
    }
    joined = PyUnicode_Join(separator, symbols);
    Py_DECREF(separator);
    if (joined == NULL) {
        return NULL;
    }
    Py_XDECREF(token_symbols_text);
    token_symbols_text = joined;
    Py_RETURN_NONE;
}

static int
is_cjk_token_character(Py_UCS4 character)
{
    return ((character >= 0x2E80 && character <= 0x9FFF) ||
            (character >= 0xF900 && character <= 0xFAFF) ||
            (character >= 0xFE00 && character <= 0xFE1F) ||
            (character >= 0xFE30 && character <= 0xFE6F) ||
            (character >= 0xFF00 && character <= 0xFFEF) ||
            (character >= 0x16FE0 && character <= 0x1B2FF) ||
            (character >= 0x1F000 && character <= 0x1F2FF) ||
            (character >= 0x20000 && character <= 0x3347F) ||
            (character >= 0xE0100 && character <= 0xE01EF));
}

static int
is_token_symbol_character(Py_UCS4 character)
{
    Py_ssize_t found;
    if (token_symbols_text == NULL) {
        PyErr_SetString(PyExc_RuntimeError,
                        "native token symbols are not configured");
        return -1;
    }
    found = PyUnicode_FindChar(
        token_symbols_text, character, 0,
        PyUnicode_GET_LENGTH(token_symbols_text), 1
    );
    if (found < 0 && PyErr_Occurred()) {
        return -1;
    }
    return found >= 0;
}

static int
append_token_slice(PyObject *output, PyObject *text,
                   Py_ssize_t start, Py_ssize_t end)
{
    PyObject *token;
    int status;
    if (start >= end) {
        return 0;
    }
    if (end - start == 4 &&
            PyUnicode_READ_CHAR(text, start) == 0xE6 &&
            PyUnicode_READ_CHAR(text, start + 1) == 0xE6 &&
            PyUnicode_READ_CHAR(text, start + 2) == 0xE6 &&
            PyUnicode_READ_CHAR(text, start + 3) == 0xE6) {
        return PyList_Append(output, pipe_token);
    }
    token = PyUnicode_Substring(text, start, end);
    if (token == NULL) {
        return -1;
    }
    status = PyList_Append(output, token);
    Py_DECREF(token);
    return status;
}

static int
unicode_matches_ascii_at(PyObject *text, Py_ssize_t start,
                         Py_ssize_t text_length, const char *pattern,
                         Py_ssize_t pattern_length)
{
    Py_ssize_t offset;
    if (start < 0 || pattern_length > text_length - start) {
        return 0;
    }
    for (offset = 0; offset < pattern_length; offset++) {
        if (PyUnicode_READ_CHAR(text, start + offset) !=
                (unsigned char)pattern[offset]) {
            return 0;
        }
    }
    return 1;
}

/* Return the length of a paragraph delimiter-adjacent tag at ``start`` and
 * set ``side`` to -1 for a leading delimiter or +1 for a trailing delimiter.
 * These are exactly the literal replacements performed by
 * utils.split_into_paragraphs before its line-start ``|}`` rule. */
static Py_ssize_t
paragraph_tag_at(PyObject *text, Py_ssize_t start, Py_ssize_t length,
                 int *side)
{
    Py_UCS4 first = PyUnicode_READ_CHAR(text, start);
    if (first == '<') {
        if (unicode_matches_ascii_at(text, start, length, "<table>", 7)) {
            *side = -1;
            return 7;
        }
        if (unicode_matches_ascii_at(text, start, length, "</table>", 8)) {
            *side = 1;
            return 8;
        }
        if (unicode_matches_ascii_at(text, start, length, "<tr>", 4)) {
            *side = -1;
            return 4;
        }
        if (unicode_matches_ascii_at(text, start, length, "</tr>", 5)) {
            *side = 1;
            return 5;
        }
    } else if (first == '{' &&
               unicode_matches_ascii_at(text, start, length, "{|", 2)) {
        *side = -1;
        return 2;
    }
    return 0;
}

static PyObject *
paragraph_stage_one(PyObject *text)
{
    Py_ssize_t length = PyUnicode_GET_LENGTH(text);
    Py_ssize_t output_length = 0;
    Py_ssize_t input_index = 0;
    Py_ssize_t output_index = 0;
    PyObject *output;
    int output_kind;
    void *output_data;

    while (input_index < length) {
        Py_UCS4 character = PyUnicode_READ_CHAR(text, input_index);
        Py_ssize_t tag_length;
        int side = 0;
        tag_length = paragraph_tag_at(
            text, input_index, length, &side
        );
        if (tag_length) {
            if (output_length > PY_SSIZE_T_MAX - tag_length - 2) {
                PyErr_SetString(PyExc_OverflowError,
                                "paragraph text is too large");
                return NULL;
            }
            output_length += tag_length + 2;
            input_index += tag_length;
        } else {
            if (output_length == PY_SSIZE_T_MAX) {
                PyErr_SetString(PyExc_OverflowError,
                                "paragraph text is too large");
                return NULL;
            }
            output_length++;
            input_index++;
            if (character == '\r' && input_index < length &&
                    PyUnicode_READ_CHAR(text, input_index) == '\n') {
                input_index++;
            }
        }
    }

    output = PyUnicode_New(
        output_length, PyUnicode_MAX_CHAR_VALUE(text)
    );
    if (output == NULL) {
        return NULL;
    }
    output_kind = PyUnicode_KIND(output);
    output_data = PyUnicode_DATA(output);
    input_index = 0;
    while (input_index < length) {
        Py_UCS4 character = PyUnicode_READ_CHAR(text, input_index);
        Py_ssize_t tag_length;
        Py_ssize_t offset;
        int side = 0;
        tag_length = paragraph_tag_at(
            text, input_index, length, &side
        );
        if (tag_length) {
            if (side < 0) {
                PyUnicode_WRITE(output_kind, output_data, output_index++, '\n');
                PyUnicode_WRITE(output_kind, output_data, output_index++, '\n');
            }
            for (offset = 0; offset < tag_length; offset++) {
                PyUnicode_WRITE(
                    output_kind, output_data, output_index++,
                    PyUnicode_READ_CHAR(text, input_index + offset)
                );
            }
            if (side > 0) {
                PyUnicode_WRITE(output_kind, output_data, output_index++, '\n');
                PyUnicode_WRITE(output_kind, output_data, output_index++, '\n');
            }
            input_index += tag_length;
        } else {
            PyUnicode_WRITE(
                output_kind, output_data, output_index++,
                character == '\r' ? '\n' : character
            );
            input_index++;
            if (character == '\r' && input_index < length &&
                    PyUnicode_READ_CHAR(text, input_index) == '\n') {
                input_index++;
            }
        }
    }
    return output;
}

static Py_ssize_t
paragraph_line_close_end(PyObject *text, Py_ssize_t line_start,
                         Py_ssize_t length)
{
    Py_ssize_t index = line_start;
    while (index < length) {
        Py_UCS4 character = PyUnicode_READ_CHAR(text, index);
        if (character != ' ' && character != '\t') {
            break;
        }
        index++;
    }
    if (index + 1 < length &&
            PyUnicode_READ_CHAR(text, index) == '|' &&
            PyUnicode_READ_CHAR(text, index + 1) == '}' &&
            (index + 2 >= length ||
             PyUnicode_READ_CHAR(text, index + 2) != '}')) {
        return index + 2;
    }
    return -1;
}

static PyObject *
paragraph_stage_two(PyObject *text)
{
    Py_ssize_t length = PyUnicode_GET_LENGTH(text);
    Py_ssize_t match_count = 0;
    Py_ssize_t index;
    Py_ssize_t next_match_end;
    Py_ssize_t output_index = 0;
    PyObject *output;
    int output_kind;
    void *output_data;

    next_match_end = paragraph_line_close_end(text, 0, length);
    if (next_match_end >= 0) {
        match_count++;
    }
    for (index = 0; index < length; index++) {
        if (PyUnicode_READ_CHAR(text, index) == '\n' && index + 1 < length) {
            Py_ssize_t match_end = paragraph_line_close_end(
                text, index + 1, length
            );
            if (match_end >= 0) {
                match_count++;
            }
        }
    }
    if (!match_count) {
        Py_INCREF(text);
        return text;
    }
    if (match_count > (PY_SSIZE_T_MAX - length) / 2) {
        PyErr_SetString(PyExc_OverflowError, "paragraph text is too large");
        return NULL;
    }
    output = PyUnicode_New(
        length + 2 * match_count, PyUnicode_MAX_CHAR_VALUE(text)
    );
    if (output == NULL) {
        return NULL;
    }
    output_kind = PyUnicode_KIND(output);
    output_data = PyUnicode_DATA(output);
    next_match_end = paragraph_line_close_end(text, 0, length);
    for (index = 0; index < length; index++) {
        PyUnicode_WRITE(
            output_kind, output_data, output_index++,
            PyUnicode_READ_CHAR(text, index)
        );
        if (index + 1 == next_match_end) {
            PyUnicode_WRITE(output_kind, output_data, output_index++, '\n');
            PyUnicode_WRITE(output_kind, output_data, output_index++, '\n');
        }
        if (PyUnicode_READ_CHAR(text, index) == '\n' && index + 1 < length) {
            next_match_end = paragraph_line_close_end(
                text, index + 1, length
            );
        }
    }
    return output;
}

static PyObject *
split_into_paragraphs_native(PyObject *self, PyObject *text)
{
    PyObject *stage_one;
    PyObject *stage_two;
    PyObject *marker;
    PyObject *replacement;
    PyObject *separator;
    PyObject *stage_three;
    PyObject *output;
    if (!PyUnicode_Check(text)) {
        PyErr_SetString(PyExc_TypeError, "paragraph splitter input must be text");
        return NULL;
    }
    stage_one = paragraph_stage_one(text);
    if (stage_one == NULL) {
        return NULL;
    }
    stage_two = paragraph_stage_two(stage_one);
    Py_DECREF(stage_one);
    if (stage_two == NULL) {
        return NULL;
    }
    marker = PyUnicode_FromString("|-\n");
    replacement = PyUnicode_FromString("\n\n|-\n");
    separator = PyUnicode_FromString("\n\n");
    if (marker == NULL || replacement == NULL || separator == NULL) {
        Py_XDECREF(marker);
        Py_XDECREF(replacement);
        Py_XDECREF(separator);
        Py_DECREF(stage_two);
        return NULL;
    }
    stage_three = PyUnicode_Replace(
        stage_two, marker, replacement, -1
    );
    Py_DECREF(stage_two);
    Py_DECREF(marker);
    Py_DECREF(replacement);
    if (stage_three == NULL) {
        Py_DECREF(separator);
        return NULL;
    }
    output = PyUnicode_Split(stage_three, separator, -1);
    Py_DECREF(stage_three);
    Py_DECREF(separator);
    return output;
}

static PyObject *
split_into_tokens_native(PyObject *self, PyObject *text)
{
    PyObject *output;
    Py_ssize_t length;
    Py_ssize_t index = 0;
    Py_ssize_t word_start = 0;
    if (!PyUnicode_Check(text)) {
        PyErr_SetString(PyExc_TypeError, "tokenizer input must be text");
        return NULL;
    }
    if (token_symbols_text == NULL || pipe_token == NULL) {
        PyErr_SetString(PyExc_RuntimeError,
                        "native tokenizer is not configured");
        return NULL;
    }
    output = PyList_New(0);
    if (output == NULL) {
        return NULL;
    }
    length = PyUnicode_GET_LENGTH(text);
    while (index < length) {
        Py_UCS4 character = PyUnicode_READ_CHAR(text, index);
        Py_ssize_t marker_length = 0;
        int symbol;
        int delimiter;
        if (index + 3 < length && character == '<' &&
                PyUnicode_READ_CHAR(text, index + 1) == '!' &&
                PyUnicode_READ_CHAR(text, index + 2) == '-' &&
                PyUnicode_READ_CHAR(text, index + 3) == '-') {
            marker_length = 4;
        } else if (index + 2 < length && character == '-' &&
                   PyUnicode_READ_CHAR(text, index + 1) == '-' &&
                   PyUnicode_READ_CHAR(text, index + 2) == '>') {
            marker_length = 3;
        } else if (index + 1 < length &&
                   ((character == '[' &&
                     PyUnicode_READ_CHAR(text, index + 1) == '[') ||
                    (character == ']' &&
                     PyUnicode_READ_CHAR(text, index + 1) == ']') ||
                    (character == '{' &&
                     PyUnicode_READ_CHAR(text, index + 1) == '{') ||
                    (character == '}' &&
                     PyUnicode_READ_CHAR(text, index + 1) == '}'))) {
            marker_length = 2;
        }
        symbol = (marker_length || character == '|') ? 1 :
            is_token_symbol_character(character);
        if (symbol < 0) {
            Py_DECREF(output);
            return NULL;
        }
        delimiter = (
            character == ' ' || character == '\n' || symbol ||
            is_cjk_token_character(character)
        );
        if (!delimiter) {
            index++;
            continue;
        }
        if (append_token_slice(output, text, word_start, index) < 0) {
            Py_DECREF(output);
            return NULL;
        }
        if (character != ' ' && character != '\n') {
            Py_ssize_t token_end = index + (marker_length ? marker_length : 1);
            if (character == '|' && marker_length == 0) {
                if (PyList_Append(output, pipe_token) < 0) {
                    Py_DECREF(output);
                    return NULL;
                }
            } else if (append_token_slice(
                    output, text, index, token_end) < 0) {
                Py_DECREF(output);
                return NULL;
            }
            index = token_end;
        } else {
            index++;
        }
        word_start = index;
    }
    if (append_token_slice(output, text, word_start, length) < 0) {
        Py_DECREF(output);
        return NULL;
    }
    return output;
}

static int
token_window_equals_tuple(PyObject *tokens, Py_ssize_t start,
                          PyObject *needle)
{
    Py_ssize_t offset;
    Py_ssize_t needle_length = PyTuple_GET_SIZE(needle);
    for (offset = 0; offset < needle_length; offset++) {
        PyObject *left = PyList_GET_ITEM(tokens, start + offset);
        PyObject *right = PyTuple_GET_ITEM(needle, offset);
        int equal;
        if (left == right) {
            continue;
        }
        equal = PyObject_RichCompareBool(left, right, Py_EQ);
        if (equal <= 0) {
            return equal;
        }
    }
    return 1;
}

/* Count a needle only up to two occurrences, using the same first/last pair
 * candidate lists as the Python fallback.  Moved-run copy safety only asks
 * whether an occurrence is absent, unique, or repeated. */
static PyObject *
count_subsequence_at_positions(PyObject *self, PyObject *args)
{
    PyObject *tokens;
    PyObject *needle;
    PyObject *first_positions;
    PyObject *last_positions;
    PyObject *positions_fast = NULL;
    Py_ssize_t token_length;
    Py_ssize_t needle_length;
    Py_ssize_t count = 0;
    Py_ssize_t index;
    int use_last;

    if (!PyArg_ParseTuple(
            args, "OOOO:count_subsequence_at_positions", &tokens, &needle,
            &first_positions, &last_positions)) {
        return NULL;
    }
    if (!PyList_Check(tokens) || !PyTuple_Check(needle)) {
        PyErr_SetString(PyExc_TypeError,
                        "tokens must be a list and needle a tuple");
        return NULL;
    }
    token_length = PyList_GET_SIZE(tokens);
    needle_length = PyTuple_GET_SIZE(needle);
    if (needle_length == 0 || needle_length > token_length) {
        return PyLong_FromLong(0);
    }
    if (needle_length == 1) {
        PyObject *target = PyTuple_GET_ITEM(needle, 0);
        for (index = 0; index < token_length; index++) {
            PyObject *token = PyList_GET_ITEM(tokens, index);
            int equal = token == target ? 1 :
                PyObject_RichCompareBool(token, target, Py_EQ);
            if (equal < 0) {
                return NULL;
            }
            if (equal && ++count == 2) {
                break;
            }
        }
        return PyLong_FromSsize_t(count);
    }

    use_last = PySequence_Size(last_positions) <
               PySequence_Size(first_positions);
    if (PyErr_Occurred()) {
        return NULL;
    }
    positions_fast = PySequence_Fast(
        use_last ? last_positions : first_positions,
        "pair positions must be a sequence"
    );
    if (positions_fast == NULL) {
        return NULL;
    }
    for (index = 0; index < PySequence_Fast_GET_SIZE(positions_fast); index++) {
        Py_ssize_t pair_index = PyLong_AsSsize_t(
            PySequence_Fast_GET_ITEM(positions_fast, index)
        );
        Py_ssize_t start;
        int equal;
        if (pair_index == -1 && PyErr_Occurred()) {
            Py_DECREF(positions_fast);
            return NULL;
        }
        start = use_last ? pair_index - needle_length + 2 : pair_index;
        if (start < 0 || start > token_length - needle_length) {
            continue;
        }
        equal = token_window_equals_tuple(tokens, start, needle);
        if (equal < 0) {
            Py_DECREF(positions_fast);
            return NULL;
        }
        if (equal && ++count == 2) {
            break;
        }
    }
    Py_DECREF(positions_fast);
    return PyLong_FromSsize_t(count);
}

enum {
    TOKEN_MARKUP_NONE = 0,
    TOKEN_MARKUP_TEMPLATE_OPEN,
    TOKEN_MARKUP_TEMPLATE_CLOSE,
    TOKEN_MARKUP_LINK_OPEN,
    TOKEN_MARKUP_LINK_CLOSE,
    TOKEN_MARKUP_COMMENT_OPEN,
    TOKEN_MARKUP_COMMENT_CLOSE,
    TOKEN_MARKUP_PIPE,
    TOKEN_MARKUP_EQUALS
};

static int
token_markup_kind(PyObject *token)
{
    Py_ssize_t length;
    Py_UCS4 first;
    if (!PyUnicode_Check(token)) {
        return TOKEN_MARKUP_NONE;
    }
    length = PyUnicode_GET_LENGTH(token);
    if (!length) {
        return TOKEN_MARKUP_NONE;
    }
    first = PyUnicode_READ_CHAR(token, 0);
    if (length == 1) {
        if (first == '|') return TOKEN_MARKUP_PIPE;
        if (first == '=') return TOKEN_MARKUP_EQUALS;
        return TOKEN_MARKUP_NONE;
    }
    if (length == 2) {
        Py_UCS4 second = PyUnicode_READ_CHAR(token, 1);
        if (first == '{' && second == '{') {
            return TOKEN_MARKUP_TEMPLATE_OPEN;
        }
        if (first == '}' && second == '}') {
            return TOKEN_MARKUP_TEMPLATE_CLOSE;
        }
        if (first == '[' && second == '[') {
            return TOKEN_MARKUP_LINK_OPEN;
        }
        if (first == ']' && second == ']') {
            return TOKEN_MARKUP_LINK_CLOSE;
        }
        return TOKEN_MARKUP_NONE;
    }
    if (length == 3 && first == '-' &&
            PyUnicode_READ_CHAR(token, 1) == '-' &&
            PyUnicode_READ_CHAR(token, 2) == '>') {
        return TOKEN_MARKUP_COMMENT_CLOSE;
    }
    if (length == 4 && first == '<' &&
            PyUnicode_READ_CHAR(token, 1) == '!' &&
            PyUnicode_READ_CHAR(token, 2) == '-' &&
            PyUnicode_READ_CHAR(token, 3) == '-') {
        return TOKEN_MARKUP_COMMENT_OPEN;
    }
    return TOKEN_MARKUP_NONE;
}

static int
token_is_any_stop(PyObject *token, const char **stops, Py_ssize_t stop_count)
{
    Py_ssize_t index;
    for (index = 0; index < stop_count; index++) {
        if (token_equals_ascii(token, stops[index])) {
            return 1;
        }
    }
    return 0;
}

static PyObject *
tokens_until(PyObject *tokens, Py_ssize_t start,
             const char **stops, Py_ssize_t stop_count,
             Py_ssize_t *end)
{
    Py_ssize_t length = PyList_GET_SIZE(tokens);
    Py_ssize_t index = start;
    PyObject *collected;
    Py_ssize_t offset;
    while (index < length && !token_is_any_stop(
            PyList_GET_ITEM(tokens, index), stops, stop_count)) {
        index++;
    }
    collected = PyTuple_New(index - start);
    if (collected == NULL) {
        return NULL;
    }
    for (offset = start; offset < index; offset++) {
        PyObject *item = PyList_GET_ITEM(tokens, offset);
        Py_INCREF(item);
        PyTuple_SET_ITEM(collected, offset - start, item);
    }
    *end = index;
    return collected;
}

static PyObject *
template_field_before(PyObject *tokens, Py_ssize_t equals_index)
{
    Py_ssize_t index = equals_index - 1;
    Py_ssize_t count = 0;
    PyObject *field;
    while (index >= 0) {
        PyObject *token = PyList_GET_ITEM(tokens, index);
        if (token_equals_ascii(token, "{{") ||
                token_equals_ascii(token, "|") ||
                token_equals_ascii(token, "}}")) {
            break;
        }
        count++;
        index--;
    }
    if (index < 0 || count == 0 ||
            !token_equals_ascii(PyList_GET_ITEM(tokens, index), "|")) {
        Py_RETURN_NONE;
    }
    field = PyTuple_New(count);
    if (field == NULL) {
        return NULL;
    }
    {
        Py_ssize_t offset;
        for (offset = 0; offset < count; offset++) {
            PyObject *item = PyList_GET_ITEM(tokens, index + 1 + offset);
            Py_INCREF(item);
            PyTuple_SET_ITEM(field, offset, item);
        }
    }
    return field;
}

static int
replace_list_item(PyObject *list, Py_ssize_t index, PyObject *replacement)
{
    PyObject *previous;
    if (replacement == NULL) {
        return -1;
    }
    previous = PyList_GET_ITEM(list, index);
    PyList_SET_ITEM(list, index, replacement);
    Py_DECREF(previous);
    return 0;
}

static int
pop_construct(ConstructFrame *stack, Py_ssize_t *depth, int type,
              ConstructFrame *result)
{
    Py_ssize_t index;
    for (index = *depth - 1; index >= 0; index--) {
        if (stack[index].type == type) {
            Py_ssize_t discard;
            *result = stack[index];
            for (discard = index + 1; discard < *depth; discard++) {
                Py_XDECREF(stack[discard].context);
            }
            *depth = index;
            return 1;
        }
    }
    return 0;
}

/* Return 1 with a borrowed occurrence, 0 for malformed hierarchy state, and
 * -1 for an exception. */
static int
ordered_occurrence(PyObject *mapping, PyObject *counts, PyObject *key,
                   PyObject **result)
{
    PyObject *occurrences = PyDict_GetItemWithError(mapping, key);
    PyObject *count_object;
    PyObject *next_count;
    Py_ssize_t count = 0;
    if (occurrences == NULL) {
        return PyErr_Occurred() ? -1 : 0;
    }
    if (!PyList_Check(occurrences)) {
        return 0;
    }
    count_object = PyDict_GetItemWithError(counts, key);
    if (count_object != NULL) {
        count = PyLong_AsSsize_t(count_object);
        if (count == -1 && PyErr_Occurred()) {
            return -1;
        }
    } else if (PyErr_Occurred()) {
        return -1;
    }
    if (count < 0 || count >= PyList_GET_SIZE(occurrences)) {
        return 0;
    }
    next_count = PyLong_FromSsize_t(count + 1);
    if (next_count == NULL) {
        return -1;
    }
    if (PyDict_SetItem(counts, key, next_count) < 0) {
        Py_DECREF(next_count);
        return -1;
    }
    Py_DECREF(next_count);
    *result = PyList_GET_ITEM(occurrences, count);
    return 1;
}

/* Append one sentence's normalized values while reproducing the Python
 * hierarchy consistency checks. */
static int
append_sentence_values(PyObject *sentence, PyObject *values,
                       PointerSet *seen_words, ParagraphCache *cache,
                       Py_ssize_t *sentence_start,
                       Py_ssize_t *sentence_length)
{
    PyObject *words = PyObject_GetAttr(sentence, attr_words);
    PyObject *splitted = NULL;
    PyObject *split_values = NULL;
    Py_ssize_t start = PyList_GET_SIZE(values);
    Py_ssize_t split_length = 0;
    Py_ssize_t index;
    int split_truth;
    int status = 1;
    if (words == NULL) {
        return -1;
    }
    splitted = PyObject_GetAttr(sentence, attr_splitted);
    if (splitted == NULL) {
        Py_DECREF(words);
        return -1;
    }
    if (!PyList_Check(words)) {
        status = 0;
        goto done;
    }
    split_truth = PyObject_IsTrue(splitted);
    if (split_truth < 0) {
        status = -1;
        goto done;
    }
    if (split_truth) {
        split_values = PySequence_Fast(
            splitted, "sentence split values are not iterable"
        );
        if (split_values == NULL) {
            status = -1;
            goto done;
        }
        split_length = PySequence_Fast_GET_SIZE(split_values);
    }
    if (PyList_GET_SIZE(words)) {
        for (index = 0; index < PyList_GET_SIZE(words); index++) {
            PyObject *word = PyList_GET_ITEM(words, index);
            PyObject *value;
            int added = pointer_set_add(seen_words, word);
            if (added <= 0) {
                status = added;
                goto done;
            }
            if (cache != NULL &&
                    paragraph_cache_append_word(cache, word) < 0) {
                status = -1;
                goto done;
            }
            value = PyObject_GetAttr(word, attr_value);
            if (value == NULL) {
                status = -1;
                goto done;
            }
            if (PyList_Append(values, value) < 0) {
                Py_DECREF(value);
                status = -1;
                goto done;
            }
            Py_DECREF(value);
        }
        if (split_length) {
            if (split_length != PyList_GET_SIZE(words)) {
                status = 0;
                goto done;
            }
            for (index = 0; index < split_length; index++) {
                int equal = PyObject_RichCompareBool(
                    PySequence_Fast_GET_ITEM(split_values, index),
                    PyList_GET_ITEM(values, start + index), Py_EQ
                );
                if (equal <= 0) {
                    status = equal;
                    goto done;
                }
            }
        }
    } else if (split_length) {
        for (index = 0; index < split_length; index++) {
            if (PyList_Append(
                    values, PySequence_Fast_GET_ITEM(
                        split_values, index
                    )) < 0) {
                status = -1;
                goto done;
            }
        }
    } else {
        status = 0;
        goto done;
    }
    *sentence_start = start;
    *sentence_length = PyList_GET_SIZE(values) - start;

done:
    Py_XDECREF(split_values);
    Py_DECREF(words);
    Py_DECREF(splitted);
    return status;
}

static int
append_cached_paragraph(
        ParagraphCache *cache, CachedParagraph *paragraph,
        PyObject *targets, Py_ssize_t paragraph_index, PyObject *values,
        PyObject *paragraph_ranges, PyObject *sentence_ranges,
        PointerSet *seen_sentences, PointerSet *seen_words)
{
    Py_ssize_t paragraph_start = PyList_GET_SIZE(values);
    Py_ssize_t value_cursor = paragraph->value_start;
    Py_ssize_t word_cursor = paragraph->word_start;
    Py_ssize_t sentence_offset;
    for (sentence_offset = 0;
            sentence_offset < paragraph->sentence_count;
            sentence_offset++) {
        CachedSentence *sentence = &cache->sentences[
            paragraph->sentence_start + sentence_offset
        ];
        Py_ssize_t word_offset;
        Py_ssize_t value_offset;
        Py_ssize_t sentence_start;
        int status = pointer_set_add(
            seen_sentences, sentence->sentence
        );
        int retain;
        if (status <= 0) {
            return status;
        }
        for (word_offset = 0;
                word_offset < sentence->word_count;
                word_offset++) {
            status = pointer_set_add(
                seen_words, cache->words[word_cursor + word_offset]
            );
            if (status <= 0) {
                return status;
            }
        }
        sentence_start = PyList_GET_SIZE(values);
        for (value_offset = 0;
                value_offset < sentence->value_length;
                value_offset++) {
            if (PyList_Append(
                    values, PyList_GET_ITEM(
                        cache->values,
                        value_cursor + value_offset
                    )) < 0) {
                return -1;
            }
        }
        word_cursor += sentence->word_count;
        value_cursor += sentence->value_length;
        if (targets == Py_None) {
            retain = 1;
        } else {
            retain = PySet_Contains(targets, sentence->sentence);
            if (retain < 0) {
                return -1;
            }
        }
        if (retain) {
            PyObject *range = Py_BuildValue(
                "(nnnn)", paragraph_index, sentence_offset,
                sentence_start, sentence->value_length
            );
            if (range == NULL || PyDict_SetItem(
                    sentence_ranges, sentence->sentence, range) < 0) {
                Py_XDECREF(range);
                return -1;
            }
            Py_DECREF(range);
        }
    }
    {
        PyObject *paragraph_key = PyLong_FromSsize_t(paragraph_index);
        PyObject *paragraph_range = Py_BuildValue(
            "(nn)", paragraph_start, PyList_GET_SIZE(values)
        );
        if (paragraph_key == NULL || paragraph_range == NULL ||
                PyDict_SetItem(
                    paragraph_ranges, paragraph_key,
                    paragraph_range) < 0) {
            Py_XDECREF(paragraph_key);
            Py_XDECREF(paragraph_range);
            return -1;
        }
        Py_DECREF(paragraph_key);
        Py_DECREF(paragraph_range);
    }
    return 1;
}

/* Return 1 with owned outputs, 0 for fail-closed hierarchy state, and -1 for
 * an exception. */
static int
build_structural_document(PyObject *revision, PyObject *targets,
                          ParagraphCache *cache, int populate_cache,
                          PyObject *cache_peer,
                          PyObject **values_result,
                          PyObject **paragraph_ranges_result,
                          PyObject **sentence_ranges_result)
{
    PyObject *ordered_paragraphs = NULL;
    PyObject *paragraphs = NULL;
    PyObject *paragraph_counts = NULL;
    PyObject *values = NULL;
    PyObject *paragraph_ranges = NULL;
    PyObject *sentence_ranges = NULL;
    PointerSet seen_paragraphs = {NULL, 0, 0};
    PointerSet seen_sentences = {NULL, 0, 0};
    PointerSet seen_words = {NULL, 0, 0};
    Py_ssize_t paragraph_index;
    int status = -1;

    ordered_paragraphs = PyObject_GetAttr(
        revision, attr_ordered_paragraphs
    );
    paragraphs = PyObject_GetAttr(revision, attr_paragraphs);
    if (ordered_paragraphs == NULL || paragraphs == NULL) {
        goto done;
    }
    if (!PyList_Check(ordered_paragraphs) || !PyDict_Check(paragraphs) ||
            (targets != Py_None && !PyAnySet_Check(targets))) {
        status = 0;
        goto done;
    }
    if (populate_cache && paragraph_cache_init(
            cache, cache_peer,
            (size_t)PyList_GET_SIZE(ordered_paragraphs)) < 0) {
        goto done;
    }
    paragraph_counts = PyDict_New();
    values = PyList_New(0);
    paragraph_ranges = PyDict_New();
    sentence_ranges = PyDict_New();
    if (paragraph_counts == NULL || values == NULL ||
            paragraph_ranges == NULL || sentence_ranges == NULL ||
            pointer_set_init(
                &seen_paragraphs,
                (size_t)PyList_GET_SIZE(ordered_paragraphs)
            ) < 0 ||
            pointer_set_init(&seen_sentences, 256) < 0 ||
            pointer_set_init(&seen_words, 1024) < 0) {
        goto done;
    }

    for (paragraph_index = 0;
            paragraph_index < PyList_GET_SIZE(ordered_paragraphs);
            paragraph_index++) {
        PyObject *paragraph_hash = PyList_GET_ITEM(
            ordered_paragraphs, paragraph_index
        );
        PyObject *paragraph;
        PyObject *ordered_sentences = NULL;
        PyObject *sentences = NULL;
        PyObject *sentence_counts = NULL;
        PyObject *paragraph_key = NULL;
        PyObject *paragraph_range = NULL;
        Py_ssize_t paragraph_start = PyList_GET_SIZE(values);
        Py_ssize_t cached_sentence_start = populate_cache ?
            (Py_ssize_t)cache->sentence_count : 0;
        Py_ssize_t cached_word_start = populate_cache ?
            (Py_ssize_t)cache->word_count : 0;
        Py_ssize_t sentence_index;
        int cache_paragraph = 0;
        int occurrence_status = ordered_occurrence(
            paragraphs, paragraph_counts, paragraph_hash, &paragraph
        );
        if (occurrence_status <= 0) {
            status = occurrence_status;
            goto done;
        }
        if (populate_cache) {
            cache_paragraph = paragraph_cache_may_reuse(
                cache, paragraph_hash
            );
            if (cache_paragraph < 0) {
                goto done;
            }
        }
        occurrence_status = pointer_set_add(&seen_paragraphs, paragraph);
        if (occurrence_status <= 0) {
            status = occurrence_status;
            goto done;
        }
        if (!populate_cache) {
            CachedParagraph *cached = paragraph_cache_lookup(
                cache, paragraph
            );
            if (cached != NULL) {
                occurrence_status = append_cached_paragraph(
                    cache, cached, targets, paragraph_index, values,
                    paragraph_ranges, sentence_ranges,
                    &seen_sentences, &seen_words
                );
                if (occurrence_status <= 0) {
                    status = occurrence_status;
                    goto done;
                }
                continue;
            }
        }
        ordered_sentences = PyObject_GetAttr(
            paragraph, attr_ordered_sentences
        );
        sentences = PyObject_GetAttr(paragraph, attr_sentences);
        if (ordered_sentences == NULL || sentences == NULL) {
            Py_XDECREF(ordered_sentences);
            Py_XDECREF(sentences);
            goto done;
        }
        if (!PyList_Check(ordered_sentences) || !PyDict_Check(sentences)) {
            Py_DECREF(ordered_sentences);
            Py_DECREF(sentences);
            status = 0;
            goto done;
        }
        sentence_counts = PyDict_New();
        if (sentence_counts == NULL) {
            Py_DECREF(ordered_sentences);
            Py_DECREF(sentences);
            goto done;
        }
        for (sentence_index = 0;
                sentence_index < PyList_GET_SIZE(ordered_sentences);
                sentence_index++) {
            PyObject *sentence_hash = PyList_GET_ITEM(
                ordered_sentences, sentence_index
            );
            PyObject *sentence;
            Py_ssize_t sentence_start;
            Py_ssize_t sentence_length;
            Py_ssize_t cached_sentence_word_start = populate_cache ?
                (Py_ssize_t)cache->word_count : 0;
            int retain;
            occurrence_status = ordered_occurrence(
                sentences, sentence_counts, sentence_hash, &sentence
            );
            if (occurrence_status <= 0) {
                Py_DECREF(sentence_counts);
                Py_DECREF(ordered_sentences);
                Py_DECREF(sentences);
                status = occurrence_status;
                goto done;
            }
            occurrence_status = pointer_set_add(
                &seen_sentences, sentence
            );
            if (occurrence_status <= 0) {
                Py_DECREF(sentence_counts);
                Py_DECREF(ordered_sentences);
                Py_DECREF(sentences);
                status = occurrence_status;
                goto done;
            }
            occurrence_status = append_sentence_values(
                sentence, values, &seen_words,
                cache_paragraph ? cache : NULL,
                &sentence_start, &sentence_length
            );
            if (occurrence_status <= 0) {
                Py_DECREF(sentence_counts);
                Py_DECREF(ordered_sentences);
                Py_DECREF(sentences);
                status = occurrence_status;
                goto done;
            }
            if (cache_paragraph && paragraph_cache_append_sentence(
                    cache, sentence, sentence_length,
                    (Py_ssize_t)cache->word_count -
                        cached_sentence_word_start) < 0) {
                Py_DECREF(sentence_counts);
                Py_DECREF(ordered_sentences);
                Py_DECREF(sentences);
                goto done;
            }
            if (targets == Py_None) {
                retain = 1;
            } else {
                retain = PySet_Contains(targets, sentence);
                if (retain < 0) {
                    Py_DECREF(sentence_counts);
                    Py_DECREF(ordered_sentences);
                    Py_DECREF(sentences);
                    goto done;
                }
            }
            if (retain) {
                PyObject *range = Py_BuildValue(
                    "(nnnn)", paragraph_index, sentence_index,
                    sentence_start, sentence_length
                );
                if (range == NULL || PyDict_SetItem(
                        sentence_ranges, sentence, range) < 0) {
                    Py_XDECREF(range);
                    Py_DECREF(sentence_counts);
                    Py_DECREF(ordered_sentences);
                    Py_DECREF(sentences);
                    goto done;
                }
                Py_DECREF(range);
            }
        }
        Py_DECREF(sentence_counts);
        Py_DECREF(ordered_sentences);
        Py_DECREF(sentences);
        if (cache_paragraph) {
            occurrence_status = paragraph_cache_insert(
                cache, paragraph, paragraph_start,
                cached_sentence_start,
                (Py_ssize_t)cache->sentence_count -
                    cached_sentence_start,
                cached_word_start
            );
            if (occurrence_status <= 0) {
                status = occurrence_status;
                goto done;
            }
        }
        paragraph_key = PyLong_FromSsize_t(paragraph_index);
        paragraph_range = Py_BuildValue(
            "(nn)", paragraph_start, PyList_GET_SIZE(values)
        );
        if (paragraph_key == NULL || paragraph_range == NULL ||
                PyDict_SetItem(
                    paragraph_ranges, paragraph_key, paragraph_range
                ) < 0) {
            Py_XDECREF(paragraph_key);
            Py_XDECREF(paragraph_range);
            goto done;
        }
        Py_DECREF(paragraph_key);
        Py_DECREF(paragraph_range);
    }
    *values_result = values;
    *paragraph_ranges_result = paragraph_ranges;
    *sentence_ranges_result = sentence_ranges;
    values = NULL;
    paragraph_ranges = NULL;
    sentence_ranges = NULL;
    status = 1;

done:
    pointer_set_clear(&seen_paragraphs);
    pointer_set_clear(&seen_sentences);
    pointer_set_clear(&seen_words);
    Py_XDECREF(ordered_paragraphs);
    Py_XDECREF(paragraphs);
    Py_XDECREF(paragraph_counts);
    Py_XDECREF(values);
    Py_XDECREF(paragraph_ranges);
    Py_XDECREF(sentence_ranges);
    return status;
}

static PyObject *
document_pair(PyObject *self, PyObject *args)
{
    PyObject *previous;
    PyObject *current;
    PyObject *previous_targets;
    PyObject *current_targets;
    PyObject *prev_values = NULL;
    PyObject *prev_paragraph_ranges = NULL;
    PyObject *prev_sentence_ranges = NULL;
    PyObject *curr_values = NULL;
    PyObject *curr_paragraph_ranges = NULL;
    PyObject *curr_sentence_ranges = NULL;
    PyObject *result = NULL;
    ParagraphCache cache = {0};
    int status;
    if (!PyArg_ParseTuple(
            args, "OOOO:document_pair", &previous, &current,
            &previous_targets, &current_targets)) {
        return NULL;
    }
    status = build_structural_document(
        previous, previous_targets, &cache, 1, current, &prev_values,
        &prev_paragraph_ranges, &prev_sentence_ranges
    );
    if (status < 0) {
        goto error;
    }
    if (!status) {
        paragraph_cache_clear(&cache);
        Py_RETURN_NONE;
    }
    cache.values = prev_values;
    status = build_structural_document(
        current, current_targets, &cache, 0, NULL, &curr_values,
        &curr_paragraph_ranges, &curr_sentence_ranges
    );
    if (status < 0) {
        goto error;
    }
    if (!status) {
        paragraph_cache_clear(&cache);
        Py_DECREF(prev_values);
        Py_DECREF(prev_paragraph_ranges);
        Py_DECREF(prev_sentence_ranges);
        Py_RETURN_NONE;
    }
    result = PyTuple_Pack(
        6, prev_values, prev_paragraph_ranges, prev_sentence_ranges,
        curr_values, curr_paragraph_ranges, curr_sentence_ranges
    );

error:
    paragraph_cache_clear(&cache);
    Py_XDECREF(prev_values);
    Py_XDECREF(prev_paragraph_ranges);
    Py_XDECREF(prev_sentence_ranges);
    Py_XDECREF(curr_values);
    Py_XDECREF(curr_paragraph_ranges);
    Py_XDECREF(curr_sentence_ranges);
    return result;
}

static PyObject *
document_index(PyObject *self, PyObject *args)
{
    PyObject *tokens;
    PyObject *structural_tokens;
    PyObject *keys = NULL;
    PyObject *prefix = NULL;
    PyObject *result = NULL;
    ConstructFrame *stack = NULL;
    uint64_t *prefix_values = NULL;
    Py_ssize_t depth = 0;
    Py_ssize_t length;
    Py_ssize_t index;
    Py_ssize_t information = 0;
    const char *template_stops[] = {"|", "}}"};
    const char *field_stops[] = {"=", "|", "}}"};
    const char *link_stops[] = {"|", "]]"};

    if (!PyArg_ParseTuple(args, "OO:document_index", &tokens,
                          &structural_tokens)) {
        return NULL;
    }
    if (!PyList_Check(tokens) || !PyAnySet_Check(structural_tokens)) {
        PyErr_SetString(PyExc_TypeError, "tokens list and structural set required");
        return NULL;
    }
    length = PyList_GET_SIZE(tokens);
    keys = PyList_New(length);
    if ((size_t)length > (SIZE_MAX / sizeof(uint64_t)) - 1) {
        PyErr_NoMemory();
        goto error;
    }
    prefix = PyByteArray_FromStringAndSize(
        NULL, (length + 1) * (Py_ssize_t)sizeof(uint64_t)
    );
    stack = (ConstructFrame *)calloc(
        (size_t)(length > 0 ? length : 1), sizeof(ConstructFrame)
    );
    if (keys == NULL || prefix == NULL || stack == NULL) {
        if (stack == NULL && !PyErr_Occurred()) {
            PyErr_NoMemory();
        }
        goto error;
    }
    prefix_values = (uint64_t *)PyByteArray_AS_STRING(prefix);
    prefix_values[0] = 0;

    for (index = 0; index < length; index++) {
        PyObject *token = PyList_GET_ITEM(tokens, index);
        PyObject *replacement = NULL;
        int markup_kind = token_markup_kind(token);
        int info = native_informative_token(token, structural_tokens);
        Py_INCREF(token);
        PyList_SET_ITEM(keys, index, token);
        if (info < 0) {
            goto error;
        }
        information += info;
        prefix_values[index + 1] = (uint64_t)information;

        if (markup_kind == TOKEN_MARKUP_TEMPLATE_OPEN) {
            Py_ssize_t end;
            PyObject *name = tokens_until(
                tokens, index + 1, template_stops, 2, &end
            );
            if (name == NULL) {
                goto error;
            }
            if (PyTuple_GET_SIZE(name)) {
                replacement = PyTuple_Pack(
                    4, context_wikitext, token, context_template, name
                );
            }
            stack[depth].type = 1;
            stack[depth].context = name;
            stack[depth].argument_index = 0;
            depth++;
        } else if (markup_kind == TOKEN_MARKUP_TEMPLATE_CLOSE) {
            ConstructFrame frame = {0, NULL, 0};
            if (pop_construct(stack, &depth, 1, &frame)) {
                if (frame.context != NULL &&
                        PyTuple_GET_SIZE(frame.context)) {
                    replacement = PyTuple_Pack(
                        4, context_wikitext, token, context_template,
                        frame.context
                    );
                }
                Py_XDECREF(frame.context);
            }
        } else if (markup_kind == TOKEN_MARKUP_LINK_OPEN) {
            Py_ssize_t end;
            PyObject *target = tokens_until(
                tokens, index + 1, link_stops, 2, &end
            );
            if (target == NULL) {
                goto error;
            }
            if (PyTuple_GET_SIZE(target)) {
                replacement = PyTuple_Pack(
                    4, context_wikitext, token, context_link, target
                );
            }
            stack[depth].type = 2;
            stack[depth].context = target;
            stack[depth].argument_index = 0;
            depth++;
        } else if (markup_kind == TOKEN_MARKUP_LINK_CLOSE) {
            ConstructFrame frame = {0, NULL, 0};
            if (pop_construct(stack, &depth, 2, &frame)) {
                if (frame.context != NULL &&
                        PyTuple_GET_SIZE(frame.context)) {
                    replacement = PyTuple_Pack(
                        4, context_wikitext, token, context_link,
                        frame.context
                    );
                }
                Py_XDECREF(frame.context);
            }
        } else if (markup_kind == TOKEN_MARKUP_COMMENT_OPEN) {
            replacement = PyTuple_Pack(
                3, context_wikitext, token, context_comment
            );
            stack[depth].type = 3;
            stack[depth].context = NULL;
            stack[depth].argument_index = 0;
            depth++;
        } else if (markup_kind == TOKEN_MARKUP_COMMENT_CLOSE) {
            ConstructFrame frame = {0, NULL, 0};
            pop_construct(stack, &depth, 3, &frame);
            replacement = PyTuple_Pack(
                3, context_wikitext, token, context_comment
            );
        } else if (markup_kind == TOKEN_MARKUP_PIPE && depth) {
            ConstructFrame *frame = &stack[depth - 1];
            if (frame->type == 2) {
                Py_ssize_t end;
                PyObject *option = tokens_until(
                    tokens, index + 1, link_stops, 2, &end
                );
                if (option == NULL) {
                    goto error;
                }
                {
                    PyObject *argument = PyLong_FromSsize_t(
                        frame->argument_index
                    );
                    if (argument != NULL) {
                        replacement = PyTuple_Pack(
                            6, context_wikitext, token, context_link,
                            frame->context, argument, option
                        );
                        Py_DECREF(argument);
                    }
                }
                frame->argument_index++;
                Py_DECREF(option);
            } else if (frame->type == 1) {
                Py_ssize_t end;
                PyObject *field = tokens_until(
                    tokens, index + 1, field_stops, 3, &end
                );
                if (field == NULL) {
                    goto error;
                }
                if (end < length &&
                        token_equals_ascii(PyList_GET_ITEM(tokens, end), "=") &&
                        PyTuple_GET_SIZE(field)) {
                    replacement = PyTuple_Pack(
                        5, context_wikitext, token,
                        context_template_field, frame->context, field
                    );
                } else {
                    PyObject *argument = PyLong_FromSsize_t(
                        frame->argument_index
                    );
                    if (argument != NULL) {
                        replacement = PyTuple_Pack(
                            5, context_wikitext, token,
                            context_template_arg, frame->context, argument
                        );
                        Py_DECREF(argument);
                    }
                }
                frame->argument_index++;
                Py_DECREF(field);
            }
        } else if (markup_kind == TOKEN_MARKUP_EQUALS && depth &&
                   stack[depth - 1].type == 1) {
            PyObject *field = template_field_before(tokens, index);
            if (field == NULL) {
                goto error;
            }
            if (field != Py_None) {
                replacement = PyTuple_Pack(
                    5, context_wikitext, token, context_template_field,
                    stack[depth - 1].context, field
                );
            }
            Py_DECREF(field);
        }
        if (replacement != NULL &&
                replace_list_item(keys, index, replacement) < 0) {
            goto error;
        }
    }

    result = PyTuple_Pack(2, keys, prefix);
    if (result == NULL) {
        goto error;
    }
    for (index = 0; index < depth; index++) {
        Py_XDECREF(stack[index].context);
    }
    free(stack);
    Py_DECREF(keys);
    Py_DECREF(prefix);
    return result;

error:
    if (stack != NULL) {
        for (index = 0; index < depth; index++) {
            Py_XDECREF(stack[index].context);
        }
    }
    free(stack);
    Py_XDECREF(keys);
    Py_XDECREF(prefix);
    Py_XDECREF(result);
    return NULL;
}

static PyObject *
token_window_tuple(PyObject *values, Py_ssize_t start, Py_ssize_t size)
{
    PyObject *window = PyTuple_New(size);
    Py_ssize_t offset;
    if (window == NULL) {
        return NULL;
    }
    for (offset = 0; offset < size; offset++) {
        PyObject *item = PyList_GET_ITEM(values, start + offset);
        Py_INCREF(item);
        PyTuple_SET_ITEM(window, offset, item);
    }
    return window;
}

static PyObject *
residual_structural_windows(PyObject *self, PyObject *args)
{
    PyObject *keys;
    PyObject *prefix;
    PyObject *residual_flags;
    Py_ssize_t range_start;
    Py_ssize_t range_end;
    PyObject *allowed;
    PyObject *sizes;
    int minimum_info;
    Py_buffer prefix_view = {0};
    Py_buffer flags_view = {0};
    PyObject *output = NULL;
    const unsigned char *flags;
    Py_ssize_t index;

    if (!PyArg_ParseTuple(
            args, "OOOnnOOi:residual_structural_windows",
            &keys, &prefix, &residual_flags, &range_start, &range_end,
            &allowed, &sizes, &minimum_info)) {
        return NULL;
    }
    if (!PyList_Check(keys) || !PyTuple_Check(sizes) ||
            (allowed != Py_None && !PyAnySet_Check(allowed)) ||
            range_start < 0 || range_end < range_start ||
            range_end > PyList_GET_SIZE(keys)) {
        PyErr_SetString(PyExc_ValueError,
                        "invalid residual structural window inputs");
        return NULL;
    }
    if (PyObject_GetBuffer(prefix, &prefix_view, PyBUF_CONTIG_RO) < 0 ||
            PyObject_GetBuffer(
                residual_flags, &flags_view, PyBUF_CONTIG_RO
            ) < 0) {
        goto error;
    }
    if (prefix_view.len !=
            (PyList_GET_SIZE(keys) + 1) *
                (Py_ssize_t)sizeof(uint64_t) ||
            flags_view.len < PyList_GET_SIZE(keys)) {
        PyErr_SetString(PyExc_ValueError,
                        "invalid structural prefix or residual flags");
        goto error;
    }
    flags = (const unsigned char *)flags_view.buf;
    output = PySet_New(NULL);
    if (output == NULL) {
        goto error;
    }
    index = range_start;
    while (index < range_end) {
        Py_ssize_t run_start;
        Py_ssize_t size_index;
        while (index < range_end && !flags[index]) {
            index++;
        }
        run_start = index;
        while (index < range_end && flags[index]) {
            index++;
        }
        if (run_start == index) {
            continue;
        }
        for (size_index = 0; size_index < PyTuple_GET_SIZE(sizes);
                size_index++) {
            Py_ssize_t size = PyLong_AsSsize_t(
                PyTuple_GET_ITEM(sizes, size_index)
            );
            Py_ssize_t start;
            if (size <= 0 || (size == -1 && PyErr_Occurred())) {
                PyErr_SetString(PyExc_ValueError,
                                "invalid structural window size");
                goto error;
            }
            for (start = run_start; start + size <= index; start++) {
                uint64_t prefix_start;
                uint64_t prefix_end;
                PyObject *window;
                int retained;
                memcpy(
                    &prefix_start,
                    (const unsigned char *)prefix_view.buf +
                        (size_t)start * sizeof(uint64_t),
                    sizeof(uint64_t)
                );
                memcpy(
                    &prefix_end,
                    (const unsigned char *)prefix_view.buf +
                        (size_t)(start + size) * sizeof(uint64_t),
                    sizeof(uint64_t)
                );
                if (prefix_end - prefix_start < (uint64_t)minimum_info) {
                    continue;
                }
                window = token_window_tuple(keys, start, size);
                if (window == NULL) {
                    goto error;
                }
                retained = allowed == Py_None ? 1 :
                    PySet_Contains(allowed, window);
                if (retained < 0 ||
                        (retained && PySet_Add(output, window) < 0)) {
                    Py_DECREF(window);
                    goto error;
                }
                Py_DECREF(window);
            }
        }
    }
    PyBuffer_Release(&prefix_view);
    PyBuffer_Release(&flags_view);
    return output;

error:
    if (prefix_view.obj != NULL) {
        PyBuffer_Release(&prefix_view);
    }
    if (flags_view.obj != NULL) {
        PyBuffer_Release(&flags_view);
    }
    Py_XDECREF(output);
    return NULL;
}

static PyObject *
unresolved_residual_windows(PyObject *self, PyObject *args)
{
    PyObject *prev_values;
    PyObject *curr_values;
    PyObject *prev_used;
    PyObject *curr_mapping;
    PyObject *structural_tokens;
    PyObject *sizes;
    int minimum_info;
    unsigned char *prev_available = NULL;
    unsigned char *curr_available = NULL;
    unsigned char *prev_informative = NULL;
    unsigned char *curr_informative = NULL;
    PyObject *triplets = NULL;
    PyObject *previous_windows = NULL;
    PyObject *output = NULL;
    Py_ssize_t prev_len;
    Py_ssize_t curr_len;
    Py_ssize_t index;

    if (!PyArg_ParseTuple(
            args, "OOOOOOi:unresolved_residual_windows",
            &prev_values, &curr_values, &prev_used, &curr_mapping,
            &structural_tokens, &sizes, &minimum_info)) {
        return NULL;
    }
    if (!PyList_Check(prev_values) || !PyList_Check(curr_values) ||
            !PyDict_Check(prev_used) || !PyList_Check(curr_mapping) ||
            !PyAnySet_Check(structural_tokens) || !PyTuple_Check(sizes)) {
        PyErr_SetString(PyExc_TypeError, "invalid unresolved-window inputs");
        return NULL;
    }
    prev_len = PyList_GET_SIZE(prev_values);
    curr_len = PyList_GET_SIZE(curr_values);
    if (PyList_GET_SIZE(curr_mapping) != curr_len) {
        PyErr_SetString(PyExc_ValueError, "current mapping length mismatch");
        return NULL;
    }
    prev_available = (unsigned char *)malloc(
        (size_t)(prev_len > 0 ? prev_len : 1)
    );
    curr_available = (unsigned char *)malloc(
        (size_t)(curr_len > 0 ? curr_len : 1)
    );
    prev_informative = (unsigned char *)malloc(
        (size_t)(prev_len > 0 ? prev_len : 1)
    );
    curr_informative = (unsigned char *)malloc(
        (size_t)(curr_len > 0 ? curr_len : 1)
    );
    if (prev_available == NULL || curr_available == NULL ||
            prev_informative == NULL || curr_informative == NULL) {
        PyErr_NoMemory();
        goto error;
    }
    memset(prev_available, 1, (size_t)prev_len);
    {
        Py_ssize_t iteration = 0;
        PyObject *key;
        PyObject *value;
        while (PyDict_Next(prev_used, &iteration, &key, &value)) {
            Py_ssize_t used_index = PyLong_AsSsize_t(key);
            if (used_index == -1 && PyErr_Occurred()) {
                goto error;
            }
            if (used_index >= 0 && used_index < prev_len) {
                prev_available[used_index] = 0;
            }
        }
    }
    for (index = 0; index < prev_len; index++) {
        int info = native_informative_token(
            PyList_GET_ITEM(prev_values, index), structural_tokens
        );
        if (info < 0) {
            goto error;
        }
        prev_informative[index] = info ? 1 : 0;
    }
    for (index = 0; index < curr_len; index++) {
        int info = native_informative_token(
            PyList_GET_ITEM(curr_values, index), structural_tokens
        );
        if (info < 0) {
            goto error;
        }
        curr_available[index] = PyList_GET_ITEM(curr_mapping, index) == Py_None;
        curr_informative[index] = info ? 1 : 0;
    }

    triplets = PySet_New(NULL);
    if (triplets == NULL) {
        goto error;
    }
    {
        Py_ssize_t run = 0;
        for (index = 0; index < prev_len; index++) {
            if (!prev_available[index]) {
                run = 0;
                continue;
            }
            run++;
            if (run >= 3) {
                PyObject *window = token_window_tuple(
                    prev_values, index - 2, 3
                );
                if (window == NULL || PySet_Add(triplets, window) < 0) {
                    Py_XDECREF(window);
                    goto error;
                }
                Py_DECREF(window);
            }
        }
    }
    if (PySet_Size(triplets) == 0) {
        output = PySet_New(NULL);
        goto success;
    }
    {
        Py_ssize_t run = 0;
        int shared = 0;
        for (index = 0; index < curr_len; index++) {
            if (!curr_available[index]) {
                run = 0;
                continue;
            }
            run++;
            if (run >= 3) {
                PyObject *window = token_window_tuple(
                    curr_values, index - 2, 3
                );
                int present;
                if (window == NULL) {
                    goto error;
                }
                present = PySet_Contains(triplets, window);
                Py_DECREF(window);
                if (present < 0) {
                    goto error;
                }
                if (present) {
                    shared = 1;
                    break;
                }
            }
        }
        if (!shared) {
            output = PySet_New(NULL);
            goto success;
        }
    }

    previous_windows = PySet_New(NULL);
    output = PySet_New(NULL);
    if (previous_windows == NULL || output == NULL) {
        goto error;
    }
    {
        Py_ssize_t run_start = -1;
        for (index = 0; index <= prev_len; index++) {
            if (index < prev_len && prev_available[index]) {
                if (run_start < 0) {
                    run_start = index;
                }
                continue;
            }
            if (run_start >= 0) {
                Py_ssize_t size_index;
                for (size_index = 0;
                        size_index < PyTuple_GET_SIZE(sizes); size_index++) {
                    Py_ssize_t size = PyLong_AsSsize_t(
                        PyTuple_GET_ITEM(sizes, size_index)
                    );
                    Py_ssize_t start;
                    if (size <= 0 || (size == -1 && PyErr_Occurred())) {
                        PyErr_SetString(PyExc_ValueError, "invalid window size");
                        goto error;
                    }
                    for (start = run_start; start + size <= index; start++) {
                        Py_ssize_t offset;
                        int info = 0;
                        PyObject *window;
                        for (offset = 0; offset < size; offset++) {
                            info += prev_informative[start + offset];
                        }
                        if (info < minimum_info) {
                            continue;
                        }
                        window = token_window_tuple(prev_values, start, size);
                        if (window == NULL ||
                                PySet_Add(previous_windows, window) < 0) {
                            Py_XDECREF(window);
                            goto error;
                        }
                        Py_DECREF(window);
                    }
                }
                run_start = -1;
            }
        }
    }
    if (PySet_Size(previous_windows) == 0) {
        goto success;
    }
    {
        Py_ssize_t run_start = -1;
        for (index = 0; index <= curr_len; index++) {
            if (index < curr_len && curr_available[index]) {
                if (run_start < 0) {
                    run_start = index;
                }
                continue;
            }
            if (run_start >= 0) {
                Py_ssize_t size_index;
                for (size_index = 0;
                        size_index < PyTuple_GET_SIZE(sizes); size_index++) {
                    Py_ssize_t size = PyLong_AsSsize_t(
                        PyTuple_GET_ITEM(sizes, size_index)
                    );
                    Py_ssize_t start;
                    for (start = run_start; start + size <= index; start++) {
                        Py_ssize_t offset;
                        int info = 0;
                        PyObject *window;
                        int present;
                        for (offset = 0; offset < size; offset++) {
                            info += curr_informative[start + offset];
                        }
                        if (info < minimum_info) {
                            continue;
                        }
                        window = token_window_tuple(curr_values, start, size);
                        if (window == NULL) {
                            goto error;
                        }
                        present = PySet_Contains(previous_windows, window);
                        if (present > 0 && PySet_Add(output, window) < 0) {
                            Py_DECREF(window);
                            goto error;
                        }
                        Py_DECREF(window);
                        if (present < 0) {
                            goto error;
                        }
                    }
                }
                run_start = -1;
            }
        }
    }

success:
    if (output == NULL) {
        goto error;
    }
    Py_XDECREF(triplets);
    Py_XDECREF(previous_windows);
    free(prev_available);
    free(curr_available);
    free(prev_informative);
    free(curr_informative);
    return output;

error:
    Py_XDECREF(triplets);
    Py_XDECREF(previous_windows);
    Py_XDECREF(output);
    free(prev_available);
    free(curr_available);
    free(prev_informative);
    free(curr_informative);
    return NULL;
}

static PyObject *
unique_anchor_occurrences(PyObject *self, PyObject *args)
{
    PyObject *prev_keys;
    PyObject *curr_keys;
    PyObject *prev_ranges_obj;
    PyObject *curr_ranges_obj;
    PyObject *prev_prefix_obj;
    PyObject *curr_prefix_obj;
    PyObject *prev_targets;
    PyObject *curr_targets;
    PyObject *sizes_obj;
    int minimum_info;
    int include_all;
    uint64_t *prev_hash_prefix = NULL;
    uint64_t *curr_hash_prefix = NULL;
    Py_ssize_t *prev_prefix = NULL;
    Py_ssize_t *curr_prefix = NULL;
    Range *prev_ranges = NULL;
    Range *curr_ranges = NULL;
    Py_ssize_t prev_range_count = 0;
    Py_ssize_t curr_range_count = 0;
    PyObject *output = NULL;
    AnchorSegment *segments = NULL;
    size_t segment_count = 0;
    size_t segment_capacity = 0;
    Py_ssize_t size_index;

    if (!PyArg_ParseTuple(
            args, "OOOOOOOOOip:unique_anchor_occurrences",
            &prev_keys, &curr_keys, &prev_ranges_obj, &curr_ranges_obj,
            &prev_prefix_obj, &curr_prefix_obj, &prev_targets, &curr_targets,
            &sizes_obj, &minimum_info, &include_all)) {
        return NULL;
    }
    if (!PyList_Check(prev_keys) || !PyList_Check(curr_keys) ||
            !PyTuple_Check(sizes_obj)) {
        PyErr_SetString(PyExc_TypeError, "keys must be lists and sizes a tuple");
        return NULL;
    }
    if (copy_rolling_hash_prefix(prev_keys, &prev_hash_prefix) < 0 ||
            copy_rolling_hash_prefix(curr_keys, &curr_hash_prefix) < 0 ||
            copy_prefix(prev_prefix_obj, PyList_GET_SIZE(prev_keys),
                        &prev_prefix) < 0 ||
            copy_prefix(curr_prefix_obj, PyList_GET_SIZE(curr_keys),
                        &curr_prefix) < 0 ||
            copy_ranges(prev_ranges_obj, prev_targets, &prev_ranges,
                        &prev_range_count) < 0 ||
            copy_ranges(curr_ranges_obj, curr_targets, &curr_ranges,
                        &curr_range_count) < 0) {
        goto error;
    }
    for (size_index = 0; size_index < PyTuple_GET_SIZE(sizes_obj);
            size_index++) {
        Py_ssize_t size = PyLong_AsSsize_t(
            PyTuple_GET_ITEM(sizes_obj, size_index)
        );
        Table table = {NULL, 0, 0};
        Py_ssize_t range_index;
        size_t entry_index;
        size_t expected = (size_t)PyList_GET_SIZE(prev_keys);
        uint64_t base_power = 1;
        Py_ssize_t power_index;
        if (size <= 0 || (size == -1 && PyErr_Occurred())) {
            table_clear(&table);
            PyErr_SetString(PyExc_ValueError, "anchor sizes must be positive");
            goto error;
        }
        for (power_index = 0; power_index < size; power_index++) {
            base_power *= STRUCTURAL_ROLLING_BASE;
        }
        if (!include_all) {
            expected = 0;
            for (range_index = 0; range_index < prev_range_count;
                    range_index++) {
                Range *range = &prev_ranges[range_index];
                if (range->target && range->end - range->start >= size) {
                    expected += (size_t)(
                        range->end - range->start - size + 1
                    );
                }
            }
            for (range_index = 0; range_index < curr_range_count;
                    range_index++) {
                Range *range = &curr_ranges[range_index];
                if (range->target && range->end - range->start >= size) {
                    expected += (size_t)(
                        range->end - range->start - size + 1
                    );
                }
            }
        }
        if (table_init(&table, expected) < 0) {
            goto error;
        }
        if (include_all) {
            for (range_index = 0; range_index < prev_range_count;
                    range_index++) {
                Range *range = &prev_ranges[range_index];
                Py_ssize_t start;
                for (start = range->start;
                        start + size <= range->end; start++) {
                    if (prev_prefix[start + size] - prev_prefix[start] <
                            minimum_info) {
                        continue;
                    }
                    if (table_add_previous(
                            &table, prev_keys, start, size,
                            range->paragraph, start - range->start,
                            range->target, rolling_window_hash(
                                prev_hash_prefix, start, size, base_power
                            )) < 0) {
                        table_clear(&table);
                        goto error;
                    }
                }
            }
            for (range_index = 0; range_index < curr_range_count;
                    range_index++) {
                Range *range = &curr_ranges[range_index];
                Py_ssize_t start;
                for (start = range->start;
                        start + size <= range->end; start++) {
                    if (curr_prefix[start + size] - curr_prefix[start] <
                            minimum_info) {
                        continue;
                    }
                    if (table_add_current(
                            &table, prev_keys, curr_keys, start, size,
                            range->paragraph, start - range->start,
                            range->target, rolling_window_hash(
                                curr_hash_prefix, start, size, base_power
                            )) < 0) {
                        table_clear(&table);
                        goto error;
                    }
                }
            }
        } else {
            /* Build only the candidate universe incident to a residual
             * paragraph.  Global scans below still establish exact revision-
             * wide uniqueness for every retained key. */
            for (range_index = 0; range_index < prev_range_count;
                    range_index++) {
                Range *range = &prev_ranges[range_index];
                Py_ssize_t start;
                if (!range->target) {
                    continue;
                }
                for (start = range->start;
                        start + size <= range->end; start++) {
                    if (prev_prefix[start + size] - prev_prefix[start] <
                            minimum_info) {
                        continue;
                    }
                    if (table_add_target_candidate(
                            &table, prev_keys, start, size,
                            rolling_window_hash(
                                prev_hash_prefix, start, size, base_power
                            )) < 0) {
                        table_clear(&table);
                        goto error;
                    }
                }
            }
            for (range_index = 0; range_index < curr_range_count;
                    range_index++) {
                Range *range = &curr_ranges[range_index];
                Py_ssize_t start;
                if (!range->target) {
                    continue;
                }
                for (start = range->start;
                        start + size <= range->end; start++) {
                    if (curr_prefix[start + size] - curr_prefix[start] <
                            minimum_info) {
                        continue;
                    }
                    if (table_add_target_candidate(
                            &table, curr_keys, start, size,
                            rolling_window_hash(
                                curr_hash_prefix, start, size, base_power
                            )) < 0) {
                        table_clear(&table);
                        goto error;
                    }
                }
            }
            if (table.used) {
                for (range_index = 0; range_index < prev_range_count;
                        range_index++) {
                    Range *range = &prev_ranges[range_index];
                    Py_ssize_t start;
                    for (start = range->start;
                            start + size <= range->end; start++) {
                        if (prev_prefix[start + size] - prev_prefix[start] <
                                minimum_info) {
                            continue;
                        }
                        if (table_count_target_previous(
                                &table, prev_keys, start, size,
                                range->paragraph, start - range->start,
                                rolling_window_hash(
                                    prev_hash_prefix, start, size, base_power
                                )) < 0) {
                            table_clear(&table);
                            goto error;
                        }
                    }
                }
                for (range_index = 0; range_index < curr_range_count;
                        range_index++) {
                    Range *range = &curr_ranges[range_index];
                    Py_ssize_t start;
                    for (start = range->start;
                            start + size <= range->end; start++) {
                        if (curr_prefix[start + size] - curr_prefix[start] <
                                minimum_info) {
                            continue;
                        }
                        if (table_count_target_current(
                                &table, curr_keys, start, size,
                                range->paragraph, start - range->start,
                                rolling_window_hash(
                                    curr_hash_prefix, start, size, base_power
                                )) < 0) {
                            table_clear(&table);
                            goto error;
                        }
                    }
                }
            }
        }
        for (entry_index = 0; entry_index < table.capacity; entry_index++) {
            Entry *entry = &table.entries[entry_index];
            if (!entry->occupied || entry->prev_count != 1 ||
                    entry->curr_count != 1) {
                continue;
            }
            if (segment_count == segment_capacity) {
                size_t new_capacity = segment_capacity ?
                    segment_capacity * 2 : 1024;
                AnchorSegment *resized = (AnchorSegment *)realloc(
                    segments, new_capacity * sizeof(AnchorSegment)
                );
                if (resized == NULL) {
                    PyErr_NoMemory();
                    table_clear(&table);
                    goto error;
                }
                segments = resized;
                segment_capacity = new_capacity;
            }
            segments[segment_count].prev_paragraph = entry->prev_paragraph;
            segments[segment_count].prev_start = entry->prev_local;
            segments[segment_count].prev_end = entry->prev_local + size;
            segments[segment_count].curr_paragraph = entry->curr_paragraph;
            segments[segment_count].curr_start = entry->curr_local;
            segments[segment_count].curr_end = entry->curr_local + size;
            segment_count++;
        }
        table_clear(&table);
    }

    output = PyList_New(0);
    if (output == NULL) {
        goto error;
    }
    if (segment_count) {
        size_t index;
        AnchorSegment merged;
        qsort(
            segments, segment_count, sizeof(AnchorSegment),
            compare_anchor_segments
        );
        merged = segments[0];
        for (index = 1; index < segment_count; index++) {
            AnchorSegment *next = &segments[index];
            Py_ssize_t merged_diagonal = merged.prev_start - merged.curr_start;
            Py_ssize_t next_diagonal = next->prev_start - next->curr_start;
            if (merged.prev_paragraph == next->prev_paragraph &&
                    merged.curr_paragraph == next->curr_paragraph &&
                    merged_diagonal == next_diagonal &&
                    next->prev_start <= merged.prev_end &&
                    next->curr_start <= merged.curr_end) {
                if (next->prev_end > merged.prev_end) {
                    merged.prev_end = next->prev_end;
                }
                if (next->curr_end > merged.curr_end) {
                    merged.curr_end = next->curr_end;
                }
                continue;
            }
            if (append_anchor_segment(output, &merged) < 0) {
                goto error;
            }
            merged = *next;
        }
        if (append_anchor_segment(output, &merged) < 0) {
            goto error;
        }
    }

    free(prev_hash_prefix);
    free(curr_hash_prefix);
    free(prev_prefix);
    free(curr_prefix);
    free(prev_ranges);
    free(curr_ranges);
    free(segments);
    return output;

error:
    Py_XDECREF(output);
    free(prev_hash_prefix);
    free(curr_hash_prefix);
    free(prev_prefix);
    free(curr_prefix);
    free(prev_ranges);
    free(curr_ranges);
    free(segments);
    return NULL;
}

static PyMethodDef methods[] = {
    {
        "configure_token_symbols",
        configure_token_symbols,
        METH_O,
        "Configure the exact symbol alphabet used by the native tokenizer."
    },
    {
        "split_into_tokens",
        split_into_tokens_native,
        METH_O,
        "Split wikitext using the exact WikiWho token contract."
    },
    {
        "split_into_paragraphs",
        split_into_paragraphs_native,
        METH_O,
        "Split wikitext using the exact WikiWho paragraph contract."
    },
    {
        "count_subsequence_at_positions",
        count_subsequence_at_positions,
        METH_VARARGS,
        "Count an exact token subsequence up to two occurrences."
    },
    {
        "unique_anchor_occurrences",
        unique_anchor_occurrences,
        METH_VARARGS,
        "Find exact globally unique structural anchor occurrences."
    },
    {
        "duplicated_candidate_windows",
        duplicated_candidate_windows,
        METH_VARARGS,
        "Return exact candidate windows occurring at least twice."
    },
    {
        "lcs_token_pairs",
        lcs_token_pairs,
        METH_VARARGS,
        "Return the exact lexicographically best bounded gap alignment."
    },
    {
        "available_residual_windows",
        available_residual_windows,
        METH_VARARGS,
        "Return exact available raw-value windows within paragraph bounds."
    },
    {
        "document_pair",
        document_pair,
        METH_VARARGS,
        "Build exact compact structural documents from adjacent revisions."
    },
    {
        "document_index",
        document_index,
        METH_VARARGS,
        "Build exact contextual token keys and informative prefix in one pass."
    },
    {
        "residual_structural_windows",
        residual_structural_windows,
        METH_VARARGS,
        "Return exact contextual windows inside residual-only gap runs."
    },
    {
        "unresolved_residual_windows",
        unresolved_residual_windows,
        METH_VARARGS,
        "Return the exact available residual ambiguity-window intersection."
    },
    {NULL, NULL, 0, NULL}
};

static struct PyModuleDef module = {
    PyModuleDef_HEAD_INIT,
    "_structural_native",
    NULL,
    -1,
    methods
};

PyMODINIT_FUNC
PyInit__structural_native(void)
{
    PyObject *created = PyModule_Create(&module);
    if (created == NULL) {
        return NULL;
    }
    context_wikitext = PyUnicode_InternFromString("wikitext");
    context_template = PyUnicode_InternFromString("template");
    context_link = PyUnicode_InternFromString("link");
    context_comment = PyUnicode_InternFromString("comment");
    context_template_field = PyUnicode_InternFromString("template-field");
    context_template_arg = PyUnicode_InternFromString("template-arg");
    attr_ordered_paragraphs = PyUnicode_InternFromString(
        "ordered_paragraphs"
    );
    attr_paragraphs = PyUnicode_InternFromString("paragraphs");
    attr_ordered_sentences = PyUnicode_InternFromString(
        "ordered_sentences"
    );
    attr_sentences = PyUnicode_InternFromString("sentences");
    attr_words = PyUnicode_InternFromString("words");
    attr_splitted = PyUnicode_InternFromString("splitted");
    attr_value = PyUnicode_InternFromString("value");
    pipe_token = PyUnicode_InternFromString("|");
    if (context_wikitext == NULL || context_template == NULL ||
            context_link == NULL || context_comment == NULL ||
            context_template_field == NULL || context_template_arg == NULL ||
            attr_ordered_paragraphs == NULL || attr_paragraphs == NULL ||
            attr_ordered_sentences == NULL || attr_sentences == NULL ||
            attr_words == NULL || attr_splitted == NULL ||
            attr_value == NULL || pipe_token == NULL) {
        Py_XDECREF(context_wikitext);
        Py_XDECREF(context_template);
        Py_XDECREF(context_link);
        Py_XDECREF(context_comment);
        Py_XDECREF(context_template_field);
        Py_XDECREF(context_template_arg);
        Py_XDECREF(attr_ordered_paragraphs);
        Py_XDECREF(attr_paragraphs);
        Py_XDECREF(attr_ordered_sentences);
        Py_XDECREF(attr_sentences);
        Py_XDECREF(attr_words);
        Py_XDECREF(attr_splitted);
        Py_XDECREF(attr_value);
        Py_XDECREF(pipe_token);
        context_wikitext = NULL;
        context_template = NULL;
        context_link = NULL;
        context_comment = NULL;
        context_template_field = NULL;
        context_template_arg = NULL;
        attr_ordered_paragraphs = NULL;
        attr_paragraphs = NULL;
        attr_ordered_sentences = NULL;
        attr_sentences = NULL;
        attr_words = NULL;
        attr_splitted = NULL;
        attr_value = NULL;
        pipe_token = NULL;
        Py_DECREF(created);
        return NULL;
    }
    return created;
}
