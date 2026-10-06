# The writing standard: ASD-STE100 Simplified Technical English

Use these rules for the `README.md` of onboardiq and for this file. Section 3 gives the project
vocabulary. Each term in Section 3 has one meaning in all of the documentation.

## 1. The writing rules

### Words

1. Use one word for one meaning, and one meaning for one word. Do not use synonyms for variety.
2. Use a word only as one part of speech. For example, "test" is a noun or a verb, "check" is a verb.
3. Do not use phrasal verbs (`set up`, `carry out`, `find out`, `pick up`, `look up`, `come up with`).
   Use one verb: "prepare", "do", "find", "get", "make".
4. Do not use an "-ing" form as a noun or an adjective ("the running job", "after indexing").
   Exception: a technical name, a file name, a command or a status value.
5. Do not use contractions (`don't`, `it's`, `can't`). Do not use slang or idioms
   (`out of the box`, `under the hood`, `at a glance`, "gotcha", "bells and whistles").
6. Do not use `and/or`. Write "A, B or both".
7. Do not use `should`, `could`, `would` or `may` for instructions. Use "must" for a rule, the
   imperative for a step and "can" for a possibility.
8. Keep the articles "a", "an" and "the" in sentences.
9. Do not make a noun cluster of more than three words. A technical name is one word.

### Sentences

1. A procedural sentence (an instruction) has a maximum of **20 words**.
2. A descriptive sentence has a maximum of **25 words**.
3. Write one instruction in one sentence.
4. Use the imperative for an instruction: "Run the tests." Not `The tests should be run.`
5. Use the active voice. Use the passive voice only when the agent of the action is not important.
6. Use only the simple present, the simple past and the simple future.
7. Put a condition before the instruction: "If the index is stale, build it again."
8. Do not use semicolons in sentences. Write two sentences.

### Paragraphs, notes and warnings

1. A paragraph has one topic and a maximum of **6 sentences**. Start with the topic sentence.
2. A warning or a caution starts with a clear command. Then it gives the reason.
3. A note gives information. It does not give an instruction.
4. Use a vertical list for a sequence or a set of conditions. Each item of a numbered procedure is one step.

### Tables, headings and diagrams

1. A table cell can be a short phrase. If a cell has a sentence, the sentence obeys the rules.
2. A heading is a noun phrase ("The cost model") or an imperative ("Run the demo").
   Do not start a heading with an "-ing" form.
3. A diagram label is a short phrase. Use the same terms as the text.

### What STE does not change

Code, commands, file names, paths, field names, environment variables, status values, enum values,
product names and URLs stay exactly as they are. They are technical names. Put them in backticks.

## 2. General words to replace

| Do not use | Use |
|---|---|
| utilize, leverage | use |
| in order to | to |
| set up | prepare, install, configure |
| carry out, perform | do |
| make sure, ensure | make sure (allowed), or "check that" |
| a lot of, lots of | many, much |
| e.g., i.e. | for example, that is |
| should (instruction) | must (rule) / imperative (step) |
| might, may (possibility) | can |
| very, really, just, simply, easily | (delete) |
| seamless, robust, powerful, blazing | (delete or give a measured fact) |

## 3. Project vocabulary

These terms have one meaning in the onboardiq documentation. The code names are in backticks.

### 3.1 Technical names (nouns)

| Term | Meaning | Do not use |
|---|---|---|
| **user** | The person who asks a question: a new member of a data team. | reader, joiner, employee, customer |
| **document** | One source file in the document folder: `.md`, `.markdown`, `.txt` or `.pdf`. | file (for a source), article, page |
| **document folder** | The folder that onboardiq reads documents from (`ONBOARDIQ_DOCS_DIR`, default `sample_docs`). | corpus folder, knowledge base |
| **corpus** | All documents in the document folder, as one set. | knowledge base, dataset |
| **front-matter** | The `---` block at the top of a document with the `role`, `level` and `topic` keys. | header, metadata block, YAML |
| **role** | The job of the user: `data_scientist`, `data_engineer` or `data_analyst`. Also the tag on a chunk. | persona, job title, audience |
| **level** | The seniority of the user: `junior`, `mid` or `senior`. Also the tag on a chunk. | seniority, grade, rank |
| **tag** | A normalized role value or level value. The value `all` matches every user. | label, category |
| **section** | The text of a document under one Markdown heading. | part, block |
| **heading path** | The headings above a section, joined with ` > `, for example `pipelines.md > Building and Running Pipelines > Backfills`. | breadcrumb, title |
| **chunk** | One unit of text that onboardiq stores and retrieves. A chunk is part of one section. | segment, piece, split, node |
| **overlap** | The text from the end of a chunk that the next chunk of the same section starts with. | carry-over, stride, window |
| **index** | The four files that `onboardiq index` writes in the index folder: chunks, BM25 statistics, vectors and the manifest. | database, vector store, cache |
| **manifest** | The file `manifest.json` in the index. It holds the fingerprint and the embedder name. | metadata file, header |
| **fingerprint** | A SHA-256 value of the documents, the chunk settings, the tokenizer version and the embedder name. | hash (alone), checksum, signature |
| **token** | A lower-case word that the tokenizer gives. BM25 and the hashing embedder use tokens. | term (in prose), keyword |
| **embedder** | The provider that changes text into a vector. | encoder, embedding service |
| **vector** | The list of numbers that the embedder gives for one text. Each vector has a length of 1. | embedding (as a noun), dense representation |
| **chat model** | The provider that writes the answer from the prompt. | LLM (in prose), generator, bot |
| **provider** | An embedder or a chat model behind one interface: offline, `openai` or `ollama`. | backend, engine, vendor |
| **offline provider** | The hashing embedder or the echo chat model. Both work with no key and no network. | fake, mock, stub (in prose) |
| **candidate** | A chunk that BM25 or vector search returns before fusion. | result, match |
| **fusion** | Reciprocal-rank fusion (RRF): one ranked list from the BM25 list and the vector list. | merge, blend, combination |
| **hit** | A chunk after fusion, with its scores and its ranks. | result, document |
| **passage** | A hit that onboardiq puts in the prompt with a number `[n]`. | context chunk, snippet, excerpt |
| **filter stage** | The step of the metadata filter that gave the allowed chunks: `role+level`, `role` or `none`. | fallback level, tier |
| **evidence check** | The rule that decides if the hits support an answer. | answerability check, guard |
| **refusal** | The fixed answer that onboardiq gives when the evidence check fails. | rejection, fallback answer |
| **marker** | A citation number in square brackets in the answer, for example `[2]`. | reference, footnote, tag |
| **citation** | A valid marker together with the chunk that it points to. | source (alone), reference |
| **source list** | The `Sources:` block after the answer. It has one line for each citation. | bibliography, references |
| **history window** | The last exchanges of the conversation that onboardiq puts in the prompt. | memory, context window |
| **feedback row** | One row in the SQLite table `feedback`, with one rating for one answer. | review, vote, record |
| **rating** | The value `helpful` or `not_helpful` in a feedback row. | score, vote, grade |
| **golden set** | The file `eval/golden_set.jsonl`: questions with the section that answers each one. | test set, benchmark, ground truth |
| **target** | One expected `source` and `heading` pair for a golden-set question. | label, answer key |

### 3.2 Technical verbs

| Verb | Meaning |
|---|---|
| **load** | Read the documents from the document folder, or read a saved index from the index folder. |
| **split** | Cut a document into sections, or cut a section into chunks. |
| **embed** | Change text into a vector with the embedder. |
| **build** | Make a new index from the corpus and save it. |
| **reuse** | Load the saved index because the fingerprint did not change. |
| **filter** | Keep only the chunks whose tags match the role and the level of the user. |
| **widen** | Go to the next, less strict filter stage when the current stage keeps no chunk. |
| **retrieve** | Get the hits for a question from the index. |
| **fuse** | Make one ranked list from two ranked lists with reciprocal-rank fusion. |
| **refuse** | Give the refusal and do not send a prompt to the chat model. |
| **cite** | Put a marker in the answer for a passage. |
| **validate** | Check each marker against the passages and remove a marker that points to no passage. |
| **escape** | Replace the HTML characters `<`, `>`, `&` and quotes with HTML entities. |
| **submit** | Send a rating, and an optional comment, to the feedback store. |
| **export** | Write all feedback rows to a CSV file. |
| **evaluate** | Measure recall@k and MRR of the retrieval on the golden set. |
