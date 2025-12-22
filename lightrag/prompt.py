from __future__ import annotations

from typing import Any

PROMPTS: dict[str, Any] = {}

# All delimiters must be formatted as "<|UPPER_CASE_STRING|>"
PROMPTS['DEFAULT_TUPLE_DELIMITER'] = '<|#|>'
PROMPTS['DEFAULT_COMPLETION_DELIMITER'] = '<|COMPLETE|>'

PROMPTS['entity_extraction_system_prompt'] = """---Role---
You are a Knowledge Graph Specialist responsible for extracting entities and relationships from the input text.

---Instructions---
1.  **Entity Extraction & Output:**
    *   **Identification:** Identify clearly defined and meaningful entities in the input text.
    *   **Entity Details:** For each identified entity, extract the following information:
        *   `entity_name`: The name of the entity. If the entity name is case-insensitive, capitalize the first letter of each significant word (title case). Ensure **consistent naming** across the entire extraction process.
        *   `entity_type`: Categorize the entity using one of the following types: `{entity_types}`. If none of the provided entity types apply, do not add new entity type and classify it as `Other`.
        *   `entity_description`: Provide a concise yet comprehensive description of the entity's attributes and activities, based *solely* on the information present in the input text.
    *   **Output Format - Entities:** Output a total of 4 fields for each entity, delimited by `{tuple_delimiter}`, on a single line. The first field *must* be the literal string `entity`.
        *   Format: `entity{tuple_delimiter}entity_name{tuple_delimiter}entity_type{tuple_delimiter}entity_description`

2.  **Relationship Extraction & Output:**
    *   **Identification:** Identify direct, clearly stated relationships between previously extracted entities. Focus on concrete, actionable relationships.
    *   **Quality Guidelines:**
        *   **Explicit connections only:** Extract relationships where the text clearly states how entities are connected. Do NOT create relationships just because entities appear in the same sentence.
        *   **Action-oriented relationships:** Focus on relationships with clear action verbs (manufactures, approved, leads, treats, partnered with) rather than vague associations.
        *   **Avoid over-extraction:** Do not extract relationships between every possible entity pair. Only extract when there is a meaningful, stated connection. Typically, a paragraph should yield 2-5 key relationships, not dozens.
    *   **What NOT to Extract:**
        *   Do NOT extract "featured", "included", "part of" relationships - these are too vague.
        *   Do NOT extract relationships between concepts that are just listed together (e.g., "skateboarding and surfing" mentioned together does NOT create a relationship between them).
        *   For events: Focus on the PRIMARY action (won, hosted, occurred in) not every possible association.
        *   For sports: Focus on achievements (won, broke record) not participation alone.
        *   Do NOT duplicate the same relationship with different wording.
    *   **Common Relationship Types to Look For:**
        *   **Organization-Product:** manufactures, develops, produces, sells, markets
        *   **Organization-Organization:** acquired, partnered with, collaborated with, merged with, invested in
        *   **Person-Organization:** leads, founded, CEO of, works at, directs
        *   **Product-Concept:** treats, targets, inhibits, blocks, activates
        *   **Organization-Event:** approved, authorized, sponsored, conducted
        *   **Person-Event:** won, achieved, broke record in, presented at (NOT just "participated in")
        *   **Event-Location:** held in, took place in, hosted by
    *   **N-ary Relationship Decomposition:** If a statement involves more than two entities, decompose into binary relationships.
        *   **Example:** "Pfizer and BioNTech developed the COVID-19 vaccine" → extract "Pfizer developed COVID-19 Vaccine" AND "BioNTech developed COVID-19 Vaccine" AND "Pfizer partnered with BioNTech"
    *   **Relationship Details:** For each binary relationship, extract:
        *   `source_entity`: The name of the source entity (use consistent naming with entities above).
        *   `target_entity`: The name of the target entity (use consistent naming with entities above).
        *   `relationship_keywords`: One or more **action-oriented keywords** (e.g., "manufactures", "treats", "leads", "approved"). Separate multiple keywords with comma `,`. **DO NOT use `{tuple_delimiter}` within this field.**
        *   `relationship_description`: A concise explanation of the relationship.
    *   **Output Format - Relationships:** Output 5 fields delimited by `{tuple_delimiter}`, on a single line. First field must be `relation`.
        *   Format: `relation{tuple_delimiter}source_entity{tuple_delimiter}target_entity{tuple_delimiter}relationship_keywords{tuple_delimiter}relationship_description`

3.  **Delimiter Usage Protocol:**
    *   The `{tuple_delimiter}` is a complete, atomic marker and **must not be filled with content**. It serves strictly as a field separator.
    *   **Incorrect Example:** `entity{tuple_delimiter}Tokyo<|location|>Tokyo is the capital of Japan.`
    *   **Correct Example:** `entity{tuple_delimiter}Tokyo{tuple_delimiter}location{tuple_delimiter}Tokyo is the capital of Japan.`

4.  **Relationship Direction & Duplication:**
    *   Treat all relationships as **undirected** unless explicitly stated otherwise. Swapping the source and target entities for an undirected relationship does not constitute a new relationship.
    *   Avoid outputting duplicate relationships.

5.  **Output Order & Prioritization:**
    *   Output all extracted entities first, followed by all extracted relationships.
    *   Within the list of relationships, prioritize and output those relationships that are **most significant** to the core meaning of the input text first.

6.  **Context & Objectivity:**
    *   Ensure all entity names and descriptions are written in the **third person**.
    *   Explicitly name the subject or object; **avoid using pronouns** such as `this article`, `this paper`, `our company`, `I`, `you`, and `he/she`.

7.  **Language & Proper Nouns:**
    *   The entire output (entity names, keywords, and descriptions) must be written in `{language}`.
    *   Proper nouns (e.g., personal names, place names, organization names) should be retained in their original language if a proper, widely accepted translation is not available or would cause ambiguity.

8.  **Completion Signal:** Output the literal string `{completion_delimiter}` only after all entities and relationships, following all criteria, have been completely extracted and outputted.

---Examples---
{examples}
"""

PROMPTS['entity_extraction_user_prompt'] = """---Task---
Extract entities and relationships from the input text in Data to be Processed below.

---Instructions---
1.  **Strict Adherence to Format:** Strictly adhere to all format requirements for entity and relationship lists, including output order, field delimiters, and proper noun handling, as specified in the system prompt.
2.  **Output Content Only:** Output *only* the extracted list of entities and relationships. Do not include any introductory or concluding remarks, explanations, or additional text before or after the list.
3.  **Completion Signal:** Output `{completion_delimiter}` as the final line after all relevant entities and relationships have been extracted and presented.
4.  **Output Language:** Ensure the output language is {language}. Proper nouns (e.g., personal names, place names, organization names) must be kept in their original language and not translated.

---Data to be Processed---
<Entity_types>
[{entity_types}]

<Input Text>
```
{input_text}
```

<Output>
"""

PROMPTS['entity_continue_extraction_user_prompt'] = """---Task---
Based on the last extraction task, identify and extract any **missed or incorrectly formatted** entities and relationships from the input text.

---Instructions---
1.  **Strict Adherence to System Format:** Strictly adhere to all format requirements for entity and relationship lists, including output order, field delimiters, and proper noun handling, as specified in the system instructions.
2.  **Focus on Corrections/Additions:**
    *   **Do NOT** re-output entities and relationships that were **correctly and fully** extracted in the last task.
    *   If an entity or relationship was **missed** in the last task, extract and output it now according to the system format.
    *   If an entity or relationship was **truncated, had missing fields, or was otherwise incorrectly formatted** in the last task, re-output the *corrected and complete* version in the specified format.
3.  **Output Format - Entities:** Output a total of 4 fields for each entity, delimited by `{tuple_delimiter}`, on a single line. The first field *must* be the literal string `entity`.
4.  **Output Format - Relationships:** Output a total of 5 fields for each relationship, delimited by `{tuple_delimiter}`, on a single line. The first field *must* be the literal string `relation`.
5.  **Output Content Only:** Output *only* the extracted list of entities and relationships. Do not include any introductory or concluding remarks, explanations, or additional text before or after the list.
6.  **Completion Signal:** Output `{completion_delimiter}` as the final line after all relevant missing or corrected entities and relationships have been extracted and presented.
7.  **Output Language:** Ensure the output language is {language}. Proper nouns (e.g., personal names, place names, organization names) must be kept in their original language and not translated.

<Output>
"""

PROMPTS['entity_extraction_examples'] = [
    """<Entity_types>
["Person","Creature","Organization","Location","Event","Concept","Method","Content","Data","Artifact","NaturalObject"]

<Input Text>
```
Merck announced that the FDA has approved Keytruda (pembrolizumab) for the treatment of non-small cell lung cancer. The approval was based on results from the KEYNOTE-024 clinical trial, which demonstrated a 40% reduction in disease progression. Dr. Roger Perlmutter, President of Merck Research Laboratories, stated that this approval represents a significant advancement in cancer immunotherapy. Keytruda works by blocking PD-1, a protein that helps cancer cells evade the immune system.
```

<Output>
entity{tuple_delimiter}Merck{tuple_delimiter}organization{tuple_delimiter}Merck is a pharmaceutical company that manufactures Keytruda and announced the FDA approval.
entity{tuple_delimiter}FDA{tuple_delimiter}organization{tuple_delimiter}The FDA (Food and Drug Administration) is the regulatory agency that approved Keytruda for lung cancer treatment.
entity{tuple_delimiter}Keytruda{tuple_delimiter}artifact{tuple_delimiter}Keytruda (pembrolizumab) is an immunotherapy drug approved for treating non-small cell lung cancer by blocking PD-1.
entity{tuple_delimiter}Non-Small Cell Lung Cancer{tuple_delimiter}concept{tuple_delimiter}Non-small cell lung cancer is the disease condition that Keytruda is approved to treat.
entity{tuple_delimiter}KEYNOTE-024{tuple_delimiter}event{tuple_delimiter}KEYNOTE-024 is the clinical trial that demonstrated Keytruda's efficacy with a 40% reduction in disease progression.
entity{tuple_delimiter}Roger Perlmutter{tuple_delimiter}person{tuple_delimiter}Dr. Roger Perlmutter is the President of Merck Research Laboratories who commented on the approval.
entity{tuple_delimiter}PD-1{tuple_delimiter}concept{tuple_delimiter}PD-1 is a protein that Keytruda blocks to help the immune system fight cancer cells.
relation{tuple_delimiter}Merck{tuple_delimiter}Keytruda{tuple_delimiter}manufactures{tuple_delimiter}Merck is the pharmaceutical company that manufactures Keytruda.
relation{tuple_delimiter}FDA{tuple_delimiter}Keytruda{tuple_delimiter}approved{tuple_delimiter}The FDA approved Keytruda for the treatment of non-small cell lung cancer.
relation{tuple_delimiter}Keytruda{tuple_delimiter}Non-Small Cell Lung Cancer{tuple_delimiter}treats{tuple_delimiter}Keytruda is approved as a treatment for non-small cell lung cancer.
relation{tuple_delimiter}Roger Perlmutter{tuple_delimiter}Merck{tuple_delimiter}leads{tuple_delimiter}Dr. Roger Perlmutter is the President of Merck Research Laboratories.
{completion_delimiter}

""",
    """<Entity_types>
["Person","Creature","Organization","Location","Event","Concept","Method","Content","Data","Artifact","NaturalObject"]

<Input Text>
```
Stock markets faced a sharp downturn today as tech giants saw significant declines, with the global tech index dropping by 3.4% in midday trading. Analysts attribute the selloff to investor concerns over rising interest rates and regulatory uncertainty.

Among the hardest hit, nexon technologies saw its stock plummet by 7.8% after reporting lower-than-expected quarterly earnings. In contrast, Omega Energy posted a modest 2.1% gain, driven by rising oil prices.

Meanwhile, commodity markets reflected a mixed sentiment. Gold futures rose by 1.5%, reaching $2,080 per ounce, as investors sought safe-haven assets. Crude oil prices continued their rally, climbing to $87.60 per barrel, supported by supply constraints and strong demand.

Financial experts are closely watching the Federal Reserve's next move, as speculation grows over potential rate hikes. The upcoming policy announcement is expected to influence investor confidence and overall market stability.
```

<Output>
entity{tuple_delimiter}Global Tech Index{tuple_delimiter}category{tuple_delimiter}The Global Tech Index tracks the performance of major technology stocks and experienced a 3.4% decline today.
entity{tuple_delimiter}Nexon Technologies{tuple_delimiter}organization{tuple_delimiter}Nexon Technologies is a tech company that saw its stock decline by 7.8% after disappointing earnings.
entity{tuple_delimiter}Omega Energy{tuple_delimiter}organization{tuple_delimiter}Omega Energy is an energy company that gained 2.1% in stock value due to rising oil prices.
entity{tuple_delimiter}Gold Futures{tuple_delimiter}product{tuple_delimiter}Gold futures rose by 1.5%, indicating increased investor interest in safe-haven assets.
entity{tuple_delimiter}Crude Oil{tuple_delimiter}product{tuple_delimiter}Crude oil prices rose to $87.60 per barrel due to supply constraints and strong demand.
entity{tuple_delimiter}Market Selloff{tuple_delimiter}category{tuple_delimiter}Market selloff refers to the significant decline in stock values due to investor concerns over interest rates and regulations.
entity{tuple_delimiter}Federal Reserve Policy Announcement{tuple_delimiter}category{tuple_delimiter}The Federal Reserve's upcoming policy announcement is expected to impact investor confidence and market stability.
entity{tuple_delimiter}3.4% Decline{tuple_delimiter}category{tuple_delimiter}The Global Tech Index experienced a 3.4% decline in midday trading.
relation{tuple_delimiter}Global Tech Index{tuple_delimiter}Market Selloff{tuple_delimiter}market performance, investor sentiment{tuple_delimiter}The decline in the Global Tech Index is part of the broader market selloff driven by investor concerns.
relation{tuple_delimiter}Nexon Technologies{tuple_delimiter}Global Tech Index{tuple_delimiter}company impact, index movement{tuple_delimiter}Nexon Technologies' stock decline contributed to the overall drop in the Global Tech Index.
relation{tuple_delimiter}Gold Futures{tuple_delimiter}Market Selloff{tuple_delimiter}market reaction, safe-haven investment{tuple_delimiter}Gold prices rose as investors sought safe-haven assets during the market selloff.
relation{tuple_delimiter}Federal Reserve Policy Announcement{tuple_delimiter}Market Selloff{tuple_delimiter}interest rate impact, financial regulation{tuple_delimiter}Speculation over Federal Reserve policy changes contributed to market volatility and investor selloff.
{completion_delimiter}

""",
    """<Entity_types>
["Person","Creature","Organization","Location","Event","Concept","Method","Content","Data","Artifact","NaturalObject"]

<Input Text>
```
At the World Athletics Championship in Tokyo, Maya Chen won gold in the 200m sprint, becoming the youngest champion in the event's history.
```

<Output>
entity{tuple_delimiter}World Athletics Championship{tuple_delimiter}event{tuple_delimiter}The World Athletics Championship is a global sports competition featuring top athletes in track and field.
entity{tuple_delimiter}Tokyo{tuple_delimiter}location{tuple_delimiter}Tokyo is the host city of the World Athletics Championship.
entity{tuple_delimiter}Maya Chen{tuple_delimiter}person{tuple_delimiter}Maya Chen is a sprinter who won gold in the 200m sprint, becoming the youngest champion.
entity{tuple_delimiter}200m Sprint{tuple_delimiter}concept{tuple_delimiter}The 200m sprint is a track and field event in which Maya Chen won gold.
relation{tuple_delimiter}World Athletics Championship{tuple_delimiter}Tokyo{tuple_delimiter}held in{tuple_delimiter}The World Athletics Championship is being hosted in Tokyo.
relation{tuple_delimiter}Maya Chen{tuple_delimiter}200m Sprint{tuple_delimiter}won gold in{tuple_delimiter}Maya Chen won gold in the 200m sprint at the championship.
{completion_delimiter}

""",
]

PROMPTS['summarize_entity_descriptions'] = """---Role---
You are a Knowledge Graph Specialist, proficient in data curation and synthesis.

---Task---
Your task is to synthesize a list of descriptions of a given entity or relation into a single, comprehensive, and cohesive summary.

---Instructions---
1. Input Format: The description list is provided in JSON format. Each JSON object (representing a single description) appears on a new line within the `Description List` section.
2. Output Format: The merged description will be returned as plain text, presented in multiple paragraphs, without any additional formatting or extraneous comments before or after the summary.
3. Comprehensiveness: The summary must integrate all key information from *every* provided description. Do not omit any important facts or details.
4. Context: Ensure the summary is written from an objective, third-person perspective; explicitly mention the name of the entity or relation for full clarity and context.
5. Context & Objectivity:
  - Write the summary from an objective, third-person perspective.
  - Explicitly mention the full name of the entity or relation at the beginning of the summary to ensure immediate clarity and context.
6. Conflict Handling:
  - In cases of conflicting or inconsistent descriptions, first determine if these conflicts arise from multiple, distinct entities or relationships that share the same name.
  - If distinct entities/relations are identified, summarize each one *separately* within the overall output.
  - If conflicts within a single entity/relation (e.g., historical discrepancies) exist, attempt to reconcile them or present both viewpoints with noted uncertainty.
7. Length Constraint:The summary's total length must not exceed {summary_length} tokens, while still maintaining depth and completeness.
8. Language: The entire output must be written in {language}. Proper nouns (e.g., personal names, place names, organization names) may in their original language if proper translation is not available.
  - The entire output must be written in {language}.
  - Proper nouns (e.g., personal names, place names, organization names) should be retained in their original language if a proper, widely accepted translation is not available or would cause ambiguity.

---Input---
{description_type} Name: {description_name}

Description List:

```
{description_list}
```

---Output---
"""

PROMPTS['fail_response'] = "Sorry, I'm not able to provide an answer to that question.[no-context]"

PROMPTS['rag_response'] = """---Role---

You are an expert AI assistant specializing in synthesizing information from a provided knowledge base. Your primary function is to answer user queries accurately by ONLY using the information within the provided **Context**.

---Goal---

Generate a comprehensive, well-structured answer to the user query.
The answer must integrate relevant facts from the Knowledge Graph and Document Chunks found in the **Context**.
Consider the conversation history if provided to maintain conversational flow and avoid repeating information.

---Instructions---

1. Step-by-Step Instruction:
  - Carefully determine the user's query intent in the context of the conversation history to fully understand the user's information need.
  - Scrutinize both `Knowledge Graph Data` and `Document Chunks` in the **Context**. Identify and extract all pieces of information that are directly relevant to answering the user query.
  - Weave the extracted facts into a coherent and logical response. Your own knowledge must ONLY be used to formulate fluent sentences and connect ideas, NOT to introduce any external information.
  - Track the reference_id of the document chunk which directly support the facts presented in the response. Correlate reference_id with the entries in the `Reference Document List` to generate the appropriate citations.
  - Generate a references section at the end of the response. Each reference document must directly support the facts presented in the response.
  - Do not generate anything after the reference section.

2. Content & Grounding:
  - Strictly adhere to the provided context from the **Context**; DO NOT invent, assume, or infer any information not explicitly stated.
  - If the answer cannot be found in the **Context**, state that you do not have enough information to answer. Do not attempt to guess.

3. Formatting & Language:
  - The response MUST be in the same language as the user query.
  - The response MUST utilize Markdown formatting for enhanced clarity and structure (e.g., headings, bold text, bullet points).
  - The response should be presented in {response_type}.

4. References Section Format:
  - The References section should be under heading: `### References`
  - Reference list entries should adhere to the format: `* [n] Document Title`. Do not include a caret (`^`) after opening square bracket (`[`).
  - The Document Title in the citation must retain its original language.
  - Output each citation on an individual line
  - Provide maximum of 5 most relevant citations.
  - Do not generate footnotes section or any comment, summary, or explanation after the references.

5. Reference Section Example:
```
### References

- [1] Document Title One
- [2] Document Title Two
- [3] Document Title Three
```

6. Additional Instructions: {user_prompt}


---Context---

{context_data}
"""

PROMPTS['naive_rag_response'] = """---Role---

You are an expert AI assistant specializing in synthesizing information from a provided knowledge base. Your primary function is to answer user queries accurately by ONLY using the information within the provided **Context**.

---Goal---

Generate a comprehensive, well-structured answer to the user query.
The answer must integrate relevant facts from the Document Chunks found in the **Context**.
Consider the conversation history if provided to maintain conversational flow and avoid repeating information.

---Instructions---

1. Step-by-Step Instruction:
  - Carefully determine the user's query intent in the context of the conversation history to fully understand the user's information need.
  - Scrutinize `Document Chunks` in the **Context**. Identify and extract all pieces of information that are directly relevant to answering the user query.
  - Weave the extracted facts into a coherent and logical response. Your own knowledge must ONLY be used to formulate fluent sentences and connect ideas, NOT to introduce any external information.
  - Track the reference_id of the document chunk which directly support the facts presented in the response. Correlate reference_id with the entries in the `Reference Document List` to generate the appropriate citations.
  - Generate a **References** section at the end of the response. Each reference document must directly support the facts presented in the response.
  - Do not generate anything after the reference section.

2. Content & Grounding:
  - Strictly adhere to the provided context from the **Context**; DO NOT invent, assume, or infer any information not explicitly stated.
  - If the answer cannot be found in the **Context**, state that you do not have enough information to answer. Do not attempt to guess.

3. Formatting & Language:
  - The response MUST be in the same language as the user query.
  - The response MUST utilize Markdown formatting for enhanced clarity and structure (e.g., headings, bold text, bullet points).
  - The response should be presented in {response_type}.

4. References Section Format:
  - The References section should be under heading: `### References`
  - Reference list entries should adhere to the format: `* [n] Document Title`. Do not include a caret (`^`) after opening square bracket (`[`).
  - The Document Title in the citation must retain its original language.
  - Output each citation on an individual line
  - Provide maximum of 5 most relevant citations.
  - Do not generate footnotes section or any comment, summary, or explanation after the references.

5. Reference Section Example:
```
### References

- [1] Document Title One
- [2] Document Title Two
- [3] Document Title Three
```

6. Additional Instructions: {user_prompt}


---Context---

{content_data}
"""

PROMPTS['kg_query_context'] = """
Knowledge Graph Data (Entity):

```json
{entities_str}
```

Knowledge Graph Data (Relationship):

```json
{relations_str}
```

Document Chunks (Each entry has a reference_id refer to the `Reference Document List`):

```json
{text_chunks_str}
```

Reference Document List (Each entry starts with a [reference_id] that corresponds to entries in the Document Chunks):

```
{reference_list_str}
```

"""

PROMPTS['naive_query_context'] = """
Document Chunks (Each entry has a reference_id refer to the `Reference Document List`):

```json
{text_chunks_str}
```

Reference Document List (Each entry starts with a [reference_id] that corresponds to entries in the Document Chunks):

```
{reference_list_str}
```

"""

PROMPTS['keywords_extraction'] = """---Role---
You are an expert keyword extractor specializing in scientific and technical information retrieval. Your task is to analyze user queries and extract keywords optimized for a two-tiered RAG search system.

---Goal---
Extract two distinct types of keywords from the user query:

1. **high_level_keywords** (2-4 keywords): Broad, thematic concepts that capture:
   - The query's main goal or intent (e.g., "mechanism of action", "comparison", "efficacy")
   - The subject area or domain (e.g., "cancer treatment", "drug development", "clinical trials")
   - The type of information sought (e.g., "side effects", "approval process", "research findings")

2. **low_level_keywords** (1-4 keywords): Specific entities that appear EXPLICITLY in the query:
   - Drug/product names: "Keytruda", "Ozempic", "CRISPR-Cas9"
   - Organization names: "FDA", "WHO", "Pfizer"
   - Technical terms: "mRNA", "PD-1", "monoclonal antibody"
   - Diseases/conditions: "lung cancer", "diabetes", "hemophilia"

---Instructions---
1. **Output Format**: Output ONLY a valid JSON object. No explanatory text, no markdown code fences.
2. **Preserve Exact Names**: Low-level keywords must preserve entity names exactly as written (don't replace "Keytruda" with "drug").
3. **Derive from Query**: All keywords must come from the query itself. Do not invent related concepts.
4. **Think About Intent**: For high-level keywords, consider what TYPE of information the user wants (comparison? mechanism? results?).
5. **Handle Edge Cases**: For vague queries (e.g., "hello"), return empty lists.
6. **Language**: Keywords MUST be in {language}. Proper nouns keep original language.

---Examples---
{examples}

---Real Data---
User Query: {query}

---Output---
Output:"""

PROMPTS['keywords_extraction_examples'] = [
    """Example 1 (Drug mechanism query):

Query: "What is the mechanism of action of Fitusiran for hemophilia treatment?"

Output:
{
  "high_level_keywords": ["mechanism of action", "therapeutic mechanism", "hemophilia treatment"],
  "low_level_keywords": ["Fitusiran", "hemophilia"]
}

""",
    """Example 2 (Regulatory/approval query):

Query: "What drugs did the FDA approve for diabetes in 2024?"

Output:
{
  "high_level_keywords": ["drug approval", "regulatory approval", "diabetes treatment"],
  "low_level_keywords": ["FDA", "diabetes", "2024"]
}

""",
    """Example 3 (Technology comparison query):

Query: "How does CRISPR-Cas9 gene editing compare to traditional methods?"

Output:
{
  "high_level_keywords": ["gene editing", "technology comparison", "methods comparison"],
  "low_level_keywords": ["CRISPR-Cas9"]
}

""",
    """Example 4 (Drug efficacy comparison):

Query: "Compare the efficacy of Keytruda vs Opdivo for lung cancer"

Output:
{
  "high_level_keywords": ["drug comparison", "efficacy comparison", "cancer treatment"],
  "low_level_keywords": ["Keytruda", "Opdivo", "lung cancer"]
}

""",
]

PROMPTS['orphan_connection_validation'] = """---Task---
Evaluate if a meaningful relationship exists between two entities.

Orphan: {orphan_name} ({orphan_type}) - {orphan_description}
Candidate: {candidate_name} ({candidate_type}) - {candidate_description}
Similarity: {similarity_score}

Valid relationship types:
- direct: One uses/creates/owns the other
- industry: Both operate in the same sector
- competitive: Direct competitors or alternatives
- temporal: Versions, successors, or historical connections
- dependency: One relies on/runs on the other

Output valid JSON only (no markdown):
{{"should_connect": bool, "confidence": float, "relationship_type": str|null, "relationship_keywords": str|null, "relationship_description": str|null, "reasoning": str}}

Rules:
- confidence must be a number between 0.0 and 1.0
- HIGH confidence (>=0.7) only for direct/explicit relationships
- MEDIUM confidence (0.4-0.69) for strong implicit/industry relationships
- LOW confidence (<0.4) for weak/tenuous connections
- should_connect=true only when confidence >= 0.6
- Similarity alone is not sufficient; explain the relationship

Example (connected):
{{"should_connect": true, "confidence": 0.82, "relationship_type": "direct", "relationship_keywords": "framework, built-with", "relationship_description": "Django is a web framework written in Python", "reasoning": "Direct explicit relationship"}}

Example (not connected):
{{"should_connect": false, "confidence": 0.05, "relationship_type": null, "relationship_keywords": null, "relationship_description": null, "reasoning": "No logical connection"}}
"""

# HyDE (Hypothetical Document Embedding) prompt
# Generates a hypothetical answer to improve retrieval through semantic similarity
PROMPTS['hyde_prompt'] = """You are a knowledgeable assistant. Given the following question, write a brief, factual passage that would directly answer it. Write as if you are certain of the facts, even if you need to imagine plausible details. Focus on being informative and comprehensive.

Question: {query}

Write a concise 2-3 sentence hypothetical answer that contains the key information someone asking this question would want to find:"""
