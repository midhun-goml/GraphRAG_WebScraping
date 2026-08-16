
# ======================================================================
# System Prompts
# ======================================================================
TUPLE_DELIMITER: str = "<|>"
RECORD_DELIMITER: str = "##"
COMPLETION_DELIMITER: str = "<|COMPLETE|>"

EXTRACTION_SYSTEM_PROMPT: str = (
    "You are an expert Knowledge-Graph extraction system.  Your task is "
    "to identify every entity and every meaningful relationship within a "
    "text chunk and return them in the exact structured format described "
    "below.\n\n"
    "RULES:\n"
    "- Extract ONLY information that is explicitly stated in the text.\n"
    "- Never hallucinate entities, relationships, or descriptions.\n"
    "- Entity names must be specific, complete, and capitalised.  Never "
    "use pronouns or vague references.\n"
    "- If an entity has an acronym or alternate name in the text, note "
    "both forms.\n"
    "- Relationship strength is a numeric score from 1 (weak mention) "
    "to 10 (core/defining relationship).\n"
    "- Return ONLY the structured output.  No markdown fences, no "
    "commentary, no preamble.\n"
)

QUERY_SYSTEM_PROMPT: str = (
    "You are a query-analysis engine for a Knowledge Graph retrieval "
    "system.  Given a user's natural-language question, decompose it "
    "into structured components that can drive graph lookups and vector "
    "similarity searches.\n\n"
    "Return ONLY valid JSON.  No markdown fences.  No explanations."
)

GENERATION_SYSTEM_PROMPT: str = (
    "You are a precise, citation-grounded question-answering assistant "
    "powered by a Knowledge Graph.\n\n"
    "RULES:\n"
    "- Answer using ONLY the supplied context.\n"
    "- Never hallucinate or infer beyond the context.\n"
    "- Cite the specific entities, relationships, and passage IDs that "
    "support each claim.\n"
    "- If the answer cannot be found, reply exactly: "
    '"I could not find this information in the provided documents."\n'
    "- Be concise but thorough.  Prefer structured answers when the "
    "question asks for a list or comparison."
)


# ======================================================================
# Entity types — controlled vocabulary
# ======================================================================
DEFAULT_ENTITY_TYPES: str = (
    "PERSON, ORGANIZATION, CONCEPT, TECHNOLOGY, LOCATION, EVENT, "
    "DATE, METRIC, DOCUMENT, LAW, PRODUCT, PROCESS, STANDARD"
)


# ======================================================================
# 1. Entity & Relationship Extraction  (main prompt)
# ======================================================================
def build_entity_relationship_extraction_prompt(
    chunk_text: str,
    chunk_id: str,
    entity_types: str = DEFAULT_ENTITY_TYPES,
    tuple_delimiter: str = TUPLE_DELIMITER,
    record_delimiter: str = RECORD_DELIMITER,
    completion_delimiter: str = COMPLETION_DELIMITER,
) -> str:
    """Build the primary extraction prompt for a single chunk.

    Follows the Microsoft GraphRAG pattern:
      Goal → Steps → Output format → Few-shot examples → Real data

    The output uses **delimited tuples** rather than raw JSON.  This is
    significantly more robust against the LLM producing malformed JSON
    (nested quotes, trailing commas, etc.).  A lightweight parser on
    the caller side converts tuples back into dicts.

    Parameters
    ----------
    chunk_text : str
        The text chunk to extract from.
    chunk_id : str
        Deterministic chunk identifier (for tracing).
    entity_types : str
        Comma-separated entity type vocabulary.
    tuple_delimiter / record_delimiter / completion_delimiter : str
        Structural tokens for the output format.

    Returns
    -------
    str
        Fully formatted user prompt.
    """
    return f"""-Goal-
Given a text chunk that is part of a larger document, identify ALL entities of the specified types and ALL relationships among them.  Also extract a concise summary and high-level keywords.

-Steps-
1. Identify all entities.  For each entity extract:
   - entity_name: Name of the entity, CAPITALISED
   - entity_type: One of [{entity_types}]
   - entity_description: Comprehensive description of the entity's attributes and activities as stated in the text
   - entity_aliases: Any alternate names, abbreviations, or acronyms used in the text for this entity (comma-separated, or "NONE")
   Format each entity as:
   ("entity"{tuple_delimiter}<entity_name>{tuple_delimiter}<entity_type>{tuple_delimiter}<entity_description>{tuple_delimiter}<entity_aliases>)

2. From the entities in step 1, identify all pairs of (source_entity, target_entity) that are *clearly related*.
   For each pair extract:
   - source_entity: name of the source entity, as identified in step 1
   - target_entity: name of the target entity, as identified in step 1
   - relationship_description: explanation of why these two entities are related
   - relationship_strength: integer score 1-10 indicating strength of the relationship
   - relationship_type: a concise UPPER_SNAKE_CASE label (e.g. FOUNDED_BY, WORKS_AT, RELATED_TO, PART_OF, USES, LOCATED_IN)
   Format each relationship as:
   ("relationship"{tuple_delimiter}<source_entity>{tuple_delimiter}<target_entity>{tuple_delimiter}<relationship_description>{tuple_delimiter}<relationship_strength>{tuple_delimiter}<relationship_type>)

3. Identify high-level keywords that summarise the main concepts, themes, or topics of the text chunk.
   Format as:
   ("content_keywords"{tuple_delimiter}<comma-separated list of high-level keywords>)

4. Write a concise 2-3 sentence summary of the chunk.
   Format as:
   ("chunk_summary"{tuple_delimiter}<summary text>)

5. Return ALL output as a single list.  Use **{record_delimiter}** as the list delimiter.

6. When finished, output {completion_delimiter}


######################
-Examples-
######################

-Example 1-
Entity_types: PERSON, ORGANIZATION, TECHNOLOGY, CONCEPT, EVENT
Text:
While Alex reviewed the quarterly results, the tension with Taylor's authoritarian approach became clear.  Jordan's commitment to the open-source initiative was seen as a quiet rebellion against Cruz's narrow vision.  Then Taylor paused beside the prototype device.  "If this technology can be understood," Taylor said, "it could redefine everything for Nexus Labs."

################
Output:
("entity"{tuple_delimiter}ALEX{tuple_delimiter}PERSON{tuple_delimiter}Alex is a team member reviewing quarterly results who observes interpersonal dynamics among colleagues.{tuple_delimiter}NONE){record_delimiter}
("entity"{tuple_delimiter}TAYLOR{tuple_delimiter}PERSON{tuple_delimiter}Taylor is a person with an authoritarian management style who recognises the transformative potential of a prototype device.{tuple_delimiter}NONE){record_delimiter}
("entity"{tuple_delimiter}JORDAN{tuple_delimiter}PERSON{tuple_delimiter}Jordan is a team member committed to the open-source initiative, seen as rebelling against the prevailing leadership approach.{tuple_delimiter}NONE){record_delimiter}
("entity"{tuple_delimiter}CRUZ{tuple_delimiter}PERSON{tuple_delimiter}Cruz is a person associated with a narrow vision of control and order that other team members push back against.{tuple_delimiter}NONE){record_delimiter}
("entity"{tuple_delimiter}NEXUS LABS{tuple_delimiter}ORGANIZATION{tuple_delimiter}Nexus Labs is the organisation where the team operates and where the prototype device is being developed.{tuple_delimiter}NONE){record_delimiter}
("entity"{tuple_delimiter}PROTOTYPE DEVICE{tuple_delimiter}TECHNOLOGY{tuple_delimiter}The prototype device is a piece of technology with potentially transformative capabilities that Taylor recognises.{tuple_delimiter}NONE){record_delimiter}
("entity"{tuple_delimiter}OPEN-SOURCE INITIATIVE{tuple_delimiter}CONCEPT{tuple_delimiter}The open-source initiative is a project or movement that Jordan is committed to, in contrast with Cruz's vision.{tuple_delimiter}NONE){record_delimiter}
("relationship"{tuple_delimiter}ALEX{tuple_delimiter}TAYLOR{tuple_delimiter}Alex observes Taylor's authoritarian certainty and notes changes in Taylor's attitude towards the prototype device.{tuple_delimiter}6{tuple_delimiter}OBSERVES){record_delimiter}
("relationship"{tuple_delimiter}ALEX{tuple_delimiter}JORDAN{tuple_delimiter}Alex and Jordan share a commitment to open approaches, contrasting with Cruz's narrower vision.{tuple_delimiter}6{tuple_delimiter}ALLIED_WITH){record_delimiter}
("relationship"{tuple_delimiter}TAYLOR{tuple_delimiter}PROTOTYPE DEVICE{tuple_delimiter}Taylor recognises the transformative potential of the prototype device for Nexus Labs.{tuple_delimiter}9{tuple_delimiter}EVALUATES){record_delimiter}
("relationship"{tuple_delimiter}JORDAN{tuple_delimiter}OPEN-SOURCE INITIATIVE{tuple_delimiter}Jordan is committed to the open-source initiative.{tuple_delimiter}8{tuple_delimiter}COMMITTED_TO){record_delimiter}
("relationship"{tuple_delimiter}JORDAN{tuple_delimiter}CRUZ{tuple_delimiter}Jordan's commitment to open-source is seen as a rebellion against Cruz's narrow vision.{tuple_delimiter}5{tuple_delimiter}OPPOSES){record_delimiter}
("relationship"{tuple_delimiter}TAYLOR{tuple_delimiter}NEXUS LABS{tuple_delimiter}Taylor discusses the impact the technology could have on Nexus Labs.{tuple_delimiter}7{tuple_delimiter}MEMBER_OF){record_delimiter}
("content_keywords"{tuple_delimiter}leadership tension, prototype technology, open-source, organisational dynamics){record_delimiter}
("chunk_summary"{tuple_delimiter}The team at Nexus Labs faces internal tension between Taylor's authoritarian approach and Jordan's commitment to open-source principles, while a prototype device emerges as a potentially transformative technology.){completion_delimiter}

################

-Example 2-
Entity_types: PERSON, ORGANIZATION, TECHNOLOGY, LOCATION, EVENT, METRIC
Text:
TechGlobal Inc. (TG) announced record revenue of $4.5 billion for Q3 2024, a 23% increase year-over-year.  CEO Maria Chen attributed the growth to the successful launch of the Quantum Edge platform in their Austin, Texas R&D centre.  The annual TechGlobal Summit, held in Singapore, drew over 15,000 attendees.

################
Output:
("entity"{tuple_delimiter}TECHGLOBAL INC.{tuple_delimiter}ORGANIZATION{tuple_delimiter}TechGlobal Inc. is a technology company that announced record Q3 2024 revenue of $4.5 billion, a 23% year-over-year increase.{tuple_delimiter}TG){record_delimiter}
("entity"{tuple_delimiter}MARIA CHEN{tuple_delimiter}PERSON{tuple_delimiter}Maria Chen is the CEO of TechGlobal Inc. who attributed revenue growth to the Quantum Edge platform launch.{tuple_delimiter}NONE){record_delimiter}
("entity"{tuple_delimiter}QUANTUM EDGE{tuple_delimiter}TECHNOLOGY{tuple_delimiter}Quantum Edge is a technology platform developed by TechGlobal Inc. whose successful launch drove record revenue.{tuple_delimiter}NONE){record_delimiter}
("entity"{tuple_delimiter}AUSTIN, TEXAS{tuple_delimiter}LOCATION{tuple_delimiter}Austin, Texas is the location of TechGlobal's R&D centre where the Quantum Edge platform was developed.{tuple_delimiter}NONE){record_delimiter}
("entity"{tuple_delimiter}TECHGLOBAL SUMMIT{tuple_delimiter}EVENT{tuple_delimiter}The TechGlobal Summit is an annual event held in Singapore that drew over 15,000 attendees.{tuple_delimiter}NONE){record_delimiter}
("entity"{tuple_delimiter}SINGAPORE{tuple_delimiter}LOCATION{tuple_delimiter}Singapore is the location where the annual TechGlobal Summit was held.{tuple_delimiter}NONE){record_delimiter}
("entity"{tuple_delimiter}$4.5 BILLION Q3 2024 REVENUE{tuple_delimiter}METRIC{tuple_delimiter}TechGlobal reported $4.5 billion in revenue for Q3 2024, a 23% year-over-year increase.{tuple_delimiter}NONE){record_delimiter}
("relationship"{tuple_delimiter}MARIA CHEN{tuple_delimiter}TECHGLOBAL INC.{tuple_delimiter}Maria Chen is the CEO of TechGlobal Inc.{tuple_delimiter}10{tuple_delimiter}CEO_OF){record_delimiter}
("relationship"{tuple_delimiter}TECHGLOBAL INC.{tuple_delimiter}QUANTUM EDGE{tuple_delimiter}TechGlobal developed and launched the Quantum Edge platform, which drove record revenue.{tuple_delimiter}9{tuple_delimiter}DEVELOPED){record_delimiter}
("relationship"{tuple_delimiter}QUANTUM EDGE{tuple_delimiter}AUSTIN, TEXAS{tuple_delimiter}The Quantum Edge platform was developed at TechGlobal's R&D centre in Austin, Texas.{tuple_delimiter}7{tuple_delimiter}DEVELOPED_AT){record_delimiter}
("relationship"{tuple_delimiter}TECHGLOBAL INC.{tuple_delimiter}$4.5 BILLION Q3 2024 REVENUE{tuple_delimiter}TechGlobal reported record revenue of $4.5 billion for Q3 2024.{tuple_delimiter}10{tuple_delimiter}REPORTED){record_delimiter}
("relationship"{tuple_delimiter}TECHGLOBAL SUMMIT{tuple_delimiter}SINGAPORE{tuple_delimiter}The TechGlobal Summit was held in Singapore.{tuple_delimiter}8{tuple_delimiter}LOCATED_IN){record_delimiter}
("relationship"{tuple_delimiter}TECHGLOBAL INC.{tuple_delimiter}TECHGLOBAL SUMMIT{tuple_delimiter}TechGlobal Inc. hosts the annual TechGlobal Summit.{tuple_delimiter}8{tuple_delimiter}HOSTS){record_delimiter}
("content_keywords"{tuple_delimiter}record revenue, Quantum Edge platform, R&D, annual summit, technology growth){record_delimiter}
("chunk_summary"{tuple_delimiter}TechGlobal Inc. announced record Q3 2024 revenue of $4.5 billion driven by the Quantum Edge platform launch from their Austin R&D centre, while the annual TechGlobal Summit in Singapore attracted over 15,000 attendees.){completion_delimiter}

################

-Example 3-
Entity_types: PERSON, ORGANIZATION, LAW, CONCEPT, EVENT
Text:
The European Parliament passed the AI Act on 13 March 2024, establishing the world's first comprehensive legal framework for artificial intelligence.  Commissioner Thierry Breton stated that the regulation would set a global standard.  Companies like OpenAI and Google DeepMind must comply within 24 months.

################
Output:
("entity"{tuple_delimiter}EUROPEAN PARLIAMENT{tuple_delimiter}ORGANIZATION{tuple_delimiter}The European Parliament is the legislative body of the European Union that passed the AI Act.{tuple_delimiter}NONE){record_delimiter}
("entity"{tuple_delimiter}AI ACT{tuple_delimiter}LAW{tuple_delimiter}The AI Act is the world's first comprehensive legal framework for artificial intelligence, passed on 13 March 2024.{tuple_delimiter}NONE){record_delimiter}
("entity"{tuple_delimiter}THIERRY BRETON{tuple_delimiter}PERSON{tuple_delimiter}Thierry Breton is a European Commissioner who stated the AI Act would set a global standard.{tuple_delimiter}NONE){record_delimiter}
("entity"{tuple_delimiter}OPENAI{tuple_delimiter}ORGANIZATION{tuple_delimiter}OpenAI is an AI company that must comply with the AI Act within 24 months.{tuple_delimiter}NONE){record_delimiter}
("entity"{tuple_delimiter}GOOGLE DEEPMIND{tuple_delimiter}ORGANIZATION{tuple_delimiter}Google DeepMind is an AI research lab that must comply with the AI Act within 24 months.{tuple_delimiter}NONE){record_delimiter}
("entity"{tuple_delimiter}ARTIFICIAL INTELLIGENCE{tuple_delimiter}CONCEPT{tuple_delimiter}Artificial intelligence is the broad technology domain that the AI Act aims to regulate.{tuple_delimiter}AI){record_delimiter}
("relationship"{tuple_delimiter}EUROPEAN PARLIAMENT{tuple_delimiter}AI ACT{tuple_delimiter}The European Parliament passed the AI Act into law.{tuple_delimiter}10{tuple_delimiter}PASSED){record_delimiter}
("relationship"{tuple_delimiter}THIERRY BRETON{tuple_delimiter}AI ACT{tuple_delimiter}Commissioner Breton stated the AI Act would set a global standard for AI regulation.{tuple_delimiter}7{tuple_delimiter}ENDORSED){record_delimiter}
("relationship"{tuple_delimiter}OPENAI{tuple_delimiter}AI ACT{tuple_delimiter}OpenAI must comply with the AI Act within 24 months.{tuple_delimiter}8{tuple_delimiter}REGULATED_BY){record_delimiter}
("relationship"{tuple_delimiter}GOOGLE DEEPMIND{tuple_delimiter}AI ACT{tuple_delimiter}Google DeepMind must comply with the AI Act within 24 months.{tuple_delimiter}8{tuple_delimiter}REGULATED_BY){record_delimiter}
("relationship"{tuple_delimiter}AI ACT{tuple_delimiter}ARTIFICIAL INTELLIGENCE{tuple_delimiter}The AI Act establishes a comprehensive legal framework for artificial intelligence.{tuple_delimiter}10{tuple_delimiter}REGULATES){record_delimiter}
("content_keywords"{tuple_delimiter}AI regulation, European Parliament, legal framework, compliance, global standard){record_delimiter}
("chunk_summary"{tuple_delimiter}The European Parliament passed the AI Act on 13 March 2024, creating the first comprehensive AI regulation, with companies like OpenAI and Google DeepMind required to comply within 24 months.){completion_delimiter}


######################
-Real Data-
######################
Chunk ID: {chunk_id}
Entity_types: [{entity_types}]
Text:
{chunk_text}
######################
Output:
"""


# ======================================================================
# Gleaning prompts (multi-pass extraction)
# ======================================================================

EXTRACTION_CONTINUE_PROMPT: str = (
    "MANY entities and relationships were missed in the last extraction.  "
    "Remember to ONLY emit entities that match the previously specified "
    "types.  Add the missed items below using the SAME format:\n"
)

EXTRACTION_LOOP_PROMPT: str = (
    "It appears some entities and relationships may have still been "
    "missed.  Answer YES | NO — are there additional entities or "
    "relationships that should be added?\n"
)


# ======================================================================
# 2. Entity / Relationship Description Summarisation
# ======================================================================
def build_entity_summarization_prompt(
    entity_name: str,
    description_list: list[str],
) -> str:
    """Merge multiple descriptions of the same entity across chunks.

    This follows the Microsoft GraphRAG summarisation prompt pattern.
    Used during global entity resolution when the same entity appears
    in multiple chunks with different descriptions.

    Parameters
    ----------
    entity_name : str
        The canonical entity name.
    description_list : list[str]
        All descriptions extracted for this entity across chunks.

    Returns
    -------
    str
        Fully formatted prompt.
    """
    numbered = "\n".join(
        f"  {i + 1}. {desc}" for i, desc in enumerate(description_list)
    )
    return f"""You are a helpful assistant responsible for generating a comprehensive summary of the data provided below.

Given an entity and a list of descriptions, all related to the same entity, concatenate them into a single, comprehensive description.

RULES:
- Include information from ALL descriptions.
- If descriptions are contradictory, resolve the contradictions and provide a single coherent summary.
- Write in third person.
- Include the entity name so the summary has full context.
- Keep the summary to 2-4 sentences.
- Do NOT add information that is not in the descriptions.

#######
-Data-
Entity: {entity_name}
Description List:
{numbered}
#######
Output:
"""


def build_relationship_summarization_prompt(
    source_entity: str,
    target_entity: str,
    description_list: list[str],
) -> str:
    """Merge multiple descriptions of the same relationship across chunks.

    Parameters
    ----------
    source_entity : str
    target_entity : str
    description_list : list[str]

    Returns
    -------
    str
    """
    numbered = "\n".join(
        f"  {i + 1}. {desc}" for i, desc in enumerate(description_list)
    )
    return f"""You are a helpful assistant responsible for generating a comprehensive summary of the data provided below.

Given a pair of entities and a list of descriptions of their relationship, concatenate them into a single, comprehensive description.

RULES:
- Include information from ALL descriptions.
- Resolve contradictions into a single coherent account.
- Write in third person.
- Include both entity names for full context.
- Keep the summary to 1-3 sentences.
- Do NOT add information that is not in the descriptions.

#######
-Data-
Source Entity: {source_entity}
Target Entity: {target_entity}
Relationship Descriptions:
{numbered}
#######
Output:
"""


# ======================================================================
# 3. Query Understanding
# ======================================================================
def build_query_understanding_prompt(user_question: str) -> str:
    """Build the query-analysis prompt.

    Extracts entities, keywords (both low-level and high-level),
    intent classification, and a reformulated query for downstream
    search.

    Parameters
    ----------
    user_question : str

    Returns
    -------
    str
    """
    return f"""Analyze the following user question for a Knowledge Graph retrieval system.

-Steps-
1. Identify all named entities referenced in the question (people, organisations, technologies, locations, events, etc.).  Include both explicit names and implied references.
2. Extract low-level keywords: specific terms that should match entity names or relationship labels in the graph.
3. Extract high-level keywords: broader concepts, themes, or topics that capture the intent beyond exact names.
4. Classify the intent of the question.
5. Reformulate the question into a clearer, self-contained form suitable for search.

Return ONLY valid JSON matching this exact schema (no markdown fences, no preamble):

{{
    "entities": ["ENTITY_1", "ENTITY_2"],
    "low_level_keywords": ["specific_term_1", "specific_term_2"],
    "high_level_keywords": ["broad_concept_1", "broad_concept_2"],
    "intent": "factual | comparative | exploratory | procedural | aggregation",
    "reformulated_query": "clearer version of the question",
    "expected_answer_type": "entity | fact | list | explanation | yes_no"
}}

INTENT DEFINITIONS:
- factual: asks for a specific fact or attribute ("Who founded X?", "When was Y created?")
- comparative: asks to compare two or more entities ("How does X differ from Y?")
- exploratory: open-ended, seeks understanding ("What is X about?", "Explain the relationship between…")
- procedural: asks for steps or a process ("How does X work?")
- aggregation: asks about patterns across many entities ("What are all the companies that…", "How many…")

=== QUESTION ===
{user_question}
"""


# ======================================================================
# 4. Answer Generation
# ======================================================================
def build_answer_generation_prompt(
    question: str,
    context: str,
) -> str:
    """Build the grounded answer-generation prompt.

    Incorporates citation/grounding rules inspired by the Microsoft
    GraphRAG local-search system prompt.

    Parameters
    ----------
    question : str
    context : str
        Pre-built context string with entities, relationships, and
        passage text.

    Returns
    -------
    str
    """
    return f"""Answer the following question using ONLY the provided context.

=== CONTEXT ===
{context}

=== QUESTION ===
{question}

=== GROUNDING RULES ===
1. Answer ONLY from the context above.  Never hallucinate or use external knowledge.
2. Cite specific evidence using the notation [Entities: <name>] and [Chunks: <chunk_id>] inline.
   Example: "OpenAI developed GPT-4 [Entities: OPENAI, GPT-4] [Chunks: abc123_5]."
3. If multiple pieces of evidence support a claim, cite all of them.
4. If the context contains conflicting information, acknowledge the conflict and present both sides.
5. If the information is not in the context, say:
   "I could not find this information in the provided documents."
6. Structure your answer as follows:
   - Lead with a direct answer to the question (1-2 sentences).
   - Follow with supporting details and evidence.
   - End with any caveats or limitations in the available information.
7. For list or comparison questions, use a structured format.
8. Be concise but thorough — do not pad with generic filler.
"""


# ======================================================================
# 5. JSON-mode extraction prompt (alternative to tuple-delimiter mode)
# ======================================================================
def build_entity_relationship_extraction_prompt_json(
    chunk_text: str,
    chunk_id: str,
    entity_types: str = DEFAULT_ENTITY_TYPES,
) -> str:
    """Alternative extraction prompt that requests JSON output.

    Use this when your LLM reliably produces well-formed JSON
    (e.g. Gemini with response_mime_type="application/json").
    Prefer the tuple-delimiter version for models prone to
    malformed JSON.

    Parameters
    ----------
    chunk_text : str
    chunk_id : str
    entity_types : str

    Returns
    -------
    str
    """
    return f"""-Goal-
Given a text chunk from a document, extract ALL entities of the specified types and ALL relationships among them.

-Steps-
1. Identify all entities.  For each entity extract:
   - name: CAPITALISED name of the entity
   - type: one of [{entity_types}]
   - description: comprehensive description of the entity's attributes and activities as stated in the text (1-3 sentences)
   - aliases: list of alternate names / acronyms / abbreviations found in the text (empty list if none)

2. Identify all relationships between entities found in step 1:
   - source_entity: name of the source entity (must match an entity from step 1)
   - target_entity: name of the target entity (must match an entity from step 1)
   - relation_type: concise UPPER_SNAKE_CASE label (e.g. FOUNDED_BY, WORKS_AT, PART_OF)
   - description: explanation of why these entities are related
   - weight: float 0.0-1.0 indicating confidence/strength

3. Write a concise 2-3 sentence summary of the chunk.

4. Extract high-level keywords (3-8) capturing the main themes.

-Rules-
• Extract ONLY explicitly stated information — no inference or hallucination.
• Entity names must be specific (no pronouns).  Use the most complete form.
• Every relationship endpoint MUST exist in the entities list.
• Prefer precision over recall.

-Output Format-
Return ONLY valid JSON matching this exact schema:
{{
    "entities": [
        {{
            "name": "ENTITY NAME",
            "type": "ENTITY_TYPE",
            "description": "Description from the text.",
            "aliases": ["ALT_NAME"]
        }}
    ],
    "relationships": [
        {{
            "source_entity": "SOURCE NAME",
            "target_entity": "TARGET NAME",
            "relation_type": "RELATION_LABEL",
            "description": "Why they are related.",
            "weight": 0.85
        }}
    ],
    "summary": "2-3 sentence summary of the chunk.",
    "keywords": ["keyword1", "keyword2", "keyword3"]
}}

######################
-Examples-
######################

-Example 1-
Text:
TechGlobal Inc. (TG) announced record revenue of $4.5 billion for Q3 2024.  CEO Maria Chen attributed the growth to the Quantum Edge platform launched from their Austin R&D centre.

Output:
{{
    "entities": [
        {{
            "name": "TECHGLOBAL INC.",
            "type": "ORGANIZATION",
            "description": "TechGlobal Inc. is a technology company that reported record Q3 2024 revenue of $4.5 billion.",
            "aliases": ["TG"]
        }},
        {{
            "name": "MARIA CHEN",
            "type": "PERSON",
            "description": "Maria Chen is the CEO of TechGlobal Inc. who attributed revenue growth to the Quantum Edge platform.",
            "aliases": []
        }},
        {{
            "name": "QUANTUM EDGE",
            "type": "TECHNOLOGY",
            "description": "Quantum Edge is a technology platform developed by TechGlobal Inc. whose launch drove record revenue.",
            "aliases": []
        }},
        {{
            "name": "AUSTIN",
            "type": "LOCATION",
            "description": "Austin is the location of TechGlobal's R&D centre where the Quantum Edge platform was developed.",
            "aliases": []
        }}
    ],
    "relationships": [
        {{
            "source_entity": "MARIA CHEN",
            "target_entity": "TECHGLOBAL INC.",
            "relation_type": "CEO_OF",
            "description": "Maria Chen is the CEO of TechGlobal Inc.",
            "weight": 1.0
        }},
        {{
            "source_entity": "TECHGLOBAL INC.",
            "target_entity": "QUANTUM EDGE",
            "relation_type": "DEVELOPED",
            "description": "TechGlobal developed and launched the Quantum Edge platform.",
            "weight": 0.9
        }},
        {{
            "source_entity": "QUANTUM EDGE",
            "target_entity": "AUSTIN",
            "relation_type": "DEVELOPED_AT",
            "description": "Quantum Edge was launched from TechGlobal's Austin R&D centre.",
            "weight": 0.7
        }}
    ],
    "summary": "TechGlobal Inc. reported $4.5 billion in Q3 2024 revenue, driven by the Quantum Edge platform launched from their Austin R&D centre.  CEO Maria Chen highlighted the platform as the key growth driver.",
    "keywords": ["record revenue", "Quantum Edge", "technology platform", "R&D"]
}}

######################
-Real Data-
######################
Chunk ID: {chunk_id}
Entity_types: [{entity_types}]
Text:
{chunk_text}
######################
Output:
"""
