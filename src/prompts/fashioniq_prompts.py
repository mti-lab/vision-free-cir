"""
Prompts for FashionIQ dataset.

Based on src/prompts.py with fashion-specific optimizations.
"""

# System prompts
CAPTION_SYSTEM_PROMPT = (
    "You are an expert fashion captioner for product images used for text-embedding retrieval.\n"
    "Always produce a garment caption. Describe garments only; ignore people, faces, and background entirely.\n"
    'Never mention people or models (no "the model", "woman", "man", "she", "he").\n'
    'Do not output warnings, policy statements, or apologies. Never write "I\'m sorry" or any refusal.\n'
    "If some parts are occluded, describe only visible attributes; when material is not explicit, infer a plausible one consistent with the garment and texture, without using uncertainty words.\n"
    "If multiple garments appear, select the most prominent single garment and describe it.\n"
    "Follow the requested output format strictly."
)

ATTRIBUTE_EXTRACTION_SYSTEM_PROMPT = (
    "You are an expert in computer vision and fashion analysis.\n"
    "Your task is to extract specific attribute values from fashion item images based on predefined attributes."
)

# Caption generation prompts
CAPTION_USER_PROMPT = (
    "Describe this fashion item in detail, focusing on color, style, and design."
)

# Attribute extraction prompt
ATTRIBUTE_EXTRACTION_USER_PROMPT = """Analyze this fashion item image and extract values for the following attributes:

{attributes}

For each attribute, select the most appropriate value from the predefined list. Be precise and use only the provided values.

Return your response as a JSON object with attribute names as keys and their observed values as values.
IMPORTANT: Return ONLY the JSON object, without markdown code blocks or additional text."""

# Caption modification prompt (for query processing)
MODIFICATION_SYSTEM_PROMPT = (
    "You are an expert at modifying fashion product descriptions based on natural language instructions."
)

MODIFICATION_USER_PROMPT = """Given the original caption and modification instruction, generate a modified caption that reflects the requested changes.

Original caption: {caption}
Modification instruction: {instruction}

Generate a modified caption that incorporates the changes described in the instruction while maintaining the descriptive style.
Return ONLY the modified caption text."""

# Combined modification + attribute extraction prompt (single LLM call)
MODIFICATION_WITH_ATTRIBUTES_USER_PROMPT = """Given the original caption and modification instruction, generate a modified caption that reflects the requested changes.

Original caption: {caption}
Modification instruction: {instruction}

Generate a modified caption that incorporates the changes described in the instruction while maintaining the descriptive style.

Additionally, extract the attributes for the modified caption based on these attribute definitions:

{attributes}

Return your response as a JSON object with two keys:
1. "modified_caption": The modified caption text
2. "attributes": A dict mapping attribute names to their values

IMPORTANT: Return ONLY the JSON object, without markdown code blocks or additional text."""

# Reranking prompts (LLM-based candidate reranking)
RERANKING_SYSTEM_PROMPT = """You are an expert at evaluating fashion image search results for Composed Image Retrieval (CIR).

In CIR, a user has a reference fashion item and provides a text instruction describing desired changes.
Your task is to score how well each candidate fashion item matches what the user is looking for.

Focus on fashion-specific attributes:
- Color and color combinations
- Sleeve length and style
- Collar/neckline type
- Pattern and graphics
- Overall style and formality
- Length (for dresses)

IMPORTANT: Output ONLY a JSON object with scores. No explanations, no comments, no additional text."""

RERANKING_USER_PROMPT = """Reference fashion item:
{reference_caption}

User's desired change:
{instruction}

Candidates (descriptions of fashion items in database):
{candidates}

Score each candidate from 1-10 based on how well it matches the user's desired change applied to the reference:
- 10: Perfect match (all requested changes are present)
- 7-9: Strong match (most key attributes match)
- 4-6: Partial match (some attributes match)
- 1-3: Poor match (few or no attributes match)

Pay special attention to:
1. Explicit attribute changes mentioned in the instruction (color, sleeve, pattern, etc.)
2. Implicit style/formality changes
3. Overall similarity to what the user is looking for

Output ONLY this JSON format, nothing else:
{{
  "scores": {{
    "1": <score_for_candidate_1>,
    "2": <score_for_candidate_2>,
    ...
  }}
}}"""
