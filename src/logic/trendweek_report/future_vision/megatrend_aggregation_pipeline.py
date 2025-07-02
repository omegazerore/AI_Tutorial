from textwrap import dedent
from typing import List

from langchain.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field

from src.logic import build_standard_chat_prompt_template


class SubTrend(BaseModel):
    name: str = Field(description="the name of subtrend")
    definition: str = Field(description='A clear and descriptive definition (2-3 sentences)')
    examples: str = Field(description='Relevant examples illustrating the subtrend')


class Result(BaseModel):
    result: List[SubTrend] = Field(description="A list of subtrends and their definition and examples")


output_parser = PydanticOutputParser(pydantic_object=Result)
format_instructions = output_parser.get_format_instructions()

system_template = dedent("""
                    You are a helpful AI assistant acting as a personal assistant to a trend manager.
                    You always strive for high quality and attention to detail.

                    Here is the feedback of a previous report from the client. The final clustering granularity level should follow the standard of `Analyst 2`.

                    Technology:
                        Analyst 1: Personalized Beauty Tech; Augmented Reality (AR) Try-Ons; AI-Powered Beauty Solutions; 
                        Smart Devices & Connected Beauty; Generative AI in Beauty & Wellness; Digital Product Passports; 
                        Tech-Enabled Beauty Experiences. 
                        Analyst 2: AI; Gen AI; Extended Reality; Gamification; Seamless Commerce; Beauty Tech . 
                        Shared subthemes: Both analysts highlight AI-driven beauty innovations (Analyst 1’s “AI-Powered 
                        Beauty Solutions” and “Generative AI in Beauty” vs Analyst 2’s AI/Gen AI) . They
                        also emphasize augmented/extended reality for virtual try-ons (AR try-ons in Analyst 1, 
                        Extended Reality in Analyst 2) . Each addresses beauty tech devices integrating smart
                        technology (Analyst 1’s “Smart Devices & Connected Beauty” and tech-enabled experiences,
                        Analyst 2’s Beauty Tech category). 
                        Unique to Analyst 1: Analyst 1 alone mentions Digital Product Passports for product lifecycle
                        transparency . Analyst 1 also specifically names tech-enabled beauty experiences, blending
                        digital tools with beauty routines (a theme not explicitly listed by Analyst 2). Additionally,
                        “Personalized Beauty Tech” is called out by Analyst 1, whereas Analyst 2 discusses
                        personalization mainly under AI/Gen AI and in the Society context. 
                        Unique to Analyst 2: Analyst 2 introduces Gamification in beauty (e.g. beauty in gaming and
                        interactive play) and Seamless Commerce (chatbots, social commerce, phygital retail), which do
                        not appear as distinct themes in Analyst 1’s tech trends . These reflect a focus on digital
                        engagement and commerce channels unique to Analyst 2’s technology outlook.

                    Industry:
                        Analyst 1: Hybrid Beauty Products; Inclusive Beauty Innovation; Science-Backed Innovation; 
                        Personalization and Customization; Sustainable and Clean Beauty; Tech-Enabled Beauty Experiences; 
                        Holistic Wellness Integration. 
                        Analyst 2: Superior Sensoriality; Skinification; Functionality; Skin Health; Superior Performance; 
                        Tweakments. 
                        Shared subthemes: Both reports note the trend of multi-functional or “hybrid” products that
                        combine benefits (Analyst 1’s Hybrid Beauty Products vs. Analyst 2’s emphasis on Multi-Functionality 
                        under “Functionality” and Skinification, which mixes skincare with cosmetics).
                        They also converge on science-driven performance in beauty – Analyst 1 highlights 
                        Science-Backed Innovation, while Analyst 2 stresses Superior Performance and Skin Health
                        (dermocosmetic efficacy and proven results) . 
                        Unique to Analyst 1: Analyst 1’s industry trends include a focus on inclusive beauty innovation
                        (expanding shades and products for diverse consumers) and personalization/
                        customization of products at an industry level – themes not explicitly listed in Analyst 2’s
                        industry section. Analyst 1 also integrates sustainability in products here (labeled Sustainable
                        and Clean Beauty) and wellness crossover (Holistic Wellness Integration), which are treated in
                        other categories by Analyst 2. 
                        Unique to Analyst 2: Analyst 2 introduces Superior Sensoriality (enhancing textures,
                        fragrances, and packaging feel) as a distinct industry theme, as well as Tweakments (cosmetic
                        procedures or products mimicking clinical treatments) . Analyst 2’s Skinification (applying
                        skincare principles to all categories) and Functionality (including convenient packaging and
                        “beauty charms”) are also unique in framing, as Analyst 1 doesn’t separate these concepts in the
                        same way.

                    Society:
                        Analyst 1: Preventative Skincare & Youthful Appearance; Transparency & Ingredient Literacy; 
                        Generational Shifts in Beauty Preferences; Holistic Wellness & Self-Care; Inclusivity & Diversity.
                        Analyst 2: Glimmers; Holistic Wellbeing; Longevity; Self-Expression; Personalization; Targeted Beauty; 
                        Inclusivity & Accessibility . 
                        Shared subthemes: Both analysts underscore holistic wellness in beauty (Analyst 1’s Holistic
                        Wellness & Self-Care and Analyst 2’s Holistic Wellbeing, focusing on beauty’s link to mental/
                        physical health) . There is a common emphasis on inclusivity: Analyst 1 speaks of 
                        Inclusivity & Diversity, while Analyst 2 highlights Inclusivity & Accessibility, both addressing
                        diverse representation and accessibility in beauty . Themes around youth and aging
                        align as well – Analyst 1 notes Preventative Skincare & Youthful Appearance (pre-aging focus) and
                        Analyst 2 discusses Longevity (prevention and pro-aging approaches) . Additionally, 
                        personalization appears in both, with Analyst 1’s emphasis on hyper-individualized beauty
                        experiences and Analyst 2’s Personalization trend (tailoring beauty to individual preferences)
                        Unique to Analyst 1: Analyst 1 uniquely highlights Transparency & Ingredient Literacy – the
                        demand for clear ingredient and sourcing information – a theme not separately identified by
                        Analyst 2. Analyst 1 also calls out Generational Shifts in Beauty Preferences, examining how
                        different age groups (like Gen Z vs. Millennials) drive trends , which has no direct counterpart
                        in Analyst 2’s societal themes. 
                        Unique to Analyst 2: Analyst 2 includes a trend termed Glimmers, encapsulating joyful, playful
                        self-care and “cult of cute” aesthetics, which does not surface in Analyst 1’s report. Analyst 2 also
                        explicitly mentions Self-Expression (encouraging creativity and breaking beauty norms) and 
                        Targeted Beauty for specific demographics (e.g. male grooming, tween “Zalpha” segment, niche
                        identities) – these specific social themes are not separately identified by Analyst 1.

                    Environment:
                        Analyst 1: Sustainable Packaging; Clean and Green Formulations; Sustainable Innovation; Circular
                        Beauty; Waterless and Low-Resource Formulations; Sustainable Sourcing; Eco-Conscious Consumerism.
                        Analyst 2: War on Waste; War on Carbon; Saving Resources; Alternative Resources; Climate-Proof.
                        Shared subthemes: Both analysts cover waste reduction and circularity in beauty. Analyst 1’s 
                        Sustainable Packaging and Circular Beauty initiatives parallel Analyst 2’s War on Waste
                        (emphasizing circular, refillable, recyclable solutions) . Both also address resource
                        conservation: Analyst 1 highlights Waterless and Low-Resource Formulations (minimizing water
                        usage) , while Analyst 2’s Saving Resources likewise targets water and energy conservation
                        . Additionally, sustainable sourcing and ingredients appear in each – Analyst 1’s Sustainable
                        Sourcing and Clean & Green Formulations correspond to Analyst 2 themes like Alternative
                        Resources (e.g. biotech, “cleanical” beauty) and aspects of Saving Resources (regenerative
                        farming) . Both reports reflect the rise of “clean” beauty and green formulations focused
                        on safety and eco-friendliness (Analyst 1 explicitly names this, and Analyst 2 includes it under 
                        Alternative Resources as clean beauty tied with biotech) . 
                        Unique to Analyst 1: Analyst 1 introduces an Eco-Conscious Consumerism theme, focusing on
                        consumers’ value-driven buying habits and demand for sustainable practices , which is not
                        explicitly called out by Analyst 2. Analyst 1’s broader Sustainable Innovation category (covering
                        eco-friendly materials, processes, and product design) is also a holistic view not broken out in
                        Analyst 2’s report, which instead divides sustainability into specific “wars” (waste, carbon, etc.). 
                        Unique to Analyst 2: Analyst 2 uniquely spotlights a “War on Carbon,” singling out carbon
                        footprint reduction and decarbonization efforts, which Analyst 1 does not isolate as its own
                        theme. Analyst 2 also features Climate-Proof beauty – trends like anti-pollution, sun protection,
                        and products for climate resilience – a distinct theme not explicitly identified in Analyst 1’s
                        list. Additionally, Analyst 2’s focus on Alternative Resources (like biotech innovations) goes a
                        step further into scientific sourcing solutions, beyond the scope of Analyst 1’s named themes.
                    """)

human_template = dedent("""
                    Given the following subtrends, please generate a summarized version by clustering conceptually similar subtrends together:

                    **Subtrends:**

                    {text}

                    Each subtrend follows this structure:
                    **<SUBTREND>:**
                    - **Definition:** <A clear and descriptive definition (2–3 sentences).>
                    - **Examples:** <Relevant examples illustrating the subtrend.>

                    ### Instructions:
                    - Identify and group subtrends with similar definitions.
                    - Create a refined, summarized subtrend for each group, ensuring clarity and coherence.
                    - The summarized definition should be concise yet descriptive (2–3 sentences).
                    - Refine and enhance the examples to better illustrate the summarized subtrend.

                    **Output format:** {format_instructions}
                    """)


def build_prompt_template():
    """
    Constructs a LangChain prompt template using a system and human prompt.

    Returns:
        PromptTemplate: A chain-compatible prompt template.
    """
    input_ = {"system": {"template": system_template},
              "human": {"template": human_template,
                        "input_variables": ['text'],
                        "partial_variables": {"format_instructions": format_instructions}}
              }

    return build_standard_chat_prompt_template(input_)