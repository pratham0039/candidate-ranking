"""
JD parser: turns a job-description text file into a structured spec the
ranker can score against.

Design principle: extract *requirements*, not keywords. The output spec has
four parts:
  1. numeric constraints  - experience band, notice preferences
  2. location policy      - preferred / welcome cities, country, relocation
  3. competency groups    - must-have and nice-to-have CONCEPTS, each backed
                            by a lexicon of surface forms (so "Pinecone",
                            "FAISS" and "vector database" all count as the
                            same requirement, and a candidate can satisfy it
                            in plain language)
  4. disqualifiers        - patterns the JD explicitly rejects

Generic numerics (experience band, locations) are parsed from the text with
regexes. Competency lexicons are seeded from the JD's own named technologies
and expanded with standard synonyms; this keeps the parser usable on other
JDs while staying interpretable.
"""
import re
from dataclasses import dataclass, field


# Concept lexicons: each concept maps to surface forms found in real profiles.
# A candidate matches a concept if ANY form appears in their career evidence.
CONCEPT_LEXICONS = {
    "embeddings_retrieval": [
        "embedding", "embeddings", "dense retrieval", "semantic search",
        "sentence-transformers", "sentence transformers", "bge", "e5",
        "vector representation", "retrieval system", "retrieval pipeline",
        "relevant information", "relevant matches", "content matching",
    ],
    "vector_search_infra": [
        "vector database", "vector db", "pinecone", "weaviate", "qdrant",
        "milvus", "opensearch", "elasticsearch", "faiss", "pgvector",
        "hybrid search", "hybrid retrieval", "vespa", "index refresh",
        "approximate nearest neighbor", "ann index", "hnsw",
    ],
    "ranking_systems": [
        "ranking", "re-ranking", "reranking", "learning-to-rank",
        "learning to rank", "recommendation system", "recommender",
        "recommendation engine", "search relevance", "candidate-jd matching",
        "matching pipeline", "personalization", "feeds ranking",
    ],
    "ranking_evaluation": [
        "ndcg", "mrr", "mean reciprocal rank", "map", "a/b test", "ab test",
        "offline evaluation", "online evaluation", "eval harness",
        "evaluation framework", "evaluation rigor", "offline metrics",
        "online metrics", "interleaving", "relevance regression",
    ],
    "python_engineering": [
        "python", "pytorch", "scikit-learn", "sklearn", "numpy", "pandas",
    ],
    "nlp": [
        "nlp", "natural language processing", "natural language",
        "text classification", "named entity", "transformers",
        "language model", "llm", "rag", "retrieval-augmented",
    ],
    # nice-to-haves
    "llm_finetuning": ["lora", "qlora", "peft", "fine-tun", "finetun", "sft"],
    "ltr_models": ["xgboost", "lightgbm", "gradient boost", "learning-to-rank"],
    "marketplace_domain": [
        "hr-tech", "hr tech", "recruiting", "recruiter", "talent",
        "marketplace", "job seeker", "hiring",
    ],
    "scale_inference": [
        "distributed", "latency", "throughput", "inference optimization",
        "billions of", "millions of users", "at scale", "high-qps", "sharding",
    ],
}

MUST_HAVE = [
    "embeddings_retrieval", "vector_search_infra",
    "ranking_systems", "ranking_evaluation", "python_engineering",
]
NICE_TO_HAVE = [
    "llm_finetuning", "ltr_models", "marketplace_domain", "scale_inference",
]

# Consulting / pure-services firms named by the JD plus common peers.
CONSULTING_FIRMS = {
    "tcs", "tata consultancy", "infosys", "wipro", "accenture", "cognizant",
    "capgemini", "mindtree", "ltimindtree", "hcl", "tech mahindra",
    "mphasis", "lti", "genpact",
}

# Signals that a career stop is research-only (a JD hard disqualifier when
# it covers the whole career).
RESEARCH_MARKERS = ["research lab", "academic", "university", "phd", "postdoc",
                    "research scientist", "research-only", "publications"]

# CV/speech/robotics-only background (JD: rejected without NLP/IR exposure).
CV_SPEECH_MARKERS = ["computer vision", "image classification", "object detection",
                     "speech recognition", "asr", "tts", "robotics", "slam"]
NLP_IR_MARKERS = ["nlp", "natural language", "retrieval", "search", "ranking",
                  "recommendation", "text", "language model", "embedding"]


@dataclass
class JDSpec:
    title: str = ""
    yoe_min: float = 0.0
    yoe_max: float = 50.0
    yoe_ideal_min: float = 0.0
    yoe_ideal_max: float = 50.0
    country: str = "India"
    preferred_cities: list = field(default_factory=list)
    welcome_cities: list = field(default_factory=list)
    visa_sponsorship: bool = False
    notice_pref_days: int = 30
    notice_buyout_days: int = 30
    must_have: list = field(default_factory=lambda: list(MUST_HAVE))
    nice_to_have: list = field(default_factory=lambda: list(NICE_TO_HAVE))
    consulting_firms: set = field(default_factory=lambda: set(CONSULTING_FIRMS))
    jd_text: str = ""


CITY_WORDS = [
    "pune", "noida", "hyderabad", "mumbai", "delhi", "delhi ncr", "bangalore",
    "bengaluru", "chennai", "kolkata", "gurgaon", "gurugram", "ahmedabad",
]


def _find_cities(text):
    found = []
    low = text.lower()
    for city in CITY_WORDS:
        if city in low and city not in found:
            found.append(city)
    return found


def parse_jd(text):
    spec = JDSpec(jd_text=text)

    m = re.search(r"Job Description:\s*(.+)", text)
    if m:
        spec.title = m.group(1).strip()

    # Experience band: "Experience Required: 5–9 years" (any dash variant
    # or the word "to")
    m = re.search(r"(\d+)\s*(?:[-–—]+|to)\s*(\d+)\s*(?:\+\s*)?years", text, re.I)
    if m:
        spec.yoe_min, spec.yoe_max = float(m.group(1)), float(m.group(2))
        spec.yoe_ideal_min, spec.yoe_ideal_max = spec.yoe_min + 1, spec.yoe_max - 1

    # Ideal band if the JD states one ("6-8 years total experience")
    m = re.search(r"(\d+)\s*[-–—]\s*(\d+)\s*years?\s+total\s+experience", text, re.I)
    if m:
        spec.yoe_ideal_min, spec.yoe_ideal_max = float(m.group(1)), float(m.group(2))

    # Location line: preferred cities
    m = re.search(r"Location:\s*(.+)", text)
    if m:
        spec.preferred_cities = _find_cities(m.group(1))

    # Welcome cities ("Candidates in Hyderabad, Pune, Mumbai, Delhi NCR
    # welcome to apply")
    m = re.search(r"Candidates in (.+?) welcome", text, re.I)
    if m:
        spec.welcome_cities = [c for c in _find_cities(m.group(1))
                               if c not in spec.preferred_cities]

    # Sponsorship is offered only if "sponsor" appears without a nearby
    # negation ("don't / do not / cannot / won't / no ... sponsor")
    low = text.lower()
    spec.visa_sponsorship = bool(
        re.search(r"\bsponsor", low)
        and not re.search(r"\b(?:don'?t|do not|cannot|can'?t|won'?t|no)\b[^.]{0,40}sponsor", low)
    )

    # Notice period preferences ("sub-30-day notice", "buy out up to 30 days")
    m = re.search(r"sub-?(\d+)-?day notice", text, re.I)
    if m:
        spec.notice_pref_days = int(m.group(1))
    m = re.search(r"buy out up to (\d+) days", text, re.I)
    if m:
        spec.notice_buyout_days = int(m.group(1))

    return spec


def jd_query_text(spec):
    """The text candidates are matched against: the JD itself plus the
    surface forms of its must-have competencies, so the query emphasizes
    what the role actually requires over boilerplate."""
    parts = [spec.jd_text]
    for concept in spec.must_have:
        parts.append(" ".join(CONCEPT_LEXICONS.get(concept, [])))
    return " ".join(parts).lower()


if __name__ == "__main__":
    import sys
    path = sys.argv[1] if len(sys.argv) > 1 else "data/job_description.md"
    spec = parse_jd(open(path).read())
    print(f"title           : {spec.title}")
    print(f"yoe band        : {spec.yoe_min}-{spec.yoe_max} (ideal {spec.yoe_ideal_min}-{spec.yoe_ideal_max})")
    print(f"preferred cities: {spec.preferred_cities}")
    print(f"welcome cities  : {spec.welcome_cities}")
    print(f"visa sponsorship: {spec.visa_sponsorship}")
    print(f"notice          : prefer <={spec.notice_pref_days}d, buyout {spec.notice_buyout_days}d")
    print(f"must-have       : {spec.must_have}")
    print(f"nice-to-have    : {spec.nice_to_have}")
