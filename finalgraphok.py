import wikipedia
import time
import requests
import re
import spacy
nlp = spacy.load("en_core_web_lg")
from neo4j import GraphDatabase
wikipedia.set_lang("en")

# Fix: set a proper user agent (Wikipedia blocks default requests)
wikipedia.set_rate_limiting(True)
import google.generativeai as genai

genai.configure(api_key="AIzaSyDrxaTXnzUwPE_tLnq1GRmhq1gOCLOciyQ")

model = genai.GenerativeModel("gemini-pro")

class LLMWrapper:
    def invoke(self, prompt):
        response = model.generate_content(prompt)
        return type("obj", (), {"content": response.text})

llm = LLMWrapper()
def fetch_pages(topic, max_pages=15):
    headers = {
        "User-Agent": "KnowledgeGraphBot/1.0 (your@email.com)"
    }

    titles = []

    #  Use Wikipedia REST API directly (more reliable than the library)
    try:
        search_url = "https://en.wikipedia.org/w/api.php"
        params = {
            "action": "query",
            "list": "search",
            "srsearch": topic,
            "srlimit": 30,
            "format": "json"
        }
        resp = requests.get(search_url, params=params, headers=headers, timeout=15)
        resp.raise_for_status()
        data = resp.json()
        titles = [r["title"] for r in data["query"]["search"]]
    except Exception as e:
        print(f"⚠️ Search failed: {e}")
        titles = []

    texts = []
    all_titles = [topic] + titles[:max_pages]

    for t in all_titles:
        #  Fetch page content via REST API directly
        try:
            url = f"https://en.wikipedia.org/w/api.php"
            params = {
                "action": "query",
                "titles": t,
                "prop": "extracts",
                "explaintext": True,
                "exlimit": 1,
                "format": "json"
            }
            resp = requests.get(url, params=params, headers=headers, timeout=15)
            resp.raise_for_status()
            data = resp.json()

            pages = data["query"]["pages"]
            for pid, page in pages.items():
                if pid == "-1":
                    continue
                extract = page.get("extract", "")
                if extract:
                    texts.append(extract[:10000])
                    print(f"   Fetched: {page['title']}")

            time.sleep(0.5)  #  polite delay to avoid rate limiting

        except Exception as e:
            print(f"   Skipping '{t}': {e}")
            continue

    if not texts:
        raise ValueError(f" No Wikipedia content found for topic: '{topic}'")

    print(f"📄 Total pages fetched: {len(texts)}")
    return "\n".join(texts)

# ---------------- ENTITY FILTER ---------------- #
USE_LLM = False
def is_valid_entity(name):
    name_low = name.lower()

    if len(name) <= 2:
        return False

    if name.isdigit():
        return False

    #  FILTER LATEX / MATH NOISE
    if any(x in name for x in ["\\", "^{", "_{", "\\frac", "\\left", "\\right", "\\log", "$", "}"]):
        return False

    if re.search(r'[{}\^\_\$\\]', name):
        return False

    bad_words = ["actor", "cast", "list", "chapter"]
    if any(x in name_low for x in bad_words):
        return False

    if any(x in name_low for x in [
        "year", "million", "billion", "percent",
        "first", "second", "third"
    ]):
        return False

    if name_low in [
        "one", "two", "three", "many", "several",
        "part", "chapter", "volume"
    ]:
        return False

    return True


def classify_entity(name, label):
    name_low = name.lower()

    if name.startswith("House "):
        return "HOUSE"

    if any(x in name_low for x in ["battle", "war", "rebellion"]):
        return "EVENT"

    if label in ["GPE", "LOC"]:
        return "LOCATION"

    if label == "ORG":
        return "ORGANIZATION"

    if label == "PERSON":
        return "PERSON"

    return "OTHER"

# ---------------- ENTITY EXTRACTION ---------------- #

def extract_entities(text):
    doc = nlp(text)
    entities = []
    seen = set()

    for ent in doc.ents:
        name = ent.text.strip()

        if not is_valid_entity(name):
            continue

        etype = classify_entity(name, ent.label_)
        key = (name.lower(), etype)

        if key not in seen:
            seen.add(key)
            entities.append({"name": name, "type": etype})

    print(f"Entities: {len(entities)}")
    return entities

# ---------------- NORMALIZE RELATION ---------------- #

def normalize_relation(rel):
    rel = rel.upper().strip().replace(" ", "_")

    mapping = {
        "IS": "IS_A",
        "ARE": "IS_A",
        "WAS": "IS_A",
        "WERE": "IS_A",
        "HAS": "HAS",
        "HAVE": "HAS",
        "MADE": "CREATE",
        "BUILT": "BUILD",
        "FOUNDED": "CREATE",
        "LED": "LEAD",
        "JOINED": "JOIN",
        "WORKED": "WORK",
        "USED": "USE",
        "IS_A": "TYPE_OF",
        "TYPE": "TYPE_OF",
        "PART": "PART_OF",
        "MEMBER": "MEMBER_OF",
        "FRIEND": "FRIEND_OF",
        "ENEMY": "ENEMY_OF"
    }

    return mapping.get(rel, rel)

# ---------------- EXTRACT RELATIONS ---------------- #

def extract_relations(text):
    doc = nlp(text)
    relations = []

    for sent in doc.sents:
        ents = [e for e in sent.ents if is_valid_entity(e.text)]

        if len(ents) < 2:
            continue

        for token in sent:
            if token.pos_ in ["VERB", "AUX"]:
                rel = normalize_relation(token.lemma_)

                for i in range(len(ents)-1):
                    relations.append({
                        "head": ents[i].text,
                        "relation": rel,
                        "tail": ents[i+1].text
                    })

    return relations

# ---------------- STATISTICAL RELATIONS ---------------- #

def extract_statistical_relations(text):
    relations = []
    t = text.lower()

    patterns = [
        ("Normal Distribution", "HAS_MEAN", "Mean"),
        ("Normal Distribution", "HAS_STD_DEV", "Standard Deviation"),
        ("Normal Distribution", "SHAPE", "Bell Curve"),
        ("Normal Distribution", "SYMMETRIC_AROUND", "Mean"),
        ("Normal Distribution", "FOLLOWS", "Gaussian Distribution"),
        ("Gaussian Distribution", "DEFINED_BY", "Mean"),
        ("Gaussian Distribution", "DEFINED_BY", "Standard Deviation"),
        ("Standard Deviation", "CONTROLS", "Spread"),
        ("Mean", "CENTER_OF", "Distribution"),
        ("Normal Distribution", "USED_IN", "Statistics"),
        ("Normal Distribution", "USED_IN", "Machine Learning"),
        ("Normal Distribution", "ASSUMED_IN", "Central Limit Theorem"),
    ]

    for h, r, ta in patterns:
        if h.lower() in t:
            relations.append({"head": h, "relation": r, "tail": ta})

    return relations
#---------------llm--------------------#
def extract_llm_relations(text):
    import json

    chunks = [text[i:i+3000] for i in range(0, len(text), 3000)]
    chunks = chunks[:3]  # limit

    relations = []

    for chunk in chunks:
        prompt = f"""
Extract relationships as JSON:
[{{"head":"","relation":"","tail":""}}]

TEXT:
{chunk}
"""
        try:
            res = llm.invoke(prompt).content.strip()
            data = json.loads(res)

            for r in data:
                if r.get("head") and r.get("tail"):
                    relations.append({
                        "head": r["head"],
                        "relation": normalize_relation(r.get("relation", "related_to")),
                        "tail": r["tail"]
                    })
        except:
            continue

    return relations

# ---------------- CLEAN RELATIONS ---------------- #

def clean_relations(relations):
    cleaned = []
    for r in relations:
        h = r["head"].strip()
        t = r["tail"].strip()
        rel = r["relation"].strip().upper()
        if not h or not t or not rel:
            continue
        if h.lower() == t.lower():
            continue
        if len(h) < 3 or len(t) < 3:
            continue
        if any(x in h.lower() for x in ["year", "million", "first"]):
            continue
        cleaned.append({"head": h, "relation": rel, "tail": t})
    return cleaned

# ---------------- REMOVE DUPLICATES ---------------- #

def remove_duplicates(relations):
    seen = set()
    unique = []

    for r in relations:
        key = (r["head"].lower(), r["relation"], r["tail"].lower())

        if key not in seen:
            seen.add(key)
            unique.append(r)

    return unique

# ---------------- build graph ---------------- #
def build_graph(topic, uri, user, password):

    from collections import defaultdict

    print(f" Building graph for topic: {topic}")
   
    print(" Fetching Wikipedia data...")
    text = fetch_pages(topic)

    print(" Extracting entities...")
    entities = extract_entities(text)

    print(" Extracting relations...")
    relations = extract_relations(text)
    if USE_LLM:
        print(" Using LLM relations...")
        relations += extract_llm_relations(text)
    print(" Cleaning...")
    relations = clean_relations(relations)
    relations = remove_duplicates(relations)

    entities = entities[:5000]
    #  FILTER: both nodes must exist
    entity_name_set = {e["name"] for e in entities}
    relations = [r for r in relations
                 if r["head"] in entity_name_set
                 and r["tail"] in entity_name_set]

    grouped = defaultdict(list)
    for r in relations:
        grouped[r["relation"]].append(r)
    
    print(" Relation types found:")
    for rel_type, rels in sorted(grouped.items(), key=lambda x: -len(x[1])):
        print(f"   {rel_type}: {len(rels)}")

    balanced = []
    for rel_type, rels in grouped.items():
        balanced.extend(rels[:500])

    for i in range(min(len(entities), 200)):
        for j in range(i+1, min(len(entities), 200)):
            balanced.append({
                "head": entities[i]["name"],
                "relation": "RELATED_TO",
                "tail": entities[j]["name"]
            })

    relations = balanced[:5000]

    print(f" FINAL ENTITIES: {len(entities)}")
    print(f" FINAL RELATIONS: {len(relations)}")
    print(f" UNIQUE RELATION TYPES: {len(grouped)}")
    db = Neo4jConnection(uri, user, password)
    db.load(entities, relations)
    db.close()
    print("🚀 GRAPH READY")
    return True

# ---------------- NEO4J ---------------- #

class Neo4jConnection:

    def __init__(self, uri, user, password):
        self.driver = GraphDatabase.driver(uri, auth=(user, password))
        self.driver.verify_connectivity()

    def close(self):
        self.driver.close()

    def clean_relation(self, rel):
        if not rel:
            return None
        rel = rel.strip().upper()
        rel = rel.replace(" ", "_")
        rel = re.sub(r'[^A-Z0-9_]', '', rel)
        rel = rel.strip("_")
        if not rel or len(rel) < 2:
            return None
        return rel

    def load(self, entities, relations):
        with self.driver.session() as s:

            print(" Clearing old graph...")
            s.run("MATCH (n) DETACH DELETE n")

            print(" Loading entities...")
            for e in entities:
                try:
                    s.run(
                        f"MERGE (n:{e['type']} {{name:$name}})",
                        name=e["name"]
                    )
                except Exception as ex:
                    print(" Entity error:", e, ex)

            print(" Loading relations...")
            skipped = 0
            inserted = 0

            for r in relations:
                rel = self.clean_relation(r.get("relation"))

                if not rel:
                    skipped += 1
                    continue

                try:
                    s.run(f"""
                    MATCH (a {{name:$h}}),(b {{name:$t}})
                    MERGE (a)-[:{rel}]->(b)
                    """, h=r["head"], t=r["tail"])
                    inserted += 1

                except Exception as ex:
                    print(" Relation error:", r, ex)
                    skipped += 1

            print(f" Inserted relations: {inserted}")
            print(f" Skipped: {skipped}")

# ---------------- MAIN ---------------- #
