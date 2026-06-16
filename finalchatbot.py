import streamlit as st
import google.generativeai as genai

from neo4j import GraphDatabase
import time
from finalgraphok import build_graph


def check_neo4j_connection(uri, user, password):
    try:
        driver = GraphDatabase.driver(uri, auth=(user, password))
        driver.verify_connectivity()
        driver.close()
        return True
    except:
        return False
    
# ---------------- UI ---------------- #

st.set_page_config(page_title="GraphRAG QA", layout="wide")
st.title(" GraphRAG Wikipedia Q&A")

st.sidebar.header("Configuration")

# ---------------- STATE ---------------- #

if "last_topic" not in st.session_state:
    st.session_state.last_topic = ""

if "building" not in st.session_state:
    st.session_state.building = False


# ---------------- SIDEBAR INPUTS ---------------- #

disabled = st.session_state.building

topic = st.sidebar.text_input("Topic", value="", disabled=disabled)

neo4j_uri = st.sidebar.text_input(
    "Neo4j URI",
    value="neo4j://127.0.0.1:7687",
    disabled=disabled
)

neo4j_user = st.sidebar.text_input(
    "Neo4j User",
    value="neo4j",
    disabled=disabled
)

neo4j_password = st.sidebar.text_input(
    "Neo4j Password",
    value="12345678",
    type="password",
    disabled=disabled
)
# ---------------- NEO4J STATUS ---------------- #

connected = False  # default

if neo4j_uri and neo4j_user and neo4j_password and not st.session_state.get("building", False):
    connected = check_neo4j_connection(neo4j_uri, neo4j_user, neo4j_password)

    if connected:
        st.sidebar.success("🟢 Neo4j Connected")
    else:
        st.sidebar.error("🔴 Neo4j Not Connected")
        
api_key = st.sidebar.text_input(
    "Gemini API Key",
    type="password",
    disabled=disabled
)




if topic != st.session_state.last_topic and not st.session_state.building:
    st.session_state.building = True


# ---------------- BUILD PROCESS ---------------- #

if st.session_state.building:

    st.warning("⏳ Knowledge Graph is building... Please wait")

    with st.spinner("Building Knowledge Graph..."):
        try:
            build_graph(topic, neo4j_uri, neo4j_user, neo4j_password)
            st.success(" Knowledge Graph Ready")

        except Exception as e:
            st.error(f" Graph build failed: {e}")

    # update state
    st.session_state.last_topic = topic
    st.session_state.building = False

    # clear old chat (important)
    st.session_state.messages = []

    st.rerun()

model_option = st.sidebar.selectbox("Gemini Model", [
    "gemini-2.5-flash",
    "gemini-2.5-flash-lite"
])

use_graph = True


# ---------------- GRAPH ---------------- #

@st.cache_resource
def get_driver(uri, user, password):
    try:
        driver = GraphDatabase.driver(uri, auth=(user, password))
        driver.verify_connectivity()
        return driver
    except:
        return None


def run_cypher(driver, query):
    try:
        with driver.session() as session:
            result = session.run(query)
            return [record.data() for record in result]
    except:
        return []


# ---------------- GEMINI ---------------- #

def call_gemini(model, prompt):
    for _ in range(3):
        try:
            response = model.generate_content(prompt)
            if response.text:
                return response.text.strip()
        except:
            time.sleep(1)
            continue
    return None


#  STEP 1: Topic check
def is_related_to_topic(question, topic, model):
    prompt = f"""
Is the following question related to "{topic}"?

Question: {question}

Answer ONLY: YES or NO
"""
    result = call_gemini(model, prompt)

    if not result:
        return True

    result = result.upper()

    if "YES" in result:
        return True
    elif "NO" in result:
        return False
    return True




def get_answer(question, topic, model):
    prompt = f"""
You are a strict assistant for the topic "{topic}".

Rules:
- If the question is NOT related to {topic}, return EXACTLY: Result not found
- Do NOT answer unrelated questions
- Answer in 2 short lines only if related

Question: {question}

Answer:
"""
    response = call_gemini(model, prompt)

    if not response:
        return None

    response = response.strip()

    #  enforce strict filtering
    if "Result not found" in response:
        return "Result not found"

    # extra safety: detect unrelated answers
    if any(phrase in response for phrase in ["not related", "unrelated", "don't know"]):
        return "Result not found"
    
    # limit to 2 lines
    lines = response.split("\n")
    return "\n".join(lines[:2])


# 🔹 STEP 3: Generate Cypher

def generate_cypher_with_gemini(question, model):
    prompt = f"""
You are an expert in Neo4j.

Convert the user question into a Cypher query.

Question: {question}

Instructions:

1. Identify the main entity (person/place/thing) from the question.
2. Match nodes using:
   toLower(n.name) CONTAINS toLower("<entity>")

3. If question is:
   - About a person → fetch relationships
   - About "who", "what", "relation" → include relationships
   - About "list all" → return nodes

4. ALWAYS try to include relationships:
   OPTIONAL MATCH (n)-[r]->(m)

5. Return useful data:
   RETURN n.name, labels(n), type(r), m.name

6. Limit results:
   LIMIT 20

7. Do NOT use DELETE or CREATE

8. Output ONLY Cypher query (no explanation)

Examples:

Question: Who is Jon Snow?
MATCH (n)
WHERE toLower(n.name) CONTAINS toLower("jon snow")
OPTIONAL MATCH (n)-[r]->(m)
RETURN n.name, labels(n), type(r), m.name
LIMIT 20

Question: Who killed Joffrey?
MATCH (a)-[r]->(b)
WHERE toLower(b.name) CONTAINS toLower("joffrey")
RETURN a.name, type(r), b.name
LIMIT 20
"""
    response = call_gemini(model, prompt)

    if not response:
        return None

    query = response.replace("```cypher", "").replace("```", "").strip()

    # safety
    if "DELETE" in query.upper() or "CREATE" in query.upper():
        return None

    if "LIMIT" not in query.upper():
        query += " LIMIT 20"

    return query

#  STEP 4: Convert DB result → Answer

def generate_db_answer(question, results, model):
    prompt = f"""
You are a expert.

Question: {question}
Graph Data: {results}

Instructions:
- Use the graph data to explain the answer
- If only a name is present, expand using your knowledge
- Give a meaningful explanation (not just name)
- Answer in 2 lines

If no useful data → return NOT_FOUND
"""
    response = call_gemini(model, prompt)

    if not response:
        return None

    return response.strip()


# ---------------- MAIN ---------------- #

def main():
    if "messages" not in st.session_state:
        st.session_state.messages = []

    if not api_key:
        st.warning("Enter Gemini API Key")
        return

    genai.configure(api_key=api_key)
    model = genai.GenerativeModel(model_option)

    driver = get_driver(neo4j_uri, neo4j_user, neo4j_password)

    # chat history
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    user_input = st.chat_input(f"Ask about {topic}...")

    if user_input:

        # show user msg
        with st.chat_message("user"):
            st.markdown(user_input)

        st.session_state.messages.append({"role": "user", "content": user_input})

        with st.chat_message("assistant"):
            thinking = st.empty()
            thinking.markdown("🤖 Thinking...")

            #  STEP 1: Topic check
            
            graph_answer = None

            #  STEP 2: GraphRAG
            if use_graph and driver:
                cypher_query = generate_cypher_with_gemini(user_input, model)

                if cypher_query:
                    results = run_cypher(driver, cypher_query)

                    if results:
                        meaningful = False

                        for row in results:
                            values = [v for v in row.values() if v is not None]

                            #  CONDITION 1: at least 2 meaningful values (relationship case)
                            if len(values) >= 2:
                                meaningful = True

                            #  CONDITION 2: detect relationship-like structure
                            if any(isinstance(v, str) and v.isupper() for v in values):
                                # e.g., "KNOWS", "KILLED", "MEMBER_OF"
                                meaningful = True

                            if meaningful:
                                break

                        if meaningful:
                            graph_answer = generate_db_answer(user_input, results, model)
            #  STEP 3: Decide answer
            if graph_answer and "NOT_FOUND" not in graph_answer:
                final_answer = graph_answer
            else:
                final_answer = get_answer(user_input, topic, model)

            #  STEP 4: Error handling
            if final_answer and final_answer.startswith("ERROR"):
                final_answer = " API limit or error"

            elif not final_answer or "Result not found" in final_answer:
                final_answer = " Result not found"

            thinking.empty()
            st.markdown(final_answer)

        st.session_state.messages.append({
            "role": "assistant",
            "content": final_answer
        })


if __name__ == "__main__":
    main()