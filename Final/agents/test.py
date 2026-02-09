from agents import (
    generate_cypher,
    run_cypher,
    retrieve_node,
    rank_node,
    answer_node,
    critic_node,
    recommendation_node,
    app
)

query = "Compare the average ratings of Apple and Samsung products."

print("\n========================")
print("UNIT TEST 1 — CYPHER GENERATION")
cypher = generate_cypher(query)
print(cypher)

print("\n========================")
print("UNIT TEST 2 — GRAPH RETRIEVAL")
graph_rows = run_cypher(cypher)
print(graph_rows)

print("\n========================")
print("UNIT TEST 3 — RETRIEVER NODE")
state = {"query": query}
retrieved = retrieve_node(state)
print(retrieved)

print("\n========================")
print("UNIT TEST 4 — RANK NODE")
ranked = rank_node({
    "query": query,
    "docs": retrieved["docs"]
})
print(ranked)

print("\n========================")
print("UNIT TEST 5 — ANSWER NODE")
answered = answer_node({
    "query": query,
    "ranked_docs": ranked["ranked_docs"]
})
print(answered)

print("\n========================")
print("UNIT TEST 6 — CRITIC NODE")
critic = critic_node({
    "answer": answered["answer"],
    "ranked_docs": ranked["ranked_docs"]
})
print(critic)

print("\n========================")
rec = recommendation_node({
    "query": query,
    "answer": answered["answer"]
})
print(rec)

print("\n========================")
print("PIPELINE TEST — LANGGRAPH END-TO-END")

result = app.invoke({
    "query": query,
    "docs": [],
    "ranked_docs": [],
    "answer": "",
    "validated": False,
    "recommendation": ""
})

print("\nFINAL OUTPUT")
print(result)
