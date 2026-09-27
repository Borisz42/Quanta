Instruction skill teaching Antigravity how to coordinate between the primary conversational model and the `flash_lite` worker:
1. **When Ingesting Large Text**:
   - The primary agent invokes a subagent with `Model: "flash_lite"` and the role `"QUANTA S-Expression Transducer"`.
   - The subagent extracts entities, relations, and temporal intervals into structured JSON.
   - The subagent calls the local MCP tool `quanta_ingest_propositions(doc_id, extracted_data)` to compile and persist the graph.
2. **When Querying Long-Term Memory**:
   - The primary agent calls `quanta_query_memory(query)` directly via MCP.
   - The local engine returns the verified, minimal subgraph in $< 5\text{ ms}$.
   - The primary agent incorporates the factual sub-graph into its response to the user.
3. **When Correcting a False Belief**:
   - Calls `quanta_revise_belief(entity, property, new_value, reason)` to update temporal intervals $[t_{\text{start}}, t_{\text{end}})$ without destroying history.