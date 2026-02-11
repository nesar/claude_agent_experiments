---
name: literature
description: Search for scientific literature and domain knowledge. Uses mcp-ke's arxiv_agent for paper search and kb-mcp for stored knowledge base documents. Use when the user needs references, methodologies, or background on a cosmology topic.
argument-hint: [search topic]
---

# Literature & Knowledge Search

Topic: $ARGUMENTS

## Search Strategy

### 1. Knowledge Base (kb-mcp) -- check stored documents first
- `kb_search(query, max_results, search_type)` -- hybrid semantic + fulltext search
  - search_type: "hybrid" (default), "semantic", or "fulltext"
- `kb_get(identifier)` -- retrieve full document content
- `kb://sources` -- list all available sources

### 2. arXiv Search (mcp-ke) -- find new papers
- `arxiv_agent(query)` -- multi-step literature search and paper analysis
  - Searches arXiv, downloads papers, summarizes findings

### 3. Web Search -- supplementary references
- Use web search for additional context, documentation, or methodology references

## Output

- Summarize findings with proper citations (author, year, arXiv ID)
- Note key equations, methods, and results relevant to the analysis
- Save findings to `methodology.md` in the experiment directory
- Include all references in BibTeX format for the final report
