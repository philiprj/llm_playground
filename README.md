# llm_playground

Repository for exploring LLMs and related frameworks

## Setup

Ensure Python 3.11 is installed with pyenv and poetry is installed with pipx. Then run:

```bash
pyenv local 3.11
make setup
```

This will install the project dependencies and pre-commit hooks.

## LangChain

LangChain is a framework for developing applications powered by large language models (LLMs).

LangChain aims to simplify every stage of the LLM application lifecycle:

- Development: Build applications using LangChain's open-source building blocks, components, and third-party integrations. Use LangGraph to build stateful agents with first-class streaming and human-in-the-loop support.
- Productionisation: Use LangSmith to inspect, monitor and evaluate your chains, so that you can continuously optimize and deploy with confidence.
- Deployment: Turn your LangGraph applications into production-ready APIs and Assistants with LangGraph Cloud.

Concretely, the framework consists of the following open-source libraries:

- langchain-core: Base abstractions and LangChain Expression Language.
- Integration packages (e.g. langchain-openai, langchain-anthropic, etc.): Important integrations have been split into lightweight packages that are co-maintained by the LangChain team and the integration developers.
- langchain: Chains, agents, and retrieval strategies that make up an application's cognitive architecture.
- langchain-community: Third-party integrations that are community maintained.
- LangGraph: Build robust and stateful multi-actor applications with LLMs by modeling steps as edges and nodes in a graph. Integrates smoothly with LangChain, but can be used without it.
- LangGraphPlatform: Deploy LLM applications built with LangGraph to production.
- LangSmith: A developer platform that lets you debug, test, evaluate, and monitor LLM applications.
