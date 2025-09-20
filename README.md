# Monitoring bad AI behaviour in production

In this project, we demonstrate an idea to deploy AI safety research into existing LLM monitoring and observability layers. We achieved this by implementing protocols for detecting two types of bad behaviour - hallucinations and chain-of-thought pathologies - and integrating these with the popular LLM observability layer [Langfuse](https://langfuse.com/).

## What we did

Our initial idea was to create a fork of Langfuse that specifically monitored for misaligned behaviours using protocols from recent safety research. We settled on implementing protocols from two papers:

- [Detecting hallucinations in large language models with semantic entropy](https://www.nature.com/articles/s41586-024-07421-0) - This Nature paper demonstrates a technique for detecting hallucinations by measuring the semantic entropy over multiple resamples of the LLM response to a given prompt. Semantic entropy is calculted by clustering responds based on their meaning, and a small number of clusters corresponds to lower entropy.
- [CoT Red-Handed: Stress Testing Chain-of-thought Monitoring](https://arxiv.org/abs/2505.23575) - This paper proposes a protocol for monitoring the chain-of-thought of reasoning models for misaligned reasoning. Specifically, they concentrate on samples where the model response does not reveal misalignment, but the chain-of-thought does. Their protocol allows a weaker model to successfully detect when a stronger model is pursuing a side-goal in addition to the goal in the user prompt, based on a hybrid analysis of the chain-of-thought and the output.

We asked Claude to build a small demo app that we integrated with our implementations and Langfuse to demonstrate how Langfuse could be modified to incorporate AI safety research to detect AI misbehaviour in production.

## Why we did it

We think that deployng AI safety research in production has a number of benefits:
- Building datasets of misbehaviour in real-world settings can provide researchers with much more representative data for safety evaluations, and can help to refine our model of how LLM misalignment might emerge.
- Safety evals suffer from under-elicitation - it is hard to know how much more dangerous current models might be without extensive iteration on their scaffolding. Maximal elicitation is highly incentivised in production deployments, so monitoring deployed models could help understand how elicitation relates to dangerous capabilities.

We chose these papers because:
- We think that hallucinations are a high-value misbehaviour to target - LLM hallucinations currently preclude their deployment in a variety of high-stakes settings, and the hackathon was centered around developing a venture-scalable idea. By detecting hallucinations in production, we can safeguard high-stakes deployments and unlock a range of new use cases for AI agents.
- CoT monitoring is a [popular safety agenda](https://arxiv.org/abs/2507.11473) with obvious near-term applications. However, concerns about scheming and CoT faithfulness make it more difficult to make strong claims about CoT monitoring in evaluation settings - so we think that production data could be highly leveraged for this research direction.

## About

This project was developed by @chrisdoyleIE and @DanielPolatajko as their submission to the (one-day) [AI Security Hackathon in London in September 2025](https://luma.com/AI-security-hackathon?tk=HiGIn4).

Some of the initial work for the project was done on a fork [here](https://github.com/DanielPolatajko/langfuse-python/pull/1), before we decided that a fork wasn't necessary to demonstrate the idea.
