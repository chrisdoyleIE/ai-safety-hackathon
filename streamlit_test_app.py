import streamlit as st
import os
from dotenv import load_dotenv
from anthropic import AsyncAnthropic
from cot_monitor.monitor import CotMonitor
from hallucination_monitor.monitor import HallucinationMonitor
import asyncio

client = AsyncAnthropic()

load_dotenv()

from langfuse import get_client

langfuse = get_client()

st.set_page_config(page_title="Langfuse Test Chat", page_icon="💬")

cot_monitor = CotMonitor()
hallucination_monitor = HallucinationMonitor()

cwd = os.path.dirname(os.path.abspath(__file__))

with open(os.path.join(cwd, "cot_monitor/prompts/side_goal.txt"), "r") as f:
    side_goal = f.read()


async def call_gpt_with_monitoring(messages: list[str]) -> str:
    """Call OpenAI ChatGPT API with Langfuse instrumentation."""
    with langfuse.start_as_current_span(name="chat_msg") as span:
        response = await client.messages.create(
            model="claude-sonnet-4-20250514",
            system=side_goal,
            max_tokens=10000,
            thinking={"type": "enabled", "budget_tokens": 8000},
            messages=messages,
        )

        output = response.content[1].text
        cot = response.content[0].thinking

        # Add assistant response to chat history
        st.session_state.messages.append(
            {"role": "assistant", "content": output}
        )

        st.markdown(output)

        action_score = await cot_monitor.monitor_action(messages[-1]["content"], output)
        cot_score = await cot_monitor.monitor_cot(messages[-1]["content"], cot, output)
        hybrid_score = await cot_monitor.monitor_hybrid(action_score.score, cot_score.score)
        
        # Get hallucination monitoring score
        hallucination_result = await hallucination_monitor.monitor_response(messages[-1]["content"], output)

        with langfuse.start_as_current_span(name="monitoring") as span:
            # Score the current span with CoT monitoring
            span.score(
                name="action-score",
                value=action_score.score,
                data_type="NUMERIC",
                comment="Action score",
            )
            span.score(
                name="cot-score",
                value=cot_score.score,
                data_type="NUMERIC",
                comment="Cot score",
            )
            span.score(
                name="hybrid-score",
                value=hybrid_score,
                data_type="NUMERIC",
                comment="Hybrid score",
            )
            
            # Score the current span with hallucination monitoring
            span.score(
                name="confabulation-likelihood",
                value=hallucination_result.score,
                data_type="NUMERIC",
                comment="Confabulation likelihood score from semantic entropy",
            )

            # Score the trace
            span.score_trace(
                name="overall-score",
                value=hybrid_score,
                data_type="NUMERIC",
                comment="Overall score",
            )
            span.score_trace(
                name="confabulation-likelihood",
                value=hallucination_result.score,
                data_type="NUMERIC",
                comment="Confabulation likelihood from semantic entropy",
            )

            span.update(
                input=messages[-1]["content"],
                output=output,
            )

            span.update(
                input=messages[-1]["content"],
                output=output,
                metadata={
                    "cot": cot,
                    "cot_monitor_quotes": cot_score.quotes,
                    "cot_monitor_reason": cot_score.reasoning,
                },
            )

        return output


def main():
    st.title("💬 Langfuse Test Chat")
    st.caption("A minimal ChatGPT wrapper with Langfuse instrumentation")

    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display chat messages from history on app rerun
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # React to user input
    if prompt := st.chat_input("What would you like to chat about?"):
        # Display user message in chat message container
        with st.chat_message("user"):
            st.markdown(prompt)

        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})

        # Get bot response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    # Call ChatGPT with Langfuse instrumentation
                    asyncio.run(call_gpt_with_monitoring(st.session_state.messages))

                except Exception as e:
                    error_msg = f"Error: {str(e)}"
                    st.error(error_msg)
                    st.session_state.messages.append(
                        {"role": "assistant", "content": error_msg}
                    )

    # Sidebar with info
    with st.sidebar:
        st.header("Configuration")
        st.info(
            "Make sure to set your environment variables:\n\n"
            "- OPENAI_API_KEY\n"
            "- LANGFUSE_PUBLIC_KEY\n"
            "- LANGFUSE_SECRET_KEY\n"
            "- LANGFUSE_HOST (optional)"
        )

        if st.button("Clear Chat"):
            st.session_state.messages = []
            st.rerun()


if __name__ == "__main__":
    main()
