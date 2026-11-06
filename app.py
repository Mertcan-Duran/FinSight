import streamlit as st

from src.agents.financial_agent import (
    run_financial_agent,
)


st.set_page_config(
    page_title="FinSight",
    page_icon="📊",
    layout="centered",
)


st.title("FinSight")

st.caption(
    "AI-powered financial research "
    "using SEC EDGAR filings."
)


if "messages" not in st.session_state:
    st.session_state.messages = []


for message in st.session_state.messages:
    with st.chat_message(
        message["role"]
    ):
        st.markdown(
            message["content"]
        )


user_question = st.chat_input(
    "Ask a question about a public company..."
)


if user_question:
    st.session_state.messages.append(
        {
            "role": "user",
            "content": user_question,
        }
    )

    with st.chat_message("user"):
        st.markdown(
            user_question
        )

    with st.chat_message("assistant"):
        with st.spinner(
            "Researching SEC filings..."
        ):
            try:
                answer = run_financial_agent(
                    user_question
                )

            except Exception as error:
                answer = (
                    "FinSight could not complete "
                    "the request.\n\n"
                    f"Error: `{error}`"
                )

        st.markdown(
            answer
        )

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": answer,
        }
    )