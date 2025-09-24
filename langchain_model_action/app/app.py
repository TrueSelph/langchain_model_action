"""This module contains the main app for the Langchain Model action."""

import requests
import streamlit as st
from jvclient.lib.utils import call_update_action
from jvclient.lib.widgets import app_controls, app_header, app_update_action
from streamlit_router import StreamlitRouter


def render(router: StreamlitRouter, agent_id: str, action_id: str, info: dict) -> None:
    """
    Renders the app for the Langchain Model action.

    :param router: The StreamlitRouter instance.
    :param agent_id: The agent ID.
    :param action_id: The action ID.
    :param info: A dictionary containing additional information
    """

    # add app header controls
    (model_key, module_root) = app_header(agent_id, action_id, info)
    # add app main controls
    app_controls(agent_id, action_id)
    # add update button to apply changes
    app_update_action(agent_id, action_id)

    provider = st.session_state[model_key]["provider"]
    endpoint = st.session_state[model_key]["endpoint"]

    if provider == "openrouter" and endpoint:
        with st.expander("Select OpenRouter model", False):
            max_price = st.text_input("Max Price", value="0")

            series = st.selectbox(
                "Series",
                [
                    "All",
                    "GPT",
                    "Claude",
                    "Gemini",
                    "Grok",
                    "Cohere",
                    "Nova",
                    "Qwen",
                    "Yi",
                    "DeepSeek",
                    "Mistral",
                    "Llama2",
                    "Llama3",
                    "Llama4",
                    "RWKV",
                    "Qwen3",
                    "Router",
                    "Media",
                    "PaLM",
                    "Other",
                ],
                index=0,
            )

            context = st.text_input("Context", value="16000")

            input_modalities = st.multiselect(
                "Input Modalities",
                ["text", "code", "image", "audio", "file"],
                default=["text"],
            )

            supported_parameters = st.multiselect(
                "Supported Parameters",
                [
                    "tools",
                    "temperature",
                    "top_p",
                    "top_k",
                    "frequency_penalty",
                    "presence_penalty",
                    "repetition_penalty",
                    "min_p",
                    "top_a",
                    "seed",
                    "max_tokens",
                    "logit_bias",
                    "logprobs",
                    "structured_outputs",
                    "stop",
                    "verbosity",
                ],
                default=["tools"],
            )

            output_modalities = st.multiselect(
                "Output Modalities",
                ["text", "json", "structured", "function_calls"],
                default=["text"],
            )

            # Initialize session state for caching
            if "last_params" not in st.session_state:
                st.session_state.last_params = {}

            current_params = {
                "max_price": max_price,
                "series": series,
                "context": context,
                "input_modalities": input_modalities,
                "supported_parameters": supported_parameters,
                "output_modalities": output_modalities,
            }
            if series == "All":
                del current_params["series"]

            params_changed = current_params != st.session_state.last_params
            if params_changed:

                # get langchain model info
                model_info = get_langchain_model_info(current_params, endpoint)
                # create a select here to select the model
                model_options = [model["name"] for model in model_info]
                if model_options:
                    selected_model = st.selectbox("Select Model", model_options)
                    if selected_model:
                        for model in model_info:
                            if model["name"] == selected_model:
                                # Create a clean, formatted display
                                st.subheader(f"{selected_model} Details")

                                col1, col2 = st.columns(2)

                                with col1:
                                    st.write(
                                        f"**Model ID:** `{model.get('id', 'N/A')}`"
                                    )
                                    st.write(
                                        f"**Name:** **{model.get('name', 'N/A')}**"
                                    )
                                    st.write(
                                        f"**Canonical Slug:** `{model.get('canonical_slug', 'N/A')}`"
                                    )

                                with col2:
                                    # Fix: Convert timestamp without pandas
                                    created_timestamp = model.get("created", 0)
                                    if created_timestamp:
                                        from datetime import datetime

                                        created_date = datetime.fromtimestamp(
                                            created_timestamp
                                        ).strftime("%Y-%m-%d %H:%M:%S")
                                        st.write(f"**Created:** {created_date}")
                                    else:
                                        st.write("**Created:** N/A")
                                    st.write(
                                        f"**Context Length:** **{model.get('context_length', 'N/A'):,}** tokens"
                                    )

                                # Description
                                if model.get("description"):
                                    st.write("### 📝 Description")
                                    st.info(model["description"])

                                # Architecture
                                st.write("### ⚙️ Architecture")
                                arch = model.get("architecture", {})
                                arch_col1, arch_col2 = st.columns(2)

                                with arch_col1:
                                    st.write(
                                        f"**Modality:** `{arch.get('modality', 'N/A')}`"
                                    )
                                    st.write("**Input Modalities:**")
                                    for modality in arch.get("input_modalities", []):
                                        st.write(f"- `{modality}`")

                                with arch_col2:
                                    st.write(
                                        f"**Tokenizer:** `{arch.get('tokenizer', 'N/A')}`"
                                    )
                                    st.write("**Output Modalities:**")
                                    for modality in arch.get("output_modalities", []):
                                        st.write(f"- `{modality}`")

                                # Pricing Information
                                st.write("### 💰 Pricing (per 1M tokens)")
                                pricing = model.get("pricing", {})

                                if any(
                                    float(pricing.get(key, 0)) > 0
                                    for key in ["prompt", "completion", "request"]
                                ):
                                    pricing_cols = st.columns(4)
                                    pricing_data = [
                                        ("Prompt", pricing.get("prompt", "0")),
                                        ("Completion", pricing.get("completion", "0")),
                                        ("Request", pricing.get("request", "0")),
                                        ("Image", pricing.get("image", "0")),
                                    ]

                                    for i, (label, price) in enumerate(pricing_data):
                                        with pricing_cols[i]:
                                            if float(price) == 0:
                                                st.metric(label, "FREE")
                                            else:
                                                st.metric(label, f"${price}")
                                else:
                                    st.success(
                                        "🎉 **This model is completely FREE to use!**"
                                    )

                                # Supported Parameters
                                st.write("### ⚡ Supported Parameters")
                                params = model.get("supported_parameters", [])
                                if params:
                                    # Display parameters in a grid
                                    cols = st.columns(3)
                                    for i, param in enumerate(sorted(params)):
                                        with cols[i % 3]:
                                            st.code(param, language="python")
                                else:
                                    st.write("No specific parameters listed")

                                # Provider Information
                                st.write("### 🏢 Provider Details")
                                provider_info = model.get("top_provider", {})
                                if provider_info:
                                    prov_col1, prov_col2 = st.columns(2)
                                    with prov_col1:
                                        st.write(
                                            f"**Max Completion Tokens:** {provider_info.get('max_completion_tokens', 'N/A')}"
                                        )
                                        st.write(
                                            f"**Moderated:** {'Yes' if provider_info.get('is_moderated') else 'No'}"
                                        )

                                    with prov_col2:
                                        st.write(
                                            f"**Provider Context Length:** {provider_info.get('context_length', 'N/A'):,}"
                                        )

                                break

                        if st.button(
                            "Update", key=f"{model_key}_btn_model_update"
                        ) and model.get("id"):
                            st.session_state[model_key]["model_name"] = model.get(
                                "id", ""
                            )
                            result = call_update_action(
                                agent_id=agent_id,
                                action_id=action_id,
                                action_data=st.session_state[model_key],
                            )
                            if result and result.get("id", "") == action_id:
                                st.success("Model updated")
                                st.rerun()
                            else:
                                st.error("Unable to update model")
                else:
                    st.warning(
                        "No models found matching your criteria. Please adjust your filters."
                    )


def get_langchain_model_info(filters: dict, endpoint: str) -> list:
    """
    Retrieve and filter language models from a LangChain-compatible endpoint.

    Fetches available models from the specified endpoint and applies multiple filters
    to return only models that match all the specified criteria.

    :param filters: Dictionary containing filter criteria for model selection.
                   Supported filters:
                   - max_price: Maximum price per token (float)
                   - series: Model series name to match (string)
                   - context: Minimum context length required (integer)
                   - input_modalities: List of required input modalities
                   - output_modalities: List of required output modalities
                   - supported_parameters: List of required supported parameters
    :param endpoint: Base URL of the LangChain model API endpoint
    :return: List of filtered model dictionaries if successful, None if API call fails
    """

    url = f"{endpoint}/models"

    response = requests.get(url)

    if response.status_code == 200:
        models_data = response.json().get("data", [])
        filtered_models = []

        for model in models_data:
            matches = True

            # Filter by price
            if filters.get("max_price"):
                try:
                    if float(model["pricing"]["prompt"]) > float(filters["max_price"]):
                        matches = False
                except (KeyError, ValueError):
                    matches = False

            # Filter by series
            if (
                filters.get("series")
                and filters["series"].lower() not in model["id"].lower()
            ):
                matches = False

            # Filter by context length
            if filters.get("context"):
                try:
                    if int(model["context_length"]) < int(filters["context"]):
                        matches = False
                except (KeyError, ValueError):
                    matches = False

            # Filter by input modalities
            if filters.get("input_modalities"):
                try:
                    if not all(
                        modality in model["architecture"]["input_modalities"]
                        for modality in filters["input_modalities"]
                    ):
                        matches = False
                except KeyError:
                    matches = False

            # Filter by supported parameters
            if filters.get("supported_parameters"):
                try:
                    if not all(
                        param in model["supported_parameters"]
                        for param in filters["supported_parameters"]
                    ):
                        matches = False
                except KeyError:
                    matches = False

            # Filter by output modalities
            if filters.get("output_modalities"):
                try:
                    if not all(
                        modality in model["architecture"]["output_modalities"]
                        for modality in filters["output_modalities"]
                    ):
                        matches = False
                except KeyError:
                    matches = False

            if matches:
                filtered_models.append(model)

        return filtered_models
    else:
        print("Failed to get model info:", response.status_code)
        return []
