import:py logging;
import:py traceback;
import:py from typing { Union }
import:py from logging { Logger }
import:py from langchain_core.prompts { ChatPromptTemplate }
import:py from langchain_openai { ChatOpenAI, AzureChatOpenAI }
import:py from langchain_community.callbacks { get_openai_callback }
import:jac from jivas.agent.action.model_action { ModelAction, ModelActionResult }

node LangChainModelAction :ModelAction: {
    # JIVAS action wrapper around LangChain library for abstracted LLM interfacing
    # you can register multiple instances of this action, each with different api keys and model configurations
    # for use in other actions

    # set up logger
    static has logger:Logger = logging.getLogger(__name__);

    has api_key:str = "";
    has api_base: str = "";
    has api_version: str = "";
    has model_name:str = "gpt-4o";
    has model_temperature:float = 0.4;
    has model_max_tokens:int = 2048;

    can invoke(
        prompt_messages:list,
        prompt_variables:dict,
        kwargs:dict = {}
    ) -> dict {

        llm = None;

        # we'll need to convert the prompt_messages dict to an array of tuples
        prompt_tuples = [(role, message) for item in prompt_messages for (role, message) in item.items()];
        template = ChatPromptTemplate.from_messages(prompt_tuples);

        op = kwargs.get('op', "chatopenai");
        functions = kwargs.get('functions', []);
        temperature = kwargs.get('model_temperature', self.model_temperature);
        model_name = kwargs.get('model_name', self.model_name);
        max_tokens = kwargs.get('model_max_tokens', self.model_max_tokens);

        if op == 'chatopenai' {
            llm = ChatOpenAI(
                temperature = temperature,
                model_name = model_name,
                openai_api_key = self.api_key,
                max_tokens = max_tokens
            );

        } elif op == 'azurchatopenai' {
            llm = AzureChatOpenAI(
                temperature = temperature,
                openai_api_version=self.api_version,
                openai_api_key=self.api_key,
                openai_api_base=self.api_base,
                deployment_name=model_name,
                max_tokens = max_tokens
            );
        }

        if(llm) {
            try {

                messages = template.format_messages(**prompt_variables);

                if(functions) {
                    llm = llm.bind_tools(functions);
                }

                with get_openai_callback() as cb {

                    if(message:=llm.invoke(messages)) {

                        return ModelActionResult(
                            prompt = messages,
                            functions = functions,
                            tokens = cb.total_tokens,
                            result = message.tool_calls if functions else message.content,
                            temperature = temperature,
                            model_name = model_name,
                            max_tokens = max_tokens
                        );

                    }

                }

            } except Exception as e {
                self.logger.error(f"an exception occurred, {traceback.format_exc()}");
            }
        }

        return None;
    }

    can healthcheck() -> Union[bool, dict] {

        if not self.api_key {
            return {
                "status": False,
                "message": "API key not set.",
                "severity": "error"
            };
        }

        test_prompt_messages = [{"system" : "Output the result of 2 + 2"}];
        test_kwargs = {
            "model_name": self.model_name,
            "model_temperature": self.model_temperature,
            "model_max_tokens": self.model_max_tokens
        };

        try {
            if( model_action_result := self.call_model(prompt_messages = test_prompt_messages, prompt_variables = {}, kwargs = test_kwargs)) {               # set the interaction message+
                interaction_message = model_action_result.get_result();
                if not interaction_message {
                    return {
                        "status": False,
                        "message": "No valid result from LLM call. Check API key and model configuration.",
                        "severity": "error"
                    };
                } else {
                    return True;
                }
            }
            return {
                "status": False,
                "message": "Unable to excute LLM call. Check API key and model configuration.",
                "severity": "error"
            };
        } except Exception as e {
            self.logger.error(f"An exception occurred in {self.label}:\n{traceback.format_exc()}\n");
            return {
                "status": False,
                "message": f"There is an issue with the action. {e}",
                "severity": "error"
            };
        }
    }

}