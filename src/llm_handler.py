import os
import requests
import json
from typing import List, Dict, Any, Generator, Tuple

def get_cloud_api_keys() -> Tuple[str, str]:
    """
    Retrieves API keys for cloud models, first checking environment variables,
    then checking Streamlit secrets (if running in a Streamlit context).
    """
    hf_token = os.environ.get("HF_TOKEN")
    groq_api_key = os.environ.get("GROQ_API_KEY")
    
    # Try streamlit secrets
    try:
        import streamlit as st
        if hasattr(st, "secrets"):
            if not hf_token:
                hf_token = st.secrets.get("HF_TOKEN")
            if not groq_api_key:
                groq_api_key = st.secrets.get("GROQ_API_KEY")
    except Exception:
        pass
        
    return hf_token, groq_api_key


def check_cloud_provider() -> Tuple[str, str, str]:
    """
    Checks for available cloud providers and returns (provider, api_key, model).
    Prioritizes Groq, then Hugging Face.
    """
    hf_token, groq_api_key = get_cloud_api_keys()
    
    if groq_api_key:
        return "groq", groq_api_key, "llama-3.1-8b-instant"
    elif hf_token:
        return "huggingface", hf_token, "meta-llama/Llama-3.2-1B-Instruct"
        
    return "none", None, None


class HybridLLMHandler:
    """
    Hybrid LLM Client that attempts to connect to local Ollama (llama3.2:1b).
    If Ollama is offline, automatically falls back to Groq API or Hugging Face Inference API.
    """
    def __init__(self, host: str = "http://localhost:11434", model: str = "llama3.2:1b"):
        self.host = host.rstrip('/')
        self.model = model

    def check_connection(self) -> Tuple[bool, str]:
        """
        Checks connection status:
        1. Checks if local Ollama is running and has the model.
        2. If Ollama is offline, checks if Cloud API keys are configured for fallback.
        Returns (is_connected, status_message).
        """
        # First check local Ollama
        try:
            response = requests.get(f"{self.host}/api/tags", timeout=2)
            if response.status_code == 200:
                data = response.json()
                models = [m.get("name") for m in data.get("models", [])]
                
                model_found = False
                for m in models:
                    if self.model in m or m in self.model:
                        model_found = True
                        break
                        
                if model_found:
                    return True, f"Connected to Ollama! Model '{self.model}' is available."
                else:
                    available = ", ".join(models) if models else "None"
                    return True, f"Ollama is running, but model '{self.model}' was not found. Available models: {available}."
        except (requests.exceptions.ConnectionError, requests.exceptions.Timeout):
            pass
        except Exception:
            pass

        # Local Ollama offline, check for Cloud fallback keys
        provider, _, cloud_model = check_cloud_provider()
        if provider != "none":
            provider_name = "Groq" if provider == "groq" else "Hugging Face"
            return True, f"Using Cloud Fallback ({provider_name}) running model '{cloud_model}'."

        return False, "Local Ollama is offline and no Cloud API credentials (GROQ_API_KEY / HF_TOKEN) were found."

    def _generate_openai_compatible_chat(
        self,
        endpoint_url: str,
        api_key: str,
        model: str,
        system_prompt: str,
        user_content: str,
        stream: bool = False,
        json_format: bool = False
    ) -> Any:
        """Helper to invoke OpenAI-compatible endpoints (Groq, Hugging Face)."""
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "model": model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content}
            ],
            "temperature": 0.1 if json_format else 0.4,
            "stream": stream
        }
        
        if json_format:
            payload["response_format"] = {"type": "json_object"}
            
        try:
            response = requests.post(endpoint_url, headers=headers, json=payload, stream=stream, timeout=(5.0, 60.0))
            if response.status_code != 200:
                raise RuntimeError(f"Cloud API returned error {response.status_code}: {response.text}")
                
            if stream:
                def response_generator() -> Generator[str, None, None]:
                    for line in response.iter_lines():
                        if line:
                            line_str = line.decode('utf-8').strip()
                            if line_str.startswith("data: "):
                                data_content = line_str[len("data: "):]
                                if data_content == "[DONE]":
                                    break
                                try:
                                    chunk_data = json.loads(data_content)
                                    delta = chunk_data.get("choices", [{}])[0].get("delta", {})
                                    content = delta.get("content", "")
                                    if content:
                                        yield content
                                except Exception:
                                    pass
                return response_generator()
            else:
                data = response.json()
                return data.get("choices", [{}])[0].get("message", {}).get("content", "")
        except Exception as e:
            raise RuntimeError(f"Cloud API execution error: {str(e)}")

    def generate_chat(self, system_prompt: str, user_content: str, stream: bool = False, json_format: bool = False) -> Any:
        """
        Generates chat completion:
        1. Attempts connection to local Ollama instance.
        2. Falls back to Cloud API (Groq or Hugging Face) if ConnectionError occurs.
        """
        # Attempt local Ollama connection
        url = f"{self.host}/api/chat"
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content}
            ],
            "options": {
                "temperature": 0.1 if json_format else 0.4,
                "top_p": 0.9,
            },
            "stream": stream
        }
        
        if json_format:
            payload["format"] = "json"

        try:
            # Set a low connection timeout to switch quickly if port is not listening,
            # but keep read timeout normal (e.g. 60s) for response generation.
            response = requests.post(url, json=payload, stream=stream, timeout=(2.0, 60.0))
            if response.status_code == 200:
                if stream:
                    def response_generator() -> Generator[str, None, None]:
                        for line in response.iter_lines():
                            if line:
                                data = json.loads(line.decode('utf-8'))
                                content = data.get("message", {}).get("content", "")
                                if content:
                                    yield content
                    return response_generator()
                else:
                    data = response.json()
                    return data.get("message", {}).get("content", "")
            else:
                # If non-200, we'll try cloud fallback rather than failing immediately
                raise requests.exceptions.ConnectionError(f"Ollama returned non-200 code: {response.status_code}")
                
        except (requests.exceptions.ConnectionError, requests.exceptions.Timeout) as e:
            # Fallback to cloud
            provider, api_key, cloud_model = check_cloud_provider()
            
            if provider == "groq":
                endpoint = "https://api.groq.com/openai/v1/chat/completions"
                return self._generate_openai_compatible_chat(
                    endpoint, api_key, cloud_model, system_prompt, user_content, stream, json_format
                )
            elif provider == "huggingface":
                endpoint = "https://api-inference.huggingface.co/v1/chat/completions"
                return self._generate_openai_compatible_chat(
                    endpoint, api_key, cloud_model, system_prompt, user_content, stream, json_format
                )
            else:
                # Re-raise the connection issue since no fallback is available
                raise RuntimeError(
                    f"Local Ollama connection failed, and no Cloud fallback key is configured. "
                    f"Error detail: {str(e)}"
                )
        except Exception as e:
            raise RuntimeError(f"Execution error: {str(e)}")


def get_qa_prompt(context_chunks: List[Dict[str, Any]], query: str) -> Tuple[str, str]:
    """Generates the system and user prompt for Context-Aware Q&A with citations."""
    system_prompt = (
        "You are EduMind AI, a specialized study assistant. Your goal is to answer the user's question "
        "as accurately as possible based ONLY on the provided context chunks.\n\n"
        "Rules:\n"
        "1. Answer the question comprehensively and clearly using the provided document texts.\n"
        "2. CITE the source document name and chunk index at the end of sentences where relevant. "
        "Use the exact format: [Source: filename (Chunk X)].\n"
        "3. Do not make up facts or add information outside the context. If you cannot find the answer "
        "in the context, state: 'I am sorry, but the indexed document does not contain the answer to this question.'\n"
        "4. Be objective, helpful, and concise."
    )
    
    context_str = ""
    for idx, chunk in enumerate(context_chunks):
        source = chunk.get("metadata", {}).get("source", "Unknown")
        chunk_idx = chunk.get("metadata", {}).get("chunk_index", idx)
        context_str += f"--- CONTEXT BLOCK {idx+1} (Source: {source}, Chunk: {chunk_idx}) ---\n{chunk['text']}\n\n"
        
    user_content = f"Context documents:\n{context_str}\nQuestion: {query}"
    return system_prompt, user_content


def get_notes_prompt(context_chunks: List[Dict[str, Any]]) -> Tuple[str, str]:
    """Generates prompts to create detailed structured study summaries/notes."""
    system_prompt = (
        "You are EduMind AI, an expert academic summary writer. Your job is to read the provided text segments "
        "and produce a comprehensive, structured set of study notes.\n\n"
        "Your output must follow this markdown format:\n"
        "1. ## Executive Summary: A high-level overview of the material.\n"
        "2. ## Core Concepts & Terminology: Key terms formatted in bold with definitions.\n"
        "3. ## Detailed Explanations: Deep-dives into the topics, using headings, subheadings, and bullet points.\n"
        "4. ## Study Takeaways: A brief bulleted summary of the most critical points.\n\n"
        "Do not include any greeting or conversational text; begin immediately with the markdown structure."
    )
    
    context_str = ""
    for idx, chunk in enumerate(context_chunks):
        context_str += f"--- TEXT SEGMENT {idx+1} ---\n{chunk['text']}\n\n"
        
    user_content = f"Generate study notes based on this content:\n{context_str}"
    return system_prompt, user_content


def get_quiz_prompt(context_chunks: List[Dict[str, Any]], num_questions: int = 3) -> Tuple[str, str]:
    """Generates prompts to create multiple-choice questions formatted in JSON."""
    system_prompt = (
        "You are EduMind AI, an expert teacher. Your task is to generate a multiple-choice practice quiz "
        "based on the provided text context.\n\n"
        "You MUST return a JSON object with a single root key 'quiz', containing a list of questions.\n"
        "Each question MUST contain these keys exactly:\n"
        "  - 'question': The question text.\n"
        "  - 'options': A list of exactly 4 strings containing choices.\n"
        "  - 'correct_answer': The string matching the exact option that is correct.\n"
        "  - 'explanation': A brief explanation explaining why that answer is correct.\n\n"
        "Example of valid output structure:\n"
        "{\n"
        "  \"quiz\": [\n"
        "    {\n"
        "      \"question\": \"Which library is used for vector search?\",\n"
        "      \"options\": [\"PyPDF2\", \"docx\", \"FAISS\", \"Streamlit\"],\n"
        "      \"correct_answer\": \"FAISS\",\n"
        "      \"explanation\": \"FAISS is specifically designed for efficient similarity searches of dense vectors.\"\n"
        "    }\n"
        "  ]\n"
        "}\n\n"
        "Do not write any introductory or trailing text. Return only the JSON."
    )
    
    context_str = ""
    for idx, chunk in enumerate(context_chunks):
        context_str += f"--- CONTEXT TEXT SEGMENT {idx+1} ---\n{chunk['text']}\n\n"
        
    user_content = f"Generate a quiz with exactly {num_questions} questions from this content:\n{context_str}"
    return system_prompt, user_content

# Alias to preserve backward compatibility for main.py imports
OllamaHandler = HybridLLMHandler
