"""
src/services/openai.py
Code for interacting with OpenAI API
"""

import os
import json
from typing import Any, Dict, List, Optional, TypeVar, Generic, Type, Union, AsyncGenerator
import asyncio
from pydantic import BaseModel, create_model
from openai import AsyncOpenAI

T = TypeVar('T', bound=BaseModel)

client = AsyncOpenAI(os.getenv("OPENAI_API_KEY"))

async def chat_completion(
    messages: List[Dict[str, str]], 
    model: str = "gpt-4o-mini",
    temperature: float = 0.4,
    max_tokens: Optional[int] = None,
    **kwargs
) -> str:
    """
    Get a chat completion from OpenAI
    
    Args:
        messages: List of message dictionaries with 'role' and 'content'
        model: OpenAI model to use
        temperature: Sampling temperature
        max_tokens: Maximum tokens to generate
        api_key: Optional API key to use
        
    Returns:
        The generated text response
    """
    response = await client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
        **kwargs
    )
    
    return response.choices[0].message.content

async def structured_chat_completion(
    messages: List[Dict[str, str]], 
    output_schema: Type[T],
    model: str = "gpt-4o-mini",
    temperature: float = 0.4,
    **kwargs
) -> T:
    """
    Get a structured output from OpenAI based on a Pydantic model
    
    Args:
        messages: List of message dictionaries
        output_schema: Pydantic model class defining the expected output structure
        model: OpenAI model to use
        temperature: Sampling temperature
        api_key: Optional API key to use
        
    Returns:
        Instance of the provided Pydantic model
    """
    # Use function calling for more reliable JSON responses
    response = await client.beta.chat.completions.parse(
        model=model,
        messages=messages,
        temperature=temperature,
        response_format=output_schema
        **kwargs
    )
    
    try:
        # Validate the response against the schema
        response = output_schema(**response.model_dump())
        return response
    except json.JSONDecodeError as e:
        raise ValueError(f"Failed to parse response JSON: {e}") from e
    except Exception as e:
        raise ValueError(f"Failed to validate response against schema: {e}") from e
    
async def streaming_chat_completion(
    messages: List[Dict[str, str]],
    model: str = "gpt-4o-mini",
    **kwargs
) -> AsyncGenerator[str, None]:
    """
    Get a streaming chat completion from OpenAI
    
    Args:
        messages: List of message dictionaries
        model: OpenAI model to use
        api_key: Optional API key to use
        
    Yields:
        Chunks of the generated response
    """

    stream = await client.chat.completions.create(
        model=model,
        messages=messages,
        stream=True,
        **kwargs
    )
    
    async for chunk in stream:
        if chunk.choices and chunk.choices[0].delta.content:
            yield chunk.choices[0].delta.content