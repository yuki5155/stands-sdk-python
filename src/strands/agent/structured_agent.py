"""Structured Agent with native Pydantic model support.

This module provides the StructuredAgent class which extends the base Agent
with built-in support for structured outputs using Pydantic models.
"""

import json
import logging
from typing import Any, Dict, Optional, Type, TypeVar, Union

from pydantic import BaseModel, ValidationError

from .agent import Agent

logger = logging.getLogger(__name__)

# Type variable for Pydantic models
T = TypeVar('T', bound=BaseModel)


class StructuredAgent(Agent):
    """Agent with native structured output support.
    
    This class extends the base Agent to provide built-in support for
    structured outputs using Pydantic models. It automatically handles
    JSON parsing, validation, and error handling.
    
    Example:
        ```python
        from pydantic import BaseModel
        from strands.agent import StructuredAgent
        
        class Analysis(BaseModel):
            sentiment: str
            score: float
            
        agent = StructuredAgent(model=model)
        result = agent("Analyze: Great product!", response_model=Analysis)
        print(result.sentiment)  # Typed result
        ```
    """
    
    def __call__(
        self, 
        prompt: str, 
        response_model: Optional[Type[T]] = None, 
        **kwargs: Any
    ) -> Union[str, T]:
        """Process a prompt with optional structured output.
        
        Args:
            prompt: The input prompt to process.
            response_model: Optional Pydantic model for structured output.
            **kwargs: Additional parameters passed to the base agent.
            
        Returns:
            If response_model is provided, returns a validated Pydantic model instance.
            Otherwise, returns the raw string response.
            
        Raises:
            ValueError: If structured output parsing fails after max retries.
            ValidationError: If the response doesn't match the Pydantic schema.
        """
        if response_model is None:
            # Standard agent behavior for unstructured output
            return super().__call__(prompt, **kwargs)
        
        return self._generate_structured_output(prompt, response_model, **kwargs)
    
    def _generate_structured_output(
        self, 
        prompt: str, 
        response_model: Type[T], 
        **kwargs: Any
    ) -> T:
        """Generate and validate structured output.
        
        Args:
            prompt: The input prompt.
            response_model: The Pydantic model class.
            **kwargs: Additional parameters.
            
        Returns:
            Validated Pydantic model instance.
            
        Raises:
            ValueError: If parsing fails after max retries.
        """
        max_retries = kwargs.pop('max_retries', 3)
        
        # Build structured prompt
        structured_prompt = self._build_structured_prompt(prompt, response_model)
        
        # Add JSON-specific system prompt
        original_system_prompt = kwargs.get('system_prompt', '')
        kwargs['system_prompt'] = original_system_prompt + self._get_json_system_prompt()
        
        last_response = None
        
        for attempt in range(max_retries):
            try:
                # Get response from base agent
                response = super().__call__(structured_prompt, **kwargs)
                last_response = str(response)
                
                # Parse and validate response
                return self._parse_and_validate_response(last_response, response_model)
                
            except (json.JSONDecodeError, ValidationError) as e:
                logger.warning(
                    f"Structured output parsing failed (attempt {attempt + 1}/{max_retries}): {e}"
                )
                
                if attempt == max_retries - 1:
                    logger.error(f"Raw response: {last_response}")
                    raise ValueError(
                        f"Failed to generate structured output after {max_retries} attempts: {e}"
                    ) from e
                    
        raise ValueError(f"Maximum retries ({max_retries}) exceeded")
    
    def _build_structured_prompt(self, prompt: str, response_model: Type[BaseModel]) -> str:
        """Build a prompt optimized for structured JSON output.
        
        Args:
            prompt: The original user prompt.
            response_model: The Pydantic model class.
            
        Returns:
            Enhanced prompt with JSON schema instructions.
        """
        schema = response_model.model_json_schema()
        
        return f"""{prompt}

Please provide your response as valid JSON that conforms to the following schema:

Schema:
{json.dumps(schema, indent=2, ensure_ascii=False)}

Requirements:
1. Output must be valid JSON
2. Include all required fields
3. Do not include any text outside the JSON object
4. Do not use code blocks or markdown formatting
5. Start your response directly with the JSON object

Format:
{{
  "field1": "value1",
  "field2": "value2"
}}"""
    
    def _get_json_system_prompt(self) -> str:
        """Get additional system prompt for JSON mode.
        
        Returns:
            System prompt addition for structured output.
        """
        return """

You are a specialized AI that generates structured JSON responses.
- Always respond with valid JSON format
- Strictly follow the provided schema
- Do not include explanatory text or code blocks
- Output only the pure JSON object"""
    
    def _parse_and_validate_response(self, response_text: str, response_model: Type[T]) -> T:
        """Parse response text and validate against Pydantic model.
        
        Args:
            response_text: Raw response from the LLM.
            response_model: The Pydantic model class.
            
        Returns:
            Validated Pydantic model instance.
            
        Raises:
            json.JSONDecodeError: If JSON parsing fails.
            ValidationError: If validation against schema fails.
        """
        # Clean the response text
        cleaned_text = self._clean_response_text(response_text)
        
        # Parse JSON
        try:
            parsed_json = json.loads(cleaned_text)
        except json.JSONDecodeError:
            # Try to extract JSON from mixed content
            cleaned_text = self._extract_json_from_text(cleaned_text)
            parsed_json = json.loads(cleaned_text)
        
        # Validate with Pydantic
        return response_model(**parsed_json)
    
    def _clean_response_text(self, text: str) -> str:
        """Clean response text to extract JSON content.
        
        Args:
            text: Raw response text.
            
        Returns:
            Cleaned text ready for JSON parsing.
        """
        text = text.strip()
        
        # Remove code blocks
        if "```json" in text:
            text = text.split("```json")[1].split("```")[0].strip()
        elif "```" in text:
            parts = text.split("```")
            if len(parts) >= 3:
                text = parts[1].strip()
        
        return text
    
    def _extract_json_from_text(self, text: str) -> str:
        """Extract JSON object from mixed text content.
        
        Args:
            text: Text containing JSON object.
            
        Returns:
            Extracted JSON string.
            
        Raises:
            json.JSONDecodeError: If no valid JSON object is found.
        """
        # Find the first { and matching }
        start_idx = text.find('{')
        if start_idx == -1:
            raise json.JSONDecodeError("No JSON object found", text, 0)
        
        brace_count = 0
        end_idx = start_idx
        
        for i, char in enumerate(text[start_idx:], start_idx):
            if char == '{':
                brace_count += 1
            elif char == '}':
                brace_count -= 1
                if brace_count == 0:
                    end_idx = i
                    break
        
        if brace_count != 0:
            raise json.JSONDecodeError("Incomplete JSON object", text, start_idx)
        
        return text[start_idx:end_idx + 1]


# Utility function for backward compatibility
def structured_query(
    agent: Agent,
    prompt: str,
    response_schema: Type[T],
    max_retries: int = 3
) -> Optional[T]:
    """Generate structured output using any Agent instance.
    
    This function provides backward compatibility and can work with both
    regular Agent and StructuredAgent instances.
    
    Args:
        agent: Agent instance (regular or structured).
        prompt: Input prompt.
        response_schema: Pydantic model class.
        max_retries: Maximum retry attempts.
        
    Returns:
        Validated Pydantic model instance or None if failed.
    """
    if isinstance(agent, StructuredAgent):
        try:
            return agent(prompt, response_model=response_schema, max_retries=max_retries)
        except Exception as e:
            logger.error(f"Structured query failed: {e}")
            return None
    else:
        # Legacy implementation for regular Agent
        return _legacy_structured_query(agent, prompt, response_schema, max_retries)


def _legacy_structured_query(
    agent: Agent,
    prompt: str,
    response_schema: Type[T],
    max_retries: int = 3
) -> Optional[T]:
    """Legacy structured query implementation for regular Agent instances.
    
    Args:
        agent: Regular Agent instance.
        prompt: Input prompt.
        response_schema: Pydantic model class.
        max_retries: Maximum retry attempts.
        
    Returns:
        Validated Pydantic model instance or None if failed.
    """
    schema_description = response_schema.model_json_schema()
    
    structured_prompt = f"""{prompt}

Please provide your response as valid JSON that conforms to the following schema:

Schema:
{json.dumps(schema_description, indent=2, ensure_ascii=False)}

Requirements:
1. Output must be valid JSON
2. Include all required fields
3. Do not include any text outside the JSON object
4. Do not use code blocks or markdown formatting"""

    for attempt in range(max_retries):
        try:
            response = agent(structured_prompt)
            response_text = str(response).strip()

            # Clean response
            if "```json" in response_text:
                response_text = response_text.split("```json")[1].split("```")[0].strip()
            elif "```" in response_text:
                response_text = response_text.split("```")[1].split("```")[0].strip()

            # Parse and validate
            parsed_json = json.loads(response_text)
            return response_schema(**parsed_json)

        except (json.JSONDecodeError, ValidationError) as e:
            logger.warning(f"Legacy structured query failed (attempt {attempt + 1}/{max_retries}): {e}")
            if attempt == max_retries - 1:
                logger.error(f"Raw response: {response_text}")

    return None 