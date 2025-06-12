"""
Prompt Research Toolkit
A comprehensive toolkit for researching and evaluating LLM prompts.
"""

import os
import json
import time
from datetime import datetime
from typing import List, Dict, Any, Optional, Callable, Union
from dataclasses import dataclass
import pandas as pd
from sqlalchemy import create_engine, Column, Integer, String, Text, Float, DateTime, ForeignKey
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, relationship, Session
from sqlalchemy.sql import func
import openai
from openai import OpenAI

import logging
logging.basicConfig(level=logging.INFO)

# Database setup
Base = declarative_base()

class Prompt(Base):
    """Stores prompt templates"""
    __tablename__ = 'prompts'
    
    id = Column(Integer, primary_key=True)
    template = Column(Text, nullable=False, unique=True)
    name = Column(String(200))  # Optional friendly name
    description = Column(Text)  # Optional description
    created_at = Column(DateTime, default=datetime.utcnow)
    metadata_ = Column(Text)  # JSON string for additional metadata
    
    # Relationships
    invocations = relationship("Invocation", back_populates="prompt")
    artifacts = relationship("Artifact", back_populates="prompt")
    evals = relationship("Eval", back_populates="prompt")

class Invocation(Base):
    """Stores raw LLM outputs"""
    __tablename__ = 'invocations'
    
    id = Column(Integer, primary_key=True)
    model = Column(String(100), nullable=False)
    run_name = Column(String(200), nullable=False)
    timestamp = Column(DateTime, default=datetime.utcnow)
    raw_output = Column(Text, nullable=False)
    prompt_id = Column(Integer, ForeignKey('prompts.id'))
    formatted_prompt = Column(Text)  # The actual prompt sent (after formatting)
    parameters = Column(Text)  # JSON string of parameters (temperature, etc.)
    
    # Relationships
    prompt = relationship("Prompt", back_populates="invocations")
    artifacts = relationship("Artifact", back_populates="invocation")

class Artifact(Base):
    """Stores parsed and postprocessed LLM outputs"""
    __tablename__ = 'artifacts'
    
    id = Column(Integer, primary_key=True)
    input_data = Column(Text, nullable=False)  # The input that generated this artifact
    parsed_output = Column(Text, nullable=False)  # Parsed/postprocessed output
    prompt_id = Column(Integer, ForeignKey('prompts.id'))
    invocation_id = Column(Integer, ForeignKey('invocations.id'))
    timestamp = Column(DateTime, default=datetime.utcnow)
    metadata_ = Column(Text)  # JSON string for additional metadata
    
    # Relationships
    prompt = relationship("Prompt", back_populates="artifacts")
    invocation = relationship("Invocation", back_populates="artifacts")
    evals = relationship("Eval", back_populates="artifact")

class Eval(Base):
    """Stores evaluation results"""
    __tablename__ = 'evals'
    
    id = Column(Integer, primary_key=True)
    eval_type = Column(String(100), nullable=False)
    run_name = Column(String(200), nullable=False)
    score = Column(Float, nullable=False)
    artifact_id = Column(Integer, ForeignKey('artifacts.id'))
    prompt_id = Column(Integer, ForeignKey('prompts.id'))
    timestamp = Column(DateTime, default=datetime.utcnow)
    metadata_ = Column(Text)  # JSON string for additional eval metadata
    
    # Relationships
    artifact = relationship("Artifact", back_populates="evals")
    prompt = relationship("Prompt", back_populates="evals")

@dataclass
class LLMConfig:
    """Configuration for LLM invocation"""
    model: str = "gpt-4.1-nano"
    temperature: float = 0.7
    max_tokens: Optional[int] = None
    api_key: Optional[str] = '<USE_ENV_VARIALBE>'

class PromptResearchToolkit:
    def __init__(self, db_path: str = "prompt_research.db", llm_config: Optional[LLMConfig] = None):
        """Initialize the toolkit with database and LLM configuration"""
        self.engine = create_engine(f'sqlite:///{db_path}')
        Base.metadata.create_all(self.engine)
        self.SessionLocal = sessionmaker(bind=self.engine)
        
        self.llm_config = llm_config or LLMConfig()
        if self.llm_config.api_key:
            self.client = OpenAI(api_key=self.llm_config.api_key)
        else:
            self.client = OpenAI()  # Uses OPENAI_API_KEY env var
    
    def get_or_create_prompt(self, template: str, name: Optional[str] = None, 
                            description: Optional[str] = None) -> int:
        """
        Get or create a prompt template
        
        Args:
            template: The prompt template text
            name: Optional friendly name
            description: Optional description
        
        Returns:
            The prompt ID
        """
        with self.SessionLocal() as session:
            # Check if prompt already exists
            prompt = session.query(Prompt).filter_by(template=template).first()
            
            if prompt:
                # Update name/description if provided and different
                if name and prompt.name != name:
                    prompt.name = name
                if description and prompt.description != description:
                    prompt.description = description
                session.commit()
                return prompt.id
            else:
                # Create new prompt
                prompt = Prompt(
                    template=template,
                    name=name,
                    description=description
                )
                session.add(prompt)
                session.commit()
                session.refresh(prompt)
                return prompt.id
    
    def invoke_llm(self, prompt_template: str, run_name: str, 
                   input_data: Optional[str] = None, 
                   prompt_name: Optional[str] = None,
                   **kwargs) -> Dict[str, Any]:
        """
        Invoke an LLM and store the raw output
        
        Args:
            prompt_template: The prompt template (may contain {input} placeholder)
            run_name: Name for this run
            input_data: Optional input to format into the prompt
            prompt_name: Optional name for the prompt template
            **kwargs: Additional parameters for the LLM call
        
        Returns:
            Dictionary with invocation_id, raw_output, and prompt_id
        """
        # Get or create prompt
        prompt_id = self.get_or_create_prompt(prompt_template, name=prompt_name)
        
        # Format the prompt if input provided
        if input_data and "{input}" in prompt_template:
            formatted_prompt = prompt_template.format(input=input_data)
        else:
            formatted_prompt = prompt_template
        
        # Merge kwargs with default config
        params = {
            "model": kwargs.get("model", self.llm_config.model),
            "temperature": kwargs.get("temperature", self.llm_config.temperature),
            "max_tokens": kwargs.get("max_tokens", self.llm_config.max_tokens),
        }
        
        # Remove None values
        params = {k: v for k, v in params.items() if v is not None}
        
        # Make the API call
        response = self.client.chat.completions.create(
            messages=[{"role": "user", "content": formatted_prompt}],
            **params
        )
        
        raw_output = response.choices[0].message.content
        
        # Store in database
        with self.SessionLocal() as session:
            invocation = Invocation(
                model=params["model"],
                run_name=run_name,
                raw_output=raw_output,
                prompt_id=prompt_id,
                formatted_prompt=formatted_prompt,
                parameters=json.dumps(params)
            )
            session.add(invocation)
            session.commit()
            session.refresh(invocation)
            
            return {
                "invocation_id": invocation.id,
                "raw_output": raw_output,
                "prompt_id": prompt_id,
                "model": params["model"]
            }
    
    def create_artifact(self, 
                       input_data: str, 
                       parsed_output: str, 
                       prompt_id: int,
                       invocation_id: Optional[int] = None,
                       metadata: Optional[Dict] = None) -> int:
        """
        Create an artifact from parsed LLM output
        
        Args:
            input_data: The input that generated this artifact
            parsed_output: The parsed/postprocessed output
            prompt_id: ID of the prompt template used
            invocation_id: Optional ID linking to the invocation
            metadata: Optional metadata dictionary
        
        Returns:
            The artifact ID
        """
        with self.SessionLocal() as session:
            artifact = Artifact(
                input_data=input_data,
                parsed_output=parsed_output,
                prompt_id=prompt_id,
                invocation_id=invocation_id,
                metadata_=json.dumps(metadata) if metadata else None
            )
            session.add(artifact)
            session.commit()
            session.refresh(artifact)
            return artifact.id
    
    def run_eval_on_artifacts(self, 
                             artifact_ids: List[int], 
                             eval_func: Callable[[str, str], float], 
                             eval_type: str,
                             run_name: str) -> List[int]:
        """
        Run an evaluation function on a set of existing artifacts
        
        Args:
            artifact_ids: List of artifact IDs to evaluate
            eval_func: Function that takes (input, output) and returns a score
            eval_type: Name/type of the evaluation
            run_name: Name for this evaluation run
        
        Returns:
            List of eval IDs created
        """
        eval_ids = []
        
        with self.SessionLocal() as session:
            for artifact_id in artifact_ids:
                artifact = session.query(Artifact).filter_by(id=artifact_id).first()
                if not artifact:
                    continue
                
                # Run the evaluation
                score = eval_func(artifact.input_data, artifact.parsed_output)
                
                # Store the result
                eval_record = Eval(
                    eval_type=eval_type,
                    run_name=run_name,
                    score=score,
                    artifact_id=artifact_id,
                    prompt_id=artifact.prompt_id
                )
                session.add(eval_record)
                session.commit()
                session.refresh(eval_record)
                eval_ids.append(eval_record.id)
        
        return eval_ids
    
    def create_and_eval_artifacts(self,
                                 inputs: List[str],
                                 prompt_template: str,
                                 run_name: str,
                                 eval_func: Callable[[str, str], float],
                                 eval_type: str,
                                 prompt_name: Optional[str] = None,
                                 parse_func: Optional[Callable[[str], str]] = None,
                                 **llm_kwargs) -> Dict[str, Any]:
        """
        Create a set of artifacts (one per input) then run an eval on them
        
        Args:
            inputs: List of inputs to process
            prompt_template: Template with {input} placeholder
            run_name: Name for this run
            eval_func: Function that takes (input, output) and returns a score
            eval_type: Name/type of the evaluation
            prompt_name: Optional name for the prompt template
            parse_func: Optional function to parse raw LLM output
            **llm_kwargs: Additional parameters for LLM invocation
        
        Returns:
            Dictionary with artifact_ids, eval_ids, and prompt_id
        """
        artifact_ids = []
        prompt_id = None
        
        for input_data in inputs:
            # Invoke the LLM
            result = self.invoke_llm(
                prompt_template=prompt_template,
                run_name=run_name,
                input_data=input_data,
                prompt_name=prompt_name,
                **llm_kwargs
            )
            
            if prompt_id is None:
                prompt_id = result["prompt_id"]
            
            # Parse the output if parse function provided
            parsed_output = parse_func(result["raw_output"]) if parse_func else result["raw_output"]
            
            # Create artifact
            artifact_id = self.create_artifact(
                input_data=input_data,
                parsed_output=parsed_output,
                prompt_id=result["prompt_id"],
                invocation_id=result["invocation_id"]
            )
            artifact_ids.append(artifact_id)
        
        # Run evaluations on all artifacts
        eval_ids = self.run_eval_on_artifacts(artifact_ids, eval_func, eval_type, run_name)
        
        return {
            "artifact_ids": artifact_ids,
            "eval_ids": eval_ids,
            "prompt_id": prompt_id
        }
    
    def display_eval_results(self, run_name: Optional[str] = None, 
                           eval_type: Optional[str] = None,
                           prompt_id: Optional[int] = None) -> pd.DataFrame:
        """
        Display evaluation results as a pandas DataFrame
        
        Args:
            run_name: Optional filter by run name
            eval_type: Optional filter by eval type
            prompt_id: Optional filter by prompt ID
        
        Returns:
            DataFrame with evaluation results
        """
        with self.SessionLocal() as session:
            query = session.query(
                Eval.id.label('eval_id'),
                Eval.eval_type,
                Eval.run_name,
                Eval.score,
                Eval.timestamp,
                Artifact.input_data.label('input'),
                Artifact.parsed_output.label('output'),
                Prompt.id.label('prompt_id'),
                Prompt.template.label('prompt_template'),
                Prompt.name.label('prompt_name')
            ).join(Artifact).join(Prompt)
            
            if run_name:
                query = query.filter(Eval.run_name == run_name)
            if eval_type:
                query = query.filter(Eval.eval_type == eval_type)
            if prompt_id:
                query = query.filter(Eval.prompt_id == prompt_id)
            
            results = query.all()
            
            df = pd.DataFrame(results)
            
            return df
    
    def compare_eval_runs(self, run_names: List[str], eval_type: Optional[str] = None) -> pd.DataFrame:
        """
        Compare evaluation runs assuming same inputs to create each evaluated set of artifacts
        
        Args:
            run_names: List of run names to compare
            eval_type: Optional filter by eval type
        
        Returns:
            DataFrame with comparative results
        """
        with self.SessionLocal() as session:
            dfs = []
            
            for run_name in run_names:
                query = session.query(
                    Eval.score,
                    Artifact.input_data.label('input'),
                    Eval.eval_type,
                    Prompt.id.label('prompt_id'),
                    Prompt.template.label('prompt_template')
                ).join(Artifact).join(Prompt).filter(Eval.run_name == run_name)
                
                if eval_type:
                    query = query.filter(Eval.eval_type == eval_type)
                
                results = query.all()
                df = pd.DataFrame(results)
                df['run_name'] = run_name
                dfs.append(df)
            
            if not dfs:
                return pd.DataFrame()
            
            # Combine all dataframes
            combined_df = pd.concat(dfs)
            
            # Pivot to show scores side by side
            pivot_df = combined_df.pivot_table(
                index=['input', 'eval_type', 'prompt_id'],
                columns='run_name',
                values='score',
                aggfunc='mean'  # In case of duplicates
            )
            
            # Calculate statistics
            pivot_df['mean'] = pivot_df[run_names].mean(axis=1)
            pivot_df['std'] = pivot_df[run_names].std(axis=1)
            pivot_df['best_run'] = pivot_df[run_names].idxmax(axis=1)
            
            return pivot_df.round(3)
    
    def compare_prompts(self, prompt_ids: List[int], eval_type: Optional[str] = None) -> pd.DataFrame:
        """
        Compare different prompts based on their evaluation scores
        
        Args:
            prompt_ids: List of prompt IDs to compare
            eval_type: Optional filter by eval type
        
        Returns:
            DataFrame with prompt comparison
        """
        with self.SessionLocal() as session:
            query = session.query(
                Prompt.id.label('prompt_id'),
                Prompt.name.label('prompt_name'),
                Prompt.template.label('prompt_template'),
                Eval.eval_type,
                func.avg(Eval.score).label('avg_score'),
                func.std(Eval.score).label('std_score'),
                func.count(Eval.id).label('eval_count'),
                func.min(Eval.score).label('min_score'),
                func.max(Eval.score).label('max_score')
            ).join(Eval).filter(Prompt.id.in_(prompt_ids))
            
            if eval_type:
                query = query.filter(Eval.eval_type == eval_type)
            
            query = query.group_by(Prompt.id, Prompt.name, Prompt.template, Eval.eval_type)
            
            results = query.all()
            df = pd.DataFrame(results)
            
            return df.round(3)
    
    def get_prompt_performance_by_input(self, prompt_id: int, eval_type: Optional[str] = None) -> pd.DataFrame:
        """
        Get detailed performance of a specific prompt broken down by input
        
        Args:
            prompt_id: ID of the prompt to analyze
            eval_type: Optional filter by eval type
        
        Returns:
            DataFrame with performance by input
        """
        with self.SessionLocal() as session:
            query = session.query(
                Artifact.input_data.label('input'),
                Eval.eval_type,
                func.avg(Eval.score).label('avg_score'),
                func.count(Eval.id).label('eval_count'),
                func.group_concat(Eval.run_name).label('runs')
            ).join(Artifact).filter(Eval.prompt_id == prompt_id)
            
            if eval_type:
                query = query.filter(Eval.eval_type == eval_type)
            
            query = query.group_by(Artifact.input_data, Eval.eval_type)
            
            results = query.all()
            df = pd.DataFrame(results)
            
            return df.round(3)

# Example evaluation functions
def length_eval(input_text: str, output_text: str) -> float:
    """Simple evaluation based on output length"""
    return len(output_text)

def contains_keywords_eval(keywords: List[str]) -> Callable[[str, str], float]:
    """Create an eval function that checks for keyword presence"""
    def eval_func(input_text: str, output_text: str) -> float:
        output_lower = output_text.lower()
        return sum(1 for keyword in keywords if keyword.lower() in output_lower) / len(keywords)
    return eval_func

def similarity_eval(input_text: str, output_text: str) -> float:
    """Simple similarity based on common words"""
    input_words = set(input_text.lower().split())
    output_words = set(output_text.lower().split())
    if not input_words:
        return 0.0
    return len(input_words.intersection(output_words)) / len(input_words)

# Example usage
if __name__ == "__main__":
    # Initialize toolkit
    toolkit = PromptResearchToolkit("research_excel.db")
    
    # Example 1: Testing different prompts for the same task
    inputs = ["tables design", "graphs", "formulas", "multiple sheets"]
    
    # Test prompt 1: Simple haiku
    prompt1 = "get me excited about with excel  - write two sentence tip about {input} "
    results1 = toolkit.create_and_eval_artifacts(
        inputs=inputs,
        prompt_template=prompt1,
        prompt_name="exiting_excel",
        run_name="test_existing",
        eval_func=length_eval,
        eval_type="length",
        temperature=0.7
    )
    
    # Test prompt 2: Creative haiku
    prompt2 = "help with excel - return some unknown feature of {input} keep it short (no more than two sentences)"
    results2 = toolkit.create_and_eval_artifacts(
        inputs=inputs,
        prompt_template=prompt2,
        prompt_name="unknown_excel",
        run_name="test_unknown",
        eval_func=length_eval,
        eval_type="length",
        temperature=0.7
    )
    
    # # Compare prompts
    # prompt_comparison = toolkit.compare_prompts([results1["prompt_id"], results2["prompt_id"]])
    # print("Prompt Comparison:")
    # print(prompt_comparison)
    
    # Example 2: Testing different temperatures for the same prompt
    # for temp in [0.3, 0.7, 1.0]:
    #     toolkit.create_and_eval_artifacts(
    #         inputs=inputs,
    #         prompt_template=prompt1,
    #         run_name=f"haiku_temp_{temp}",
    #         eval_func=length_eval,
    #         eval_type="length",
    #         temperature=temp
    #     )
    
    # # Compare runs
    # comparison = toolkit.compare_eval_runs([f"haiku_temp_{t}" for t in [0.3, 0.7, 1.0]])
    # print("\nTemperature Comparison:")
    # print(comparison)
    
    # # Example 3: Detailed analysis of a specific prompt
    # performance = toolkit.get_prompt_performance_by_input(results1["prompt_id"])
    # print("\nPrompt Performance by Input:")
    # print(performance)