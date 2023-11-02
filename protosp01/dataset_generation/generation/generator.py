from typing import Dict, List, Any
import pandas as pd
from pandas import DataFrame
import numpy as np
import numpy.typing as npt
from sklearn.metrics.pairwise import cosine_similarity
import torch
from gen_prompt_template import PROMPT_TEMPLATE
from api_key import API_KEY
import openai
from openai.error import (
    RateLimitError,
    ServiceUnavailableError,
    APIError,
    APIConnectionError,
    Timeout,
    InvalidRequestError,
)

import time

class SkillsGenerator():

    def __init__(self, 
                 taxonomy: DataFrame, 
                 taxonomy_is_embedded: bool,
                 combination_dist: npt.ArrayLike,
                 popularity: Dict,
                 emb_model: Any = None,
                 emb_tokenizer: Any = None,
                 reference_df: DataFrame = None):
        
        if(not taxonomy_is_embedded):
            if("name+definition" not in emb_tax.columns):
                raise ValueError("The taxonomy must contain a 'name+definition' column")
            emb_tax = SkillsGenerator.embedd_df(taxonomy, "name+defintion", emb_model, emb_tokenizer)
        else :
            emb_tax = taxonomy
        
        ## embedded taxonomy
        self.emb_tax = emb_tax 


        ## combination dist
        self.combination_dist = SkillsGenerator.softmax(combination_dist)

        ## popularity measures
        self.popularity = popularity

        ## computing sim matrix
        self.compute_sim_matrix()

        ## Label to Idx and opposite
        self.idx_to_label = {k: v for k, v in enumerate(taxonomy.name.values)}
        self.label_to_idx = {v: k for k, v in enumerate(taxonomy.name.values)}




    @staticmethod
    def embedd_df(df: DataFrame, key_to_embed:str, model: Any, tokenizer: Any):
        pass


    def compute_sim_matrix(self):
        """
            Creates a new field in the class = the pairwise
            similarity matrix between all skill embeddings
        """
        skills_embeddings = torch.cat(list(self.emb_tax["embeddings"].values)).numpy()
        pairwise_sims = cosine_similarity(skills_embeddings, skills_embeddings)
        self.pairwise_sims = pairwise_sims


    def get_combination_for_(self, skill: str, k: int, threshold: float, temperature: float=1, frequency_select: bool=False, temperature_sample_size: float=1):
        """
            Creates combination of skill to pair with skill

            skill : the skill we consider
            k : the number of close neighbor of skill to consider
            threshold : maximum allowed distance between skill and a candidate
            temperature : flattening of the frequency distribution of the neighbors
            frequency_select : are the neighbors selected according to their frequency ?

        """
        skill_idx = self.label_to_idx[skill]
        sims_with_skill = self.pairwise_sims[skill_idx, :]
        kNN = (-sims_with_skill).argsort()[1:k+1]
        (-sims_with_skill).argsort()
        kNN_skills = [self.idx_to_label[nn] for nn in kNN if self.pairwise_sims[skill_idx, nn] > threshold]

        nb_associated_skills = self.get_combination_size(temperature_sample_size) - 1 ## we remove the skill selected first

        if(nb_associated_skills == 0):
            return []
        if(frequency_select):
            F = np.array([self.popularity[nn] for nn in kNN_skills])
            F = SkillsGenerator.softmax(F / F.sum(), T=temperature) ## we get a dist, potentially dumped
            # with frequency dist
            kNN_skills = list(np.random.choice(kNN_skills, size=min(nb_associated_skills, len(kNN_skills)), replace=False, p=F))
        else :
            # with uniform dist
            kNN_skills = list(np.random.choice(kNN_skills, size=min(nb_associated_skills, len(kNN_skills)), replace=False))
        return kNN_skills
    
    def get_combination_size(self, T):
        temp_dist = SkillsGenerator.softmax(self.combination_dist, T)
        return np.random.choice(np.arange(1, len(self.combination_dist) + 1), p=temp_dist)

    @staticmethod
    def softmax(X, T=1):
        return np.exp(X / T) / np.sum(np.exp(X / T))

    def stochastic_inf_iter(self, 
                            total_generations=1e7, 
                            threshold=0.0,
                            beam_size=20,
                            temperature_skill=1,
                            temperature_pairing=1,
                            temperature_sample_size=1,
                            frequency_select=True):
        all_skills = list(self.label_to_idx.keys())
        F = np.array([self.popularity[sk] for sk in self.label_to_idx.keys()])
        F = SkillsGenerator.softmax(F / F.sum(), temperature_skill)
        
        for gen in range(total_generations):
            skill = np.random.choice(all_skills, p=F) ## chosing the skill to generate
            combs = [skill] + self.get_combination_for_(skill, 
                                                        threshold=threshold,
                                                        k=beam_size,
                                                        temperature=temperature_pairing,
                                                        frequency_select=frequency_select,
                                                        temperature_sample_size=temperature_sample_size) ## get the tuple to generate
            yield combs


    def deterministic_iter(self):
        pass



class DatasetGenerator():


    def __init__(self,
                 emb_tax: DataFrame,
                 reference_df: DataFrame = None,
                 emb_model: Any = None,
                 emb_tokenizer: Any = None):
        openai.api_key = API_KEY

        ## reference dataset, use for precise few shots
        if(reference_df is not None):
            if("skill+sentence" not in reference_df.columns):
                raise ValueError("The taxonomy must contain a 'skill+sentence' column")
            self.references = SkillsGenerator.embedd_df(reference_df, "skill+sentence", emb_model, emb_tokenizer)


    def generate_ds(self, skills):
        pass

    def query(self, 
              messages:List[Dict[str]],
              model: str="gpt-3.5-turbo"):
        try:
            response = openai.ChatCompletion.create(
                model=model, messages=messages, request_timeout=20
            )
            return response
        except (
            RateLimitError,
            ServiceUnavailableError,
            APIError,
            Timeout,
        ) as e:  # Exception
            print(f"Timed out {e}. Waiting for 5 seconds.")
            time.sleep(10)


    def create_prompt_for(self, 
                          mode:str,
                          skills: List[str],
                          specific_few_shots:bool):
        (system_prompt, instruction_field, shots_field) = (..., ..., ...)
        

        ## basic system prompt to get in the role
        system_prompt = PROMPT_TEMPLATE[mode]["role_instruction"]
        ## only one sample for now
        instruction_field = PROMPT_TEMPLATE[mode]["instruction"].format(N_EXAMPLES=1)

        messages = [
            {
                'role': "system",
                "content":system_prompt
            },
            {
                "role": "user",
                "content": instruction_field
            }
            ]
        shots = None
        if(specific_few_shots):
            shots = self.generate_specific_few_shots(skills)
        else :
            shots = PROMPT_TEMPLATE[mode]["shots"]
        
        for shot in shots:
            skills, posting = ... ## get sen
            messages.append({'role':'user', 'content':skills})
            messages.append({'role':'assistant', 'content':posting})

        messages.append({'role': 'user', 'content': skills})

        return messages




    def generate_specific_few_shots(self, skills):
        pass

