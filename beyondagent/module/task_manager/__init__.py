

import copy
import os
import time
from typing import Callable, Optional, Sequence

import hydra
from loguru import logger
from omegaconf import DictConfig
from beyondagent.client.llm_client import DashScopeClient, LlmClient
from beyondagent.module.agent_flow.agent_flow import AgentFlow
from beyondagent.module.agent_flow.base_agent_flow import BaseAgentFlow
from beyondagent.module.task_manager.env_worker import EnvWorker
from beyondagent.module.task_manager.prompt_explore import AGENT_INTERACTION_SYSTEM_PROMPT
from beyondagent.module.task_manager.prompt_summarize import AGENT_SUMMARIZE_SYSTEM_PROMPT, get_task_summarize_prompt, parse_tasks_from_response
from beyondagent.schema.task import Task
from beyondagent.schema.trajectory import Trajectory


class TaskManager(object):

    def __init__(self, config:DictConfig, llm_client: LlmClient,tokenizer, env_service_url:str,max_llm_retries: int = 3):
        self._config=config
        self._llm_client = llm_client
        self._env_service_url = env_service_url
        self._max_llm_retries = max_llm_retries
        
        self._tokenizer = tokenizer # TODO: 这玩意不该在这
    
    def _step_explore_batch(self,tasks:Sequence[Task]):
        # TODO: I have no idea what data_id and rollout_id are.
        raise NotImplementedError()
    
    def _step_explore(self,task:Task, data_id: str, rollout_id: str,max_step:Optional[int]=None):
        """
        Step 1: explore the environment to find out possible actions and their results.
        """
        # reset env every time
        env_worker=EnvWorker(env_type=task.env_type, task_id=task.task_id, instance_id=None, env_service_url=self._env_service_url)
        llm_chat_fn = self._get_llm_chat_fn() # TODO: better sampling_params for exploring
        agent_flow: BaseAgentFlow = AgentFlow(enable_context_generator=False,
                                            llm_chat_fn=llm_chat_fn, 
                                            tokenizer=self._tokenizer, 
                                            config=self._config)
        if max_step is not None:
            agent_flow.max_steps=max_step
        
        assert isinstance(task.query,str)
        traj=env_worker.execute(data_id=data_id, rollout_id=rollout_id,system_prompt=AGENT_INTERACTION_SYSTEM_PROMPT, agent_flow=agent_flow)
        
        return traj
    
    
    def _step_summarize(self,trajectories:Sequence[Trajectory]):
        """
        Step 2: summarize the results of the exploration to generate the TASK (query and gt).
        """
        llm_fn=self._get_llm_chat_fn()
        system_prompt,user_prompt=get_task_summarize_prompt(trajectories)
        
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        llm_output=llm_fn(messages=messages)['content']
        tasks=parse_tasks_from_response(llm_output)
        return tasks
    
    
    def _get_llm_chat_fn(self, sampling_params: Optional[dict] = None) -> Callable:
        def llm_chat(messages: list[dict[str, str]],
                     custom_sampling_params: Optional[dict] = None,
                     request_id: Optional[str] = None) -> dict:
            """
            input messages: [{"role": "system", "value": "..."}, {"role": "user", "value": "..."}]
            output messages: [{"role": "assistant", "value": "..."}]
            """
            # TODO: sending sampling_params to rollout server
            updated_sampling_params = {}
            if sampling_params:
                updated_sampling_params.update(sampling_params)
            if custom_sampling_params:
                updated_sampling_params.update(custom_sampling_params)

            # output_messages = []
            input_messages = copy.deepcopy(messages)
            res=None
            for i in range(self._max_llm_retries):
                try:
                    res=self._llm_client.chat(messages=input_messages,sampling_params=updated_sampling_params)
                    break

                except Exception as e:
                    logger.exception(f"rollout_server.{i} error: {e.args}")
                    time.sleep(i + 1)
            
            assert res is not None, f"rollout_server error"
            return {
                "role": "assistant",
                "content": res,
            }
        return llm_chat



@hydra.main(config_path="/Users/cc/projects/BeyondAgent/config", config_name="beyond_agent_dataflow", version_base=None)
def test(config):
    import transformers
    tokenizer=transformers.AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct", trust_remote_code=True)
    manager=TaskManager(config,DashScopeClient(),tokenizer=tokenizer,env_service_url="http://localhost:8000")
    if not os.path.exists('test-explore.json'):
        traj=manager._step_explore(Task(task_id="0a9d82a_1",env_type="appworld"),"123","123",max_step=3)
        with open('test-explore.json','w') as f:
            f.write(traj.json(indent=2))
    else:
        with open('test-explore.json','r') as f:
            traj=Trajectory.parse_raw(f.read())
    
    t=manager._step_summarize([traj])

if __name__=="__main__":
    test()