from typing import Optional, Sequence, Tuple

from beyondagent.schema.trajectory import Trajectory


AGENT_SUMMARIZE_SYSTEM_PROMPT = """
You are a *Task Abstraction Expert*. Your specialty is to inspect an agent's interaction history and distill concrete, goal-oriented tasks from it.

========================  YOUR JOB  ========================
1. Inspect the interactions.
2. Identify the specific goal or task the agent is attempting to achieve.
3. Abstract each goal into a clear, concise **task description**, a **query** (suitable for search or training), and the **minimal action sequence** that successfully completes the task.

=====================  ABSTRACTION RULES  ==================
- Focus on clear, goal-directed behaviour; ignore purely random exploration.  
- Group similar behaviour patterns into the same task.  
- Every task must have **at least one** action sequence that was executed successfully.  
- Each task needs an explicit completion criterion.  
- All actions listed in an action sequence must be valid and directly executable by the agent.
- All actions listed in an action sequence must be included in the available APIs of the current environment state.
- Ensure that all actions listed in an action sequence are combined into a minimum sequence from the initial state of the environment to the completion of the task. No additional information or skipped steps are allowed.

========================  OUTPUT FORMAT  ===================
For every task you identify, output exactly one block in the form below:

<task>
Description: [A precise task description. File paths or locations are allowed.]
Query: [A succinct search / training query—results only, no extra guidance.]
Confidence: [0.0 – 1.0, your confidence in this abstraction]
ActionSequence: [A minimal sequence]
</task>

===========================  EXAMPLE  ======================
<task>
Description: Get the most-liked song in my Spotify playlists.
Query: Using these APIs, now generate code to solve the actual task:\n\nMy name is: Joyce Weaver. My personal email is joyce-weav@gmail.com and phone number is 3155673041.\n\nTask:\n\nWhat is the title of the most-liked song in my Spotify playlists.
Confidence: 1.0
ActionSequence: 
# step0
print(apis.api_docs.show_app_descriptions())
# step1
print(apis.api_docs.show_api_descriptions(app_name='supervisor'))
# step2
print(apis.api_docs.show_api_doc(app_name='supervisor', api_name='show_account_passwords'))
# step3
print(apis.supervisor.show_account_passwords())
passwords = apis.supervisor.show_account_passwords()
# step4
spotify_password = [account_password for account_password in passwords if account_password["account_name"] == "spotify"][0]["password"]
print(spotify_password)
# step5
print(apis.api_docs.show_api_descriptions(app_name='spotify'))
# step6
print(apis.api_docs.show_api_doc(app_name='spotify', api_name='show_playlist_library'))
# step7
print(apis.api_docs.show_api_doc(app_name='spotify', api_name='login'))
print(apis.api_docs.show_api_doc(app_name='supervisor', api_name='show_profile'))
# step8
email = apis.supervisor.show_profile()['email']
access_token = apis.spotify.login(username=email, password=spotify_password)['access_token']
playlist_0 = apis.spotify.show_playlist_library(page_index=0, access_token=access_token)
print(apis.api_docs.show_api_doc(app_name='spotify', api_name='show_song'))
like_count = apis.spotify.show_song(song_id=136)['like_count']
# step9
page_index = 0
song_ids_all = []
while True:
    playlists = apis.spotify.show_playlist_library(page_index=page_index, access_token=access_token)
    if not playlists:
        break
    for _ in playlists:
        song_ids_all.extend(_['song_ids'])
    page_index += 1
print(song_ids_all)

max_id = -1
max_like_count = 0
for _ in song_ids_all:
    like_count = apis.spotify.show_song(song_id=_)['like_count']
    max_like_count = max(max_like_count, like_count)
    if max_like_count == like_count:
        max_id = _
answer = apis.spotify.show_song(song_id=max_id)['title']
print(answer)
apis.supervisor.complete_task(answer=answer)
</task>
"""

def _get_action_observation_pair(traj:Trajectory)->list[tuple[str,str]]:
    res=[]
    for idx,step in enumerate(traj.steps):
        # TODO: 我们并不能确定来自环境的交互结果到底是 user message 还是 tool results
        if step['role'] == 'assistant' and (idx+1<len(traj.steps) and traj.steps[idx+1]['role'] in ['tool','user']):
            # FIXME: extract action from content and use it
            res.append((step['content'],traj.steps[idx+1]['content']))
    
    return res

def get_task_summarize_prompt(trajectories:Sequence[Trajectory],len_history:int=2) -> tuple[str,str]:
    """获取任务摘要 prompt"""
    x=""
    idx=0
    for traj in trajectories:
        pairs=_get_action_observation_pair(traj)
        for k,v in enumerate(pairs):
            histories=pairs[max(0,k-len_history):k]
            # TODO: do we need reward? FIXME: reward 从哪来的？探索阶段有 query 吗？
            x+=f"""
Record {idx}:
    History: {" -> ".join([f"{_[0]}->{_[1]}" for _ in histories])}
    Action: {v[0]}
    Observation: {v[1]}
    Reward: unknown
"""
            idx+=1
    
    user_prompt=f"""Please analyze the following agent interaction sequence and abstract specific tasks from it:

{x}

Please identify the specific tasks the agent is attempting to complete in these interactions, and abstract them into clear task descriptions and queries following the specified format.
"""
    return AGENT_SUMMARIZE_SYSTEM_PROMPT,user_prompt

def parse_tasks_from_response(response: str) -> list:
    """从响应中解析任务列表"""
    tasks = []
    try:
        import re
        
        # 找到所有<task>标签中的内容
        task_matches = re.findall(r'<task>(.*?)</task>', response, re.DOTALL)
        
        for task_content in task_matches:
            task_info = {}
            lines = task_content.strip().split('\n')
            
            # 初始化ActionSequence收集相关变量
            collecting_action_sequence = False
            action_sequence_lines = []
            
            for line in lines:
                line = line.strip()
                if line.startswith('Description:'):
                    task_info['description'] = line.replace('Description:', '').strip()
                elif line.startswith('Query:'):
                    task_info['query'] = line.replace('Query:', '').strip()
                elif line.startswith('Confidence:'):
                    confidence_str = line.replace('Confidence:', '').strip()
                    try:
                        task_info['confidence'] = float(confidence_str)
                    except ValueError:
                        task_info['confidence'] = 1.0
                elif line.startswith('ActionSequence:'):
                    # ActionSequence可能有多行，需要收集所有后续行
                    collecting_action_sequence = True
                    action_sequence_text = line.replace('ActionSequence:', '').strip()
                    if action_sequence_text:
                        action_sequence_lines.append(action_sequence_text)
                elif collecting_action_sequence:
                    # 如果遇到下一个字段的开始，停止收集
                    if (line.startswith('Description:') or 
                        line.startswith('Query:') or 
                        line.startswith('Confidence:')):
                        collecting_action_sequence = False
                    else:
                        action_sequence_lines.append(line)
            
            # 组装ActionSequence
            if action_sequence_lines:
                task_info['gt'] = '\n'.join(action_sequence_lines)
            else:
                task_info['gt'] = ""
            
            # 检查必要字段
            if 'description' in task_info and 'query' in task_info:
                if 'confidence' not in task_info:
                    task_info['confidence'] = 1.0
                tasks.append(task_info)
                
    except Exception as e:
        print(f"Error parsing tasks: {e}")
    
    return tasks
