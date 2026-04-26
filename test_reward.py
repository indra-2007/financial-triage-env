def reward_fn(prompts, completions, **kwargs):
    print("Prompts:", len(prompts))
    print("Completions:", len(completions))
    print("task_id len:", len(kwargs.get('task_id', [])))
