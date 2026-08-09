# Framework

- TaskData

- Task
    - init(TaskData)
    - infer -> TaskData
    - eval -> TaskData

- Dataset
    - iter -> TaskData
    - summary(TaskData)

- Runner (RL runner, Eval Runner)
    - run(Dataset) -> summary
