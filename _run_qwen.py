from chatbots.centurio_qwen import CenturioQwen
from task_loader.task_loader import TaskLoader

qwen = CenturioQwen()
loader = TaskLoader("samples.csv")

print("Centurio Qwen\n")

for task in loader.tasks():
    print(task, "\n")
    print(qwen.chat(task.image, task.prompt))