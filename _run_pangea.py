from chatbots.pangea import Pangea
from task_loader.task_loader import TaskLoader

pangea = Pangea()
loader = TaskLoader("samples.csv")

print("Pangea\n")

for task in loader.tasks():
    print(task, "\n")
    print(pangea.chat(task.image, task.prompt))