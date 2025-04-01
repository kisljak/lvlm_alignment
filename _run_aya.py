from chatbots.aya_vision import AyaVision8B
from task_loader.task_loader import TaskLoader

aya = AyaVision8B()
loader = TaskLoader("samples.csv")

print("Aya Vision 8B\n")

for task in loader.tasks():
    print("\n_____________________\n")
    print(task, "\n")
    print(aya.chat(task.image, task.prompt))