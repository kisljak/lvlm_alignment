import pandas as pd

class Task:
    def __init__(self, id, lang, lang_answer, image, prompt):
        self.id = id
        self.lang = lang
        self.lang_answer = lang_answer
        self.image = image
        self.prompt = prompt

    def __str__(self):
        return f'Task(id: {self.id}, language: {self.lang}, image: {self.image}, prompt: {self.prompt})'

class TaskLoader:
    def __init__(self, task_file_name: str):
        self.task_file_name = task_file_name
        self.dataFrame = pd.read_csv(self.task_file_name)
        assert (list(self.dataFrame.columns) == ['id', 'lang', 'image', 'prompt', 'lang_answer'])


    def tasks(self):
        for _, row in self.dataFrame.iterrows():
            yield Task(row['id'], row['lang'], row['lang_answer'], "images/" + row['image'], row['prompt'])



if __name__ == "__main__":
    l = TaskLoader("../samples.csv")

    for t in l.tasks():
        print(t)