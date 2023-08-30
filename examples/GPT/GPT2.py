from plato.servers import fedavg
from plato.clients import simple
from plato.trainers import basic
from transformers import GPT2ForQuestionAnswering
import GPT_trainer
def main():
    model = GPT2ForQuestionAnswering.from_pretrained("gpt2")

    trainer = GPT_trainer.Trainer(model=model)
    client = simple.Client(trainer=trainer)
    server = fedavg.Server(trainer=trainer)
    server.run(client)


if __name__ == "__main__":
    main()
