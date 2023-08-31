from plato.servers import fedavg
from plato.clients import simple
from plato.trainers import basic
from transformers import GPT2ForQuestionAnswering
import GPT_trainer, GPTModelWrapper
def main():
    model = GPTModelWrapper.GPTModelWrapper
    trainer = GPT_trainer.Trainer()
    client = simple.Client(model=model)
    server = fedavg.Server(model=model,trainer=trainer)
    server.run(client)


if __name__ == "__main__":
    main()
