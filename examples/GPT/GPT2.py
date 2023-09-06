from plato.servers import fedavg
from plato.clients import simple
import GPT_trainer, GPTModelWrapper
def main():
    model = GPTModelWrapper.GPTModelWrapper
    server = fedavg.Server(model=model,trainer=GPT_trainer.Trainer)
    client = simple.Client(model=model,trainer=GPT_trainer.Trainer)
    server.run(client)


if __name__ == "__main__":
    main()
