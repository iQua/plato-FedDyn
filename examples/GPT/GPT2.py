from plato.servers import fedavg
from plato.clients import simple
from plato.trainers import basic

def main():
    """A Plato federated learning training session using FedProx."""
    trainer = basic.Trainer
    client = simple.Client(trainer=trainer)
    server = fedavg.Server(trainer=trainer)
    server.run(client)


if __name__ == "__main__":
    main()
