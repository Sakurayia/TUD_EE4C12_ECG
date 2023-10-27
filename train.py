from trainer import Trainer

def main():
    # set Trainer
    trainer = Trainer()
    trainer.train_network()
    trainer.metrics(True)

if __name__ == '__main__':
    main()