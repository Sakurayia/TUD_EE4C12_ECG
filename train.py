from trainer import Trainer

def main():
    # set Trainer
    trainer = Trainer('linear', 'model_save_linear', compute_class = True)
    trainer.train_network()

if __name__ == '__main__':
    main()