from trainer import Trainer

def main():
    # set Trainer
    trainer = Trainer('cnn', 'model_save_cnn', compute_class = False)
    trainer.metrics(True)

if __name__ == '__main__':
    main()