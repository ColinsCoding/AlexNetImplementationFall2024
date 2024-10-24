# Import your custom modules
import load_alexnet
import prepare_dataset
import train_alexnet_model
import save_model

def main():
    # Step 1: Load the AlexNet model
    model = load_alexnet.load_model()

    # Step 2: Prepare the dataset
    trainloader, testloader = prepare_dataset.get_dataloaders()

    # Step 3: Train the model
    trained_model = train_alexnet_model.train_model(model, trainloader)

    # Step 4: Save the trained model
    save_model.save(trained_model)

if __name__ == "__main__":
    main()
