import flwr as fl
import tensorflow as tf
from model import create_model

class Strategy(fl.server.strategy.FedAvg):
    def __init__(self, num_rounds=3, **kwargs):
        super().__init__(**kwargs)
        self.global_model = create_model((66,))  # Update input_shape accordingly
        self.num_rounds = num_rounds  # Default to 3 if not provided
        self.final_parameters = None  # To store the final parameters
    
    def aggregate_fit(self, rnd, results, failures):
        # Call the superclass method to perform standard aggregation
        aggregated_parameters, aggregated_metrics = super().aggregate_fit(rnd, results, failures)
        
        # Save the model after the last round
        if rnd == self.num_rounds:
            self.final_parameters = aggregated_parameters
        
        return aggregated_parameters, aggregated_metrics
    
    def save_model(self):
        if self.final_parameters is None:
            print("No final parameters to save.")
            return
        
        # Extract the list of tensors from the Parameters object
        parameters_list = self.final_parameters.tensors

        # Convert parameters to tensors and set the model weights
        print('Saving model...')
        try:
            self.global_model.set_weights([tf.convert_to_tensor(param) for param in parameters_list])
        except ValueError as e:
            print(f"Error setting weights: {e}")
        
        # Save the model to disk
        self.global_model.save("/Users/mansahaj/cybersecurity_nrc/fl_implementation/cic_fl/final_global_model.h5")
        print("Model saved as final_global_model.h5")
    
    def aggregate_evaluate(self, rnd, results, failures):
        aggregated, metrics = super().aggregate_evaluate(rnd, results, failures)
        accuracies = [res[1].metrics['accuracy'] for res in results if res[1].metrics is not None]
        if accuracies:
            average_accuracy = sum(accuracies) / len(accuracies)
            print(f"Round {rnd}: Average accuracy: {average_accuracy:.3f}")
        else:
            print(f"Round {rnd}: No accuracies to average.")
        return aggregated, metrics

# Setup the server with the custom strategy
strategy = Strategy(num_rounds=5)  # Set the number of rounds as neede
# Start the Flower server with the custom strategy
history = fl.server.start_server(server_address="127.0.0.1:8080", config=fl.server.ServerConfig(num_rounds=5), strategy=strategy)
# Save the final model after training
strategy.save_model()
