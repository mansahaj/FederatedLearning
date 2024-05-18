import flwr as fl

import flwr as fl

class Strategy(fl.server.strategy.FedAvg):
    def aggregate_evaluate(self, rnd, results, failures):
        # Call the superclass method to perform standard aggregation
        aggregated = super().aggregate_evaluate(rnd, results, failures)

        # Extract accuracy from results and compute average
        accuracies = [res[1].metrics['accuracy'] for res in results if res[1].metrics is not None]
        if accuracies:
            average_accuracy = sum(accuracies) / len(accuracies)
            print(f"Round {rnd}: Average accuracy: {average_accuracy:.3f}")
        else:
            print(f"Round {rnd}: No accuracies to average.")

        return aggregated

# Setup the server with the custom strategy
strategy = Strategy()

# Start the Flower server with the custom strategy
fl.server.start_server(server_address="127.0.0.1:8080", config=fl.server.ServerConfig(num_rounds=5), strategy=strategy)
