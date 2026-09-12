# Separate quantum federated extension

This extension is a separate Package C artifact, not a contribution or experiment in the present DRL manuscript. Its target title is **Hybrid Quantum Classical Hierarchical Federated Learning for Network Selection in TN NTN Systems**.

The extension begins with centralized, FedAvg, FedProx, and hierarchical-FL classifiers trained from canonical scenario contexts. The best topology is selected by validation macro-F1 and balanced accuracy; test data remain untouched during this decision. Only then does the selected topology replace the client-side classical MLP with the hybrid VQC/QNN classifier.

The hybrid model encodes four to eight latent values into equally many simulated qubits and produces five network-selection classes. It is evaluated with five seeds in ideal state-vector, finite-shot, and simple readout/depolarizing-noise modes. Required outputs are macro-F1, balanced accuracy, convergence rounds, trainable parameter count, model-update bytes, simulated wall-clock time, and tier-dropout robustness.

The pass condition is no more than a two percentage-point loss in macro-F1 relative to the selected classical topology while reducing trainable parameters or model-update payload. The extension makes no quantum-advantage or hardware-deployment claim.
