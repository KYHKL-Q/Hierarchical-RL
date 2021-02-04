# Folder for trained models
## The following files will be saved under the subfolder of corresponding city when RL_train.py is run:
- bed_action_model.pth: DQN for hospital beds.
- mask_action_model.pth: DQN for surgical masks.
- bed_Actor_model.pth: Continuous actor for hospital beds.
- mask_Actor_model.pth: Continuous actor for surgical masks.

## The following files will be saved under the subfolder of corresponding city when RNN_train.py is run:
- L_RNN.pth: information rebuilding model for 'Exposed'.
- Iut_RNN.pth: information rebuilding model for 'Infected but untested'.
- R_RNN.pth: information rebuilding model for 'Recovered'.

All models are saved in the form of 'state_dict' in PyTorch.