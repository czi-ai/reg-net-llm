# Save the model
import lightning.pytorch as pl
import wandb
                
# class WandBModelSaver(pl.Callback):
#     def __init__(self, checkpoint_path):
#         self.checkpoint_path = checkpoint_path
    
#     def on_save_checkpoint(self, trainer, pl_module, checkpoint):
#         # Path to the checkpoint file
#         # checkpoint_path = checkpoint["callbacks"][trainer.checkpoint_callback.__class__.__name__]["filepath"]
#         wandb.save(self.checkpoint_path) # Save local checkpoint to wandb cloud
#         print("+_+_+_+_+_+_+_+_+_+_+_+_+_+_+_+_+_+_+_+_+_+_+_+_", checkpoint_path)


class SaveModelEveryNSteps(pl.Callback):
    def __init__(self, save_freq=5000):
        self.save_freq = save_freq

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        # Save every 'self.save_freq' steps
        if trainer.global_step % self.save_freq == 0:
            checkpoint_path = trainer.checkpoint_callback.last_model_path
            wandb.save(checkpoint_path) # Save local to wandb cloud
            print(f"Model saved to wandb at step {trainer.global_step}")
