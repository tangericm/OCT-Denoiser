import os
from typing import List, Optional

class LiveLossPlot:

    def __init__(
        self,
        out_dir: str,
        *,
        title: str = "Training / Validation Loss",
        filename: str = "loss_curve.png",
        save_every_epoch: bool = True,
        show_window: bool = True,
    ):
        self.out_dir = out_dir
        self.title = title
        self.filename = filename
        self.save_every_epoch = save_every_epoch
        self.show_window = show_window

        self.train_epochs: List[int] = []
        self.train_losses: List[float] = []
        self.val_epochs: List[int] = []
        self.val_losses: List[float] = []

        self._enabled = False
        self._plt = None
        self._fig = None
        self._ax = None
        self._train_line = None
        self._val_line = None

        # Lazy import + backend safety
        try:
            import matplotlib
            import matplotlib.pyplot as plt
            self._plt = plt

            os.makedirs(self.out_dir, exist_ok=True)

            self._fig, self._ax = plt.subplots()
            self._ax.set_title(self.title)
            self._ax.set_xlabel("Epoch")
            self._ax.set_ylabel("Loss")
            self._ax.grid(True, linestyle="--", alpha=0.4)

            (self._train_line,) = self._ax.plot([], [], label="train")
            (self._val_line,) = self._ax.plot([], [], label="val")
            self._ax.legend()

            if self.show_window:
                plt.ion()
                self._fig.show()
                self._fig.canvas.draw()

            self._enabled = True
        except Exception:
            # Headless / missing matplotlib: silently disable live plotting
            self._enabled = False

    def update(
        self,
        epoch: int,
        train_loss: float,
        val_loss: Optional[float],
        *,
        also_save_epoch_snapshot: bool = False,
    ) -> None:
        # Always keep the lists updated (even if plotting disabled)
        self.train_epochs.append(epoch)
        self.train_losses.append(float(train_loss))
        if val_loss is not None:
            self.val_epochs.append(epoch)
            self.val_losses.append(float(val_loss))

        if not self._enabled:
            return

        # Update lines
        self._train_line.set_data(self.train_epochs, self.train_losses)
        self._val_line.set_data(self.val_epochs, self.val_losses)

        # Rescale axes
        self._ax.relim()
        self._ax.autoscale_view()

        # Draw
        self._fig.canvas.draw()
        self._fig.canvas.flush_events()
        if self.show_window:
            self._plt.pause(0.001)

        # Save snapshot(s)
        out_path = os.path.join(self.out_dir, self.filename)
        self._fig.savefig(out_path, dpi=160, bbox_inches="tight")

        if self.save_every_epoch or also_save_epoch_snapshot:
            epoch_path = os.path.join(self.out_dir, f"loss_curve_epoch_{epoch:04d}.png")
            self._fig.savefig(epoch_path, dpi=160, bbox_inches="tight")
