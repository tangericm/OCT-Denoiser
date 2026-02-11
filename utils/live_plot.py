from __future__ import annotations

import os


class LiveLossPlot:
    """Live-updating loss curve saved to PNG each validation epoch."""

    def __init__(
        self,
        out_dir: str,
        *,
        title: str = "Training / Validation Loss",
        filename: str = "loss_curve.png",
        show_window: bool = True,
    ):
        self.out_dir = out_dir
        self.title = title
        self.filename = filename
        self.show_window = show_window

        self.train_epochs: list[int] = []
        self.train_losses: list[float] = []
        self.val_epochs: list[int] = []
        self.val_losses: list[float] = []
        self.val_snrs: list[float] = []
        self.val_snr_epochs: list[int] = []

        self._enabled = False
        self._plt = None
        self._fig = None
        self._ax = None
        self._train_line = None
        self._val_line = None
        self._snr_ax = None
        self._snr_line = None

        try:
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

            self._snr_ax = self._ax.twinx()
            self._snr_ax.set_ylabel("SNR (dB)")
            (self._snr_line,) = self._snr_ax.plot([], [], label="val_snr", color="tab:green")

            lines = [self._train_line, self._val_line, self._snr_line]
            self._ax.legend(lines, [l.get_label() for l in lines], loc="best")

            if self.show_window:
                plt.ion()
                self._fig.show()
                self._fig.canvas.draw()

            self._enabled = True
        except Exception:
            self._enabled = False

    def update(
        self,
        epoch: int,
        train_loss: float,
        val_loss: float | None,
        val_snr: float | None = None,
    ) -> None:
        self.train_epochs.append(epoch)
        self.train_losses.append(float(train_loss))
        if val_loss is not None:
            self.val_epochs.append(epoch)
            self.val_losses.append(float(val_loss))
        if val_snr is not None:
            self.val_snrs.append(float(val_snr))
            self.val_snr_epochs.append(epoch)

        if not self._enabled:
            return

        self._train_line.set_data(self.train_epochs, self.train_losses)
        self._val_line.set_data(self.val_epochs, self.val_losses)
        if self.val_snrs:
            self._snr_line.set_data(self.val_snr_epochs, self.val_snrs)

        self._ax.relim()
        self._ax.autoscale_view()
        if self.val_snrs:
            self._snr_ax.relim()
            self._snr_ax.autoscale_view()

        self._fig.canvas.draw()
        self._fig.canvas.flush_events()
        if self.show_window:
            self._plt.pause(0.001)

        self._fig.savefig(
            os.path.join(self.out_dir, self.filename),
            dpi=160, bbox_inches="tight",
        )
