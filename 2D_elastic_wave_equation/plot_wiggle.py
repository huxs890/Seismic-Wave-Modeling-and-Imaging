import numpy as np

class WigglePlotter:

    def __init__(
        self,
        seismogram,
        time,
        distance,
    ):

        self.seismogram = np.asarray(
            seismogram,
            dtype=np.float64
        )

        self.time = np.asarray(
            time,
            dtype=np.float64
        )

        self.distance = np.asarray(
            distance,
            dtype=np.float64
        )

        self.nt, self.ntrace = self.seismogram.shape

        # Average trace spacing
        if self.ntrace > 1:
            self.dx_trace = np.mean(
                np.diff(self.distance)
            )
        else:
            self.dx_trace = 1.0


    def plot(
        self,
        ax,
        scale=0.8,
        normalize="global",
        fill=True,
        xlabel="Distance (m)",
        ylabel="Time (s)",
        title="Wiggle seismogram",
        linewidth=0.7,
        color="black",
    ):

        # ====================================================
        # Global normalization
        # ====================================================

        if normalize == "global":

            global_max = np.max(
                np.abs(self.seismogram)
            )

            if global_max == 0.0:
                global_max = 1.0

        # ====================================================
        # Plot each trace
        # ====================================================

        for itrace in range(self.ntrace):

            trace = self.seismogram[
                :,
                itrace
            ].copy()

            # -----------------------------------------------
            # Normalization
            # -----------------------------------------------

            if normalize == "trace":

                amp = np.max(
                    np.abs(trace)
                )

                if amp > 0.0:
                    trace /= amp

            elif normalize == "global":

                trace /= global_max

            elif normalize == "none":

                pass

            else:

                raise ValueError(
                    "normalize must be "
                    "'global', 'trace', or 'none'."
                )

            # -----------------------------------------------
            # Display scaling
            # -----------------------------------------------

            trace *= (
                scale
                *
                self.dx_trace
            )

            # Wiggle x-coordinate
            x_wiggle = (
                self.distance[itrace]
                +
                trace
            )

            # -----------------------------------------------
            # Wiggle line
            # -----------------------------------------------

            ax.plot(
                x_wiggle,
                self.time,
                color=color,
                linewidth=linewidth
            )

            # -----------------------------------------------
            # Fill positive amplitude
            # -----------------------------------------------

            if fill:

                ax.fill_betweenx(
                    self.time,
                    self.distance[itrace],
                    x_wiggle,
                    where=(trace >= 0.0),
                    color=color,
                    interpolate=True
                )

        # ====================================================
        # Axis settings
        # ====================================================

        ax.set_ylim(
            self.time[-1],
            self.time[0]
        )

        ax.set_xlim(
            self.distance[0] - self.dx_trace,
            self.distance[-1] + self.dx_trace
        )

        # x-axis on top
        ax.xaxis.tick_top()
        ax.xaxis.set_label_position("top")

        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(title)

        return ax