import marimo

__generated_with = "0.18.0"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import torch
    import torch.nn as nn
    import torch.optim as optim
    import numpy as np
    import matplotlib.pyplot as plt
    import time
    import matplotlib.animation as animation
    return mo, np, plt, torch


@app.cell
def _(np, torch):
    def spiralData(nPoints=500):
        n = np.sqrt(np.random.rand(nPoints)) * 540 * (2*np.pi / 360)
        d1x = -np.cos(n) * n + np.random.rand(nPoints) * 0.1
        d1y = np.sin(n) * n + np.random.rand(nPoints) * 0.1
        data = np.stack([d1x, d1y], axis=1)
        return torch.tensor((data - data.mean(0)) / data.std(0), dtype=torch.float32)
    return (spiralData,)


@app.cell
def _(plt, spiralData):
    data = spiralData().numpy()

    plt.scatter(data[:, 0], data[:, 1], s=10)
    plt.axis('equal')
    plt.show()
    return


@app.cell
def _(np, torch):
    def sdeStep(xCurr, beta, dt):
        drift = -0.5 * beta * xCurr

        diffusion = np.sqrt(beta)

        r = torch.randn_like(xCurr)
        dW = r * np.sqrt(dt)

        xNext = xCurr + (drift * dt) + (diffusion * dW)

        return xNext
    return (sdeStep,)


@app.cell
def _(mo, sdeStep, spiralData):
    @mo.cache
    def generateTrajectory():
        xPoints = spiralData(800)
        beta = 1.0
        dt = 0.01
        steps = 100
    
        history = [xPoints.numpy()]
    
        for i in range(steps):
            xPoints = sdeStep(xPoints, beta, dt)
            history.append(xPoints.numpy())
        
        return history
    return (generateTrajectory,)


@app.cell
def _(generateTrajectory, mo):
    trajectory = generateTrajectory()


    timeSlider = mo.ui.slider(start=0, stop=100, step=1, value=0, label="Time Step ($t$)")

    return timeSlider, trajectory


@app.cell
def _(plt, trajectory):
    def showFrame(stepIndex):
        data = trajectory[stepIndex]

        fig, ax = plt.subplots(figsize=(5, 5))
        ax.scatter(data[:, 0], data[:, 1], s=10, c='dodgerblue', alpha=0.6, edgecolors='black', linewidth=0.1)

        # Styling
        ax.set_xlim(-3.5, 3.5); ax.set_ylim(-3.5, 3.5)
        ax.set_title(f"Spiral melting into Noise (Step {stepIndex})")
        ax.grid(True, linestyle='--', alpha=0.3)

        return fig
    return (showFrame,)


@app.cell
def _(mo, showFrame, timeSlider):
    mo.vstack([
        mo.md("### SDE Time Evolution"),
        mo.md("Drag the slider to watch the **Forward Diffusion Process** (Equation 1)."),
        timeSlider,
        showFrame(timeSlider.value)
    ])
    return


if __name__ == "__main__":
    app.run()
