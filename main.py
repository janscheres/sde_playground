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
    return np, plt, torch


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
    return


if __name__ == "__main__":
    app.run()
