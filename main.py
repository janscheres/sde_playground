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
    return mo, nn, np, optim, plt, torch


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
        mo.md("# SDE Time Evolution"),
        mo.md("Drag the slider to watch the **Forward Diffusion Process** (Equation 1)."),
        timeSlider,
        showFrame(timeSlider.value)
    ])
    return


@app.cell
def _(torch):
    def getMarginalParams(t):
        beta = 1.0

        # X_t = alpha * X_0 + sigma * noise
        logMeanCoeff = -0.5 * beta * t
        alpha = torch.exp(logMeanCoeff)
        sigma = torch.sqrt(1 - torch.exp(2 * logMeanCoeff))

        return alpha.view(-1, 1), sigma.view(-1, 1)
    return (getMarginalParams,)


@app.cell
def _(nn, torch):
    class ScoreNet(nn.Module):
        def __init__(self):
            super().__init__()
            # Input: (x,y,t)
            self.net = nn.Sequential(
                nn.Linear(3, 64),
                nn.Tanh(),
                nn.Linear(64, 128),
                nn.Tanh(),
                nn.Linear(128, 2) # Output: Estimated Noise Direction
            )

        def forward(self, x, t):
            inputData = torch.cat([x, t], dim=1)
            return self.net(inputData)
    return (ScoreNet,)


@app.cell
def _(ScoreNet, getMarginalParams, mo, optim, plt, spiralData, torch):
    def trainAndShow(dataPoints, epochs):
        model = ScoreNet()
        optimizer = optim.Adam(model.parameters(), lr=0.01)
        dataset = torch.utils.data.TensorDataset(dataPoints)
        loader = torch.utils.data.DataLoader(dataset, batch_size=256, shuffle=True)

        lossHistory = []

        for epoch in range(epochs):
            epochLoss = 0
            for (xBatch,) in loader:
                optimizer.zero_grad()
                t = torch.rand(xBatch.shape[0], 1)
                alpha, sigma = getMarginalParams(t)
                noise = torch.randn_like(xBatch)
                xNoisy = alpha * xBatch + sigma * noise
                predictedNoise = model(xNoisy, t)
                loss = torch.mean((predictedNoise - noise)**2)
                loss.backward()
                optimizer.step()
                epochLoss += loss.item()

            lossHistory.append(epochLoss / len(loader))

            # every 50 epochs update the loss graph
            if epoch % 50 == 0:
                fig, ax = plt.subplots(figsize=(6, 4))
                ax.plot(lossHistory, color='red', linewidth=2)
                ax.set_title(f"Live Training: Epoch {epoch}/{epochs}")
                ax.set_xlabel("Epochs")
                ax.set_ylabel("Loss (MSE)")
                ax.grid(True, alpha=0.3)
                ax.set_xlim(0, epochs)

                mo.output.replace(fig)

                plt.close(fig)

        return model, lossHistory


    cleanData = spiralData(1000)

    trainedModel, history = trainAndShow(cleanData, epochs=1000)

    mo.md("# **Training Complete!**")
    return history, trainedModel


@app.cell
def _(history, plt):
    plt.figure(figsize=(8, 5))
    plt.plot(history, label='Training Loss', color='darkblue')
    plt.title('Final Training Loss Curve')
    plt.xlabel('Epochs')
    plt.ylabel('Loss (MSE)')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend()
    plt.show()
    return


@app.cell
def _(getMarginalParams, np, torch):
    def reverseSdeStep(xCurr, model, beta, dt, t):
        # get t to tell the nn what the current time is
        tTensor = torch.full((xCurr.shape[0], 1), t)

        # to scale output
        _, sigmaT = getMarginalParams(tTensor)

        with torch.no_grad():
            # nn predicts the noise epsilon
            # ccore = -epsilon / sigma
            predNoise = model(xCurr, tTensor)
            score = -predNoise / sigmaT

        # calculate reverse drift -f(x) + g^2 * score, f(x) = -0.5*beta*x
        reverseDrift = (0.5 * beta * xCurr) + (beta * score)

        diffusion = np.sqrt(beta)
        randomKick = torch.randn_like(xCurr)
        dW = randomKick * np.sqrt(dt)

        # 5. Update (Note: we add drift because we are simulating "time flowing backwards")
        xNext = xCurr + (reverseDrift * dt) + (diffusion * dW)

        return xNext
    return (reverseSdeStep,)


@app.cell
def _(mo, reverseSdeStep, torch, trainedModel):
    @mo.cache
    def generateReverseTrajectory(trainedModel):
        # start with random data, similar to t=1 the end
        xPoints = torch.randn(800, 2)

        beta = 1.0
        steps = 100
        dt = 0.01

        history = [xPoints.numpy()]

        for i in reversed(range(steps)):
            tCurr = (i + 1) * dt
            xPoints = reverseSdeStep(xPoints, trainedModel, beta, dt, tCurr)
            history.append(xPoints.numpy())

        return history

    revTraj = generateReverseTrajectory(trainedModel)
    return


if __name__ == "__main__":
    app.run()
