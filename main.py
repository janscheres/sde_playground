import marimo

__generated_with = "0.18.1"
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
    return animation, mo, nn, np, optim, plt, torch


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
        mo.md("# SDE Playground"),
        mo.md("## 1. The Physics (Forward Process)"),
        timeSlider,
        showFrame(timeSlider.value),
    ], align="center")
    return


@app.cell
def _(animation, plt, trajectory):
    def meltingVid(trajectory):
        fig, ax = plt.subplots(figsize=(5, 5))
        ax.set_xlim(-3.5, 3.5); ax.set_ylim(-3.5, 3.5)
        ax.grid(True, linestyle='--', alpha=0.3)
        ax.set_title("Phase 1: Melting (Forward SDE)", fontsize=10)

        scat = ax.scatter([], [], s=10, c='dodgerblue', alpha=0.6, edgecolors='black', linewidth=0.1)
        def update(frame):
            if frame >= len(trajectory): frame = len(trajectory) - 1

            data = trajectory[frame]
            scat.set_offsets(data)
            return (scat,)

        ani = animation.FuncAnimation(fig, update, frames=len(trajectory), blit=True)

        ani.save("forward.mp4", writer="ffmpeg", fps=15, dpi=100)

        plt.close(fig)

    meltingVid(trajectory)
    return


@app.cell
def _(mo):
    mo.vstack([
        mo.video("forward.mp4", width=600),
    ], align="center")
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

                mo.output.replace(mo.vstack([
                    fig,
                    mo.md("Epoch "+ str(epoch)+ " out of "+str(epochs))
                ], align="center"))

                plt.close(fig)

        return model, lossHistory


    cleanData = spiralData(1000)

    trainedModel, history = trainAndShow(cleanData, epochs=500)

    mo.md("# **Training Complete!**")
    return history, trainedModel


@app.cell
def _(history, plt):
    def finalLossPlot():
        fig, ax = plt.subplots(figsize=(6, 3))
        ax.plot(history, color='darkblue')
        ax.set_title("Model Training History")
        ax.set_xlabel("Epochs"); ax.set_ylabel("Loss")
        ax.grid(True, alpha=0.3)
        return fig
    return (finalLossPlot,)


@app.cell
def _(finalLossPlot, mo):
    mo.vstack([
        mo.md("## 2. The Brain (Training)"),
        mo.md("Below is the final loss curve, proving the AI learned the score function."),
        finalLossPlot(),
    ], align="center")
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
            # score = -epsilon / sigma
            predNoise = model(xCurr, tTensor)
            score = -predNoise /(sigmaT + 1e-5)

        # calculate reverse drift -f(x) + g^2 * score, f(x) = -0.5*beta*x
        reverseDrift = (0.5 * beta * xCurr) + (beta * score)

        diffusion = np.sqrt(beta)
        randomKick = torch.randn_like(xCurr)
        dW = randomKick * np.sqrt(dt)

        xNext = xCurr + (reverseDrift * dt) + (diffusion * dW)# use our learned drift to reverse timestep

        return xNext
    return (reverseSdeStep,)


@app.cell
def _(mo, reverseSdeStep, torch, trainedModel):
    @mo.cache
    def generateReverseTrajectory(trainedModel):
        xPoints = torch.randn(800, 2)# start with random data, similar to t=1 the end of many sde forward steps

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
    return (revTraj,)


@app.cell
def _(getMarginalParams, np, plt, torch):
    def plot_interactive_explorer(model, trajectory_data, step_index, show_points, show_vectors):
        t_val = max(0.01, 1.0 - (step_index / 100.0))#convert step into time for ai, step 0 is time1, step100 is time0

        fig, ax = plt.subplots(figsize=(6, 6))

        if show_vectors:
            x = np.linspace(-3.5, 3.5, 20)
            y = np.linspace(-3.5, 3.5, 20)
            gridX, gridY = np.meshgrid(x, y)#make a 20x20 grid for the arrow bases
            gridPoints = torch.tensor(np.stack([gridX, gridY], axis=-1)).float().reshape(-1, 2)#flatted to feed into nn

            tTensor = torch.full((len(gridPoints), 1), t_val)
            with torch.no_grad():
                _, sigmaT = getMarginalParams(tTensor)
                predNoise = model(gridPoints, tTensor)
                score = -predNoise / (sigmaT + 1e-5)

            vectors = score.numpy()
            ax.quiver(gridPoints[:,0], gridPoints[:,1], vectors[:,0], vectors[:,1], 
                      color='teal', alpha=0.6, scale=100, headwidth=4, label='Learned Drift')

        if show_points:
            idx = step_index
            if idx >= len(trajectory_data): idx = len(trajectory_data) - 1

            data_np = trajectory_data[idx]

            ax.scatter(data_np[:, 0], data_np[:, 1], s=10, c='crimson', alpha=0.5, label='AI Particles')

        ax.set_title(f"Reconstruction Step: {step_index} (t={t_val:.2f})")
        ax.set_xlim(-3.5, 3.5); ax.set_ylim(-3.5, 3.5)
        ax.grid(True, linestyle='--', alpha=0.2)
        if show_points or show_vectors: ax.legend(loc='upper right')

        return fig
    return (plot_interactive_explorer,)


@app.cell
def _(mo):
    dispVecs = mo.ui.checkbox(value=True, label="Display Learned Vector Field")
    dispPoints = mo.ui.checkbox(value=True, label="Display Sample Data")

    reconstructSlider = mo.ui.slider(start=0, stop=100, step=1, value=95, label="Reconstruction Step")
    return dispPoints, dispVecs, reconstructSlider


@app.cell
def _(
    dispPoints,
    dispVecs,
    mo,
    plot_interactive_explorer,
    reconstructSlider,
    revTraj,
    trainedModel,
):
    mo.vstack([
        mo.md("## 2. The Brain: Vector Field Explorer"),
        mo.md("Investigate what the AI has learned. Drag the slider to see how the **Vector Field** (Blue Arrows) guides the **Data** (Red Dots) back to the spiral."),

        mo.hstack([
            mo.vstack([
                mo.md("### View Options"),
                dispVecs,
                dispPoints,
                reconstructSlider,
            ], align="center"),

            plot_interactive_explorer(
                trainedModel,
                revTraj,
                reconstructSlider.value, 
                dispPoints.value, 
                dispVecs.value
            )
        ], align="center", gap=2)
    ])
    return


@app.cell
def _(animation, plt, revTraj):
    def unmeltingVid(revTrajectory):
        fig, ax = plt.subplots(figsize=(5, 5))
        ax.set_xlim(-3.5, 3.5); ax.set_ylim(-3.5, 3.5)
        ax.grid(True, linestyle='--', alpha=0.3)
        ax.set_title("Phase 3: Unmelting (Reverse SDE)", fontsize=10)

        scat = ax.scatter([], [], s=10, c='crimson', alpha=0.6, edgecolors='black', linewidth=0.1)

        def update(frame):
            if frame >= len(revTrajectory): frame = len(revTrajectory) - 1

            data = revTrajectory[frame]
            scat.set_offsets(data)
            return (scat,)

        ani = animation.FuncAnimation(fig, update, frames=len(revTrajectory), blit=True)

        ani.save("reverse.mp4", writer="ffmpeg", fps=15, dpi=100)

        plt.close(fig)

    unmeltingVid(revTraj)
    return


@app.cell
def _(mo):
    mo.vstack([
        mo.video("reverse.mp4", width=600),
    ], align="center")
    return


@app.cell
def _(
    getMarginalParams,
    np,
    plt,
    revTraj,
    spiralData,
    torch,
    trainedModel,
    trajectory,
):
    def exportFigs():
        times = [0, 5, 50, 100]

        fig1, axes = plt.subplots(1, 4, figsize=(16, 4), constrained_layout=True)

        for i, ax in enumerate(axes):
            data = trajectory[times[i]]
            ax.scatter(data[:, 0], data[:, 1], s=5, c='dodgerblue', alpha=0.6)

            ax.set_title(f"Diffusion Step ${times[i]}$", fontsize=20)
            ax.set_xlim(-3.5, 3.5)
            ax.set_ylim(-3.5, 3.5)
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_aspect('equal')

        plt.savefig("melting.png", dpi=300, bbox_inches='tight')
        plt.close(fig1)


        indices_rev = [0, 50, 95, 100]

        fig2, axes = plt.subplots(1, 4, figsize=(16, 4), constrained_layout=True)
        x = np.linspace(-3.5, 3.5, 20)
        y = np.linspace(-3.5, 3.5, 20)
        gridX, gridY = np.meshgrid(x, y)
        gridPoints = torch.tensor(np.stack([gridX, gridY], axis=-1)).float().reshape(-1, 2)
        spiralExample = spiralData(500).numpy()

        for i, ax in enumerate(axes):
            idx = indices_rev[i]
            t_val = max(0.01, 1.0 - (idx / 100.0))
            tTensor = torch.full((len(gridPoints), 1), t_val)

            with torch.no_grad():
                _, sigmaT = getMarginalParams(tTensor)
                predNoise = trainedModel(gridPoints, tTensor)
                score = -predNoise / (sigmaT + 1e-5)

            vectors = score.numpy()

            ax.scatter(spiralExample[:, 0], spiralExample[:, 1], s=5, c='black', alpha=0.1)
            ax.quiver(gridPoints[:,0], gridPoints[:,1], vectors[:,0], vectors[:,1], color='teal', alpha=0.8, scale=100, headwidth=4, width=0.005)

            ax.set_title(f"Reconstruction Step {idx}", fontsize=18)
            ax.set_xlim(-3.5, 3.5)
            ax.set_ylim(-3.5, 3.5)
            ax.set_xticks([])
            ax.set_yticks([])
            ax.grid(True, linestyle='--', alpha=0.3)
            ax.set_aspect('equal')

        plt.savefig("vecfield.png", dpi=300, bbox_inches='tight')
        plt.close(fig2)


        fig3, axes = plt.subplots(1, 4, figsize=(16, 4), constrained_layout=True)

        for i, ax in enumerate(axes):
            idx = indices_rev[i]

            data = revTraj[idx]
            ax.scatter(data[:, 0], data[:, 1], s=5, c='crimson', alpha=0.6)

            ax.set_title(f"Reconstruction Step {idx}", fontsize=20)
            ax.set_xlim(-3.5, 3.5)
            ax.set_ylim(-3.5, 3.5)
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_aspect('equal')

        plt.savefig("unmelting.png", dpi=300, bbox_inches='tight')
        plt.close(fig3)

    exportFigs()
    return


if __name__ == "__main__":
    app.run()
