import matplotlib.pyplot as plt

# Path to your file
filename = "tabularQL/tabular_results.txt"

data = []

with open(filename, "r") as f:
    i = 0
    for line in f:
        if i == 0:
            i+=1
            continue
        # Remove parentheses and spaces
        line = line.strip().replace("(", "").replace(")", "")
        if not line:
            continue
        
        episodes, epsilon, epsilon_decay, gamma, alpha, wins, losses, draws = map(float, line.split(","))

        total = wins + losses + draws
        win_pct = wins / total * 100
        loss_pct = losses / total * 100
        draw_pct = draws / total * 100

        data.append({
            "episodes": episodes,
            "epsilon": epsilon,
            "epsilon_decay": epsilon_decay,
            "gamma": gamma,
            "alpha": alpha,
            "wins": wins,
            "losses": losses,
            "draws": draws,
            "win_pct": win_pct,
            "loss_pct": loss_pct,
            "draw_pct": draw_pct
        })

# --- FIND BEST RESULT ---
# Max win percentage, and among equal wins choose the one with min losses
best = max(data, key=lambda x: (x["win_pct"], -x["loss_pct"]))

print("Best configuration:")
print(best)

# --- PLOT RESULTS ---
episodes_list = [d["episodes"] for d in data]
win_pct_list = [d["win_pct"] for d in data]
loss_pct_list = [d["loss_pct"] for d in data]
draw_pct_list = [d["draw_pct"] for d in data]

plt.figure(figsize=(12, 6))
plt.plot(episodes_list, win_pct_list, label="Win %")
plt.plot(episodes_list, loss_pct_list, label="Loss %")
plt.plot(episodes_list, draw_pct_list, label="Draw %")

plt.xlabel("Episodes")
plt.ylabel("Percentage (%)")
plt.title("Win / Loss / Draw Percentages Across Experiments")
plt.legend()
plt.grid()

# ---- ADD BEST CONFIGURATION ON THE PLOT ----
best_text = (
    f"Best configuration:\n"
    f"episodes = {best['episodes']}\n"
    f"epsilon = {best['epsilon']}\n"
    f"epsilon_decay = {best['epsilon_decay']}\n"
    f"gamma = {best['gamma']}\n"
    f"alpha = {best['alpha']}\n"
    f"win % = {best['win_pct']:.2f}\n"
    f"loss % = {best['loss_pct']:.2f}\n"
    f"draw % = {best['draw_pct']:.2f}"
)

plt.gcf().text(
    0.5, 0.3, best_text, fontsize=10,
    bbox=dict(facecolor="white", alpha=0.8)
)

#plt.show()
plt.savefig("tabular_results.png", dpi=300, bbox_inches="tight")

