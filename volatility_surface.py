#Calibrate an SVI surface to historical SPY option-chain data.

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")  

import matplotlib.pyplot as plt
from matplotlib import cm
from scipy.optimize import brentq, least_squares
from scipy.stats import norm


DEFAULT_DATA = Path(__file__).parent / "data" / "spy_options_2025-12-12.csv"


@dataclass(frozen=True)
class SVI:
    #Raw-SVI parameters for total implied variance.

    a: float
    b: float
    rho: float
    m: float
    sigma: float

    def total_variance(self, log_moneyness: np.ndarray) -> np.ndarray:
        x = np.asarray(log_moneyness) - self.m
        return self.a + self.b * (
            self.rho * x + np.sqrt(x * x + self.sigma * self.sigma)
        )


def black_price(
    forward: float,
    strike: float,
    expiry: float,
    volatility: float,
    discount: float,
    option_type: str,
) -> float:
    #Black-76 option value expressed using the equity forward.

    vol_time = volatility * np.sqrt(expiry)
    d1 = (np.log(forward / strike) + 0.5 * vol_time**2) / vol_time
    d2 = d1 - vol_time
    if option_type == "call":
        return discount * (forward * norm.cdf(d1) - strike * norm.cdf(d2))
    return discount * (strike * norm.cdf(-d2) - forward * norm.cdf(-d1))


def implied_volatility(
    price: float,
    forward: float,
    strike: float,
    expiry: float,
    discount: float,
    option_type: str,
) -> float:
    #Invert a mid price to Black implied volatility.

    objective = lambda vol: black_price(  # noqa: E731
        forward, strike, expiry, vol, discount, option_type
    ) - price
    try:
        return brentq(objective, 0.01, 2.00)
    except ValueError:
        return np.nan


def infer_forward(smile: pd.DataFrame, spot: float, expiry: float) -> tuple[float, float]:
    #Infer forward and discount factor from call-put

    pairs = smile.pivot_table(
        index="strike", columns="type", values=["mark", "bid"], aggfunc="first"
    ).dropna()
    pairs = pairs[
        (pairs.index > 0.875 * spot)
        & (pairs.index < 1.125 * spot)
        & (pairs[("bid", "call")] > 0)
        & (pairs[("bid", "put")] > 0)
    ]
    if len(pairs) < 5:
        raise ValueError("Not enough paired call/put quotes to infer the forward")

    strikes = pairs.index.to_numpy()
    call_minus_put = (
        pairs[("mark", "call")] - pairs[("mark", "put")]
    ).to_numpy()

    def parity_errors(theta: np.ndarray) -> np.ndarray:
        forward, discount = theta
        return discount * (forward - strikes) - call_minus_put

    fit = least_squares(
        parity_errors,
        x0=[spot * np.exp(0.02 * expiry), np.exp(-0.04 * expiry)],
        bounds=([0.90 * spot, 0.85], [1.20 * spot, 1.00]),
        loss="soft_l1",
        f_scale=0.20,
    )
    if not fit.success:
        raise RuntimeError(f"Forward inference failed: {fit.message}")
    return float(fit.x[0]), float(fit.x[1])


def load_historical_quotes(path: Path | str = DEFAULT_DATA) -> pd.DataFrame:
    #Load, clean, and transform the snapshot

    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Historical-data CSV not found: {path}")
    raw = pd.read_csv(path, parse_dates=["quote_date", "expiration"])
    required = {
        "quote_date", "expiration", "strike", "type", "bid", "ask", "mark",
        "open_interest", "spot",
    }
    missing = required.difference(raw.columns)
    if missing:
        raise ValueError(f"Missing historical-data columns: {sorted(missing)}")

    cleaned: list[pd.DataFrame] = []
    for expiration, smile in raw.groupby("expiration", sort=True):
        quote_date = smile["quote_date"].iloc[0]
        expiry = (expiration - quote_date).days / 365.0
        spot = float(smile["spot"].iloc[0])
        forward, discount = infer_forward(smile, spot, expiry)

        quotes = smile.copy()
        quotes["expiry"] = expiry
        quotes["forward"] = forward
        quotes["discount_factor"] = discount
        quotes["log_moneyness"] = np.log(quotes["strike"] / forward)
        quotes["mid"] = (quotes["bid"] + quotes["ask"]) / 2
        quotes["relative_spread"] = (quotes["ask"] - quotes["bid"]) / quotes["mid"]

        # Keep the more liquid OTM side; it carries the cleanest volatility signal.
        is_otm = ((quotes["type"] == "put") & (quotes["strike"] < forward)) | (
            (quotes["type"] == "call") & (quotes["strike"] >= forward)
        )
        quotes = quotes[
            is_otm
            & (quotes["bid"] > 0)
            & (quotes["ask"] > quotes["bid"])
            & (quotes["relative_spread"] < 0.18)
            & (quotes["log_moneyness"].abs() <= 0.18)
            & (quotes["open_interest"] >= 25)
        ].copy()
        quotes["source_implied_vol"] = quotes["implied_volatility"]
        quotes["implied_vol"] = [
            implied_volatility(mid, forward, strike, expiry, discount, option_type)
            for mid, strike, option_type in zip(
                quotes["mid"], quotes["strike"], quotes["type"], strict=True
            )
        ]
        quotes = quotes[quotes["implied_vol"].between(0.05, 0.80)]
        if len(quotes) < 8:
            raise ValueError(f"Too few clean quotes for expiry {expiration.date()}")
        cleaned.append(quotes)

    return pd.concat(cleaned, ignore_index=True)


def calibrate_smile(
    log_moneyness: np.ndarray, implied_vol: np.ndarray, expiry: float
) -> SVI:
    #Calibrate one raw-SVI slice by least squares

    market_variance = implied_vol**2 * expiry
    x0 = np.array(
        [0.65 * market_variance.min(), 0.08 * np.sqrt(expiry), -0.35, 0.0, 0.18]
    )
    # Equity-smile bounds stabilize the weakly identified far-wing parameters.
    lower = np.array([-0.10, 1e-5, -0.999, -0.50, 0.01])
    upper = np.array([1.00, 0.50, 0.000, 0.50, 2.00])

    def residuals(theta: np.ndarray) -> np.ndarray:
        model = SVI(*theta).total_variance(log_moneyness)
        return (model - market_variance) / (2 * implied_vol * expiry)

    fit = least_squares(
        residuals,
        x0=x0,
        bounds=(lower, upper),
        loss="soft_l1",
        f_scale=0.001,
        max_nfev=10_000,
    )
    if not fit.success:
        raise RuntimeError(f"SVI calibration failed: {fit.message}")
    return SVI(*fit.x)


def calibrate_surface(quotes: pd.DataFrame) -> tuple[dict[float, SVI], pd.DataFrame]:
    #Fit all expiries and return parameters

    required = {"expiry", "log_moneyness", "implied_vol"}
    missing = required.difference(quotes.columns)
    if missing:
        raise ValueError(f"Missing quote columns: {sorted(missing)}")

    calibrated: dict[float, SVI] = {}
    diagnostics = quotes.copy()
    diagnostics["fitted_vol"] = np.nan
    for expiry, smile in diagnostics.groupby("expiry", sort=True):
        params = calibrate_smile(
            smile["log_moneyness"].to_numpy(),
            smile["implied_vol"].to_numpy(),
            float(expiry),
        )
        calibrated[float(expiry)] = params
        fitted_variance = params.total_variance(smile["log_moneyness"].to_numpy())
        diagnostics.loc[smile.index, "fitted_vol"] = np.sqrt(fitted_variance / expiry)

    diagnostics["error_vol_points"] = (
        diagnostics["fitted_vol"] - diagnostics["implied_vol"]
    )
    # Diagnostics are returned for future analysis...
    return calibrated, diagnostics


# Plotting parameters.........
def _style() -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": "#08111f",
            "axes.facecolor": "#0d192a",
            "axes.edgecolor": "#64748b",
            "axes.labelcolor": "#dbeafe",
            "axes.titlecolor": "#f8fafc",
            "xtick.color": "#cbd5e1",
            "ytick.color": "#cbd5e1",
            "text.color": "#e2e8f0",
            "grid.color": "#334155",
            "grid.alpha": 0.35,
            "font.size": 11,
        }
    )


def plot_diagnostics(
    diagnostics: pd.DataFrame, calibrated: dict[float, SVI], output: Path
) -> None:
    #Plot market smiles, fitted SVI curves, and calibration errors.

    _style()
    fig, (ax_smile, ax_error) = plt.subplots(
        2, 1, figsize=(12, 8), gridspec_kw={"height_ratios": [3, 1]}, sharex=True
    )
    palette = cm.viridis(np.linspace(0.08, 0.92, len(calibrated)))
    for color, (expiry, smile) in zip(
        palette, diagnostics.groupby("expiry", sort=True), strict=True
    ):
        dense_k = np.linspace(
            smile["log_moneyness"].min(), smile["log_moneyness"].max(), 250
        )
        label = f"{round(expiry * 365):d}d"
        fitted = np.sqrt(calibrated[float(expiry)].total_variance(dense_k) / expiry)
        ax_smile.plot(dense_k, 100 * fitted, color=color, lw=2.2, label=label)
        ax_smile.scatter(
            smile["log_moneyness"],
            100 * smile["implied_vol"],
            color=color,
            edgecolor="#f8fafc",
            linewidth=0.35,
            s=20,
            zorder=3,
        )
        ax_error.scatter(
            smile["log_moneyness"],
            10_000 * smile["error_vol_points"],
            color=color,
            edgecolor="#f8fafc",
            linewidth=0.3,
            s=18,
        )

    ax_smile.set_title("SPY · Historical SVI calibration", loc="left", fontsize=18, weight="bold")
    ax_smile.set_ylabel("Implied volatility (%)")
    ax_smile.legend(title="Expiry", ncol=5, frameon=False, loc="upper right")
    ax_smile.grid(True)
    ax_error.axhline(0, color="#94a3b8", lw=1)
    ax_error.set_xlabel("Log-moneyness  ·  ln(K/F)")
    ax_error.set_ylabel("Error (bp)")
    ax_error.grid(True)
    fig.suptitle(
        "Market snapshot: 12 Dec 2025   ·   OTM bid/ask mid-prices",
        x=0.125,
        y=0.94,
        ha="left",
        color="#94a3b8",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(output, dpi=180, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)


def plot_surface(calibrated: dict[float, SVI], output: Path) -> None:
    # Render the calibrated implied-volatility surface.

    _style()
    expiries = np.array(sorted(calibrated))
    # Common liquid-moneyness range across all five expiries.
    k_grid = np.linspace(-0.17, 0.12, 121)
    k_mesh, t_mesh = np.meshgrid(k_grid, expiries)
    vol_mesh = np.vstack(
        [np.sqrt(calibrated[t].total_variance(k_grid) / t) for t in expiries]
    )

    fig = plt.figure(figsize=(12, 7.3))
    ax = fig.add_subplot(111, projection="3d")
    surface = ax.plot_surface(
        k_mesh,
        t_mesh,
        100 * vol_mesh,
        cmap="viridis",
        linewidth=0,
        antialiased=True,
        alpha=0.96,
    )
    ax.plot_wireframe(
        k_mesh,
        t_mesh,
        100 * vol_mesh,
        rstride=1,
        cstride=15,
        color="#dbeafe",
        alpha=0.18,
        linewidth=0.55,
    )
    ax.set_title(
        "SPY implied-volatility surface · 12 Dec 2025",
        loc="left",
        pad=20,
        fontsize=18,
        weight="bold",
    )
    ax.set_xlabel("Log-moneyness  ·  ln(K/F)", labelpad=12)
    ax.set_ylabel("Years to expiry", labelpad=12)
    ax.set_zlabel("Implied volatility (%)", labelpad=10)
    ax.view_init(elev=28, azim=-128)
    ax.xaxis.pane.set_facecolor("#0d192a")
    ax.yaxis.pane.set_facecolor("#0d192a")
    ax.zaxis.pane.set_facecolor("#0d192a")
    ax.grid(True)
    colorbar = fig.colorbar(surface, ax=ax, shrink=0.6, pad=0.08)
    colorbar.set_label("Implied volatility (%)")
    fig.tight_layout()
    fig.savefig(output, dpi=180, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)


def run(output_dir: Path, data_path: Path) -> None:
    quotes = load_historical_quotes(data_path)
    calibrated, diagnostics = calibrate_surface(quotes)

    output_dir.mkdir(parents=True, exist_ok=True)
    figures_dir = output_dir / "figures"
    results_dir = output_dir / "results"
    figures_dir.mkdir(exist_ok=True)
    results_dir.mkdir(exist_ok=True)
    parameters = pd.DataFrame(
        [{"expiry": expiry, **asdict(params)} for expiry, params in calibrated.items()]
    )

    diagnostics.to_csv(results_dir / "calibration.csv", index=False)
    parameters.to_csv(results_dir / "svi_parameters.csv", index=False)
    plot_diagnostics(diagnostics, calibrated, figures_dir / "calibration_diagnostics.png")
    plot_surface(calibrated, figures_dir / "volatility_surface.png")

    rmse_bp = 10_000 * np.sqrt(np.mean(diagnostics["error_vol_points"] ** 2))
    max_error_bp = 10_000 * diagnostics["error_vol_points"].abs().max()
    quote_date = diagnostics["quote_date"].iloc[0].date()
    print(f"SPY market snapshot: {quote_date} | spot: {diagnostics['spot'].iloc[0]:.2f}")
    print(f"Calibrated {len(calibrated)} expiries / {len(quotes)} cleaned quotes")
    print(f"Volatility RMSE: {rmse_bp:.2f} bp | max error: {max_error_bp:.2f} bp")
    print(f"Results written to {output_dir.resolve()}")


if __name__ == "__main__":
    run(Path("."), DEFAULT_DATA)
