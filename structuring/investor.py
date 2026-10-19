"""
investor.py - Investor profile: objectives, market convictions and risk constraints.

The questionnaire of the interface fills an `InvestorProfile`. Market views come
in two forms: a simple `Conviction` (direction, move, volatility view), or a
detailed `ScenarioView` made of weighted `ScenarioCase`s ("+8 % within six
months with falling volatility: 50 %; stagnation: 30 %; -15 %: 20 %"), whose
probabilities are subjective. `view_on` always returns a ScenarioView, so the
engine works with one representation. Everything the
investor declares is a PREFERENCE: suitability.py checks it against the
contractual features of each structure, and nothing here is ever treated as a
guarantee that a product respects it.

Conventions: amounts in the reference currency, percentages as fractions
(0.10 = 10 %), horizons in years.
"""
from dataclasses import asdict, dataclass, field

OBJECTIVES = ("capital growth", "income", "capital protection", "diversification")
DIRECTIONS = ("up", "flat", "down")
VOL_VIEWS = ("rising", "stable", "falling")
CONFIDENCE = ("low", "medium", "high")
FEARED_SCENARIOS = ("crash", "moderate decline", "stagnant market", "high volatility", "rising rates")
KNOWLEDGE = ("basic", "intermediate", "advanced")
ASSET_CLASSES = ("equity index", "single stock", "fx", "commodity", "rates")


@dataclass(frozen=True)
class Objectives:
    capital: float = 100_000.0
    target_return: float = 0.05          # per year
    horizon: float = 5.0                 # years
    main_objective: str = "capital growth"
    needs_regular_income: bool = False
    reference_currency: str = "EUR"

    def __post_init__(self):
        if self.capital <= 0:
            raise ValueError("The capital must be positive.")
        if self.horizon <= 0:
            raise ValueError("The horizon must be positive.")
        if self.main_objective not in OBJECTIVES:
            raise ValueError(f"Unknown objective {self.main_objective!r}.")


@dataclass(frozen=True)
class Conviction:
    """A view on one underlying: direction, expected move over the horizon, volatility view, confidence."""
    underlying: str = "Euro Stoxx 50"
    direction: str = "up"
    expected_move: float = 0.10          # expected change over the horizon (0.10 = +10 %); its sign follows direction
    horizon: float = 3.0                 # years
    volatility_view: str = "stable"
    confidence: str = "medium"

    def __post_init__(self):
        if self.direction not in DIRECTIONS:
            raise ValueError(f"Unknown direction {self.direction!r}.")
        if self.volatility_view not in VOL_VIEWS:
            raise ValueError(f"Unknown volatility view {self.volatility_view!r}.")
        if self.confidence not in CONFIDENCE:
            raise ValueError(f"Unknown confidence {self.confidence!r}.")
        if self.horizon <= 0:
            raise ValueError("The conviction horizon must be positive.")
        if self.expected_move <= -1:
            raise ValueError("A move cannot be below -100 %.")

    @property
    def signed_move(self) -> float:
        """Expected move with the sign implied by the direction (0 for a flat view)."""
        if self.direction == "flat":
            return 0.0
        magnitude = abs(self.expected_move)
        return magnitude if self.direction == "up" else max(-magnitude, -0.99)


@dataclass(frozen=True)
class ScenarioCase:
    """
    One case of the investor's scenario: 'the market moves `move` within `horizon` years, with the
    volatility changed by `vol_change`' (absolute, -0.03 = 3 points lower), with a SUBJECTIVE probability.
    After its horizon the market is assumed flat (median unchanged) until the product matures.
    """
    label: str = "Rise"
    probability: float = 1.0
    move: float = 0.08
    horizon: float = 0.5
    vol_change: float = 0.0

    def __post_init__(self):
        if not 0 <= self.probability <= 1:
            raise ValueError("A probability lies between 0 and 1.")
        if self.horizon <= 0:
            raise ValueError("A scenario horizon must be positive.")
        if self.move <= -1:
            raise ValueError("A move cannot be below -100 %.")
        if self.vol_change < -1:
            raise ValueError("The volatility change is absolute (e.g. -0.03).")


@dataclass(frozen=True)
class ScenarioView:
    """The investor's described scenario on one underlying: weighted cases whose probabilities sum to 1."""
    underlying: str = "Euro Stoxx 50"
    cases: tuple[ScenarioCase, ...] = (ScenarioCase(),)
    confidence: str = "medium"

    def __post_init__(self):
        if not self.cases:
            raise ValueError("A scenario needs at least one case.")
        total = sum(c.probability for c in self.cases)
        if abs(total - 1) > 1e-6:
            raise ValueError(f"The probabilities of the cases sum to {total:.0%}, not 100 %.")
        if self.confidence not in CONFIDENCE:
            raise ValueError(f"Unknown confidence {self.confidence!r}.")

    @classmethod
    def from_conviction(cls, conviction: "Conviction") -> "ScenarioView":
        vol = {"rising": 0.05, "stable": 0.0, "falling": -0.03}[conviction.volatility_view]
        return cls(conviction.underlying, (ScenarioCase(conviction.direction, 1.0, conviction.signed_move,
                                                        conviction.horizon, vol),), conviction.confidence)

    @property
    def expected_move(self) -> float:
        return sum(c.probability*c.move for c in self.cases)

    @property
    def expected_vol_change(self) -> float:
        return sum(c.probability*c.vol_change for c in self.cases)

    def probability(self, kind: str, flat_band: float = 0.03) -> float:
        """Subjective probability of 'up', 'down' or 'flat' (|move| <= flat_band)."""
        test = {"up": lambda m: m > flat_band, "down": lambda m: m < -flat_band, "flat": lambda m: abs(m) <= flat_band}
        return sum(c.probability for c in self.cases if test[kind](c.move))

    @property
    def main_direction(self) -> str:
        return max(("up", "flat", "down"), key=self.probability)

    @property
    def upside_target(self) -> float:
        """Largest positive move among the cases (0 if none): where a call spread should stop."""
        return max([c.move for c in self.cases if c.move > 0], default=0.0)

    @property
    def downside_target(self) -> float:
        return min([c.move for c in self.cases if c.move < 0], default=0.0)

    @property
    def shortest_horizon(self) -> float:
        return min(c.horizon for c in self.cases)

    def describe(self) -> str:
        return "; ".join(f"{c.probability:.0%}: {c.label} ({c.move:+.0%} in {c.horizon:g}y, vol {c.vol_change:+.0%})"
                         for c in self.cases)


@dataclass(frozen=True)
class RiskConstraints:
    max_loss_fraction: float = 0.10      # maximum acceptable loss at maturity, fraction of capital
    max_temporary_loss_fraction: float = 0.20   # acceptable mark-to-market drawdown during the life
    accepts_total_loss: bool = False
    may_need_to_sell_early: bool = True
    max_lockup_years: float = 5.0
    accepts_issuer_credit_risk: bool = True
    accepts_capital_at_risk: bool = True
    knowledge: str = "intermediate"
    feared_scenarios: tuple[str, ...] = ("crash",)

    def __post_init__(self):
        if not 0 <= self.max_loss_fraction <= 1 or not 0 <= self.max_temporary_loss_fraction <= 1:
            raise ValueError("Loss limits are fractions between 0 and 1.")
        if self.knowledge not in KNOWLEDGE:
            raise ValueError(f"Unknown knowledge level {self.knowledge!r}.")
        unknown = set(self.feared_scenarios) - set(FEARED_SCENARIOS)
        if unknown:
            raise ValueError(f"Unknown feared scenarios {sorted(unknown)}.")


@dataclass(frozen=True)
class InvestorProfile:
    objectives: Objectives = field(default_factory=Objectives)
    convictions: tuple[Conviction, ...] = (Conviction(),)
    constraints: RiskConstraints = field(default_factory=RiskConstraints)
    scenarios: tuple[ScenarioView, ...] = ()     # detailed scenarios; they take precedence over the convictions

    def __post_init__(self):
        if not self.convictions:
            raise ValueError("At least one conviction is needed.")

    def view_on(self, underlying: str) -> ScenarioView:
        """The detailed scenario on this underlying, else the one implied by the simple conviction."""
        for view in self.scenarios:
            if view.underlying == underlying:
                return view
        return ScenarioView.from_conviction(self.conviction_on(underlying))

    def max_loss_amount(self) -> float:
        """The declared maximum loss in currency: a preference, not a property of any product."""
        return self.constraints.max_loss_fraction*self.objectives.capital

    def conviction_on(self, underlying: str) -> Conviction:
        """The investor's view on this underlying, or a neutral view if none was given."""
        for conviction in self.convictions:
            if conviction.underlying == underlying:
                return conviction
        return Conviction(underlying, "flat", 0.0, self.objectives.horizon, "stable", "low")

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "InvestorProfile":
        constraints = dict(data["constraints"])
        constraints["feared_scenarios"] = tuple(constraints.get("feared_scenarios", ()))
        scenarios = tuple(ScenarioView(v["underlying"], tuple(ScenarioCase(**c) for c in v["cases"]),
                                       v.get("confidence", "medium")) for v in data.get("scenarios", ()))
        return cls(Objectives(**data["objectives"]), tuple(Conviction(**c) for c in data["convictions"]),
                   RiskConstraints(**constraints), scenarios)
