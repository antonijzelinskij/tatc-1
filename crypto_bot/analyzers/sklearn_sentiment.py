"""
SklearnSentimentAnalyzer
------------------------
Локальная ML-модель: TF-IDF (1-2 граммы) + Logistic Regression.

Обучается при инициализации на встроенном датасете крипто-заголовков
(~200 размеченных примеров). Работает полностью офлайн, не требует
GPU, transformers или интернет-соединения.

Интерфейс: идентичен CryptoBERTAnalyzer — возвращает Signal с
action=BUY/SELL/HOLD, confidence [0..1], raw_label=Bullish/Bearish/Neutral.

Применение:
    from crypto_bot.analyzers.sklearn_sentiment import SklearnSentimentAnalyzer
    analyzer = SklearnSentimentAnalyzer(
        inference_latency_ms=5,
        network_latency_ms=50,
    )
"""

from __future__ import annotations

import logging
from datetime import timedelta

from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from crypto_bot.analyzers.base import BaseAnalyzer
from crypto_bot.models.domain import Action, NewsItem, Signal

logger = logging.getLogger(__name__)

# ── Размеченный обучающий датасет ────────────────────────────────────────────
#
# Заголовки основаны на реальных новостях 2023-2025 годов.
# Каждый класс: Bullish / Bearish / Neutral
# ─────────────────────────────────────────────────────────────────────────────

_BULLISH: list[str] = [
    # ETF-событие (январь 2024)
    "SEC approves 11 spot Bitcoin ETFs including BlackRock and Fidelity products",
    "Bitcoin ETFs debut with record 4.6 billion in volume on first day",
    "Bitcoin ETF inflows surpass 10 billion in first month of trading",
    "BlackRock Bitcoin ETF surpasses 250000 BTC in holdings",
    "Fidelity spot Bitcoin ETF attracts institutional mega-buyers",
    # ATH / ралли
    "Bitcoin breaks all-time high as institutional demand surges",
    "Bitcoin matches 2021 all-time high near 69000",
    "Bitcoin sets new record at 73737 as ETF demand explodes",
    "Bitcoin crosses 100000 for first time in history milestone",
    "Bitcoin tops 90000 as institutional demand reaches euphoric levels",
    "Bitcoin surges past 80000 on Trump election victory",
    "Bitcoin breaks 75000 all-time high on US election results",
    "Crypto market cap hits 3 trillion institutional buying wave",
    # Халвинг / майнинг
    "Bitcoin halving complete supply squeeze begins historically bullish",
    "Bitcoin mining difficulty reaches all-time high network strength",
    "Post-halving scarcity narrative drives bitcoin higher analyst",
    # MicroStrategy / корпоративные закупки
    "MicroStrategy purchases 18300 BTC brings holdings to 252000",
    "MicroStrategy adds 9245 BTC for 623 million treasury reserve",
    "Strategy buys bitcoin dip adds 7633 BTC total holdings grow",
    "Major hedge fund reveals massive bitcoin long position 13F filing",
    "Public company adds bitcoin to balance sheet shareholder approval",
    # Ethereum хорошие новости
    "SEC approves spot Ethereum ETF applications unexpected move",
    "Ethereum spot ETFs begin trading 1 billion first day volume",
    "Ethereum Dencun upgrade activated EIP-4844 reduces L2 fees 99 percent",
    "Ethereum ETF approval triggers massive crypto market rally",
    "Ethereum breaks 4000 as ETF demand and DeFi adoption surge",
    "Ethereum staking yields attract billions from institutional investors",
    "Ethereum layer 2 ecosystem reaches 40 billion TVL milestone",
    # Регуляторное позитивное
    "Trump nominates pro-crypto SEC chair huge win for industry",
    "Congress passes comprehensive crypto regulation framework clarity",
    "US regulators approve crypto custody for major banks",
    "Federal Reserve cuts rates 50bps bitcoin surges 8 percent",
    "Fed delivers first rate cut since 2020 bitcoin rallies above 65000",
    "Strategic bitcoin reserve executive order signed US government",
    # Adoption
    "PayPal expands crypto payment options to 35 million merchants globally",
    "El Salvador bitcoin adoption model spreading to other nations",
    "Coinbase gets full banking licence approval EU expansion",
    "Major payment network integrates bitcoin lightning instantly",
    "Apple Pay integrates crypto payments for 1 billion users",
    "Microsoft adds bitcoin treasury holding shareholder vote approved",
    # DeFi / Staking
    "DeFi total value locked surpasses 200 billion record high",
    "Bitcoin lightning network capacity reaches all-time high",
    "Ethereum staking ratio climbs to 30 percent of supply",
    "Layer 2 rollup transactions exceed ethereum mainnet volume",
    # Цена / технический
    "Bitcoin breaks above 50000 for first time since December 2021",
    "Bitcoin bulls reclaim 60000 level strong weekly close",
    "Crypto market green across board bitcoin leads recovery",
    "Bitcoin spot volume surges indicating strong buyer demand",
    "Bitcoin RSI reaches buy zone after correction analysts bullish",
    "Institutional wallets accumulate bitcoin at fastest pace 2024",
    "Bitcoin open interest rises bullish sentiment returns market",
    "Glassnode long term holders accumulate record bitcoin supply",
    "Bitcoin hash rate hits record high miner confidence strong",
    "Bitcoin network transaction fees drop record lows adoption rising",
    # Восстановление
    "Bitcoin rebounds sharply after flash crash buyers return dip",
    "Bitcoin recovers above 63000 after Mt Gox distribution complete",
    "Crypto market recovers losses institutional support confirmed",
]

_BEARISH: list[str] = [
    # Банкротства / коллапсы
    "Major crypto exchange files for bankruptcy amid liquidity crisis",
    "FTX collapse triggers industry wide contagion fears billions lost",
    "Crypto lender halts withdrawals amid insolvency rumors",
    "Celsius network freezes withdrawals depositors funds at risk",
    "Genesis crypto trading desk files for bankruptcy",
    "BlockFi files chapter 11 bankruptcy contagion spreading",
    # Регуляторное негативное
    "SEC sues largest crypto exchange securities violations lawsuit",
    "China bans all cryptocurrency transactions and mining nationwide",
    "EU imposes strict crypto travel rule compliance nightmare",
    "IRS launches crypto tax crackdown thousands audited",
    "US Treasury sanctions crypto mixer billions frozen",
    "SEC rejects spot bitcoin ETF application market crashes",
    "CFTC charges major exchange manipulation billions in fines",
    # Хаки / эксплойты
    "Ethereum defi protocol suffers 500 million exploit hackers",
    "Crypto exchange hacked 400 million stolen user funds at risk",
    "Bitcoin bridge exploit 300 million drained cross chain attack",
    "DeFi protocol reentrancy attack 100 million stolen",
    "NFT marketplace hacked user wallets drained overnight",
    # Крупные продажи / давление
    "German government sells 50000 bitcoin contributing to price decline",
    "Mt Gox creditors begin receiving bitcoin repayments sell pressure",
    "Bitcoin whale dumps 15000 BTC in single transaction market drops",
    "Miner capitulation begins forced selling pressure bitcoin down",
    "Bitcoin miners offload holdings at record pace post halving",
    "US government sells confiscated bitcoin 30000 BTC auction",
    # Крупные дропы
    "Bitcoin crashes 30 percent single day market panic selloff",
    "Crypto market flash crash bitcoin drops to 49000 liquidations",
    "Bitcoin plunges below 60000 macro fears dominate sentiment",
    "Ethereum drops 20 percent broader crypto market selloff",
    "Bitcoin breaks below key support 58000 bearish trend confirmed",
    "Crypto market loses 500 billion market cap single session",
    "Bitcoin liquidations hit 1 billion as price crashes 15 percent",
    # Макро негативное
    "Federal Reserve hikes rates aggressively crypto markets bleed",
    "US inflation surges unexpectedly bitcoin drops risk-off",
    "Recession fears mount bitcoin falls alongside stocks",
    "Dollar strengthens sharply crypto assets under pressure",
    "Risk assets selloff bitcoin correlation equities spikes",
    # Стейблкоин / Terra / Tether
    "Tether stablecoin depegs market panic contagion fears",
    "Terra Luna collapse wipes 40 billion industry shockwave",
    "USDC depegs Silicon Valley Bank exposure fears spread",
    # Прочие негативные
    "Binance faces regulatory action multiple major jurisdictions",
    "Crypto market manipulation probe major exchanges subpoenaed",
    "Bitcoin network congestion fees spike 100x users flee chain",
    "Environmental concerns bitcoin mining ban proposed legislation",
    "Crypto winter deepens no bottom in sight analyst warns",
    "Bitcoin funding rates deeply negative bearish sentiment extreme",
    "On-chain data shows long term holders distributing bitcoin",
    "Crypto VC funding dries up market sentiment extremely bearish",
    "Stablecoin redemptions surge $10 billion exits crypto market",
    "Bitcoin fear greed index hits extreme fear lowest 2023",
]

_NEUTRAL: list[str] = [
    "Bitcoin price consolidates near 40000 support level range bound",
    "Ethereum developers announce next testnet upgrade date schedule",
    "Crypto market volume remains stable amid low volatility period",
    "Bitcoin hash rate steady as miners await next difficulty adjustment",
    "DeFi protocols report normal activity levels this week",
    "Bitcoin trades sideways ahead of key economic data release",
    "Ethereum gas fees remain moderate average transaction costs",
    "Crypto regulatory talks continue lawmakers review proposals",
    "Bitcoin weekly close unchanged traders watch price levels",
    "Ethereum layer 2 total value locked holds steady weekly",
    "Bitcoin options expiry this week traders position carefully",
    "Crypto exchange reports normal trading volumes no issues",
    "Bitcoin mining pool distribution shows no concentration concerns",
    "Ethereum core developers hold weekly call progress update",
    "Crypto market awaits Federal Reserve decision on rates",
    "Bitcoin dominance unchanged altcoins mixed performance",
    "Stablecoin supply holds steady no significant flows detected",
    "Crypto market cap unchanged week over week consolidation",
    "Bitcoin technical analysis shows key levels traders watch",
    "Ethereum upgrade delayed by two weeks testing continues",
    "Crypto market mixed as investors await macro clarity",
    "Bitcoin long short ratio neutral sentiment balanced market",
    "Blockchain analytics show steady on-chain activity metrics",
    "Crypto exchange lists new trading pairs volume impact minimal",
    "Bitcoin price holds 65000 support analysts watch closely",
    "Ethereum staking withdrawals remain stable no unusual activity",
    "Crypto regulatory framework progress slow lawmakers disagree",
    "Bitcoin futures basis normalizes after period of volatility",
    "Weekly crypto review mixed signals across major assets",
    "Crypto market participants await halving with neutral outlook",
]

# ── Обучение модели ───────────────────────────────────────────────────────────

def _build_model() -> Pipeline:
    X = _BULLISH + _BEARISH + _NEUTRAL
    y = (
        ["Bullish"] * len(_BULLISH)
        + ["Bearish"] * len(_BEARISH)
        + ["Neutral"] * len(_NEUTRAL)
    )
    base_clf = LogisticRegression(
        C=2.0,
        max_iter=2000,
        class_weight="balanced",
        solver="lbfgs",
    )
    pipeline = Pipeline([
        ("tfidf", TfidfVectorizer(
            ngram_range=(1, 3),
            max_features=10_000,
            sublinear_tf=True,
            min_df=1,
        )),
        ("clf", CalibratedClassifierCV(base_clf, cv=5, method="isotonic")),
    ])
    pipeline.fit(X, y)
    logger.info(
        "SklearnSentimentAnalyzer trained | "
        "examples=%d (%dB / %dBe / %dN)",
        len(X), len(_BULLISH), len(_BEARISH), len(_NEUTRAL),
    )
    return pipeline


_LABEL_TO_ACTION = {
    "Bullish": Action.BUY,
    "Bearish": Action.SELL,
    "Neutral": Action.HOLD,
}


# ── Основной класс ────────────────────────────────────────────────────────────

class SklearnSentimentAnalyzer(BaseAnalyzer):
    """Офлайн ML-анализатор на TF-IDF + Logistic Regression.

    Args:
        inference_latency_ms:  Симулируемое время инференса (мс).
        network_latency_ms:    Симулируемая сетевая задержка (мс).
        buy_threshold:         Мин. уверенность для BUY сигнала.
        sell_threshold:        Мин. уверенность для SELL сигнала.
    """

    def __init__(
        self,
        inference_latency_ms: int = 5,
        network_latency_ms: int = 50,
        buy_threshold: float = 0.45,
        sell_threshold: float = 0.45,
    ) -> None:
        self._inference_latency_ms = inference_latency_ms
        self._network_latency_ms = network_latency_ms
        self._total_latency_ms = inference_latency_ms + network_latency_ms
        self._buy_threshold = buy_threshold
        self._sell_threshold = sell_threshold

        logger.info("Training SklearnSentimentAnalyzer...")
        self._model = _build_model()
        logger.info(
            "SklearnSentimentAnalyzer ready | latency=%dms | "
            "buy_thr=%.2f | sell_thr=%.2f",
            self._total_latency_ms,
            buy_threshold,
            sell_threshold,
        )

    # ------------------------------------------------------------------

    def analyze(self, news_item: NewsItem) -> Signal:
        text = news_item.headline[:512]
        probs = self._model.predict_proba([text])[0]
        classes = self._model.classes_  # ['Bearish', 'Bullish', 'Neutral'] (sorted)

        class_prob = dict(zip(classes, probs))
        best_label = max(class_prob, key=class_prob.get)  # type: ignore[arg-type]
        confidence = class_prob[best_label]

        raw_action = _LABEL_TO_ACTION.get(best_label, Action.HOLD)

        # Применяем пороги уверенности
        if raw_action == Action.BUY and confidence < self._buy_threshold:
            raw_action = Action.HOLD
        elif raw_action == Action.SELL and confidence < self._sell_threshold:
            raw_action = Action.HOLD

        signal = Signal(
            action=raw_action,
            confidence=confidence,
            raw_label=best_label,
            text=text,
            news_timestamp=news_item.timestamp,
            total_latency_ms=self._total_latency_ms,
        )

        logger.debug(
            "Signal | %s → %s (conf=%.3f) | latency=%dms | text=%r",
            news_item.coin,
            raw_action.value,
            confidence,
            self._total_latency_ms,
            text[:60],
        )
        return signal
