from sparkit.orchestrator.federation import FederationConfig, build_evidence_pack
from sparkit.providers.consensus import ConsensusProvider
from sparkit.providers.elicit import ElicitProvider
from sparkit.providers.paperqa2 import PaperQA2Provider
from sparkit.providers.scite import SciteProvider


def main() -> None:
    question = "Which intervention most improves progression-free survival in EGFR-mutant NSCLC?"
    providers = [
        PaperQA2Provider(),
        ElicitProvider(),
        ConsensusProvider(),
        SciteProvider(),
    ]
    pack = build_evidence_pack(
        question=question,
        providers=providers,
        config=FederationConfig(top_k=20, provider_max_items=15),
    )
    print("question:", pack.question)
    print("provider_stats:", pack.provider_stats)
    print("evidence_items:", len(pack.items))


if __name__ == "__main__":
    main()

