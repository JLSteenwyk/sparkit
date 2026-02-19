from __future__ import annotations

import json
from pathlib import Path


STEMS = [
    "Explain the current evidence and mechanisms for {topic} in {context}.",
    "Compare leading methods for {topic} in {context}, including known tradeoffs.",
    "What are major failure modes and mitigation strategies for {topic} in {context}?",
    "Summarize benchmarked progress on {topic} in {context} and identify open problems.",
]


BLUEPRINTS = [
    {
        "domain": "physics",
        "context": "modern experimental and computational physics",
        "subdomains": [
            ("quantum_information", "superconducting qubit coherence", ["decoherence", "flux noise", "dielectric loss"], ["transmon", "TLS", "error correction"]),
            ("condensed_matter", "topological phases in quantum materials", ["topological", "band structure", "edge states"], ["Berry phase", "Chern", "transport"]),
            ("plasma", "magnetic confinement fusion stability", ["tokamak", "instability", "confinement"], ["ELM", "transport", "disruption"]),
            ("astrophysics", "gravitational-wave source inference", ["waveform", "parameter estimation", "detector"], ["LIGO", "Bayesian", "noise"]),
            ("photonics", "integrated nonlinear photonics", ["waveguide", "nonlinear", "frequency conversion"], ["dispersion", "Q factor", "microresonator"]),
        ],
    },
    {
        "domain": "chemistry",
        "context": "molecular and materials chemistry",
        "subdomains": [
            ("catalysis", "electrocatalyst design for CO2 reduction", ["catalyst", "selectivity", "overpotential"], ["Faradaic efficiency", "active site", "stability"]),
            ("computational_chemistry", "DFT-based reaction pathway prediction", ["DFT", "transition state", "energy barrier"], ["functional", "solvation", "validation"]),
            ("polymer", "polymer electrolyte transport", ["ionic conductivity", "polymer", "electrolyte"], ["segmental motion", "salt", "battery"]),
            ("spectroscopy", "ultrafast spectroscopy for charge transfer", ["spectroscopy", "excited state", "charge transfer"], ["femtosecond", "kinetics", "relaxation"]),
            ("supramolecular", "self-assembly of supramolecular systems", ["self-assembly", "noncovalent", "morphology"], ["host-guest", "thermodynamics", "kinetics"]),
        ],
    },
    {
        "domain": "biology",
        "context": "quantitative and systems biology",
        "subdomains": [
            ("genomics", "single-cell RNA-seq integration", ["single-cell", "batch effect", "cell type"], ["UMAP", "normalization", "integration"]),
            ("protein_science", "protein structure prediction reliability", ["protein", "structure", "confidence"], ["MSA", "folding", "validation"]),
            ("systems_biology", "gene regulatory network inference", ["regulatory", "network", "perturbation"], ["causal", "transcription", "dynamics"]),
            ("neuroscience", "large-scale neural decoding", ["neural", "decoding", "representation"], ["spiking", "latent", "generalization"]),
            ("synthetic_biology", "CRISPR circuit design", ["CRISPR", "circuit", "off-target"], ["guide RNA", "specificity", "safety"]),
        ],
    },
    {
        "domain": "medicine",
        "context": "clinical and translational research",
        "subdomains": [
            ("oncology", "biomarker-guided cancer therapy selection", ["biomarker", "response", "stratification"], ["precision oncology", "trial", "survival"]),
            ("imaging", "AI triage in radiology workflows", ["radiology", "sensitivity", "specificity"], ["AUC", "workflow", "false positives"]),
            ("epidemiology", "causal inference in observational cohorts", ["confounding", "causal", "cohort"], ["propensity", "bias", "sensitivity analysis"]),
            ("cardiology", "risk prediction for heart failure", ["risk model", "heart failure", "calibration"], ["EHR", "time-to-event", "external validation"]),
            ("infectious_disease", "antimicrobial resistance surveillance", ["resistance", "pathogen", "surveillance"], ["genomics", "stewardship", "trend"]),
        ],
    },
    {
        "domain": "computer_science",
        "context": "modern machine learning and systems",
        "subdomains": [
            ("nlp", "retrieval-augmented generation fidelity", ["retrieval", "grounding", "hallucination"], ["RAG", "citation", "faithfulness"]),
            ("vision", "vision transformer robustness", ["transformer", "robustness", "distribution shift"], ["augmentation", "adversarial", "OOD"]),
            ("systems", "LLM inference optimization", ["latency", "throughput", "quantization"], ["KV cache", "batching", "memory"]),
            ("security", "ML model extraction defenses", ["model extraction", "query", "defense"], ["watermark", "rate limit", "privacy"]),
            ("theory", "generalization bounds for deep learning", ["generalization", "capacity", "regularization"], ["PAC-Bayes", "margin", "implicit bias"]),
        ],
    },
    {
        "domain": "materials_science",
        "context": "materials discovery and characterization",
        "subdomains": [
            ("battery_materials", "solid-state battery interfaces", ["interface", "interphase", "ionic transport"], ["dendrite", "stability", "electrolyte"]),
            ("alloys", "high-entropy alloy design", ["alloy", "phase stability", "mechanical"], ["entropy", "microstructure", "strength"]),
            ("semiconductors", "defect engineering in wide-bandgap semiconductors", ["defect", "bandgap", "dopant"], ["trap", "mobility", "recombination"]),
            ("catalytic_materials", "operando catalyst characterization", ["operando", "catalyst", "active phase"], ["spectroscopy", "structure", "turnover"]),
            ("ml_materials", "foundation models for materials screening", ["foundation model", "materials", "screening"], ["representation", "transfer", "benchmark"]),
        ],
    },
    {
        "domain": "electrical_engineering",
        "context": "electronic and energy systems",
        "subdomains": [
            ("power_systems", "grid stability with inverter-dominated systems", ["inverter", "stability", "grid"], ["frequency", "control", "resilience"]),
            ("communications", "massive MIMO channel estimation", ["MIMO", "channel estimation", "spectral efficiency"], ["beamforming", "pilot", "noise"]),
            ("signal_processing", "compressed sensing in imaging", ["compressed sensing", "reconstruction", "sparsity"], ["regularization", "sampling", "noise"]),
            ("embedded", "edge AI deployment under power constraints", ["edge", "power", "latency"], ["quantization", "compiler", "throughput"]),
            ("devices", "GaN power device reliability", ["GaN", "reliability", "breakdown"], ["thermal", "trap", "lifetime"]),
        ],
    },
    {
        "domain": "mathematics",
        "context": "applied and computational mathematics",
        "subdomains": [
            ("optimization", "nonconvex optimization in deep models", ["nonconvex", "optimization", "convergence"], ["saddle point", "stochastic", "landscape"]),
            ("numerical_pde", "stable solvers for stiff PDEs", ["PDE", "stiff", "stability"], ["time stepping", "error", "adaptivity"]),
            ("probability", "uncertainty quantification in Bayesian models", ["uncertainty", "Bayesian", "posterior"], ["prior", "sampling", "credible interval"]),
            ("graph_theory", "spectral graph methods for clustering", ["graph", "spectral", "clustering"], ["Laplacian", "cut", "eigenvalue"]),
            ("information_theory", "coding limits in noisy channels", ["coding", "capacity", "noise"], ["Shannon", "error rate", "decoder"]),
        ],
    },
    {
        "domain": "earth_science",
        "context": "climate and geophysical systems",
        "subdomains": [
            ("climate", "attribution of extreme weather events", ["attribution", "extreme event", "climate model"], ["counterfactual", "uncertainty", "ensemble"]),
            ("hydrology", "flood forecasting with hybrid models", ["flood", "forecast", "hydrology"], ["remote sensing", "rainfall", "lead time"]),
            ("geophysics", "seismic inversion with neural operators", ["seismic", "inversion", "wavefield"], ["operator learning", "regularization", "resolution"]),
            ("remote_sensing", "satellite methane plume detection", ["satellite", "methane", "detection"], ["spectral", "false alarm", "validation"]),
            ("oceanography", "ocean heat uptake diagnostics", ["ocean", "heat uptake", "circulation"], ["stratification", "reanalysis", "trend"]),
        ],
    },
    {
        "domain": "aerospace_engineering",
        "context": "aeronautics and space systems",
        "subdomains": [
            ("aerodynamics", "turbulence modeling for high-lift flows", ["turbulence", "high-lift", "CFD"], ["RANS", "LES", "validation"]),
            ("propulsion", "detonation and advanced propulsion cycles", ["propulsion", "combustion", "efficiency"], ["instability", "emissions", "cycle"]),
            ("guidance_control", "autonomous guidance in uncertain environments", ["guidance", "control", "uncertainty"], ["robust", "MPC", "safety"]),
            ("structures", "fatigue prediction for composite airframes", ["fatigue", "composite", "damage"], ["delamination", "inspection", "lifetime"]),
            ("space_systems", "onboard autonomy for planetary missions", ["autonomy", "spacecraft", "navigation"], ["fault tolerance", "planning", "resource"]),
        ],
    },
]


def _difficulty(stem_idx: int) -> str:
    if stem_idx == 0:
        return "medium"
    if stem_idx == 1:
        return "hard"
    if stem_idx == 2:
        return "hard"
    return "medium"


def _citations(stem_idx: int) -> int:
    return 3 if stem_idx in {1, 2} else 2


def build_questions() -> list[dict]:
    questions: list[dict] = []
    q_index = 1
    for domain_blueprint in BLUEPRINTS:
        domain = domain_blueprint["domain"]
        context = domain_blueprint["context"]
        for subdomain, topic, required, optional in domain_blueprint["subdomains"]:
            for stem_idx, stem in enumerate(STEMS):
                question_text = stem.format(topic=topic, context=context)
                questions.append(
                    {
                        "id": f"q{q_index:03d}",
                        "question": question_text,
                        "domain": domain,
                        "subdomain": subdomain,
                        "required_keywords": required,
                        "optional_keywords": optional,
                        "must_have_citations": _citations(stem_idx),
                        "difficulty": _difficulty(stem_idx),
                    }
                )
                q_index += 1
    return questions


def write_questions_file(path: str | Path) -> int:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)

    questions = build_questions()
    if len(questions) != 200:
        raise RuntimeError(f"Expected 200 questions, generated {len(questions)}")

    output.write_text(json.dumps(questions, indent=2))
    return len(questions)
