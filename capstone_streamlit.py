"""
capstone_streamlit.py — Quantum Study Buddy
Run: streamlit run capstone_streamlit.py
"""

import streamlit as st
import uuid
import os
import re
import math
import chromadb
from dotenv import load_dotenv
from typing import TypedDict, List
from sentence_transformers import SentenceTransformer
from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver

load_dotenv()

# ── Page Config ────────────────────────────────────────────
st.set_page_config(
    page_title="Quanta — Quantum Study Buddy",
    page_icon="⚛️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Custom CSS ─────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:ital,wght@0,400;0,700;1,400&family=Syne:wght@400;600;700;800&display=swap');

:root {
    --bg-primary:    #050a14;
    --bg-secondary:  #0a1628;
    --bg-card:       #0d1f3c;
    --accent-blue:   #00d4ff;
    --accent-purple: #7c3aed;
    --accent-green:  #00ff9f;
    --accent-orange: #ff6b35;
    --text-primary:  #e8f4fd;
    --text-muted:    #6b8cae;
    --border:        #1a3a5c;
}

html, body, .stApp {
    background-color: var(--bg-primary) !important;
    font-family: 'Syne', sans-serif;
    color: var(--text-primary);
}

/* ── Header ── */
.quanta-header {
    background: linear-gradient(135deg, #050a14 0%, #0a1628 50%, #050a14 100%);
    border-bottom: 1px solid var(--border);
    padding: 1.5rem 2rem;
    display: flex;
    align-items: center;
    gap: 1rem;
    position: relative;
    overflow: hidden;
}

.quanta-header::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0; bottom: 0;
    background: radial-gradient(ellipse at 20% 50%, rgba(0,212,255,0.05) 0%, transparent 60%),
                radial-gradient(ellipse at 80% 50%, rgba(124,58,237,0.05) 0%, transparent 60%);
}

.quanta-logo {
    font-size: 2.5rem;
    animation: pulse 3s ease-in-out infinite;
}

@keyframes pulse {
    0%, 100% { transform: scale(1); }
    50% { transform: scale(1.1); }
}

.quanta-title {
    font-family: 'Syne', sans-serif;
    font-size: 1.8rem;
    font-weight: 800;
    background: linear-gradient(90deg, var(--accent-blue), var(--accent-purple));
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin: 0;
}

.quanta-subtitle {
    font-family: 'Space Mono', monospace;
    font-size: 0.75rem;
    color: var(--text-muted);
    margin: 0;
    letter-spacing: 0.1em;
}

/* ── Sidebar ── */
section[data-testid="stSidebar"] {
    background: var(--bg-secondary) !important;
    border-right: 1px solid var(--border);
}

section[data-testid="stSidebar"] * {
    color: var(--text-primary) !important;
}

.sidebar-section {
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 1rem;
    margin-bottom: 1rem;
}

.sidebar-title {
    font-family: 'Space Mono', monospace;
    font-size: 0.65rem;
    letter-spacing: 0.15em;
    color: var(--accent-blue) !important;
    text-transform: uppercase;
    margin-bottom: 0.75rem;
    border-bottom: 1px solid var(--border);
    padding-bottom: 0.5rem;
}

.topic-item {
    display: flex;
    align-items: center;
    gap: 0.5rem;
    padding: 0.25rem 0;
    font-size: 0.78rem;
    color: var(--text-muted) !important;
    font-family: 'Space Mono', monospace;
}

.topic-item.covered {
    color: var(--accent-green) !important;
}

.topic-dot {
    width: 6px;
    height: 6px;
    border-radius: 50%;
    background: var(--border);
    flex-shrink: 0;
}

.topic-dot.covered {
    background: var(--accent-green);
    box-shadow: 0 0 6px var(--accent-green);
}

.stat-row {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 0.4rem 0;
    border-bottom: 1px solid var(--border);
    font-size: 0.8rem;
}

.stat-label {
    color: var(--text-muted) !important;
    font-family: 'Space Mono', monospace;
    font-size: 0.7rem;
}

.stat-value {
    color: var(--accent-blue) !important;
    font-weight: 700;
    font-family: 'Space Mono', monospace;
}

/* ── Chat Messages ── */
.chat-container {
    padding: 1rem 0;
}

.message-user {
    display: flex;
    justify-content: flex-end;
    margin: 0.75rem 0;
}

.message-assistant {
    display: flex;
    justify-content: flex-start;
    margin: 0.75rem 0;
    gap: 0.75rem;
}

.bubble-user {
    background: linear-gradient(135deg, var(--accent-purple), #4f1db8);
    color: white;
    padding: 0.85rem 1.2rem;
    border-radius: 18px 18px 4px 18px;
    max-width: 70%;
    font-size: 0.9rem;
    line-height: 1.5;
    box-shadow: 0 4px 20px rgba(124,58,237,0.3);
}

.bubble-assistant {
    background: var(--bg-card);
    border: 1px solid var(--border);
    color: var(--text-primary);
    padding: 0.85rem 1.2rem;
    border-radius: 4px 18px 18px 18px;
    max-width: 75%;
    font-size: 0.9rem;
    line-height: 1.6;
    box-shadow: 0 4px 20px rgba(0,0,0,0.3);
}

.avatar {
    width: 36px;
    height: 36px;
    border-radius: 50%;
    background: linear-gradient(135deg, var(--accent-blue), var(--accent-purple));
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 1.1rem;
    flex-shrink: 0;
    box-shadow: 0 0 15px rgba(0,212,255,0.3);
}

.message-meta {
    display: flex;
    align-items: center;
    gap: 0.5rem;
    margin-top: 0.4rem;
    flex-wrap: wrap;
}

.badge {
    font-family: 'Space Mono', monospace;
    font-size: 0.62rem;
    padding: 0.2rem 0.5rem;
    border-radius: 20px;
    letter-spacing: 0.05em;
}

.badge-faith {
    background: rgba(0,255,159,0.1);
    color: var(--accent-green);
    border: 1px solid rgba(0,255,159,0.2);
}

.badge-faith.low {
    background: rgba(255,107,53,0.1);
    color: var(--accent-orange);
    border: 1px solid rgba(255,107,53,0.2);
}

.badge-source {
    background: rgba(0,212,255,0.08);
    color: var(--accent-blue);
    border: 1px solid rgba(0,212,255,0.15);
}

.badge-viz {
    background: rgba(124,58,237,0.1);
    color: #a78bfa;
    border: 1px solid rgba(124,58,237,0.2);
}

/* ── Banners ── */
.banner {
    padding: 0.6rem 1rem;
    border-radius: 8px;
    font-size: 0.82rem;
    font-family: 'Space Mono', monospace;
    margin-bottom: 0.75rem;
    display: flex;
    align-items: center;
    gap: 0.5rem;
}

.banner-confusion {
    background: rgba(255,196,0,0.08);
    border: 1px solid rgba(255,196,0,0.2);
    color: #ffd700;
}

.banner-analogy {
    background: rgba(124,58,237,0.08);
    border: 1px solid rgba(124,58,237,0.2);
    color: #a78bfa;
}

/* ── Viz Panel ── */
.viz-panel {
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 1rem;
    margin-top: 0.75rem;
}

.viz-title {
    font-family: 'Space Mono', monospace;
    font-size: 0.65rem;
    letter-spacing: 0.15em;
    color: var(--accent-purple);
    text-transform: uppercase;
    margin-bottom: 0.5rem;
}

/* ── Input Area ── */
.stChatInput {
    background: var(--bg-secondary) !important;
    border-top: 1px solid var(--border) !important;
}

.stChatInput input {
    background: var(--bg-card) !important;
    color: var(--text-primary) !important;
    border: 1px solid var(--border) !important;
    border-radius: 12px !important;
    font-family: 'Syne', sans-serif !important;
}

.stChatInput input:focus {
    border-color: var(--accent-blue) !important;
    box-shadow: 0 0 0 2px rgba(0,212,255,0.15) !important;
}

/* ── Buttons ── */
.stButton button {
    background: var(--bg-card) !important;
    color: var(--text-primary) !important;
    border: 1px solid var(--border) !important;
    border-radius: 8px !important;
    font-family: 'Space Mono', monospace !important;
    font-size: 0.75rem !important;
    transition: all 0.2s ease !important;
}

.stButton button:hover {
    border-color: var(--accent-blue) !important;
    color: var(--accent-blue) !important;
    box-shadow: 0 0 10px rgba(0,212,255,0.15) !important;
}

/* New session button */
.new-session-btn button {
    background: rgba(255,107,53,0.1) !important;
    border-color: rgba(255,107,53,0.3) !important;
    color: var(--accent-orange) !important;
    width: 100% !important;
}

/* ── Formula Box ── */
.formula-box {
    background: #020812;
    border: 1px solid var(--accent-blue);
    border-left: 3px solid var(--accent-blue);
    border-radius: 8px;
    padding: 0.75rem 1rem;
    font-family: 'Space Mono', monospace;
    font-size: 0.82rem;
    color: var(--accent-blue);
    margin: 0.5rem 0;
    white-space: pre-wrap;
}

/* ── Welcome Screen ── */
.welcome-card {
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: 16px;
    padding: 2rem;
    text-align: center;
    max-width: 600px;
    margin: 2rem auto;
}

.welcome-atom {
    font-size: 4rem;
    margin-bottom: 1rem;
    animation: spin 10s linear infinite;
}

@keyframes spin {
    from { transform: rotate(0deg); }
    to { transform: rotate(360deg); }
}

.welcome-title {
    font-family: 'Syne', sans-serif;
    font-size: 1.6rem;
    font-weight: 800;
    background: linear-gradient(90deg, var(--accent-blue), var(--accent-purple));
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-bottom: 0.5rem;
}

.welcome-text {
    color: var(--text-muted);
    font-size: 0.9rem;
    line-height: 1.6;
    margin-bottom: 1.5rem;
}

.topic-pill {
    display: inline-block;
    background: rgba(0,212,255,0.08);
    border: 1px solid rgba(0,212,255,0.15);
    color: var(--accent-blue);
    padding: 0.3rem 0.7rem;
    border-radius: 20px;
    font-size: 0.75rem;
    font-family: 'Space Mono', monospace;
    margin: 0.2rem;
}

/* ── Scrollbar ── */
::-webkit-scrollbar { width: 4px; }
::-webkit-scrollbar-track { background: var(--bg-primary); }
::-webkit-scrollbar-thumb { background: var(--border); border-radius: 2px; }
::-webkit-scrollbar-thumb:hover { background: var(--accent-blue); }

/* ── Hide Streamlit defaults ── */
#MainMenu, footer { visibility: hidden; }
.block-container { padding-top: 0 !important; }
</style>
""", unsafe_allow_html=True)


# ── Constants ──────────────────────────────────────────────
PLANCK_H      = 6.626e-34
PLANCK_HBAR   = 1.055e-34
SPEED_LIGHT   = 3.0e8
ELECTRON_MASS = 9.109e-31
ELECTRON_VOLT = 1.602e-19
FAITHFULNESS_THRESHOLD = 0.7
MAX_EVAL_RETRIES = 2

ALL_TOPICS = [
    "What is Quantum Physics and Why It Exists",
    "History and Timeline of Quantum Physics",
    "Key Scientists of Quantum Physics",
    "Quantum Physics in Modern Technology",
    "Blackbody Radiation and the Ultraviolet Catastrophe",
    "The Photoelectric Effect",
    "Wave Particle Duality — Concept and Experiments",
    "Wave Particle Duality — de Broglie Hypothesis and Maths",
    "Bohr Model of the Atom — Concept and Revolution",
    "Bohr Model — Energy Formula and Calculations",
    "Schrödinger Equation — What It Means Simply",
    "Schrödinger Equation — Time Dependent and Independent Forms",
    "Wave Function and Probability — What ψ Really Means",
    "Wave Function — Normalisation and Common Misconceptions",
    "Heisenberg Uncertainty Principle — Intuition and Meaning",
    "Heisenberg Uncertainty Principle — Formula and Calculations",
    "Quantum Superposition — What It Really Means",
    "Quantum Superposition — Measurement, Collapse and Interpretations",
    "Quantum Tunnelling — Concept and Intuition",
    "Quantum Tunnelling — Applications and Real World Impact",
]

VIZ_MAP = {
    "What is Quantum Physics and Why It Exists":                    "intro_quantum.html",
    "History and Timeline of Quantum Physics":                      "timeline.html",
    "Key Scientists of Quantum Physics":                            "scientists.html",
    "Quantum Physics in Modern Technology":                         "technology.html",
    "Blackbody Radiation and the Ultraviolet Catastrophe":          "blackbody.html",
    "The Photoelectric Effect":                                     "photoelectric.html",
    "Wave Particle Duality — Concept and Experiments":              "double_slit.html",
    "Wave Particle Duality — de Broglie Hypothesis and Maths":      "de_broglie.html",
    "Bohr Model of the Atom — Concept and Revolution":              "bohr_model.html",
    "Bohr Model — Energy Formula and Calculations":                 "bohr_formula.html",
    "Schrödinger Equation — What It Means Simply":                  "schrodinger_simple.html",
    "Schrödinger Equation — Time Dependent and Independent Forms":  "schrodinger_forms.html",
    "Wave Function and Probability — What ψ Really Means":          "wave_function.html",
    "Wave Function — Normalisation and Common Misconceptions":      "wave_function_norm.html",
    "Heisenberg Uncertainty Principle — Intuition and Meaning":     "uncertainty_intuition.html",
    "Heisenberg Uncertainty Principle — Formula and Calculations":  "uncertainty_formula.html",
    "Quantum Superposition — What It Really Means":                 "superposition.html",
    "Quantum Superposition — Measurement, Collapse and Interpretations": "superposition_collapse.html",
    "Quantum Tunnelling — Concept and Intuition":                   "tunnelling_concept.html",
    "Quantum Tunnelling — Applications and Real World Impact":      "tunnelling_apps.html",
}


# ── Load Agent (cached) ────────────────────────────────────
@st.cache_resource
def load_agent():
    llm      = ChatGroq(model="llama-3.3-70b-versatile", temperature=0.5)
    embedder = SentenceTransformer("all-MiniLM-L6-v2")

    client = chromadb.Client()
    try:
        client.delete_collection("quantum_kb")
    except:
        pass
    collection = client.create_collection("quantum_kb")

    DOCUMENTS = [
        {"id": "doc_001", "topic": "What is Quantum Physics and Why It Exists",
         "text": """Quantum physics is the study of matter and energy at the most fundamental level. It aims to uncover the properties and behaviors of the very building blocks of nature. While many quantum experiments examine very small objects such as electrons and photons, quantum phenomena are all around us acting on every scale. The field of quantum physics arose in the late 1800s and early 1900s from a series of experimental observations of atoms that did not make intuitive sense in the context of classical physics. Before this, scientists believed physics was nearly complete. Newton's laws explained motion perfectly, and Maxwell's equations described electricity and light without fault. However, certain experiments produced results that classical physics simply could not explain. Hot glowing objects emitted light in patterns that defied classical prediction. The breakthrough came when scientists realized that matter and energy are not continuous like a flowing river but come in discrete packets called quanta. Without quantum mechanics there would be no transistors, no lasers, no MRI machines, no solar panels, and no modern computers."""},

        {"id": "doc_002", "topic": "History and Timeline of Quantum Physics",
         "text": """The story of quantum physics begins at the turn of the twentieth century. In 1900 Max Planck proposed energy is emitted in discrete units called quanta. In 1905 Albert Einstein proposed light travels as discrete particles called photons to explain the photoelectric effect. In 1913 Niels Bohr applied quantum ideas to the atom proposing electrons orbit in fixed energy levels. In 1923 Arthur Compton demonstrated photons carry momentum. In 1924 Louis de Broglie proposed all matter has wave-like properties. By 1925 Werner Heisenberg along with Max Born developed matrix mechanics. In 1926 Erwin Schrödinger published his wave equation. That same year Max Born proposed the wave function represents probability. In 1927 Heisenberg published the uncertainty principle and Bohr established the Copenhagen interpretation. In 1935 Einstein Podolsky and Rosen introduced the concept of entanglement. In 1957 Hugh Everett proposed the Many-Worlds interpretation. In 1964 John Bell published his famous inequality providing a way to test quantum entanglement experimentally."""},

        {"id": "doc_003", "topic": "Key Scientists of Quantum Physics",
         "text": """Quantum physics was built by a remarkable generation of scientists. Max Planck is the father of quantum theory proposing energy comes in discrete quanta in 1900. Albert Einstein proposed light consists of photons and won the Nobel Prize for the photoelectric effect not relativity. Niels Bohr developed the atomic model in 1913 and the Copenhagen interpretation. Werner Heisenberg developed matrix mechanics in 1925 and the uncertainty principle in 1927. Erwin Schrödinger published his wave equation in 1926. Paul Dirac unified quantum mechanics with special relativity. Max Born provided the probabilistic interpretation of the wave function. Louis de Broglie proposed matter has wave properties. Wolfgang Pauli formulated the exclusion principle. Richard Feynman developed quantum electrodynamics and invented Feynman diagrams. Alain Aspect John Clauser and Anton Zeilinger won the 2022 Nobel Prize for proving quantum entanglement is real."""},

        {"id": "doc_004", "topic": "Quantum Physics in Modern Technology",
         "text": """Quantum physics is the invisible foundation beneath almost every piece of modern technology. Every transistor in every smartphone and computer chip operates based on quantum mechanical principles. Transistors control electron flow through semiconductors using quantum band theory. A modern smartphone contains over ten billion transistors each a working quantum device. MRI scanners exploit the quantum spin property of hydrogen atoms to create detailed images without radiation. Lasers work through stimulated emission where electrons dropping to lower energy levels release photons in synchrony. Solar panels use the photoelectric effect to convert photons to electricity. GPS depends on atomic clocks using quantum energy states accurate to one second in three hundred million years. Quantum computers use qubits exploiting superposition and entanglement. Quantum communications use Quantum Key Distribution for theoretically unhackable encryption."""},

        {"id": "doc_005", "topic": "Blackbody Radiation and the Ultraviolet Catastrophe",
         "text": """A black body absorbs all radiation and emits based purely on temperature. At room temperature objects emit infrared. At 500 degrees Celsius they glow red. As temperature rises color shifts from red to orange to yellow to white to bluish white. Classical physics predicted black bodies should emit infinite energy at short wavelengths — the ultraviolet catastrophe. Max Planck solved this in 1900 by proposing energy is not continuous but comes in discrete packets called quanta. Each quantum has energy E = hf where f is frequency and h is Planck's constant 6.626 × 10⁻³⁴ joule-seconds. This quantisation means high frequency radiation requires large energy quanta which are less likely to be emitted eliminating the ultraviolet catastrophe. Planck thought it was just a mathematical trick but Einstein showed it was physically real."""},

        {"id": "doc_006", "topic": "The Photoelectric Effect",
         "text": """The photoelectric effect is where light shining on metal ejects electrons. Observed by Hertz in 1887. Classical wave theory predicted brighter light gives electrons more energy and higher frequency ejects more electrons. Both predictions were wrong. Experiments showed increasing brightness increases number of electrons but not their energy. Increasing frequency increases electron kinetic energy. Below a threshold frequency no electrons are ejected regardless of brightness. Einstein proposed in 1905 that light travels as discrete photons each with energy E = hf. When a photon hits metal it gives all energy to one electron. If energy exceeds the work function Φ the electron escapes. The photoelectric equation is KE = hf - Φ. Einstein won the Nobel Prize in 1921 for this explanation not for relativity. This was the first direct evidence light has particle properties."""},

        {"id": "doc_007", "topic": "Wave Particle Duality — Concept and Experiments",
         "text": """Wave-particle duality states every quantum entity exhibits both wave and particle behavior depending on how it is observed. The double slit experiment is the clearest demonstration. When electrons are fired at a barrier with two slits an interference pattern appears proving wave behavior. When detectors observe which slit electrons pass through the interference pattern disappears and electrons behave as particles. The act of observation changes the outcome. Light showed particle behavior through the photoelectric effect and wave behavior through interference. Davisson and Germer confirmed in 1927 that electrons produce diffraction patterns like waves and won the Nobel Prize in 1937. Niels Bohr captured this in the complementarity principle — wave and particle aspects are complementary not contradictory. You cannot observe both simultaneously."""},

        {"id": "doc_008", "topic": "Wave Particle Duality — de Broglie Hypothesis and Maths",
         "text": """In 1924 Louis de Broglie proposed all matter has an associated wavelength. The de Broglie wavelength formula is λ = h / p = h / mv where λ is wavelength h is Planck's constant 6.626 × 10⁻³⁴ J·s p is momentum m is mass and v is velocity. An electron moving at 2 × 10⁶ m/s has mass 9.11 × 10⁻³¹ kg giving wavelength approximately 3.6 × 10⁻¹⁰ metres about the size of an atom. A cricket ball of mass 0.16 kg at 30 m/s has wavelength 1.4 × 10⁻³⁴ metres — unmeasurably tiny explaining why everyday objects show no wave behavior. The formula became the foundation for Schrödinger's wave equation. Electron microscopes use electron wavelengths shorter than visible light to image objects smaller than optical microscopes can resolve."""},

        {"id": "doc_009", "topic": "Bohr Model of the Atom — Concept and Revolution",
         "text": """Before Bohr the best atomic model was Rutherford's 1911 planetary model with electrons orbiting the nucleus. Classical physics predicted orbiting electrons should radiate energy and spiral into the nucleus within a fraction of a second. Yet atoms are stable. Bohr solved this in 1913 by declaring electrons only occupy specific allowed orbits called stationary states where they do not radiate. Electrons jump between orbits by absorbing or emitting photons with energy equal to the difference between levels. This explained why hydrogen emits light at specific colors forming spectral lines. Bohr won the Nobel Prize in 1922. However the model fails for multi-electron atoms cannot explain fine spectral structure and treats electrons as particles in fixed orbits which is wrong. Modern quantum mechanics shows electrons exist as probability clouds called orbitals."""},

        {"id": "doc_010", "topic": "Bohr Model — Energy Formula and Calculations",
         "text": """The Bohr model energy formula for hydrogen is Eₙ = -13.6 / n² eV where n is the principal quantum number any positive integer. The ground state n=1 has energy -13.6 eV. The first excited state n=2 has energy -3.4 eV. The second excited state n=3 has energy -1.51 eV. The negative sign means the electron is bound. When an electron jumps between levels the photon energy is ΔE = E_upper - E_lower = hf. For n=3 to n=2 the energy difference is 1.89 eV corresponding to red light at 656 nanometres exactly matching hydrogen's brightest spectral line. Bohr also quantized angular momentum L = nℏ where ℏ is h/2π. The allowed orbits are where an integer number of electron wavelengths fit around the circumference forming standing waves."""},

        {"id": "doc_011", "topic": "Schrödinger Equation — What It Means Simply",
         "text": """The Schrödinger equation is the quantum counterpart of Newton's second law. Just as F=ma governs particle motion the Schrödinger equation governs the evolution of the quantum wave function ψ over time. Erwin Schrödinger published it in 1926 earning the Nobel Prize in 1933. The wave function contains all information about a quantum system but tells us probability of finding particles not exact positions. The equation was inspired by de Broglie's matter waves. When applied to hydrogen it correctly predicts all energy levels Bohr derived but from rigorous mathematics. Max Born provided the physical meaning in 1926 — |ψ|² gives the probability density of finding the particle at each location. The equation completely replaced the Bohr model describing electrons as three-dimensional probability clouds called orbitals with spherical s dumbbell p and complex d and f shapes."""},

        {"id": "doc_012", "topic": "Schrödinger Equation — Time Dependent and Independent Forms",
         "text": """The Schrödinger equation has two forms. The time-dependent form iℏ ∂ψ/∂t = Hψ describes how the wave function changes over time. Here ℏ is Planck's constant divided by 2π i is the imaginary unit and H is the Hamiltonian operator representing total energy. This form applies when a system is changing — a moving particle evolving quantum state or interacting with an external field. The time-independent form Hψ = Eψ applies to stationary states where energy is fixed. This eigenvalue equation gives allowed energy levels of quantum systems. Solving it for hydrogen gives exactly Eₙ = -13.6/n² eV derived from first principles. It applies to particle in a box quantum harmonic oscillator and hydrogen atom problems. The key insight is quantum mechanics is about wave functions and probabilities not definite particle trajectories."""},

        {"id": "doc_013", "topic": "Wave Function and Probability — What ψ Really Means",
         "text": """The wave function ψ is the central object of quantum mechanics containing all information about a quantum system. Unlike classical mechanics where particles have definite positions a quantum particle described by a wave function does not have a definite position before measurement. The Born rule states the probability of finding a particle at a location is proportional to |ψ|² — the square of the wave function magnitude. Where |ψ|² is large the particle is likely found. Where |ψ|² is zero the particle will never be found. The wave function itself can take complex values but |ψ|² is always real and non-negative. The electron in hydrogen exists as a three-dimensional probability cloud — the orbital — not a ball in a fixed orbit. The wave function is not a physical wave like sound but a mathematical object encoding probabilities. Whether it is physically real is debated between interpretations."""},

        {"id": "doc_014", "topic": "Wave Function — Normalisation and Common Misconceptions",
         "text": """Normalisation requires that integrating |ψ|² over all space equals 1 because the particle must exist somewhere. A normalised wave function gives physically meaningful probabilities summing to 100 percent. Wave function collapse occurs on measurement — before measurement the particle exists in superposition across many positions. After measurement it collapses to a definite location. This happens instantaneously regardless of how spread the wave function was. Common misconceptions include thinking the wave function represents ignorance about where the particle really is — this is wrong. Experiments by Alain Aspect proved particles genuinely have no definite properties before measurement. Another misconception is the wave function is a physical wave in space — it is not. A third is collapse takes time — according to Copenhagen it is instantaneous. For two entangled particles the wave function links them so measuring one instantly informs about the other."""},

        {"id": "doc_015", "topic": "Heisenberg Uncertainty Principle — Intuition and Meaning",
         "text": """The Heisenberg uncertainty principle introduced in 1927 states there is a fundamental limit to how precisely certain pairs of properties can be known simultaneously. Position and momentum is the most important pair. The most critical point is what this is NOT — it is not about measurement disturbance or instrument limitations. The uncertainty is fundamental in nature itself. A particle genuinely does not have simultaneously definite position and momentum. The reason comes from matter's wave nature. A precisely located wave must contain many wavelengths meaning many momenta. A single-wavelength wave spreads across all space meaning undefined position. Like sound — a pure note has defined frequency but no moment in time while a sharp click has a defined moment but contains all frequencies. Consequences include atomic stability — electrons cannot spiral into nuclei because confinement would cause enormous momentum uncertainty. Zero-point energy at absolute zero is also a consequence."""},

        {"id": "doc_016", "topic": "Heisenberg Uncertainty Principle — Formula and Calculations",
         "text": """The uncertainty principle formula for position and momentum is Δx · Δp ≥ ℏ/2 where Δx is position uncertainty Δp is momentum uncertainty and ℏ = 1.055 × 10⁻³⁴ J·s. The energy-time form is ΔE · Δt ≥ ℏ/2. For an electron in an atom of size Δx = 1 × 10⁻¹⁰ m the minimum momentum uncertainty is Δp ≥ ℏ / (2 × Δx) = 5.275 × 10⁻²⁵ kg·m/s giving velocity uncertainty of 5.8 × 10⁵ m/s — 0.2 percent of light speed. For a cricket ball of 0.16 kg the velocity uncertainty is only 3.3 × 10⁻²⁴ m/s — completely unmeasurable. This explains why quantum effects matter only at atomic scales. If an electron were confined in a nucleus of size 10⁻¹⁵ m the momentum uncertainty would be 10⁵ times larger causing kinetic energy far exceeding nuclear binding forces proving electrons cannot exist inside nuclei."""},

        {"id": "doc_017", "topic": "Quantum Superposition — What It Really Means",
         "text": """Quantum superposition states a quantum system can exist in multiple possible states simultaneously until measurement. This is not ignorance — the system genuinely exists in all states at once. A classical coin is secretly heads or tails even unobserved. A quantum coin in superposition is genuinely both simultaneously. The double slit experiment demonstrates this — an electron goes through both slits simultaneously as a superposition of both paths creating interference. If detectors observe which slit the electron took superposition is destroyed and interference disappears. Schrödinger created his cat thought experiment in 1935 where a cat is simultaneously alive and dead until observed to show how strange superposition is at larger scales. Superposition is the foundation of quantum computing. A classical bit is 0 or 1. A qubit in superposition is both simultaneously allowing quantum computers to process 2ⁿ states with n qubits."""},

        {"id": "doc_018", "topic": "Quantum Superposition — Measurement, Collapse and Interpretations",
         "text": """When a quantum system in superposition is measured it collapses into one definite state — wave function collapse. The Born rule gives probabilities — a superposition with amplitudes α and β collapses to first state with probability |α|² and second with |β|². The probabilities always sum to 1. Individual measurement outcomes are fundamentally random — quantum mechanics only predicts probabilities never certainties. The Copenhagen interpretation says before measurement systems exist in superposition and measurement causes instantaneous collapse. The wave function is a calculational tool not a physical reality. The Many-Worlds interpretation says collapse never happens — the universe splits into branches for each outcome. Decoherence explains why superposition vanishes in everyday objects — interaction with even one air molecule entangles the system with the environment destroying superposition almost instantly. For a grain of dust decoherence occurs in less than 10⁻³¹ seconds."""},

        {"id": "doc_019", "topic": "Quantum Tunnelling — Concept and Intuition",
         "text": """Quantum tunnelling is where a particle passes through a barrier it classically cannot cross due to insufficient energy. A classical ball without enough energy rolls back from a hill. A quantum particle has a small real probability of appearing on the other side without going over — it tunnels through. This is possible because quantum particles are described by wave functions that penetrate into forbidden regions decaying exponentially. If the barrier is thin enough the wave function emerges on the other side with nonzero value giving finite probability of tunnelling. Three factors determine tunnelling probability — barrier height higher barriers reduce probability, barrier width thicker barriers exponentially reduce probability, and particle mass heavier particles tunnel less. Tunnelling powers the sun — protons tunnel through electrostatic repulsion to fuse. Without tunnelling the sun would not shine. Alpha radioactive decay is also explained by tunnelling."""},

        {"id": "doc_020", "topic": "Quantum Tunnelling — Applications and Real World Impact",
         "text": """The scanning tunnelling microscope uses electrons tunnelling across a 1 nanometre gap to image surfaces atom by atom. Tunnelling current changes by a factor of 10 for every 0.1 nanometre change in distance. Its inventors Gerd Binnig and Heinrich Rohrer won the Nobel Prize in 1986. Flash memory in USB drives smartphones and solid state drives uses controlled tunnelling. Electrons tunnel through thin insulating oxide layers to store or erase data. This happens billions of times per second in your devices. Tunnel diodes are used in high-frequency oscillators and amplifiers. Quantum computing uses tunnelling in certain qubit designs. Transistors below 1 nanometre suffer uncontrolled tunnelling leakage creating short circuits — this is a fundamental limit on miniaturisation. In biology hydrogen atoms tunnel through energy barriers in enzyme reactions. There is evidence tunnelling plays a role in DNA mutation when hydrogen tunnels between base pairs."""},
    ]

    texts      = [d["text"]  for d in DOCUMENTS]
    ids        = [d["id"]    for d in DOCUMENTS]
    embeddings = embedder.encode(texts).tolist()

    collection.add(
        documents=texts,
        embeddings=embeddings,
        ids=ids,
        metadatas=[{"topic": d["topic"]} for d in DOCUMENTS]
    )

    # ── State ──────────────────────────────────────────────
    class QuantumState(TypedDict):
        question:       str
        messages:       List[dict]
        route:          str
        retrieved:      str
        sources:        List[str]
        tool_result:    str
        answer:         str
        faithfulness:   float
        eval_retries:   int
        student_name:   str
        topics_covered: List[str]
        confusion_flag: bool
        analogy_mode:   bool
        viz_type:       str

    # ── Helper Functions ───────────────────────────────────
    def extract_number(text):
        text = (text.replace("×10⁻³⁴","e-34").replace("×10⁻³¹","e-31")
                    .replace("×10⁻¹⁰","e-10").replace("×10⁻¹⁵","e-15")
                    .replace("×10⁶","e6").replace("×10⁸","e8")
                    .replace("×10¹⁴","e14").replace("×10¹⁵","e15"))
        matches = re.findall(r'[-+]?\d+\.?\d*[eE][-+]?\d+|[-+]?\d+\.?\d+', text)
        return float(matches[0]) if matches else None

    def extract_all_numbers(text):
        text = (text.replace("×10⁻³⁴","e-34").replace("×10⁻³¹","e-31")
                    .replace("×10⁻¹⁰","e-10").replace("×10⁻¹⁵","e-15")
                    .replace("×10⁶","e6").replace("×10⁸","e8")
                    .replace("×10¹⁴","e14").replace("×10¹⁵","e15"))
        return [float(m) for m in re.findall(r'[-+]?\d+\.?\d*[eE][-+]?\d+|[-+]?\d+\.?\d+', text)]

    def quantum_calculator(question):
        q = question.lower()
        try:
            if "photon energy" in q or ("energy" in q and "frequency" in q and "photon" in q):
                freq = extract_number(question)
                if freq is None: return "Please provide the frequency value in Hz."
                energy_j = PLANCK_H * freq
                return f"📐 E = hf = 6.626×10⁻³⁴ × {freq:.3e} = {energy_j:.3e} J = {energy_j/ELECTRON_VOLT:.3e} eV"
            elif "energy" in q and "wavelength" in q and "photon" in q:
                wl = extract_number(question)
                if wl is None: return "Please provide wavelength in metres."
                e = (PLANCK_H * SPEED_LIGHT) / wl
                return f"📐 E = hc/λ = {e:.3e} J = {e/ELECTRON_VOLT:.3e} eV"
            elif "de broglie" in q or ("wavelength" in q and any(p in q for p in ["electron","proton","particle"])):
                v = extract_number(question)
                if v is None: return "Please provide velocity in m/s."
                m = 1.673e-27 if "proton" in q else ELECTRON_MASS
                wl = PLANCK_H / (m * v)
                return f"📐 λ = h/mv = 6.626×10⁻³⁴ / ({m:.3e} × {v:.3e}) = {wl:.3e} m"
            elif ("energy level" in q or "hydrogen" in q) and any(f"n={i}" in q.replace(" ","") for i in range(1,11)):
                n = next((i for i in range(1,11) if f"n={i}" in q.replace(" ","")), None)
                if n is None: return "Please specify n."
                e = -13.6 / (n**2)
                return f"📐 Eₙ = -13.6/n² = -13.6/{n}² = {e:.4f} eV = {e*ELECTRON_VOLT:.3e} J"
            elif "transition" in q or ("from n" in q and "to n" in q):
                nums = re.findall(r'n\s*=\s*(\d+)', q)
                if len(nums) < 2: return "Please specify both energy levels e.g. n=3 to n=2"
                n1, n2 = sorted([int(nums[0]), int(nums[1])], reverse=True)
                de = abs(-13.6/n1**2 - (-13.6/n2**2))
                return f"📐 ΔE = |E{n1} - E{n2}| = {de:.4f} eV, λ = {PLANCK_H*SPEED_LIGHT/(de*ELECTRON_VOLT)*1e9:.1f} nm"
            elif "uncertainty" in q or "delta x" in q or "Δx" in question:
                n = extract_number(question)
                if n is None: return "Please provide a value."
                dp = PLANCK_HBAR / (2 * n)
                return f"📐 Δp ≥ ℏ/(2Δx) = {PLANCK_HBAR:.3e}/(2×{n:.3e}) = {dp:.3e} kg·m/s"
            elif "frequency" in q and "wavelength" in q:
                wl = extract_number(question)
                if wl is None: return "Please provide wavelength in metres."
                return f"📐 f = c/λ = 3×10⁸/{wl:.3e} = {SPEED_LIGHT/wl:.3e} Hz"
            elif "angular momentum" in q:
                n = extract_number(question)
                if n is None: return "Please provide n."
                return f"📐 L = nℏ = {int(n)} × 1.055×10⁻³⁴ = {int(n)*PLANCK_HBAR:.3e} J·s"
            else:
                return """I can calculate:\n1. Photon energy E=hf\n2. Photon energy E=hc/λ\n3. de Broglie wavelength λ=h/mv\n4. Hydrogen energy Eₙ=-13.6/n²\n5. Energy transition ΔE\n6. Uncertainty Δx·Δp≥ℏ/2\n7. Frequency f=c/λ\n8. Angular momentum L=nℏ"""
        except Exception as e:
            return f"Calculator error: {str(e)}"

    # ── Nodes ──────────────────────────────────────────────
    def memory_node(state: QuantumState) -> dict:
        question = state["question"].lower()
        msgs     = state.get("messages", [])
        msgs     = msgs + [{"role": "user", "content": state["question"]}]
        if len(msgs) > 6: msgs = msgs[-6:]
        student_name = state.get("student_name", "")
        if "my name is" in question:
            try:
                name = state["question"].split("my name is")[-1].strip().split()[0]
                student_name = name.replace(".", "").replace(",", "").capitalize()
            except: pass
        confusion_flag = any(w in question for w in [
            "confused","confusing","don't understand","dont understand",
            "not clear","unclear","lost","i don't get","i dont get",
            "too hard","simplify","simpler","explain again","unable to understand"])
        analogy_mode = any(w in question for w in [
            "analogy","example","real world","real-world","like what",
            "compare","similar to","relate to","everyday","in simple terms"])
        topics_covered   = state.get("topics_covered", [])
        previous_sources = state.get("sources", [])
        for topic in previous_sources:
            if topic and topic not in topics_covered:
                topics_covered = topics_covered + [topic]
        return {"messages": msgs, "student_name": student_name,
                "confusion_flag": confusion_flag, "analogy_mode": analogy_mode,
                "topics_covered": topics_covered}

    def router_node(state: QuantumState) -> dict:
        question = state["question"]
        messages = state.get("messages", [])
        recent   = "; ".join(f"{m['role']}: {m['content'][:60]}" for m in messages[-3:-1]) or "none"
        prompt = f"""You are a router for a quantum physics study assistant.
- retrieve: quantum concept theory history scientist phenomenon question
- tool: numerical calculation with numbers and units like energy wavelength momentum
- memory_only: follow-up about conversation topics covered or casual chat

Recent: {recent}
Question: {question}

Reply ONE word only: retrieve / memory_only / tool"""
        response = llm.invoke(prompt)
        decision = response.content.strip().lower()
        if "memory" in decision:   decision = "memory_only"
        elif "tool" in decision:   decision = "tool"
        else:                      decision = "retrieve"
        return {"route": decision}

    def retrieval_node(state: QuantumState) -> dict:
        q_emb   = embedder.encode([state["question"]]).tolist()
        results = collection.query(query_embeddings=q_emb, n_results=3)
        chunks  = results["documents"][0]
        topics  = [m["topic"] for m in results["metadatas"][0]]
        context = "\n\n---\n\n".join(f"[{topics[i]}]\n{chunks[i]}" for i in range(len(chunks)))
        return {"retrieved": context, "sources": topics}

    def skip_retrieval_node(state: QuantumState) -> dict:
        return {"retrieved": "", "sources": []}

    def tool_node(state: QuantumState) -> dict:
        result = quantum_calculator(state["question"])
        return {"tool_result": result}

    def viz_selector_node(state: QuantumState) -> dict:
        sources = state.get("sources", [])
        route   = state.get("route", "retrieve")
        viz_type = ""
        if route == "retrieve" and sources:
            for topic in sources:
                if topic in VIZ_MAP:
                    viz_type = VIZ_MAP[topic]
                    break
        return {"viz_type": viz_type}

    def answer_node(state: QuantumState) -> dict:
        question     = state["question"]
        retrieved    = state.get("retrieved", "")
        tool_result  = state.get("tool_result", "")
        messages     = state.get("messages", [])
        eval_retries = state.get("eval_retries", 0)
        confusion    = state.get("confusion_flag", False)
        analogy      = state.get("analogy_mode", False)
        student_name = state.get("student_name", "")
        context_parts = []
        if retrieved:    context_parts.append(f"KNOWLEDGE BASE:\n{retrieved}")
        if tool_result:  context_parts.append(f"CALCULATOR RESULT:\n{tool_result}")
        context = "\n\n".join(context_parts)
        name_part = f" The student's name is {student_name}." if student_name else ""
        confusion_inst = "\nUse very simple language. Short sentences. Step by step. Add an everyday analogy." if confusion else ""
        analogy_inst   = "\nProvide a clear real-world analogy and connect it back to the quantum concept." if analogy else ""
        retry_inst     = "\nIMPORTANT: Use ONLY information explicitly stated in the context." if eval_retries > 0 else ""
        if context:
            system_content = f"""You are Quanta — a friendly patient quantum physics tutor for beginners.{name_part}
STRICT RULE: Answer using ONLY the context below. If not in context say you don't have that information.{confusion_inst}{analogy_inst}{retry_inst}

CONTEXT:
{context}"""
        else:
            system_content = f"""You are Quanta — a friendly quantum physics tutor.{name_part}
Answer based on conversation history. Be brief and conversational."""
        lc_msgs = [SystemMessage(content=system_content)]
        for msg in messages[:-1]:
            lc_msgs.append(HumanMessage(content=msg["content"]) if msg["role"]=="user" else AIMessage(content=msg["content"]))
        lc_msgs.append(HumanMessage(content=question))
        response = llm.invoke(lc_msgs)
        return {"answer": response.content}

    def eval_node(state: QuantumState) -> dict:
        answer  = state.get("answer", "")
        context = state.get("retrieved", "")[:500]
        retries = state.get("eval_retries", 0)
        if not context:
            return {"faithfulness": 1.0, "eval_retries": 0}
        if retries >= MAX_EVAL_RETRIES:
            return {"faithfulness": 1.0, "eval_retries": retries}
        prompt = f"""Does this answer use ONLY information from the context?
Reply with a single number: 1.0 if yes, 0.7 if mostly, 0.5 if partially, 0.0 if no.

Context: {context}
Answer: {answer[:300]}

Number only:"""
        result = llm.invoke(prompt).content.strip()
        try:
            score = float(result.split()[0].replace(",","."))
            score = max(0.0, min(1.0, score))
        except:
            score = 0.5
        return {"faithfulness": score, "eval_retries": retries + 1}

    def save_node(state: QuantumState) -> dict:
        messages = state.get("messages", [])
        messages = messages + [{"role": "assistant", "content": state["answer"]}]
        return {"messages": messages}

    def route_decision(state: QuantumState) -> str:
        route = state.get("route", "retrieve")
        if route == "tool":        return "tool"
        if route == "memory_only": return "skip"
        return "retrieve"

    def eval_decision(state: QuantumState) -> str:
        score   = state.get("faithfulness", 1.0)
        retries = state.get("eval_retries", 0)
        if retries >= MAX_EVAL_RETRIES: return "save"
        if score >= FAITHFULNESS_THRESHOLD: return "save"
        return "answer"

    # ── Build Graph ────────────────────────────────────────
    graph = StateGraph(QuantumState)
    graph.add_node("memory",   memory_node)
    graph.add_node("router",   router_node)
    graph.add_node("retrieve", retrieval_node)
    graph.add_node("skip",     skip_retrieval_node)
    graph.add_node("tool",     tool_node)
    graph.add_node("viz",      viz_selector_node)
    graph.add_node("answer",   answer_node)
    graph.add_node("eval",     eval_node)
    graph.add_node("save",     save_node)
    graph.set_entry_point("memory")
    graph.add_edge("memory",   "router")
    graph.add_edge("retrieve", "viz")
    graph.add_edge("skip",     "viz")
    graph.add_edge("tool",     "viz")
    graph.add_edge("viz",      "answer")
    graph.add_edge("answer",   "eval")
    graph.add_edge("save",     END)
    graph.add_conditional_edges("router", route_decision,
        {"retrieve": "retrieve", "skip": "skip", "tool": "tool"})
    graph.add_conditional_edges("eval", eval_decision,
        {"answer": "answer", "save": "save"})
    app = graph.compile(checkpointer=MemorySaver())
    return app, embedder, collection, llm


# ── Load ───────────────────────────────────────────────────
try:
    app, embedder, collection, llm = load_agent()
    agent_loaded = True
except Exception as e:
    agent_loaded = False
    st.error(f"Failed to load agent: {e}")
    st.stop()


# ── Session State ──────────────────────────────────────────
if "messages"       not in st.session_state: st.session_state.messages       = []
if "thread_id"      not in st.session_state: st.session_state.thread_id      = str(uuid.uuid4())[:8]
if "student_name"   not in st.session_state: st.session_state.student_name   = ""
if "topics_covered" not in st.session_state: st.session_state.topics_covered = []
if "last_faith"     not in st.session_state: st.session_state.last_faith     = None
if "last_sources"   not in st.session_state: st.session_state.last_sources   = []
if "last_viz"       not in st.session_state: st.session_state.last_viz       = ""
if "last_route"     not in st.session_state: st.session_state.last_route     = ""


# ── Header ─────────────────────────────────────────────────
st.markdown("""
<div class="quanta-header">
    <div class="quanta-logo">⚛️</div>
    <div>
        <div class="quanta-title">QUANTA</div>
        <div class="quanta-subtitle">QUANTUM PHYSICS STUDY BUDDY · AI-POWERED TUTOR</div>
    </div>
</div>
""", unsafe_allow_html=True)


# ── Sidebar ────────────────────────────────────────────────
with st.sidebar:
    st.markdown('<div class="sidebar-section">', unsafe_allow_html=True)
    st.markdown('<div class="sidebar-title">⚡ Session Info</div>', unsafe_allow_html=True)

    name_display = st.session_state.student_name if st.session_state.student_name else "Anonymous"
    thread_short = st.session_state.thread_id[:6]

    st.markdown(f"""
    <div class="stat-row">
        <span class="stat-label">STUDENT</span>
        <span class="stat-value">{name_display}</span>
    </div>
    <div class="stat-row">
        <span class="stat-label">SESSION ID</span>
        <span class="stat-value">#{thread_short}</span>
    </div>
    <div class="stat-row">
        <span class="stat-label">QUESTIONS</span>
        <span class="stat-value">{len(st.session_state.messages) // 2}</span>
    </div>
    <div class="stat-row">
        <span class="stat-label">TOPICS COVERED</span>
        <span class="stat-value">{len(st.session_state.topics_covered)}/20</span>
    </div>
    """, unsafe_allow_html=True)

    if st.session_state.last_faith is not None:
        faith_color = "#00ff9f" if st.session_state.last_faith >= 0.7 else "#ff6b35"
        st.markdown(f"""
        <div class="stat-row">
            <span class="stat-label">LAST FAITHFULNESS</span>
            <span class="stat-value" style="color:{faith_color}">{st.session_state.last_faith:.2f}</span>
        </div>
        """, unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="sidebar-section">', unsafe_allow_html=True)
    st.markdown('<div class="sidebar-title">📚 Topic Map</div>', unsafe_allow_html=True)

    for topic in ALL_TOPICS:
        short = topic[:35] + "..." if len(topic) > 35 else topic
        covered = topic in st.session_state.topics_covered
        cls = "covered" if covered else ""
        dot_cls = "covered" if covered else ""
        st.markdown(f"""
        <div class="topic-item {cls}">
            <div class="topic-dot {dot_cls}"></div>
            {short}
        </div>
        """, unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="sidebar-section">', unsafe_allow_html=True)
    st.markdown('<div class="sidebar-title">🔢 Formula Sheet</div>', unsafe_allow_html=True)
    st.markdown("""
    <div style="font-family:'Space Mono',monospace;font-size:0.68rem;color:#6b8cae;line-height:2;">
    E = hf &nbsp;&nbsp;&nbsp; (photon energy)<br>
    E = hc/λ &nbsp; (from wavelength)<br>
    λ = h/mv &nbsp; (de Broglie)<br>
    Eₙ = -13.6/n² eV<br>
    KE = hf - Φ &nbsp; (photoelectric)<br>
    Δx·Δp ≥ ℏ/2<br>
    ΔE·Δt ≥ ℏ/2<br>
    f = c/λ<br>
    L = nℏ
    </div>
    """, unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="new-session-btn">', unsafe_allow_html=True)
    if st.button("🗑️ New Conversation"):
        st.session_state.messages       = []
        st.session_state.thread_id      = str(uuid.uuid4())[:8]
        st.session_state.student_name   = ""
        st.session_state.topics_covered = []
        st.session_state.last_faith     = None
        st.session_state.last_sources   = []
        st.session_state.last_viz       = ""
        st.session_state.last_route     = ""
        st.rerun()
    st.markdown('</div>', unsafe_allow_html=True)


# ── Main Chat Area ─────────────────────────────────────────
main_col, viz_col = st.columns([3, 2]) if st.session_state.last_viz else [st.container(), None]

with (main_col if st.session_state.last_viz else st.container()):

    # Welcome screen
    if not st.session_state.messages:
        st.markdown("""
        <div class="welcome-card">
            <div class="welcome-atom">⚛️</div>
            <div class="welcome-title">Welcome to Quanta</div>
            <div class="welcome-text">
                Your personal quantum physics tutor. Ask me anything about quantum mechanics —
                from the basics to mind-bending concepts. I can explain ideas, solve problems,
                and show 3D visualizations.
            </div>
            <div>
                <span class="topic-pill">Wave-Particle Duality</span>
                <span class="topic-pill">Uncertainty Principle</span>
                <span class="topic-pill">Superposition</span>
                <span class="topic-pill">Quantum Tunnelling</span>
                <span class="topic-pill">Schrödinger Equation</span>
                <span class="topic-pill">Bohr Model</span>
                <span class="topic-pill">Photoelectric Effect</span>
                <span class="topic-pill">Calculator 🔢</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

    # Chat history
    for msg in st.session_state.messages:
        if msg["role"] == "user":
            st.markdown(f"""
            <div class="message-user">
                <div class="bubble-user">{msg["content"]}</div>
            </div>
            """, unsafe_allow_html=True)
        else:
            content = msg["content"]
            # Detect formula blocks
            if "📐" in content:
                parts = content.split("📐")
                formatted = parts[0]
                for part in parts[1:]:
                    lines = part.strip().split("\n")
                    formatted += f'<div class="formula-box">📐 {chr(10).join(lines)}</div>'
                content = formatted

            st.markdown(f"""
            <div class="message-assistant">
                <div class="avatar">⚛️</div>
                <div>
                    <div class="bubble-assistant">{content}</div>
                </div>
            </div>
            """, unsafe_allow_html=True)


# ── Viz Panel ──────────────────────────────────────────────
if st.session_state.last_viz and viz_col:
    with viz_col:
        st.markdown(f"""
        <div class="viz-panel">
            <div class="viz-title">🌀 3D Visualization</div>
        </div>
        """, unsafe_allow_html=True)

        viz_path = os.path.join("visualizations", st.session_state.last_viz)
        if os.path.exists(viz_path):
            with open(viz_path, "r") as f:
                html_content = f.read()
            st.components.v1.html(html_content, height=400, scrolling=False)
        else:
            st.markdown(f"""
            <div style="background:#0a1628;border:1px dashed #1a3a5c;border-radius:12px;
                        padding:2rem;text-align:center;color:#6b8cae;font-family:'Space Mono',monospace;
                        font-size:0.75rem;">
                🌀 3D Scene<br><br>
                <span style="color:#00d4ff">{st.session_state.last_viz}</span><br><br>
                Place HTML files in<br>visualizations/ folder
            </div>
            """, unsafe_allow_html=True)


# ── Chat Input ─────────────────────────────────────────────
if prompt := st.chat_input("Ask me anything about quantum physics... or give me a calculation!"):

    # Display user message
    st.markdown(f"""
    <div class="message-user">
        <div class="bubble-user">{prompt}</div>
    </div>
    """, unsafe_allow_html=True)
    st.session_state.messages.append({"role": "user", "content": prompt})

    # Run agent
    with st.spinner(""):
        config = {"configurable": {"thread_id": st.session_state.thread_id}}
        try:
            result = app.invoke(
                {
                    "question":     prompt,
                    "eval_retries": 0,
                    "tool_result":  "",
                    "retrieved":    "",
                    "viz_type":     "",
                    "sources":      [],
                    "faithfulness": 0.0,
                    "answer":       "",
                },
                config=config
            )

            answer         = result.get("answer", "Sorry, I could not generate an answer.")
            faithfulness   = result.get("faithfulness", 0.0)
            sources        = result.get("sources", [])
            viz_type       = result.get("viz_type", "")
            route          = result.get("route", "")
            student_name   = result.get("student_name", "")
            topics_covered = result.get("topics_covered", [])
            confusion_flag = result.get("confusion_flag", False)
            analogy_mode   = result.get("analogy_mode", False)

            # Update session state
            st.session_state.last_faith     = faithfulness
            st.session_state.last_sources   = sources
            st.session_state.last_viz       = viz_type
            st.session_state.last_route     = route
            if student_name:
                st.session_state.student_name = student_name
            for t in topics_covered:
                if t not in st.session_state.topics_covered:
                    st.session_state.topics_covered.append(t)

        except Exception as e:
            answer       = f"I encountered an error: {str(e)}"
            faithfulness = 0.0
            sources      = []
            viz_type     = ""
            route        = ""
            confusion_flag = False
            analogy_mode   = False

    # Show banners
    if confusion_flag:
        st.markdown('<div class="banner banner-confusion">💡 Simplified explanation mode is ON</div>', unsafe_allow_html=True)
    if analogy_mode:
        st.markdown('<div class="banner banner-analogy">🔗 Real-world analogy mode is ON</div>', unsafe_allow_html=True)

    # Display assistant message
    content = answer
    if "📐" in content:
        parts = content.split("📐")
        formatted = parts[0]
        for part in parts[1:]:
            lines = part.strip().split("\n")
            formatted += f'<div class="formula-box">📐 {chr(10).join(lines)}</div>'
        content = formatted

    faith_class = "badge-faith" if faithfulness >= 0.7 else "badge-faith low"
    faith_text  = f"✅ {faithfulness:.2f}" if faithfulness >= 0.7 else f"⚠️ {faithfulness:.2f}"

    sources_html = "".join(
        f'<span class="badge badge-source">📖 {s[:30]}</span>'
        for s in sources[:2]
    ) if sources else ""

    viz_html = f'<span class="badge badge-viz">🌀 {viz_type}</span>' if viz_type else ""

    st.markdown(f"""
    <div class="message-assistant">
        <div class="avatar">⚛️</div>
        <div>
            <div class="bubble-assistant">{content}</div>
            <div class="message-meta">
                <span class="badge {faith_class}">{faith_text}</span>
                {sources_html}
                {viz_html}
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.session_state.messages.append({"role": "assistant", "content": answer})
    st.rerun()
