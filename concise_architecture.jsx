import { useState } from "react";

const C = {
  bg: "#06060c",
  card: "#0e0e18",
  border: "#1e1e30",
  indigo: "#6366f1",
  indigoL: "#a5b4fc",
  violet: "#8b5cf6",
  emerald: "#10b981",
  emeraldD: "#059669",
  amber: "#f59e0b",
  rose: "#f43f5e",
  cyan: "#06b6d4",
  sky: "#38bdf8",
  text: "#e2e8f0",
  muted: "#94a3b8",
  dim: "#475569",
  faint: "#334155",
};

const B = ({ x, y, w, h, label, sub, sub2, color, dashed, thick, radius = 6 }) => (
  <g>
    <rect x={x} y={y} width={w} height={h} rx={radius}
      fill={color + "12"} stroke={color} strokeWidth={thick ? 2.5 : 1.4}
      strokeDasharray={dashed ? "6,3" : "none"} />
    <text x={x + w / 2} y={y + (sub ? (sub2 ? h / 2 - 10 : h / 2 - 4) : h / 2 + 2)}
      textAnchor="middle" fill={C.text} fontSize={11}
      fontWeight={600} fontFamily="'JetBrains Mono', monospace">{label}</text>
    {sub && <text x={x + w / 2} y={y + (sub2 ? h / 2 + 4 : h / 2 + 10)}
      textAnchor="middle" fill={C.muted} fontSize={8.5}
      fontFamily="'JetBrains Mono', monospace">{sub}</text>}
    {sub2 && <text x={x + w / 2} y={y + h / 2 + 16}
      textAnchor="middle" fill={C.dim} fontSize={8}
      fontFamily="'JetBrains Mono', monospace">{sub2}</text>}
  </g>
);

const Arr = ({ d, color = C.dim, label, labelX, labelY, dashed, id }) => (
  <g>
    <defs>
      <marker id={id || `a-${Math.random().toString(36).slice(2,6)}`}
        markerWidth="7" markerHeight="5" refX="7" refY="2.5" orient="auto">
        <polygon points="0 0, 7 2.5, 0 5" fill={color} />
      </marker>
    </defs>
    <path d={d} fill="none" stroke={color} strokeWidth={1.3}
      strokeDasharray={dashed ? "5,3" : "none"}
      markerEnd={`url(#${id || `a-${Math.random().toString(36).slice(2,6)}`})`} />
    {label && <text x={labelX} y={labelY} textAnchor="middle"
      fill={C.muted} fontSize={8} fontFamily="'JetBrains Mono', monospace">{label}</text>}
  </g>
);

const Section = ({ x, y, w, h, title, color }) => (
  <g>
    <rect x={x} y={y} width={w} height={h} rx={10}
      fill="none" stroke={color} strokeWidth={1.8} strokeDasharray="8,4" opacity={0.5} />
    <text x={x + 10} y={y + 14} fill={color} fontSize={10}
      fontWeight={700} fontFamily="'JetBrains Mono', monospace" opacity={0.8}>{title}</text>
  </g>
);

const MathText = ({ x, y, text, color = C.muted, size = 9 }) => (
  <text x={x} y={y} textAnchor="middle" fill={color} fontSize={size}
    fontFamily="'JetBrains Mono', monospace" fontStyle="italic">{text}</text>
);

const Plus = ({ x, y }) => (
  <g>
    <circle cx={x} cy={y} r={8} fill={C.bg} stroke={C.dim} strokeWidth={1} />
    <text x={x} y={y + 3.5} textAnchor="middle" fill={C.text} fontSize={12}
      fontWeight={700} fontFamily="'JetBrains Mono', monospace">+</text>
  </g>
);

export default function ConciseArchitecture() {
  return (
    <div style={{
      background: C.bg, minHeight: "100vh",
      display: "flex", flexDirection: "column", alignItems: "center",
      padding: "24px 12px",
      fontFamily: "'JetBrains Mono', 'Fira Code', monospace",
    }}>
      <h1 style={{ color: C.text, fontSize: 20, fontWeight: 700, letterSpacing: "-0.02em", marginBottom: 2 }}>
        VibClustNet — Concise System Architecture
      </h1>
      <p style={{ color: C.muted, fontSize: 11, marginBottom: 20 }}>
        Detailed module-level data flow with tensor shapes and proper block notation
      </p>

      <svg viewBox="0 0 1100 1380" width="100%" style={{ maxWidth: 1100 }}>

        {/* ═══ SECTION 1: DATA INGESTION ═══ */}
        <Section x={15} y={5} w={340} h={260} title="§1  DATA INGESTION" color={C.cyan} />

        <B x={40} y={28} w={140} h={40} label="Labeled CSV" sub="surface_id ∈ ℤ⁺" color={C.cyan} />
        <B x={195} y={28} w={140} h={40} label="Unlabeled CSV" sub="surface_id = ∅" color={C.dim} />

        <B x={40} y={88} w={295} h={36} label="Window Grouping" sub="groupby(window_id) → (N, 3, T)" color={C.cyan} />

        {/* arrows down */}
        <line x1={110} y1={68} x2={110} y2={88} stroke={C.cyan} strokeWidth={1.2} markerEnd="url(#a-d1)" />
        <line x1={265} y1={68} x2={265} y2={88} stroke={C.dim} strokeWidth={1.2} markerEnd="url(#a-d2)" />
        <defs>
          <marker id="a-d1" markerWidth="7" markerHeight="5" refX="7" refY="2.5" orient="auto"><polygon points="0 0,7 2.5,0 5" fill={C.cyan}/></marker>
          <marker id="a-d2" markerWidth="7" markerHeight="5" refX="7" refY="2.5" orient="auto"><polygon points="0 0,7 2.5,0 5" fill={C.dim}/></marker>
          <marker id="a-ind" markerWidth="7" markerHeight="5" refX="7" refY="2.5" orient="auto"><polygon points="0 0,7 2.5,0 5" fill={C.indigo}/></marker>
          <marker id="a-em" markerWidth="7" markerHeight="5" refX="7" refY="2.5" orient="auto"><polygon points="0 0,7 2.5,0 5" fill={C.emerald}/></marker>
          <marker id="a-amb" markerWidth="7" markerHeight="5" refX="7" refY="2.5" orient="auto"><polygon points="0 0,7 2.5,0 5" fill={C.amber}/></marker>
          <marker id="a-ros" markerWidth="7" markerHeight="5" refX="7" refY="2.5" orient="auto"><polygon points="0 0,7 2.5,0 5" fill={C.rose}/></marker>
          <marker id="a-vio" markerWidth="7" markerHeight="5" refX="7" refY="2.5" orient="auto"><polygon points="0 0,7 2.5,0 5" fill={C.violet}/></marker>
          <marker id="a-sky" markerWidth="7" markerHeight="5" refX="7" refY="2.5" orient="auto"><polygon points="0 0,7 2.5,0 5" fill={C.sky}/></marker>
        </defs>

        <B x={40} y={140} w={295} h={36} label="Z-Normalise" sub="x̂ = (x − μ) / σ   per channel" color={C.cyan} />
        <line x1={187} y1={124} x2={187} y2={140} stroke={C.cyan} strokeWidth={1.2} markerEnd="url(#a-d1)" />

        <B x={40} y={192} w={140} h={40} label="Merge Labels" sub="remap → super-class" color={C.cyan} />
        <B x={195} y={192} w={140} h={40} label="Strat. Split" sub="80 / 20 (labeled)" color={C.cyan} />
        <line x1={110} y1={176} x2={110} y2={192} stroke={C.cyan} strokeWidth={1.2} markerEnd="url(#a-d1)" />
        <line x1={265} y1={176} x2={265} y2={192} stroke={C.cyan} strokeWidth={1.2} markerEnd="url(#a-d1)" />

        <MathText x={187} y={252} text="X_train ∈ ℝ^{N×3×T}  ·  y ∈ {−1, 0..K}" color={C.muted} />

        {/* ═══ SECTION 2: VibClustNet ENCODER ═══ */}
        <Section x={15} y={275} w={720} h={680} title="§2  VibClustNet AUTOENCODER" color={C.indigo} />

        {/* Input split into 3 axes */}
        <B x={40} y={300} w={200} h={36} label="Input  x ∈ ℝ^{B×3×T}" sub="split along axis dim" color={C.indigo} />

        {/* 3 parallel axis branches */}
        {[0, 1, 2].map(i => {
          const bx = 40 + i * 115;
          const labels = ["x_X ∈ ℝ^{B×1×T}", "x_Y ∈ ℝ^{B×1×T}", "x_Z ∈ ℝ^{B×1×T}"];
          const axNames = ["Axis X", "Axis Y", "Axis Z"];
          return (
            <g key={i}>
              <B x={bx} y={350} w={105} h={30} label={axNames[i]} sub="(B,1,T)" color={C.violet} />
              <line x1={bx + 52} y1={336} x2={bx + 52} y2={350} stroke={C.violet} strokeWidth={1} markerEnd="url(#a-vio)" />
            </g>
          );
        })}

        {/* Shared MSTCB x2 */}
        <B x={40} y={395} w={345} h={50} label="MSTCB₁ → MSTCB₂  (shared weights)"
          sub="Conv1d(k=3) ‖ Conv1d(k=7) ‖ Conv1d(k=15) ‖ MaxPool→Conv1d(k=1)"
          sub2="+ BatchNorm + ReLU + Residual(1×1)   →  (B, 96, T) per axis" color={C.indigo} />
        {[0, 1, 2].map(i => (
          <line key={i} x1={40 + i * 115 + 52} y1={380} x2={40 + i * 115 + 52} y2={395}
            stroke={C.indigo} strokeWidth={1} markerEnd="url(#a-ind)" />
        ))}

        {/* CAIM */}
        <B x={40} y={465} w={345} h={55} label="CAIM — Cross-Axis Interaction"
          sub="Q = K = V = [pool(axis_i)]₃   ∈ ℝ^{B×3×96}"
          sub2="MultiHeadAttn(h=4) → LayerNorm → broadcast × axis features" color={C.violet} />
        <line x1={212} y1={445} x2={212} y2={465} stroke={C.violet} strokeWidth={1.2} markerEnd="url(#a-vio)" />
        <MathText x={212} y={458} text="3 × (B, 96, T)" color={C.dim} size={8} />

        {/* Concat */}
        <B x={40} y={535} w={345} h={30} label="Concat axes → (B, 288, T)" color={C.indigo} />
        <line x1={212} y1={520} x2={212} y2={535} stroke={C.indigo} strokeWidth={1.2} markerEnd="url(#a-ind)" />

        {/* FAAG */}
        <B x={40} y={580} w={345} h={55} label="FAAG — Frequency-Aware Attention Gate"
          sub="t_gate: DepthwiseConv1d(k=7) → σ"
          sub2="f_gate: FFT → |·| → MLP → σ   ⇒  x ⊙ t_attn ⊙ f_attn" color={C.violet} />
        <line x1={212} y1={565} x2={212} y2={580} stroke={C.violet} strokeWidth={1.2} markerEnd="url(#a-vio)" />

        {/* MSTCB3 */}
        <B x={40} y={650} w={345} h={40} label="MSTCB₃"
          sub="(B, 288, T) → (B, 96, T)" color={C.indigo} />
        <line x1={212} y1={635} x2={212} y2={650} stroke={C.indigo} strokeWidth={1.2} markerEnd="url(#a-ind)" />

        {/* Fork: temporal features go two ways */}
        <MathText x={212} y={703} text="after₃ ∈ ℝ^{B×96×T}" color={C.indigoL} size={9} />

        {/* Left: Reconstruction head */}
        <B x={40} y={715} w={165} h={55} label="Reconstruction Head"
          sub="Conv1d(96→96→48→3)"
          sub2="→ x̂ ∈ ℝ^{B×3×T}" color={C.rose} />
        <line x1={122} y1={690} x2={122} y2={715} stroke={C.rose} strokeWidth={1.2} markerEnd="url(#a-ros)" />

        {/* Right: Global pool + encoder head */}
        <B x={220} y={715} w={165} h={36} label="Global Avg Pool"
          sub="mean(dim=T) → (B, 96)" color={C.indigo} />
        <line x1={302} y1={690} x2={302} y2={715} stroke={C.indigo} strokeWidth={1.2} markerEnd="url(#a-ind)" />

        <B x={220} y={765} w={165} h={36} label="Linear → z"
          sub="z ∈ ℝ^{B×128}" color={C.emerald} />
        <line x1={302} y1={751} x2={302} y2={765} stroke={C.emerald} strokeWidth={1.2} markerEnd="url(#a-em)" />

        {/* Classification head */}
        <B x={220} y={818} w={165} h={44} label="Classifier (opt)"
          sub="Linear(128→64)→ReLU"
          sub2="→Dropout→Linear(64→K)" color={C.amber} />
        <line x1={302} y1={801} x2={302} y2={818} stroke={C.amber} strokeWidth={1.2} markerEnd="url(#a-amb)" />

        {/* Loss computation */}
        <B x={40} y={800} w={165} h={36} label="ℒ_rec = log-cosh"
          sub="Σ log(cosh(x − x̂)) / n" color={C.rose} />
        <B x={220} y={878} w={165} h={30} label="ℒ_cls = CrossEntropy"
          sub="" color={C.amber} />
        <line x1={302} y1={862} x2={302} y2={878} stroke={C.amber} strokeWidth={1} markerEnd="url(#a-amb)" />

        {/* Total loss */}
        <B x={40} y={878} w={165} h={36} label="ℒ = ℒ_rec + λ·ℒ_cls"
          sub="Adam · lr=1e-3 · clip=5" color={C.rose} thick />

        {/* Feedback arrow */}
        <path d="M 122 914 L 122 940 L 25 940 L 25 310 L 40 310"
          fill="none" stroke={C.rose} strokeWidth={1.2} strokeDasharray="5,3" markerEnd="url(#a-ros)" />
        <MathText x={24} y={630} text="backprop" color={C.rose} size={8} />

        {/* ═══ SECTION 3: POST-PROCESSING (right side) ═══ */}
        <Section x={415} y={275} w={315} h={218} title="§3  POST-PROCESSING" color={C.emerald} />

        <B x={435} y={300} w={270} h={36} label="L2 Normalise" sub="z / ‖z‖₂" color={C.emerald} />

        <B x={435} y={352} w={270} h={50} label="PCA"
          sub="fit on train · 95% variance"
          sub2="128d → ~Kd (data-dependent)" color={C.emerald} />
        <line x1={570} y1={336} x2={570} y2={352} stroke={C.emerald} strokeWidth={1.2} markerEnd="url(#a-em)" />

        <B x={435} y={420} w={130} h={36} label="Train PCA" sub="fit_transform" color={C.emerald} />
        <B x={575} y={420} w={130} h={36} label="Test PCA" sub="transform only" color={C.emerald} />
        <line x1={500} y1={402} x2={500} y2={420} stroke={C.emerald} strokeWidth={1} markerEnd="url(#a-em)" />
        <line x1={640} y1={402} x2={640} y2={420} stroke={C.emerald} strokeWidth={1} markerEnd="url(#a-em)" />

        {/* Connect encoder output to post-processing */}
        <path d="M 385 783 L 420 783 L 420 310 L 435 310"
          fill="none" stroke={C.emerald} strokeWidth={1.3} markerEnd="url(#a-em)" />
        <MathText x={420} y={550} text="z embed" color={C.emeraldD} size={8} />

        {/* ═══ SECTION 4: CLUSTERING ═══ */}
        <Section x={415} y={505} w={315} h={290} title="§4  CLUSTERING" color={C.amber} />

        <B x={435} y={530} w={270} h={36} label="K Selection" sub="silhouette sweep K∈[K*−2, K*+2]" color={C.amber} />

        {/* Clustering methods - compact grid */}
        {[
          ["KMeans", "n_init=20", 435, 582],
          ["Agglomerative", "cosine·avg", 570, 582],
          ["GMM", "n_init=5", 435, 624],
          ["SBScan", "auto-ε kNN", 570, 624],
          ["PSO", "15 particles", 435, 666],
          ["GSA", "gravitational", 570, 666],
          ["RandAssign", "baseline", 435, 708],
          ["RandCentroid", "baseline", 570, 708],
        ].map(([l, s, x, y], i) => (
          <B key={i} x={x} y={y} w={128} h={32} label={l} sub={s}
            color={i < 6 ? C.amber : C.dim} />
        ))}

        <line x1={570} y1={566} x2={570} y2={582} stroke={C.amber} strokeWidth={1} markerEnd="url(#a-amb)" />

        {/* Arrow from PCA to clustering */}
        <line x1={570} y1={456} x2={570} y2={505} stroke={C.amber} strokeWidth={1.2} markerEnd="url(#a-amb)" />

        {/* ═══ SECTION 5: EVALUATION ═══ */}
        <Section x={415} y={810} w={315} h={200} title="§5  EVALUATION" color={C.rose} />

        <B x={435} y={838} w={130} h={80} label="Unsupervised" sub=""
          color={C.rose} />
        <text x={500} y={870} textAnchor="middle" fill={C.dim} fontSize={8}
          fontFamily="'JetBrains Mono', monospace">Silhouette</text>
        <text x={500} y={882} textAnchor="middle" fill={C.dim} fontSize={8}
          fontFamily="'JetBrains Mono', monospace">Davies-Bouldin</text>
        <text x={500} y={894} textAnchor="middle" fill={C.dim} fontSize={8}
          fontFamily="'JetBrains Mono', monospace">Calinski-Harabasz</text>
        <text x={500} y={906} textAnchor="middle" fill={C.dim} fontSize={8}
          fontFamily="'JetBrains Mono', monospace">Dunn Index</text>

        <B x={575} y={838} w={130} h={80} label="Supervised" sub=""
          color={C.sky} />
        <text x={640} y={870} textAnchor="middle" fill={C.dim} fontSize={8}
          fontFamily="'JetBrains Mono', monospace">ARI</text>
        <text x={640} y={882} textAnchor="middle" fill={C.dim} fontSize={8}
          fontFamily="'JetBrains Mono', monospace">NMI</text>
        <text x={640} y={894} textAnchor="middle" fill={C.dim} fontSize={8}
          fontFamily="'JetBrains Mono', monospace">c2s mapping</text>
        <text x={640} y={906} textAnchor="middle" fill={C.dim} fontSize={8}
          fontFamily="'JetBrains Mono', monospace">Train−Test gap</text>

        <B x={435} y={935} w={270} h={36} label="NMI Probe" sub="every 10 epochs during training" color={C.sky} />

        <line x1={570} y1={795} x2={570} y2={810} stroke={C.rose} strokeWidth={1.2} markerEnd="url(#a-ros)" />

        {/* ═══ SECTION 6: VISUALISATION ═══ */}
        <Section x={415} y={1025} w={315} h={175} title="§6  VISUALISATION" color={C.sky} />

        <B x={435} y={1048} w={270} h={36} label="t-SNE" sub="perp=30 · 1000 iter · on PCA embs" color={C.sky} />

        <B x={435} y={1098} w={130} h={36} label="3×3 Grid"
          sub="8 methods + GT" color={C.sky} />
        <B x={575} y={1098} w={130} h={36} label="Individual"
          sub="per-method plots" color={C.sky} />

        <B x={435} y={1148} w={270} h={36} label="VCN Diagnostics"
          sub="Temporal Attn · CAIM heatmap · Recon" color={C.violet} />

        <line x1={570} y1={1084} x2={570} y2={1098} stroke={C.sky} strokeWidth={1} markerEnd="url(#a-sky)" />
        <line x1={500} y1={1134} x2={500} y2={1148} stroke={C.violet} strokeWidth={1} markerEnd="url(#a-vio)" />
        <line x1={640} y1={1134} x2={640} y2={1148} stroke={C.violet} strokeWidth={1} markerEnd="url(#a-vio)" />

        <line x1={570} y1={1010} x2={570} y2={1025} stroke={C.sky} strokeWidth={1.2} markerEnd="url(#a-sky)" />

        {/* ═══ NOTATION KEY ═══ */}
        <rect x={15} y={1215} width={720} height={150} rx={10}
          fill={C.card} stroke={C.border} strokeWidth={1} />
        <text x={35} y={1238} fill={C.text} fontSize={11} fontWeight={700}
          fontFamily="'JetBrains Mono', monospace">NOTATION KEY</text>

        {/* Column 1: Block types */}
        <text x={35} y={1260} fill={C.muted} fontSize={9} fontWeight={600}
          fontFamily="'JetBrains Mono', monospace">Block Types</text>
        {[
          [C.indigo, "Encoder module"],
          [C.violet, "Attention / gate"],
          [C.emerald, "Embedding / PCA"],
          [C.amber, "Clustering"],
          [C.rose, "Loss / evaluation"],
        ].map(([c, l], i) => (
          <g key={i}>
            <rect x={35} y={1268 + i * 18} width={10} height={10} rx={2}
              fill={c + "25"} stroke={c} strokeWidth={1} />
            <text x={52} y={1277 + i * 18} fill={C.muted} fontSize={8}
              fontFamily="'JetBrains Mono', monospace">{l}</text>
          </g>
        ))}

        {/* Column 2: Tensor notation */}
        <text x={200} y={1260} fill={C.muted} fontSize={9} fontWeight={600}
          fontFamily="'JetBrains Mono', monospace">Tensor Shapes</text>
        {[
          "B = batch size",
          "T = window length (samples)",
          "C = channel dim (96 internal)",
          "z ∈ ℝ^128 (embedding)",
          "K = num super-classes",
        ].map((t, i) => (
          <text key={i} x={200} y={1278 + i * 18} fill={C.dim} fontSize={8}
            fontFamily="'JetBrains Mono', monospace">{t}</text>
        ))}

        {/* Column 3: Operators */}
        <text x={420} y={1260} fill={C.muted} fontSize={9} fontWeight={600}
          fontFamily="'JetBrains Mono', monospace">Operations</text>
        {[
          "‖  = concatenation",
          "⊙  = element-wise multiply",
          "σ  = sigmoid activation",
          "→  = data flow (solid)",
          "⇢  = feedback / optional (dashed)",
        ].map((t, i) => (
          <text key={i} x={420} y={1278 + i * 18} fill={C.dim} fontSize={8}
            fontFamily="'JetBrains Mono', monospace">{t}</text>
        ))}

        {/* Column 4: Architecture summary */}
        <text x={610} y={1260} fill={C.muted} fontSize={9} fontWeight={600}
          fontFamily="'JetBrains Mono', monospace">Key Design</text>
        {[
          "Shared-weight axis branches",
          "Temporal + freq dual gating",
          "Recon from pre-pool features",
          "Semi-supervised (ℒ_cls on y≥0)",
          "log-cosh robust recon loss",
        ].map((t, i) => (
          <text key={i} x={610} y={1278 + i * 18} fill={C.dim} fontSize={8}
            fontFamily="'JetBrains Mono', monospace">{t}</text>
        ))}

      </svg>
    </div>
  );
}
