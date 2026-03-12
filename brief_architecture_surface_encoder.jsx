import { useState } from "react";

const COLORS = {
  bg: "#0a0a0f",
  card: "#12121a",
  cardHover: "#1a1a28",
  border: "#2a2a3a",
  accent: "#6366f1",
  accentLight: "#818cf8",
  green: "#10b981",
  amber: "#f59e0b",
  rose: "#f43f5e",
  cyan: "#06b6d4",
  text: "#e2e8f0",
  textMuted: "#94a3b8",
  textDim: "#64748b",
};

const Block = ({ x, y, w, h, label, sublabel, color, icon }) => (
  <g>
    <rect
      x={x} y={y} width={w} height={h} rx={8}
      fill={color + "18"} stroke={color} strokeWidth={1.5}
    />
    <text x={x + w / 2} y={y + (sublabel ? h / 2 - 6 : h / 2 + 1)}
      textAnchor="middle" fill={COLORS.text}
      fontSize={13} fontWeight={600} fontFamily="'JetBrains Mono', monospace">
      {icon && <tspan>{icon} </tspan>}
      {label}
    </text>
    {sublabel && (
      <text x={x + w / 2} y={y + h / 2 + 12}
        textAnchor="middle" fill={COLORS.textMuted}
        fontSize={10} fontFamily="'JetBrains Mono', monospace">
        {sublabel}
      </text>
    )}
  </g>
);

const Arrow = ({ x1, y1, x2, y2, label, color = COLORS.textDim, dashed = false }) => {
  const midX = (x1 + x2) / 2;
  const midY = (y1 + y2) / 2;
  return (
    <g>
      <defs>
        <marker id={`ah-${color.replace('#','')}`} markerWidth="8" markerHeight="6"
          refX="8" refY="3" orient="auto">
          <polygon points="0 0, 8 3, 0 6" fill={color} />
        </marker>
      </defs>
      <line x1={x1} y1={y1} x2={x2} y2={y2}
        stroke={color} strokeWidth={1.5}
        strokeDasharray={dashed ? "6,4" : "none"}
        markerEnd={`url(#ah-${color.replace('#','')})`} />
      {label && (
        <text x={midX} y={midY - 8}
          textAnchor="middle" fill={COLORS.textMuted}
          fontSize={9} fontFamily="'JetBrains Mono', monospace">
          {label}
        </text>
      )}
    </g>
  );
};

export default function BriefArchitecture() {
  return (
    <div style={{
      background: COLORS.bg,
      minHeight: "100vh",
      display: "flex",
      flexDirection: "column",
      alignItems: "center",
      padding: "32px 16px",
      fontFamily: "'JetBrains Mono', 'Fira Code', monospace",
    }}>
      <h1 style={{
        color: COLORS.text,
        fontSize: 22,
        fontWeight: 700,
        letterSpacing: "-0.02em",
        marginBottom: 4,
      }}>
        VibClustNet — Brief System Architecture
      </h1>
      <p style={{ color: COLORS.textMuted, fontSize: 12, marginBottom: 28 }}>
        High-level pipeline: Data → Encoder → Clustering → Evaluation
      </p>

      <svg viewBox="0 0 900 520" width="100%" style={{ maxWidth: 900 }}>
        {/* Stage 1: Input */}
        <Block x={30} y={20} w={180} h={60} label="Raw CSV Windows"
          sublabel="(N, 3, T) · X/Y/Z accel" color={COLORS.cyan} />
        <Block x={30} y={100} w={180} h={50} label="Z-Normalise"
          sublabel="μ=0, σ=1 per ch" color={COLORS.cyan} />
        <Arrow x1={120} y1={80} x2={120} y2={100} color={COLORS.cyan} />

        <Block x={30} y={170} w={180} h={50} label="Stratified Split"
          sublabel="80/20 · labeled only" color={COLORS.cyan} />
        <Arrow x1={120} y1={150} x2={120} y2={170} color={COLORS.cyan} />

        {/* Unlabeled branch */}
        <Block x={30} y={250} w={180} h={50} label="Unlabeled Windows"
          sublabel="y = −1 appended" color={COLORS.textDim} />
        <Arrow x1={120} y1={220} x2={120} y2={250} color={COLORS.textDim} dashed />

        {/* Stage 2: VibClustNet Encoder (center) */}
        <rect x={280} y={20} width={310} height={290} rx={12}
          fill={COLORS.accent + "0a"} stroke={COLORS.accent} strokeWidth={2}
          strokeDasharray="8,4" />
        <text x={435} y={46} textAnchor="middle" fill={COLORS.accentLight}
          fontSize={14} fontWeight={700} fontFamily="'JetBrains Mono', monospace">
          VibClustNet Autoencoder
        </text>

        <Block x={310} y={60} w={250} h={44} label="MSTCB ×2"
          sublabel="Multi-Scale Conv (k=3,7,15)" color={COLORS.accent} />
        <Block x={310} y={118} w={250} h={44} label="CAIM + FAAG"
          sublabel="Cross-Axis Attn · Freq Gate" color={COLORS.accentLight} />
        <Block x={310} y={176} w={250} h={44} label="MSTCB ×1 → Pool → z"
          sublabel="emb ∈ ℝ^128 (L2-norm)" color={COLORS.accent} />
        <Block x={310} y={240} w={120} h={44} label="Rec Head"
          sublabel="log-cosh ℒ_rec" color={COLORS.rose} />
        <Block x={440} y={240} w={120} h={44} label="Cls Head"
          sublabel="CE ℒ_cls (opt)" color={COLORS.amber} />

        <Arrow x1={435} y1={104} x2={435} y2={118} color={COLORS.accent} />
        <Arrow x1={435} y1={162} x2={435} y2={176} color={COLORS.accent} />
        <Arrow x1={370} y1={220} x2={370} y2={240} color={COLORS.rose} />
        <Arrow x1={500} y1={220} x2={500} y2={240} color={COLORS.amber} />

        {/* Arrow from input to encoder */}
        <Arrow x1={210} y1={130} x2={310} y2={130} label="(B, 3, T)" color={COLORS.cyan} />

        {/* Stage 3: Post-processing */}
        <Block x={660} y={60} w={200} h={50} label="PCA"
          sublabel="95% var · dim reduction" color={COLORS.green} />
        <Arrow x1={560} y1={198} x2={660} y2={85} label="z embed" color={COLORS.green} />

        {/* Stage 4: Clustering */}
        <Block x={660} y={140} w={200} h={110} label="Clustering"
          sublabel="" color={COLORS.amber} />
        <text x={760} y={175} textAnchor="middle" fill={COLORS.textMuted}
          fontSize={9} fontFamily="'JetBrains Mono', monospace">
          KMeans · Agg · GMM
        </text>
        <text x={760} y={190} textAnchor="middle" fill={COLORS.textMuted}
          fontSize={9} fontFamily="'JetBrains Mono', monospace">
          SBScan · PSO · GSA
        </text>
        <text x={760} y={205} textAnchor="middle" fill={COLORS.textMuted}
          fontSize={9} fontFamily="'JetBrains Mono', monospace">
          RandomAssign · RandomClust
        </text>
        <text x={760} y={225} textAnchor="middle" fill={COLORS.textDim}
          fontSize={9} fontFamily="'JetBrains Mono', monospace">
          best K via silhouette sweep
        </text>
        <Arrow x1={760} y1={110} x2={760} y2={140} color={COLORS.amber} />

        {/* Stage 5: Evaluation */}
        <Block x={660} y={280} w={200} h={70} label="Evaluation"
          sublabel="" color={COLORS.rose} />
        <text x={760} y={315} textAnchor="middle" fill={COLORS.textMuted}
          fontSize={9} fontFamily="'JetBrains Mono', monospace">
          Sil · DB · CH · Dunn
        </text>
        <text x={760} y={330} textAnchor="middle" fill={COLORS.textMuted}
          fontSize={9} fontFamily="'JetBrains Mono', monospace">
          ARI · NMI (vs GT)
        </text>
        <Arrow x1={760} y1={250} x2={760} y2={280} color={COLORS.rose} />

        {/* Stage 6: Visualization */}
        <Block x={660} y={380} w={200} h={50} label="Visualisation"
          sublabel="t-SNE · 3×3 grid · diag" color={COLORS.cyan} />
        <Arrow x1={760} y1={350} x2={760} y2={380} color={COLORS.cyan} />

        {/* Loss feedback loop */}
        <path d="M 370 284 L 370 320 L 250 320 L 250 130 L 280 130"
          fill="none" stroke={COLORS.rose} strokeWidth={1.2}
          strokeDasharray="5,3" markerEnd={`url(#ah-${COLORS.rose.replace('#','')})`} />
        <text x={250} y={338} textAnchor="middle" fill={COLORS.rose}
          fontSize={9} fontFamily="'JetBrains Mono', monospace">
          ℒ = ℒ_rec + λ·ℒ_cls → backprop
        </text>

        {/* Legend */}
        <rect x={30} y={420} width={560} height={80} rx={8}
          fill={COLORS.card} stroke={COLORS.border} strokeWidth={1} />
        <text x={50} y={445} fill={COLORS.textMuted} fontSize={10}
          fontFamily="'JetBrains Mono', monospace" fontWeight={600}>LEGEND</text>
        {[
          { c: COLORS.cyan, l: "Data / I/O", x: 50 },
          { c: COLORS.accent, l: "Encoder Modules", x: 180 },
          { c: COLORS.rose, l: "Loss / Eval", x: 340 },
          { c: COLORS.amber, l: "Clustering", x: 470 },
        ].map(({ c, l, x }, i) => (
          <g key={i}>
            <rect x={x} y={458} width={12} height={12} rx={2}
              fill={c + "30"} stroke={c} strokeWidth={1} />
            <text x={x + 18} y={469} fill={COLORS.textMuted} fontSize={10}
              fontFamily="'JetBrains Mono', monospace">{l}</text>
          </g>
        ))}
        <text x={50} y={490} fill={COLORS.textDim} fontSize={9}
          fontFamily="'JetBrains Mono', monospace">
          Dashed arrows = optional / feedback paths · Solid arrows = data flow
        </text>
      </svg>
    </div>
  );
}
