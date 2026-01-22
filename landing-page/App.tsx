import React from 'react';
import StickyNav from './components/StickyNav';
import CodeWindow from './components/CodeWindow';
import { ChevronRight, Scale, Eye, Database, BrainCircuit, ShieldCheck, VideoOff, Activity, FileText, Lock, ArrowRight, Github, Download, Terminal } from 'lucide-react';

// Code snippets provided in the prompt
const CODE_EXHIBIT_A = `class Camera:
    """
    Models a pinhole camera with full intrinsic and extrinsic matrices.
    """
    def __init__(self, K, Tw, dist_coeffs):
        self.K = K # Intrinsic matrix (fx, fy, cx, cy)
        self.Tw = Tw # Extrinsic pose matrix (4x4)
        self.dist_coeffs = dist_coeffs # Lens distortion (k1, k2, p1, p2, k3)

    def triangulate(self, p1, p2, P1, P2):
        """
        Uses DLT (Direct Linear Transform) to reconstruct 3D points 
        from 2D correspondences across stereo cameras.
        """
        A = np.zeros((4, 4))
        A[0] = p1[0] * P1[2] - P1[0]
        A[1] = p1[1] * P1[2] - P1[1]
        A[2] = p2[0] * P2[2] - P2[0]
        A[3] = p2[1] * P2[2] - P2[1]
        
        _, _, Vh = np.linalg.svd(A)
        X = Vh[-1]
        return X[:3] / X[3]`;

const CODE_EXHIBIT_B = `def solve_assignment(cost_matrix):
    """
    Binary Integer Programming (BIP) solver for cross-camera association.
    Formulates transitivity constraints (x_ij + x_ik - x_jk <= 1)
    to ensure consistent person tracks across time and cameras.
    """
    prob = pulp.LpProblem("Association", pulp.LpMinimize)
    
    # Binary decision variables: x_ij = 1 if detection i matches detection j
    x = pulp.LpVariable.dicts("match", (indices, indices), cat='Binary')
    
    # Objective: Minimize matching cost (geometric + pose + reid)
    prob += pulp.lpSum([cost_matrix[i,j] * x[i][j] for i,j in edges])
    
    # Transitivity constraints for consistent clustering
    for i, j, k in triplets:
        prob += x[i][j] + x[j][k] - x[i][k] <= 1
        
    prob.solve(pulp.GLPK_CMD(msg=0))
    return x`;

const CODE_EXHIBIT_C = `class LATransformer(nn.Module):
    """
    Locally-Aware Transformer for persistent person re-identification.
    Splits ViT feature maps into 14 local parts, fusing global CLS 
    tokens with local tokens weighted by lambda=8.
    """
    def __init__(self, model, lmbd=8):
        super().__init__()
        self.part = 14 
        self.model = model # ViT Base
        self.lmbd = lmbd

    def forward(self, x):
        x = self.model.forward_features(x) # [Batch, 197, 768]
        cls_token = x[:, 0].unsqueeze(1)
        parts = x[:, 1:].reshape(x.shape[0], self.part, -1)
        
        predict = {}
        for i in range(self.part):
            # Local Attention: Combine global context with local details
            local_feat = (cls_token.squeeze() + self.lmbd * parts[:, i]) / (1 + self.lmbd)
            predict[i] = self.classifiers[i](local_feat)
            
        return predict`;

const CODE_EXHIBIT_D = `def fuse_evidence(vision_event, weight_event):
    """
    Probabilistic fusion of computer vision and weight telemetry.
    Uses temporal IoU and confidence weighting (vision=0.3, weight=0.7).
    """
    # 1. Calculate temporal synchronization
    overlap = min(vision_event.end, weight_event.end) - max(vision_event.start, weight_event.start)
    iou = max(0, overlap) / (vision_event.duration + weight_event.duration - overlap)
    
    # 2. Correlate weight delta with SKU expected weight
    weight_score = exp(-abs(weight_event.delta - vision_event.sku_weight) / sigma)
    
    # 3. Final Evidence Score
    confidence = (0.3 * vision_event.conf + 0.7 * weight_score) * iou
    
    if confidence > ADMISSIBILITY_THRESHOLD:
        return generate_evidence_packet(vision_event, weight_event, confidence)
    return None`;

const REPO_URL = "https://github.com/phamtrung0633/retail-ai";

const App: React.FC = () => {
  return (
    <div className="min-h-screen bg-[#050505] text-neutral-200 font-sans selection:bg-borr-red selection:text-white">
      <StickyNav />

      {/* Hero Section */}
      <section className="relative pt-32 pb-20 md:pt-48 md:pb-32 px-6 overflow-hidden">
        <div className="absolute inset-0 bg-[radial-gradient(circle_at_top_right,_var(--tw-gradient-stops))] from-borr-red/10 via-[#0a0a0a] to-[#050505] pointer-events-none" />
        <div className="absolute top-0 inset-x-0 h-px bg-gradient-to-r from-transparent via-white/10 to-transparent"></div>
        
        <div className="relative z-10 max-w-7xl mx-auto">
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-12 items-center">
                <div className="max-w-4xl">
                    <div className="flex items-center gap-2 mb-8">
                        <span className="h-px w-8 bg-borr-red"></span>
                        <span className="text-borr-red text-xs font-bold tracking-widest uppercase">Winner UQ Validate Innovation Competition</span>
                    </div>
                    
                    <h1 className="font-display text-3xl md:text-5xl lg:text-6xl leading-[1.05] tracking-tight text-white mb-8 uppercase font-extrabold">
                        The Geometry <br />
                        <span className="text-neutral-500">of Evidence.</span>
                    </h1>
                    
                    <p className="text-lg md:text-2xl text-neutral-400 font-light max-w-2xl leading-relaxed mb-12">
                        Borr AI is an open standard for moving retail security from subjective video to <span className="text-white font-medium">mathematical certainty</span>. Fusing stereo vision and weight telemetry into auditable evidence.
                    </p>
                    
                    <div className="flex flex-col sm:flex-row gap-4">
                        <a href={REPO_URL} target="_blank" rel="noopener noreferrer" className="group flex items-center justify-center gap-2 px-8 py-4 bg-white text-black text-sm font-semibold rounded hover:bg-neutral-200 transition-all">
                            <Github size={18} />
                            View Repository
                        </a>
                        <a href={`${REPO_URL}#readme`} target="_blank" rel="noopener noreferrer" className="flex items-center justify-center gap-2 px-8 py-4 border border-neutral-800 text-neutral-300 text-sm font-medium rounded hover:bg-white/5 transition-all">
                            <Terminal size={18} />
                            Read the Docs
                        </a>
                    </div>
                </div>

                <div className="relative group">
                    <div className="absolute -inset-1 bg-gradient-to-r from-borr-red/50 to-purple-600/50 rounded-xl blur opacity-25 group-hover:opacity-50 transition duration-1000 group-hover:duration-200"></div>
                    <div className="relative aspect-video rounded-xl overflow-hidden border border-white/10 bg-black shadow-2xl">
                        <iframe 
                            className="w-full h-full"
                            src="https://www.youtube.com/embed/vcH0PEo4aMo" 
                            title="Borr AI Demonstration" 
                            frameBorder="0" 
                            allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share" 
                            referrerPolicy="strict-origin-when-cross-origin" 
                            allowFullScreen
                        ></iframe>
                    </div>
                </div>
            </div>
        </div>
      </section>

      {/* Stats / Trust Bar */}
      <section className="border-y border-neutral-900 bg-[#080808]">
        <div className="max-w-7xl mx-auto px-6 py-12">
            <div className="grid grid-cols-2 md:grid-cols-4 gap-8">
                <div>
                    <div className="text-4xl font-display font-black text-white mb-1">MIT</div>
                    <div className="text-xs uppercase tracking-wider text-neutral-500">Open License</div>
                </div>
                <div>
                    <div className="text-4xl font-display font-black text-white mb-1">100%</div>
                    <div className="text-xs uppercase tracking-wider text-neutral-500">Self-Hosted</div>
                </div>
                <div>
                    <div className="text-4xl font-display font-black text-white mb-1">Zero</div>
                    <div className="text-xs uppercase tracking-wider text-neutral-500">Telemetry</div>
                </div>
                <div>
                    <div className="text-4xl font-display font-black text-white mb-1">Auditable</div>
                    <div className="text-xs uppercase tracking-wider text-neutral-500">Codebase</div>
                </div>
            </div>
        </div>
      </section>

      {/* Narrative Section - Side by Side */}
      <section className="py-24 px-6 md:px-12" id="platform">
        <div className="max-w-7xl mx-auto grid grid-cols-1 lg:grid-cols-2 gap-20 items-center">
            <div className="relative">
                <div className="absolute -inset-4 bg-gradient-to-r from-borr-red/20 to-purple-900/20 blur-2xl opacity-50 rounded-full" />
                <div className="relative border border-white/10 bg-neutral-900/50 p-8 rounded-xl backdrop-blur-sm">
                   <div className="flex items-center gap-3 mb-6 border-b border-white/5 pb-4">
                        <Activity size={20} className="text-borr-red" />
                        <span className="font-mono text-sm text-neutral-400">LIVE RECONSTRUCTION STREAM</span>
                   </div>
                   <div className="space-y-4 font-mono text-xs">
                        <div className="flex justify-between text-neutral-500">
                            <span>TIMESTAMP</span>
                            <span>EVENT</span>
                            <span>CONFIDENCE</span>
                        </div>
                        <div className="flex justify-between text-white">
                            <span>14:02:22.45</span>
                            <span className="text-borr-red">PROXIMITY_EVENT_START</span>
                            <span>0.89</span>
                        </div>
                        <div className="flex justify-between text-white">
                            <span>14:02:23.10</span>
                            <span className="text-white">HAND_SEGMENTATION_COMPLETE</span>
                            <span>0.98</span>
                        </div>
                        <div className="flex justify-between text-white">
                            <span>14:02:23.12</span>
                            <span className="text-white">WEIGHT_FUSION_SYNC</span>
                            <span>0.99</span>
                        </div>
                        <div className="mt-4 p-3 bg-borr-red/10 border border-borr-red/20 rounded text-borr-red text-center">
                            FORENSIC EVIDENCE GENERATED
                        </div>
                   </div>
                </div>
            </div>
            
            <div>
                <h2 className="text-borr-red font-mono text-sm tracking-widest uppercase mb-4">The Truth Gap</h2>
                <h3 className="font-display text-4xl md:text-5xl text-white mb-8 uppercase font-bold">
                    Democratizing forensic truth.
                </h3>
                <div className="space-y-6 text-neutral-400 font-light leading-relaxed text-lg">
                    <p>
                        For decades, forensic security has been trapped in proprietary "black boxes"—closed systems where the algorithms of guilt are hidden from public audit.
                    </p>
                    <p>
                        <strong className="text-white">Borr AI changes this.</strong> We provide the primitives to mathematically reconstruct an event in real-time, completely open source.
                    </p>
                    <p>
                        By fusing 3D computer vision, weight telemetry, and vector databases on the edge, we enable developers to build systems of "predictive reconstruction" without vendor lock-in.
                    </p>
                </div>
            </div>
        </div>
      </section>

      {/* Feature Grid */}
      <section className="py-24 px-6 md:px-12 bg-[#080808] border-t border-neutral-900">
        <div className="max-w-7xl mx-auto">
            <div className="text-center mb-16">
                <h2 className="font-display text-3xl md:text-4xl text-white mb-4 uppercase font-bold">Don't Trust Pixels. <span className="text-neutral-500">Trust Code.</span></h2>
                <p className="text-neutral-400 max-w-2xl mx-auto">
                    Standard retail AI relies on hidden, proprietary models. Borr exposes the geometry pipeline, allowing full auditability of the logic used to determine intent.
                </p>
            </div>

            <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
                <div className="group p-8 bg-[#0a0a0a] border border-neutral-800 rounded hover:border-borr-red/30 transition-all duration-300">
                    <div className="w-12 h-12 bg-neutral-900 rounded-lg flex items-center justify-center mb-6 text-white group-hover:bg-borr-red group-hover:text-white transition-colors">
                        <Eye size={24} />
                    </div>
                    <h3 className="text-xl font-medium text-white mb-3">Stereo Vision</h3>
                    <p className="text-neutral-400 text-sm leading-relaxed">
                        Dual-camera triangulation using DLT and weighted temporal decay (\(\lambda_t\)). We reconstruct 3D human skeletons in real-time to detect precise shelf interactions.
                    </p>
                </div>

                <div className="group p-8 bg-[#0a0a0a] border border-neutral-800 rounded hover:border-borr-red/30 transition-all duration-300">
                    <div className="w-12 h-12 bg-neutral-900 rounded-lg flex items-center justify-center mb-6 text-white group-hover:bg-borr-red group-hover:text-white transition-colors">
                        <Database size={24} />
                    </div>
                    <h3 className="text-xl font-medium text-white mb-3">Swin-V2 Embeddings</h3>
                    <p className="text-neutral-400 text-sm leading-relaxed">
                        State-of-the-art product recognition using Swin Transformer V2. We extract high-dimensional visual embeddings and query against a Milvus vector database for zero-shot SKU identification.
                    </p>
                </div>

                <div className="group p-8 bg-[#0a0a0a] border border-neutral-800 rounded hover:border-borr-red/30 transition-all duration-300">
                    <div className="w-12 h-12 bg-neutral-900 rounded-lg flex items-center justify-center mb-6 text-white group-hover:bg-borr-red group-hover:text-white transition-colors">
                        <BrainCircuit size={24} />
                    </div>
                    <h3 className="text-xl font-medium text-white mb-3">Transparent Fusion</h3>
                    <p className="text-neutral-400 text-sm leading-relaxed">
                        Inspectable probabilistic truth scores derived from the Intersection over Union (IoU) of vision events and weight telemetry.
                    </p>
                </div>
            </div>
        </div>
      </section>

      {/* Technical Exhibits Section - Dark Mode Commercial */}
      <section className="py-24 px-6 md:px-12" id="technology">
        <div className="max-w-7xl mx-auto">
            <div className="flex flex-col md:flex-row justify-between items-end mb-12 border-b border-neutral-800 pb-8">
                <div className="max-w-2xl">
                    <h2 className="font-display text-3xl md:text-4xl text-white mb-4 uppercase font-bold">Forensic Core Architecture</h2>
                    <p className="text-neutral-400">
                        The mathematical heart of the system. Transparent, auditable, and built for court admissibility.
                    </p>
                </div>
                <div className="flex items-center gap-2 text-sm text-borr-red font-mono mt-4 md:mt-0">
                    <FileText size={16} />
                    <a href={REPO_URL} target="_blank" rel="noopener noreferrer" className="hover:text-white transition-colors">VIEW DOCUMENTATION</a>
                </div>
            </div>

            <div className="space-y-24">
                <div className="grid grid-cols-1 lg:grid-cols-12 gap-12">
                    <div className="lg:col-span-4 space-y-6">
                        <h3 className="text-xl font-display text-white uppercase font-bold">Exhibit A: Stereo Triangulation</h3>
                        <p className="text-sm text-neutral-400 leading-relaxed">
                            Borr uses calibrated stereo camera pairs to reconstruct 3D coordinates from 2D correspondences. By solving the Direct Linear Transform (DLT), we map human keypoints into physical space with millimeter precision, bypassing the perspective ambiguity of single-camera systems.
                        </p>
                        <div className="flex flex-wrap gap-2 text-xs font-mono text-neutral-500">
                            <span className="px-2 py-1 bg-neutral-900 rounded border border-neutral-800">DLT Solver</span>
                            <span className="px-2 py-1 bg-neutral-900 rounded border border-neutral-800">Intrinsic K</span>
                            <span className="px-2 py-1 bg-neutral-900 rounded border border-neutral-800">Extrinsic Tw</span>
                        </div>
                    </div>
                    <div className="lg:col-span-8">
                        <CodeWindow filename="camera.py" code={CODE_EXHIBIT_A} />
                    </div>
                </div>

                <div className="grid grid-cols-1 lg:grid-cols-12 gap-12">
                    <div className="lg:col-span-4 space-y-6 order-last lg:order-first">
                         <div className="p-6 bg-neutral-900/30 border border-neutral-800 rounded-lg">
                            <h3 className="text-lg font-display text-white mb-2 uppercase font-bold">Technical Insight</h3>
                            <p className="text-xs text-neutral-400">
                                "The Arbiter of Truth." By correlating vision-detected proximity events with real-time weight telemetry (0.7 weight factor), Borr generates a combined probability score that meets the highest standards for forensic admissibility.
                            </p>
                         </div>
                         <div className="space-y-4">
                            <h3 className="text-xl font-display text-white uppercase font-bold">Exhibit B: BIP Association</h3>
                            <p className="text-sm text-neutral-400 leading-relaxed">
                                To maintain identity across multiple cameras, we use a Binary Integer Programming (BIP) solver. This ensures transitivity in person matching, preventing the "identity swap" common in traditional trackers.
                            </p>
                         </div>
                    </div>
                     <div className="lg:col-span-8">
                        <div className="mb-4 flex items-center justify-between">
                            <h3 className="text-xl font-display text-white uppercase font-bold">Exhibit C: Probabilistic Fusion</h3>
                        </div>
                        <CodeWindow filename="fusion.py" code={CODE_EXHIBIT_D} />
                    </div>
                </div>
                
                <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
                    <div>
                         <h3 className="text-lg font-display text-white mb-4 uppercase font-bold">BIP Solver (GLPK)</h3>
                         <CodeWindow filename="bip_solver.py" code={CODE_EXHIBIT_B} />
                    </div>
                    <div>
                         <h3 className="text-lg font-display text-white mb-4 uppercase font-bold">Identity Custody (LATransformer)</h3>
                         <CodeWindow filename="transformer.py" code={CODE_EXHIBIT_C} />
                    </div>
                </div>
            </div>
        </div>
      </section>

      {/* Use Cases (Formerly Solutions) */}
      <section className="py-24 px-6 md:px-12 bg-neutral-900" id="solutions">
        <div className="max-w-7xl mx-auto">
            <div className="grid grid-cols-1 md:grid-cols-2 gap-16">
                <div>
                    <div className="inline-flex items-center gap-2 px-3 py-1 rounded-full bg-borr-red/10 border border-borr-red/20 text-borr-red text-xs font-bold tracking-wider mb-8 font-mono">
                        <Scale size={14} />
                        OPEN SOURCE EVIDENCE
                    </div>
                    <h2 className="font-display text-4xl text-white mb-6 uppercase font-bold">Build the future of verifiable truth.</h2>
                    <p className="text-neutral-400 text-lg mb-8 font-light">
                        Borr AI turns video into math. Developers can use our primitives to generate structured, unambiguous evidence for a variety of use cases.
                    </p>
                    
                    <ul className="space-y-4">
                        {[
                            { title: 'Theft & Shrinkage Evidence', desc: 'Generate mathematical proof of loss with vision and weight correlation.' },
                            { title: 'Identity Custody', desc: 'Maintain persistent person tracks across camera blind spots using LATransformers.' },
                            { title: 'Interaction Forensics', desc: 'Isolate hand regions via U-Net segmentation to verify product contact.' }
                        ].map((item, i) => (
                            <li key={i} className="flex gap-4 p-4 border border-white/5 rounded-lg hover:bg-white/5 transition-colors">
                                <div className="mt-1">
                                    <div className="w-2 h-2 bg-borr-red rounded-full shadow-[0_0_10px_rgba(217,4,41,0.5)]"></div>
                                </div>
                                <div>
                                    <h4 className="text-white font-medium text-sm">{item.title}</h4>
                                    <p className="text-neutral-500 text-sm">{item.desc}</p>
                                </div>
                            </li>
                        ))}
                    </ul>
                </div>
                
                <div className="relative">
                    <div className="absolute inset-0 bg-gradient-to-tr from-borr-red/5 to-transparent rounded-2xl"></div>
                    <div className="h-full border border-white/10 rounded-2xl p-8 md:p-12 flex flex-col justify-between bg-black">
                        <div>
                            <div className="flex items-center gap-2 text-neutral-500 mb-8 uppercase tracking-widest text-xs font-bold font-mono">
                                <Lock size={14} /> Data Privacy
                            </div>
                            <h3 className="font-display text-3xl text-white mb-4 uppercase font-bold">Evidence, Not Surveillance.</h3>
                            <p className="text-neutral-400 leading-relaxed mb-8">
                                We don't send footage of a face to a server. We send a JSON packet. We aren't watching people. We are mapping reality.
                            </p>
                        </div>
                        
                        <div className="bg-[#0a0a0a] rounded border border-white/10 p-6 font-mono text-xs">
                             <div className="flex items-center gap-2 text-neutral-500 mb-4 pb-4 border-b border-white/5">
                                <div className="w-2 h-2 rounded-full bg-green-500"></div>
                                Secure Evidence Packet
                             </div>
                             <div className="text-green-500/80 space-y-1">
                                <div className="flex"><span className="w-24 text-purple-400">event_id:</span> <span className="text-white">"evt_8839201"</span></div>
                                <div className="flex"><span className="w-24 text-purple-400">type:</span> <span className="text-white">"EVIDENCE_PACKET"</span></div>
                                <div className="flex"><span className="w-24 text-purple-400">sku_match:</span> <span className="text-white">"SKU_9928_COKE"</span></div>
                                <div className="flex"><span className="w-24 text-purple-400">weight_delta:</span> <span className="text-white">-0.355kg</span></div>
                                <div className="flex"><span className="w-24 text-purple-400">confidence:</span> <span className="text-white">0.9982</span></div>
                                <div className="flex"><span className="w-24 text-purple-400">hash:</span> <span className="text-white">"0x7f8a92b..."</span></div>
                             </div>
                        </div>
                    </div>
                </div>
            </div>
        </div>
      </section>

      {/* Call to Action */}
      <section className="py-24 px-6 border-t border-white/5 relative overflow-hidden">
        <div className="max-w-4xl mx-auto text-center relative z-10">
            <h2 className="font-display text-4xl md:text-5xl text-white mb-6 uppercase font-bold">Ready to inspect the code?</h2>
            <p className="text-neutral-400 mb-10 text-lg">Download the primitives and start building your own forensic tools today.</p>
            <div className="flex flex-col sm:flex-row justify-center gap-4">
                 <a href={`${REPO_URL}/archive/refs/heads/main.zip`} className="px-8 py-3 bg-borr-red text-white font-medium rounded hover:bg-red-700 transition-colors shadow-lg shadow-red-900/20">
                    <Download size={18} className="inline mr-2"/>
                    Download Core
                </a>
                <a href={`${REPO_URL}/pulls`} className="px-8 py-3 bg-transparent border border-white/20 text-white font-medium rounded hover:bg-white/5 transition-colors">
                    <Github size={18} className="inline mr-2"/>
                    Contribute
                </a>
            </div>
        </div>
      </section>

      {/* Minimal Footer */}
      <footer className="py-12 px-6 bg-black border-t border-white/10">
        <div className="max-w-7xl mx-auto flex flex-col md:flex-row justify-between items-center gap-6">
            <div className="flex items-center gap-3">
                <div className="w-6 h-6 bg-white/10 rounded flex items-center justify-center text-white font-display font-bold text-sm">B</div>
                <div className="text-white font-display font-black text-lg uppercase tracking-tighter">Borr<span className="text-neutral-600">AI</span></div>
                <span className="text-neutral-600 text-sm ml-2 hidden sm:inline font-mono uppercase tracking-tight">| Open Source Forensic Computing</span>
            </div>
            
            <div className="flex gap-8 text-sm text-neutral-500">
                <a href={REPO_URL} target="_blank" rel="noopener noreferrer" className="hover:text-white transition-colors">GitHub</a>
                <a href={`${REPO_URL}/blob/main/LICENSE`} target="_blank" rel="noopener noreferrer" className="hover:text-white transition-colors">License</a>
            </div>
            
            <div className="text-neutral-700 text-xs">
                © 2025 Borr AI Open Source Project.
            </div>
        </div>
      </footer>
    </div>
  );
};

export default App;