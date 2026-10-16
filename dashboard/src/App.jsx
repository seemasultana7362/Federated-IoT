import React, { Suspense, useEffect, useMemo, useState } from 'react';
import { Canvas, useFrame } from '@react-three/fiber';
import { Float, Line, OrbitControls, Stars, Text } from '@react-three/drei';
import { Activity, ArrowDownRight, ArrowUpRight, Bell, BookOpen, Braces, Check, ChevronDown, CircleHelp, Cpu, Database, Download, Github, GitBranch, HardDrive, LockKeyhole, Network, Play, Radio, RotateCcw, ShieldCheck, SlidersHorizontal, Square, Terminal, Timer, Waves, Zap } from 'lucide-react';
import { Area, AreaChart, CartesianGrid, ResponsiveContainer, Tooltip, XAxis, YAxis, BarChart, Bar } from 'recharts';

const API = import.meta.env.VITE_API_BASE || 'http://127.0.0.1:8765';
const initialConfig = { clients: 5, rounds: 3, epochs: 3, lr: 0.001, split: 'non_iid', dp: true, epsilon: 2, delta: 0.00001, clip: 1, secagg: true, shares: 5, threshold: 3, adaptive: false };
const blankMetrics = { accuracy: null, precision: null, recall: null, f1: null, loss: null, epsilon: null };
const nav = [{ label: 'Experiment lab', icon: Activity }, { label: 'Topology', icon: Network }, { label: 'Privacy controls', icon: ShieldCheck }, { label: 'Run history', icon: Timer }];

function NodeOrb({ position, color = '#62e8d1', radius = .22, active = false, label }) {
  const ref = React.useRef();
  useFrame((s, dt) => { if (ref.current) { ref.current.rotation.y += dt * .25; ref.current.rotation.x += dt * .12; if(active) ref.current.scale.setScalar(1 + Math.sin(s.clock.elapsedTime*2)*.06); } });
  return <group position={position}>
    <mesh ref={ref}><icosahedronGeometry args={[radius, 2]} /><meshStandardMaterial color={color} emissive={color} emissiveIntensity={active ? 1.4 : .4} metalness={.6} roughness={.24} /></mesh>
    <mesh><sphereGeometry args={[radius*1.45, 20, 20]} /><meshBasicMaterial color={color} wireframe transparent opacity={.22} /></mesh>
    {label && <Text position={[0, -radius-.25, 0]} fontSize={.13} color="#dce7f5" anchorX="center" outlineWidth={0.004} outlineColor="#07101d">{label}</Text>}
    {active && <pointLight color={color} intensity={.6} distance={2.2} />}
  </group>;
}

function NetworkScene({ config, running, stage }) {
  const clients = Math.min(config.clients, 8);
  const nodes = useMemo(() => Array.from({ length: clients }, (_, i) => {
    const a = (i / clients) * Math.PI * 2;
    return [Math.cos(a)*2.45, Math.sin(a)*.55, Math.sin(a)*2.1];
  }), [clients]);
  return <Canvas camera={{ position: [0, 4, 7.3], fov: 45 }} dpr={[1, 1.7]}>
    <color attach="background" args={['#07101d']} />
    <fog attach="fog" args={['#07101d', 8, 18]} />
    <ambientLight intensity={.7} />
    <pointLight position={[0, 3, 0]} color="#63e7d2" intensity={2.1} />
    <Stars radius={70} depth={30} count={1000} factor={2} fade speed={.35} />
    <Float speed={.45} rotationIntensity={.15} floatIntensity={.2}>
      <NodeOrb position={[0, .5, 0]} color={stage >= 4 ? '#b6a3ff' : '#6debd6'} radius={.48} active={running || stage === 6} label="GLOBAL MODEL" />
    </Float>
    {nodes.map((p,i) => <React.Fragment key={i}>
      <Line points={[p, [0,.5,0]]} color={running ? (stage >= 3 ? '#a994ff' : '#57d9c9') : '#244153'} transparent opacity={running ? .82 : .52} lineWidth={running ? 1.4 : .7} />
      <NodeOrb position={p} color={running ? ['#66e8d3','#8da9ff','#e5b5ff'][i%3] : '#4b6a83'} radius={.2} active={running && stage >= 2} label={'EDGE '+String(i+1).padStart(2,'0')} />
    </React.Fragment>)}
    {[1.15,1.7,2.15].map((r,i)=><mesh key={i} rotation={[-Math.PI/2,0,0]} position={[0,-.2-i*.008,0]}><torusGeometry args={[r,.006,6,90]} /><meshBasicMaterial color="#2b6170" transparent opacity={.32} /></mesh>)}
    <OrbitControls enablePan={false} minDistance={4.8} maxDistance={11} autoRotate={!running} autoRotateSpeed={.25} />
  </Canvas>;
}

function MetricCard({ label, value, suffix='', foot, icon: Icon, color='mint' }) {
  return <div className="metric-card">
    <div className="metric-top"><span>{label}</span><span className={'metric-icon '+color}><Icon size={16}/></span></div>
    <div className="metric-value">{value == null ? '—' : value}{value != null ? suffix : ''}</div>
    <div className="metric-foot">{foot}</div>
  </div>;
}

function Field({ label, hint, children }) {
  return <label className="field"><span>{label}{hint && <small>{hint}</small>}</span>{children}</label>;
}

export default function App() {
  const [config, setConfig] = useState(initialConfig);
  const [status, setStatus] = useState('idle');
  const [stage, setStage] = useState(0);
  const [job, setJob] = useState(null);
  const [logs, setLogs] = useState(['[SYSTEM] Federated IoT experiment console ready.', '[INFO] Waiting for local Python API.']);
  const [metrics, setMetrics] = useState(blankMetrics);
  const [history, setHistory] = useState([]);
  const [health, setHealth] = useState({ api: false, repo: true, clients: null, testData: false, message: 'Checking local API…', details: null });
  const [uploadingTest, setUploadingTest] = useState(false);
  const [error, setError] = useState('');
  const [activeNav, setActiveNav] = useState('Experiment lab');
  const [polling, setPolling] = useState(false);
  const update = (key, value) => setConfig(old => {
    const next = {...old, [key]: value};
    if (key === 'clients') {
      next.shares = Math.min(old.shares, Number(value));
      next.threshold = Math.min(old.threshold, next.shares);
    }
    if (key === 'shares') next.threshold = Math.min(old.threshold, Number(value));
    return next;
  });
  const running = status === 'running';
  const progress = stage <= 0 ? 0 : Math.min(100, Math.round(stage/6*100));
  useEffect(() => { checkHealth(); refreshHistory(); }, []);
  useEffect(() => { if(!job || !polling) return; let stopped = false; const timer = setInterval(async () => {
    try {
      const r = await fetch(API+'/api/jobs/'+job);
      const d = await r.json();
      if(stopped) return;
      setLogs(d.logs || []); setStage(d.stage || 0); setStatus(d.status || 'running');
      if(d.metrics) setMetrics(d.metrics);
      if(d.status === 'succeeded' || d.status === 'failed') { setPolling(false); refreshHistory(); }
      if(d.error) setError(d.error);
    } catch(e) { if(!stopped) { setHealth(h=>({...h,api:false,message:'API disconnected'})); setError('Could not reach the local experiment API. Is python dashboard_api.py running?'); setPolling(false); setStatus('failed'); } }
  }, 1200); return () => { stopped = true; clearInterval(timer); }; }, [job, polling]);
  async function checkHealth() {
    try { const r = await fetch(API+'/api/health'); const d = await r.json(); setHealth({api:true, repo:d.repo, clients:d.clients, testData:d.test_data, message:d.message, details:d.details || null}); }
    catch { setHealth({api:false, repo:true, clients:null, testData:false, message:'Local API offline. Start dashboard_api.py.', details:null}); }
  }
  async function uploadTestFile(event) {
    const file = event.target.files?.[0];
    event.target.value = '';
    if (!file) return;
    if (!file.name.toLowerCase().endsWith('.csv')) { setError('Choose a processed CSV file with a label column and 42 numeric features.'); return; }
    setUploadingTest(true); setError('');
    try {
      const response = await fetch(API+'/api/upload-test', {
        method:'POST',
        headers:{'Content-Type':'text/csv','X-Filename':file.name},
        body: await file.arrayBuffer()
      });
      const data = await response.json();
      if (!response.ok) throw new Error(data.error || 'Upload rejected');
      await checkHealth();
      setLogs(old => [...old, '[DATA] '+data.message]);
    } catch(e) { setError(e.message); }
    finally { setUploadingTest(false); }
  }
  async function refreshHistory() {
    try { const r = await fetch(API+'/api/history'); const d = await r.json(); setHistory(d.items || []); } catch {}
  }
  async function runExperiment() {
    setError(''); setMetrics(blankMetrics); setStage(0); setStatus('running'); setPolling(true);
    setLogs(['[UI] Submitting configuration to the local runner…']);
    try {
      const r = await fetch(API+'/api/run', {method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify(config)});
      const d = await r.json();
      if(!r.ok) throw new Error([d.error, ...(d.details || [])].filter(Boolean).join('\n'));
      setJob(d.job_id); setLogs(d.logs || ['[SYSTEM] Job created. Waiting for worker.']);
      if(d.status === 'failed') { setStatus('failed'); setPolling(false); setError(d.error || 'Run failed.'); }
    } catch(e) { setStatus('failed'); setPolling(false); setError(e.message + (health.api ? '' : ' Start the API with: python dashboard_api.py')); }
  }
  function cancelJob() { setPolling(false); setStatus('stopped'); setLogs(old=>[...old,'[UI] Stopped polling; this does not terminate a backend Flower process.']); }
  function downloadJSON() {
    const data = {createdAt:new Date().toISOString(),config,metrics,status,logs,jobId:job};
    const url=URL.createObjectURL(new Blob([JSON.stringify(data,null,2)],{type:'application/json'}));
    const a=document.createElement('a');a.href=url;a.download='federated-iot-experiment.json';a.click();URL.revokeObjectURL(url);
  }
  const chartData = useMemo(() => history.slice(-10).map((h,i)=>({ round: h.round ?? i+1, accuracy: h.accuracy ?? 0, loss: h.loss ?? 0 })), [history]);
  return <div className="app-shell">
    <aside className="sidebar">
      <div className="brand"><div className="brand-symbol"><Network size={19}/></div><div><strong>FEDERATED<span>.LAB</span></strong><small>PRIVACY SYSTEMS</small></div></div>
      <div className="workspace-label">WORKSPACE <span>● LIVE</span></div>
      <nav>{nav.map(n=><button key={n.label} className={'nav-link '+(activeNav===n.label?'selected':'')} onClick={()=>setActiveNav(n.label)}><n.icon size={17}/>{n.label}{n.label==='Experiment lab'&&<i/>}</button>)}</nav>
      <div className="side-spacer"/>
      <div className="repo-card"><div className="repo-top"><Github size={15}/><span>CONNECTED REPOSITORY</span></div><b>Federated-IoT</b><small>seemasultana7362 / main</small><a href="https://github.com/seemasultana7362/Federated-IoT" target="_blank" rel="noreferrer">View source <ArrowUpRight size={13}/></a></div>
      <div className="side-footer"><span className={health.api?'dot ok':'dot warn'}/>{health.api?'Local API connected':'Local API offline'}<button aria-label="Refresh API status" onClick={checkHealth}><RotateCcw size={13}/></button></div>
    </aside>
    <main className="main-content">
      <header className="topbar"><div className="breadcrumbs"><span>PROJECTS</span><b>/</b><span>FEDERATED IOT</span><b>/</b><strong>{activeNav.toUpperCase()}</strong></div><div className="top-actions"><span className={'status-chip '+(running?'working':status==='succeeded'?'good':'')}><i/>{running?'EXPERIMENT RUNNING':status==='succeeded'?'RUN COMPLETED':'EXPERIMENT CONSOLE'}</span><button className="icon-button" title="Repository"><a href="https://github.com/seemasultana7362/Federated-IoT" target="_blank" rel="noreferrer"><Github size={16}/></a></button></div></header>
      <section className="hero-row"><div><div className="eyebrow"><span className="eyebrow-line"/> PRIVACY-PRESERVING MACHINE LEARNING / UNSW-NB15</div><h1>Federated <span>IoT</span> Lab</h1><p className="hero-copy">Train across edge clients. Keep raw traffic local. Observe the global model converge in a live 3D topology.</p></div><div className="hero-badge"><ShieldCheck size={17}/><div><b>LOCAL DATA BY DESIGN</b><small>Only model updates are aggregated</small></div></div></section>
      <section className="metric-grid">
        <MetricCard label="GLOBAL ACCURACY" value={metrics.accuracy==null?null:(metrics.accuracy>1?metrics.accuracy.toFixed(2): (metrics.accuracy*100).toFixed(2))} suffix={metrics.accuracy==null?'':'%'} foot="Held-out global evaluation" icon={Activity}/>
        <MetricCard label="GLOBAL F1 SCORE" value={metrics.f1==null?null:(metrics.f1>1?metrics.f1.toFixed(2):(metrics.f1*100).toFixed(2))} suffix={metrics.f1==null?'':'%'} foot="Weighted F1" icon={Waves} color="violet"/>
        <MetricCard label="PRIVACY BUDGET" value={metrics.epsilon==null?(config.dp?'Target '+config.epsilon:'Off'):Number(metrics.epsilon).toFixed(3)} foot={config.dp?'ε · Opacus accountant':'Differential privacy disabled'} icon={LockKeyhole} color="amber"/>
        <MetricCard label="FEDERATION NODES" value={config.clients} foot="Participating edge clients" icon={Network} color="blue"/>
      </section>
      <section className="main-grid">
        <div className="panel visual-panel">
          <div className="panel-head"><div><span className="panel-kicker">SYSTEM TOPOLOGY</span><h2>Federation in space</h2></div><div className="live-indicator"><span className={running?'pulse':''}/>{running?'LIVE UPDATES':'INTERACTIVE 3D'}</div></div>
          <div className="canvas-wrap"><NetworkScene config={config} running={running} stage={stage}/><div className="canvas-overlay top-left"><span className="legend-dot mint"/> CLIENT MODEL <span className="legend-dot violet"/> GLOBAL MODEL</div><div className="canvas-overlay bottom-right"><span>DRAG TO ORBIT</span><span>SCROLL TO ZOOM</span></div><div className="canvas-overlay bottom-left"><span className="coordinate">FL / SECAGG+ / DP-SGD</span></div></div>
          <div className="flow-strip">
            {['LOCAL DATA','DP-SGD','SECAGG+','FEDAVG','GLOBAL EVAL'].map((s,i)=><React.Fragment key={s}><div className={'flow-step '+(stage>i?'done':stage===i&&running?'current':'')}><span>{stage>i?<Check size={12}/>:String(i+1).padStart(2,'0')}</span>{s}</div>{i<4&&<div className="flow-connector"/>}</React.Fragment>)}
          </div>
        </div>
        <div className="panel config-panel">
          <div className="panel-head"><div><span className="panel-kicker">RUN CONFIGURATION</span><h2>Experiment controls</h2></div><SlidersHorizontal size={17} className="muted"/></div>
          <div className="config-scroll">
            <div className="section-label"><Database size={14}/> DATA PARTITION</div>
            <Field label="Client split" hint="Local CSVs"><select value={config.split} onChange={e=>update('split',e.target.value)}><option value="non_iid">Non-IID · heterogeneous</option><option value="iid">IID · balanced</option></select></Field>
            <div className="two-fields"><Field label="Edge clients"><input type="number" min="2" max="5" value={config.clients} onChange={e=>update('clients',Math.max(2,Math.min(5,Number(e.target.value)||2)))}/></Field><Field label="Global rounds"><input type="number" min="1" max="100" value={config.rounds} onChange={e=>update('rounds',Math.max(1,Math.min(100,Number(e.target.value)||1)))}/></Field></div>
            <div className="section-label"><Cpu size={14}/> LOCAL TRAINING</div>
            <div className="two-fields"><Field label="Local epochs"><input type="number" min="1" max="20" value={config.epochs} onChange={e=>update('epochs',Math.max(1,Math.min(20,Number(e.target.value)||1)))}/></Field><Field label="Learning rate"><input type="number" step="0.0001" min="0.00001" max="1" value={config.lr} onChange={e=>update('lr',Number(e.target.value))}/></Field></div>
            <div className="section-label"><ShieldCheck size={14}/> DIFFERENTIAL PRIVACY <span className="section-tag">OPACUS</span></div>
            <div className="toggle-row"><div><b>Enable DP-SGD</b><small>Clip per-sample gradients and add noise</small></div><button className={'toggle '+(config.dp?'on':'')} role="switch" aria-checked={config.dp} onClick={()=>update('dp',!config.dp)}><i/></button></div>
            {config.dp&&<><Field label="Target epsilon (ε)" hint="Lower = stronger privacy"><div className="range-row"><input type="range" min=".5" max="12" step=".5" value={config.epsilon} onChange={e=>update('epsilon',Number(e.target.value))}/><strong>{config.epsilon.toFixed(1)}</strong></div></Field><Field label="Gradient clipping norm"><div className="range-row"><input type="range" min=".1" max="5" step=".1" value={config.clip} onChange={e=>update('clip',Number(e.target.value))}/><strong>{config.clip.toFixed(1)}</strong></div></Field><Field label="Target delta (δ)"><select value={config.delta} onChange={e=>update('delta',Number(e.target.value))}><option value={0.000001}>1e-6</option><option value={0.00001}>1e-5</option><option value={0.0001}>1e-4</option></select></Field></>}
            <div className="section-label"><LockKeyhole size={14}/> SECURE AGGREGATION <span className="section-tag">FLOWER SECAGG+</span></div>
            <div className="toggle-row"><div><b>Secure aggregation</b><small>Mask and aggregate client updates</small></div><button className={'toggle '+(config.secagg?'on':'')} role="switch" aria-checked={config.secagg} onClick={()=>update('secagg',!config.secagg)}><i/></button></div>
            {config.secagg&&<div className="two-fields"><Field label="Shares"><input type="number" min="2" max={config.clients} value={config.shares} onChange={e=>update('shares',Math.max(2,Math.min(config.clients,Number(e.target.value)||2)))}/></Field><Field label="Reconstruction threshold"><input type="number" min="2" max={config.shares} value={config.threshold} onChange={e=>update('threshold',Math.max(2,Math.min(config.shares,Number(e.target.value)||2)))}/></Field></div>}
            <div className="section-label"><Radio size={14}/> COMMUNICATION</div>
            <div className="toggle-row"><div><b>Adaptive communication</b><small>Threshold-based update filtering</small></div><button className={'toggle '+(config.adaptive?'on':'')} role="switch" aria-checked={config.adaptive} onClick={()=>update('adaptive',!config.adaptive)}><i/></button></div>
            {config.adaptive&&<div className="inline-note">UI setting only: the current Flower server app does not pass this option to a communication policy yet.</div>}
          </div>
          <div className="config-footer"><button className="reset-btn" onClick={()=>{setConfig(initialConfig);setError('')}}><RotateCcw size={14}/> Reset</button><button className="run-btn" onClick={runExperiment} disabled={running || config.threshold>config.shares}><Play size={15} fill="currentColor"/>{running?'Running experiment…':'Run experiment'}<ArrowUpRight size={15}/></button></div>
          <div className="panel-footnote"><LockKeyhole size={12}/> Runs execute on this machine using your repository checkout.</div>
        </div>
      </section>
      <section className="lower-grid">
        <div className="panel chart-panel"><div className="panel-head"><div><span className="panel-kicker">MODEL PERFORMANCE</span><h2>Global round trajectory</h2></div><span className="chart-tag">FROM RUN HISTORY</span></div>
          {chartData.length? <div className="chart-box"><ResponsiveContainer width="100%" height="100%"><AreaChart data={chartData} margin={{top:12,right:8,left:-18,bottom:0}}><defs><linearGradient id="accFill" x1="0" y1="0" x2="0" y2="1"><stop offset="0%" stopColor="#62e8d1" stopOpacity={.25}/><stop offset="100%" stopColor="#62e8d1" stopOpacity={0}/></linearGradient></defs><CartesianGrid stroke="#1c2a3a" strokeDasharray="4 5"/><XAxis dataKey="round" stroke="#71839a" tickLine={false} axisLine={false}/><YAxis domain={[0,100]} stroke="#71839a" tickLine={false} axisLine={false}/><Tooltip contentStyle={{background:'#101b2a',border:'1px solid #26374b',borderRadius:8,color:'#e7eef9'}}/><Area type="monotone" dataKey="accuracy" stroke="#62e8d1" fill="url(#accFill)" strokeWidth={2.3}/></AreaChart></ResponsiveContainer></div>:<div className="empty-chart"><Activity size={25}/><b>No experiment results yet</b><span>Run an experiment to populate accuracy and loss history.</span></div>}
        </div>
        <div className="panel telemetry-panel"><div className="panel-head"><div><span className="panel-kicker">RUNNER TELEMETRY</span><h2>Execution stream</h2></div><span className={'status-dot '+(running?'run':'')}/></div>
          <div className="telemetry-info"><div><small>JOB ID</small><b>{job ? job.slice(0,12) : '—'}</b></div><div><small>PIPELINE STAGE</small><b>{['Ready','Preflight','Local fit','Privacy step','Secure aggregation','Global evaluation','Complete'][Math.min(stage,6)]}</b></div><div><small>PROGRESS</small><b>{status==='succeeded'?'100%':status==='failed'?'Stopped':progress+'%'}</b></div></div>
          <div className="progress-track"><span style={{width:(status==='succeeded'?100:progress)+'%'}}/></div>
          <div className="log-window">{logs.slice(-90).map((line,i)=><div className="log-line" key={i}><span>{String(i+1).padStart(2,'0')}</span><code>{line}</code></div>)}</div>
          <div className="telemetry-actions"><button onClick={downloadJSON}><Download size={13}/> Export run JSON</button>{running&&<button onClick={cancelJob}><Square size={12}/> Stop updates</button>}</div>
        </div>
      </section>
      <section className="bottom-strip"><div><GitBranch size={15}/><span>BACKEND</span><b>Flower · PyTorch · Opacus</b></div><div><HardDrive size={15}/><span>DATASET</span><b>UNSW-NB15</b></div><div><ShieldCheck size={15}/><span>PRIVACY</span><b>{config.dp?'DP-SGD enabled':'DP-SGD disabled'}</b></div><div><Zap size={15}/><span>AGGREGATION</span><b>{config.secagg?'Flower SecAgg+':'Flower workflow without SecAgg+'}</b></div></section>
      {!health.api&&<div className="api-warning"><CircleHelp size={16}/><span><b>Local API is not connected.</b> Start <code>python dashboard_api.py</code> from the repository root to run real experiments. The 3D topology is interactive, but no metrics are generated until the backend is connected.</span></div>}
      {error&&<div className="error-banner"><CircleHelp size={16}/><span><b>Experiment issue:</b> {error}</span><button onClick={()=>setError('')}>Dismiss</button></div>}
      {!health.testData&&health.api&&<div className="api-warning"><CircleHelp size={16}/><span><b>Global test data not found.</b> The evaluator expects a processed CSV at <code>dataset/processed/test_processed.csv</code>, with a label and 42 numeric feature columns. Upload that held-out test CSV here.</span><label className="upload-btn"><input type="file" accept=".csv,text/csv" onChange={uploadTestFile} disabled={uploadingTest}/>{uploadingTest?'Uploading…':'Select processed CSV'}</label></div>}
      {health.api&&health.details&&!health.details.ready&&<div className="api-warning"><CircleHelp size={16}/><span><b>Dataset preflight details</b><br/>{[...(health.details.clients||[]).filter(c=>!c.ok).slice(0,3).map(c=>c.message), ...(health.details.test_data?[]:[health.details.test_message])].map((m,i)=><React.Fragment key={i}>{m}<br/></React.Fragment>)}{(health.details.clients||[]).some(c=>!c.ok&&String(c.message).includes('Git LFS pointer'))&&<>Resolve pointer files with <code>git lfs install</code> and <code>git lfs pull</code> in the repository root.</>}</span></div>}
      <footer className="footer"><span>FEDERATED.LAB · RESEARCH CONSOLE</span><span>BUILT AROUND <a href="https://github.com/seemasultana7362/Federated-IoT" target="_blank" rel="noreferrer">YOUR FEDERATED-IOT REPOSITORY <ArrowUpRight size={11}/></a></span></footer>
    </main>
  </div>;
}
