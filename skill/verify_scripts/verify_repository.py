"""离线核查交付结构、IP封装、冻结比特流及真实证据，不要求GPU或板卡。"""
from pathlib import Path
import json,hashlib,ast,re,sys,subprocess
from xml.etree import ElementTree as ET
R=Path(__file__).resolve().parents[2]
subprocess.run([sys.executable,str(R/'skill/verify_scripts/verify_privacy.py')],cwd=R,check=True)
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def load(p):return json.loads((R/p).read_text(encoding='utf-8'))
manifest=load('file_manifest_sha256.json')
for name,expected_sha in manifest.items():
    path=(R/name).resolve()
    assert path.is_relative_to(R) and path.is_file(),name
    assert sha(path)==expected_sha,name
required_directories=['src/pl/rtl','src/pl/ip','src/pl/constraints','src/ps/driver','src/ps/preprocess','src/ps/utils','sim/testbench','sim/scripts','sim/results','build/scripts','build/reports','board/bitstream','board/scripts','board/results','data/eeg_dataset','data/golden_ref','skill/prompt_workflow','skill/case_templates','skill/verify_scripts','skill/pitfall_list','skill/pynq_common_skill','report/llm_trace','report/figures']
assert all((R/p).is_dir() for p in required_directories),[p for p in required_directories if not (R/p).is_dir()]
required=['README.md','src/pl/rtl/h144_decoder_core.sv','src/pl/constraints/pynq_z2_overlay_leds.xdc','src/pl/ip/h144_accel/component.xml','src/ps/utils/stream_decoder.py','sim/testbench/tb_activity.sv','sim/scripts/simulate_activity.py','build/scripts/build_overlay.tcl','board/bitstream/h144_dma_overlay.bit','board/bitstream/h144_dma_overlay.hwh','board/scripts/board_server.py','board/scripts/ui/index.html','board/results/board_results.json','data/eeg_dataset/README.md','data/golden_ref/checkpoints/fpga_candidate.pt','data/golden_ref/rtl_integer_parameters.npz','skill/pynq_common_skill/SKILL.md','report/design_report.pdf','skill/verify_scripts/verify_privacy.py']
assert all((R/p).is_file() for p in required),[p for p in required if not (R/p).is_file()]
board=load('board/results/board_results.json')
assert board['complete'] and board['passed'] and len(board['rows'])==228
assert all(x['mode']==2 and x['mismatches']==0 and all(x['checks'].values()) for x in board['rows'])
assert sha(R/'board/bitstream/h144_dma_overlay.bit')==board['bitstream_sha256']
assert sha(R/'data/golden_ref/activity_golden_all.npz')==board['golden_data_sha256']
sim=load('sim/results/rtl_simulation.json');assert sim['passed'] and sim['compared_logits']==1248
rank=load('board/results/demo_ranking.json')['records'];assert len(rank)==228 and len({x['record'] for x in rank})==228
assert all(x['rank']==i+1 for i,x in enumerate(rank))
assert rank==sorted(rank,key=lambda x:(x['wer'],x['cer'],x['seconds'],x['record']))
for p in R.rglob('*.py'):
    if any(x in p.parts for x in ['.git','vivado_project','ip_packager','activity_sim']):continue
    ast.parse(p.read_text(encoding='utf-8-sig'),filename=str(p))
ip=R/'src/pl/ip/h144_accel';root=ET.parse(ip/'component.xml').getroot()
for f in root.findall('.//{*}file/{*}name'):
    assert (ip/f.text).is_file(),f.text
for p in (ip/'src').rglob('*.sv'):
    canonical=R/'src/pl/rtl'/p.name
    if canonical.exists():assert sha(p)==sha(canonical),p.name
policy=load('data/golden_ref/frozen_activity_policy.json');assert policy['low_bits']==[8,12,16] and not policy['qat_performed']
summary=load('board/results/software_summary.json');assert summary['records']==228
assert summary['metrics']['2']['char_errors']==1095 and summary['metrics']['2']['word_errors']==821
secret=[]
for p in R.rglob('*'):
    if not p.is_file() or p.suffix not in ['.py','.md','.json','.html','.tcl','.txt','.xml','.xdc','.sv','.v']:continue
    if any(x in p.parts for x in ['.git','vivado_project','ip_packager','activity_sim']):continue
    t=p.read_text(encoding='utf-8',errors='replace')
    if re.search(r'-----BEGIN (?:OPENSSH |RSA |EC )?PRIVATE KEY-----',t):secret.append(str(p.relative_to(R)))
assert not secret,secret
if (R/'board/results/demo_validation_20261103.json').exists():
    demo=load('board/results/demo_validation_20261103.json');assert demo['passed']
    assert all(t['mismatches']==0 and t['error_flags']==0 for t in demo['tests'])
print(json.dumps(dict(passed=True,required_files=len(required),required_directories=len(required_directories),verified_file_hashes=len(manifest),official_sentences=228,board_mismatches=0,core_sim_logits=1248,sorted_demo_records=228,bitstream_sha256=sha(R/'board/bitstream/h144_dma_overlay.bit')),indent=2))
