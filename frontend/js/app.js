function openPage(id,btn){
 document.querySelectorAll('.page').forEach(p=>p.classList.remove('active'));
 document.getElementById(id).classList.add('active');
 document.querySelectorAll('.nav button').forEach(b=>b.classList.remove('active'));
 if(btn)btn.classList.add('active');
 const titles={overview:'Overview',investigation:'Consumer Investigation',risk:'Risk Assessment',analytics:'Model & Analytics'};
 document.getElementById('title').textContent=titles[id];
 window.scrollTo(0,0);
}
function investigate(){
 const id=document.getElementById('cid').value||'Unknown';
 document.getElementById('investigationResult').innerHTML='<h2>'+id+'</h2><p><b>Locality:</b> Substation-Zone-4</p><p><b>City:</b> GridCity</p><p><b>Consumer type:</b> Residential</p><p><b>Behavioural anomaly:</b> 0.78</p><p><b>Zero-consumption days:</b> 22</p><p><b>Sudden-drop days:</b> 5</p><p class="risk high"><b>Demo risk:</b> HIGH</p><div class="note">This standalone page is a frontend prototype. Connect it to FastAPI to replace demo values with your real consumer record.</div>';
}
function runRisk(){
 const s=parseFloat(document.getElementById('r_anomaly').value)||0;
 const risk=s>=.8?'CRITICAL':s>=.65?'HIGH':s>=.4?'MEDIUM':'LOW';
 const theft=s>=.65;
 const cls=risk==='HIGH'||risk==='CRITICAL'?'high':risk==='MEDIUM'?'medium':'low';
 document.getElementById('riskResult').innerHTML='<h2 class="risk '+cls+'">'+(theft?'POSSIBLE THEFT':'NORMAL USAGE')+'</h2><p><b>Risk level:</b> '+risk+'</p><p><b>Demo theft probability:</b> '+Math.round(s*100)+'%</p><hr><p><b>Risk factors</b></p><p>• Behavioural anomaly score: '+s+'</p><p>• Zero-consumption days: '+document.getElementById('r_zero').value+'</p><p>• Sudden-drop days: '+document.getElementById('r_drop').value+'</p><div class="note">This is the offline frontend fallback. The real result comes from your FastAPI /predict endpoint when the backend is connected.</div>';
}