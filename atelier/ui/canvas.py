"""Canvas de composition Ideogram 4 (HTML/JS autonome, sans dépendance).

L'utilisateur dessine des boîtes (objet / texte) à la souris. Les coordonnées
sont normalisées 0–1000 dans `window.__atelier_boxes`, lues au moment de
« Construire » via un hook JS — pas de communication serveur compliquée.

NB : Gradio nettoie les <script> dans gr.HTML, donc le JS (CANVAS_JS) est injecté
dans le <head> du Blocks ; seul le balisage (CANVAS_MARKUP) passe par gr.HTML.
"""
from __future__ import annotations

CANVAS_MARKUP = """
<div id="atelier-canvas-wrap" style="border:1px solid #e5e7eb;border-radius:12px;
     padding:12px;background:#fff;">
  <div style="display:flex;gap:8px;flex-wrap:wrap;align-items:center;margin-bottom:8px;">
    <label>Type :
      <select id="atl-type">
        <option value="obj">objet</option>
        <option value="text">texte</option>
      </select>
    </label>
    <input id="atl-desc" placeholder="description de l'élément" style="flex:1;min-width:220px;"/>
    <input id="atl-text" placeholder="texte à afficher (si type=texte)" style="min-width:160px;"/>
    <label>Couleur : <input id="atl-color" type="color" value="#add645"/></label>
    <label>Ratio :
      <select id="atl-ratio">
        <option value="1">1:1</option>
        <option value="1.5">3:2</option>
        <option value="0.667">2:3</option>
        <option value="1.333">4:3</option>
        <option value="0.75">3:4</option>
        <option value="1.777">16:9</option>
        <option value="0.5625">9:16</option>
        <option value="2.4">21:9</option>
      </select>
    </label>
    <button id="atl-clear" type="button">Vider</button>
    <button id="atl-undo" type="button">Annuler la dernière</button>
  </div>
  <div style="font-size:.8rem;color:#6b7280;margin-bottom:6px;">
    Renseignez type/description (et le texte si besoin), puis <b>dessinez la boîte</b>
    en glissant la souris. Recommencez pour chaque élément.
  </div>
  <canvas id="atl-canvas" width="900" height="900"
          style="width:100%;max-width:760px;background:#0e0f14;border-radius:8px;
                 cursor:crosshair;display:block;"></canvas>
  <div id="atl-list" style="font-size:.78rem;color:#374151;margin-top:8px;"></div>
</div>
"""

CANVAS_JS = """
(function(){
  function init(){
    var c = document.getElementById('atl-canvas');
    if(!c || c.dataset.init === '1') return;
    c.dataset.init = '1';
    window.__atelier_boxes = window.__atelier_boxes || [];
    var ctx = c.getContext('2d');
    var drawing=false, sx=0, sy=0, cx=0, cy=0;
    function applyRatio(){
      var r = parseFloat((document.getElementById('atl-ratio')||{}).value || '1');
      c.width = 900; c.height = Math.round(900 / r); redraw();
    }
    function pos(e){
      var rect = c.getBoundingClientRect();
      var x = (e.clientX - rect.left) / rect.width * c.width;
      var y = (e.clientY - rect.top) / rect.height * c.height;
      return [Math.max(0,Math.min(c.width,x)), Math.max(0,Math.min(c.height,y))];
    }
    function redraw(prev){
      ctx.clearRect(0,0,c.width,c.height);
      ctx.fillStyle='#0e0f14'; ctx.fillRect(0,0,c.width,c.height);
      (window.__atelier_boxes||[]).forEach(function(b){
        var x=b.x1/1000*c.width, y=b.y1/1000*c.height;
        var w=(b.x2-b.x1)/1000*c.width, h=(b.y2-b.y1)/1000*c.height;
        ctx.strokeStyle=b.color||'#add645'; ctx.lineWidth=2; ctx.strokeRect(x,y,w,h);
        ctx.globalAlpha=0.12; ctx.fillStyle=(b.color||'#add645'); ctx.fillRect(x,y,w,h);
        ctx.globalAlpha=1; ctx.fillStyle='#fff'; ctx.font='13px sans-serif';
        ctx.fillText((b.type==='text'?'T: '+(b.text||''):(b.desc||'objet')).slice(0,28), x+4, y+15);
      });
      if(prev){ ctx.strokeStyle='#add645'; ctx.setLineDash([5,4]);
        ctx.strokeRect(prev[0],prev[1],prev[2]-prev[0],prev[3]-prev[1]); ctx.setLineDash([]); }
    }
    function renderList(){
      var el=document.getElementById('atl-list'); if(!el) return;
      var bs=window.__atelier_boxes||[];
      el.innerHTML = bs.length ? bs.map(function(b,i){
        return (i+1)+'. '+(b.type==='text'?'TEXTE « '+(b.text||'')+' »':'OBJET')+
               ' — '+(b.desc||'')+' ['+b.x1+','+b.y1+','+b.x2+','+b.y2+']';
      }).join('<br>') : '<i>Aucune boîte. Dessinez sur le canvas.</i>';
    }
    c.addEventListener('mousedown',function(e){drawing=true;var p=pos(e);sx=p[0];sy=p[1];});
    c.addEventListener('mousemove',function(e){ if(!drawing)return; var p=pos(e);cx=p[0];cy=p[1];
      redraw([Math.min(sx,cx),Math.min(sy,cy),Math.max(sx,cx),Math.max(sy,cy)]); });
    function finish(e){
      if(!drawing) return; drawing=false; var p=pos(e); cx=p[0]; cy=p[1];
      var x1=Math.min(sx,cx), y1=Math.min(sy,cy), x2=Math.max(sx,cx), y2=Math.max(sy,cy);
      if(Math.abs(x2-x1)<4 || Math.abs(y2-y1)<4){ redraw(); return; }
      var n=function(v,m){return Math.round(v/m*1000);};
      window.__atelier_boxes.push({
        type:(document.getElementById('atl-type')||{}).value||'obj',
        desc:(document.getElementById('atl-desc')||{}).value||'',
        text:(document.getElementById('atl-text')||{}).value||'',
        color:(document.getElementById('atl-color')||{}).value||'#add645',
        x1:n(x1,c.width), y1:n(y1,c.height), x2:n(x2,c.width), y2:n(y2,c.height)
      });
      redraw(); renderList();
    }
    c.addEventListener('mouseup', finish);
    c.addEventListener('mouseleave', function(e){ if(drawing) finish(e); });
    var clr=document.getElementById('atl-clear');
    if(clr) clr.addEventListener('click',function(){window.__atelier_boxes=[];redraw();renderList();});
    var und=document.getElementById('atl-undo');
    if(und) und.addEventListener('click',function(){(window.__atelier_boxes||[]).pop();redraw();renderList();});
    var rat=document.getElementById('atl-ratio');
    if(rat) rat.addEventListener('change', applyRatio);
    applyRatio(); renderList();
  }
  // Le canvas peut apparaître tardivement (changement d'onglet) : on réessaie.
  setInterval(init, 700);
})();
"""

# Hook JS du bouton « Construire » : place les boîtes dans le 1er input Python.
READ_BOXES_JS = (
    "(boxes, mode, hl, aes, light, med, style, bg, colors) => {"
    " return [JSON.stringify(window.__atelier_boxes||[]), mode, hl, aes, light,"
    " med, style, bg, colors]; }"
)
