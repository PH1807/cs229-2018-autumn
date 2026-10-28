import subprocess,glob,os,re,sys,json
from PIL import Image
B="/Users/philipp/Coding/_git/cs229-2018-autumn/zusammenfassung_handnotes"
S="/private/tmp/claude-501/-Users-philipp-Coding--git-cs229-2018-autumn/c2559a12-b36a-45a5-a3fe-cf61f8c895bd/scratchpad/fit"
os.makedirs(S,exist_ok=True)
bg=(251,247,236)
def fill(png):
    im=Image.open(png).convert("RGB"); w,h=im.size; px=im.load()
    ft=int(h*(1-13/297)); top=int(h*8/297); last=0
    for y in range(ft-1,0,-1):
        if any(abs(px[x,y][0]-bg[0])+abs(px[x,y][1]-bg[1])+abs(px[x,y][2]-bg[2])>30 for x in range(0,w,2)):
            last=y;break
    return (last-top)/(ft-top)
def trial(page,s):
    open(f"{B}/fit_tmp.tex","w").write("\\input{preamble}\n\\begin{document}\n\\shorthandoff{\"}\n\\hnscale{%.3f}\\input{pages/%s}\n\\end{document}\n"%(s,page))
    r=subprocess.run(["xelatex","-interaction=nonstopmode","-halt-on-error","fit_tmp.tex"],cwd=B,capture_output=True,text=True)
    if r.returncode!=0: return None,None
    info=subprocess.run(["pdfinfo",f"{B}/fit_tmp.pdf"],capture_output=True,text=True).stdout
    n=int(re.search(r"Pages:\s+(\d+)",info).group(1))
    if n!=1: return n,None
    subprocess.run(["pdftoppm","-png","-r","50",f"{B}/fit_tmp.pdf",f"{S}/t"])
    f=glob.glob(S+"/t-*.png")[0]; v=fill(f); os.remove(f)
    return n,v
def fit(pg,lo=0.60,hi=1.20,target=0.955):
    n,v=trial(pg,lo)
    if n!=1: return lo,"OVERFLOW"
    best=lo
    for it in range(6):
        mid=(lo+hi)/2
        n,v=trial(pg,mid)
        if n==1 and v is not None and v<=target: best=mid; lo=mid
        else: hi=mid
    return round(best,3),"ok"
if __name__=="__main__":
    pages=sys.argv[1:] or json.load(open(B+"/pages_order.json"))
    m=open(B+"/main.tex").read(); mk=open(B+"/main_kompakt.tex").read()
    for pg in pages:
        s,st=fit(pg)
        print(pg,s,st,flush=True)
        pat=r"\\hnscale\{[\d.]+\}\\input\{pages/%s\}"%pg; rp=lambda mo:"\\hnscale{%.3f}\\input{pages/%s}"%(s,pg)
        m=re.sub(pat,rp,m); mk=re.sub(pat,rp,mk)
    open(B+"/main.tex","w").write(m); open(B+"/main_kompakt.tex","w").write(mk)
    for f in glob.glob(B+"/fit_tmp.*"): os.remove(f)
