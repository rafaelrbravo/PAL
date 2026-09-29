import math
import multiprocessing as mp
from pathlib import Path
import queue
import time


def _Color(color):
    if isinstance(color, int):
        if color < 0 or color > 0xFFFFFFFF: raise ValueError(f"integer color must be between 0x00000000 and 0xFFFFFFFF color:{color}")
        if color <= 0xFFFFFF: return ((color >> 16) & 255, (color >> 8) & 255, color & 255, 255)
        return ((color >> 24) & 255, (color >> 16) & 255, (color >> 8) & 255, color & 255)
    if len(color) not in (3,4): raise ValueError("color must be an RGB/RGBA sequence or packed integer")
    vals=tuple(color)+(255,) if len(color)==3 else tuple(color)
    if any(not isinstance(v,(int,float)) or not math.isfinite(v) or v<0 or v>255 for v in vals): raise ValueError(f"color channels must be finite values from 0 to 255 color:{color}")
    return tuple(int(v) for v in vals)


def _Finite(name,*vals):
    if any(not isinstance(v,(int,float)) or not math.isfinite(v) for v in vals): raise ValueError(f"{name} must be finite")


def _MatMul(a,b):
    return [[sum(a[r][k]*b[k][c] for k in range(4)) for c in range(4)] for r in range(4)]


def _Perspective(fov,aspect,near,far):
    f=1.0/math.tan(math.radians(fov)/2.0)
    return [[f/aspect,0,0,0],[0,f,0,0],[0,0,(far+near)/(near-far),(2*far*near)/(near-far)],[0,0,-1,0]]


def _LookAt(eye,target,up):
    def norm(v):
        n=math.sqrt(sum(x*x for x in v)); return [x/n for x in v]
    def cross(a,b): return [a[1]*b[2]-a[2]*b[1],a[2]*b[0]-a[0]*b[2],a[0]*b[1]-a[1]*b[0]]
    f=norm([target[i]-eye[i] for i in range(3)]); s=norm(cross(f,up)); u=cross(s,f)
    return [[s[0],s[1],s[2],-sum(s[i]*eye[i] for i in range(3))],
            [u[0],u[1],u[2],-sum(u[i]*eye[i] for i in range(3))],
            [-f[0],-f[1],-f[2],sum(f[i]*eye[i] for i in range(3))],[0,0,0,1]]


def _FlattenColumnMajor(m):
    return tuple(m[r][c] for c in range(4) for r in range(4))


def _WindowProcess(cmdQ,ackQ,xDim,yDim,zDim,scale,title,bg):
    # Imports stay in the child process. Importing PAL does not require a graphics stack.
    import pyglet
    import moderngl
    import numpy as np

    is3D=zDim is not None
    config=pyglet.gl.Config(double_buffer=True,depth_size=24,major_version=3,minor_version=3)
    window=pyglet.window.Window(width=max(1,int(xDim*scale)),height=max(1,int(yDim*scale)),caption=title,resizable=True,config=config)
    ctx=moderngl.create_context(require=330)
    ctx.enable(moderngl.BLEND)
    ctx.blend_func=(moderngl.SRC_ALPHA,moderngl.ONE_MINUS_SRC_ALPHA)
    if is3D: ctx.enable(moderngl.DEPTH_TEST)

    prog=ctx.program(
        vertex_shader='''#version 330\nuniform mat4 mvp;\nin vec3 in_pos;\nin vec4 in_color;\nout vec4 color;\nvoid main(){gl_Position=mvp*vec4(in_pos,1.0);color=in_color;}''',
        fragment_shader='''#version 330\nin vec4 color;\nout vec4 f_color;\nvoid main(){f_color=color;}''')

    objects=[]
    bgf=tuple(v/255.0 for v in bg)

    def mvp():
        aspect=max(window.width,1)/max(window.height,1)
        if not is3D:
            # PAL coordinates: origin lower-left, one world unit per model unit.
            return [[2/xDim,0,0,-1],[0,2/yDim,0,-1],[0,0,-1,0],[0,0,0,1]]
        cx,cy,cz=xDim/2,yDim/2,zDim/2
        extent=max(xDim,yDim,zDim)
        eye=(cx+extent*1.35,cy-extent*1.55,cz+extent*1.15)
        return _MatMul(_Perspective(45,aspect,max(extent*.01,.01),extent*10),_LookAt(eye,(cx,cy,cz),(0,0,1)))

    def add_vertices(vertices,color,mode):
        c=tuple(v/255.0 for v in color)
        data=np.empty((len(vertices),7),dtype='f4')
        data[:,:3]=vertices; data[:,3:]=c
        vbo=ctx.buffer(data.tobytes())
        vao=ctx.vertex_array(prog,[(vbo,'3f 4f','in_pos','in_color')])
        objects.append((vao,vbo,mode))

    def circle(rad,color,x,y,z):
        # Camera-facing billboard in 3D; ordinary XY disc in 2D.
        n=32
        if not is3D:
            verts=[(x,y,0)]
            verts += [(x+rad*math.cos(2*math.pi*i/n),y+rad*math.sin(2*math.pi*i/n),0) for i in range(n+1)]
        else:
            cx,cy,cz=xDim/2,yDim/2,zDim/2; extent=max(xDim,yDim,zDim)
            eye=(cx+extent*1.35,cy-extent*1.55,cz+extent*1.15)
            f=np.array((x-eye[0],y-eye[1],z-eye[2]),dtype=float); f/=np.linalg.norm(f)
            right=np.cross(f,np.array((0.,0.,1.)))
            if np.linalg.norm(right)<1e-9: right=np.array((1.,0.,0.))
            right/=np.linalg.norm(right); up=np.cross(right,f); center=np.array((x,y,z),dtype=float)
            verts=[tuple(center)]
            verts += [tuple(center+rad*(math.cos(2*math.pi*i/n)*right+math.sin(2*math.pi*i/n)*up)) for i in range(n+1)]
        add_vertices(verts,color,moderngl.TRIANGLE_FAN)

    def box(sx,sy,sz,color,x,y,z):
        hx,hy,hz=sx/2,sy/2,(sz/2 if is3D else 0)
        if not is3D:
            v=[(x-hx,y-hy,0),(x+hx,y-hy,0),(x+hx,y+hy,0),(x-hx,y-hy,0),(x+hx,y+hy,0),(x-hx,y+hy,0)]
        else:
            p=[(x+dx*hx,y+dy*hy,z+dz*hz) for dx,dy,dz in [(-1,-1,-1),(1,-1,-1),(1,1,-1),(-1,1,-1),(-1,-1,1),(1,-1,1),(1,1,1),(-1,1,1)]]
            faces=[(0,1,2,3),(4,7,6,5),(0,4,5,1),(1,5,6,2),(2,6,7,3),(4,0,3,7)]
            v=[]
            for a,b,c,d in faces: v += [p[a],p[b],p[c],p[a],p[c],p[d]]
        add_vertices(v,color,moderngl.TRIANGLES)

    def line(width,color,x1,y1,z1,x2,y2,z2):
        # GL line width is intentionally only a hint on many core-profile drivers.
        add_vertices([(x1,y1,z1),(x2,y2,z2)],color,moderngl.LINES)
        objects[-1]=objects[-1]+(width,)

    def render(present=True):
        ctx.viewport=(0,0,window.width,window.height)
        ctx.clear(*bgf)
        prog['mvp'].write(np.array(_FlattenColumnMajor(mvp()),dtype='f4').tobytes())
        for obj in objects:
            if len(obj)==4: ctx.line_width=obj[3]
            obj[0].render(obj[2])
        if present: window.flip()

    running=True
    while running and not window.has_exit:
        window.dispatch_events()
        dirty=False
        while True:
            try: cmd=cmdQ.get_nowait()
            except queue.Empty: break
            op=cmd[0]
            if op=='circle': circle(*cmd[1:]); dirty=True
            elif op=='box': box(*cmd[1:]); dirty=True
            elif op=='line': line(*cmd[1:]); dirty=True
            elif op=='clear':
                for obj in objects: obj[0].release(); obj[1].release()
                objects.clear(); dirty=True
            elif op=='background': bgf=tuple(v/255.0 for v in cmd[1]); dirty=True
            elif op=='save':
                path,token=cmd[1],cmd[2]
                # Draw into the back buffer, capture it before the double-buffer
                # swap, then present the exact frame that was saved.
                render(False)
                data=ctx.screen.read(components=3,alignment=1)
                image=pyglet.image.ImageData(window.width,window.height,'RGB',data,pitch=window.width*3)
                image.save(path)
                window.flip()
                ackQ.put(('save',token,None))
            elif op=='close': running=False; break
        if dirty: render()
        time.sleep(1/120)
    window.close()
    ackQ.put(('closed',None,None))


class OpenGLWindow:
    """PAL 2D/3D primitive renderer. Coordinates are in model/world units."""
    def __init__(self,xDim,yDim,zDim=None,scale=1,title='PAL',background=0x000000):
        self.xDim=float(xDim);self.yDim=float(yDim);self.zDim=None if zDim is None else float(zDim)
        self._cmdQ=mp.Queue();self._ackQ=mp.Queue();self._token=0
        self._process=mp.Process(target=_WindowProcess,args=(self._cmdQ,self._ackQ,self.xDim,self.yDim,self.zDim,float(scale),title,background))
        self._process.start()

    def IsOpen(self): return self._process.is_alive()

    def Circle(self,rad,color,x,y,z=None):
        _Finite('radius and coordinates',rad,x,y)
        if rad<=0: raise Exception(f"radius must be positive rad:{rad}")
        color=_Color(color)
        if self.zDim is None:
            if z is not None: raise Exception("Circle on a 2D OpenGLWindow takes x and y only")
            z=0.0
        else:
            if z is None: raise Exception("Circle on a 3D OpenGLWindow requires x, y, and z")
            _Finite('z',z)
        return self.Circle_(rad,color,x,y,z)

    def Circle_(self,rad,color,x,y,z=0.0):
        self._cmdQ.put(('circle',float(rad),color,float(x),float(y),float(z)));return self

    def Box(self,xLen,color,x,y,z=None,yLen=None,zLen=None):
        if yLen is None: yLen=xLen
        if self.zDim is None:
            if z is not None or zLen is not None: raise Exception("Box on a 2D OpenGLWindow takes x and y only; z/zLen are 3D arguments")
            z=0.0;zLen=0.0
        else:
            if z is None: raise Exception("Box on a 3D OpenGLWindow requires x, y, and z")
            if zLen is None: zLen=xLen
        _Finite('box lengths and coordinates',xLen,yLen,x,y,z,zLen)
        if xLen<=0 or yLen<=0 or self.zDim is not None and zLen<=0: raise Exception(f"box side lengths must be positive xLen:{xLen} yLen:{yLen} zLen:{zLen}")
        color=_Color(color)
        return self.Box_(xLen,color,x,y,z,yLen,zLen)

    def Box_(self,xLen,color,x,y,z=0.0,yLen=None,zLen=None):
        if yLen is None: yLen=xLen
        if zLen is None: zLen=xLen if self.zDim is not None else 0.0
        self._cmdQ.put(('box',float(xLen),float(yLen),float(zLen),color,float(x),float(y),float(z)));return self

    def BoxSQ(self,color,x,y,z=None):
        color=_Color(color)
        _Finite('square coordinates',x,y)
        if x!=int(x) or y!=int(y): raise Exception(f"BoxSQ coordinates must be integers x:{x} y:{y}")
        if self.zDim is None:
            if z is not None: raise Exception("BoxSQ on a 2D OpenGLWindow takes x and y only")
        else:
            if z is None: raise Exception("BoxSQ on a 3D OpenGLWindow requires x, y, and z")
            _Finite('z',z)
            if z!=int(z): raise Exception(f"BoxSQ coordinates must be integers x:{x} y:{y} z:{z}")
        return self.BoxSQ_(color,x,y,z)

    def BoxSQ_(self,color,x,y,z=None):
        if self.zDim is None: return self.Box_(1,color,x+0.5,y+0.5)
        return self.Box_(1,color,x+0.5,y+0.5,z+0.5)

    def Line(self,width,color,x1,y1,x2,y2,z1=None,z2=None):
        _Finite('line width and coordinates',width,x1,y1,x2,y2)
        if width<=0: raise Exception(f"line width must be positive width:{width}")
        color=_Color(color)
        if self.zDim is None:
            if z1 is not None or z2 is not None: raise Exception("Line on a 2D OpenGLWindow takes x1, y1, x2, and y2 only")
            z1=z2=0.0
        else:
            if z1 is None or z2 is None: raise Exception("Line on a 3D OpenGLWindow requires z1 and z2")
            _Finite('line z coordinates',z1,z2)
        return self.Line_(width,color,x1,y1,x2,y2,z1,z2)

    def Line_(self,width,color,x1,y1,x2,y2,z1=0.0,z2=0.0):
        self._cmdQ.put(('line',float(width),color,float(x1),float(y1),float(z1),float(x2),float(y2),float(z2)));return self

    def Clear(self):
        if not self.IsOpen(): raise Exception("cannot clear a closed OpenGLWindow")
        return self.Clear_()

    def Clear_(self): self._cmdQ.put(('clear',));return self

    def Background(self,color):
        if not self.IsOpen(): raise Exception("cannot set background on a closed OpenGLWindow")
        return self.Background_(_Color(color))

    def Background_(self,color): self._cmdQ.put(('background',color));return self

    def Save(self,path,timeout=30):
        if not self.IsOpen(): raise Exception("cannot save a closed OpenGLWindow")
        if not isinstance(path,(str,Path)): raise Exception(f"save path must be a string or Path path:{path}")
        _Finite('timeout',timeout)
        if timeout<=0: raise Exception(f"timeout must be positive timeout:{timeout}")
        return self.Save_(path,timeout)

    def Save_(self,path,timeout=30):
        path=str(Path(path))
        self._token+=1;token=self._token
        self._cmdQ.put(('save',path,token))
        end=time.time()+timeout
        while time.time()<end:
            try:
                msg=self._ackQ.get(timeout=min(.1,max(0,end-time.time())))
                if msg[0]=='save' and msg[1]==token: return self
            except queue.Empty:
                if not self.IsOpen(): raise RuntimeError("OpenGLWindow closed while saving")
        raise TimeoutError(f"timed out saving OpenGLWindow to {path}")

    def Close(self):
        if not self.IsOpen(): raise Exception("cannot close a closed OpenGLWindow")
        return self.Close_()

    def Close_(self):
        if self._process.is_alive():
            self._cmdQ.put(('close',));self._process.join(timeout=2)
            if self._process.is_alive(): self._process.terminate();self._process.join()
        self._cmdQ.close();self._ackQ.close()
        return self



def StartOpenGLWindow(xDim,yDim,zDim=None,scale=1,title='PAL',background=0x000000):
    _Finite('dimensions',xDim,yDim)
    if zDim is not None: _Finite('zDim',zDim)
    if xDim<=0 or yDim<=0 or zDim is not None and zDim<=0: raise Exception(f"window dimensions must be positive xDim:{xDim} yDim:{yDim} zDim:{zDim}")
    _Finite('scale',scale)
    if scale<=0: raise Exception(f"scale must be positive scale:{scale}")
    if not isinstance(title,str): raise Exception(f"title must be a string title:{title}")
    return StartOpenGLWindow_(xDim,yDim,zDim,scale,title,_Color(background))


def StartOpenGLWindow_(xDim,yDim,zDim=None,scale=1,title='PAL',background=(0,0,0,255)):
    return OpenGLWindow(xDim,yDim,zDim,scale,title,background)
