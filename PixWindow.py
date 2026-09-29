import multiprocessing as mp
from multiprocessing import shared_memory
import numpy as np


def _WindowProcess(shmName, saveShmName, xDim, yDim, scale, closeEvent, savePath, saveEvent):
    saveShm = shared_memory.SharedMemory(name=saveShmName)
    savePix = np.ndarray((xDim, yDim, 3), dtype=np.uint8, buffer=saveShm.buf)
    import pygame
    shm = shared_memory.SharedMemory(name=shmName)
    pix = np.ndarray( (xDim, yDim, 3), dtype=np.uint8, buffer=shm.buf)
    pygame.init()
    screen = pygame.display.set_mode( (xDim * scale, yDim * scale))
    clock = pygame.time.Clock()
    running = True

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT: running = False
        surf = pygame.surfarray.make_surface(pix)
        if saveEvent.is_set():
            pygame.image.save(pygame.surfarray.make_surface(savePix), savePath.value)
            saveEvent.clear()
        if scale != 1:
            surf = pygame.transform.scale( surf, (xDim * scale, yDim * scale))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        clock.tick(60)
    closeEvent.set()
    pygame.quit()
    shm.close()
    saveShm.close()


class PixWindow:

    def __init__(self,shm,saveShm,pix,savePix,process,closeEvent,savePath,saveEvent):
        self.shm=shm
        self.saveShm=saveShm
        self.pix=pix
        self.savePix=savePix
        self.process=process
        self.closeEvent=closeEvent
        self.savePath=savePath
        self.saveEvent=saveEvent

    def IsOpen(self): return self.process.is_alive() and not self.closeEvent.is_set()

    def Save(self,path):
        if not self.IsOpen(): raise Exception("cannot save a closed PixWindow")
        if not isinstance(path,str): raise Exception(f"save path must be a string path:{path}")
        path=str(path)
        if len(path)>=len(self.savePath): raise Exception(f"save path must be shorter than {len(self.savePath)} characters")
        if self.saveEvent.is_set(): raise Exception("previous PixWindow save is still in progress")
        return self.Save_(path)

    def Save_(self,path):
        self.savePix[:]=self.pix
        self.savePath.value=str(path)
        self.saveEvent.set()
        return self

    def Close(self):
        if not self.IsOpen(): raise Exception("cannot close a closed PixWindow")
        return self.Close_()

    def Close_(self):
        if self.process.is_alive():
            self.process.terminate()
            self.process.join()
        self.shm.close()
        self.saveShm.close()
        try: self.shm.unlink()
        except FileNotFoundError: pass
        try: self.saveShm.unlink()
        except FileNotFoundError: pass
        return self


def StartPixWindow(xDim,yDim,scale=1):
    if not np.isfinite(xDim) or xDim<=0 or xDim!=int(xDim): raise Exception(f"xDim must be a positive integer xDim:{xDim}")
    if not np.isfinite(yDim) or yDim<=0 or yDim!=int(yDim): raise Exception(f"yDim must be a positive integer yDim:{yDim}")
    if not np.isfinite(scale) or scale<=0 or scale!=int(scale): raise Exception(f"scale must be a positive integer scale:{scale}")
    return StartPixWindow_(int(xDim),int(yDim),int(scale))


def StartPixWindow_(xDim,yDim,scale=1):
    nBytes=xDim*yDim*3
    shm=shared_memory.SharedMemory(create=True,size=nBytes)
    saveShm=shared_memory.SharedMemory(create=True,size=nBytes)
    pix=np.ndarray((xDim,yDim,3),dtype=np.uint8,buffer=shm.buf)
    savePix=np.ndarray((xDim,yDim,3),dtype=np.uint8,buffer=saveShm.buf)
    pix.fill(0)
    savePix.fill(0)
    closeEvent=mp.Event()
    savePath=mp.Array('u',4096)
    saveEvent=mp.Event()
    process=mp.Process(target=_WindowProcess,args=(shm.name,saveShm.name,xDim,yDim,scale,closeEvent,savePath,saveEvent))
    process.start()
    win=PixWindow(shm,saveShm,pix,savePix,process,closeEvent,savePath,saveEvent)
    return pix,win
