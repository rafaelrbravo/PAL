import multiprocessing as mp
from multiprocessing import shared_memory
import numpy as np


def _WindowProcess(shmName, xDim, yDim, scale, closeEvent):
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
        if scale != 1:
            surf = pygame.transform.scale( surf, (xDim * scale, yDim * scale))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        clock.tick(60)
    closeEvent.set()
    pygame.quit()
    shm.close()


class PixWindow:

    def __init__(self, shm, process, closeEvent):
        self.shm = shm
        self.process = process
        self.closeEvent = closeEvent

    def IsOpen(self):
        return ( self.process.is_alive() and not self.closeEvent.is_set())

    def Close(self):
        if self.process.is_alive():
            self.process.terminate()
            self.process.join()

        self.shm.close()

        try: self.shm.unlink()
        except FileNotFoundError: pass


def StartPixWindow(xDim, yDim, scale=1):
    nBytes = xDim * yDim * 3
    shm = shared_memory.SharedMemory( create=True, size=nBytes)
    pix = np.ndarray( (xDim, yDim, 3), dtype=np.uint8, buffer=shm.buf)
    pix.fill(0)
    closeEvent = mp.Event()
    process = mp.Process( target=_WindowProcess, args=( shm.name, xDim, yDim, scale, closeEvent))
    process.start()
    win = PixWindow( shm, process, closeEvent)
    return pix, win