import pygame
import random
import asyncio
import platform

LEBAR = 600
TINGGI = 400
UKURAN_GRID = 20
BARIS = TINGGI // UKURAN_GRID
KOLOM = LEBAR // UKURAN_GRID
JUMLAH_ROBOT = 5
JUMLAH_TITIK_KOTOR = 30
JUMLAH_HAMBATAN = 10

PUTIH = (255, 255, 255)
HITAM = (0, 0, 0)
HIJAU = (0, 255, 0)
MERAH = (255, 0, 0)

pygame.init()
layar = pygame.display.set_mode((LEBAR, TINGGI))
pygame.display.set_caption("Robot Bersih-Bersih")
jam = pygame.time.Clock()

titik_kotor = {(random.randint(0, KOLOM - 1), random.randint(0, BARIS - 1)) for _ in range(JUMLAH_TITIK_KOTOR)}
hambatan = {(random.randint(0, KOLOM - 1), random.randint(0, BARIS - 1)) for _ in range(JUMLAH_HAMBATAN)}

class Robot:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.titik_bersih = set()

    def gerak(self):
        langkah_mungkin = [(0, 1), (1, 0), (0, -1), (-1, 0)]
        random.shuffle(langkah_mungkin)
        for dx, dy in langkah_mungkin:
            x_baru = max(0, min(self.x + dx, KOLOM - 1))
            y_baru = max(0, min(self.y + dy, BARIS - 1))
            if (x_baru, y_baru) not in hambatan:
                self.x, self.y = x_baru, y_baru
                break
        
        # Bersihin kalo ketemu titik kotor
        posisi_sekarang = (self.x, self.y)
        if posisi_sekarang in titik_kotor:
            titik_kotor.remove(posisi_sekarang)
            self.titik_bersih.add(posisi_sekarang)

robot_list = [Robot(random.randint(0, KOLOM - 1), random.randint(0, BARIS - 1)) for _ in range(JUMLAH_ROBOT)]

def setup():
    pass

def update_loop():
    layar.fill(PUTIH)
    for debu in titik_kotor:
        pygame.draw.rect(layar, MERAH, (debu[0] * UKURAN_GRID, debu[1] * UKURAN_GRID, UKURAN_GRID, UKURAN_GRID))
    for h in hambatan:
        pygame.draw.rect(layar, HITAM, (h[0] * UKURAN_GRID, h[1] * UKURAN_GRID, UKURAN_GRID, UKURAN_GRID))
    for robot in robot_list:
        robot.gerak()
        pygame.draw.rect(layar, HIJAU, (robot.x * UKURAN_GRID, robot.y * UKURAN_GRID, UKURAN_GRID, UKURAN_GRID))
    pygame.display.flip()

FPS = 5

async def main():
    setup()
    while True:
        update_loop()
        await asyncio.sleep(1.0 / FPS)

if platform.system() == "Emscripten":
    asyncio.ensure_future(main())
else:
    if __name__ == "__main__":
        asyncio.run(main())
