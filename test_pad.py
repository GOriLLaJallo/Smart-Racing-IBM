import pygame
import time

def main():
    pygame.init()
    pygame.joystick.init()

    if pygame.joystick.get_count() == 0:
        print("Nessun controller collegato!")
        return

    joy = pygame.joystick.Joystick(0)
    joy.init()
    
    print("=====================================================")
    print(f"🎮 Controller rilevato: {joy.get_name()}")
    print("=====================================================")
    print("Muovi lo sterzo, premi i pedali e i pulsanti del cambio.")
    print("Vedrai a schermo il 'Numero' che Pygame assegna a quel tasto.")
    print("Premi Ctrl+C nel terminale per uscire.")
    print("=====================================================\n")

    try:
        while True:
            pygame.event.pump()
            
            # Test Assi (Levette e Grilletti)
            for i in range(joy.get_numaxes()):
                val = joy.get_axis(i)
                # Mostra solo se l'asse è mosso significativamente (evita spam per le deadzone)
                if abs(val) > 0.15:
                    print(f"Asse MOSSO: {i}  (Valore: {val:.2f})           ", end="\r")
                    
            # Test Pulsanti
            for i in range(joy.get_numbuttons()):
                if joy.get_button(i):
                    print(f"\n[!] Pulsante PREMUTO: {i}")
                    time.sleep(0.3)  # Pausa per evitare spam quando tieni premuto
                    
            time.sleep(0.05)
            
    except KeyboardInterrupt:
        print("\n\nTest terminato. Usa i numeri scoperti per mappare il gioco!")
        pygame.quit()

if __name__ == "__main__":
    main()
