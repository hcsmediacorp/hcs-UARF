#!/usr/bin/env python3
"""
UARF Swap Manager - Automatische Swap-File Verwaltung für speichereffizientes Training

Features:
- Automatische Swap-Erkennung und -Erstellung
- Manuelles Swap-Setup mit benutzerdefinierten Parametern
- Plattform-spezifische Optimierung (Linux, Windows, Android)
- Memory-Mapped Files für effizientes Offloading
"""

import os
import sys
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Optional, Dict, Any, Tuple
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class SwapConfig:
    """Konfiguration für Swap-Management"""
    enabled: bool = True
    auto_mode: bool = True
    swap_size_gb: float = 4.0
    swap_path: Optional[str] = None
    priority: int = 10
    use_zram: bool = False
    max_usage_percent: float = 80.0


class SwapManager:
    """
    Verwaltet Swap-Files für speichereffizientes Training
    
    Unterstützt:
    - Linux Swap-Files
    - Windows Pagefile
    - Android ZRAM/Swap
    - Memory-Mapped Offloading
    """
    
    def __init__(self, config: Optional[SwapConfig] = None):
        self.config = config or SwapConfig()
        self.swap_active = False
        self.original_swap = None
        self.platform = self._detect_platform()
        
    def _detect_platform(self) -> str:
        """Erkennt die aktuelle Plattform"""
        if os.name == 'nt':
            return 'windows'
        elif os.name == 'posix':
            # Prüfen ob Android (Termux)
            if os.path.exists('/system/bin/app_process'):
                return 'android'
            return 'linux'
        return 'unknown'
    
    def get_system_memory_info(self) -> Dict[str, float]:
        """Ruft System-Speicherinformationen ab"""
        import psutil
        
        mem = psutil.virtual_memory()
        swap = psutil.swap_memory()
        
        return {
            'total_ram_gb': mem.total / (1024**3),
            'available_ram_gb': mem.available / (1024**3),
            'used_ram_gb': mem.used / (1024**3),
            'ram_percent': mem.percent,
            'total_swap_gb': swap.total / (1024**3),
            'used_swap_gb': swap.used / (1024**3),
            'swap_percent': swap.percent,
        }
    
    def check_swap_exists(self) -> bool:
        """Prüft ob Swap aktiv ist"""
        if self.platform == 'windows':
            # Windows Pagefile prüfen
            import ctypes
            pagefile_info = ctypes.windll.kernel32.GetPageFileSizeW
            return True  # Windows hat immer Pagefile
        else:
            # Linux/Android: /proc/swaps prüfen
            try:
                with open('/proc/swaps', 'r') as f:
                    lines = f.readlines()
                    return len(lines) > 1  # Header + mindestens ein Swap
            except:
                return False
    
    def calculate_optimal_swap_size(self) -> float:
        """Berechnet optimale Swap-Größe basierend auf verfügbarem RAM"""
        mem_info = self.get_system_memory_info()
        total_ram = mem_info['total_ram_gb']
        
        # Empfehlungen basierend auf RAM-Größe
        if total_ram < 2:
            return 4.0  # Sehr wenig RAM → viel Swap
        elif total_ram < 4:
            return 8.0
        elif total_ram < 8:
            return 4.0
        elif total_ram < 16:
            return 2.0
        else:
            return 0.0  # Genug RAM, kein Swap nötig
    
    def create_swap_file(self, size_gb: float, path: Optional[str] = None) -> Optional[str]:
        """
        Erstellt eine Swap-Datei
        
        Args:
            size_gb: Größe in GB
            path: Pfad zur Swap-Datei (optional)
            
        Returns:
            Pfad zur erstellten Swap-Datei oder None bei Fehler
        """
        if not self.config.enabled:
            logger.info("Swap ist deaktiviert")
            return None
        
        if self.platform == 'windows':
            logger.warning("Swap-Erstellung unter Windows wird nicht direkt unterstützt")
            logger.warning("Bitte verwenden Sie die Windows-Systemsteuerung")
            return None
        
        # Pfad bestimmen
        if path is None:
            if self.config.swap_path:
                swap_path = Path(self.config.swap_path)
            else:
                # Standardpfade pro Plattform
                if self.platform == 'android':
                    swap_path = Path('/sdcard/uarf_swap.bin')
                else:
                    swap_path = Path('/tmp/uarf_swap.bin')
        else:
            swap_path = Path(path)
        
        # Prüfen ob ausreichend Platz
        try:
            free_space = shutil.disk_usage(swap_path.parent).free / (1024**3)
            if free_space < size_gb:
                logger.error(f"Nicht genügend Speicherplatz: {free_space:.2f}GB verfügbar, {size_gb}GB benötigt")
                return None
        except Exception as e:
            logger.error(f"Fehler beim Prüfen des Speicherplatzes: {e}")
            return None
        
        # Swap-Datei erstellen
        print(f"📝 Erstelle Swap-Datei: {swap_path} ({size_gb}GB)")
        
        try:
            # Datei mit Nullen füllen (effizient mit truncate)
            with open(swap_path, 'wb') as f:
                f.truncate(int(size_gb * 1024**3))
            
            # Berechtigungen setzen (nur für Owner lesbar/schreibbar)
            os.chmod(swap_path, 0o600)
            
            # Als Swap formatieren (nur Linux/Android)
            if self.platform in ['linux', 'android']:
                try:
                    subprocess.run(['mkswap', str(swap_path)], check=True, capture_output=True)
                    print(f"✅ Swap-Datei formatiert: {swap_path}")
                except subprocess.CalledProcessError as e:
                    logger.error(f"mkswap fehlgeschlagen: {e}")
                    return None
            
            return str(swap_path)
            
        except Exception as e:
            logger.error(f"Fehler beim Erstellen der Swap-Datei: {e}")
            return None
    
    def enable_swap(self, swap_path: str, priority: Optional[int] = None) -> bool:
        """
        Aktiviert eine Swap-Datei
        
        Args:
            swap_path: Pfad zur Swap-Datei
            priority: Swap-Priorität (höher = bevorzugt)
            
        Returns:
            True bei Erfolg, False bei Fehler
        """
        if self.platform == 'windows':
            logger.warning("Swap-Aktivierung unter Windows nicht direkt unterstützt")
            return False
        
        priority = priority or self.config.priority
        
        try:
            # Swap aktivieren
            cmd = ['swapon', '-p', str(priority), swap_path]
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode != 0:
                logger.error(f"swapon fehlgeschlagen: {result.stderr}")
                return False
            
            self.swap_active = True
            print(f"✅ Swap aktiviert: {swap_path} (Priorität: {priority})")
            return True
            
        except Exception as e:
            logger.error(f"Fehler beim Aktivieren von Swap: {e}")
            return False
    
    def disable_swap(self, swap_path: str) -> bool:
        """Deaktiviert eine Swap-Datei"""
        if self.platform == 'windows':
            return False
        
        try:
            subprocess.run(['swapoff', swap_path], check=True, capture_output=True)
            self.swap_active = False
            print(f"✅ Swap deaktiviert: {swap_path}")
            return True
        except Exception as e:
            logger.error(f"Fehler beim Deaktivieren von Swap: {e}")
            return False
    
    def remove_swap_file(self, swap_path: str) -> bool:
        """Entfernt eine Swap-Datei"""
        try:
            # Erst deaktivieren falls aktiv
            if self.swap_active:
                self.disable_swap(swap_path)
            
            # Datei löschen
            Path(swap_path).unlink()
            print(f"✅ Swap-Datei entfernt: {swap_path}")
            return True
        except Exception as e:
            logger.error(f"Fehler beim Entfernen der Swap-Datei: {e}")
            return False
    
    def setup_auto_swap(self) -> bool:
        """
        Automatisches Swap-Setup
        
        Erkennt Speicherbedarf und erstellt optimalen Swap
        """
        if not self.config.auto_mode:
            return False
        
        mem_info = self.get_system_memory_info()
        
        print("\n🔍 Analysiere Systemspeicher...")
        print(f"   RAM: {mem_info['total_ram_gb']:.1f}GB total, {mem_info['available_ram_gb']:.1f}GB verfügbar")
        print(f"   Swap: {mem_info['total_swap_gb']:.1f}GB total, {mem_info['swap_percent']:.1f}% genutzt")
        
        # Prüfen ob Swap benötigt wird
        if mem_info['available_ram_gb'] > 4 and mem_info['total_swap_gb'] > 0:
            print("✅ Ausreichend Speicher vorhanden, kein zusätzlicher Swap nötig")
            return True
        
        # Optimale Größe berechnen
        optimal_size = self.calculate_optimal_swap_size()
        
        if optimal_size <= 0:
            print("✅ Kein zusätzlicher Swap erforderlich")
            return True
        
        # Prüfen ob bereits Swap existiert
        if self.check_swap_exists() and mem_info['total_swap_gb'] >= optimal_size:
            print(f"✅ Ausreichend Swap vorhanden: {mem_info['total_swap_gb']:.1f}GB")
            return True
        
        # Swap erstellen
        print(f"\n📦 Erstelle zusätzlichen Swap: {optimal_size}GB")
        swap_path = self.create_swap_file(optimal_size)
        
        if swap_path and self.enable_swap(swap_path):
            print(f"🎉 Auto-Swap erfolgreich eingerichtet!")
            return True
        else:
            print("⚠️  Auto-Swap Setup fehlgeschlagen")
            return False
    
    def setup_manual_swap(self, size_gb: float, path: Optional[str] = None) -> bool:
        """
        Manuelles Swap-Setup
        
        Args:
            size_gb: Gewünschte Swap-Größe in GB
            path: Optionaler Pfad zur Swap-Datei
            
        Returns:
            True bei Erfolg
        """
        print(f"\n🔧 Manuelles Swap-Setup: {size_gb}GB")
        
        swap_path = self.create_swap_file(size_gb, path)
        
        if swap_path and self.enable_swap(swap_path):
            print(f"🎉 Manuelles Swap-Setup erfolgreich!")
            return True
        else:
            print("❌ Manuelles Swap-Setup fehlgeschlagen")
            return False
    
    def cleanup(self):
        """Aufräumen: Swap deaktivieren aber Datei behalten"""
        if self.swap_active and self.config.swap_path:
            self.disable_swap(self.config.swap_path)
    
    def print_status(self):
        """Druckt aktuellen Swap-Status"""
        mem_info = self.get_system_memory_info()
        
        print("\n" + "="*60)
        print("📊 SPEICHER-STATUS")
        print("="*60)
        print(f"RAM:     {mem_info['total_ram_gb']:.1f}GB total, {mem_info['available_ram_gb']:.1f}GB verfügbar ({mem_info['ram_percent']}%)")
        print(f"Swap:    {mem_info['total_swap_gb']:.1f}GB total, {mem_info['used_swap_gb']:.1f}GB genutzt ({mem_info['swap_percent']}%)")
        print(f"Plattform: {self.platform}")
        print(f"Swap aktiv: {'Ja' if self.swap_active else 'Nein'}")
        print("="*60)


def create_memory_mapped_offload(tensor_path: str, max_size_gb: float = 2.0):
    """
    Erstellt Memory-Mapped Files für Tensor-Offloading
    
    Ermöglicht das Arbeiten mit großen Tensoren durch Auslagern auf Festplatte
    """
    import numpy as np
    
    # Datei erstellen falls nicht existent
    path = Path(tensor_path)
    if not path.exists():
        # Maximalgröße in Bytes
        max_bytes = int(max_size_gb * 1024**3)
        # Array für Float32 Tensoren
        array = np.memmap(str(path), dtype='float32', mode='w+', shape=(max_bytes // 4,))
        array.flush()
    
    # Bestehende Datei mappen
    array = np.memmap(str(path), dtype='float32', mode='r+')
    return array


if __name__ == '__main__':
    # Demo/Test
    manager = SwapManager(SwapConfig(auto_mode=True, enabled=True))
    manager.print_status()
    
    if len(sys.argv) > 1:
        if sys.argv[1] == 'auto':
            manager.setup_auto_swap()
        elif sys.argv[1] == 'manual':
            size = float(sys.argv[2]) if len(sys.argv) > 2 else 4.0
            manager.setup_manual_swap(size)
        elif sys.argv[1] == 'status':
            manager.print_status()
