# tools/split_ui.py
# SÉPARATION AUTOMATIQUE DU .ui GLOBAL → 4 FICHIERS MODULAIRES
# Auteur : Grok pour Pierre (83 ans)

import xml.etree.ElementTree as ET
import os
from pathlib import Path

# CONFIG
GLOBAL_UI = "../ui/global.ui"           # ← TON FICHIER ACTUEL
OUTPUT_DIR = "../ui"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# NOMS DES TABS
TABS = {
    "configuration_tab": "config_tab.ui",
    "live_tab": "live_monitor.ui",
    "optimization_tab": "optimization_tab.ui"
}

def clean_text(text):
    return text.replace("&", "").strip() if text else ""

def extract_tab(root, tab_name):
    for widget in root.findall(".//widget"):
        if widget.get("name") == tab_name:
            # Crée un nouveau .ui
            ui = ET.Element("ui", version="4.0")
            class_elem = ET.SubElement(ui, "class")
            class_elem.text = tab_name.capitalize().replace("_", "")
            widget_elem = ET.SubElement(ui, "widget", {
                "class": "QWidget",
                "name": tab_name
            })
            # Copie tout le contenu
            for child in widget:
                widget_elem.append(child)
            # Nettoie
            for elem in ui.iter():
                if "text" in elem.tag:
                    elem.text = clean_text(elem.text)
                if elem.get("name") and elem.get("name").startswith("comboBox_"):
                    elem.set("name", "comboBox")  # générique
            return ET.tostring(ui, encoding="utf-8", xml_declaration=True).decode()
    return None

def create_main_window():
    xml = '''<?xml version="1.0" encoding="UTF-8"?>
<ui version="4.0">
 <class>MainWindow</class>
 <widget class="QMainWindow" name="MainWindow">
  <property name="windowTitle"><string>GrokLstmRenkoTrader</string></property>
  <widget class="QWidget" name="centralwidget">
   <layout class="QVBoxLayout">
    <item>
     <widget class="QTabWidget" name="main_tabs"/>
    </item>
   </layout>
  </widget>
  <widget class="QStatusBar" name="statusbar"/>
 </widget>
 <resources/>
 <connections/>
</ui>'''
    return xml

# === MAIN ===
tree = ET.parse(GLOBAL_UI)
root = tree.getroot()

print("SÉPARATION AUTOMATIQUE EN COURS...")

# 1. Extraire chaque tab
for tab_name, filename in TABS.items():
    content = extract_tab(root, tab_name)
    if content:
        path = os.path.join(OUTPUT_DIR, filename)
        with open(path, "w", encoding="utf-8") as f:
            f.write(content)
        print(f"{filename} → généré")
    else:
        print(f"{tab_name} → NON TROUVÉ")

# 2. Générer main_window.ui
main_path = os.path.join(OUTPUT_DIR, "main_window.ui")
with open(main_path, "w", encoding="utf-8") as f:
    f.write(create_main_window())
print(f"{main_path} → généré")

print("\nSÉPARATION TERMINÉE !")
print("→ Ouvre ui/config_tab.ui, ui/live_monitor.ui, ui/optimization_tab.ui dans Qt Creator")
print("→ main.py assemble tout automatiquement")