from PyQt6.QtWidgets import QDateEdit
from PyQt6.QtWidgets import QComboBox, QLineEdit, QCheckBox
from PyQt6.QtCore import QDate

from utils.utils import to_number

# =================== les liste ou dict fournis sont ceux de config

def reset_all_widget(widgetList):
    for widget in widgetList:
        if isinstance(widget, QComboBox):
             widget.setCurrentIndex(0)
        elif isinstance(widget, QLineEdit):
            widget.setText(str(""))
        elif isinstance(widget, QCheckBox):
            widget.setChecked(False)

# dans les appels de liste les value sont les données mais pas les widgets = il faut un mapping
def set_widget_from_list(qui, name, wlist):
    i = 0
    widgetList = qui.mapping[name]
    reset_all_widget(widgetList)
    widgetName = [widgetList[k].objectName() for k in range(len(widgetList))]
    for value in wlist:     # liste du contenu de config
        if i > len(widgetList):
            return
        if value in widgetName: # si la donnée est aussi le nom du widget
            widget = widgetList[widgetName.index(value)]
            if isinstance(widget, QCheckBox):
                widget.setChecked(True)
        else:
            while widgetName[i] == value:
                i += 1
            if isinstance(widgetList[i], QComboBox):
                itemList = [widgetList[i].itemText(j) for j in range(widgetList[i].count())]
                if value in  itemList:
                    widgetList[i].setCurrentText(str(value))
                else:
                    continue
            elif isinstance(widgetList[i], QLineEdit):
                widgetList[i].setText(str(value))
            elif isinstance(widgetList[i], QCheckBox):
                widgetList[i].setChecked(value)
            i += 1

def set_widget_from_dict(qui, wdict: dict):
    for name, value in wdict.items():
        if isinstance(value, dict):
            set_widget_from_dict(qui, value)
            continue
        if isinstance(value, list):
            set_widget_from_list(qui, name, value)
            continue
        try:
            widget = getattr(qui, name)
            if widget is not None:
                if isinstance(widget, QComboBox):
                    itemList = [widget.itemText(j) for j in range(widget.count())]
                    if value in itemList:
                        widget.setCurrentText(str(value))
                    else:
                        continue
                elif isinstance(widget, QLineEdit):
                    widget.setText(str(value))
                elif isinstance(widget, QCheckBox):
                    widget.setChecked(value)
                elif isinstance(widget, QDateEdit):
                    widget.setDate(QDate.fromString(value, "yyyy-MM-dd"))

        except Exception as e:
            print(f"{qui} {e}")

def get_widget_from_list(qui, name):
    i = 0
    widgetList = qui.mapping[name]
    futurList = []
    widgetName = [widgetList[k].objectName() for k in range(len(widgetList))]
    for i in range(len(widgetList)):
        if isinstance(widgetList[i], QComboBox):
            if widgetList[i].currentIndex != 0:
                futurList.append(widgetList[i].currentText())
        elif isinstance(widgetList[i], QLineEdit):
            futurList.append(to_number(widgetList[i].text()))
        elif isinstance(widgetList[i], QCheckBox):
            if widgetList[i].isChecked():
                futurList.append(widgetName[i])
    return futurList

def get_widget_from_dict(qui, dlist: dict):
    for name, value in dlist.items():
        if isinstance(value, dict):
            get_widget_from_dict(qui, value)
            continue
        if isinstance(value, list):
            dlist[name] = get_widget_from_list(qui, name)
            continue
        try:
            widget = getattr(qui, name)
            if widget is not None:
                if isinstance(widget, QComboBox):
                    dlist[name] = widget.currentText()
                elif isinstance(widget, QLineEdit):
                    dlist[name] = to_number(widget.text())
                elif isinstance(widget, QCheckBox):
                    dlist[name] = True if widget.isChecked() else False
                elif isinstance(widget, QDateEdit):
                    dlist[name] = widget.date().toString("yyyy-MM-dd")
        except Exception as e:
            print(f"err de saisie élément {e}")
