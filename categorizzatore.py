import pandas as pd
import numpy as np
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import make_pipeline

# --- CONFIGURAZIONE ---
file_nuovo = "CASA ROSSA Ale Nuovi movimenti.xlsx"
nome_foglio_mov = "Scheda Movimenti"
nome_foglio_cat = "Categorie" 

def pulisci_percorso(p):
    # Rimuove & iniziale, apici e spazi bianchi
    return p.strip().lstrip('&').strip().strip("'").strip('"').strip()

print("--- ASSISTENTE CATEGORIZZATORE PRO (V10) ---")
path_raw = input("Trascina qui il file dello storico): ")
path_storico = pulisci_percorso(path_raw)

# 1. Caricamento Dati
try:
    df_storico = pd.read_excel(path_storico, sheet_name=nome_foglio_mov, dtype=str)
    df_nuovo = pd.read_excel(file_nuovo, sheet_name=nome_foglio_mov, dtype=str)
    df_cat = pd.read_excel(path_storico, sheet_name=nome_foglio_cat, dtype=str)
except Exception as e:
    print(f"\nERRORE: Impossibile caricare i file. Verifica il percorso.\n{e}")
    exit()

# Mappa info includendo il TIPO
mappa_info_cat = {
    str(r['Code']): f"{r['Categoria']} > {r['Sottocategoria']} ({r['Tipo']})" 
    for _, r in df_cat.iterrows()
}

def prepara_testo(df):
    return (df['Cod conto'].fillna('') + " " + df['Dettaglio'].fillna('') + " " + df['Tipologia'].fillna(''))

# 2. Allenamento Multi-Target
print("\n[1/3] Studio dello storico in corso...")
X_train = prepara_testo(df_storico)

def crea_modello():
    return make_pipeline(TfidfVectorizer(), RandomForestClassifier(n_estimators=100, random_state=42))

model_code = crea_modello().fit(X_train, df_storico['Code'].fillna('NON_DEF'))
model_luogo = crea_modello().fit(X_train, df_storico['Luogo'].fillna(''))
model_persona = crea_modello().fit(X_train, df_storico['Persona'].fillna(''))

# 3. Analisi e Interazione
X_nuovo = prepara_testo(df_nuovo)
probs_code = model_code.predict_proba(X_nuovo)
classes_code = model_code.classes_

stats = {"auto": 0, "manual": 0, "saltate": 0, "paypal": 0, "luoghi_auto": 0, "persone_auto": 0}
stop_manuale = False

for i, row in df_nuovo.iterrows():
    desc = str(row['Dettaglio']).lower()
    if "paga in 3 rate" in desc or "paypal *paga" in desc:
        stats["paypal"] += 1
        continue
    
    if pd.notna(row['Code']) and str(row['Code']).strip().lower() != 'nan':
        continue

    # Compilazione Silenziosa Luogo e Persona (>70%)
    p_luogo = model_luogo.predict_proba(X_nuovo.iloc[[i]])
    if np.max(p_luogo) >= 0.50:
        df_nuovo.at[i, 'Luogo'] = model_luogo.classes_[np.argmax(p_luogo)]
        stats["luoghi_auto"] += 1
    
    p_pers = model_persona.predict_proba(X_nuovo.iloc[[i]])
    if np.max(p_pers) >= 0.70:
        df_nuovo.at[i, 'Persona'] = model_persona.classes_[np.argmax(p_pers)]
        stats["persone_auto"] += 1

    # Logica Codice
    p_max = np.max(probs_code[i])
    cod_predetto = str(classes_code[np.argmax(probs_code[i])])

    # Regola Extra < 40€
    try:
        importo_abs = abs(float(str(row['Importo']).replace(',', '.')))
        if cod_predetto in ['SP005', 'SP006'] and importo_abs < 40:
            cod_predetto = 'SP006'
    except: pass

    if p_max >= 0.51:
        df_nuovo.at[i, 'Code'] = cod_predetto
        df_nuovo.at[i, 'Machine learning'] = f"AI: OK ({int(p_max*100)}%)"
        stats["auto"] += 1
    elif not stop_manuale:
        print("\n" + "="*60)
        print(f"DATA: {row['Data']} | IMPORTO: {row['Importo']}€")
        print(f"VOCE: {row['Cod conto']}")
        print("-" * 60)
        
        idx_v = sorted([idx for idx, p in enumerate(probs_code[i]) if p >= 0.10], key=lambda x: probs_code[i][x], reverse=True)
        opzioni = []
        for n, idx in enumerate(idx_v):
            c = str(classes_code[idx])
            opzioni.append(c)
            # QUI AGGIUNTA INFO COMPLETA CON TIPO
            print(f"[{n+1}] {c} | {int(probs_code[i][idx]*100)}% | {mappa_info_cat.get(c, 'N/D')}")
        
        print(f"[Invio] Salta riga | [fine] Salva tutto e chiudi domande")
        scelta = input("\nScegli numero, codice o comando: ").strip()
        
        if scelta.lower() == 'fine': 
            stop_manuale = True
            stats["saltate"] += 1
        elif scelta == "": 
            stats["saltate"] += 1
        elif scelta.isdigit() and 1 <= int(scelta) <= len(opzioni):
            cod_scelto = opzioni[int(scelta)-1]
            df_nuovo.at[i, 'Code'] = cod_scelto
            df_nuovo.at[i, 'Machine learning'] = f"MANUALE (Scelta {scelta})"
            stats["manual"] += 1
        else: 
            df_nuovo.at[i, 'Code'] = scelta.upper()
            df_nuovo.at[i, 'Machine learning'] = "MANUALE (Input testuale)"
            stats["manual"] += 1
    else:
        stats["saltate"] += 1

# 4. Salvataggio e Report Finale
print("\n[3/3] Salvataggio dei dati...")
try:
    with pd.ExcelWriter(file_nuovo, engine='openpyxl', mode='a', if_sheet_exists='overlay') as writer:
        df_nuovo.to_excel(writer, sheet_name=nome_foglio_mov, index=False)
except Exception as e:
    print(f"\nERRORE SALVATAGGIO: Assicurati che il file '{file_nuovo}' sia CHIUSO.\n{e}")

print("\n" + "=" * 30)
print("LAVORO COMPLETATO!")
print("=" * 30)
print("\n" + f"✅ IA (Automatici): {stats['auto']}")
print(f"👤 TU (Manuali): {stats['manual']}")
print(f"⏭ SALTATI/SOSPESI: {stats['saltate']}")
print(f"💳 PAYPAL (Ignorati): {stats['paypal']}")
print(f"📍 EXTRA: Luoghi: {stats['luoghi_auto']} | Persone: {stats['persone_auto']}")
print("\n" + "=" * 30)

