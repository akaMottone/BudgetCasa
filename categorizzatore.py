import pandas as pd
import numpy as np
import os
import shutil
import re
from datetime import datetime
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import make_pipeline

# --- CONFIGURAZIONE ---
file_nuovo_nome = "CASA ROSSA Ale Nuovi movimenti.xlsx"
nome_foglio_mov = "Scheda Movimenti"
nome_foglio_cat = "Categorie" 

def pulisci_percorso(p):
    return p.strip().lstrip('&').strip().strip("'").strip('"').strip()

print("--- ASSISTENTE CATEGORIZZATORE PRO (V19.3) ---")

if not os.path.exists(file_nuovo_nome):
    print(f"\nERRORE: Non trovo '{file_nuovo_nome}'!")
    exit()

path_raw = input("Trascina qui il file con lo storico dei movimenti: ")
path_storico = pulisci_percorso(path_raw)

# Backup automatico
#if not os.path.exists("Backup dati"): os.makedirs("Backup dati")
#shutil.copy(file_nuovo_nome, f"Backup dati/BACKUP_{datetime.now().strftime('%Y%m%d_%H%M')}_{file_nuovo_nome}")

# 1. Caricamento dati
print("\n[1/3] Caricamento file e categorie...")
df_storico = pd.read_excel(path_storico, sheet_name=nome_foglio_mov, dtype=str)
df_nuovo = pd.read_excel(file_nuovo_nome, sheet_name=nome_foglio_mov, dtype=str)
df_cat = pd.read_excel(path_storico, sheet_name=nome_foglio_cat, dtype=str)

# Mappa per mostrare Categoria > Sottocategoria (Tipo)
mappa_info_cat = {str(r['Code']): f"{r['Categoria']} > {r['Sottocategoria']} ({r['Tipo']})" for _, r in df_cat.iterrows()}

def prepara_testo(df):
    return (df['Cod conto'].fillna('') + " " + df['Dettaglio'].fillna('') + " " + df['Tipologia'].fillna(''))

# 2. Allenamento
print("[2/3] Allenamento Intelligenza Artificiale...")
X_train = prepara_testo(df_storico)
def crea_modello(): return make_pipeline(TfidfVectorizer(), RandomForestClassifier(n_estimators=100, random_state=42))

model_code = crea_modello().fit(X_train, df_storico['Code'].fillna('NON_DEF'))
model_luogo = crea_modello().fit(X_train, df_storico['Luogo'].fillna(''))
model_persona = crea_modello().fit(X_train, df_storico['Persona'].fillna(''))

# 3. Analisi
print("[3/3] Analisi nuovi movimenti e ricerca storica...")
X_nuovo = prepara_testo(df_nuovo)
probs_code = model_code.predict_proba(X_nuovo)
classes_code = model_code.classes_

stats = {"auto": 0, "manual": 0, "saltate": 0, "paypal": 0, "luoghi_auto": 0, "persone_auto": 0}
stop_manuale = False
STOPWORDS = {'delle', 'dalla', 'degli', 'nella', 'della', 'dallo', 'dalle', 'con', 'del', 'per', 'presso', 'mediante', 'effettuato'}

# --- CALCOLO CONTATORE (CORRETTO PER PANDAS) ---
p_max_series = np.max(probs_code, axis=1)
cond_ia_incerta = p_max_series < 0.50
cond_senza_codice = df_nuovo['Code'].isna() | (df_nuovo['Code'].astype(str).str.strip().str.lower() == 'nan')
cond_no_paypal = ~df_nuovo['Dettaglio'].fillna('').str.lower().str.contains("paga in 3 rate|paypal *paga", na=False)

totale_manuali = len(df_nuovo[cond_ia_incerta & cond_senza_codice & cond_no_paypal])
contatore_manuale = 0

for i, row in df_nuovo.iterrows():
    desc_low = str(row['Dettaglio']).lower()
    if "paga in 3 rate" in desc_low or "paypal *paga" in desc_low:
        stats["paypal"] += 1; continue
    if pd.notna(row['Code']) and str(row['Code']).strip().lower() != 'nan': continue

    # Gestione Luogo/Persona (55%)
    for model, col, s_key in [(model_luogo, 'Luogo', "luoghi_auto"), (model_persona, 'Persona', "persone_auto")]:
        prob_array = model.predict_proba(X_nuovo.iloc[[i]])
        if np.max(prob_array) >= 0.55:
            df_nuovo.at[i, col] = model.classes_[np.argmax(prob_array)]
            stats[s_key] += 1

    p_values = probs_code[i]
    p_max = np.max(p_values)
    cod_ia = str(classes_code[np.argmax(p_values)])

    # Ricerca Storica
    parole_pulite = [re.escape(p) for p in desc_low.split() if len(p) > 3 and p not in STOPWORDS]
    voci_storiche = []
    if parole_pulite:
        regex_search = '|'.join(parole_pulite)
        filtro = df_storico['Dettaglio'].str.lower().str.contains(regex_search, na=False)
        voci_storiche = df_storico[filtro]['Code'].value_counts().head(3).index.tolist()

    # Logica di Output Manuale (Testo semplificato)
    if p_max >= 0.50:
        df_nuovo.at[i, 'Code'] = cod_ia
        df_nuovo.at[i, 'Machine learning'] = f"AI: OK ({int(p_max*100)}%)"
        stats["auto"] += 1
    elif not stop_manuale:
        contatore_manuale += 1
        print("\n" + "="*70)
        data_pulita = str(row['Data'])[:10]
        print(f"DATA: {data_pulita} | IMPORTO: {row['Importo']}€")
        # Visualizzazione semplificata richiesta: Cod conto + Dettaglio
        print(f"VOCE: {row['Cod conto']}")
        print(f"{contatore_manuale} di {totale_manuali}")
        print("-" * 70)
        
        opzioni = []
        for idx in np.argsort(p_values)[::-1][:3]:
            c = str(classes_code[idx])
            if c not in opzioni:
                opzioni.append(c)
                print(f"[{len(opzioni)}] {c} | {int(p_values[idx]*100)}% | {mappa_info_cat.get(c, 'N/D')}")
        
        for c in voci_storiche:
            if c not in opzioni:
                opzioni.append(c)
                print(f"[{len(opzioni)}] {c} (Storico) | {mappa_info_cat.get(c, 'N/D')}")

        print(f"\n[Invio] Salta riga | [q] Salva tutto e chiudi domande")
        scelta = input("Scegli numero, codice o comando: ").strip()
        
        if scelta.lower() == 'q': 
            stop_manuale = True
            stats["saltate"] += 1
        elif scelta == "": 
            stats["saltate"] += 1
        elif scelta.isdigit() and 1 <= int(scelta) <= len(opzioni):
            df_nuovo.at[i, 'Code'] = opzioni[int(scelta)-1]
            df_nuovo.at[i, 'Machine learning'] = "Manuale (Suggerito)"
            stats["manual"] += 1
        else: 
            df_nuovo.at[i, 'Code'] = scelta.upper()
            df_nuovo.at[i, 'Machine learning'] = "Manuale (Nuovo)"
            stats["manual"] += 1
    else:
        stats["saltate"] += 1

# 4. Salvataggio e Report Finale
print("\nSalvataggio dei dati in corso...")
with pd.ExcelWriter(file_nuovo_nome, engine='openpyxl', mode='a', if_sheet_exists='overlay') as writer:
    df_nuovo.to_excel(writer, sheet_name=nome_foglio_mov, index=False)

print("\n" + "=" * 30)
print("LAVORO COMPLETATO!")
print("=" * 30)
print("\n" + f"✅ IA (Automatici): {stats['auto']}")
print(f"👤 TU (Manuali): {stats['manual']}")
print(f"⏭ SALTATI/SOSPESI: {stats['saltate']}")
print(f"💳 PAYPAL (Ignorati): {stats['paypal']}")
print(f"📍 EXTRA: Luoghi: {stats['luoghi_auto']} | Persone: {stats['persone_auto']}")
print("\n" + "=" * 30)