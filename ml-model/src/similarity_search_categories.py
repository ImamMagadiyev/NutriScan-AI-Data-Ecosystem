import pandas as pd
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
import time

FICHIER_A_COMPLETER = 'CSV_Avec_distributeur.csv'
FICHIER_REFERENCE = 'en.openfoodfacts.org.products.csv'
FICHIER_SORTIE = 'CSV_Avec_distributeur_FAISS.csv'
SEUIL_SIMILARITE = 0.55
MAX_REFERENCE = 200000

start_total = time.time()
print("=" * 60)
print("=" * 60)

# 1. Charger le modèle (utilise GPU si dispo)
print("\n[1/6] Modèle...")
t = time.time()
model = SentenceTransformer('all-MiniLM-L6-v2')
model.max_seq_length = 64  # Réduire pour aller plus vite
print(f"  OK ({time.time()-t:.1f}s)")

# 2. Charger référentiel en parallèle
print("\n[2/6] Référentiel...")
t = time.time()
ref_df = pd.read_csv(
    FICHIER_REFERENCE, 
    sep='\t',
    usecols=['product_name', 'categories_en'],
    dtype={'product_name': 'string', 'categories_en': 'string'},
    on_bad_lines='skip',
    engine='c'  # Engine C ultra rapide
)
# Nettoyage vectorisé
ref_df = ref_df.dropna()
mask_valid = (ref_df['product_name'].str.len() > 0) & (ref_df['categories_en'].str.len() > 0)
ref_df = ref_df[mask_valid]
ref_df['product_name'] = ref_df['product_name'].str.lower()

if len(ref_df) > MAX_REFERENCE:
    ref_df = ref_df.sample(n=MAX_REFERENCE, random_state=42)
print(f"  {len(ref_df)} produits ({time.time()-t:.1f}s)")

# 3. Charger fichier à compléter
print("\n[3/6] Fichier à compléter...")
t = time.time()
df = pd.read_csv(FICHIER_A_COMPLETER)
if 'categories_en' not in df.columns:
    df['categories_en'] = ''
mask = df['categories_en'].isna() | (df['categories_en'].astype(str).str.strip() == '')
nb_sans_cat = mask.sum()
print(f"  {nb_sans_cat} sans catégorie ({time.time()-t:.1f}s)")

if nb_sans_cat == 0:
    print("\n[OK] Tous ont une catégorie!")
    exit()

# 4. Vectorisation ULTRA FAST
print("\n[4/6] Vectorisation...")
t = time.time()
noms_ref = ref_df['product_name'].tolist()
cats_ref = ref_df['categories_en'].tolist()

# Batch énorme + multi-thread
embeddings_ref = model.encode(
    noms_ref,
    show_progress_bar=True,
    convert_to_numpy=True,
    batch_size=256,
    normalize_embeddings=True  # Normalise directement
)
print(f"  Référentiel vectorisé ({time.time()-t:.1f}s)")

# 5. Index FAISS IVF (approximatif mais ultra rapide)
print("\n[5/6] Index FAISS IVF...")
t = time.time()
dim = embeddings_ref.shape[1]
nlist = min(256, len(noms_ref) // 40)  # Nombre de clusters
quantizer = faiss.IndexFlatIP(dim)
index = faiss.IndexIVFFlat(quantizer, dim, nlist, faiss.METRIC_INNER_PRODUCT)
index.train(embeddings_ref)
index.add(embeddings_ref)
index.nprobe = 16  # Chercher dans 16 clusters (balance vitesse/précision)
print(f"  Index IVF créé ({time.time()-t:.1f}s)")

# 6. Recherche
print("\n[6/6] Recherche...")
t = time.time()
noms_query = df.loc[mask, 'product_name'].astype(str).str.lower().tolist()
indices_df = df[mask].index.tolist()

embeddings_query = model.encode(
    noms_query,
    show_progress_bar=True,
    convert_to_numpy=True,
    batch_size=256,
    normalize_embeddings=True
)

distances, indices_faiss = index.search(embeddings_query, 1)
print(f"  Recherche terminée ({time.time()-t:.1f}s)")

# Appliquer vectorisé
t = time.time()
mask_ok = distances[:, 0] >= SEUIL_SIMILARITE
categories_trouvees = [cats_ref[idx[0]] if ok else '' for idx, ok in zip(indices_faiss, mask_ok)]
df.loc[mask, 'categories_en'] = categories_trouvees
nb_completes = mask_ok.sum()

# Résultat
print(f"\n{'='*60}")
print(f"RÉSULTAT:")
print(f"  Complétés: {nb_completes}/{nb_sans_cat} ({nb_completes/nb_sans_cat*100:.1f}%)")
print(f"  Temps total: {time.time()-start_total:.1f}s")
print(f"{'='*60}")

df.to_csv(FICHIER_SORTIE, index=False)
print(f"\n[OK] Sauvegardé: {FICHIER_SORTIE}")
