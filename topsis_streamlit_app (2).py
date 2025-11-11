import streamlit as st
import numpy as np
import pandas as pd

st.set_page_config(page_title="TOPSIS avec Entropie", layout="wide", page_icon="📊")

# Titre principal
st.title("🎯 Méthode TOPSIS avec Entropie et Poids Hiérarchiques")
st.markdown("---")

# Initialisation de session_state
if 'decision_matrix' not in st.session_state:
    st.session_state.decision_matrix = None
if 'main_criteria' not in st.session_state:
    st.session_state.main_criteria = []

# Sidebar pour la configuration
with st.sidebar:
    st.header("⚙️ Configuration")
    
    num_alternatives = st.number_input("Nombre d'alternatives", min_value=2, max_value=20, value=4)
    num_main_criteria = st.number_input("Nombre de critères principaux", min_value=1, max_value=10, value=3)
    
    st.markdown("---")
    st.info("📚 **Méthode TOPSIS**\n\nCombine les poids d'entropie (objectifs) et les poids subjectifs pour un classement optimal des alternatives.")

# Fonction pour calculer l'entropie
def calculate_entropy_weights(matrix):
    """Calcule les poids d'entropie pour chaque critère"""
    m, n = matrix.shape
    k = 1 / np.log(m)
    
    # Normalisation
    P = matrix / matrix.sum(axis=0)
    
    # Calcul de Z (éviter log(0))
    P_safe = np.where(P == 0, 1e-10, P)
    Z = P_safe / P_safe.sum(axis=0)
    
    # Calcul de l'entropie
    e = -k * np.sum(Z * np.log(Z), axis=0)
    
    # Calcul des poids
    w_entropy = (1 - e) / np.sum(1 - e)
    
    return w_entropy, e

# Fonction pour combiner les poids
def combine_weights(w_entropy, w_subjective):
    """Combine les poids d'entropie et subjectifs"""
    w_combined = (w_entropy * w_subjective) / np.sum(w_entropy * w_subjective)
    return w_combined

# Fonction TOPSIS complète
def topsis_analysis(matrix, weights, criteria_types):
    """
    Effectue l'analyse TOPSIS complète
    matrix: matrice de décision
    weights: poids combinés
    criteria_types: 'benefit' ou 'cost' pour chaque critère
    """
    m, n = matrix.shape
    
    # Étape 2: Normalisation
    P = matrix / matrix.sum(axis=0)
    
    # Étape 6: Application des poids
    U = P * weights
    
    # Étape 7: Solutions idéales
    A_plus = np.zeros(n)
    A_minus = np.zeros(n)
    
    for j in range(n):
        if criteria_types[j] == 'benefit':
            A_plus[j] = np.max(U[:, j])
            A_minus[j] = np.min(U[:, j])
        else:  # cost
            A_plus[j] = np.min(U[:, j])
            A_minus[j] = np.max(U[:, j])
    
    # Étape 8: Calcul des distances
    S_plus = np.sqrt(np.sum((U - A_plus)**2, axis=1))
    S_minus = np.sqrt(np.sum((U - A_minus)**2, axis=1))
    
    # Étape 9: Proximité relative
    C = S_minus / (S_plus + S_minus)
    
    # Étape 10: Classement
    ranking = np.argsort(-C) + 1
    
    return {
        'normalized_matrix': P,
        'weighted_matrix': U,
        'A_plus': A_plus,
        'A_minus': A_minus,
        'S_plus': S_plus,
        'S_minus': S_minus,
        'proximity': C,
        'ranking': ranking
    }

def get_color_gradient(value, min_val=0, max_val=1):
    """Génère une couleur en dégradé rouge-jaune-vert"""
    normalized = (value - min_val) / (max_val - min_val) if max_val > min_val else 0.5
    
    if normalized < 0.5:
        # Rouge vers Jaune
        r = 255
        g = int(255 * (normalized * 2))
        b = 0
    else:
        # Jaune vers Vert
        r = int(255 * (2 - normalized * 2))
        g = 255
        b = 0
    
    return f'rgba({r}, {g}, {b}, 0.3)'

# Onglets principaux
tab1, tab2, tab3, tab4 = st.tabs(["📝 Critères", "📊 Matrice de Décision", "🧮 Calculs", "📈 Résultats"])

with tab1:
    st.header("Définition des Critères Principaux et Sous-critères")
    
    main_criteria_data = []
    
    for i in range(num_main_criteria):
        st.subheader(f"🔹 Critère Principal {i+1}")
        
        col1, col2, col3 = st.columns([2, 1, 1])
        with col1:
            main_name = st.text_input(f"Nom", value=f"Critère Principal {i+1}", key=f"main_{i}")
        with col2:
            main_weight = st.number_input(f"Poids subjectif", min_value=0.0, max_value=1.0, value=round(1/num_main_criteria, 3), step=0.01, key=f"weight_{i}")
        with col3:
            num_sub = st.number_input(f"Nb sous-critères", min_value=1, max_value=10, value=2, key=f"numsub_{i}")
        
        sub_criteria = []
        for j in range(num_sub):
            col_a, col_b = st.columns([3, 1])
            with col_a:
                sub_name = st.text_input(f"  └─ Sous-critère {i+1}.{j+1}", value=f"Sous-critère {i+1}.{j+1}", key=f"sub_{i}_{j}")
            with col_b:
                sub_type = st.selectbox(f"Type", ["benefit", "cost"], key=f"type_{i}_{j}")
            
            sub_criteria.append({
                'name': sub_name,
                'type': sub_type
            })
        
        main_criteria_data.append({
            'name': main_name,
            'weight': main_weight,
            'sub_criteria': sub_criteria
        })
        
        st.markdown("---")
    
    st.session_state.main_criteria = main_criteria_data
    
    # Vérification des poids
    total_weight = sum([mc['weight'] for mc in main_criteria_data])
    if abs(total_weight - 1.0) > 0.01:
        st.warning(f"⚠️ La somme des poids subjectifs est {total_weight:.3f}. Elle devrait être égale à 1.0")
    else:
        st.success(f"✅ Somme des poids subjectifs = {total_weight:.3f}")

with tab2:
    st.header("Matrice de Décision")
    
    if st.session_state.main_criteria:
        # Créer la liste de tous les sous-critères
        all_sub_criteria = []
        criteria_types = []
        main_criteria_indices = []
        
        for mc_idx, mc in enumerate(st.session_state.main_criteria):
            for sc in mc['sub_criteria']:
                all_sub_criteria.append(f"{mc['name']}: {sc['name']}")
                criteria_types.append(sc['type'])
                main_criteria_indices.append(mc_idx)
        
        num_sub_criteria = len(all_sub_criteria)
        
        st.info(f"📋 Total: {num_alternatives} alternatives × {num_sub_criteria} sous-critères")
        
        # Créer le DataFrame pour la saisie
        if st.session_state.decision_matrix is None or st.session_state.decision_matrix.shape != (num_alternatives, num_sub_criteria):
            st.session_state.decision_matrix = pd.DataFrame(
               matrix = np.random.uniform(1, 10, size=(num_alternatives, num_sub_criteria))
                columns=all_sub_criteria,
                index=[f"Alternative {i+1}" for i in range(num_alternatives)]
            )
        
        st.markdown("### 📝 Saisir les valeurs de performance")
        edited_df = st.data_editor(
            st.session_state.decision_matrix,
            use_container_width=True,
            num_rows="fixed"
        )
        
        st.session_state.decision_matrix = edited_df
        
        # Afficher les types de critères
        st.markdown("### 🏷️ Types de critères")
        types_df = pd.DataFrame({
            'Sous-critère': all_sub_criteria,
            'Type': criteria_types,
            'Critère Principal': [st.session_state.main_criteria[idx]['name'] for idx in main_criteria_indices]
        })
        st.dataframe(types_df, use_container_width=True)

with tab3:
    st.header("Calculs Détaillés")
    
    if st.session_state.decision_matrix is not None and st.button("🚀 Lancer les calculs", type="primary", use_container_width=True):
        
        matrix = st.session_state.decision_matrix.values
        
        # Calculer les poids d'entropie pour chaque sous-critère
        w_entropy, entropy_values = calculate_entropy_weights(matrix)
        
        # Préparer les poids subjectifs des sous-critères
        w_subjective = []
        for mc_idx, mc in enumerate(st.session_state.main_criteria):
            main_weight = mc['weight']
            num_sub = len(mc['sub_criteria'])
            # Répartir le poids principal également entre les sous-critères
            for _ in range(num_sub):
                w_subjective.append(main_weight / num_sub)
        
        w_subjective = np.array(w_subjective)
        
        # Normaliser les poids subjectifs
        w_subjective = w_subjective / w_subjective.sum()
        
        # Combiner les poids
        w_combined = combine_weights(w_entropy, w_subjective)
        
        # Extraire les types de critères
        criteria_types = []
        for mc in st.session_state.main_criteria:
            for sc in mc['sub_criteria']:
                criteria_types.append(sc['type'])
        
        # Analyse TOPSIS
        results = topsis_analysis(matrix, w_combined, criteria_types)
        
        st.session_state.results = {
            'w_entropy': w_entropy,
            'entropy_values': entropy_values,
            'w_subjective': w_subjective,
            'w_combined': w_combined,
            'topsis': results,
            'criteria_types': criteria_types
        }
        
        # Affichage des résultats intermédiaires
        st.success("✅ Calculs terminés avec succès!")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Nombre d'alternatives", num_alternatives)
        with col2:
            st.metric("Nombre de sous-critères", len(w_entropy))
        with col3:
            best_alt = np.argmax(results['proximity']) + 1
            st.metric("Meilleure alternative", f"Alternative {best_alt}", delta="Optimal")
        
        st.markdown("---")
        
        # Tableau des poids
        st.markdown("### ⚖️ Comparaison des Poids")
        weights_df = pd.DataFrame({
            'Sous-critère': st.session_state.decision_matrix.columns,
            'Entropie (Objectif)': w_entropy,
            'Subjectif': w_subjective,
            'Combiné': w_combined,
            'Type': criteria_types
        })
        
        st.dataframe(
            weights_df.style.format({
                'Entropie (Objectif)': '{:.4f}',
                'Subjectif': '{:.4f}',
                'Combiné': '{:.4f}'
            }).background_gradient(subset=['Combiné'], cmap='YlGn'),
            use_container_width=True
        )
        
        # Graphique simple des poids avec des barres
        st.markdown("### 📊 Visualisation des Poids")
        chart_data = pd.DataFrame({
            'Entropie': w_entropy,
            'Subjectif': w_subjective,
            'Combiné': w_combined
        }, index=[f'C{i+1}' for i in range(len(w_entropy))])
        st.bar_chart(chart_data)
        
        # Matrice normalisée
        st.markdown("### 📐 Matrice Normalisée")
        normalized_df = pd.DataFrame(
            results['normalized_matrix'],
            columns=st.session_state.decision_matrix.columns,
            index=st.session_state.decision_matrix.index
        )
        st.dataframe(normalized_df.style.format('{:.4f}').background_gradient(cmap='Blues'), use_container_width=True)
        
        # Matrice pondérée
        st.markdown("### 🎯 Matrice Pondérée (U)")
        weighted_df = pd.DataFrame(
            results['weighted_matrix'],
            columns=st.session_state.decision_matrix.columns,
            index=st.session_state.decision_matrix.index
        )
        st.dataframe(weighted_df.style.format('{:.4f}').background_gradient(cmap='Greens'), use_container_width=True)
        
        # Solutions idéales
        st.markdown("### 🎯 Solutions Idéales")
        col_a, col_b = st.columns(2)
        with col_a:
            st.info("**Solution Positive Idéale (A+)**")
            ideal_pos_df = pd.DataFrame({
                'Critère': [f'C{i+1}' for i in range(len(results['A_plus']))],
                'Valeur': results['A_plus']
            })
            st.dataframe(ideal_pos_df.style.format({'Valeur': '{:.4f}'}), use_container_width=True)
        
        with col_b:
            st.warning("**Solution Négative Idéale (A-)**")
            ideal_neg_df = pd.DataFrame({
                'Critère': [f'C{i+1}' for i in range(len(results['A_minus']))],
                'Valeur': results['A_minus']
            })
            st.dataframe(ideal_neg_df.style.format({'Valeur': '{:.4f}'}), use_container_width=True)

with tab4:
    st.header("Résultats Finaux")
    
    if 'results' in st.session_state and st.session_state.results is not None:
        results = st.session_state.results
        topsis = results['topsis']
        
        # Tableau de classement
        st.markdown("### 🏆 Classement Final")
        
        ranking_df = pd.DataFrame({
            'Alternative': st.session_state.decision_matrix.index,
            'S+ (Distance PIS)': topsis['S_plus'],
            'S- (Distance NIS)': topsis['S_minus'],
            'Proximité Relative (Ci)': topsis['proximity'],
            'Rang': topsis['ranking']
        }).sort_values('Rang')
        
        st.dataframe(
            ranking_df.style.format({
                'S+ (Distance PIS)': '{:.4f}',
                'S- (Distance NIS)': '{:.4f}',
                'Proximité Relative (Ci)': '{:.4f}'
            }).background_gradient(subset=['Proximité Relative (Ci)'], cmap='RdYlGn'),
            use_container_width=True
        )
        
        # Affichage visuel du classement
        st.markdown("### 📊 Visualisation des Proximités")
        prox_chart = pd.DataFrame({
            'Proximité': topsis['proximity']
        }, index=st.session_state.decision_matrix.index)
        st.bar_chart(prox_chart)
        
        # Comparaison des distances
        st.markdown("### 📏 Comparaison des Distances")
        dist_chart = pd.DataFrame({
            'Distance à PIS (S+)': topsis['S_plus'],
            'Distance à NIS (S-)': topsis['S_minus']
        }, index=st.session_state.decision_matrix.index)
        st.bar_chart(dist_chart)
        
        # Podium des 3 meilleures alternatives
        st.markdown("### 🥇 Podium")
        col1, col2, col3 = st.columns(3)
        
        sorted_indices = np.argsort(-topsis['proximity'])
        
        if len(sorted_indices) >= 1:
            with col1:
                st.success("**🥇 1ère Place**")
                idx = sorted_indices[0]
                st.metric(
                    st.session_state.decision_matrix.index[idx],
                    f"{topsis['proximity'][idx]:.4f}",
                    delta="Meilleure"
                )
        
        if len(sorted_indices) >= 2:
            with col2:
                st.info("**🥈 2ème Place**")
                idx = sorted_indices[1]
                st.metric(
                    st.session_state.decision_matrix.index[idx],
                    f"{topsis['proximity'][idx]:.4f}"
                )
        
        if len(sorted_indices) >= 3:
            with col3:
                st.warning("**🥉 3ème Place**")
                idx = sorted_indices[2]
                st.metric(
                    st.session_state.decision_matrix.index[idx],
                    f"{topsis['proximity'][idx]:.4f}"
                )
        
        st.markdown("---")
        
        # Analyse comparative détaillée
        st.markdown("### 🔍 Analyse Comparative Détaillée")
        
        selected_alts = st.multiselect(
            "Sélectionner les alternatives à comparer",
            options=list(st.session_state.decision_matrix.index),
            default=list(st.session_state.decision_matrix.index)[:min(3, num_alternatives)]
        )
        
        if selected_alts:
            comparison_data = []
            for alt in selected_alts:
                idx = list(st.session_state.decision_matrix.index).index(alt)
                comparison_data.append({
                    'Alternative': alt,
                    'Proximité': topsis['proximity'][idx],
                    'Rang': topsis['ranking'][idx],
                    'Distance PIS': topsis['S_plus'][idx],
                    'Distance NIS': topsis['S_minus'][idx]
                })
            
            comparison_df = pd.DataFrame(comparison_data)
            st.dataframe(
                comparison_df.style.format({
                    'Proximité': '{:.4f}',
                    'Distance PIS': '{:.4f}',
                    'Distance NIS': '{:.4f}'
                }),
                use_container_width=True
            )
            
            # Performances par critère pour les alternatives sélectionnées
            st.markdown("#### 📊 Performances Pondérées par Critère")
            perf_data = {}
            for alt in selected_alts:
                idx = list(st.session_state.decision_matrix.index).index(alt)
                perf_data[alt] = topsis['weighted_matrix'][idx]
            
            perf_df = pd.DataFrame(
                perf_data,
                index=[f'C{i+1}' for i in range(len(st.session_state.decision_matrix.columns))]
            )
            st.bar_chart(perf_df)
        
        # Statistiques globales
        st.markdown("### 📉 Statistiques Globales")
        col_a, col_b, col_c, col_d = st.columns(4)
        
        with col_a:
            st.metric("Proximité Moyenne", f"{np.mean(topsis['proximity']):.4f}")
        with col_b:
            st.metric("Écart-type", f"{np.std(topsis['proximity']):.4f}")
        with col_c:
            st.metric("Minimum", f"{np.min(topsis['proximity']):.4f}")
        with col_d:
            st.metric("Maximum", f"{np.max(topsis['proximity']):.4f}")
        
        # Analyse de la distribution
        st.markdown("### 📊 Distribution des Proximités")
        bins = np.linspace(0, 1, 11)
        hist, _ = np.histogram(topsis['proximity'], bins=bins)
        hist_df = pd.DataFrame({
            'Fréquence': hist
        }, index=[f'{bins[i]:.1f}-{bins[i+1]:.1f}' for i in range(len(bins)-1)])
        st.bar_chart(hist_df)
        
        # Téléchargement des résultats
        st.markdown("### 💾 Exporter les Résultats")
        
        col_x, col_y, col_z = st.columns(3)
        
        with col_x:
            csv = ranking_df.to_csv(index=False)
            st.download_button(
                label="📥 Classement (CSV)",
                data=csv,
                file_name="topsis_ranking.csv",
                mime="text/csv",
                use_container_width=True
            )
        
        with col_y:
            weights_csv = pd.DataFrame({
                'Critère': st.session_state.decision_matrix.columns,
                'Poids_Entropie': results['w_entropy'],
                'Poids_Subjectif': results['w_subjective'],
                'Poids_Combiné': results['w_combined']
            }).to_csv(index=False)
            st.download_button(
                label="📥 Poids (CSV)",
                data=weights_csv,
                file_name="topsis_weights.csv",
                mime="text/csv",
                use_container_width=True
            )
        
        with col_z:
            # Export complet
            all_data = ranking_df.copy()
            all_data['Matrice_Originale'] = [
                str(st.session_state.decision_matrix.iloc[i].to_dict())
                for i in range(len(st.session_state.decision_matrix))
            ]
            export_csv = all_data.to_csv(index=False)
            st.download_button(
                label="📥 Export Complet (CSV)",
                data=export_csv,
                file_name="topsis_export_complet.csv",
                mime="text/csv",
                use_container_width=True
            )
    
    else:
        st.info("👆 Veuillez d'abord lancer les calculs dans l'onglet 'Calculs'")
        st.markdown("""
        ### 📋 Instructions:
        1. Allez dans l'onglet **Critères** pour définir vos critères
        2. Allez dans l'onglet **Matrice de Décision** pour saisir les données
        3. Cliquez sur **Lancer les calculs** dans l'onglet **Calculs**
        4. Revenez ici pour voir les résultats
        """)

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: gray;'>
    <p>📊 <b>Application TOPSIS avec Entropie et Poids Hiérarchiques</b></p>
    <p>Méthode combinant poids objectifs (entropie) et subjectifs pour une prise de décision optimale</p>
    <p style='font-size: 0.8em;'>Développé avec Streamlit | Calculs basés sur NumPy et Pandas</p>
</div>
""", unsafe_allow_html=True)
