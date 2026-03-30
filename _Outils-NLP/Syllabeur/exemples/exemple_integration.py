"""Exemple d'intégration — Lectura Syllabeur.

Montre comment brancher un phonémiseur custom
et utiliser les données structurées dans une application.
"""

from lectura_syllabeur import LecturaSyllabeur, Phonemizer


# ══════════════════════════════════════════════════════════════════════════════
# 1. Phonémiseur custom (n'importe quel objet avec .phonemize)
# ══════════════════════════════════════════════════════════════════════════════

class MonPhonemiseur:
    """Exemple de phonémiseur custom basé sur un dictionnaire."""

    def __init__(self):
        self.lexique = {
            "bonjour": "bɔ̃ʒuʁ",
            "monde": "mɔ̃d",
            "chocolat": "ʃɔkɔla",
            "maison": "mɛzɔ̃",
        }

    def phonemize(self, word: str) -> str:
        return self.lexique.get(word.lower(), "")


syl = LecturaSyllabeur(phonemizer=MonPhonemiseur())
r = syl.analyze("chocolat")
print("Custom phonemizer:", r.format_simple())


# ══════════════════════════════════════════════════════════════════════════════
# 2. Intégration avec Lectura G2P (si acheté séparément)
# ══════════════════════════════════════════════════════════════════════════════

# from lectura_g2p import LecturaG2P
#
# g2p = LecturaG2P("modele/g2p_model_crf.json",
#                    corrections_path="modele/g2p_corrections_crf.json")
#
# # LecturaG2P a une méthode .predict() → adapté automatiquement
# syl = LecturaSyllabeur(phonemizer=g2p)
# r = syl.analyze("extraordinaire")
# print(r.format_detail())


# ══════════════════════════════════════════════════════════════════════════════
# 3. Utilisation des spans pour le surlignage (GUI / web)
# ══════════════════════════════════════════════════════════════════════════════

def highlight_syllables(word: str, result) -> str:
    """Génère du HTML avec chaque syllabe colorée."""
    colors = ["#FF6B6B", "#4ECDC4", "#45B7D1", "#96CEB4", "#FFEAA7"]
    html_parts = []

    for i, s in enumerate(result.syllabes):
        color = colors[i % len(colors)]
        start, end = s.span
        text = word[start:end] if start < end else s.ortho
        html_parts.append(f'<span style="color:{color}">{text}</span>')

    return "".join(html_parts)


syl_espeak = LecturaSyllabeur()
r = syl_espeak.analyze("extraordinaire")
print("\nHTML surlignage:", highlight_syllables("extraordinaire", r))


# ══════════════════════════════════════════════════════════════════════════════
# 4. Compteur de syllabes pour la poésie
# ══════════════════════════════════════════════════════════════════════════════

def compter_syllabes_vers(vers: str) -> int:
    """Compte le nombre de syllabes dans un vers."""
    syl_espeak = LecturaSyllabeur()
    results = syl_espeak.analyze_text(vers)
    return sum(r.nb_syllabes for r in results)


vers = "Je suis le ténébreux le veuf l'inconsolé"
n = compter_syllabes_vers(vers)
print(f"\n«{vers}» → {n} syllabes")
