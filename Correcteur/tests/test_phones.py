"""Tests pour l'estimateur de phone par regles."""

from lectura_correcteur._phones import estimer_phone, generer_phones_d1


# --- Tests estimer_phone ---

def test_estimer_phone_chat():
    phone = estimer_phone("chat")
    assert "ʃ" in phone
    assert "a" in phone


def test_estimer_phone_pharmacie():
    phone = estimer_phone("pharmacie")
    assert phone.startswith("f")  # ph -> f


def test_estimer_phone_maison():
    phone = estimer_phone("maison")
    assert "m" in phone
    assert "ɛ" in phone or "e" in phone  # ai -> ɛ
    assert "ɔ̃" in phone  # on -> ɔ̃


def test_estimer_phone_gnou():
    phone = estimer_phone("gnou")
    assert "ɲ" in phone  # gn -> ɲ
    assert "u" in phone  # ou -> u


def test_estimer_phone_chapeau():
    phone = estimer_phone("chapeau")
    assert "ʃ" in phone  # ch -> ʃ
    assert "o" in phone  # eau -> o


def test_estimer_phone_enfant():
    phone = estimer_phone("enfant")
    assert "ɑ̃" in phone  # en/an -> ɑ̃


def test_estimer_phone_oiseau():
    phone = estimer_phone("oiseau")
    assert "wa" in phone  # oi -> wa
    assert "o" in phone  # eau -> o


def test_estimer_phone_garcon():
    phone = estimer_phone("garçon")
    assert "s" in phone  # ç -> s (via simples)
    assert "ɔ̃" in phone  # on -> ɔ̃


def test_estimer_phone_s_intervocalique():
    """s entre voyelles -> z."""
    phone = estimer_phone("maison")
    assert "z" in phone


def test_estimer_phone_c_devant_e():
    """c devant e/i -> s."""
    phone = estimer_phone("ceci")
    assert phone.count("s") >= 1


def test_estimer_phone_c_devant_a():
    """c devant a -> k."""
    phone = estimer_phone("car")
    assert "k" in phone


def test_estimer_phone_g_devant_e():
    """g devant e/i -> ʒ."""
    phone = estimer_phone("geste")
    assert "ʒ" in phone


def test_estimer_phone_gu_devant_e():
    """gu devant e/i -> g."""
    phone = estimer_phone("guerre")
    assert "g" in phone


def test_estimer_phone_consonnes_muettes():
    """Consonnes finales muettes (sauf CaReFuL)."""
    phone_chat = estimer_phone("chat")
    # 't' final devrait etre muet
    assert not phone_chat.endswith("t")

    phone_vert = estimer_phone("vert")
    # 't' final muet
    assert not phone_vert.endswith("t")


def test_estimer_phone_consonne_finale_non_muette():
    """c, r, f, l finaux ne sont pas muets."""
    phone_sac = estimer_phone("sac")
    assert "k" in phone_sac

    phone_mer = estimer_phone("mer")
    assert "ʁ" in phone_mer


def test_estimer_phone_qu():
    phone = estimer_phone("quand")
    assert "k" in phone


def test_estimer_phone_retourne_string():
    """Le resultat est une chaine non vide pour des mots normaux."""
    assert isinstance(estimer_phone("bonjour"), str)
    assert len(estimer_phone("bonjour")) > 0


def test_estimer_phone_mot_vide():
    assert estimer_phone("") == ""


def test_estimer_phone_nasales_pas_devant_voyelle():
    """'animal' : le 'an' n'est pas nasal car suivi de 'i'."""
    phone = estimer_phone("animal")
    # Devrait avoir 'a' + 'n' separes, pas 'ɑ̃'
    assert "ɑ̃" not in phone


# --- Tests de pipeline : estimer_phone + generer_phones_d1 ---

def test_pipeline_farmasi():
    """estimer_phone('farmasi') suivi de generer_phones_d1 produit des variantes."""
    phone = estimer_phone("farmasi")
    assert len(phone) > 0
    variantes = generer_phones_d1(phone)
    assert len(variantes) > 0
    # Les variantes sont differentes de l'original
    assert phone not in variantes
