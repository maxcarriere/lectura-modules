"""Lectura Pseudo-Ortho — IPA vers pseudo-orthographe lisible, zero dependance.

Convertit une syllabe IPA en pseudo-orthographe "lisible" pour injection
dans un TTS sans entree phonemique. Strategie en 2 niveaux :

  1. Lookup lexical : table de 3493 entrees (Lexique383, monosyllabiques homophones)
  2. Regles fallback : conversion deterministe phoneme → grapheme

Usage :
    from lectura_pseudo_ortho import LecturaPseudoOrtho

    p2g = LecturaPseudoOrtho()
    p2g.predict("kɑ̃")           # → "quand"  (lookup)
    p2g.predict("tʁa")          # → "tra"    (regles)
    p2g.predict_word(["pa", "pi", "jɔ̃"])  # → "pas pi yon"

Copyright (c) 2025 Lectura — Licence CC BY-SA 4.0.
"""

from __future__ import annotations

import base64
import json
import unicodedata
import zlib
from pathlib import Path

__version__ = "1.0.0"


# ══════════════════════════════════════════════════════════════════════════════
# Utilitaires IPA
# ══════════════════════════════════════════════════════════════════════════════

def iter_phonemes(ipa: str) -> list[str]:
    """Segmente une chaine IPA en phonemes individuels (combining marks groupees).

    Exemples :
        >>> iter_phonemes("ʃa")
        ['ʃ', 'a']
        >>> iter_phonemes("ɑ̃")
        ['ɑ̃']
    """
    if not ipa:
        return []
    phonemes: list[str] = []
    current = ""
    for ch in ipa:
        cat = unicodedata.category(ch)
        if cat.startswith("M"):
            current += ch
        else:
            if current:
                phonemes.append(current)
            current = ch
    if current:
        phonemes.append(current)
    return phonemes


# ══════════════════════════════════════════════════════════════════════════════
# Regles P2G deterministes (fallback)
# ══════════════════════════════════════════════════════════════════════════════

_PHONEME_TO_GRAPHEME: dict[str, str] = {
    # Voyelles orales
    "a": "a", "ɑ": "a", "e": "é", "ɛ": "è", "i": "i",
    "o": "au", "ɔ": "o", "u": "ou", "y": "u",
    "ø": "eu", "œ": "eu", "ə": "eu",
    # Voyelles nasales
    "ɑ̃": "an", "ɛ̃": "in", "ɔ̃": "on", "œ̃": "un",
    # Semi-voyelles
    "j": "y", "w": "ou", "ɥ": "u",
    # Consonnes
    "p": "p", "b": "b", "t": "t", "d": "d",
    "k": "k", "ɡ": "g", "g": "g",
    "f": "f", "v": "v", "s": "s", "z": "z",
    "ʃ": "ch", "ʒ": "j", "m": "m", "n": "n",
    "ɲ": "gn", "ŋ": "ng", "l": "l", "ʁ": "r",
}


def _apply_rules(ipa: str) -> str:
    """Convertit une syllabe IPA en orthographe par regles deterministes."""
    parts: list[str] = []
    for ph in iter_phonemes(ipa):
        parts.append(_PHONEME_TO_GRAPHEME.get(ph, ph))
    return "".join(parts)


# ══════════════════════════════════════════════════════════════════════════════
# Table P2G embarquee (zlib + base85)
# ══════════════════════════════════════════════════════════════════════════════

_P2G_TABLE_B85 = (
    "c-m#XyN={Ywx;_kgHg6JI1g~%#>3s|LFct;MrJt&BQ!#6=|rVc2xYQ?!pUl(kJ3T|TMU$th5b-t<p2GP"
    "-hGseT)*4>vTWVV_&@*q>A(Nq|E>J<KgvHZ`+r~b|5ikM-O0;NUH1Q1`~R!`|M&g>bxX2)+UZBysb(Wy"
    "Pk);AW1}B`g33E}&X9lnd80r7dY%Nd%3MlTVePo5Z%5Oa1QgKz_@)H6nG~H0?DTu5+B^N$>$hINjVfv!"
    "A=R{zl1Ddfq-iQi1*J|zex;t20Zb})`OieV)bP*0k|MkKO!13=*V6*9ssIrbH)W_711~QBk3V0Of4T(H"
    "D9ssCjh?#roe`KuSMvK`KyJOEKi(+h7357xf__&&Zaew1FRT|0(~H*Dix$@lt84$e(-)DnxBN$CmH)V~"
    "N_15@$-eSl9{k+s=QEUwFN>+=cj{|R_oDkAI?>o<ngSDx_eHZ-{^V<)K{WCkHj!u%ut2U90;w^5Io({V"
    "mTUoBiA$VJGHlaJBw(n@+!~>&)1Qt~JbaUfPjj`6$g2^O+;{T26Z)@4_Vc&LPSzS{{a?+gU!Yn@MzvPk"
    "YMNGS&gwE}qE&G1E&Wof<+U4LQ=W_KM0TPyW#!LKjiN?Tn!cK1nY=nCuh!(%*(Bp!XAN|{l7Zh>YF??D"
    "iL3jY=pWx)z>S%&4Fs(LuRY#(ZHLE6l*aq<`J2c=i3d?X{-6JPOC;|T*jEkCV`u?Bl;>1C#Rubktqd@p"
    "S1<blxt*Wir@;P;wq7;1t(w}NpqS>?yR{W(eQC{G3zA5Zpjp{%EbZ27yJc~=UV$x3u)R{M#-!E4D}O{a"
    "kPS3#V0E;ffVQ3tP<&1C8sm1*?rmqt8FDqvB4{51`2FGXXUhNlr|2sQe*f7iu1>PeBVFvz2jsD0YTF-Q"
    "soEr69h~QD{@14e`08?p{&GN<!05q`U;ed;ESdJ|az4m%btxa7*}r*ae|*r~t9fIAvdhT<F?^@`NaaSd"
    "WxH5>U0HlPq$#<c)V={GJ!_c_9VAT|;L)>&ZD5C;=JP869Bqc_X@-Ycy`3vav!nSAVCm&$Za~hEt!Z<t"
    "*@$y=&{~Caf+V&1OlbqVZ4TBk%;p_)Ak)pj#mFVnxa=tfM31+O#@^iIS0heCk}ulb18n092ix%Z_v{3#"
    "Qy6dq&E39SDf9C;fZAgzlFc#9*ZC(Qu{}0Sv;QTQNXe%)qCKbO@TfG{Wx^Qk2S6MkkM;$G^YNiWVx}|="
    "8ME<_WL8Kz`heE@c&z-&k3YUq+s(DpfqAk7@bIbnpJlXFi93FsD*?a%tTjGERq;CjrItZqa5tzi)&Mb3"
    "bSxN)`N{?$7<l(@5RyUfb8y2fb8wbz&>vA4iw2G~KxmMM7f8BLL9&e=knEvzW14mQlZ$2xwUF$f4pI|1"
    "STOtXmF(s|W`;xIJqd&^4L)mreDzwJE&*JOc=k4s+NM6rvgBL9TeQKuDyBI&@X!a%D0M7Fm@g9#>pj){"
    "_<@l1oniZ6#+gn|?l`y$z&^gVH$!UtHHHdKU084mq@kL`bilel|CXI(btVIh=I8eqPV;TE?G1P_@)8j3"
    "nffDJ2FkAoM)_Bm-3GaDQ}oR{>u={HCyDAv<V^tc`0=YnmD!@ok6*>9f>HhWRh(*u{r&N0vBT*UG-XUK"
    "qd<3%I4KI&fO&5f$QdJF56~M(t}I=nz6ETFa_H%VVD!hI)rkqo=#M|E4^#93xqPZ$qF7|{$-?-CxZK)W"
    "RRQV7CWA*-`cb<dh}Tc`!Ahs?KR~OID4JC>A(8#@XC3zwa?Kxqo}p(K!jU{Zoyi6440cA6krcIYtW860"
    "L}$q~o1_?v69r*<=01>-e$Ats#2GTpDS*bvG4&7U%YnG}vwP3cFkRguj)rG|6+O*fF~RANMYKU)X?={+"
    "6Ca;;M(*~)%cmh|kKQqO!QrOC`)vbV_0T^-tFiStVdVdps(<;n4#EY?Mmo}<wM=L&8^QozPfb8{`15b="
    "Q_d0a`xsC%bfjWxLRj6f#mSZfTiv$wRE1~pWYK2_-<<tvw9JjBIz|g#GtG5AATPQ)R$T#Ou~5yvc`yI5"
    "|94;5my`Z4=5&EKU1T`9$S`*ipSqkLq8d?+C^E9IF4E!3AIKiEhaN?bqLQeyM&=@!vHT%1a&Gz0Bw#VH"
    "0Ep>dtPWmIPB2|e)1avXco)6%z{}17F%Mm?O8WVml3o*}e%%NCq9MOvquBrMv<9qJt+G&Te|Ad4S8Mph"
    "8846K3aLKV7%VU1l9%g=f;E7je`f&2GcWZ|P%14<5*JwL1rB<tA(3h>^&)y9`ATY}mkTzE3pR=iHi}Dq"
    "a|0Oi+C$DFyLgGAAYY^(Uu2-T)F&tpoK3(5VnkR`cb3$H5xBF4>QnCwUKbB94B0&{*gY<?dz7DN$YeSJ"
    "tOiyAv*>{g52TJuJx{JxeD+dbNn;jTO6d>8HTBdaHd#tG*|_aPGa2hZI)D#)^WAhZ;8H)B*3a)On=rv_"
    "sa)`fTPWUfX*3s&=AskTnd%{xw$qSy3~8t3-_=kgrr_=iWeqc>#ppzNrOmXk=;_Q`uWY?CY1Rhyad^=_"
    "ohA2P1~%tE;4I9kr*v@P|8bga5x+3agASxiVx=WkTA~+;UL+q)moymh5IT((${_XgGP9b8*{fNi#zm5b"
    "S|PEbKxdL<Xr^vnpciML+4D*Qx;E&{zo3m@LL0v@{3zdr>`CA~gYvy0D?k(~!`Vg3_hnPQMb|m%sk4kB"
    "m!exIH@~dLZ1BG#rG{x2k}ldQ4~@F49~UXx7a3y;l_6D>WT(rF;&gfFyqM)bxJYWcSU$T*xxTQAk&P}4"
    "6B`y&!R7a#@6ZxJ3ortV&^}(SGv|xd=nJ-mi$$r6B&ka*b8q;A2NdKQkWZanOcAayKxlx9Zkr6SWCCEB"
    "wUF8i@C%gNi<R2~sAI!uIz8_!RNafrei!Sx7wfqf>$#U<$^lTvM8CaLhCT;CH)K255!%s@f#m;RF4r!@"
    "!9~K|1uhFKG~+{tyIf@k*k}&v?@gqzYwU9U3G>(mL9ZlZ1V7*=-*HuKNN<QPejvNl9@dZaZ;fFq#75W<"
    ">fW#kP^qrAEL?(BU#FlufbJTQ+v6ekCOg$#0EitJXLiWvb05?BtGENK8Qt~beU3PJL7Lp?{O^D_R!xI{"
    "6|0_Ez`rt0Gbh#tu+6U;aP;BJbQ+U^79>)10E|IF=*txHU1+fvwAc$;>;*0Mf);z3YLoEjqQ$<LB^R^;"
    "tPI=%EXi7lEl^`10hs?32>EEjiE{}B^@4(WK|w7oo=wxD&woT+9B<PQou2`miIb%O4-mM1(M@w8IUtuK"
    "Ry@T3VqGi<Jl!hf1%>LeX^bBE>wF8${W3E=)W{2J<i&d7MS9^*j{svJE_^tqH5dw7;l*m+Q>$2I6RRxr"
    "vXbPOkk}>@_8TzIW$(!>%lg7VV9dDu@vafQpG@w#?kCty0agL)3ZTVz_u_k(NO8Y<Jl^F|ykD<${~dw&"
    "{%|D&b<)lE<2b#?c?x=A3icl^%)+~7`0o6_|E#rizsj<)L(Uj;_s&G!hoE<f>n?lj4n#BD2xj2!8F)8C"
    "E=UtlZ|KODH3<7U-G_|-?zumI+w)4_LfxwNQL0Y*zqW8?ouQI`TK+)ZMSAa6nzu@R`|3pQ7Aiw#(feps"
    "*RCUPub+ESiIVs+kvYN4E3Js}6>ofnTvdR{DX_a&Cz;(hfKn$HE2dV7r&aq1RoE7*ou5_O<5f*uM){A+"
    "*D|i8dzL>d<SM#yhH9cbjaCen*j!bgq13keCw^^74b?$*(AmYe+2t$A=ifUwkP=&y8begnO2<vLIf_Ui"
    "{!rBpxv6SbVjszd&=QW!3fFjL*Z2lR<Ib;cowi?9%P?D6A+16oEl}fZr*q^=jq{;u7mR+K)FdgPRjpIK"
    "k`h{>gjRR?_$p4sD)h&apFSN5<c6^{?4hsfp$!DLL#ab{jtZTz<f~oQ2ic-NG0VzY<c2Kn=;!yvJ+*Wa"
    "9g*L@{K=4vm>u`!KMjDby%oOv?Jv{Q5-U>vlU|LygJ{{yQ(xh!ujHw(<f*Ul)R(_!sz4*J+J{bvO0M_{"
    "e|9BTe1$8%lH<LgKS5?{`_D`Z9}6iKXa)k`eAUXbTD7jPXbc9Ht08A9oR)^hOxs^y;hV4Io3HGfukg)R"
    "_RVj7aqBFV7i<d11&c#hxaN-*vMgFL?2Tr7x0aYId*@50n5$R=?WvvudUH<iu=bEQ%e+|NEHFoF899#D"
    "DhvV@27yYGb&`?38sNGZ&8q<pX;rTpveQE^(2Trv{R)dtm7)s?dq}0_z-pK|N(+5qI3-|Jhm)XU%i1kp"
    "=*B>-bZPU|z@oEnz6#%bH8VoP>;al1&19OvEr<tK(p`m4Us++V>@BaPuvb>tD>=(6oaKe=O*5un>Zq4l"
    "{lfYfW}U}X$R4E{O_O#Fv-E(6Y2Qm5ueK?IgFWbK^we7@oZAJ_rk}Ov*~6^Y&wqOOt}%R{|6D;dvur|g"
    "pw(B`!{AadPMGV=5fi&I+~Jiq`AV9673=e=*q0sfA}JWz4i=5Zzav!F=Od9%^!^5>9X~)c5)uX<Cg$>k"
    "*B)AZ#m+{Duk2l^f*qiw??4CA9cjpBNn?Yu^BoYa9j0ilmw`v)>oFR`F&bYz8bg0RAS#IA_~m)VSaLq7"
    "etcNCbG%=LP+1{UZge)_Ovo7DJ80O4m%0zTt39HXwe`x{dWBTFwItFkBv}rGPH!h95Qu#@Rp{^4HAWt{"
    "a>;fF@M+%%9jmK82Gp8Lg`%<}*~vVhQrRnZx;6Gf9|^qi*y$+X)pRo1K%Wit0KI_`fcWJNO3syn0TvVy"
    "c(M!}T6%?+UZJH|Xz3MNdNr{MtZWy+?Lmy3>du8CN8&Hufb2r92joO~HMBZg6vW_6p1}=zbN#Qz2D&ht"
    "z8JVTiOv+1^vW*e>Tn@fR?`cHgpNvHP0sY7>vIDDQ=Bke?#OD$-3o@a`DN1+=vMZZC(!fem>u9`__=_X"
    "JM%7U^_8^xou<Iy9Eh8PaS!Zb@N#`ZJxt~r*7hs3{mR;YHJxI*Ez<7p+Q3-0S8x5@EI{k7&&$;NJzTa`"
    "xM!=wJzE{_*;3K>=}77RUF-e{yTn>TA+N$iTWucNz0jT9%{B_DP^7bjq#F05EK;&ZX*0ZPFGvHE0#?gP"
    "`TD@2sxw{<()M_@aSAmHElXSFRoktGsnrPm+9CC-|952L<^h|9ts?bFwhQY{4-|pB)biJd1MYF{Jx*A*"
    "aqZ#RyGyrzocL`KwTM!!kJEHX2DC*V+f4zWcL6((k@|%T@U9WB^?#a)v-YmBcxWu{^zR`?DXk5q?Jlca"
    "q4r}&UP7FN_nCvV(43_AeP2KHGm_6kA|ATRgB^WmKNnh)b^rb67NCVDKL(pn7x6Y8x}trJcSYeb!u?I<"
    "XyeNLSO~4SL3`1v<pa&qo=6;#ca)GaeXkjSMh#QY4M1Z=tsIymvxRzaD+^F5NnZ*1g^r5)5OpHI;y8-Z"
    "N@DSV2bTOhlE4E=;0`}{EkIAFBnnvVwePghbV@9s&?_|eWYB-ZjfxjlPSmqjZNnBgfBqHer69_GKD|sJ"
    "7tt4!PH=I8N^~XaF3M_1qg`}%NLZs?bUyj#mfzbWK?l+S^rG<)H9FeQ3d3l86Y0EPMWSDcUM6xRDoIfZ"
    "qWYiaQ=S5PQ!4z0jW+d}Ahbr3!ujMvWjt?QKmU4j%RDxdNd|AaL;^bJYepMH|E8l9ddSVA4ap$c2yNxS"
    "Z(t|qlTN+nO7C?3G_}ZD#NMh#Cwe2+-H3NLvakK=P)HiF^QM*=xe-He#L>&244FlbA~iQ+>W$lwDVh_k"
    "Q8>r;XQx(vUPUszxF#AnT_aAnks5*mB<2e-s1)H*5yNl9@tcUfXdHXdi0?Kq-bM_xfrB>UnGH;{Dce6W"
    "7(=Wi_I#;ST4>{4-FRC!-nAu-LmnF0r5fEQX>68iY?f*;OP$-;I9R0$(is>5M(70Xb>C_Q?|((pf9{(!"
    "wqi9{u^L&i&RDS;S+N>fu^Oyc=eCYE)ii+#FMks`<SHxw(MhiV2r*z<hqnH+BZCSZC_<uZ=@+7F)5_pd"
    "QuGE^X91X@XUdqJ<BRxU`-KT0=wky|9VOA{-^~?dcx!ayrBF0=rlk++on7z4BRn(0zyJI~k{1I`a&;OX"
    "kEM&WPA6X@6I){wTWML0Jg%t%je$h_rvsv3$1pZn*&3{D4OX^NHSvdy@3J&4r%V$jwX;oXO<ZL;bCso0"
    "L{6iKobsoEY@mB+QoFTHr}MZVYFJ0Zj+mP!Vs4sV3$m}F8psAZL8~XdwsX&R?q%*Ouq;B2)|tkbCH46a"
    "mejx*VqK__4X-3%2g`=noMdoqSb-<6gn_o7nB=74NgAFczLVHzN!>~{6x<*o^p*cy04~rCbOYT%chEg_"
    "51pVBG_6ZaXMzzXbHW5GOy-3Nd2&an1;MaDS0`fOCJAjjpmnlAG=`F0u$XLu$tG>M1Qoj_dAB6*mQn%*"
    "9(A}1)>3l2w8+bNe>dG4XpM|UMkB5oBWx29RmZ^o_|;=DM_s>0>9dO$I~o+M)65r0_?hjw=dkBC3h-`-"
    "?{4PN1@QCO8bFYe^4^%?X{nzFI|v#DYMlp6&dnTEPRYQFk(Wr5B2hp{8gap@A*AaZAzh7_>!0SKhP==?"
    "1?3h%LgV0jumx!-LpGg;Xn?bW0UDyf8b^UOHY#s<rb|Gwp%l?)%A~uJkEJbBbAze5kgmC=X87NEGc3FE"
    "jTs?bTyt_fcM~S(GbZO|U%qE4Ld`=XN*G`x3-lR_d_(+Lp{a$_g*jI^Su<0v=9lS}x99B1<kUIOQ_vc5"
    "-<4~$N=xHs(;7`oGK~z=4Qm*~^m)F99Pf>I9=s84gmw$PFU!!&db1;qW#S%p<c=cXcH;r5f9Qiv7^WK>"
    "j18P+fMBh@K(eFg?T@dOTU%AGmNRtt6C^vRHOZv6u1@0CfM2P>rhRs#TH_N~u^|mc?(<-)ZG#GW9<nn?"
    "race0D8kL0#uNyBnJz~ns`j0%CXC|+OGh&5oAQy2)#;sSR(Xa~#wD$@FKG#CjkQl8{BY=BrT`gKo93D@"
    "kTrv+(nj@C+?bD47Pp2zPX{U|OhZH+P^7$B#S?@7lbNoC@KX(m!q-r+81kR+UvG3{AQ~>^yCL#pqb~+n"
    "&L7AGwo)}m%v=*Ob4^J5zEIZ0N#cWK0~#X@kath*1^>atk_}oZN#Sq}LM!2LjaeJEyED;m8=U}-Rf)m<"
    "Y=irTEM_3Ksx@JSKVyY&j&m)JwGB#!Qb6Vlutd!Fjm-BOsTDV!SaAiZGD)gzs3DOj$WeDCk!)~{?f&P<"
    "ua-2vPj;R{3o4KckFwboQ|Qmw_)AH@8PHH-ux}z3uOQwOHvFHb`e-By@)nC{q$5!mX-3kF0ccL9hj)lQ"
    "2ImOaYvPXC`PG-q8v79%oDq#(24}krns6C31obs|2hP)QcIO8AcKYI0OWZ3}4K`I2EHRL?5eaYj5H@<G"
    "WJ@Y`D{N^Z6P-Pz6FV9T7UO*=WYd#Ew&4NS0qWSD+>8ffH^;J%+h+x|Kevg1!t*pAD?w-F*&dAM2p?=5"
    "KG+;Cj5AIrAo{-@ZLXM;Oa6zQegSw1gmp9=>&WLx+)3Lpk6g|N<}M6J7eXQ?BC)ZLqQOVe*hg`mxPQAN"
    "tkf;EI52wj`M2@FvazEXA8tm3STu*@VXLK2PoO2<4Zlp!$#dW;dtaFCe(g&}6WL5!F_H2|OvEM>TToh?"
    "GTW1b&02|}t%T6_R5?L43Al@Vh-yTgp`5rOaVN5)b{RfeLWHr!gej&K3u;e}qi(&pTTD^dYFey~Hr2F<"
    "wylkct&NGTjfrg-6E}!brsaTiq!x_8hBh9<Y5f?E1Dx7;49BJ+EJHF3VH0c>?bs6TC@T^6oYp~pZH)=J"
    "mGauwdQqd5!Lj_w<kmn-mE{eS7GBv(k!|shVQ(sbM$-v^vLHR|+%en#iq6n1t{OT>H{SPfT3IPuE}gYL"
    "DqEkFt@v#l*U)UxvOTNCl-c?kn%`w<VPCDxl&!9rwK$quJWYkBxb10~WGIoQGr~Zf`bfiA#2U0;sz59%"
    "d!Y)#I*=Wp)b*Rlfof*R)$}R=rrP2va$H->O|v$9K^sC6bI#VmgYDrEYV8qfF(tIJB((PLv^aR$uq?E-"
    "cwCDqwsjm~YX?kg4@_(0Y-{6eYp2VGL@|uqHAXH%DcU32unliZNjW`X?@o9N%q`VxLNag?>W4A2J<a^K"
    "Lat2Yti-Y%w0e)F9R@9E*)_|r%B<tJ_37I>qN|OFu2zPC)`kEc`6;w!bdC(H5kI`r#=|Qu3V$04Ki1S%"
    "`hKC{#l!5t)E4!<wff%r7%!Q$E~KZo*3(<*>4la=&=M2YI<C9`F0!V`BLHBlr@t|R*66pgWzuXzac@!F"
    "W$2Ziw|&t5nX}|!o;FMKX&)MrSwr+tTw*&b?6`f9o6K5qpa1FIE5*M355fgo-4kn>;0I-$`4J8_!@(Zb"
    "I0_OxO5vbowqax5l_BZ0$T=<dCR!rdT3*26<tB{6E!Ke+>p&Z~%Ua(q3o~|ving`~Y#rNy*ynCU1bDQ5"
    "k)@6Bjsi7-jGV3Gk%;q@n2R>-1FZws+T$UYlFUcl@dUt@_w~Tuosc1f+YAsAE(T-^<g?xU?&O+Pm=VSS"
    "LRoE`w{5;fmcygr0NIF-ZGEex9irV|y*s890Q-=Q88(3DG^}aw%Yv{SZU(&UrC_&V0BAP@KnlzaI!5yA"
    "vA24c_RI6Cf&H5M{`l~&`TWkd;fa@)Cth0g^!D1AM6e*fw{`Sf>*%=_9kfLSbpUXCWg_6_4yM}x-uP%Y"
    "Mg9)RatcAgVK=Rxg=rmgSA-qcoZUZ<lRXsMt6!4~S}11;AckR?A}3tFUISsouQvmh0BcXUACD;|O>UFr"
    "w(l|JAJJWR)~8P&^b0~(%=jUaw~htcK_=wU{}d1+nC?aH2*+iAlto`^FS3urinD=7`93UTMajU?%D~ZL"
    ";Anj&VBlzFyKTcr(fW~y0tILCqgwj{*_VfWRB};u98w($LnrN^lXlQa8R$;o2rr1z^AsZ<u1*R<$4jo="
    "=_iqtP0?z?iAMCIFeG*-a<UdX!WFs*SLmcYls~UDg>~1VJai6T=p4Mz5uVWbVa@L3EQl)I@lI;DKa^^a"
    "%BQFt^J90i`LXjQpaKODO`7simRse(@Gex3&Z<X;>d}Sj(Vd(G=B(@doe5Qq@}ax^T@CRf>P3`hR8F%_"
    "p$(m$vgo9BbkaJy+EdHiN(<@GLU=KtgQs-TGRhw+6{F}LC>31|%?nSRwTiCx&`5IWFwt~q3>_LnhsMyc"
    "f4bWFeV%U!l}xUbVXW<(=%iM3ihk%E@YXrvts~^E02gRl!8*P3kZ%E!2VeR@hnujIe$b&GbV&bQjp?Mj"
    "$2ncn&~ya*PJ;bLj?g+Sy9mVU>fz{&Pp%F>Vu!fj@zg};@T!hmL!CwY0(h(4&F8mVV)UKEGfFaR=Us^Q"
    "onNHrLW=K@;=ACi9jvv}Xh^Q_2Kb4D#7_3K&Tl|;1Dk3ltJahxgig-o&KE?w!J7*YPbZIYCy#L_muiPg"
    "wVPEG1bn><=VCYO6~Tg3W^S{Q1IR|y9cdaJnnpL%Ui-(p+386hwVL0cHFUeyph~1RbW$4%MI-#gTgO`v"
    "B|&e1T2R<KYZ0AofgRUEw%0sh3Ksu6IrIW#@&*pB?)jGLhI+Vm&7gxD@&e;Fh5*n(W@wFrg#C_DlW^a~"
    "pcUz!?8)zGc8V$KW+$ORoMi~|omH=nEq7(yKmXZ6xBfjlfq@lZ7w$W`{s_A$)Qb!7jq}bYYQZFT><A0M"
    "^8b$Ir-Y22)dCNJwSw--uy245qHAYkT^GtiXJw&#y4A9QQ-B;<c0zTbFm$J<N*IUUZ23?ax?N#7Qu^5P"
    "<B#2;GIS0s?_gsCgxWf|sb5*kUdkv8ofL)+g`tC&6&}cNj<n({^kAAL91GYQLdPAN9f;LCdGSV=Ye$%C"
    "7yiU<Amp$MRid*yu?u%%=essLv;Zv!#1IZPFLuGchZsK8kZ!m`d++r@Gk!7G_9J$Kr*K2OzzUeBBPA{E"
    "^Ai*)BjGR43mrCLCc(SEg}y*bTtHRnP*pm;M%nqSaWsF29@6=h;LeASV|hE2j?Px4?(iNKEE)-~y2hjG"
    "M<7%eYDR|w(GmFDZR&$}h9BUL$Avpz1=tbw^EzD)h#fz*2b7%A!(%D!drYbsC~p;Xp$Zha<$R>cP82KG"
    "Z&G%Cld?n3?~wDmEvyDsd7YgrILB;J!jE@!Tmjriyx)BVdYws+GxgNh+ZxqN^64e`^pf>^8UB08^}Pbu"
    "dfv_7|BBX__P$=-pGd&(YJ+@2y+Sm5-$Lpo>h~7)d#>c~e|H+vyS%rg-y`Ywauf8p33^%8${$Ftg8p<P"
    "z3Iu#g&*|X=<kugONU;e-zh;E_L=Fg`#Mwnox<;W9HKo@YP}5my&{QwiQB!zZ6fu0`$>Cl>i68#?`w*p"
    "q)t>PN<*qu*2@LZZ!Q20QB6r_N=jr-<YtKRm~MX=k{XYFUz36a8q%5$uN1dZoJ8{;(Y&vJd?Ue)1TLMj"
    "8C}OD_mawc8T)%n<$YY?@40x=^QK&nKcL4S&`SjGiSX|wg7=8vy<(R8>ldgMQe%aH-V?Cbhk3sDU5!4>"
    "^S#aU{T=&P{}>tr9YANG2j~sB;KAj7b-8f#9tpnpd;Pu5^L=d(WN&Ny2Au=)7@M*6A)WW!bm-gDUDMj0"
    "9_q$nobNYl|9%uHM6}l_+Yhh>iR`^Z_MUjheqi&?BxbM9-m8Abt^fRoEwg=MVJB90L>2aqD(p33c*lEP"
    "<nEu|VCx-**E?3PN0jad%(=Y;>3-1Diam03zZt;+s1;*-FJpU2z$%ily(cKH_igLmx2=1gFYn{|@_rYK"
    "d#xkg0_<mRZwbZyjP%n>?(ONL%+<Xl)?V^%FKM(#-0fM758De0pb2w3u1r!~yYA!Kb&rVKBjWZFaeEd$"
    "503Y|H{OSJySH__k8R;y;Y|?)Yk`#<;$`nWBk8aoAlT)DH@SBd8>)~b;F$3<YCX@W^$3*x;NP`29iG;E"
    "13EmdM}q7J9w_Y*A$!8nHrkWyaf?lhjk%9|+5KQAb8qpk=bBU>8=JlV6f0KR_Y!4$(0p;Smq6xeff<}W"
    "|IP^7e%#xB+#{qFfaZNLlNIEhpC8Z~WXWv3&C9*b%RN$Dk1@H>io+?u=teNnbvzW%+lJ8FgwWeo-1~lZ"
    "j~LfqgY-EZp)5TzT#pRbA2M8lZjs0<ajH@vu5HQzEru3D+Bbe1yblhEytV@xzehK|dUj=i?dMa_-hej8"
    "^+;>|_JX((@ecye`Y_D)Hq7>pl<sHGcL;R7*_5DU%*zBRujh%jJ|fZjIcDzH7^oS>x^s4&ih7UJ&v6d%"
    "8Lge#*Y_)Lm&qz~np;er;`y&}4Pf94TPV|J`6sg;KDgoUZH)V1jvC96vwlSC9VLa1l0rwtM31^iH%cZQ"
    "<09QCsbJ(euaR$sjs95JC@Eq5{HxRJ9^Ll$O7*<}IK(L7wETfgv^7Cffo0LrvgjC}<r*c8j!2^;m+9r<"
    "9Wg<VHbIYmUP`x*MoXe2-{2Y@(K4QRieR*%dGu>jV|<3IKoY@51lSP)c4Wbh2(aV6SjYcM|JRUz{-!bC"
    "SM4b2W0bjgtocdPA!&A$G&{y@NS+-n&yG6g$J#$@s4-iOn}tdAe^8Z%zGmoae)iBKvyKi^8Efba#X#Dg"
    "JR+5j4lx;Pryzl(&QZ4D5zFt0<+spumc@t2IU;fvIz^SoN83h8q9cJlqa@L>9x^`<49W>x)QI>w5*;%7"
    "D)9!;^?RJg8@Nn?0$NTTLryK__aHhHN2JoxQfX<D*L>KP$J*A3F(lKGNRko3bi}kgS_mCO2pw%T9^+2m"
    "262UX%Xn)IMt7zu60$@iNMsBFc0_<3740_ibk#`o+DQ1?NPyZHHw4F)4c_u_?no3``Fjqa1Ce$#5}3CA"
    "brR2Xdu($DgO&f#T*ZTpTofD?0yehnBx1V8xG6Xi$Tha^97{ezwMIupjeM<YBo=9GAFf2}i@2Gw9o(0h"
    "zUXsC<p&9NBVj28sHx*2dZw8VO5pnkBf%F(0@iDbb#((`ctYG9BN$=>jm7fhQr~#QI&4r3Oo)yfYF7Ic"
    "w=lNHrwc|$B8)t?HHJff8;+L)@_gA<KjNz&an_G`>r3g#TcbaHRUjU|8p9($hD(05i@lU&_GW+jmMRiu"
    "BT>4?$ZyY#;Xv8TI-tvu(nfgwkOTH=Meryy#uQ7T8@cm0qArX*W~5O+^Leci4&}E@N0yYouQd*wKF(~`"
    "bPCR-aIKMxr{fR_w1{=lZ{(uiD7Vd+9#HOS`5)=)Kmg2(lbJDB^G09I8;?tNqc7QQ*hgTDZ6cqu%_0X#"
    "?TuuLT4ySKxLxj;?-*@}I8tJ)xpg<jO}&wudP0(+1oqWM50eq+b8lFz+~3>i-aw>qDTsfgx;M;vxpA}~"
    "Yz#lx2){79km@}?sota9Vxu2XEA(i9qkA@BIzgMSIRtMU;%)bFX3<DJ9p^|fnH~>_FRYF*v!T|bA|vjy"
    "0<_Rn9W3>Wzd2E`B}ze57(!OCabUC$ZRAee=sR&^c*{x}X+h(&w`@GTWx{i$6{xkPWx>jtC4Lujgk}Ws"
    "ppAIYMtCv-apQwuT{10T=)ztw`g-Ds$85w_GO|iBV~jY=3jNvwj7GRzSs-e$@F9c3$U^6i_oyr0W1I$d"
    "jg7Gp*auIJtTCQv+fmQ^sq<?V<FRM1yvjZvXNIHSM%f$20fPDSDTDQ3<gt{|l9J5_BTo1cAN+{*U>uT3"
    "rZh!;06}|Qd0EB2#z8fPOKG%AX@{7e!7(CZh-kxsc=h!;fLMFS0Qx>!!J0tmm@j0EVTc%Qh#13#G!Dmj"
    "{@0s()T3w|AN3h$f(gf`!*<XE`6DMSiv!kml4)-AA&UuP#AY#`9@HzF#VDJ_h@v~%B|e(-#o=23uMYQy"
    ")A-3|G1@{=1|dhP(C2)2ZS=7nGHn<x#=~$iVz}7y8DOZc!Bm1D+eACY7t%(5A#I0ZA5530g!FncBMz5T"
    "!OllA`V`CQBbL+kSWfXVwXNd2F(r$Ssg1$mFprGS{Ox$<ZVWHiSRVr(+c7xajvcx4Q*J8>HSZja&*5<!"
    ">a;SmjM!*L%Q^%^kG|bE&cX2yo62Y>*~s;~QP=NAZn}*V0aPP5-9~P@jlQrp`oh}iyJ+L|w4!{`Z1hF5"
    "k!xk6E;^0ALRKn9K=%c*(HF=zghGZWWCoZGaPavvW+S(~M&0%r{n4^<@|QP8-{Bg4hilyKa4m+GYhVl("
    "9~T>sYh5K>e@RcEm%?-~Y+vm<dd!C)3s42pfSdziScWbTv;AoppUO@inm|~!an)-)u6k|djh-4<5SP71"
    "U-laP5wI~Xdu>p(1I7_g(vQb`bfey*JJ4)2A0U{og$Z-l3VQU-++qKS*wG|+rS6{~uRU+Kr+Z=>y7zfc"
    "lXlOqUfTHle@`N0b4zxwCEp86e&Q-gu};#flMHWZH@s1{H0`8F&twm9vIjVwsG@Zh?E<g>bfP;^Srmn5"
    "7bhN^PNsvTeR%8>%6LK-PyWz$dhu9F>qsWld|s+gdZ9g4Qb|(037W=N^XgO`vA+eb75yvaMImQ?qvY(%"
    "usl)13nii1oNG2ql07ZSo|a@!J3kGwM9w6EvrsDjadz=kCuyDsPtG;AO&&bqS&qcBMByVYfTn2syVL4g"
    "O~1HSTl-=BBzs#T$}H&)Skg;XNqjHC_7b+Y<fAXuF!+MF@g9sR=mvm2m1LkxXZeMb{ldx7n$`oY1}1KN"
    "CEnCazRR^0!NQOMC1IaKZ?xK&<S9-y--AlN%T)@%h~Wob&W`6yHrQ=1zN|tl?IvIFN@1BhN83V%ErnZ7"
    "@nbQ(6_fm;U*dD_$@aX2J@3$SODTg({&1&d9NmzH;s5;Z)#KREGqIt`v7sqqLz9hs$*=A02n)e%F4^vv"
    "{03hN3twr(wt@O5n72~u?JKTPCi4`wze86~gq9|~)0gzIv&5r$<u7wzfKfGW{}JLgoR~Yk`L_KLrB*jr"
    "d886gpOqgZT@)Z@vZ<}PBpxbDyi}HW5G(N_R*Dy|lApXvdRD2lh!k38&0vz24+AY%;AjEnOr2Ip)AouB"
    "z4<a2`67UJQ0jvbni(KDl?%N>yOnOF*Jx^3quX;Gf3Ug+qUck+&`~hAL}zQ_t&~E4+!<SJs2k*>nrjpl"
    "pV3X*%Litget94@JTIK$eSqW#0@Cq9K*?K=*&t&eK`1d*%oG=jlW!C!F3%<&YfCbxmOmO$ZakE~57C1d"
    "a_3KbrOwxs3brdmiCC%fS6-<^4AAyh5BtdGlz1sW4HcwLfq{e5QO9ZEpp?LvBrX;bjxlhqBEhA>N2QnR"
    ";G+@$Tap8A`y)#B@BKeiDfdts8GX5gwm+hLHpbD4t$Y6~dVprZ+w8pkGE;9@gOkm{X(XF-X`75}6Rad@"
    "WI>Nd0nU#GUb18ZZY&yi3dOlV&h<fezng9hG=W$oDOxBpTN)2B0Rvka{jx?XVjO$d>O~s8oULCJqz<5W"
    "K8tT5qHY<#A5aQR<!23TK0&dga)@^sX)g0(Y57m%G>WY#|5*W&lhCz{%$0qXW*?<ly~=bdt4q0tY9QGL"
    "xu(j8Tjh0>YibxMU(=nnld~zOhFj&B{mks=2#?;N%Yh|hgnbnBE3F&cI*dI984b8XbGZzc%W%2O>|`b<"
    "d);KdNW3AGdL`O%gIw-*Fs1|SN$DF!8|og`{egH~d+I=!O?f&6&=`p{%#~MQ_6p2X?dtq=%Z4=vl#QLU"
    "L1qRXoBiVXhD1ixzJd)7q}bsPH{|;Pb%AvUI0iT#pvP|Vh7*e*QY`5ex#U;mk`3G&LVDd*C_u$+g#tw)"
    "zl@b+_fE&RLlS4SjoOLl=2AR2mt^KX+L9qT0q3YizJcTzwhcKu%^L{HY+8zxrh=9$Z=xl5DW8c;yc}2P"
    "3feWXu@u;re2Y9Ouq_eTmbe{K=vNDjVcSkz1(zp8j*z4$+>%VYN%y{!LeCN_K_3S#6iu0}7sA6~;n=kk"
    "cI~8V-$~cLldRmyC*6azB<0-I_w4M8!4-qsg^U)^EuYI$jY}r+uv_wXL=#VUC!X$(_ZE|%dE1eS>NMuT"
    "jBw%X0F3zR7sJxiwT_Zd0bJ?HkaFJbC`SS4*zEL!m0h1I$si7m7Z~OFta^G(JZ+DOyB4V%1B4ql*bZPu"
    "DN9XqOkSbgB@`6&lR~lu+e;*1mA9@^+kLaM8xikwL_)Z~61TWhn8}mP<lzv*AC{gX>{3M7rMN|re2XH5"
    "{X6j;({$LYlYPUymgzO};};FXm6`a!9zyG6zdnC{cNHI=&;q$7A1t%*q$Qsh*Rx5c_a|q8`CiZNG%A}-"
    "-y5KXcx@{KnIw@%qkHzP3%~1GfH7J!QuP>ss&>@hj#%!$j<KwG|IOC1<adz#Ok@gcSMrOsiTmKm_rVj!"
    "^D;K80SpMIupmSrNlA5!-c$5-$^pE9yhPvr_%oK1>vVp~*@N{@2}d%IM>6{32Lw|D7M5g7Y`Rcv({;K<"
    "<~RCjZXRs)T6+uoL_gR4p8^@N+>8f;{@#u4Q4~fq8`#s6f&E+`TR-lUr#hCmLK+jocp(>PPl*ie2}66r"
    "(4J<m`Cn0MW=<Vj-cwlKlb_W~epW9X&+4UkRxcgb)03~Kr^9HUY_#7Wk>#v8cACQ;<6YTwytbG8Zs6XE"
    "yjwygEK63ls2q|qf^wGGGwWO}e=2^xi?&QqfKJGx6WxeBLs9zK$+f@p=_f6vOu%AEp=O@Q;Pfr!wG>Eg"
    "RMv}vX=a5kW<nS9iMp7K%nkbe?@p`g&%b7VU-a`LO8c*reiN+;%Dj@4ZGp|Uz-C7-X712rztErkLVqSG"
    "F~_ZT>l9fJYDuABpGlp_S0%)>nz>h#Wm(O#tY$66@`s$7r<f**wOMA=j2U(RyVI>xLuP2riL9zcq}Qm+"
    "sG9LKX4wz&V>C%&BZX-aydyGJ)y&14Ttk{-lpVvEWn|4VvSt}svu@#JZsBBZ;bd&CnOiuSuWsf#NSA(d"
    "=>~#+CxK|$U$X)vGsf18u{HbYtXyM|$}*p2V$-s0s~OvBwk<Pb%gnZAW^9=`Y?+ymos20n6Lh%IOuRWj"
    "_WhXb=c}^qn0dEj!U4h}Sx^ZTIZTXOLP10a?tgb$JtCwphsh#ivB-pwWkSfZgS@hXys{o@$Q?!KZHDd7"
    "PRW{+JDh)6_tY~N)iZa*GOxL3-_yvxs3Dy>fBr+24KNU5bMgSkknL#GZAG$Nsoj+itj>X}cUg|e%*$qZ"
    "FcZKQp5<1`dcZx~OOtWaWVu{2fnHg@k}PLQ#@~>6e?5miIwyY&CQIwcQaf^pd>NT9BZy^Upt7Zp4ai2^"
    "C^W`=9J!@#iPl-8&OQCC7^u9dgYY7>#tl--q#x!(Kg@ayAoCVLmb#dEqAW`{%;<)B&aRNmE3^h0RWTF%"
    "l>H8Qj(5njzgCpvYegC5F^BS)r6^{-C6Eb@DrD=Q-DXxZ50aG>$tj{G%;^`i?1G4COS{b0F7p?7^Ovi9"
    "EVARsjFY80W^Ns1>5h3Op~jt+Elo1}{#2Icn0<dLD^NAZ8)8{5@BGReVp-~AK0aqzkPHM(@8V6c4WU}a"
    "-)}$<K(SISF#=_D%ZzTBrCa(}qM))=<&3IXetkgG0K}eb=X+)q1;4p=63p|H3Yo1!<|h?0hbumZ23Zm|"
    "NAA%DUZ1emV0+|+IpCVgp={<8@@76>3&`=jSbowpvp-vvkJ#NDF9>8QpLu|Aw+8o0?+4`J0?D?lAk{Z%"
    "rrC!JXN!*(uCQcO%)F_Xbo(nibTr8p^+11hbEI43r?&)hyd{w1nSktP0&=_wko_jW4zabSH4yLD=@<}0"
    "a*$*{5|DYTEC&mO1IANj*>9ESr#P@Y9E)L#17nE#rVR`x5;OYC^YVqgqUYwgeU^RuETh%tcog48(0r9;"
    "HCwCA*PnvU06DTCjPjyc_EB)9p&aqAz5<(l#1re9(L*yf@$3+d?E7fhe#^|;W!YziBQtVns@a-qMlszg"
    "_T-|O<_PNwi*EL1h8!Od&HjMsVd32kyaDCNnvcBiG~y0m9$;-|f33{Yj7=T+DK;w~u~~=7*cRg)nq!X8"
    "to-BK#)~xQx5AAWcWH@PTH;3Dp&tO^{V0hUB{9pexRLPey7jI@CCuE4E5JaJv+30UJB3SfdG@LdeKA{K"
    "%!j_1tuJnv_wYPKvJVnP{a$NUAa^JmI#Ri)!=G)_JC^qx-pm4^rxfS{axriRco=8^&H-d2av^2E^Io7R"
    "w*%-5i~zVAvfq8*%DF~4$z`YL!veBIK9(R~d$s$(-UB-ivb<LtNuR>D70!el@4z3<gw4O;%F(ALD^#0T"
    "m#)@6j2vs1<K6ch@4jzX4r~Ql4RAV5C_(Y=d*<Eu9AApf@uk>I(B9^`FzwNwJ^Tah_Q#R31NpK8`7&X9"
    "nU~-<+RI~@U!re_lGBqc={Mu^{$_lXmW=rY_v}Ex>^I`=oWVD<krlM981~i_$P4b_As`++J6JFiEtj{r"
    "z`K#U4a}qIeRu>5X5N4Ya?isUo}WHgyQNLfC{yc5ZqPrX;A5taSBN&;%N5B&F^GpVDied4BL*=OgSaD}"
    "oVHDYdCU*Q-5dJz6ZQh=Q&^~Hu^XS!KaLWPe*E%p;|-&-mv=`lhAsi2m=vr!#`VKf^|YeQ<|ebb_||S2"
    "xX-`$M>@KEeOhH^h|{S)W+aZw>4<~garnvS-<AVg-J<otB76OmIaW{IKVg^Hw!CqbpCTT!UFLw6KBqby"
    "9XUSYTNZhJJ)mqXKPPl;v$@2Zw%PghWZL%L4VzeZZ&wiW$+sE+M)@{43)B2I0}KIlfxy|{e*bv{yP{71"
    "@y2@DpbgoJQ5n)+xxgdla<CR%-g))q9kH20a%PCdyv9wPSxmOjns(`}r|lyZA~jI)k~WI#>*nK307)q("
    ")08NwdfCTyIqCmm5(|gtqLXq7dusubkVg_tdIwCF|B;|g<WSh@;;_?2D(@oRw46vzagrIAcw}_pwA}sz"
    "KvUhyul%UfW;xxQUBj_RcV4i6F8;#G!a2L>m|gbOvzz*secdcts0+r=1>@%;1$*H};v(zkB0Kz|qj=F9"
    "LyHdKMdqWWrW}u_j@5-{f)<_$T6iXC;fP(ZRxULpV_|dUVsqspb7lGC`LMNrv9W*A5xrnwSTI^H7N(cl"
    "8AvZ1<f0JJ#UY?eJPov9mRvAPE;dURK%*|J<bqXlkyUcRD!C})bK!fY3(=j6qdS*-1eKIJaUm`?WiIv5"
    "8bG`+5c2&J=FA0i=7KqM!JN6#%vNyf#b&UAJwcuGbw;w0Y{dP>(6u~8TQ0S|QcGAp7h64-Fl#QDHJ7J%"
    "h?ln3vAXD?t3@weEgD9hi;LfUSoUXrc1V#K3s)%@QP{s0@sTAycC)}Z7A#(i#F_=EX7TI!%i!nH7QT+R"
    "Aq@aYFmElGw-%eX7R+0V-`iho-dc#1SY*puY|C1{!~!0|U$yK4)}n67D78o|T(BoCbb__yrG%+y!Bn)!"
    "RJ4eJFIaXL47&^R)^ZKry>MM)@okL)aMTq@hL*)5*@8s2#N2vMK3R|!TacDxm0!yue@c0AaaJgoklGev"
    "9hQ*T77J|4QxL)uK?uts#4Ub#evzoQ3=nFBOg;-j+M*YM7VkPXoxe|4=uAVt0=N}^&Sx2-3B+qIepY_5"
    "lDG_0&<ViBGK7N%=9lN0V^d=#S-hu@EsaTJA)IirNo2v%zj(_dnnldda&0fvC{b<MuDH3H(R=Q1z8AS5"
    "ye-#sJ`jrX5q4a1TjG-2Vt%q2L)=L-l&goq&h_Pr6?!>*>&y1lwl+XoYgQOn)waS8h?Y(HfH)5dyL!sX"
    "lwGHovxBn*W3#(`+1%|Q=5+ln2hw=$*Z%zC5;tl}%CHtMAzv<tmrE%4ixvFE>idHFzJ&I?pgk{W&kNe~"
    "g7&;fdtR(PFH<ZSe=%)?4A5%y)&(84R5C+La9k$j#>G0^f=;o74!0qXfLyQ{#-YM3sBp`Zs<vUt;8$u4"
    "n%SH9lEQM_&srvIn#-X>EqsY<vHJ7WR5sOgsso^Y6rv@xk`3DDM~4Pi;y4ZpuWv;ncM_0p)^Eqg7kjkh"
    "%c+WzBS`K`jToy<?ziB|LGk}7JG^O_#hi$#iwBp#pu7#UoFZT7&ft#(e(^>vDI{+BiRSn~%TF9=ch0l@"
    "9MXX47}6;uHwaD;1=PJ-zL!hndbmUsQ}^nxq!3j1I^XuW;yk;Cf5hGpJ451Xxo?(ZNqDsjYTc(~J?a12"
    "JS%%=m1%61Wo(tSzsfAOYFDi~+t(8r$=8YMMD?P2QAt#Sg4J#Rn*?Ri8Op!=XD#8Cz=!qZ-zSqadZN|p"
    "|Ej&d?mK<eMql>@wAT7ReMg30CBv@@U|1!>uM*)`@q<+&{3;QCl?cDq6hkTMyHw>>!3eAFGOSv$tJdF&"
    "EWaYluQeppB5DyupT%lciQuck09L-CRmfzzGte2yMpV2+@D&k!l?c8fg0FGOX7xixE7zb_U4vR3DX<bL"
    "u!d;8B3iGA)+?g*YMFVp%)B}*VD(q6){vQ3%gk#?$15*{th(B;>Sn{LuVt+IT*m4rs8`;%UwKt=<^8_Z"
    "{`%Ga`c+-0=Q1LeKiVq;Yga<@I(X?ZD6<kPt0S9NM>elg^$V?KBN2PWr@hXRT&cw?S8ihzaxpF0ay5fo"
    "%^}wa3SRxWsdZ;Tt6IbdqE<c-wLT!B)YX}M>M0!+6Iy90OlPH`I>c566d8I&hF*!)U5V9QV<~wh*`w;3"
    "eYTlj#KE?5uC0efamApxvKj{nZC%NzE2hO&mc`X);K~uWTD)8(Uan7W#C3Bc8f0X3NUpKIS>8GyJqoOK"
    "mU_75*3B(<1yif8v`RAR$}5m70%xIHr`P;ksZOWU%Hg#7a9S-zt{hG)XVV&6$vY{ezzyjGS_~~fo{qW?"
    "K!;G_YG*<6Lai)S?0BmV=G8X56_3&y9-Y-59W5c$Fdot7?IYF`%jx&O=%L5k3F-oQfo`B1=oY$#?w~v9"
    "1f8HYMz?<Z+F)SsB)SXGJCZl?%HRK@M<j2xsDZSz$aur{=3V;xU;J9^Z3f_t$=j7oy!`q^w+31xF5~J#"
    ";$6y{HjWVYuctHdsOxQbX$=_(AOmCr(Fs?^>c90vl>E!THk#)zZ`uPJ-5PK`JU4n{iT?f<%gFuBLNe@m"
    "fIA}}4_6(pzyGXvFr5}#zz}cnd7NJT#F@E`n9W9MJW3JRt2~a{XivKM$&TDm#ks@uq=owf4^JPFzgmfj"
    "(2iGsBkl<7I61B(=FI+<Cp|sX^rrxh{ic91M7!hZwS!ao0}x`AgZ{hgiu#@}Jd-a$z%qxbf?`~K`v$dN"
    "T7Mz<*&RjFZbkU1(vBaoXpf>KWUIAWcd!1!+-3=&<eN)|o=jboW4gre7bm$KdDPm;AHs_Vg5!PQxV}Iv"
    "$?=JZ_(H@U5AqIVL<8=JExapqBs|@Bi0ck<+hM!xkc_B#Vx~tJNCDc&lt2PtsQMhpsw8UwbU??P01CF#"
    "MSD~lTsfrnCn&yaJxCk!qk5cHS`IRnU@Ku;+Z5{S4%fDF6`cxV`gM%G`crq@ba=gLIdFB7|L=W7*@9VY"
    "<_~^mSHue)xV3y|0gu)RLw4LwIQeqI@p5r|e8OksNrH9QQ(kRP2{Xy7%_N8Q<2v1FnO4Co&jaq>u??tW"
    "&TY-`_oaV)+mE#yUnVQ0xYF(9&)F2{B7_O~rr7tJW8VvK7w~|jgw{W|F90t_Isx99-=wH<eiU*PIYK6p"
    "ZjUv$n>DwB?Q?W<$a(=5rC3a_qE|?XVmCjwTM&(9L7-j7bPa%km@WrI)3~L#L2tCgTk{*o_>If4KfYZ>"
    "*7RpT#&42Y3dIDV4Hn?S@2*uk;H_r3rRd$5@^5H!H?+9|tmIG5Pv7S{I*(d+WA%L_@S!A$UPw{9p{SK5"
    "t%cKiR{TZrn?EVOK?%g8<Cjryd?&jgDz=ScH5zSgk|G@5u|X@)YQTloE_9%h*>qz7_g%qeL4(s+V7j$_"
    "*sp*OAx<NC=|)I*0esAF5MviElUwU!`kOzIdh<6@cPIn-)cF?Aliv6=>IVJtZ&iwQmJ$0NKsJ&CT4+JN"
    "MA;v!v2<Xr&&x+z>`hwiMrLS=R?PaX^L+93zFSv2L3F?Kc7FPFPs#W_HZ5O_D59MC03VIG`Ku8>zG)^0"
    "h{~ICt|qf51>oF<6RS>|^KJ0j9$3q#Io7hK-mdCGyWHcNZviQH4n`;!_$^%E8<GvVC<8I6bCiu>zeAV6"
    "j%3a{xh&T8Fq?BoHNT;nZ**s%GtwLBjkwv~s<G0o8SK!Tp%^BaId6kmQ)u}&Yx!U?e03kgr=7#1^%tRU"
    "d=a`J8r_*mQ^fXdrWY~^+-53~P2lD=L9F&I2L0jEDO{$-+j=tDKsNzoBiT?3(i%tr$p9zB>LWrXz1!-&"
    "AgkVOtxZy7?IK}jt2sZWw%h8>vznEyW@WMp-Bz<Q*;Q_9kksLU3@gTM^#WKAX@3L9NGZwthPh%#7)UG<"
    "8}wla?Hw!yNMB%${@%$JHM#@lTPRRB{0ke<I)^2N;HJS$qc^8MrvpNu=*&z*F}elI4SnVo3eC48DE)>J"
    "w_snNISd*E8tG0~0F@E8TnI0y61O4Pf?4~JO>^W9N3FDD@R#&9%-d+X9*B8>MgMPcMhyKW9ni(tQ!>{P"
    "wf^!iQ^HAt6#fkf`4$rLH)Q17{Q0*rAokuh7+eU%Te=*4+2;-4>(~G%b2RjZd4A1M;*M((KDwy&*MAj~"
    "3VjE71d=T02q6OZHk}UO0^&g`Pz5x;9#{)zUf_`00{Le84vx+{K7ne+R0P1Iz)ZpzGl$mRrjSta1>PpJ"
    "6P$s!>F7~SYinOW+QB=ZHKA{I65ggddQj8YdK`~bw4n$IM`=vbWLXe*;*CJZ9Y{t}lr;m*0gOhD3OT=~"
    "IKT=NOJEwj28dL>O>^YrfSBf|@rUJ1*QhfX$P^sp@L86zHgk9m4&C5e)B`)1h67;ohNsDk4LuDZLvJV>"
    "ZW?X}7QKtku??Tv%}%rfQ_zdCm+KSC3b=;$wSf)b`G8me{^Fv*o(|Vfjs8dmpJvnBd<vN|qBl2ec}f_A"
    "%j_xe|24jL!`^*m?-Ka`IunayBke0j+Kt9^Ht(^c6yjNYbr3)MdpenV5&2?rucz!o%xm8M-u>SD)2Rx&"
    "1&ADX0lcGVk3J6ljU>l8awvXdIyI3a1Gt@TS7hmwpff#+9w904?4J{D(Crm-mkICnMr-^g9GB)b8>?}n"
    "v`{BWGFw~%Ii^UV%`u_0!kpkug`$AZ-zZ_9<Ef^PD6vx=V=passxPjfzEDGbxr1mr<muc<%81zzqnpI?"
    "eT#qxiOllgcz%`FkMjC90Z*mIRf37@*uF;quiZ#Wcfe;{E11x%3jpnbue0iW<kzTGBaas5YY2V=c%x~i"
    "ZfX}mjAi1gcHC3onnLjB2^}>qDa0Lx4Pj&%Sb1qD1xjutv*X7yH|PR#F~IPgyz+TKbK;T84M+fyf_fh("
    "SFA-->uZ>W9{7ER76+gBzTyrrK>or_!6ZmNB{G+v{Dx-1%pRsX*u(LR=GN%+1m;NTI*^5@$8ZURK^odX"
    "nBJi?@<Bm$RQbWUT_e{A<A!T_hq9p~A6>G$dB@zPD>*fM1DW*s#p&}O=4S(pVU8s`z-#{DEi}MuX1XxI"
    "TxNzxH>~`;G^hIZxHP|WO+~j<jNO4OfE1{|fc1>0bmu8_=YnJqPXd>eCx51KL*7Bmg(N^sy0*NL)Kgy~"
    "fh+Q|w6;50+ub1-?60&H@CsxeN1)Eu`y5DetNrb|9X&v>U4EH74Z6!b&2FS_+3+HoWt-`BHrG&S&NOoZ"
    "Nd29Tp?lbzQm|}3AXoV`y9Fx={4;!HhL1d4!em;bKL;COa|WdX@rOplm;=TR;cJRrSr7_M%xG-Ws~5(T"
    "59@d<TyRfr2ZRdcbVN6Jia(otANlE`P}~y|=3~#y_3&LCH;0~O!w?NE7)mbOC`lWb^Y&{*?2M{4qFLr0"
    "!MH`VT_Q3N6CVJpw@XIEJ%-*i31TfEpH4&c6WKIG(?c+w9ND@D-?1yE0N8%VIDGXsxf@{ihL}BTo|83C"
    "Za2B~Zk(QA8t>N@1=4=AZAr(Ix&4n{XvEcx_&_FexgUsi!2<|sm<VYI0eQl6v;%62ZXWMIb`rM(R*u&="
    "9wYQqgdtB){<uUuO*mag(}^r805>H?Ti_`Upb*3WhPgCk^8KP!6z6o;Lskw+J+4`82#ezL`>`tQnA*Vh"
    "$ZJ_JBn*eoPVxDNL++paT$Lb%roMTClCh_BoWL>X5!!JGJ}45?rbrl(lsVTy_W*q~-yt4_`TE8wSG3Nr"
    "s0wcgKm|YmdXrpV03V%iCz6Dm{wBG+q!|q)0B^=O!rL||8#0m)gm<`J<CAgwr7_Y?DUgeyOF*5W?#SqN"
    "O=Cd3q+PIEq%)*_qzh?fK3B+!HvJ~~Tldq>uWYlon#R<$r6#?fF5VFLp{{J6FDN6cCzcA`L-)=$GK}@;"
    "ECzz+G3VDhhLz{TUXi@sBxsiu3-k)Tn(m?u4_zJYoeaFWv8?mAJ4*-Q`R#u7u#iVGZ2=I=-uy6pf#w*6"
    "5b(~gA>qA+gtrwZ2LINb7$3YR1vyf6XfJO)t>}?e-uwmK4MGNaqhWlWyU>+^n6kmk$)d;G;58l6#@nzM"
    "T8uEb78AEPk-4iF;O!c6H&*m~Cg}{xybV6D-o7xxe2sepJXC`x=1Z(zzb#X+s}T>v7nG+BGaGsP5+gH&"
    "&GL`8haejDAjKR%y9G!xZ#){xmCEhjrBhhg*`XX5a}aO)Z=p8_^b)(>@fpHB_!eZ|o?q-NaDGm_9Qg8#"
    "$40m8@+R4(jDSh@<Zw0s3mL}9;X%te8S+D$uS0b2tGkDQ!sXLrpmbW7#&|Ph!73>#1GMJzdoW)=X8G#F"
    "_l>i*6yyuZ8<w0Z3eQH|!jBlH&+o@DeXP-}fMQ0c12IF0?<4K-n+tpbV<&qcem$_+AZC>Fs|_jd`{KWU"
    "{%<H9On0A4@6vDXHd(#voVuURjwmWn5@a%q^iqHKadWph=>22>GyEU#vOMemGCK2_%R<tm+D+0yaOWI&"
    "$4qkzGtIj={vEr_JErJ(|DhnV<?3a`Lh9-QWArZWf3KZb%>Qny%LdT?+M>}rezekPw_$NagXjGHh;aZA"
    "YriXay^vRto*6OryNwm^_lIbcVhMMfCEjhjxcg%t1@iv56N|n<TFCdX2fW8IaDR9VRJCiCN4t<L{P)2V"
    "2=95{ne3f!KoyeT2mdY}d*}YkMl-AXrO5yz+sXN^F4#Z~P@9fCwUd}PZrP#%;saaJoY?PS_`mxjiC`CJ"
    "kC$rxc-PotrFfVAd`EwNmk;zl#{!f7e8=%n==87I6Zylo_l`^X?%V7I0SKB_&Cw+B*}H8M?<@$fhtGd%"
    "+OD7f*++_%BrFh_wj#^#Ka(Y9I>s_a(s88sxjz5|`GVa%o8k4~DzA_it()%U`V4wGaphAeoCi(s{Yg*1"
    "9eVnEoLcwOb3K`K`6*dsj=Wzr&!$+Ct!b|VH_Dr-=#S}u8nBp49YHautUMn;?-3J!_fu6H)_^ny#I>$&"
    "jC8--)&9Q?k~ih^Z=Nj9qq~ozcdK@{Q0?A1q3$7yyoX!VpOX?K?~lX(gwclSsSTdw*g{V`=yrYDK`2jm"
    "?_9sqTY$=?#gTR2POSg+e*m&q%ij"
)


_p2g_cache: dict[str, str] | None = None


def _load_embedded_table() -> dict[str, str]:
    """Decompresse la table P2G embarquee (lazy, une seule fois)."""
    global _p2g_cache
    if _p2g_cache is None:
        raw = zlib.decompress(base64.b85decode(_P2G_TABLE_B85))
        _p2g_cache = json.loads(raw)
    return _p2g_cache


# ══════════════════════════════════════════════════════════════════════════════
# Classe principale
# ══════════════════════════════════════════════════════════════════════════════

class LecturaPseudoOrtho:
    """Convertisseur IPA → pseudo-orthographe avec lookup lexical + regles fallback."""

    def __init__(self, table_path: Path | str | None = None) -> None:
        self._overrides: dict[str, str] = {}
        if table_path is not None:
            path = Path(table_path)
            with open(path, encoding="utf-8") as f:
                self._table: dict[str, str] = json.load(f)
        else:
            self._table = _load_embedded_table()

    def set_overrides(self, overrides: dict[str, str]) -> None:
        """Definit des overrides P2G (prioritaires sur la table et les regles)."""
        self._overrides = overrides

    def predict(self, ipa_syllable: str) -> str:
        """IPA syllabe → orthographe optimale (overrides > lookup > regles)."""
        override = self._overrides.get(ipa_syllable)
        if override is not None:
            return override
        result = self._table.get(ipa_syllable)
        if result is not None:
            return result
        return _apply_rules(ipa_syllable)

    def predict_word(self, syllables_ipa: list[str]) -> str:
        """Liste de syllabes IPA → mot en pseudo-ortho, separe par espaces."""
        return " ".join(self.predict(s) for s in syllables_ipa)
