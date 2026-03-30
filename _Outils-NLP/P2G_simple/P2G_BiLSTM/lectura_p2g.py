"""Lectura P2G — Convertisseur phoneme-grapheme pour le francais (backend BiLSTM).

Utilise un modele BiLSTM via ONNX Runtime pour la conversion IPA → orthographe.

Usage rapide :
    from lectura_p2g import LecturaP2G

    p2g = LecturaP2G("modele/p2g_bilstm_int8.onnx",
                       vocab_path="modele/p2g_vocab.json")
    p2g.predict("bɔ̃ʒuʁ")                     # → "bonjour"
    p2g.predict_candidates("vɛʁ", k=5)        # → [("vert", 0.35), ...]
    p2g.predict_syllable("kɑ̃")                # → "quand"

Pre-requis : pip install onnxruntime numpy

Copyright (c) 2025 Lectura — Licence CC BY-SA 4.0.
Voir LICENCE.txt et ATTRIBUTION.md.
"""

from __future__ import annotations

import json
import math
import unicodedata
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
    "ø": "eu", "œ": "eu", "ə": "e",
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
    """Convertit une chaine IPA en orthographe par regles deterministes."""
    parts: list[str] = []
    for ph in iter_phonemes(ipa):
        parts.append(_PHONEME_TO_GRAPHEME.get(ph, ph))
    return "".join(parts)


# ══════════════════════════════════════════════════════════════════════════════
# Table P2G embarquee (zlib + base85)
# ══════════════════════════════════════════════════════════════════════════════

_P2G_TABLE_B85 = (
    "c-m#XyN)EuwWj+j11eh?%md8ZL`H;1dia%Y=3X5>7=cKU(n*%mNM(vZ<5W}NQ791DVxom=?1vh2|G$5E"
    "kVmPk^@rUr%ht`z|JNT*fB2vO?GK;-=MVq=zy0C#vi-mOyZ^rx(O!4*vQwA+|JDBgYXARz|9{<*?4EY|"
    "k#?%th}Y8}XZ_gd$Dg3`PMtI4?|<4`^4IeupjGBlvI=X*J$*Zx&Lp6K_Qy9RxXq;KRA8syJJsIlw_d;X"
    "`fXHE;|Qsyjg&mPX(LTjNh&CHBJwNsqzqtExyyeh+NFkn{*@Hj#b=6N1iYRWfK>&EptvbRy%>0L`M>|^"
    "qWse(kVa|FkZSbQ#qW&3G`f=C{tR;K4gLN`A+I2BN)q(D`f=OImwjcuXqaBKwqCTjURYiG-<`gQq`l=o"
    "Dy#g*g;k=f%1QQ>_wwN9Mn9jSRD4-XEx%J=Yq}TR_t1&PCesv{SiCQqwelxl`wXIy->`{9i+}}kr4UGs"
    ">C5TnTD4>g;7VNLWRhW<ULpZQRp!<RO`ZODl;Yu=Jbap~ZA4y;kmSCT*PYORHL{<-K6bL!IP3pvPW=Ma"
    "LNcng+E&xFT60#HITNjdYj5e7S}m{L@S5^mTqm*<r70_ac4`zgiqiDe6wBn*F?qEnug)eJ=Q?Yk>y-@r"
    "zEbl_)l6L7-$eiT<^pcad~G0T4S4PGzH2)?PNFp4kI!F44oW<T`tkq#*IOcapTNFqa2`Vo@S!}X+9^I5"
    "?`vg%@w|H37s&1W{5}QtU$phAv2E4V_5{T=x8AL-IO|Jm-dd1Ek_64lZewY;UfV5;yY&ieS%U4AQZ**6"
    "7GC)ys)1~vX#=aH^#ru_WPsvpiq{yogLZE_L(Y(^X%<2I5Ww#bmp@bf=RZYXN$}fWoZ{*v+dR_6{(L|l"
    "E2g&n@s+Ag($&FvzUF^z`j4+Jcjzw%bP0?e{P_7_o5+%BuP*0<ELWHE;hFuLXZFVj&Apm87AU)%91z2I"
    "s*hA|BwMzN#n+X^w?mqe>q+e!VA8Xe+0a4KlmQ++YuE;M*l9k$0>IH`n4V^MnAO|4f;2mt?*NuwUgiem"
    "4B47C$C`~eM+dD{I44L_o6nRsu-oQf9m8ziF$Xf;3|x#{B8|(QQb6>0%V_M)O@1}vG$i?=-95lIzHqP&"
    "pMTFzusVeSH_+Ve%at-ee+8&LmLl04!+f285)#{E!!-L}Vu_S|S|i$XS`Lp&gIy+!(S88L0rF^HKsX;C"
    "IwWRF(~vP64@qW)q@xdLt&hjbul)Gq8@1hBI~|xOO8^g_s{dI=Ta~!u*SQk#+h4TCXQ(QE2cXn4C=Bif"
    "HO3ks28xaaV=-UZ00aZ?{tZGh=zR`um}L&mvJLt@3S-g0kp>72^6&yl7b-}$(F2k_bZ$(uZhvyoY@rsC"
    "9n?W;0tX9bKfaRP+{estD7+_u(51m=&5y5MYttowixJP>22$J9M_HD93wVn*cvr<V2L~Sdpc$o(r3mw7"
    ";$gj~njb$9vc5BHAIvz@$;llDcLCVP*Y;*ejlafF!Kn)iPJuL3lb8-z_vhcTldR5UfYJQ?9>ZzAZMMAu"
    "FGgMhqCHc8WXnML^}s0q3bWfF_ic*4d1w9YeB>li9f`aNU>-kyv8XazRQd6XI8`vJAHRrG&9J}U|0H%e"
    "or0!}$z>Gi4iYCt!5T2{tpYh?<m&-?1Id-8Yt*-ZEm00Voe+$E|C2f~AsPMtC-q^9J|LG*^-C0sEIwHn"
    "-w>BuTdOJ{-PmOC$VxwI_XF|zsXkcgwEYKYH4;U$Y9=JI-~XiJenPJK{ZBLW>_Rw_r>8TyfStk4NHUV5"
    "HjcGvsEz0>nP!s|V{xJ&OwZg0GSaVkbdxwkra1-B7&)f?;e0s|_kMQoIU1&`Tg1`u46vf7*()YE{jrEP"
    "$SbXnQF`Lz)6U4<UU>O51nto~1}`|=G<d&lpsODGCulXcJ|~R)|5EkO|JFgcVA)7V8nl)Ptz|<P;OnUg"
    "hz@`Lt$oTl0)8I@N`{V9Oic)@8@4#va$u|5mY%BcES@a-?BJWTKaG~T(NxE1!E2_u&Ija0SI4R=U@R7@"
    "**EXyKlcCb3;S}?|HYgx@TQ9lCl?v!F5**{(?e7vsu4v-_SHo?T=@gpL-x?4=uuP>b=JsSBr}#jBu36H"
    "|Ct0V1{MG@{fpJX%gG6*i)k7(bpY?8cOH1zIUwet%T-A~e^t_Jg4D13pkFlP7i<*!-<{Tg)v8q%itW!%"
    "Y4~alpE%>?(Oe<b=Ng0MMO^Z7JyEa*@bm8spm^q`{s~H@g-PN93%$TWFEu1m&81#MFC<?{jr4NCMsdML"
    "aluA$sc&uoLtcBxS!5S4F%;yB^y7;R6qou0<$<#axIl~uE9%aYdN2Za)=+)wox$tk;e{c)#|68`MRt$!"
    "^9-3xCxF$!Dqt2pkl}&UajECYwTjPP>MLo?LQ5(Afw-ogn#3kc$tD}OeP||Q9Y_c8L2tgBP6k}+2h;lb"
    "on;dym@SnH{%{M$J1&jpqS0J*qB>JOq|$a8(vBhRwEVjoio_J$eW9#jrnDHHD6h1c78X67dFz#}S0>Hc"
    "pgs;S+NZPR-pjz|{0E$cIrWqdF8n`Evn}Emrg_kTbV;nV#7ax_BGHTFqv?_cBOXGh(Lx!deqLr)^Duig"
    "OVqeX(oicTRut$=k_^q%%?tG63^aRQNkG>Io%t8E@k?mq7lt3@yO2Ezyk}6pH)I8fLS;C+Ncp~O%D3n`"
    "M?H0xG2~Ko>*VH_)tC+bSEST1?LyK;JLREKm-XW!W&0vyOrbKQijwSfnNgfB51kjY{0A3FO&7~&7b({l"
    "b}_Qig<)dDVk)@&_Ln=f1keJE03)=Im+Q>=Vm11LZQ){3>LN+%63g5h{@?)xxd!A@rx#O%D+~}CprYF*"
    "11y;USY|Dx_5%C@<@RFbwgBqbFq%%!I}26!BD3GcdhW$~?!|iUWtegR)G^U-@06j>0niQE4t9ig^kX3T"
    "|Ch_P%W!a!aCd>r!V1m!kl`*@nE^JML;8CYDeM}%Tz|qmc0tf9$r!;8xXE{1RU6V9qKhBMF13gCBmG-r"
    "*b1=`HiWu2YywoOt1SzcVAa<t=nkN}2ITg5$i2x<br%3)2gaEl^7-7ybp9gl0Bc5fy?CD^PF|2EH#+}2"
    ";Eh$&;9tb5XBO};Ow-JXwE=ANs|Fl>_%fZwWS|9!6deF#P!Reug?txU>;*0Mf);y0i@l)5UZ&b4Ji2JH"
    "FJ{RFtpF<ncK}PWR$>d(7)Su-KLtWQnsDM=LP5Qtpk7c=ON(dIwCM96Q5VPCG(_iT0B7Q4DZm2+u3vQ1"
    "97qnx<%kteF@RVX3j$BK3VA`Hx@;PwNB%nB0&~C23=cK(f*N_TUU-pSxYHxR7>ElWj%f{sf>wC3n)lQy"
    "R@uZV3%#r)IVL2w$%Op|%yZd$GRv~QFc26sF2BEPMDHh)d#?Klc2j^=z`6ox@!h@n-X&7puO5$gc@*#0"
    "E8TxbAih6b$v~ZS^ZhtZ?{S`jUYLUYhYPdtt{J{N|8IZMTDo6l+1MdxjJbPfqV7Y`yTo;uJ$4788EynK"
    "aQ6(nn;{pZ38*)8WXl?aeVy(@#(($RpTF*TrEj5bRr@GaC;eYrxU$YrNk1)rAnzi*cPq_XCBJ=jB6kaw"
    "A+zXxw5n^@k+;{+y{JS<{Fum`VCI!p#Q2IgzCx}lz~mIz-K&$#?i)a<lZzEotHjf)eS|7(i`CB0D(&&A"
    "CN87=$K`7oSJFMppA~WyT{%NFQJzLC21{(Ns?ShrTm2KiwxovYAUo*n;@j-<mE`m9of}Artx1g`Dr%+U"
    "rrI1uBoKe7YKPoZwJWiYWJ72PM`nd<ys~S21EO)~SGP{vuc~F3t*nq%p^z4+akkSra;3)kP_+w2KTc|r"
    "l+dcysa{D5tx!U%yL@~VCt?-)W64jS4h3?<SQ_@wSM|^ag4?0gAv;Hf&RFu*F6)DAQJ<J)Wi4_;7I*aX"
    "`{JHjI*E?RZ(sgoNJh+#`|_U#z}DUhU;g%&>1l}-DgQ~YM&3cR?B%Ji@YGlG)K~J<S9t2n-!oO9kyq_Q"
    "CqyMze1$)|k}JN#6<^8mUeKQ)GqwF^riG7%6bm#1fp5NQWm&CS*H<(K1IyKrGZjutLu01xudndUSMtqQ"
    "_RUxL<}3T=x4yV_mdXn@1>}Oop(|YTM+;dNtr+%3v%OnO%$2?KB~#2*tbz7aPXWC-r*~L;$eU$eEN~W>"
    "qqU42M{5-ZfeM2_rO7(UNM8+bU5w_{0Ee`yR}I<ep%-XIUb=pTMW;&9g@iq%(sE!m%p9eKzA&5;u&TpJ"
    "P_bq0mM?T;AXd7x`D$R%**9N>Z@!utp<(s_&5>p@&EOWqgDdH-LZ`2+uvhk$S5nw3E9{k=<rU8ILiVN^"
    "(=T<@%dCE3eGIeC<0@p2QjMlbyM|eMz{9lfrHxnH6v4qBbTxYFEfmh}f@ssv+VkvT*6Zg#J$%;~zR!QI"
    "Aevb=Avw_MtLtHKDHtcrb>@hPT^a81%9?y7O}>itc~$Jo4tS9ijBE#sM&sWRs_XNS$R~P#1JjNlAQ}k?"
    "gAWsP`N3-st-fMsqr+GBu2jJeP||mx1L=-5WV58PLD~5Zh}I5MwARbOqw)0^jo}!LuO5w|KOYbk#Blul"
    "JYy_5A5=d+EZjNXuR^G-kSRAh8*nCMjPD&Z?88glhuzg4(aPF-Wo^AeD&1NVX%><!2STT}6A}o-zMCra"
    "_v#uWk6XE9y94;N?}LujRUZRtO{GFn*^%sI9#E<56+7J;d!dg6-gxYE6!2;~nQWlX26}+rzz9J6@&+a6"
    "O2GgNiU>Se1`aK~LQAjE(kry|3N5{wSOr$L3*h!3Mox9-LXji!7jHmzA=d+PqP!Yfoh=Gta3;^-2EDod"
    "S7QTRm`z^{T%1H_3QBrq7jkvDkSnX{1w%qdC9fuDdeHT`0e~q^m@ao@HRNsu!`l3^=?Qcz`^yvPd2`GT"
    "a5DT{K+K(am$mvzTK!H_;BXGa&B3?_b}@LlKA|2aa}8_z721AfZNHjMG2IqvcXw@IEZeKM{%#hab=T)*"
    ">ir%r+bZ0%)#0A44)<)S==*e}bpNh(|Abv)EuoNC;i0WI5A9y)PVQzKg;XfgSwd2c`%xAt*`u@>UbPpb"
    "fk^?YWu<(5;84{WuLfy*yxKU0nuV67t@5hvR>Rb4gnsRide#3svT^f(&B9iZ`Xt+hb*Bf4z+Gzj>%#%}"
    "xb_|=EZeyD@a)~CTR%?xwuo9psn*A7x+DYIqL1yS0MNUDoySQ1!UcHOh}Zf*&BR%I*H}C>7I*sh5TlgV"
    "hSGMIRjyF`F(WS_PQv@lL0V`|()+%zANm=|=OGafUFE@!zO$bTt;xFo_Lml*g(g1+n@|_=HXgd7eT{cT"
    ";W5JfP336g%Kcaft++vZ(W>PG&C;Go9FcdFkTQL*8GuF&Q_u}SV??bSm?N`=dT=WXP$@}Y3Hh0hiuw?B"
    "BERA|iqcAA@qh=G{5z7s14-ZxKX@%bPp2dbSnaj%w9s@)ETGUUH1}lCf5VN67gbKwvsP`x7C3+Y73rlQ"
    "%6~q+OduE07n4qKae_*8CF(B9YDlA9baqHsqg`}9`RA73+ao~-(gF0M@enmS+RqBZXnYgtykAA4Ux{8O"
    "awIBAQ3;~@ALmn^0(w&_{DqA+^_d{FMv}t$<U(aUZ(cwDdUMM>Hj_yPZ@NSRI_GOf8$|!6qZE3`&7%#;"
    "AlV3Q<-l)XC+Cw+z2-{qbpAB8$XdkSszxVzBi7xBcQ>-H{qayp8nN@HmKnJbLvO^<%byIHMUNsiH)86I"
    "+mI=m6Rc4<$M$EZR(@VZGQ7AZ8aQ1ePPdU7f&wJw3o)n^;ZYI8Z^ZGNh`neWd(nvRHZa~s477oRHsYBL"
    "OtUH5KQS0XtR(h)sZ?5M<6Yf&TQ}adC5=NK8rh{9-6v^mmTGL4YA{Qk+t@f*r3%s+7y(A;1nqU-Y6b6q"
    "Mbm%on>4m!HCVA4S+UMou^L&i8d<R#tXSu^jyBaafe9~v6FKB6EC10+uKx%zU|NT^{<9;43LPjyqHF0F"
    "qHEL2;8Ifb23BVQn4xFNn4RN`_+a~m2_WcW16Um;(dXaI6=Zm8bmOH^G<Bw>59*y=@53WJGs3_9<%J|K"
    "2At&TG(H|n7ipbNzD6dt#wNDXvKV<>Qw16WiS|ziM8S?>Y_PI5SlJq^Y^7@A4;$ZQX<AO1CQNE)o79@P"
    "%5vr^OQVRKMiDvXPXpON_t2zvYnx8zaY59uj)omEH%-LcG`$vNUqdyJ4RnH5PkL?Vp6%Sr+*M#%gc_|g"
    "jWJ8=^B*j!fiuLqP$L^&Nx%-44X-)L;M%YPPhJTFZ9g%|NyC#gJV|^fvCopam1-!sK|<&&|G5BMpd080"
    "x`XbZd*~iIK__Thmzd53BTVLm309cQ3ls9>j!+ANVS%nr#KKJy+IB$eWP@l7CA(lT*#whK+HeUfc1!YZ"
    "N!~4`1PVOra22ej<aTM1m+}5?x;4-m8I6oaTs216CL*eif&KA|$6$`Seu>g&7cX`+C|IYNFOcvv+jGxh"
    "&utXo-4Ng1%%cn7=dm?_ASLC!F~ieRKM!^gGz!!@515>rIjWqJffpk$ktRi=fRHref>lFE*EvGE8Zp;D"
    "&O;4(p>YbzEr5i^!S`Sb(olwMIt|ePX9oi`M1eJq0&8qk-ttVBfMi1{qR*5`cO@T7Tc+j)Q*$9*b4|_g"
    "zw>5TcI6v0Lb|x-<aq8TOwMOa&dt7j&r*b%henhzz(yA6GZy)V__0D$3#SWnu5hwurd-W0(<^V!*^|ks"
    "bDpQ5HR8T2*JzcN#?PiTnwVr78KxW7Fox;#d<{9?8}U4NBiab<7J6Tnp_lb$M;gn-J?_XIMZoRG15*Fc"
    "2b(ZVH#is@ILiRRT77|JN736KUn{q^s$4B+=<r8Kc2H}QNpD@9#H|6pQiDzV>`1l7C$M5e8jRfM!B*P_"
    "74$r0XOK*L9&S;Dn>mds5c)D*jzm=LJ6TN_#|xH@WYjn1BN?mHJJYQ445y4sT4!I<64V-NpFsHG(7#Lp"
    "GN?AqHDMrY22Z7p>ZQ0bAE_*E4Sk*tR8E+Nh&rH1d9#Wq2LDGhT?^r-8WM%Cp<prOKjOdM=*B=aT*`Mt"
    "<i|!|46vL(kO^$1YL1w>CSvBAkoJ9{tcjDv2gwFBMj9aRp4tokgNr2_v{I77;TnWi!r>aTHf(oiqTe<;"
    "0UWCmgZtSA_YGOhKx|cO!U})J3f~;(S{!Q|lnkYS%okvZnC}~z?>AB_ZaT5z3Q}c~RM}8NB2SQ`?n)xr"
    ";2PWg&y!y*X?&mTJcSliAQv8Gvn{63pRw_ml72Oyp~PU{L@ZuGyeVw>KTq}1NEGBP7SBjWqAt>mq!|Oy"
    "oJ<ez5PJ;H5wO?99kcVRFPSy=BQ!W88oLb6b{RC`GH3|uYw!-7r{V0*4fO5w#jTdOSE?Fpswh}uAZH^I"
    "-tZx8^hn8;RP0vR(nKaYdq^jCG!!hx`%uWHCxvXo1Fi$qu{*gL55{heWgoZC3TS_B69I+iX+Bnh&dRer"
    "7|jtr*f@N!Ib0ZLoK8UWe>>V-F(;S&4?Fz=@Dd2?XgJo9&y%>5wqqW-oDa-h7>+K4L`+0tV;@C>kD{@U"
    ";yiKxc1KvLTWE1$^yu?%<AG&kM>9U$j0mx44#&e*OP`)VOS~I?nVys9z*Y9XFx&mwmy9N|nY3ae<&T(%"
    "O(?dYv^Hh7CkLCg5<^=Fq3x-1f@%_Q7x@s?h&n?#aYN!xWJm2Xe6)lJV~Yt>Oe+@Do*YNrdU3ayqOjGp"
    "SQ~AsX%TH(8xva_6I&Y-+b||>5T#7Z0qICB7=aCKJciTyF&qatwec8^O+#3QWEjFG*ecqwCEQU~BJ4S>"
    "gZ$bW6LKr%wXOA{Mk|A3`IE`5ftD)E8zwEhvXvs+;vd7_RQ`;n698pFdf2&Rw*M8Kp;=rtbdYYm@8Ptv"
    "Qnp+=YkgF<J}F!A+cvJD*`Q^6R*NaK^))oV%hJNWTA3+ZT{CNOG_`n|3Qcj_(=y3WB28z6fjae(hOvk>"
    "Xuni}SXA~x6@+ylJ3y)HSCIqN%#f?;RRB!2#Z~0Eww9Y_ZTNyVge2yit%C>K!y(k#Bh+F_Xk|%g?cZr}"
    "@U&rBXlwDf7E^5NIKtKrnARSc*2dY^#@W_Rmko(x7`bbVT!d1zN3>xZ-j<SbdcxkF@D`X`s@H^M;3U)!"
    "V`h7r`E7+<naEj*Wjko~9!on6TF|m<mS2@w$8YP?w{=8U8xdWt3<0eT0X*_kXwB#x8CWBJc%_YpS6USQ"
    "HWYrWsjc+=LcxoN*@3Ao>U(STz4b9(GHG2%Pj9WKx6;!KEs3BdCaiT_c>!ExO_4_cz*bLxV+5_yZ)MA*"
    "*@oiYqPWY@D?4xdp#3vv$-_KtmgdtwG$gZz=%Kj8c2?MN`ye-&wc<Yi)4NxSefuAT3%0r^)-u5l$~yBS"
    "9BhVzJ*;sQBzTm<LCb8z#=I*-(rJ-%TJBA>M6$KKfWym87=>G`11;8pHg1=-zFih(><ATYZ4cNwwga)x"
    "-G~VAX#FBf8{r)VY62NKTgM|2=P5B4ZP*7|2d=fpLoOwmkGkUtfGzLqfxSB+LkPDSAS7H2$QH<FyZPP8"
    "HLEZqj01$S+Bk39e2XlHN5cWK5g*(7R!KWVyT5vOOep~NAsaJn0MBVy)83Z_VLRLmc-c$AZo>f3ZU%r9"
    "m>YDA<kw?w^)Bt_=T!syHTV7T;a&6joo&MtFD*~JwCL&WwK0icL40rP=(*O>b1gb(iwf!h;P%Qyz|9>@"
    "w*kEI(Qb<T9gyV|f`Y?tT0aZZI_9njJFYpqe;y}$D7II>CKt3&&JsWj!!kuqxO}|^!iZmQ1}p*Ao^U@N"
    "Q%aiLCd+N#W6D3GyY8${pFZdpgshnHLnLn<3$%kw$fN%$AVe_Ti`)^8%l;^fzSLf1ABPoZ1CR23SjLKy"
    "fuogyqs73{`b@yU(aLt)hLNK6BM}7(&g4h6_64#p5BaF%qUt!LIuwRZ+CeAnpp!Dtox%}b5T)lSMm$`d"
    "6oihKT)WdxA}O1q)r1p`=tW^j>`vrlEp~(}bP=x5NqZ=NUTF&Ju0wg~9K6suc%dUaq4UF<-N{)HRl4Jy"
    ")NX$$)gYBmQ90(v?qu_0=Sx5Z3Lu&^<)bXO%7Niss2-hFj}FzN3)Q1LISI^J*ZDgWsv6}(cl)~<;ziVp"
    "D9xyxW}QMCIz464N$cpOb#%3-mbsM{(xHX$Vn7E^>7-?pKU6A4(LGQqx*D1no;qt4UG1Tf<kDfH>ChND"
    "G=>h1p=1Aawe$Nt-w-O9Tq(m?+d0umt>_f}&^h3(bHrOm$Xfv}(6oYedgmeE0wfQ<^n(sJVJH2dLqF({"
    "{<|8}Nq3KPx}>4$2=<)>`;8o-by{{2h}G4@(HWmy9e%_PalhlKiO%6w9k+%$i}nTZR=b<eZ@I+iJBMeK"
    "WY*5R5bZm^NYRB9-yy|!!C5<4Yp2nWT;C1w6A6i(>}j3ffanG`)l62cDM<*OoXedrh;)ND7apEY9^+0P"
    "<4!Ks4wq^-t0)NgdKb>cZq_S;1*y#3W+MlXji@`)G&(emZl=BVk9V`vlRRoQzd>v0cCA5`NNwn(HWZ3R"
    "_=&fUw;)P_-T<|ruy@uXI@<y}u7zx`dB7Ab{&#Zd1<2$L99-S=E!7S6aP68w2RGye#%&A%po7fN8V3pc"
    "9it}UzKcOC(mmOe-_z_AQ_{^&LW4NV5ac_nUL9NR%D8|2vxRQ`dv*c?E5I(?cXIs^c2TGo7vLM`oln$)"
    "N$%JY7J%jd9m`J%89l279s+9x-IZbA03Srx&c?bfl!ea9Licp5Wdo-GIkN18>Ox`YPEVCE4!zm(p)hp2"
    "!f>SYvE|1fyF+E@99Z7L#s&zrb#PO^vX;G+Q5ZTY3>^wX2QMo;kl`F@#aHOTG)p)Zur-8^J2X2Gt9SC^"
    "jWE}aFxM{piQPcRVHc`IXLn*3?!?Y_ZFXn@S`LUI9Bf|ff_)D$e5fJaaEJEZ>w{+eVy^8+>;_NahIoM$"
    "Fi%HHTH5C)C{jkkpPv^xY{E=}cYh0gftI*{s?wpVbb5`l^I79){ti8)^DDuf4<E<!b|@X4txDbDJuFx>"
    "5?*zUN7au&s4mou4h5nk@VDF42k#6&z#WeZcfJa+BkJdMx*QNYergXWIirWiQrh>JR5MWCD(FHLD00jB"
    "NRyo?R<7Tq?EEHWhn(Lb=XYCJ4XpAyJ6CXy*`kCW@94M!xQ}?h`wH|rlOAX4sjs&+s+Z)`OYrF>>-RGJ"
    "_mb;-1+MkHo4@}RtugI=y}Cb<fZx>y`Gk6fX!gE^)JxRwE$a7N$>0C(G^BTVZ%MyL((mOa=y4PDvZ|Fo"
    "kX{A-=|+0flbZ`a=(*9~BY&3;y+XfJf->wg(_i;>ruaLB-}N{|d!p2O8TNZc6893fdx_gb>h<=M_T1F("
    "xvAgR6h%p$s7{oIRI99)3!vXz02-p2lFpQr$ehT{5aTi3{xT#r9{au~1qn2yH630lZlySh<~^c$U;p?<"
    "f*T23I%PAuj!Et%mG?6C_m;~0xWeCa@ucTXxgLK&k3XQ72;LLn-%AAV5y5-KEce$hP%EUy3IV+*V6PAJ"
    "eDAv&eVFHao9FvG_OJdiGzL0=&Oi^)8*ssc%l+zd;pjaQeDC-Adz<I`+8)T>*7yxN2jnp}W9vgY@44yF"
    "x2L<NwLLx5jl($KZ`S_(C{l=MuT!=kU<(r2dx`8l@s9n#=AB8*UY)&H{ft}x`43xW`^3Udtn7#?>>X9u"
    "Yr^o3_qxd4KfS@$I}ERPtX_{O-4B>^dkND0pr;jk<mP@ef&)-1#`a#u_L6{ABx8F|P+afZ*1d0A_dH+T"
    "$Mfa=E*AG%N4f>r&)(h=iu)Pqr<dH@(?^-Bdr7Rl<lSD<Xpgwtvl<_^7ZgAf=5}0}q_}q7$F=Jo5w}Og"
    "?Iq&&EP5Us?|E;$59@Ys>vkX8!n?woA_&$3D>=l=-g`#UVLw2y%Li|A?<h7@AxXe7<7d=*o>A)&DEq;`"
    "Yi&9_t@j3Wcv_DH*$+HW+9N{tgrjY=C)wi`n-&{$ANR8R!A|Dh;$6=*sXjI~d;cj`thDbX%J!i7;$$y@"
    "%+mrhIDP(|5w!idxBa+BNGkx%`(P$3$U8qjpf$*n*?OCodz+Vgq_`eqa-kK6Q-INpV4~}ID4@3up|=U4"
    "x2?GM{p=nwuD=H9b2vg-dStjB8LmHMxB}fGkyqkWr9fQUlml7}Erzsj{5E(W91?kL2Q+?<ZhZCZ$^hHX"
    "r=Yz7ZI0`a*81%QaU<d%1fKO_nC)$t?HwuI&z|oP=z6m$LCKhx2~u9q6K#D&qV;pk+%GXuGmLfT>^c?o"
    "9;ct<9O5%tJGHOxSKcm@RpvCem^#JtU*j6Uz!$bqrp@wCW<7jx!{6H&_rV-BmLq5Vh}1hu3LPbdj*5vM"
    "b&+nAOghFzx=~WW$a7vJ-wGT3v9M86!ua`Dr`J8Y?eCT9djW8WQNn5Y1DR-Rf~Eq?qN8QeF+R&RN*W!J"
    "Mn^8w%fmZjf*x&x9{s$OZXb=7L`S~CH9DeYJn<C4XhZYp*QUn!3|E09f{h5UBLeKmf*lcH$9=Jm|3m-R"
    "kbnNFG2d71DCuLAxp=JkNz)-|c9b+b#%xHQ9WBp}I_AgPKWnHlTaBBAN%VhEm4?1%=xcuV&?B>s4pSLx"
    "=nTa`+MYZjm5vTE8EdB?fuzn+w%`%V?}+8M&~%o?hsZf1auzy8mB&ZhMoFS0fjy%n(Xk#fKMxGb30u^N"
    "_&E|CGWsg<2GI3;oW>itOo0MgP8~x|E#>zhIuu8w($P|BX_D7`*p|oI)`>AB(~(G$5y5oCv^-h}9YY8m"
    "Z8aX_PTvM`g?Y<(YYj$srYRD#L?cLK3;}jTfE^X>Hu7}UNc7rB_}WN-+88$k$CeG=@^J1*6k7Rv4xj^("
    "b~F;0w*7Sy&vbiia|eT!|Il2;gN<Ai92EjKw(KNgy2iLEI1<P;w(cBDK0>udM@5Z%t!g9|X>1>^MC*&V"
    "nXw(*mzlokb4KL{33el4DFvvh;~{#cnGZ_f`v)Vz7e@ltYm0St17dhW+#DkqVgrrE^5jzAc*HtvPz+3n"
    "jvH!L`xLh@w#TOnMn@uyJhnB4Lw_5Nmjm*A*;PN{s~>UJk9g}#>Bw87KYdjo9=;mGBR_^qezc3dlw|g1"
    "fBKdx5@jP%y2r?G&y3+f*~>bh%aPJXc>RzA_G(4&C^N<sOQ9RN^EaX{j6G(gQ9tu}tq~68w@gQtl)tYv"
    "4xB#DY}RxN&ZKayk&CC}5DB!1b<uC+qTeXD&6plg?rHfS>FYoM%!`wmF<0|OU(FkjOLn6#*=^WIV2o`d"
    "pR>&(2T1LWWQtm6Dt)+J?wIcwZHPEhVywA!H^xo9k(+u#lA#3l)kP1J5$AJnSghRN+vwgvq;M&Sf1|oL"
    "%zC+Tv>$8?KiCMrFuRcIJwB=4qugSnA5kmxXn>=8HefnIo3A+pZye%n_i<*?NIf0rNHLin4~Q?Ujxe*K"
    ")}taL?y>^3&{Q2P^^3nbQLrUSK~xw*R<Lnkv=43MPTc4_abtMPN*ZZF<FmJHJiKMXbEFlhwWVdj%9<s9"
    "7jlGV1o5Dac+f_8G68YpgI`@TEnw)vUNHK4;)us=#8xu0N-<-MILr$D+5(J5xLjEvYO(MkgTlx{=Z^QN"
    "E8b(A26l~&u@Tq@PmZiHo@d)p&-<zKYZl|NXRf@;J|1U=qu)l^8^!^G`SU4*^<d<&l+lus%?Be+_z@rc"
    "i1lC`l1ZjCMSTE4dtG^1#lFTtHHJ%Rv`cA+n4ZBgB4dbX!-079^*Ml8d&dC!K3c(=K<Jn+WQ<{m7;T6c"
    "!-X^s$9VqNn|suwXdEB)8E1kC$EU+~&;$7+CoPKu)^(C;ZuB9G31h@&F`gdOE1ShAo5hHtJK7~an)Ais"
    "TL7;P_lDE>$!0OyLQw`GN2<{0e0Od1u^lpP7%s-ca4}-I*zy@*sII|Of*;#NJH{8%Mt>n~hhiU0m#2jE"
    "dNLypmsG*dM>6^p%jqMQ)Ad+R@iDcn;=3^=i;t;|!Qn8EjL-b-c;#*kFV|Qf10LHkINpvOx${$QD+x94"
    "9F5Q6aU1HiGP8`>Xh+LB1VoR%-8as`@eiBIXeZgo^}A8m??!IAjS~S>BRAbfZn}-Wur~U_+UUDz<Mgzm"
    "e9>(5MYEA>Wuq=SjlM!wDn>x}1+vi>$ToyRhA3nPm<@37`7>rCx4lN)_8R@svT^d4H%8y#8hwXr-0pBK"
    "hL&q!3>F_38;@&UC0&0>PoS5=bTDjR?K*nQhad}31=4_=17TQ(E)cW*X&9f%P9B;-ShaE0Ydo%cZRL%g"
    "8dwmQy+&X58vPNlF)n*;P_zTa5l_;O$9r_6-lIFvY&0Jrn6HHibJq%b^v&F1|A^SpBzL9mpCGS2Z?~s="
    "VjH^mc~6se&#zwE`22rQB4l$*cCRJh3rl|DDoL?U(yNmUZ)rEYQMNSgq)5+X4{)*vIGw1XbrtObumE(T"
    "J5gB_g=ZHh9-L05gQR_U>=Vj(LKjc|&~|$9SW4?iCe(ahs!w{MJylXkQoIS8##i&|R2{Lu1+EqSE9FHY"
    "XMUsP?8~q`QNjx)q1l{kHcOH{Ey<piWKTOk4YEYeB!RO~D*kbH@l+>io(E6PHMUJ2JmFc6#Ir=<BQAiZ"
    "X#2a<>RL^|xK>;HVf-X}TO!IV=?+-ZOI1mHFTwT_wzuS?FV!&kg1GS>j49{_fIXFDpi5`@g_Hfl$<dnD"
    "1FZ%oZhR%))Jwk0wH3j_kO3uOpF?l7+L+`iPBq_yO1{fg3c!fr2VTyO=S()(Z7;s8LM-hjU+_v{nL9_@"
    "LWV7cTTby~F}xL%{GwmtbMDFZyo5dP&~r;EgG>H!r)3=7kcQ#^{O;A`*w8bvp~<nKDPlvDjeN<k?d=E)"
    "!E7$s?w9-qUkVFfX~edH`X`vTQtIt1u2ClQ6t=%ZS5JhNCcV>_^s%$Vqj}{ob6<c_HE#bA;x?R^JH7d~"
    "{Sl>BH&=P25>KC%A0%BAAZD_ut+^x~DoeanmUs{=@gi1=7q60^yh?gjskDd`T4v2)l9mqxEmz=Z0p?7d"
    "R!GzKiVMB@G8p+HfOb&ogAtk;AUTx_y+XT{Zlu>}YFMM&a~;34x(1@?Q@qeoFt<c!YvQexLVw&DTWqKs"
    "<f58u6cwM*P20-{W}AL_AT>NMoZ@|e<Oc%M@j^h!TaVcwV<15&F;&bI7mAZ_6eljvCLU``GN+b58c=RL"
    "l)n$rgBWt>PkW`#*OUsjD@2J{sq$A|sYDFW_E!)4$mW!IDL)Mrq)vf>gVRyRY2cugz?dX177~szaIPZ3"
    "rNKv~m+Ro85&v6~18(~xO7`#lKU69AP#PJ1xrDYqqI@>S(Tc5m|0{ZcX2IL+y!|p$Z&-tq&B19Tn{#QK"
    "jBFFEBxz(pk4FK{j|N_{WCLz28g~lCxj@eKL3h8KZVWVmSR^T0C^B0b4>17)TN?ecMk-<)d)Mkk8oivY"
    "UlgPcpm#otZy};?8NVM;3QXl^4Q)O_v7>T`cNu9e^I~cFPvbO-ttkIl0g{u@wT#S_eU@e)rCGhobSkS$"
    "xrS;W*#^0$%7<I!b(Cvr7${%Uowbv*DW`^8<(U1<?B@uN-k{5YC1Zqr6!a^t8{9gKJp~yJxI%Ne442Dr"
    "xy<ZjCMSE{WWGqeA(VP0+Hr$i?shPy1MErZ8$}!H9@hPVcwBqxK$lH<It9=ei8RcWS77!E%v0^^{B+BP"
    "H3yW9owGq^1|FOJ;`xR|M%2E74GyH(;SV?D`vG-<bq6>GI3A$KZt{i`iy%@g=@q%;SLBin+#5oA-Bl<+"
    "#chQGMIyh9m1Or$$G1ZgXS9vliRb20JU5qQ=04hzAvpo(s71bk<QTRMIXlf82+C|)ij<~;mMd?fC3q>H"
    "iA%g3SLh1bHL<Z2*p_^YJSng(5!jZv9a8953yfjgPFw|-Cq#~rq$k{xOuI?<zLP@F5-UL;2Q3s$nXMPX"
    "!(rjrwG(#jq-)<v*S?di+{q{1gR~^&+|~E&?2Ew_gWH9S7SJu9%TkR?Ch@Ra@^?fNPj@Gt?vD2slb?Cp"
    "k&5ax=E00`;p_m6`05wK($lq$l28F$>B*3C-t8zy0qEH5^n;aMpDM{94viNW<@v07dQ3cRkBPe$sT%`?"
    "8#mYvU`8oRO>#_Lq1`1E6!eoqvIW~qBw&@du2S25v$Go!?{h>#xW5v&xKo(Plg;Ge5W^pqo+9j0MA)Uc"
    "MUi}qB8B}s@g382*sGI$!@QR1HS*(U4a1e0_`n`Q>tw$^e|~orAD++xxg{Sgv+$%PpBLA&Nv8KFXMy=%"
    "&+arTn@!&vpoMsCD+8G%kw>F@_N@!Q>sf#?S~61g7=WsF)ZdO+?!S()ta$&;*0JPwko-(!3Ts#Li?xaS"
    ";K}#F6UOs0Hmd;)2&b?hL?B5?b&K9p^mWPsynwt!-+uoSmXqsre#+T{^-l>$GLJ_x`r`)#Qv?>4WJ_$i"
    "P;Aq6x<%$U`e|++Z1q}u3;RSr*Zm&@8L`}q2ZH|IjqOnsMl&1O)02VyTpwFM?v$rGmbXG06T)~Q7idq3"
    "4DAU+d&1D3X0Q2QP-|vR9b4X0Sl*ML)k}U>FCEY7rFd2^9oN&7ucxQOXrFAf-yf0XtT}d?!ye;Z*>t?N"
    "m;7$v-io|iLM1FqR<@`dk}`sFmf17wTrGbpe!YvfOi+MM$fFb8h&)44`q{~~zw_xQEu~DrVo9N9p2*<z"
    "E#<WoNNrTsi-Kupg)U}77xRg_n2gK~`u*=ttLo3cW`1Av^CC+7uatfhtqID!l9X+M&9=a1M=oaW&}6^R"
    "pZ!9ACMYq-t#<1aSr2MSp<tg$oyb=u#I%~ZSCeH~&9ba!EyePOoSLVYCW*CKX4H%sb^p85ty4p0Xw8YN"
    "szs#NsLQCD@ib=H5AtI)Nns;}X%f66GFH{h#hYA1nqrh4!<c1c%`&oP8CkP#;bd;%WNzVPY_6GGIGL|*"
    "<~m51esk#tf_^7~XxU%00wgoW){L<=`{}G)W01-+pJigxvTUmv+iJEgGh@rlwq<5)nK^8knUI}~DKisv"
    "xY10!IY9RPnC$1PvhA38w`0Nq!XjBv2^Bd^j9WrML<jDFcUnCnq%VibB4e@0gpg%I$g+dHvV**`9%{%P"
    "Md)pY?axlhnv*-6e_8j`GZ)n}cf>NUxo6+g$iAo{ojHI0LzN9M5Mp!k0LPH+Xwz*)vRtX%l@F}Wfva~}"
    "j>yc*W_d6Zz!sk6R?2$7J=;r@anoeETrz=PS-z4iXGzB2ka>SShdnwce+(u|>&Q|&a)^8xnJ*)VWn!SR"
    "rH>8BM%*Yg#(W&PrEZDVS)<N9{j3<Mys3lmBDBU0Qp=<t=0iWsdJ7=)7C@G|n0cZsOE=8uhI!7ekjyKz"
    "1{zf{6a19@4tb7u$g{szl;dkf8RapD@|dM4X1yhl363gc>!00bRx=Nhl@!S-q9x4f7qjewh-pi^%+@aR"
    "7kKlRt9&f7<H(GYr8;JA9c1Z_c_yL8os}(3GW-5imgbm!e<~|bHOCubSuXGV${S)?>SI1WXIYR81WoVa"
    "O|T82TE*XQKo3B%QY|q8WpvAoZkeT9`d6Z$vQ*`as#$)0K+^!ko^0oPW)%g$xporF^OFjhtwQD}6*7k_"
    "K8FTb5;jNf(FI<gu-0IE<b^rln#!SU<`eQ}K3)sR@w`}m(loO_Ta}O4-5f6nWGSC{fN-}4_e$>v<lzFz"
    "wyYr4H)*EXhYM$mj~1@5WK_(&shD*8D?4;F$rklMe|2-DTjZy=1aiD3kmH$v>}LXUya|x~CcqA{wWc)?"
    "@7L)V5JPg1WIqy+d8;f33xos4Q)St2mFK58usj@#VT%J}i29}t3?>pY`t$Shg}tKZ=D2;9efun<)#i8<"
    "-$u}Um1Q+stIgM+g3bUrvLKA|qFMG)aHXLf@vpuDn|;I+>zdI+GdA(;5RL5nXxV<t%-dzzXN4m(a%igA"
    "nrcQd-75CvqM7Ch>k5l*_GN|~9}vy{faqc2-3`0}<;a?kyzeyP4qzT&ZDxP1%+ica9r-CXD<830hsoF$"
    ";~bh}j?k?9<J-oIH0Za&jTm=niCJ3WM&6+x0OI{9i5Vp^%dohS@a($vu0tiv+=?r}K#{ZQ)c`w%OLBSk"
    "stkQGTVKqFzL>2qZkYG*JVde&5=H%9YgQn4C>uIbxv0aRZPPoJ_Z;5L0-&c9=mK&va0hr8XaLRuWFvAR"
    "Wxw-YpeVNk=nRYixEiwGec#HtMmfo4r|821vP3?XAYXg6`@r4<I}ftFR~t#6!nPI8gdFd{AI^l$zu?Nz"
    "rzR^@n^%{v);^3JYnS8Q_Z;uOZ&(g&1zHVoI!!1+@$P%(-S-?{ip}w**i6vg=D9HK(Vsp11MT+5k+K8%"
    "vIF@tVSAaE;5XXKW0+r}Z-|o9lPu{s<MaMze3X`q`33jvK)~!b;_aNlH?xryw5=HS))dGK?%^RI9y~i("
    "FcU48x46K&k-81cqv?Hk1Pf-~fCqBV!x)~QK3Kb@P0uJ(>qu_U-=pATrjA#LHr&e<$wD!RhchY@gP0=*"
    "F%yHhBc7bLO@VpL55(OY`tuX^0_am%sAsVopV8ls5{`cS{BPq8qq3KGM=pjg0il=_tUAW^!&CLNqRi$d"
    "v$^=zZW*}GzxPKvx_o_FWoC%esXk^Tj?3wYgWhrY$>-me16$pq^}r%~{ggRYPu)LZm)N$vah0DU9<yEM"
    "fR;X|IvgE2KH^&zd3`;gY%D(~bZxV_#GAI+`SoPl_TCMfSa)w%5cA2m8URN5HaH8@{5As&0d#@D+24Nq"
    "%LsNwo&5cc^|C=5vKON=q`h*1N6h75ExNq(>dQM~Glk^L5Q}+@n>w?YY@s#n(pyj4M=C^WpyDNM6xY|y"
    "$C&_<QcR{PQBw7?kLz;M|HUL04$nm=<r4PR0wf`iB%1UNm@NMzL7T{-u+zn1r;AkHMZ9S_k(}ZrGcNJS"
    "=)!5a{RM!gx|Lt~QK!vvx;eXsW0CH>VE<hFg_VVKcF{4r?5k%t^)36lS+r0WjGqg}&qWIM!i~g5*3U(D"
    "_(ezYqBn*X9m0#uM@vmP9#0*s3(o{CJQK9=Owhs+yI`$cYDmVy=E}w9%0=eN^2hUGYyV<n|Dq#$!N9O!"
    "v|KDqFSRp}UN*=@A)t#xK$mzLXu&MGV3u5LmMnlqT~^5jtK=f9<bqXlQN-uM_e>X}I~PZHF82s3DRtsP"
    "Tx`l*>Y+7&cwZpo`z6eo3+BuPbLN6MbEBE9;M9xFU<G@EI_K+*WFy&#`;DP%d5X4NYI~)YuzD`GdM;ts"
    "Trg`cPwx;fZLMQ<(L+~@Ub<Q|j5-$=zxlB2&;0C=A~P1QQZAyfe=Xu8OML8RfpIKYycUTy3sTME*YlUb"
    "&!a7T9dAP#0Fq$dS}<=dHg7GMw-&#*zu3IB5Gk?9mbKWHwS0*MJcPe$*#)dc-I7sikyyB3Pg>{%YspIq"
    "Q_+H{XpyOC5d&Ya>@FB~7v!zw8oYbqy2j$$8U^5}E07E=i$$^piEN3v^`3mPAT72aEypUqmPP)Q^5WvG"
    "P%a_0EyOx3A+aqM*p{atge8IymP3eJ{PO%FQEeF@)Cie;7KF4#F9I#zb!<9+pRUlEhI|EZEBu_#GDH)I"
    "*IfLp{9+|>8K$5UfQw}a2M^58&ojrS#!9kyPaj(vlgL6i;bN1>f}?-&mPa&;n4jg^UZ_!`+Ol17b2X#)"
    "+~0gJazS`ouIYRr6y+oAxa79PCAY==WHW}ilV&Ja4}+cS%M~m1a`@Jl?W=8VfV9@EFs`a?g&hzroALp1"
    "9u#)<l$9yFPBCW(XA8z=cl)xr+d<6f`dbd9@!GHb`Nbt})RdHAEnY&tTo5mpQ1BNk_>0x|1@(Oi?Ri0a"
    "UeKNwwC4rwd6D+KSbJWkSTO!#+6Eb*)#$AYI%=t8hL+&COvsIkb+`qcVhJ5?LmmOSU^9$Eg<DYJmM2wh"
    "!;-<T)EG3gH}NHf<+z`<OxQG+Lx)=U64zq&=c%b|s_9e*K>a90OK2q<w9k(Y4X(s-928#PibU=tAl<Cr"
    "j*Tz&XvddR6(vWI+?N_LR-4>!!IgvJ|5J8&(=dxU5mOfrE`LFJ8)i90zR;b)9|`>8japJj-0~C6@qw0~"
    "IMD8#XZty%0n;(0Q%G(QoFEFQd$oKom&o;Si72M-)n7>=sP1*X?Q_L>b`AfCy&-mn#MN@&EXR`YY8TYH"
    "Psw`H|FwBm_RcEP*ec7|DrtX}S!~s=T6MOsCo+<+6V-|8MfIYRs00P8+x|BR%AzxrfA`N?!YhFf>&d@Q"
    "CTa9UtJVKidwtz^`l^k-?h9zG^?&+~48KZ-UlqWxN`zk}!mr{7t3>!!BK#^5eyu5nQq*^;%BzAAR^4S-"
    "wPIJTzZF@2MV4P{NT@~BB8onX)vOZ1SA_wrd_${{$#iF+Gm?#{c!}UEBKRs1d_@Fb<C4wlhm2OPL9Mz5"
    "wK`H@B~oAw(RxL+UJ<QVMC;Wu^J<xSby&dauUf4kGq0AJ*N~1^UI<xrwPDrGhE-q7SoOJ#)lX2byl=nq"
    "s^ZG~eXITTtNrz>x=zn!L@a-_R|eLugyePb(qm9&C015PHm{CsUZv_6TFXWv_KHt?og=wYi&w7P#wg@s"
    "TC(M82DzF;t`QWx`g2q3&Vp98hz~@qd?0FlKtidjGx^k0Iw~f#(o&etN<(#stqdqK^ok6<605rstGmWh"
    "@=CHt)iwKUGrx#~ZRK2B4~yc8L2+d@4iMV9l2KPoi>oY)tIxodBXG5Nxk|iTpWKM+=0-Hg$m)<>V|}x{"
    "bv}9&SnDkHaLcWmTkZ;`R$Xb8WYU#aAXfy=Lbpz@`MFY^PN$W_Y4zc>T8dmboL0`JHMWv>Qb>Ut(g(B{"
    "T7WzqbsvBZp~BV9g5-r-S*qCaRvpZ%ZFnmlr8PV{t35heLa1RpqRrb!tS6S!Z-3T9kGB)l1@Z#jKsV4W"
    "bPL@<chCtsL2Ha|{r0uNz}`u87oc|}Z{n4|{aKGl-fB?;X=jn~hV9L}^tV6zwb<JXz#EgdE17ut^@(l`"
    "v_@RU)rG{nls9c0A?{yKXW~)U+wjsFG7>-r$OfVlu8h@x>xU@$=YMT9&tKlO2R6Dj;Cgs&^u`kX?a!8x"
    "`<sPi*z*8)Mm`>{I$nSKi{8O>T5JJByus&jdi4`$<~CwB8>R6mMPRS;IBKIk>Eb6lazhp84%3qs?hiaX"
    "eMJ6hB_={UUj2=@Be3J-xQ>`J`&*v$^ib2E0yy@Y0>%*Sj;GfSPU#Ooh*1vu@2)HAd%o~Yz61fw9I6V6"
    "arNyR)P8CGh2Up*6iK@k;ipPFe#D|Zijt76)@t3o`U`WLC4iD|E*W|<by1G#62D)Z<Z|RuYbSpQFCGYv"
    "_krX30<k2=CnDkt5qmtyJCG3#xF5FguF#S2bl)MaJH%~=?Xp8MqUMR29$_E_Xd_bs34o#Mb0DjdtO3vg"
    "9diOG*iIMiQE71HklLT1_^$OJZOD)6aaw6P$XJ4{gl%n8sINO*+sajRDv0UVG4|?D-Eq_5^{VB-)k*%p"
    "_Yq|aX0e$+_?cZ1FLdD6@|^`dS|<$IaXaDU%L&KJ#qse8pOGgC)?rV1wLK-wB(FA;9M+HPbfaZj1+P30"
    "xO>MoppH4WHN)SR{_$-;)^2>6tdQbLx063-Q=p3wCghu9-*1k6FTh>E1CkP2|J=R+ycp>OcxQf-qQ?1A"
    "$Wi17nMArh*4%E^+zPhO(aj<21z40~F};djAtj34{Mc?mG?oQ{b{*3-00v^Z91u<8mfi-v(GqXXZye({"
    "F30}(b`@FEpZyrWNoFY&6M!~YfD6C7R_TDZn&FnBcVo)Gq0QaU<_fTqKRG{rpX=y6YTb?1_l>}Zk|cT|"
    "MeT;7R+h9DPU~6m7sYS>r1%CU5RZ;uM!oT!?1HG+Hj33~w6#f!aCpZCtw5^*7h1c}fl6l6jRD+u1)Bv8"
    "PGf=T*7{+;0zQN|jpU^pA>9S=F~32KUARnct&iz%{zU4{-$>n|4CGViTRcyC<I|`c^z*+}Db`s=?0W#&"
    "NDgSB1@#hTf2_vRfw?{}A8D~SX|Wrbp($E1>$lGH#n=08UF`(X{m$F@>C-(W<M-IKd@-Vka^?ekG~(v3"
    "M*R4unH(T0Z_2rv%$^j0a~n>qI%&?g!E1Y9EuZFC%bI$-stfIMk88dKq}(|ep<Lj%aDi_~HsqoV#H7wq"
    "HiG>QT>?9jIqT%ISl7dB&LP$OhHAdioq^6sZ=^TkW_zo~O1EaPLvMy+m}KU>4Q5TD<=?F3gT?UGeGH#="
    "4vW@bgud}b=z?f;XC_S%+qaos$Ru!^sYEt`o7V)f+P4_=hfAk$nHF#B$z%iF1dxqnLoG;aAOR!;oDi#z"
    "2$}S5tM`JedbhPUNs+aSgqf}8{FvHqt2fVTR<@dz$trYP&B|m~xvfD`hX*pO7`N36U_GS$4ICq-B<~yM"
    "iXCAfu}Eythat3guoNJDfi?PjCtK9$4w!GDK;7^!Y(VQAmK1`U1~ZM`ocf#&2!WzAGY!S)7A!aPnOi6{"
    "-;SX48%ErMeSPLIXb@<mJ6!=(M%Z#8yr4?lhF}Y3?L#)rkvkl<(vHDj(%Ud^qv?7e<^dM{zr`6b^p|u%"
    "7h_M!Tu0RU^S?|9CkaycHzed+NXXxik#F<o-^PI0d)HuaArNora`0uJH+-*S1Dwp!&>QCYHA9Izu0{Ch"
    "qSjyjRY)rI9pDj2vYaD?2;AFrI)Dp^2dO|6(D-^_Etq+MLuw1;o9R0^I`8-dsu@!e0FMGQ31iG0T6>#9"
    "Ld6$&o6Jsd2HvKlM>Vajef?+$?|{~XzS&85o9gI6O=IhEJW|nyA|xE8F-enULEMQq0v&fC8A(yr3^WHY"
    "8aXQD{F>qbD^M(fY492#QuQ{?k&^>rnxDoWmNQ+W&R`%@aFD}iS;pGT;W;>TgKtp}>|h!WfXN%4CNDPh"
    "G=vPjp=`KmxE)yZE;`3Hd}=p4(GE;OFUDT3Pbe$k8rs(eHh|{?VgdMzivoK(Tt7AXBN=>}O>grlWXg!%"
    "+_2>-VGJ&_r@;T$_}UG7_m#a%;Q#APERKz|uNY}J8q?Xl$Bt5nXYtiR{Os@PWa>rai^;v7vJWw@dHZ|!"
    "d+$%DD(Ds<a@+;*j-oyKIP^D?9OuZP_>Jk*M2-yLcDh}WrBi~=^eB3Sq`<R(PP9R{SIk`|yw@A8@tbg5"
    "n%8Ws#*NZKog~R@aS7y@B84``gwhIgf;Sb40zQAGgnf>unm(e$PIZjEwAiV>xPtmZ4fW*?qUn&Qb0aAu"
    "W<!i_63h230v;qX%Y)<jRbD^J>)Ql8l^RzGCaz=q9s#^|BPrbhpK+~VLbEOav<JS<s`HUwqgIVPT9~gP"
    "_zmEVrkT2_T>vqbiL2UiPl0O+!Jj8|)VQP&cN8{+k!4`zrJWQgxsl9{AIsdJ3&_O)!*lY==K;-$M=CcU"
    "0YnPweVkme7EP_MVHSGe_ZeCoeB%3xJHP<>3pWLmAo-NYTz>K!ngug^nCf5;$1|E+qtg?ZBc<y=7M>o%"
    "B@hN_Xaiw-ht9|c1=UgI2jg~)Tpx@ZuH_xdhK_u6$?oPIbC<5<)bI^t(&rba&wrSo4KRi|mh1qp`G>dA"
    "0I!+p!T@ub86MrR^7GQ1>f7Ve{LVEM-BK}j2eJTCp#B2ZGoI3&r_h}Xl0iHPTvDF=nZ^xy2Qe3t05R#>"
    "@<vimeTf9F$jj2&?qqFuhg`6~(pJDLka-+|I$Q5^Ajz%vx9fKF0Ks<oW%4xWF7q_Ik-BBWi)fZ@rq|hA"
    "L!mj-%n2a%cRGgdVRK5sviX2q<<smItR(Qy@R1ol@^A^0X^s9IY=q4jlnTTj8WCd-7(0ZoDRyN+C^Rvn"
    "u}!aD7*9T|<E?PPJ-HnaDwNX^-QX$yZ1R2Nr;9>yPe_=LJu}zCcXiwxdX^1CG_+tSxp1Q-ZD7vZuMx2`"
    "s@8~RnRf)^7SVQz$Usbd0Ic3F84>pwdebC`wSas&4be|z(-2J$!E|zD>mGc^u9yN~`yJ!()!XE5fY}>j"
    "_N;kM);zi0<kGuwdV*=ZUt1JN`^~l`9Z%-=KYpeWS2yAVnat&WAl3yBAf#a;q#*?43D40Es42R6yaU-u"
    "+zwbdUgLO-&`%MDJU#j267e+QbRA76vZMgqloV}&r#OH@5Ca(I(vZpbi&jyb(_If)IVAPCX0;(KiqG%I"
    "s<2~f1KT67Wx<dz96me6=N}HafAVuxf)JYe<_Sv1p3-pw$DBuK$07KjNKBg|VMJ2qTnF6)^wE5WcogRA"
    "8>d{+I=`YSydeM;00HPta(w}ObiSQP5_0;R<noecG>`zi8Q%zR+n{X7NInqW;d+fv#_gBJNH?WGE`}}v"
    "b%welquVu&0r8S{!ETYxkoJ)-q?P$xAuHPStK@IpPdmS|&E9GnQ`453^nSW{L)?eDvU$FsjI5qmDs&Is"
    "JKM-G)}yl+2%5*7U+WlFo)3FP@_Lh?T~aL2EA(o*i!wZPb+mUf@aD#{&fo4V9faq%`_;oj9?7%?KrDOn"
    "!|(-~V-!NbJHLj6_ZAY~R-738TX$l7@SYUpNY$aey!EuAM^<_B7j!oW8RU(I@p<k-R|aCr1}`Uz9&dx!"
    "bVwU-!(wPL!r)p=+~P#$u3~_<YslSL(es(4GbHmi_`G`i!U*#<?hWuz4W5`Uv3mWsOu?>3JP2P<o;J*E"
    "<n2q0%n&xq-`^gBXw-ugbNuWUAj!P(Xed`Iw|kdPVPR*7a$w9syzRe*-W<?N>~_az2>0Myka>H4v9rMW"
    "Iq`Df%QGGu-LlJ@WS24mCfSq2*#InL7$b)VE$3v&4{g2<(Y>$k9s&xNPmh7pX<ZuQ&5Q-Bq^Jzgn$PdS"
    "eEpc^s}J8d&e~FtFC=eRa;hji8*vLiVwgU^AH(#qMzaEn8J!Nq3?aUcw8L*M@Cl5a?1A{@z-EJ(QO>V6"
    "q`dEo|Ni-ZpmZ?ZeJ;IAzq#9F^{#X3emXm%s6a`O$t==K{oTjS-R7Y8lL5@|zrV}!tpCgC%x5kONt0?f"
    "Ne985bKo5_%`MC{@8bA(>@x3|qT~ICg2<Mumk|r8s|$?LySV?oc49IAyR9x8K>KTpM(_C1N~7I|#Ssmj"
    "^Y<gh0YI$%uHf}TUPXFl#MtjPR=nRIqD_h=+-;V4x9#HYk9`!#`{Pb5`UYts-@_j89>>7_;W1Fvu2~-K"
    "LbCAR2Tvfp=Y40gcfJ8tNPZvuyLjxK`!5^KtnQa41B`4Z=exRK12sTxI`Y&`V&1rAiw1}fY(;ZozlY)f"
    "?vEsbU7S5$s`>q0V~>^MUHbDK{rO!!(EA(<O#1U3$3vmhzh+P558K{5F6Fy#vlj#)Xj(N#lf-B5woSaV"
    "AiN$v|EX!ae*R}4DOQrOKxEpAEWiDQEHTqDmNAl!BfZc40U*d1?B>}FuLoCog}i9pbT8Lu(94M{pF-h0"
    "XnOBYdiw3q)8FINx}Tow$)w9q$s%*){i=C3#gc4IdmXq@-b_V*Ob66}#a!wLia}-N`2c#4nE1P&s@kvy"
    "q%k0_b#-H;``xbg|80=GDW8AyWN{wdeH^`8wY!CC_s$7*4^iYj+@k)RlpuM39R4SaHcU@#@Fd3;dfGv^"
    ">(dTGdAfV&`jy@SR4y%!towFi{qO$^ZBNYM"
)


def _load_embedded_table() -> dict[str, str]:
    """Decompresse la table P2G embarquee (base85 + zlib)."""
    import base64
    import zlib
    data = zlib.decompress(base64.b85decode(_P2G_TABLE_B85))
    return json.loads(data.decode("utf-8"))


# ══════════════════════════════════════════════════════════════════════════════
# Modele BiLSTM ONNX
# ══════════════════════════════════════════════════════════════════════════════

_CONT = "_CONT"


def _reconstruct_ortho(labels: list[str]) -> str:
    """Reconstruit l'orthographe depuis les labels grapheme."""
    return "".join(label for label in labels if label != _CONT)


class OnnxP2gBiLSTM:
    """Modele P2G BiLSTM via ONNX Runtime.

    Charge un modele ONNX + vocabulaire JSON (char2idx pour IPA, idx2label pour graphemes).
    Sequence-labeling : un label grapheme par caractere IPA d'entree.

    Necessite : pip install onnxruntime numpy
    """

    def __init__(self, onnx_path: Path | str, vocab_path: Path | str) -> None:
        import numpy as np
        import onnxruntime as ort

        self._np = np
        self._session = ort.InferenceSession(
            str(onnx_path),
            providers=["CPUExecutionProvider"],
        )
        with open(vocab_path, encoding="utf-8") as f:
            vocab_data = json.load(f)
        self._char2idx: dict[str, int] = vocab_data["char2idx"]
        self._idx2label: dict[int, str] = {
            int(k): v for k, v in vocab_data["idx2label"].items()
        }
        self._pad_idx: int = vocab_data.get("pad_idx", 0)
        self._unk_idx: int = vocab_data.get("unk_idx", 1)

    def _run_model(self, ipa: str):
        """Execute le modele et retourne les logits."""
        np = self._np
        ipa_chars = iter_phonemes(ipa)
        char_ids = [self._char2idx.get(ch, self._unk_idx) for ch in ipa_chars]
        input_array = np.array([char_ids], dtype=np.int64)
        lengths = np.array([len(char_ids)], dtype=np.int64)

        input_name = self._session.get_inputs()[0].name
        inputs = {input_name: input_array}
        if len(self._session.get_inputs()) > 1:
            len_name = self._session.get_inputs()[1].name
            inputs[len_name] = lengths

        outputs = self._session.run(None, inputs)
        return outputs[0], ipa_chars  # logits, ipa_chars

    def predict(self, ipa: str) -> str:
        """Predit l'orthographe pour une chaine IPA (greedy argmax)."""
        if not ipa:
            return ""
        np = self._np
        logits, ipa_chars = self._run_model(ipa)
        pred_ids = np.argmax(logits[0], axis=-1)
        labels = [self._idx2label.get(int(idx), _CONT) for idx in pred_ids]
        return _reconstruct_ortho(labels)

    def predict_candidates(
        self,
        ipa: str,
        k: int = 5,
    ) -> list[tuple[str, float]]:
        """Retourne les K meilleures orthographes avec probabilites.

        Strategie : softmax sur logits, chemin greedy = top-1,
        puis permuter les positions les plus incertaines pour generer des variantes.
        """
        if not ipa:
            return [("", 1.0)]

        np = self._np
        logits, ipa_chars = self._run_model(ipa)
        logits_seq = logits[0]  # shape: (seq_len, n_labels)

        # Softmax par position
        def softmax(x):
            e = np.exp(x - x.max(axis=-1, keepdims=True))
            return e / e.sum(axis=-1, keepdims=True)

        probs = softmax(logits_seq)
        n_pos = len(ipa_chars)

        # Chemin greedy et sa probabilite jointe
        greedy_ids = np.argmax(probs, axis=-1)
        greedy_labels = [self._idx2label.get(int(idx), _CONT) for idx in greedy_ids]
        greedy_log_prob = sum(
            float(np.log(probs[t, greedy_ids[t]] + 1e-30)) for t in range(n_pos)
        )

        # Trouver les positions les plus incertaines
        sorted_probs = np.sort(probs, axis=-1)[:, ::-1]
        uncertainty = sorted_probs[:, 0] - sorted_probs[:, 1]  # marge top1 - top2

        # Positions triees par incertitude croissante (plus incertaines d'abord)
        uncertain_positions = np.argsort(uncertainty)

        # Generer des variantes en permutant les positions incertaines
        candidates: list[tuple[str, float]] = [
            (_reconstruct_ortho(greedy_labels), greedy_log_prob)
        ]
        seen = {candidates[0][0]}

        for pos_idx in uncertain_positions:
            if len(candidates) >= k:
                break

            pos = int(pos_idx)
            # Top-3 labels pour cette position
            top_labels = np.argsort(probs[pos])[::-1][:3]

            for label_idx in top_labels:
                if len(candidates) >= k:
                    break
                label_idx = int(label_idx)
                if label_idx == greedy_ids[pos]:
                    continue

                # Creer variante
                variant_labels = list(greedy_labels)
                variant_labels[pos] = self._idx2label.get(label_idx, _CONT)
                variant_word = _reconstruct_ortho(variant_labels)

                if variant_word in seen or not variant_word:
                    continue
                seen.add(variant_word)

                # Calculer la log-prob de la variante
                variant_log_prob = greedy_log_prob
                variant_log_prob -= float(np.log(probs[pos, greedy_ids[pos]] + 1e-30))
                variant_log_prob += float(np.log(probs[pos, label_idx] + 1e-30))
                candidates.append((variant_word, variant_log_prob))

        # Trier par log-prob decroissante
        candidates.sort(key=lambda x: x[1], reverse=True)
        candidates = candidates[:k]

        # Convertir en probabilites normalisees
        if candidates:
            scores = [s for _, s in candidates]
            max_s = max(scores)
            exp_scores = [math.exp(s - max_s) for s in scores]
            total = sum(exp_scores)
            candidates = [(w, e / total) for (w, _), e in zip(candidates, exp_scores)]

        return candidates


# ══════════════════════════════════════════════════════════════════════════════
# Classe principale LecturaP2G
# ══════════════════════════════════════════════════════════════════════════════

class LecturaP2G:
    """Convertisseur Phoneme → Grapheme pour le francais (backend BiLSTM).

    Strategies (par ordre de priorite pour predict()) :
    1. Modele BiLSTM ONNX (si model_path fourni et onnxruntime disponible)
    2. Table P2G (embarquee ou externe)
    3. Regles deterministes (fallback universel)

    Args:
        model_path: Chemin vers le modele ONNX BiLSTM.
        vocab_path: Chemin vers le vocabulaire JSON.
        table_path: Chemin vers un fichier p2g_table.json externe.
    """

    def __init__(
        self,
        model_path: Path | str | None = None,
        vocab_path: Path | str | None = None,
        table_path: Path | str | None = None,
    ) -> None:
        self._model: OnnxP2gBiLSTM | None = None
        self._table: dict[str, str] = {}
        self._backend = "bilstm"

        # Charger le modele BiLSTM si disponible
        if model_path is not None and vocab_path is not None:
            try:
                self._model = OnnxP2gBiLSTM(model_path, vocab_path)
            except Exception:
                pass

        # Charger la table
        if table_path is not None:
            p = Path(table_path)
            if p.exists():
                with open(p, encoding="utf-8") as f:
                    self._table = json.load(f)
        else:
            self._table = _load_embedded_table()

    @property
    def backend(self) -> str:
        """Retourne le type de backend : 'bilstm'."""
        return self._backend

    @property
    def has_model(self) -> bool:
        """True si un modele BiLSTM est charge."""
        return self._model is not None

    def predict(self, ipa: str) -> str:
        """Meilleure orthographe pour une chaine IPA."""
        if not ipa:
            return ""

        if self._model is not None:
            return self._model.predict(ipa)

        result = self._table.get(ipa)
        if result is not None:
            return result

        return _apply_rules(ipa)

    def predict_candidates(
        self,
        ipa: str,
        k: int = 5,
    ) -> list[tuple[str, float]]:
        """Top-K orthographes avec probabilites normalisees."""
        if not ipa:
            return [("", 1.0)]

        if self._model is not None:
            return self._model.predict_candidates(ipa, k=k)

        return [(self.predict(ipa), 1.0)]

    def predict_syllable(self, ipa_syllable: str) -> str:
        """Lookup rapide pour une syllabe IPA (table ou regles)."""
        result = self._table.get(ipa_syllable)
        if result is not None:
            return result
        return _apply_rules(ipa_syllable)


# ══════════════════════════════════════════════════════════════════════════════
# CLI rapide
# ══════════════════════════════════════════════════════════════════════════════

def _main() -> None:
    """Point d'entree CLI minimal."""
    import sys

    model_path = None
    vocab_path = None
    table_path = None
    top_k = 5
    ipa_args: list[str] = []

    args = sys.argv[1:]
    i = 0
    while i < len(args):
        if args[i] == "--model" and i + 1 < len(args):
            model_path = args[i + 1]
            i += 2
        elif args[i] == "--vocab" and i + 1 < len(args):
            vocab_path = args[i + 1]
            i += 2
        elif args[i] == "--table" and i + 1 < len(args):
            table_path = args[i + 1]
            i += 2
        elif args[i] == "-k" and i + 1 < len(args):
            top_k = int(args[i + 1])
            i += 2
        else:
            ipa_args.append(args[i])
            i += 1

    p2g = LecturaP2G(model_path=model_path, vocab_path=vocab_path, table_path=table_path)
    mode = "BiLSTM ONNX" if p2g.has_model else "table+regles"
    print(f"LecturaP2G v{__version__} (backend: {mode})")

    if ipa_args:
        for ipa in ipa_args:
            candidates = p2g.predict_candidates(ipa, k=top_k)
            print(f"\n  /{ipa}/")
            for word, prob in candidates:
                print(f"    {word:<20s} {prob:.1%}")
    else:
        print("Usage: python lectura_p2g.py [--model FILE --vocab FILE] [-k N] <ipa> ...")
        print("\nExemple :")
        test_words = ["bɔ̃ʒuʁ", "mɛzɔ̃", "ʃa", "o", "pɛʃœʁ", "ɑ̃fɑ̃"]
        for ipa in test_words:
            candidates = p2g.predict_candidates(ipa, k=top_k)
            top = candidates[0] if candidates else ("?", 0.0)
            others = ", ".join(f"{w}" for w, _ in candidates[1:3])
            suffix = f" (aussi: {others})" if others else ""
            print(f"  /{ipa}/ → {top[0]}{suffix}")


if __name__ == "__main__":
    _main()
