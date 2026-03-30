"""Lectura P2G — Convertisseur phoneme-grapheme pour le francais (backend Seq2Seq).

Utilise un modele Seq2Seq (encoder-decoder + attention) via ONNX Runtime
pour la conversion IPA → orthographe.

Usage rapide :
    from lectura_p2g import LecturaP2G

    p2g = LecturaP2G("modele/p2g_seq2seq_v4_encoder_int8.onnx",
                       decoder_path="modele/p2g_seq2seq_v4_decoder_int8.onnx",
                       vocab_path="modele/p2g_seq2seq_v4_vocab.json")
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
)


def _load_embedded_table() -> dict[str, str]:
    """Decompresse la table P2G embarquee (base85 + zlib)."""
    import base64
    import zlib
    data = zlib.decompress(base64.b85decode(_P2G_TABLE_B85))
    return json.loads(data.decode("utf-8"))


# ══════════════════════════════════════════════════════════════════════════════
# Modele Seq2Seq ONNX (encoder-decoder + attention)
# ══════════════════════════════════════════════════════════════════════════════

class OnnxP2gSeq2Seq:
    """Modele P2G Seq2Seq (encoder-decoder + attention) via ONNX Runtime.

    Supporte le decodage greedy et beam search.

    Necessite : pip install onnxruntime numpy
    """

    def __init__(
        self,
        encoder_path: Path | str,
        decoder_path: Path | str,
        vocab_path: Path | str,
    ) -> None:
        import numpy as np
        import onnxruntime as ort

        self._np = np
        self._encoder = ort.InferenceSession(
            str(encoder_path),
            providers=["CPUExecutionProvider"],
        )
        self._decoder = ort.InferenceSession(
            str(decoder_path),
            providers=["CPUExecutionProvider"],
        )
        with open(vocab_path, encoding="utf-8") as f:
            vocab_data = json.load(f)
        self._src_char2idx: dict[str, int] = vocab_data["src_char2idx"]
        self._tgt_idx2char: dict[int, str] = {
            int(k): v for k, v in vocab_data["tgt_idx2char"].items()
        }
        self._sos_idx: int = vocab_data["sos_idx"]
        self._eos_idx: int = vocab_data["eos_idx"]
        self._max_len: int = vocab_data.get("max_len", 50)
        self._special = {"<PAD>", "<SOS>", "<EOS>", "<UNK>"}

    def _encode(self, ipa: str) -> tuple:
        """Encode une chaine IPA, retourne (enc_out, h, c)."""
        np = self._np
        ids = [self._src_char2idx.get(ch, 1) for ch in ipa]
        src = np.array([ids], dtype=np.int64)
        return self._encoder.run(None, {"src": src})

    def _tokens_to_str(self, tokens: list[int]) -> str:
        """Convertit une liste de token IDs en chaine."""
        chars: list[str] = []
        for t in tokens:
            if t == self._eos_idx:
                break
            ch = self._tgt_idx2char.get(t, "")
            if ch not in self._special:
                chars.append(ch)
        return "".join(chars)

    def predict_greedy(self, ipa: str) -> str:
        """Decodage greedy (top-1 a chaque pas)."""
        if not ipa:
            return ""
        np = self._np
        enc_out, h, c = self._encode(ipa)
        inp = np.array([self._sos_idx], dtype=np.int64)
        tokens: list[int] = []

        for _ in range(self._max_len):
            logits, h, c = self._decoder.run(
                None,
                {
                    "input_token": inp,
                    "h": h, "c": c,
                    "encoder_outputs": enc_out,
                },
            )
            idx = int(np.argmax(logits[0]))
            if idx == self._eos_idx:
                break
            tokens.append(idx)
            inp = np.array([idx], dtype=np.int64)

        return self._tokens_to_str(tokens)

    def predict_beam(
        self,
        ipa: str,
        beam_width: int = 5,
    ) -> list[tuple[str, float]]:
        """Beam search : retourne les K meilleures hypotheses avec scores normalises.

        Chaque hypothese est (mot, score) ou score est le log-prob normalise
        par la longueur, converti en probabilite normalisee sur les K candidats.
        """
        if not ipa:
            return [("", 1.0)]

        np = self._np
        enc_out, h, c = self._encode(ipa)

        # (log_prob, token_ids, h, c)
        beams: list[tuple[float, list[int], object, object]] = [
            (0.0, [], h, c),
        ]
        finished: list[tuple[float, list[int]]] = []

        for _ in range(self._max_len):
            candidates: list[tuple[float, list[int], object, object]] = []

            for log_prob, tokens, h_b, c_b in beams:
                last_token = tokens[-1] if tokens else self._sos_idx
                inp = np.array([last_token], dtype=np.int64)

                logits, h_new, c_new = self._decoder.run(
                    None,
                    {
                        "input_token": inp,
                        "h": h_b, "c": c_b,
                        "encoder_outputs": enc_out,
                    },
                )

                # Log-softmax
                raw = logits[0].astype(np.float64)
                max_val = raw.max()
                log_sum_exp = max_val + np.log(np.exp(raw - max_val).sum())
                log_probs = raw - log_sum_exp

                # Top-K tokens
                top_k_idx = np.argsort(log_probs)[-beam_width:][::-1]

                for idx in top_k_idx:
                    idx_int = int(idx)
                    new_lp = log_prob + float(log_probs[idx_int])
                    new_tokens = tokens + [idx_int]

                    if idx_int == self._eos_idx:
                        finished.append((new_lp, new_tokens))
                    else:
                        candidates.append((new_lp, new_tokens, h_new, c_new))

            # Garder les beam_width meilleurs
            candidates.sort(key=lambda x: x[0], reverse=True)
            beams = candidates[:beam_width]

            if not beams:
                break

            # Early stopping si tous les beams ont un score < meilleur fini
            if finished:
                best_finished = max(f[0] / max(len(f[1]), 1) for f in finished)
                worst_beam = beams[-1][0] / max(len(beams[-1][1]), 1)
                if worst_beam < best_finished:
                    break

        # Ajouter les beams non termines
        for lp, tok, _, _ in beams:
            finished.append((lp, tok))

        # Normaliser par longueur et convertir en chaines
        results: list[tuple[str, float]] = []
        for lp, tok in finished:
            word = self._tokens_to_str(tok)
            seq_len = max(len(word), 1)
            norm_score = lp / seq_len
            results.append((word, norm_score))

        # Trier par score normalise
        results.sort(key=lambda x: x[1], reverse=True)

        # Dedupliquer
        seen: set[str] = set()
        unique: list[tuple[str, float]] = []
        for w, s in results:
            if w not in seen:
                seen.add(w)
                unique.append((w, s))

        unique = unique[:beam_width]

        # Convertir log-probs en probabilites normalisees
        if unique:
            scores = [s for _, s in unique]
            max_s = max(scores)
            exp_scores = [math.exp(s - max_s) for s in scores]
            total = sum(exp_scores)
            unique = [(w, e / total) for (w, _), e in zip(unique, exp_scores)]

        return unique


# ══════════════════════════════════════════════════════════════════════════════
# Classe principale LecturaP2G
# ══════════════════════════════════════════════════════════════════════════════

class LecturaP2G:
    """Convertisseur Phoneme → Grapheme pour le francais (backend Seq2Seq).

    Strategies (par ordre de priorite pour predict()) :
    1. Modele Seq2Seq ONNX (si encoder/decoder/vocab fournis)
    2. Table P2G (embarquee ou externe)
    3. Regles deterministes (fallback universel)

    Args:
        encoder_path: Chemin vers l'encoder ONNX.
        decoder_path: Chemin vers le decoder ONNX.
        vocab_path: Chemin vers le fichier vocab JSON.
        table_path: Chemin vers un fichier p2g_table.json externe.
                    Si None, utilise la table embarquee.
        beam_width: Largeur du beam search par defaut.
    """

    def __init__(
        self,
        encoder_path: Path | str | None = None,
        decoder_path: Path | str | None = None,
        vocab_path: Path | str | None = None,
        table_path: Path | str | None = None,
        beam_width: int = 5,
    ) -> None:
        self._model: OnnxP2gSeq2Seq | None = None
        self._table: dict[str, str] = {}
        self._beam_width = beam_width

        # Charger le modele Seq2Seq si disponible
        if encoder_path is not None and decoder_path is not None and vocab_path is not None:
            try:
                self._model = OnnxP2gSeq2Seq(encoder_path, decoder_path, vocab_path)
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
    def has_model(self) -> bool:
        """True si un modele Seq2Seq est charge."""
        return self._model is not None

    @property
    def backend(self) -> str:
        """Nom du backend actif."""
        return "seq2seq" if self._model is not None else "table+regles"

    def predict(self, ipa: str) -> str:
        """Meilleure orthographe pour une chaine IPA.

        Utilise le modele Seq2Seq si disponible, sinon table, sinon regles.

        Args:
            ipa: Chaine IPA (ex: "bɔ̃ʒuʁ")

        Returns:
            Orthographe predite (ex: "bonjour")
        """
        if not ipa:
            return ""

        # 1. Modele Seq2Seq
        if self._model is not None:
            return self._model.predict_greedy(ipa)

        # 2. Table
        result = self._table.get(ipa)
        if result is not None:
            return result

        # 3. Regles
        return _apply_rules(ipa)

    def predict_candidates(
        self,
        ipa: str,
        k: int = 5,
    ) -> list[tuple[str, float]]:
        """Top-K orthographes avec probabilites normalisees.

        Utilise le beam search du modele Seq2Seq. Sans modele, retourne un seul
        candidat (table ou regles) avec probabilite 1.0.

        Args:
            ipa: Chaine IPA
            k: Nombre de candidats a retourner

        Returns:
            Liste de (orthographe, probabilite), triee par probabilite decroissante.
        """
        if not ipa:
            return [("", 1.0)]

        if self._model is not None:
            return self._model.predict_beam(ipa, beam_width=k)

        # Fallback sans modele
        return [(self.predict(ipa), 1.0)]

    def predict_syllable(self, ipa_syllable: str) -> str:
        """Lookup rapide pour une syllabe IPA (table ou regles, pas de modele).

        Args:
            ipa_syllable: Syllabe IPA (ex: "kɑ̃")

        Returns:
            Orthographe (ex: "quand")
        """
        result = self._table.get(ipa_syllable)
        if result is not None:
            return result
        return _apply_rules(ipa_syllable)


# ══════════════════════════════════════════════════════════════════════════════
# CLI rapide
# ══════════════════════════════════════════════════════════════════════════════

def _main() -> None:
    """Point d'entree CLI minimal."""
    import argparse
    import sys

    parser = argparse.ArgumentParser(description="Lectura P2G — Seq2Seq")
    parser.add_argument("ipa", nargs="*", help="Chaine(s) IPA a convertir")
    parser.add_argument("--encoder", default=None, help="Chemin encoder ONNX")
    parser.add_argument("--decoder", default=None, help="Chemin decoder ONNX")
    parser.add_argument("--vocab", default=None, help="Chemin vocab JSON")
    parser.add_argument("--model-dir", default=None,
                        help="Repertoire contenant encoder/decoder/vocab (auto-detection)")
    parser.add_argument("--table", default=None, help="Chemin p2g_table.json externe")
    parser.add_argument("-k", type=int, default=5, help="Nombre de candidats")
    parser.add_argument("--interactive", action="store_true", help="Mode interactif")
    args = parser.parse_args()

    enc_path = args.encoder
    dec_path = args.decoder
    voc_path = args.vocab

    # Auto-detection depuis un repertoire
    if args.model_dir and not enc_path:
        model_dir = Path(args.model_dir)
        if model_dir.is_dir():
            for f in sorted(model_dir.iterdir()):
                name = f.name.lower()
                if "encoder" in name and name.endswith("_int8.onnx"):
                    enc_path = str(f)
                elif "decoder" in name and name.endswith("_int8.onnx"):
                    dec_path = str(f)
                elif name.endswith("_vocab.json") or name == "vocab.json":
                    voc_path = str(f)
            # Fallback float32
            if not enc_path or not dec_path:
                for f in sorted(model_dir.iterdir()):
                    name = f.name.lower()
                    if not enc_path and "encoder" in name and name.endswith(".onnx"):
                        enc_path = str(f)
                    elif not dec_path and "decoder" in name and name.endswith(".onnx"):
                        dec_path = str(f)

    p2g = LecturaP2G(
        encoder_path=enc_path,
        decoder_path=dec_path,
        vocab_path=voc_path,
        table_path=args.table,
    )

    mode = "Seq2Seq" if p2g.has_model else "table+regles"
    print(f"LecturaP2G v{__version__} (backend: {mode})")

    if args.interactive:
        print("Mode interactif. Tapez 'q' pour quitter.")
        while True:
            try:
                ipa = input("\nIPA> ").strip()
            except (EOFError, KeyboardInterrupt):
                break
            if ipa in ("q", "quit", "exit"):
                break
            if not ipa:
                continue
            candidates = p2g.predict_candidates(ipa, k=args.k)
            for word, prob in candidates:
                bar = "\u2588" * int(prob * 20)
                print(f"  {word:<25s} {prob:>6.1%} {bar}")
        return

    if args.ipa:
        for ipa in args.ipa:
            candidates = p2g.predict_candidates(ipa, k=args.k)
            print(f"\n  /{ipa}/")
            for word, prob in candidates:
                bar = "\u2588" * int(prob * 20)
                print(f"    {word:<25s} {prob:>6.1%} {bar}")
    else:
        print("Usage: python lectura_p2g.py [--model-dir DIR | --encoder E --decoder D --vocab V] <ipa> ...")
        print("\nExemple (table + regles) :")
        test_words = ["\u0062\u0254\u0303\u0292u\u0281", "m\u025bz\u0254\u0303",
                      "\u0283a", "o", "p\u025b\u0283\u0153\u0281",
                      "\u0251\u0303f\u0251\u0303"]
        for ipa in test_words:
            candidates = p2g.predict_candidates(ipa, k=args.k)
            top = candidates[0] if candidates else ("?", 0.0)
            others = ", ".join(f"{w}" for w, _ in candidates[1:3])
            suffix = f" (aussi: {others})" if others else ""
            print(f"  /{ipa}/ \u2192 {top[0]}{suffix}")


if __name__ == "__main__":
    _main()
