theorem Ideal.isRadical_iff_quotient_reduced {R : Type*} [CommRing R] (I : Ideal R) :
    I.IsRadical ↔ IsReduced (R ⧸ I) := by
  /-
    R : Type u_1
    inst✝ : CommRing R
    I : Ideal R
    ⊢ Iff I.IsRadical (IsReduced (HasQuotient.Quotient R I))
  -/
  conv_lhs => rw [← @Ideal.mk_ker R _ I]
  /-
    R : Type u_1
    inst✝ : CommRing R
    I : Ideal R
    ⊢ Iff (RingHom.ker (Ideal.Quotient.mk I)).IsRadical (IsReduced (HasQuotient.Qu …
  -/
  exact RingHom.ker_isRadical_iff_reduced_of_surjective (@Ideal.Quotient.mk_surjective R _ I)
  /-
    🎉 no goals
  -/


/-- Let `P` be a property on ideals. If `P` holds for square-zero ideals, and if
  `P I → P (J ⧸ I) → P J`, then `P` holds for all nilpotent ideals. -/
theorem Ideal.IsNilpotent.induction_on (hI : IsNilpotent I)
    {P : ∀ ⦃S : Type _⦄ [CommRing S], Ideal S → Prop}
    (h₁ : ∀ ⦃S : Type _⦄ [CommRing S], ∀ I : Ideal S, I ^ 2 = ⊥ → P I)
    (h₂ : ∀ ⦃S : Type _⦄ [CommRing S], ∀ I J : Ideal S, I ≤ J → P I →
      P (J.map (Ideal.Quotient.mk I)) → P J) :
    P I := by
  /-
    S : Type u_1
    inst✝ : CommRing S
    I : Ideal S
    hI : IsNilpotent I
    P : ⦃S : Type u_1⦄ → [inst : CommRing S] → Ideal S → Prop
    h₁ : ∀ ⦃S : Type u_1⦄ [inst : CommRing S] (I : Ideal S), Eq (HPow.hPow I 2) Bo …
    h₂ : ∀ ⦃S : Type u_1⦄ [inst : CommRing S] (I J : Ideal S), LE.le I J → P I → P …
    ⊢ P I
  -/
  obtain ⟨n, hI : I ^ n = ⊥⟩ := hI
  /-
    case intro
    S : Type u_1
    inst✝ : CommRing S
    I : Ideal S
    P : ⦃S : Type u_1⦄ → [inst : CommRing S] → Ideal S → Prop
    h₁ : ∀ ⦃S : Type u_1⦄ [inst : CommRing S] (I : Ideal S), Eq (HPow.hPow I 2) Bo …
    h₂ : ∀ ⦃S : Type u_1⦄ [inst : CommRing S] (I J : Ideal S), LE.le I J → P I → P …
    n : Nat
    hI : Eq (HPow.hPow I n) Bot.bot
    ⊢ P I
  -/
  induction' n using Nat.strong_induction_on with n H generalizing S
  /-
    case intro.h
    P : ⦃S : Type u_1⦄ → [inst : CommRing S] → Ideal S → Prop
    h₁ : ∀ ⦃S : Type u_1⦄ [inst : CommRing S] (I : Ideal S), Eq (HPow.hPow I 2) Bo …
    h₂ : ∀ ⦃S : Type u_1⦄ [inst : CommRing S] (I J : Ideal S), LE.le I J → P I → P …
    n : Nat
    H : ∀ (m : Nat), LT.lt m n → ∀ {S : Type u_1} [inst : CommRing S] (I : Ideal S …
    S : Type u_1
    inst✝ : CommRing S
    I : Ideal S
    hI : Eq (HPow.hPow I n) Bot.bot
    ⊢ P I
  -/
  by_cases hI' : I = ⊥
    /-
      case pos
      P : ⦃S : Type u_1⦄ → [inst : CommRing S] → Ideal S → Prop
      h₁ : ∀ ⦃S : Type u_1⦄ [inst : CommRing S] (I : Ideal S), Eq (HPow.hPow I 2) Bo …
      h₂ : ∀ ⦃S : Type u_1⦄ [inst : CommRing S] (I J : Ideal S), LE.le I J → P I → P …
      n : Nat
      H : ∀ (m : Nat), LT.lt m n → ∀ {S : Type u_1} [inst : CommRing S] (I : Ideal S …
      S : Type u_1
      inst✝ : CommRing S
      I : Ideal S
      hI : Eq (HPow.hPow I n) Bot.bot
      hI' : Eq I Bot.bot
      ⊢ P I
    -/
  · subst hI'
    /-
      case pos
      P : ⦃S : Type u_1⦄ → [inst : CommRing S] → Ideal S → Prop
      h₁ : ∀ ⦃S : Type u_1⦄ [inst : CommRing S] (I : Ideal S), Eq (HPow.hPow I 2) Bo …
      h₂ : ∀ ⦃S : Type u_1⦄ [inst : CommRing S] (I J : Ideal S), LE.le I J → P I → P …
      n : Nat
      H : ∀ (m : Nat), LT.lt m n → ∀ {S : Type u_1} [inst : CommRing S] (I : Ideal S …
      S : Type u_1
      inst✝ : CommRing S
      hI : Eq (HPow.hPow Bot.bot n) Bot.bot
      ⊢ P Bot.bot
    -/
    apply h₁
    /-
      case pos.a
      P : ⦃S : Type u_1⦄ → [inst : CommRing S] → Ideal S → Prop
      h₁ : ∀ ⦃S : Type u_1⦄ [inst : CommRing S] (I : Ideal S), Eq (HPow.hPow I 2) Bo …
      h₂ : ∀ ⦃S : Type u_1⦄ [inst : CommRing S] (I J : Ideal S), LE.le I J → P I → P …
      n : Nat
      H : ∀ (m : Nat), LT.lt m n → ∀ {S : Type u_1} [inst : CommRing S] (I : Ideal S …
      S : Type u_1
      inst✝ : CommRing S
      hI : Eq (HPow.hPow Bot.bot n) Bot.bot
      ⊢ Eq (HPow.hPow Bot.bot 2) Bot.bot
    -/
    rw [← Ideal.zero_eq_bot, zero_pow two_ne_zero]
    /-
      🎉 no goals
    -/
  /-
    case neg
    P : ⦃S : Type u_1⦄ → [inst : CommRing S] → Ideal S → Prop
    h₁ : ∀ ⦃S : Type u_1⦄ [inst : CommRing S] (I : Ideal S), Eq (HPow.hPow I 2) Bo …
    h₂ : ∀ ⦃S : Type u_1⦄ [inst : CommRing S] (I J : Ideal S), LE.le I J → P I → P …
    n : Nat
    H : ∀ (m : Nat), LT.lt m n → ∀ {S : Type u_1} [inst : CommRing S] (I : Ideal S …
    S : Type u_1
    inst✝ : CommRing S
    I : Ideal S
    hI : Eq (HPow.hPow I n) Bot.bot
    hI' : Not (Eq I Bot.bot)
    ⊢ P I
  -/
  cases' n with n
    /-
      case neg.zero
      P : ⦃S : Type u_1⦄ → [inst : CommRing S] → Ideal S → Prop
      h₁ : ∀ ⦃S : Type u_1⦄ [inst : CommRing S] (I : Ideal S), Eq (HPow.hPow I 2) Bo …
      h₂ : ∀ ⦃S : Type u_1⦄ [inst : CommRing S] (I J : Ideal S), LE.le I J → P I → P …
      S : Type u_1
      inst✝ : CommRing S
      I : Ideal S
      hI' : Not (Eq I Bot.bot)
      H : ∀ (m : Nat), LT.lt m 0 → ∀ {S : Type u_1} [inst : CommRing S] (I : Ideal S …
      hI : Eq (HPow.hPow I 0) Bot.bot
      ⊢ P I
    -/
  · rw [pow_zero, Ideal.one_eq_top] at hI
    /-
      case neg.zero
      P : ⦃S : Type u_1⦄ → [inst : CommRing S] → Ideal S → Prop
      h₁ : ∀ ⦃S : Type u_1⦄ [inst : CommRing S] (I : Ideal S), Eq (HPow.hPow I 2) Bo …
      h₂ : ∀ ⦃S : Type u_1⦄ [inst : CommRing S] (I J : Ideal S), LE.le I J → P I → P …
      S : Type u_1
      inst✝ : CommRing S
      I : Ideal S
      hI' : Not (Eq I Bot.bot)
      H : ∀ (m : Nat), LT.lt m 0 → ∀ {S : Type u_1} [inst : CommRing S] (I : Ideal S …
      hI : Eq Top.top Bot.bot
      ⊢ P I
    -/
    haveI := subsingleton_of_bot_eq_top hI.symm
    /-
      case neg.zero
      P : ⦃S : Type u_1⦄ → [inst : CommRing S] → Ideal S → Prop
      h₁ : ∀ ⦃S : Type u_1⦄ [inst : CommRing S] (I : Ideal S), Eq (HPow.hPow I 2) Bo …
      h₂ : ∀ ⦃S : Type u_1⦄ [inst : CommRing S] (I J : Ideal S), LE.le I J → P I → P …
      S : Type u_1
      inst✝ : CommRing S
      I : Ideal S
      hI' : Not (Eq I Bot.bot)
      H : ∀ (m : Nat), LT.lt m 0 → ∀ {S : Type u_1} [inst : CommRing S] (I : Ideal S …
      hI : Eq Top.top Bot.bot
      this : Subsingleton (Ideal S)
      ⊢ P I
    -/
    exact (hI' (Subsingleton.elim _ _)).elim
    /-
      🎉 no goals
    -/
  /-
    case neg.succ
    P : ⦃S : Type u_1⦄ → [inst : CommRing S] → Ideal S → Prop
    h₁ : ∀ ⦃S : Type u_1⦄ [inst : CommRing S] (I : Ideal S), Eq (HPow.hPow I 2) Bo …
    h₂ : ∀ ⦃S : Type u_1⦄ [inst : CommRing S] (I J : Ideal S), LE.le I J → P I → P …
    S : Type u_1
    inst✝ : CommRing S
    I : Ideal S
    hI' : Not (Eq I Bot.bot)
    n : Nat
    H : ∀ (m : Nat), LT.lt m (HAdd.hAdd n 1) → ∀ {S : Type u_1} [inst : CommRing S …
    hI : Eq (HPow.hPow I (HAdd.hAdd n 1)) Bot.bot
    ⊢ P I
  -/
  cases' n with n
    /-
      case neg.succ.zero
      P : ⦃S : Type u_1⦄ → [inst : CommRing S] → Ideal S → Prop
      h₁ : ∀ ⦃S : Type u_1⦄ [inst : CommRing S] (I : Ideal S), Eq (HPow.hPow I 2) Bo …
      h₂ : ∀ ⦃S : Type u_1⦄ [inst : CommRing S] (I J : Ideal S), LE.le I J → P I → P …
      S : Type u_1
      inst✝ : CommRing S
      I : Ideal S
      hI' : Not (Eq I Bot.bot)
      H : ∀ (m : Nat), LT.lt m (HAdd.hAdd 0 1) → ∀ {S : Type u_1} [inst : CommRing S …
      hI : Eq (HPow.hPow I (HAdd.hAdd 0 1)) Bot.bot
      ⊢ P I
    -/
  · rw [pow_one] at hI
    /-
      case neg.succ.zero
      P : ⦃S : Type u_1⦄ → [inst : CommRing S] → Ideal S → Prop
      h₁ : ∀ ⦃S : Type u_1⦄ [inst : CommRing S] (I : Ideal S), Eq (HPow.hPow I 2) Bo …
      h₂ : ∀ ⦃S : Type u_1⦄ [inst : CommRing S] (I J : Ideal S), LE.le I J → P I → P …
      S : Type u_1
      inst✝ : CommRing S
      I : Ideal S
      hI' : Not (Eq I Bot.bot)
      H : ∀ (m : Nat), LT.lt m (HAdd.hAdd 0 1) → ∀ {S : Type u_1} [inst : CommRing S …
      hI : Eq I Bot.bot
      ⊢ P I
    -/
    exact (hI' hI).elim
    /-
      🎉 no goals
    -/
  /-
    case neg.succ.succ
    P : ⦃S : Type u_1⦄ → [inst : CommRing S] → Ideal S → Prop
    h₁ : ∀ ⦃S : Type u_1⦄ [inst : CommRing S] (I : Ideal S), Eq (HPow.hPow I 2) Bo …
    h₂ : ∀ ⦃S : Type u_1⦄ [inst : CommRing S] (I J : Ideal S), LE.le I J → P I → P …
    S : Type u_1
    inst✝ : CommRing S
    I : Ideal S
    hI' : Not (Eq I Bot.bot)
    n : Nat
    H : ∀ (m : Nat), LT.lt m (HAdd.hAdd (HAdd.hAdd n 1) 1) → ∀ {S : Type u_1} [ins …
    hI : Eq (HPow.hPow I (HAdd.hAdd (HAdd.hAdd n 1) 1)) Bot.bot
    ⊢ P I
  -/
  apply h₂ (I ^ 2) _ (Ideal.pow_le_self two_ne_zero)
    /-
      case neg.succ.succ.a
      P : ⦃S : Type u_1⦄ → [inst : CommRing S] → Ideal S → Prop
      h₁ : ∀ ⦃S : Type u_1⦄ [inst : CommRing S] (I : Ideal S), Eq (HPow.hPow I 2) Bo …
      h₂ : ∀ ⦃S : Type u_1⦄ [inst : CommRing S] (I J : Ideal S), LE.le I J → P I → P …
      S : Type u_1
      inst✝ : CommRing S
      I : Ideal S
      hI' : Not (Eq I Bot.bot)
      n : Nat
      H : ∀ (m : Nat), LT.lt m (HAdd.hAdd (HAdd.hAdd n 1) 1) → ∀ {S : Type u_1} [ins …
      hI : Eq (HPow.hPow I (HAdd.hAdd (HAdd.hAdd n 1) 1)) Bot.bot
      ⊢ P (HPow.hPow I 2)
    -/
  · apply H n.succ _ (I ^ 2)
      /-
        case neg.succ.succ.a
        P : ⦃S : Type u_1⦄ → [inst : CommRing S] → Ideal S → Prop
        h₁ : ∀ ⦃S : Type u_1⦄ [inst : CommRing S] (I : Ideal S), Eq (HPow.hPow I 2) Bo …
        h₂ : ∀ ⦃S : Type u_1⦄ [inst : CommRing S] (I J : Ideal S), LE.le I J → P I → P …
        S : Type u_1
        inst✝ : CommRing S
        I : Ideal S
        hI' : Not (Eq I Bot.bot)
        n : Nat
        H : ∀ (m : Nat), LT.lt m (HAdd.hAdd (HAdd.hAdd n 1) 1) → ∀ {S : Type u_1} [ins …
        hI : Eq (HPow.hPow I (HAdd.hAdd (HAdd.hAdd n 1) 1)) Bot.bot
        ⊢ Eq (HPow.hPow (HPow.hPow I 2) n.succ) Bot.bot
      -/
    · rw [← pow_mul, eq_bot_iff, ← hI, Nat.succ_eq_add_one]
      /-
        case neg.succ.succ.a
        P : ⦃S : Type u_1⦄ → [inst : CommRing S] → Ideal S → Prop
        h₁ : ∀ ⦃S : Type u_1⦄ [inst : CommRing S] (I : Ideal S), Eq (HPow.hPow I 2) Bo …
        h₂ : ∀ ⦃S : Type u_1⦄ [inst : CommRing S] (I J : Ideal S), LE.le I J → P I → P …
        S : Type u_1
        inst✝ : CommRing S
        I : Ideal S
        hI' : Not (Eq I Bot.bot)
        n : Nat
        H : ∀ (m : Nat), LT.lt m (HAdd.hAdd (HAdd.hAdd n 1) 1) → ∀ {S : Type u_1} [ins …
        hI : Eq (HPow.hPow I (HAdd.hAdd (HAdd.hAdd n 1) 1)) Bot.bot
        ⊢ LE.le (HPow.hPow I (HMul.hMul 2 (HAdd.hAdd n 1))) (HPow.hPow I (HAdd.hAdd (H …
      -/
      apply Ideal.pow_le_pow_right (by omega)
      /-
        🎉 no goals
      -/
      /-
        P : ⦃S : Type u_1⦄ → [inst : CommRing S] → Ideal S → Prop
        h₁ : ∀ ⦃S : Type u_1⦄ [inst : CommRing S] (I : Ideal S), Eq (HPow.hPow I 2) Bo …
        h₂ : ∀ ⦃S : Type u_1⦄ [inst : CommRing S] (I J : Ideal S), LE.le I J → P I → P …
        S : Type u_1
        inst✝ : CommRing S
        I : Ideal S
        hI' : Not (Eq I Bot.bot)
        n : Nat
        H : ∀ (m : Nat), LT.lt m (HAdd.hAdd (HAdd.hAdd n 1) 1) → ∀ {S : Type u_1} [ins …
        hI : Eq (HPow.hPow I (HAdd.hAdd (HAdd.hAdd n 1) 1)) Bot.bot
        ⊢ LT.lt n.succ (HAdd.hAdd (HAdd.hAdd n 1) 1)
      -/
    · exact n.succ.lt_succ_self
      /-
        🎉 no goals
      -/
    /-
      case neg.succ.succ.a
      P : ⦃S : Type u_1⦄ → [inst : CommRing S] → Ideal S → Prop
      h₁ : ∀ ⦃S : Type u_1⦄ [inst : CommRing S] (I : Ideal S), Eq (HPow.hPow I 2) Bo …
      h₂ : ∀ ⦃S : Type u_1⦄ [inst : CommRing S] (I J : Ideal S), LE.le I J → P I → P …
      S : Type u_1
      inst✝ : CommRing S
      I : Ideal S
      hI' : Not (Eq I Bot.bot)
      n : Nat
      H : ∀ (m : Nat), LT.lt m (HAdd.hAdd (HAdd.hAdd n 1) 1) → ∀ {S : Type u_1} [ins …
      hI : Eq (HPow.hPow I (HAdd.hAdd (HAdd.hAdd n 1) 1)) Bot.bot
      ⊢ P (Ideal.map (Ideal.Quotient.mk (HPow.hPow I 2)) I)
    -/
  · apply h₁
    /-
      case neg.succ.succ.a.a
      P : ⦃S : Type u_1⦄ → [inst : CommRing S] → Ideal S → Prop
      h₁ : ∀ ⦃S : Type u_1⦄ [inst : CommRing S] (I : Ideal S), Eq (HPow.hPow I 2) Bo …
      h₂ : ∀ ⦃S : Type u_1⦄ [inst : CommRing S] (I J : Ideal S), LE.le I J → P I → P …
      S : Type u_1
      inst✝ : CommRing S
      I : Ideal S
      hI' : Not (Eq I Bot.bot)
      n : Nat
      H : ∀ (m : Nat), LT.lt m (HAdd.hAdd (HAdd.hAdd n 1) 1) → ∀ {S : Type u_1} [ins …
      hI : Eq (HPow.hPow I (HAdd.hAdd (HAdd.hAdd n 1) 1)) Bot.bot
      ⊢ Eq (HPow.hPow (Ideal.map (Ideal.Quotient.mk (HPow.hPow I 2)) I) 2) Bot.bot
    -/
    rw [← Ideal.map_pow, Ideal.map_quotient_self]
    /-
      🎉 no goals
    -/


theorem IsNilpotent.isUnit_quotient_mk_iff {R : Type*} [CommRing R] {I : Ideal R}
    (hI : IsNilpotent I) {x : R} : IsUnit (Ideal.Quotient.mk I x) ↔ IsUnit x := by
  /-
    R : Type u_2
    inst✝ : CommRing R
    I : Ideal R
    hI : IsNilpotent I
    x : R
    ⊢ Iff (IsUnit ((Ideal.Quotient.mk I) x)) (IsUnit x)
  -/
  refine ⟨?_, fun h => h.map <| Ideal.Quotient.mk I⟩
  /-
    R : Type u_2
    inst✝ : CommRing R
    I : Ideal R
    hI : IsNilpotent I
    x : R
    ⊢ IsUnit ((Ideal.Quotient.mk I) x) → IsUnit x
  -/
  revert x
  /-
    R : Type u_2
    inst✝ : CommRing R
    I : Ideal R
    hI : IsNilpotent I
    ⊢ ∀ {x : R}, IsUnit ((Ideal.Quotient.mk I) x) → IsUnit x
  -/
  apply Ideal.IsNilpotent.induction_on (S := R) I hI <;> clear hI I
  /-
    case h₁
    R : Type u_2
    inst✝ : CommRing R
    ⊢ ∀ ⦃S : Type u_2⦄ [inst : CommRing S] (I : Ideal S), Eq (HPow.hPow I 2) Bot.b …
  -/
  swap
    /-
      case h₂
      R : Type u_2
      inst✝ : CommRing R
      ⊢ ∀ ⦃S : Type u_2⦄ [inst : CommRing S] (I J : Ideal S), LE.le I J → (∀ {x : S} …
    -/
  · introv e h₁ h₂ h₃
    /-
      case h₂
      R : Type u_2
      inst✝¹ : CommRing R
      S : Type u_2
      inst✝ : CommRing S
      I J : Ideal S
      e : LE.le I J
      h₁ : ∀ {x : S}, IsUnit ((Ideal.Quotient.mk I) x) → IsUnit x
      h₂ : ∀ {x : HasQuotient.Quotient S I}, IsUnit ((Ideal.Quotient.mk (Ideal.map ( …
      x : S
      h₃ : IsUnit ((Ideal.Quotient.mk J) x)
      ⊢ IsUnit x
    -/
    apply h₁
    /-
      case h₂.a
      R : Type u_2
      inst✝¹ : CommRing R
      S : Type u_2
      inst✝ : CommRing S
      I J : Ideal S
      e : LE.le I J
      h₁ : ∀ {x : S}, IsUnit ((Ideal.Quotient.mk I) x) → IsUnit x
      h₂ : ∀ {x : HasQuotient.Quotient S I}, IsUnit ((Ideal.Quotient.mk (Ideal.map ( …
      x : S
      h₃ : IsUnit ((Ideal.Quotient.mk J) x)
      ⊢ IsUnit ((Ideal.Quotient.mk I) x)
    -/
    apply h₂
    exact
      h₃.map
        ((DoubleQuot.quotQuotEquivQuotSup I J).trans
              (Ideal.quotEquivOfEq (sup_eq_right.mpr e))).symm.toRingHom
    /-
      case h₁
      R : Type u_2
      inst✝ : CommRing R
      ⊢ ∀ ⦃S : Type u_2⦄ [inst : CommRing S] (I : Ideal S), Eq (HPow.hPow I 2) Bot.b …
    -/
  · introv e H
    /-
      case h₁
      R : Type u_2
      inst✝¹ : CommRing R
      S : Type u_2
      inst✝ : CommRing S
      I : Ideal S
      e : Eq (HPow.hPow I 2) Bot.bot
      x : S
      H : IsUnit ((Ideal.Quotient.mk I) x)
      ⊢ IsUnit x
    -/
    obtain ⟨y, hy⟩ := Ideal.Quotient.mk_surjective (↑H.unit⁻¹ : S ⧸ I)
    have : Ideal.Quotient.mk I (x * y) = Ideal.Quotient.mk I 1 := by
      rw [map_one, _root_.map_mul, hy, IsUnit.mul_val_inv]
    /-
      case h₁.intro
      R : Type u_2
      inst✝¹ : CommRing R
      S : Type u_2
      inst✝ : CommRing S
      I : Ideal S
      e : Eq (HPow.hPow I 2) Bot.bot
      x : S
      H : IsUnit ((Ideal.Quotient.mk I) x)
      y : S
      hy : Eq ((Ideal.Quotient.mk I) y) ↑(Inv.inv H.unit)
      this : Eq ((Ideal.Quotient.mk I) (HMul.hMul x y)) ((Ideal.Quotient.mk I) 1)
      ⊢ IsUnit x
    -/
    rw [Ideal.Quotient.eq] at this
    have : (x * y - 1) ^ 2 = 0 := by
      rw [← Ideal.mem_bot, ← e]
      exact Ideal.pow_mem_pow this _
    have : x * (y * (2 - x * y)) = 1 := by
      rw [eq_comm, ← sub_eq_zero, ← this]
      ring
    /-
      case h₁.intro
      R : Type u_2
      inst✝¹ : CommRing R
      S : Type u_2
      inst✝ : CommRing S
      I : Ideal S
      e : Eq (HPow.hPow I 2) Bot.bot
      x : S
      H : IsUnit ((Ideal.Quotient.mk I) x)
      y : S
      hy : Eq ((Ideal.Quotient.mk I) y) ↑(Inv.inv H.unit)
      this✝¹ : Membership.mem I (HSub.hSub (HMul.hMul x y) 1)
      this✝ : Eq (HPow.hPow (HSub.hSub (HMul.hMul x y) 1) 2) 0
      this : Eq (HMul.hMul x (HMul.hMul y (HSub.hSub 2 (HMul.hMul x y)))) 1
      ⊢ IsUnit x
    -/
    exact isUnit_of_mul_eq_one _ _ this
    /-
      🎉 no goals
    -/

