/-- The cardinality of `(M ⧸ S)`, if `(M ⧸ S)` is finite, and `0` otherwise.
This is used to define the absolute ideal norm `Ideal.absNorm`.
-/
noncomputable def cardQuot (S : Submodule R M) : ℕ :=
  AddSubgroup.index S.toAddSubgroup


theorem cardQuot_apply (S : Submodule R M) : cardQuot S = Nat.card (M ⧸ S) := by
  /-
    R : Type u_1
    M : Type u_2
    inst✝² : Ring R
    inst✝¹ : AddCommGroup M
    inst✝ : Module R M
    S : Submodule R M
    ⊢ Eq S.cardQuot (Nat.card (HasQuotient.Quotient M S))
  -/
  rfl
  /-
    🎉 no goals
  -/


@[simp]
theorem cardQuot_bot [Infinite M] : cardQuot (⊥ : Submodule R M) = 0 :=
  AddSubgroup.index_bot.trans Nat.card_eq_zero_of_infinite


@[simp]
theorem cardQuot_top : cardQuot (⊤ : Submodule R M) = 1 :=
  AddSubgroup.index_top


@[simp]
theorem cardQuot_eq_one_iff {P : Submodule R M} : cardQuot P = 1 ↔ P = ⊤ :=
                                     /-
                                       R : Type u_1
                                       M : Type u_2
                                       inst✝² : Ring R
                                       inst✝¹ : AddCommGroup M
                                       inst✝ : Module R M
                                       P : Submodule R M
                                       ⊢ Iff (Eq P.toAddSubgroup Top.top) (Eq P Top.top)
                                     -/
  AddSubgroup.index_eq_one.trans (by simp [SetLike.ext_iff])
                                     /-
                                       🎉 no goals
                                     -/


/-- Multiplicity of the ideal norm, for coprime ideals.
This is essentially just a repackaging of the Chinese Remainder Theorem.
-/
theorem cardQuot_mul_of_coprime
    {I J : Ideal S} (coprime : IsCoprime I J) : cardQuot (I * J) = cardQuot I * cardQuot J := by
  rw [cardQuot_apply, cardQuot_apply, cardQuot_apply,
    Nat.card_congr (Ideal.quotientMulEquivQuotientProd I J coprime).toEquiv,
    Nat.card_prod]


/-- If the `d` from `Ideal.exists_mul_add_mem_pow_succ` is unique, up to `P`,
then so are the `c`s, up to `P ^ (i + 1)`.
Inspired by [Neukirch], proposition 6.1 -/
theorem Ideal.mul_add_mem_pow_succ_inj (P : Ideal S) {i : ℕ} (a d d' e e' : S) (a_mem : a ∈ P ^ i)
    (e_mem : e ∈ P ^ (i + 1)) (e'_mem : e' ∈ P ^ (i + 1)) (h : d - d' ∈ P) :
    a * d + e - (a * d' + e') ∈ P ^ (i + 1) := by
  have : a * d - a * d' ∈ P ^ (i + 1) := by
    simp only [← mul_sub]
    exact Ideal.mul_mem_mul a_mem h
  /-
    S : Type u_1
    inst✝ : CommRing S
    P : Ideal S
    i : Nat
    a d d' e e' : S
    a_mem : Membership.mem (HPow.hPow P i) a
    e_mem : Membership.mem (HPow.hPow P (HAdd.hAdd i 1)) e
    e'_mem : Membership.mem (HPow.hPow P (HAdd.hAdd i 1)) e'
    h : Membership.mem P (HSub.hSub d d')
    this : Membership.mem (HPow.hPow P (HAdd.hAdd i 1)) (HSub.hSub (HMul.hMul a d) …
    ⊢ Membership.mem (HPow.hPow P (HAdd.hAdd i 1)) (HSub.hSub (HAdd.hAdd (HMul.hMu …
  -/
  convert Ideal.add_mem _ this (Ideal.sub_mem _ e_mem e'_mem) using 1
  /-
    case h.e'_5
    S : Type u_1
    inst✝ : CommRing S
    P : Ideal S
    i : Nat
    a d d' e e' : S
    a_mem : Membership.mem (HPow.hPow P i) a
    e_mem : Membership.mem (HPow.hPow P (HAdd.hAdd i 1)) e
    e'_mem : Membership.mem (HPow.hPow P (HAdd.hAdd i 1)) e'
    h : Membership.mem P (HSub.hSub d d')
    this : Membership.mem (HPow.hPow P (HAdd.hAdd i 1)) (HSub.hSub (HMul.hMul a d) …
    ⊢ Eq (HSub.hSub (HAdd.hAdd (HMul.hMul a d) e) (HAdd.hAdd (HMul.hMul a d') e')) …
  -/
  ring
  /-
    🎉 no goals
  -/


/-- If `a ∈ P^i \ P^(i+1)` and `c ∈ P^i`, then `a * d + e = c` for `e ∈ P^(i+1)`.
`Ideal.mul_add_mem_pow_succ_unique` shows the choice of `d` is unique, up to `P`.
Inspired by [Neukirch], proposition 6.1 -/
theorem Ideal.exists_mul_add_mem_pow_succ [IsDedekindDomain S] (hP : P ≠ ⊥)
    {i : ℕ} (a c : S) (a_mem : a ∈ P ^ i)
    (a_not_mem : a ∉ P ^ (i + 1)) (c_mem : c ∈ P ^ i) :
    ∃ d : S, ∃ e ∈ P ^ (i + 1), a * d + e = c := by
  suffices eq_b : P ^ i = Ideal.span {a} ⊔ P ^ (i + 1) by
    rw [eq_b] at c_mem
    simp only [mul_comm a]
    exact Ideal.mem_span_singleton_sup.mp c_mem
  refine (Ideal.eq_prime_pow_of_succ_lt_of_le hP (lt_of_le_of_ne le_sup_right ?_)
    (sup_le (Ideal.span_le.mpr (Set.singleton_subset_iff.mpr a_mem))
      (Ideal.pow_succ_lt_pow hP i).le)).symm
  /-
    S : Type u_1
    inst✝¹ : CommRing S
    P : Ideal S
    P_prime : P.IsPrime
    inst✝ : IsDedekindDomain S
    hP : Ne P Bot.bot
    i : Nat
    a c : S
    a_mem : Membership.mem (HPow.hPow P i) a
    a_not_mem : Not (Membership.mem (HPow.hPow P (HAdd.hAdd i 1)) a)
    c_mem : Membership.mem (HPow.hPow P i) c
    ⊢ Ne (HPow.hPow P (HAdd.hAdd i 1)) (Max.max (Ideal.span (Singleton.singleton a …
  -/
  contrapose! a_not_mem with this
  /-
    S : Type u_1
    inst✝¹ : CommRing S
    P : Ideal S
    P_prime : P.IsPrime
    inst✝ : IsDedekindDomain S
    hP : Ne P Bot.bot
    i : Nat
    a c : S
    a_mem : Membership.mem (HPow.hPow P i) a
    c_mem : Membership.mem (HPow.hPow P i) c
    this : Eq (HPow.hPow P (HAdd.hAdd i 1)) (Max.max (Ideal.span (Singleton.single …
    ⊢ Membership.mem (HPow.hPow P (HAdd.hAdd i 1)) a
  -/
  rw [this]
  /-
    S : Type u_1
    inst✝¹ : CommRing S
    P : Ideal S
    P_prime : P.IsPrime
    inst✝ : IsDedekindDomain S
    hP : Ne P Bot.bot
    i : Nat
    a c : S
    a_mem : Membership.mem (HPow.hPow P i) a
    c_mem : Membership.mem (HPow.hPow P i) c
    this : Eq (HPow.hPow P (HAdd.hAdd i 1)) (Max.max (Ideal.span (Singleton.single …
    ⊢ Membership.mem (Max.max (Ideal.span (Singleton.singleton a)) (HPow.hPow P (H …
  -/
  exact mem_sup.mpr ⟨a, mem_span_singleton_self a, 0, by simp, by simp⟩
  /-
    🎉 no goals
  -/


theorem Ideal.mem_prime_of_mul_mem_pow [IsDedekindDomain S] {P : Ideal S} [P_prime : P.IsPrime]
    (hP : P ≠ ⊥) {i : ℕ} {a b : S} (a_not_mem : a ∉ P ^ (i + 1)) (ab_mem : a * b ∈ P ^ (i + 1)) :
    b ∈ P := by
  simp only [← Ideal.span_singleton_le_iff_mem, ← Ideal.dvd_iff_le, pow_succ, ←
    Ideal.span_singleton_mul_span_singleton] at a_not_mem ab_mem ⊢
  /-
    S : Type u_1
    inst✝¹ : CommRing S
    inst✝ : IsDedekindDomain S
    P : Ideal S
    P_prime : P.IsPrime
    hP : Ne P Bot.bot
    i : Nat
    a b : S
    a_not_mem : Not (Dvd.dvd (HMul.hMul (HPow.hPow P i) P) (Ideal.span (Singleton. …
    ab_mem : Dvd.dvd (HMul.hMul (HPow.hPow P i) P) (HMul.hMul (Ideal.span (Singlet …
    ⊢ Dvd.dvd P (Ideal.span (Singleton.singleton b))
  -/
  exact (prime_pow_succ_dvd_mul (Ideal.prime_of_isPrime hP P_prime) ab_mem).resolve_left a_not_mem
  /-
    🎉 no goals
  -/


/-- The choice of `d` in `Ideal.exists_mul_add_mem_pow_succ` is unique, up to `P`.
Inspired by [Neukirch], proposition 6.1 -/
theorem Ideal.mul_add_mem_pow_succ_unique [IsDedekindDomain S] (hP : P ≠ ⊥)
    {i : ℕ} (a d d' e e' : S)
    (a_not_mem : a ∉ P ^ (i + 1)) (e_mem : e ∈ P ^ (i + 1)) (e'_mem : e' ∈ P ^ (i + 1))
    (h : a * d + e - (a * d' + e') ∈ P ^ (i + 1)) : d - d' ∈ P := by
  have h' : a * (d - d') ∈ P ^ (i + 1) := by
    convert Ideal.add_mem _ h (Ideal.sub_mem _ e'_mem e_mem) using 1
    ring
  /-
    S : Type u_1
    inst✝¹ : CommRing S
    P : Ideal S
    P_prime : P.IsPrime
    inst✝ : IsDedekindDomain S
    hP : Ne P Bot.bot
    i : Nat
    a d d' e e' : S
    a_not_mem : Not (Membership.mem (HPow.hPow P (HAdd.hAdd i 1)) a)
    e_mem : Membership.mem (HPow.hPow P (HAdd.hAdd i 1)) e
    e'_mem : Membership.mem (HPow.hPow P (HAdd.hAdd i 1)) e'
    h : Membership.mem (HPow.hPow P (HAdd.hAdd i 1)) (HSub.hSub (HAdd.hAdd (HMul.h …
    h' : Membership.mem (HPow.hPow P (HAdd.hAdd i 1)) (HMul.hMul a (HSub.hSub d d'))
    ⊢ Membership.mem P (HSub.hSub d d')
  -/
  exact Ideal.mem_prime_of_mul_mem_pow hP a_not_mem h'
  /-
    🎉 no goals
  -/


/-- Multiplicity of the ideal norm, for powers of prime ideals. -/
theorem cardQuot_pow_of_prime [IsDedekindDomain S] (hP : P ≠ ⊥) {i : ℕ} :
    cardQuot (P ^ i) = cardQuot P ^ i := by
  /-
    S : Type u_1
    inst✝¹ : CommRing S
    P : Ideal S
    P_prime : P.IsPrime
    inst✝ : IsDedekindDomain S
    hP : Ne P Bot.bot
    i : Nat
    ⊢ Eq (Submodule.cardQuot (HPow.hPow P i)) (HPow.hPow (Submodule.cardQuot P) i)
  -/
  induction' i with i ih
    /-
      case zero
      S : Type u_1
      inst✝¹ : CommRing S
      P : Ideal S
      P_prime : P.IsPrime
      inst✝ : IsDedekindDomain S
      hP : Ne P Bot.bot
      ⊢ Eq (Submodule.cardQuot (HPow.hPow P 0)) (HPow.hPow (Submodule.cardQuot P) 0)
    -/
  · simp
    /-
      🎉 no goals
    -/
  /-
    case succ
    S : Type u_1
    inst✝¹ : CommRing S
    P : Ideal S
    P_prime : P.IsPrime
    inst✝ : IsDedekindDomain S
    hP : Ne P Bot.bot
    i : Nat
    ih : Eq (Submodule.cardQuot (HPow.hPow P i)) (HPow.hPow (Submodule.cardQuot P) …
    ⊢ Eq (Submodule.cardQuot (HPow.hPow P (HAdd.hAdd i 1))) (HPow.hPow (Submodule. …
  -/
  have : P ^ (i + 1) < P ^ i := Ideal.pow_succ_lt_pow hP i
  suffices hquot : map (P ^ i.succ).mkQ (P ^ i) ≃ S ⧸ P by
    rw [pow_succ' (cardQuot P), ← ih, cardQuot_apply (P ^ i.succ), ←
      card_quotient_mul_card_quotient (P ^ i) (P ^ i.succ) this.le, cardQuot_apply (P ^ i),
      cardQuot_apply P, Nat.card_congr hquot]
  /-
    case succ
    S : Type u_1
    inst✝¹ : CommRing S
    P : Ideal S
    P_prime : P.IsPrime
    inst✝ : IsDedekindDomain S
    hP : Ne P Bot.bot
    i : Nat
    ih : Eq (Submodule.cardQuot (HPow.hPow P i)) (HPow.hPow (Submodule.cardQuot P) …
    this : LT.lt (HPow.hPow P (HAdd.hAdd i 1)) (HPow.hPow P i)
    ⊢ Equiv (Subtype fun x => Membership.mem (Submodule.map (Submodule.mkQ (HPow.h …
  -/
  choose a a_mem a_not_mem using SetLike.exists_of_lt this
  choose f g hg hf using fun c (hc : c ∈ P ^ i) =>
    Ideal.exists_mul_add_mem_pow_succ hP a c a_mem a_not_mem hc
  choose k hk_mem hk_eq using fun c' (hc' : c' ∈ map (mkQ (P ^ i.succ)) (P ^ i)) =>
    Submodule.mem_map.mp hc'
  /-
    case succ
    S : Type u_1
    inst✝¹ : CommRing S
    P : Ideal S
    P_prime : P.IsPrime
    inst✝ : IsDedekindDomain S
    hP : Ne P Bot.bot
    i : Nat
    ih : Eq (Submodule.cardQuot (HPow.hPow P i)) (HPow.hPow (Submodule.cardQuot P) …
    this : LT.lt (HPow.hPow P (HAdd.hAdd i 1)) (HPow.hPow P i)
    a : S
    a_mem : Membership.mem (HPow.hPow P i) a
    a_not_mem : Not (Membership.mem (HPow.hPow P (HAdd.hAdd i 1)) a)
    f g : (c : S) → Membership.mem (HPow.hPow P i) c → S
    hg : ∀ (c : S) (hc : Membership.mem (HPow.hPow P i) c), Membership.mem (HPow.h …
    hf : ∀ (c : S) (hc : Membership.mem (HPow.hPow P i) c), Eq (HAdd.hAdd (HMul.hM …
    k : (c' : HasQuotient.Quotient S (HPow.hPow P i.succ)) → Membership.mem (Submo …
    hk_mem : ∀ (c' : HasQuotient.Quotient S (HPow.hPow P i.succ)) (hc' : Membershi …
    hk_eq : ∀ (c' : HasQuotient.Quotient S (HPow.hPow P i.succ)) (hc' : Membership …
    ⊢ Equiv (Subtype fun x => Membership.mem (Submodule.map (Submodule.mkQ (HPow.h …
  -/
  refine Equiv.ofBijective (fun c' => Quotient.mk'' (f (k c' c'.prop) (hk_mem c' c'.prop))) ⟨?_, ?_⟩
    /-
      case succ.refine_1
      S : Type u_1
      inst✝¹ : CommRing S
      P : Ideal S
      P_prime : P.IsPrime
      inst✝ : IsDedekindDomain S
      hP : Ne P Bot.bot
      i : Nat
      ih : Eq (Submodule.cardQuot (HPow.hPow P i)) (HPow.hPow (Submodule.cardQuot P) …
      this : LT.lt (HPow.hPow P (HAdd.hAdd i 1)) (HPow.hPow P i)
      a : S
      a_mem : Membership.mem (HPow.hPow P i) a
      a_not_mem : Not (Membership.mem (HPow.hPow P (HAdd.hAdd i 1)) a)
      f g : (c : S) → Membership.mem (HPow.hPow P i) c → S
      hg : ∀ (c : S) (hc : Membership.mem (HPow.hPow P i) c), Membership.mem (HPow.h …
      hf : ∀ (c : S) (hc : Membership.mem (HPow.hPow P i) c), Eq (HAdd.hAdd (HMul.hM …
      k : (c' : HasQuotient.Quotient S (HPow.hPow P i.succ)) → Membership.mem (Submo …
      hk_mem : ∀ (c' : HasQuotient.Quotient S (HPow.hPow P i.succ)) (hc' : Membershi …
      hk_eq : ∀ (c' : HasQuotient.Quotient S (HPow.hPow P i.succ)) (hc' : Membership …
      ⊢ Function.Injective fun c' => Quotient.mk'' (f (k ↑c' ⋯) ⋯)
    -/
  · rintro ⟨c₁', hc₁'⟩ ⟨c₂', hc₂'⟩ h
    rw [Subtype.mk_eq_mk, ← hk_eq _ hc₁', ← hk_eq _ hc₂', mkQ_apply, mkQ_apply,
      Submodule.Quotient.eq, ← hf _ (hk_mem _ hc₁'), ← hf _ (hk_mem _ hc₂')]
    /-
      case succ.refine_1.mk.mk
      S : Type u_1
      inst✝¹ : CommRing S
      P : Ideal S
      P_prime : P.IsPrime
      inst✝ : IsDedekindDomain S
      hP : Ne P Bot.bot
      i : Nat
      ih : Eq (Submodule.cardQuot (HPow.hPow P i)) (HPow.hPow (Submodule.cardQuot P) …
      this : LT.lt (HPow.hPow P (HAdd.hAdd i 1)) (HPow.hPow P i)
      a : S
      a_mem : Membership.mem (HPow.hPow P i) a
      a_not_mem : Not (Membership.mem (HPow.hPow P (HAdd.hAdd i 1)) a)
      f g : (c : S) → Membership.mem (HPow.hPow P i) c → S
      hg : ∀ (c : S) (hc : Membership.mem (HPow.hPow P i) c), Membership.mem (HPow.h …
      hf : ∀ (c : S) (hc : Membership.mem (HPow.hPow P i) c), Eq (HAdd.hAdd (HMul.hM …
      k : (c' : HasQuotient.Quotient S (HPow.hPow P i.succ)) → Membership.mem (Submo …
      hk_mem : ∀ (c' : HasQuotient.Quotient S (HPow.hPow P i.succ)) (hc' : Membershi …
      hk_eq : ∀ (c' : HasQuotient.Quotient S (HPow.hPow P i.succ)) (hc' : Membership …
      c₁' : HasQuotient.Quotient S (HPow.hPow P i.succ)
      hc₁' : Membership.mem (Submodule.map (Submodule.mkQ (HPow.hPow P i.succ)) (HPo …
      c₂' : HasQuotient.Quotient S (HPow.hPow P i.succ)
      hc₂' : Membership.mem (Submodule.map (Submodule.mkQ (HPow.hPow P i.succ)) (HPo …
      h : Eq ((fun c' => Quotient.mk'' (f (k ↑c' ⋯) ⋯)) ⟨c₁', hc₁'⟩) ((fun c' => Quo …
      ⊢ Membership.mem (HPow.hPow P i.succ) (HSub.hSub (HAdd.hAdd (HMul.hMul a (f (k …
    -/
    refine Ideal.mul_add_mem_pow_succ_inj _ _ _ _ _ _ a_mem (hg _ _) (hg _ _) ?_
    simpa only [Submodule.Quotient.mk''_eq_mk, Submodule.Quotient.mk''_eq_mk,
      Submodule.Quotient.eq] using h
    /-
      case succ.refine_2
      S : Type u_1
      inst✝¹ : CommRing S
      P : Ideal S
      P_prime : P.IsPrime
      inst✝ : IsDedekindDomain S
      hP : Ne P Bot.bot
      i : Nat
      ih : Eq (Submodule.cardQuot (HPow.hPow P i)) (HPow.hPow (Submodule.cardQuot P) …
      this : LT.lt (HPow.hPow P (HAdd.hAdd i 1)) (HPow.hPow P i)
      a : S
      a_mem : Membership.mem (HPow.hPow P i) a
      a_not_mem : Not (Membership.mem (HPow.hPow P (HAdd.hAdd i 1)) a)
      f g : (c : S) → Membership.mem (HPow.hPow P i) c → S
      hg : ∀ (c : S) (hc : Membership.mem (HPow.hPow P i) c), Membership.mem (HPow.h …
      hf : ∀ (c : S) (hc : Membership.mem (HPow.hPow P i) c), Eq (HAdd.hAdd (HMul.hM …
      k : (c' : HasQuotient.Quotient S (HPow.hPow P i.succ)) → Membership.mem (Submo …
      hk_mem : ∀ (c' : HasQuotient.Quotient S (HPow.hPow P i.succ)) (hc' : Membershi …
      hk_eq : ∀ (c' : HasQuotient.Quotient S (HPow.hPow P i.succ)) (hc' : Membership …
      ⊢ Function.Surjective fun c' => Quotient.mk'' (f (k ↑c' ⋯) ⋯)
    -/
  · intro d'
    /-
      case succ.refine_2
      S : Type u_1
      inst✝¹ : CommRing S
      P : Ideal S
      P_prime : P.IsPrime
      inst✝ : IsDedekindDomain S
      hP : Ne P Bot.bot
      i : Nat
      ih : Eq (Submodule.cardQuot (HPow.hPow P i)) (HPow.hPow (Submodule.cardQuot P) …
      this : LT.lt (HPow.hPow P (HAdd.hAdd i 1)) (HPow.hPow P i)
      a : S
      a_mem : Membership.mem (HPow.hPow P i) a
      a_not_mem : Not (Membership.mem (HPow.hPow P (HAdd.hAdd i 1)) a)
      f g : (c : S) → Membership.mem (HPow.hPow P i) c → S
      hg : ∀ (c : S) (hc : Membership.mem (HPow.hPow P i) c), Membership.mem (HPow.h …
      hf : ∀ (c : S) (hc : Membership.mem (HPow.hPow P i) c), Eq (HAdd.hAdd (HMul.hM …
      k : (c' : HasQuotient.Quotient S (HPow.hPow P i.succ)) → Membership.mem (Submo …
      hk_mem : ∀ (c' : HasQuotient.Quotient S (HPow.hPow P i.succ)) (hc' : Membershi …
      hk_eq : ∀ (c' : HasQuotient.Quotient S (HPow.hPow P i.succ)) (hc' : Membership …
      d' : HasQuotient.Quotient S P
      ⊢ Exists fun a => Eq ((fun c' => Quotient.mk'' (f (k ↑c' ⋯) ⋯)) a) d'
    -/
    refine Quotient.inductionOn' d' fun d => ?_
    /-
      case succ.refine_2
      S : Type u_1
      inst✝¹ : CommRing S
      P : Ideal S
      P_prime : P.IsPrime
      inst✝ : IsDedekindDomain S
      hP : Ne P Bot.bot
      i : Nat
      ih : Eq (Submodule.cardQuot (HPow.hPow P i)) (HPow.hPow (Submodule.cardQuot P) …
      this : LT.lt (HPow.hPow P (HAdd.hAdd i 1)) (HPow.hPow P i)
      a : S
      a_mem : Membership.mem (HPow.hPow P i) a
      a_not_mem : Not (Membership.mem (HPow.hPow P (HAdd.hAdd i 1)) a)
      f g : (c : S) → Membership.mem (HPow.hPow P i) c → S
      hg : ∀ (c : S) (hc : Membership.mem (HPow.hPow P i) c), Membership.mem (HPow.h …
      hf : ∀ (c : S) (hc : Membership.mem (HPow.hPow P i) c), Eq (HAdd.hAdd (HMul.hM …
      k : (c' : HasQuotient.Quotient S (HPow.hPow P i.succ)) → Membership.mem (Submo …
      hk_mem : ∀ (c' : HasQuotient.Quotient S (HPow.hPow P i.succ)) (hc' : Membershi …
      hk_eq : ∀ (c' : HasQuotient.Quotient S (HPow.hPow P i.succ)) (hc' : Membership …
      d' : HasQuotient.Quotient S P
      d : S
      ⊢ Exists fun a => Eq ((fun c' => Quotient.mk'' (f (k ↑c' ⋯) ⋯)) a) (Quotient.m …
    -/
    have hd' := (mem_map (f := mkQ (P ^ i.succ))).mpr ⟨a * d, Ideal.mul_mem_right d _ a_mem, rfl⟩
    /-
      case succ.refine_2
      S : Type u_1
      inst✝¹ : CommRing S
      P : Ideal S
      P_prime : P.IsPrime
      inst✝ : IsDedekindDomain S
      hP : Ne P Bot.bot
      i : Nat
      ih : Eq (Submodule.cardQuot (HPow.hPow P i)) (HPow.hPow (Submodule.cardQuot P) …
      this : LT.lt (HPow.hPow P (HAdd.hAdd i 1)) (HPow.hPow P i)
      a : S
      a_mem : Membership.mem (HPow.hPow P i) a
      a_not_mem : Not (Membership.mem (HPow.hPow P (HAdd.hAdd i 1)) a)
      f g : (c : S) → Membership.mem (HPow.hPow P i) c → S
      hg : ∀ (c : S) (hc : Membership.mem (HPow.hPow P i) c), Membership.mem (HPow.h …
      hf : ∀ (c : S) (hc : Membership.mem (HPow.hPow P i) c), Eq (HAdd.hAdd (HMul.hM …
      k : (c' : HasQuotient.Quotient S (HPow.hPow P i.succ)) → Membership.mem (Submo …
      hk_mem : ∀ (c' : HasQuotient.Quotient S (HPow.hPow P i.succ)) (hc' : Membershi …
      hk_eq : ∀ (c' : HasQuotient.Quotient S (HPow.hPow P i.succ)) (hc' : Membership …
      d' : HasQuotient.Quotient S P
      d : S
      hd' : Membership.mem (Submodule.map (Submodule.mkQ (HPow.hPow P i.succ)) (HPow …
      ⊢ Exists fun a => Eq ((fun c' => Quotient.mk'' (f (k ↑c' ⋯) ⋯)) a) (Quotient.m …
    -/
    refine ⟨⟨_, hd'⟩, ?_⟩
    simp only [Submodule.Quotient.mk''_eq_mk, Ideal.Quotient.mk_eq_mk, Ideal.Quotient.eq,
      Subtype.coe_mk]
    refine
      Ideal.mul_add_mem_pow_succ_unique hP a _ _ _ _ a_not_mem (hg _ (hk_mem _ hd')) (zero_mem _) ?_
    /-
      case succ.refine_2
      S : Type u_1
      inst✝¹ : CommRing S
      P : Ideal S
      P_prime : P.IsPrime
      inst✝ : IsDedekindDomain S
      hP : Ne P Bot.bot
      i : Nat
      ih : Eq (Submodule.cardQuot (HPow.hPow P i)) (HPow.hPow (Submodule.cardQuot P) …
      this : LT.lt (HPow.hPow P (HAdd.hAdd i 1)) (HPow.hPow P i)
      a : S
      a_mem : Membership.mem (HPow.hPow P i) a
      a_not_mem : Not (Membership.mem (HPow.hPow P (HAdd.hAdd i 1)) a)
      f g : (c : S) → Membership.mem (HPow.hPow P i) c → S
      hg : ∀ (c : S) (hc : Membership.mem (HPow.hPow P i) c), Membership.mem (HPow.h …
      hf : ∀ (c : S) (hc : Membership.mem (HPow.hPow P i) c), Eq (HAdd.hAdd (HMul.hM …
      k : (c' : HasQuotient.Quotient S (HPow.hPow P i.succ)) → Membership.mem (Submo …
      hk_mem : ∀ (c' : HasQuotient.Quotient S (HPow.hPow P i.succ)) (hc' : Membershi …
      hk_eq : ∀ (c' : HasQuotient.Quotient S (HPow.hPow P i.succ)) (hc' : Membership …
      d' : HasQuotient.Quotient S P
      d : S
      hd' : Membership.mem (Submodule.map (Submodule.mkQ (HPow.hPow P i.succ)) (HPow …
      ⊢ Membership.mem (HPow.hPow P (HAdd.hAdd i 1)) (HSub.hSub (HAdd.hAdd (HMul.hMu …
    -/
    rw [hf, add_zero]
    /-
      case succ.refine_2
      S : Type u_1
      inst✝¹ : CommRing S
      P : Ideal S
      P_prime : P.IsPrime
      inst✝ : IsDedekindDomain S
      hP : Ne P Bot.bot
      i : Nat
      ih : Eq (Submodule.cardQuot (HPow.hPow P i)) (HPow.hPow (Submodule.cardQuot P) …
      this : LT.lt (HPow.hPow P (HAdd.hAdd i 1)) (HPow.hPow P i)
      a : S
      a_mem : Membership.mem (HPow.hPow P i) a
      a_not_mem : Not (Membership.mem (HPow.hPow P (HAdd.hAdd i 1)) a)
      f g : (c : S) → Membership.mem (HPow.hPow P i) c → S
      hg : ∀ (c : S) (hc : Membership.mem (HPow.hPow P i) c), Membership.mem (HPow.h …
      hf : ∀ (c : S) (hc : Membership.mem (HPow.hPow P i) c), Eq (HAdd.hAdd (HMul.hM …
      k : (c' : HasQuotient.Quotient S (HPow.hPow P i.succ)) → Membership.mem (Submo …
      hk_mem : ∀ (c' : HasQuotient.Quotient S (HPow.hPow P i.succ)) (hc' : Membershi …
      hk_eq : ∀ (c' : HasQuotient.Quotient S (HPow.hPow P i.succ)) (hc' : Membership …
      d' : HasQuotient.Quotient S P
      d : S
      hd' : Membership.mem (Submodule.map (Submodule.mkQ (HPow.hPow P i.succ)) (HPow …
      ⊢ Membership.mem (HPow.hPow P (HAdd.hAdd i 1)) (HSub.hSub (k ((Submodule.mkQ ( …
    -/
    exact (Submodule.Quotient.eq _).mp (hk_eq _ hd')
    /-
      🎉 no goals
    -/


/-- Multiplicativity of the ideal norm in number rings. -/
theorem cardQuot_mul [IsDedekindDomain S] [Module.Free ℤ S] (I J : Ideal S) :
    cardQuot (I * J) = cardQuot I * cardQuot J := by
  /-
    S : Type u_1
    inst✝² : CommRing S
    inst✝¹ : IsDedekindDomain S
    inst✝ : Module.Free Int S
    I J : Ideal S
    ⊢ Eq (Submodule.cardQuot (HMul.hMul I J)) (HMul.hMul (Submodule.cardQuot I) (S …
  -/
  let b := Module.Free.chooseBasis ℤ S
  /-
    S : Type u_1
    inst✝² : CommRing S
    inst✝¹ : IsDedekindDomain S
    inst✝ : Module.Free Int S
    I J : Ideal S
    b : Basis (Module.Free.ChooseBasisIndex Int S) Int S := Module.Free.chooseBasi …
    ⊢ Eq (Submodule.cardQuot (HMul.hMul I J)) (HMul.hMul (Submodule.cardQuot I) (S …
  -/
  haveI : Infinite S := Infinite.of_surjective _ b.repr.toEquiv.surjective
  exact UniqueFactorizationMonoid.multiplicative_of_coprime cardQuot I J (cardQuot_bot _ _)
      (fun {I J} hI => by simp [Ideal.isUnit_iff.mp hI, Ideal.mul_top])
      (fun {I} i hI =>
        have : Ideal.IsPrime I := Ideal.isPrime_of_prime hI
        cardQuot_pow_of_prime hI.ne_zero)
      fun {I J} hIJ => cardQuot_mul_of_coprime <| Ideal.isCoprime_iff_sup_eq.mpr
        (Ideal.isUnit_iff.mp
          (hIJ (Ideal.dvd_iff_le.mpr le_sup_left) (Ideal.dvd_iff_le.mpr le_sup_right)))


/-- The absolute norm of the ideal `I : Ideal R` is the cardinality of the quotient `R ⧸ I`. -/
noncomputable def Ideal.absNorm [Nontrivial S] [IsDedekindDomain S] [Module.Free ℤ S] :
    Ideal S →*₀ ℕ where
  toFun := Submodule.cardQuot
                     /-
                       S : Type u_1
                       inst✝³ : CommRing S
                       inst✝² : Nontrivial S
                       inst✝¹ : IsDedekindDomain S
                       inst✝ : Module.Free Int S
                       I J : Ideal S
                       ⊢ Eq ({ toFun := Submodule.cardQuot, map_zero' := ⋯ }.toFun (HMul.hMul I J)) ( …
                     -/
                 /-
                   S : Type u_1
                   inst✝³ : CommRing S
                   inst✝² : Nontrivial S
                   inst✝¹ : IsDedekindDomain S
                   inst✝ : Module.Free Int S
                   ⊢ Eq ({ toFun := Submodule.cardQuot, map_zero' := ⋯ }.toFun 1) 1
                 -/
  map_mul' I J := by dsimp only; rw [cardQuot_mul]
    /-
      S : Type u_1
      inst✝³ : CommRing S
      inst✝² : Nontrivial S
      inst✝¹ : IsDedekindDomain S
      inst✝ : Module.Free Int S
      ⊢ Eq (Submodule.cardQuot 0) 0
    -/
                             /-
                               🎉 no goals
                             -/
    /-
      S : Type u_1
      inst✝³ : CommRing S
      inst✝² : Nontrivial S
      inst✝¹ : IsDedekindDomain S
      inst✝ : Module.Free Int S
      this : Infinite S
      ⊢ Eq (Submodule.cardQuot 0) 0
    -/
                                 /-
                                   🎉 no goals
                                 -/
    /-
      🎉 no goals
    -/
  map_one' := by dsimp only; rw [Ideal.one_eq_top, cardQuot_top]
  map_zero' := by
    have : Infinite S := Module.Free.infinite ℤ S
    rw [Ideal.zero_eq_bot, cardQuot_bot]


theorem absNorm_apply (I : Ideal S) : absNorm I = cardQuot I := rfl


@[simp]
                                                      /-
                                                        S : Type u_1
                                                        inst✝³ : CommRing S
                                                        inst✝² : Nontrivial S
                                                        inst✝¹ : IsDedekindDomain S
                                                        inst✝ : Module.Free Int S
                                                        ⊢ Eq (Ideal.absNorm Bot.bot) 0
                                                      -/
theorem absNorm_bot : absNorm (⊥ : Ideal S) = 0 := by rw [← Ideal.zero_eq_bot, _root_.map_zero]
                                                      /-
                                                        🎉 no goals
                                                      -/


@[simp]
                                                      /-
                                                        S : Type u_1
                                                        inst✝³ : CommRing S
                                                        inst✝² : Nontrivial S
                                                        inst✝¹ : IsDedekindDomain S
                                                        inst✝ : Module.Free Int S
                                                        ⊢ Eq (Ideal.absNorm Top.top) 1
                                                      -/
theorem absNorm_top : absNorm (⊤ : Ideal S) = 1 := by rw [← Ideal.one_eq_top, map_one]
                                                      /-
                                                        🎉 no goals
                                                      -/


@[simp]
theorem absNorm_eq_one_iff {I : Ideal S} : absNorm I = 1 ↔ I = ⊤ := by
  /-
    S : Type u_1
    inst✝³ : CommRing S
    inst✝² : Nontrivial S
    inst✝¹ : IsDedekindDomain S
    inst✝ : Module.Free Int S
    I : Ideal S
    ⊢ Iff (Eq (Ideal.absNorm I) 1) (Eq I Top.top)
  -/
  rw [absNorm_apply, cardQuot_eq_one_iff]
  /-
    🎉 no goals
  -/


theorem absNorm_ne_zero_iff (I : Ideal S) : Ideal.absNorm I ≠ 0 ↔ Finite (S ⧸ I) :=
  ⟨fun h => Nat.finite_of_card_ne_zero h, fun h =>
    (@AddSubgroup.finiteIndex_of_finite_quotient _ _ _ h).finiteIndex⟩


theorem absNorm_dvd_absNorm_of_le {I J : Ideal S} (h : J ≤ I) : Ideal.absNorm I ∣ Ideal.absNorm J :=
  map_dvd absNorm (dvd_iff_le.mpr h)


theorem irreducible_of_irreducible_absNorm {I : Ideal S} (hI : Irreducible (Ideal.absNorm I)) :
    Irreducible I :=
  irreducible_iff.mpr
    ⟨fun h =>
                      /-
                        S : Type u_1
                        inst✝³ : CommRing S
                        inst✝² : Nontrivial S
                        inst✝¹ : IsDedekindDomain S
                        inst✝ : Module.Free Int S
                        I : Ideal S
                        hI : Irreducible (Ideal.absNorm I)
                        h : IsUnit I
                        ⊢ IsUnit (Ideal.absNorm I)
                      -/
      hI.not_unit (by simpa only [Ideal.isUnit_iff, Nat.isUnit_iff, absNorm_eq_one_iff] using h),
                      /-
                        🎉 no goals
                      -/
      by
      /-
        S : Type u_1
        inst✝³ : CommRing S
        inst✝² : Nontrivial S
        inst✝¹ : IsDedekindDomain S
        inst✝ : Module.Free Int S
        I : Ideal S
        hI : Irreducible (Ideal.absNorm I)
        ⊢ ∀ (a b : Ideal S), Eq I (HMul.hMul a b) → Or (IsUnit a) (IsUnit b)
      -/
      rintro a b rfl
      simpa only [Ideal.isUnit_iff, Nat.isUnit_iff, absNorm_eq_one_iff] using
        hI.isUnit_or_isUnit (_root_.map_mul absNorm a b)⟩


theorem isPrime_of_irreducible_absNorm {I : Ideal S} (hI : Irreducible (Ideal.absNorm I)) :
    I.IsPrime :=
  isPrime_of_prime
    (UniqueFactorizationMonoid.irreducible_iff_prime.mp (irreducible_of_irreducible_absNorm hI))


theorem prime_of_irreducible_absNorm_span {a : S} (ha : a ≠ 0)
    (hI : Irreducible (Ideal.absNorm (Ideal.span ({a} : Set S)))) : Prime a :=
  (Ideal.span_singleton_prime ha).mp (isPrime_of_irreducible_absNorm hI)


theorem absNorm_mem (I : Ideal S) : ↑(Ideal.absNorm I) ∈ I := by
  rw [absNorm_apply, cardQuot, ← Ideal.Quotient.eq_zero_iff_mem, map_natCast,
    Quotient.index_eq_zero]


theorem span_singleton_absNorm_le (I : Ideal S) : Ideal.span {(Ideal.absNorm I : S)} ≤ I := by
  /-
    S : Type u_1
    inst✝³ : CommRing S
    inst✝² : Nontrivial S
    inst✝¹ : IsDedekindDomain S
    inst✝ : Module.Free Int S
    I : Ideal S
    ⊢ LE.le (Ideal.span (Singleton.singleton ↑(Ideal.absNorm I))) I
  -/
  simp only [Ideal.span_le, Set.singleton_subset_iff, SetLike.mem_coe, Ideal.absNorm_mem I]
  /-
    🎉 no goals
  -/


theorem span_singleton_absNorm {I : Ideal S} (hI : (Ideal.absNorm I).Prime) :
    Ideal.span (singleton (Ideal.absNorm I : ℤ)) = I.comap (algebraMap ℤ S) := by
  have : Ideal.IsPrime (Ideal.span (singleton (Ideal.absNorm I : ℤ))) := by
    rwa [Ideal.span_singleton_prime (Int.ofNat_ne_zero.mpr hI.ne_zero), ← Nat.prime_iff_prime_int]
  /-
    S : Type u_1
    inst✝³ : CommRing S
    inst✝² : Nontrivial S
    inst✝¹ : IsDedekindDomain S
    inst✝ : Module.Free Int S
    I : Ideal S
    hI : Nat.Prime (Ideal.absNorm I)
    this : (Ideal.span (Singleton.singleton ↑(Ideal.absNorm I))).IsPrime
    ⊢ Eq (Ideal.span (Singleton.singleton ↑(Ideal.absNorm I))) (Ideal.comap (algeb …
  -/
  apply (this.isMaximal _).eq_of_le
  · exact ((isPrime_of_irreducible_absNorm
      ((Nat.irreducible_iff_nat_prime _).mpr hI)).comap (algebraMap ℤ S)).ne_top
    /-
      case IJ
      S : Type u_1
      inst✝³ : CommRing S
      inst✝² : Nontrivial S
      inst✝¹ : IsDedekindDomain S
      inst✝ : Module.Free Int S
      I : Ideal S
      hI : Nat.Prime (Ideal.absNorm I)
      this : (Ideal.span (Singleton.singleton ↑(Ideal.absNorm I))).IsPrime
      ⊢ LE.le (Ideal.span (Singleton.singleton ↑(Ideal.absNorm I))) (Ideal.comap (al …
    -/
  · rw [span_singleton_le_iff_mem, mem_comap, algebraMap_int_eq, map_natCast]
    /-
      case IJ
      S : Type u_1
      inst✝³ : CommRing S
      inst✝² : Nontrivial S
      inst✝¹ : IsDedekindDomain S
      inst✝ : Module.Free Int S
      I : Ideal S
      hI : Nat.Prime (Ideal.absNorm I)
      this : (Ideal.span (Singleton.singleton ↑(Ideal.absNorm I))).IsPrime
      ⊢ Membership.mem I ↑(Ideal.absNorm I)
    -/
    exact absNorm_mem I
    /-
      🎉 no goals
    -/
    /-
      S : Type u_1
      inst✝³ : CommRing S
      inst✝² : Nontrivial S
      inst✝¹ : IsDedekindDomain S
      inst✝ : Module.Free Int S
      I : Ideal S
      hI : Nat.Prime (Ideal.absNorm I)
      this : (Ideal.span (Singleton.singleton ↑(Ideal.absNorm I))).IsPrime
      ⊢ Ne (Ideal.span (Singleton.singleton ↑(Ideal.absNorm I))) Bot.bot
    -/
  · rw [Ne, span_singleton_eq_bot]
    /-
      S : Type u_1
      inst✝³ : CommRing S
      inst✝² : Nontrivial S
      inst✝¹ : IsDedekindDomain S
      inst✝ : Module.Free Int S
      I : Ideal S
      hI : Nat.Prime (Ideal.absNorm I)
      this : (Ideal.span (Singleton.singleton ↑(Ideal.absNorm I))).IsPrime
      ⊢ Not (Eq (↑(Ideal.absNorm I)) 0)
    -/
    exact Int.ofNat_ne_zero.mpr hI.ne_zero
    /-
      🎉 no goals
    -/


/-- Let `e : S ≃ I` be an additive isomorphism (therefore a `ℤ`-linear equiv).
Then an alternative way to compute the norm of `I` is given by taking the determinant of `e`.
See `natAbs_det_basis_change` for a more familiar formulation of this result. -/
theorem natAbs_det_equiv (I : Ideal S) {E : Type*} [EquivLike E S I] [AddEquivClass E S I] (e : E) :
    Int.natAbs
        (LinearMap.det
          ((Submodule.subtype I).restrictScalars ℤ ∘ₗ AddMonoidHom.toIntLinearMap (e : S →+ I))) =
      Ideal.absNorm I := by
  -- `S ⧸ I` might be infinite if `I = ⊥`, but then `e` can't be an equiv.
  /-
    S : Type u_1
    inst✝⁶ : CommRing S
    inst✝⁵ : Nontrivial S
    inst✝⁴ : IsDedekindDomain S
    inst✝³ : Module.Free Int S
    inst✝² : Module.Finite Int S
    I : Ideal S
    E : Type u_2
    inst✝¹ : EquivLike E S (Subtype fun x => Membership.mem I x)
    inst✝ : AddEquivClass E S (Subtype fun x => Membership.mem I x)
    e : E
    ⊢ Eq (LinearMap.det ((↑Int (Submodule.subtype I)).comp (↑e).toIntLinearMap)).n …
  -/
  by_cases hI : I = ⊥
    /-
      case pos
      S : Type u_1
      inst✝⁶ : CommRing S
      inst✝⁵ : Nontrivial S
      inst✝⁴ : IsDedekindDomain S
      inst✝³ : Module.Free Int S
      inst✝² : Module.Finite Int S
      I : Ideal S
      E : Type u_2
      inst✝¹ : EquivLike E S (Subtype fun x => Membership.mem I x)
      inst✝ : AddEquivClass E S (Subtype fun x => Membership.mem I x)
      e : E
      hI : Eq I Bot.bot
      ⊢ Eq (LinearMap.det ((↑Int (Submodule.subtype I)).comp (↑e).toIntLinearMap)).n …
    -/
  · subst hI
    /-
      case pos
      S : Type u_1
      inst✝⁶ : CommRing S
      inst✝⁵ : Nontrivial S
      inst✝⁴ : IsDedekindDomain S
      inst✝³ : Module.Free Int S
      inst✝² : Module.Finite Int S
      E : Type u_2
      e : E
      inst✝¹ : EquivLike E S (Subtype fun x => Membership.mem Bot.bot x)
      inst✝ : AddEquivClass E S (Subtype fun x => Membership.mem Bot.bot x)
      ⊢ Eq (LinearMap.det ((↑Int (Submodule.subtype Bot.bot)).comp (↑e).toIntLinearM …
    -/
    have : (1 : S) ≠ 0 := one_ne_zero
    /-
      case pos
      S : Type u_1
      inst✝⁶ : CommRing S
      inst✝⁵ : Nontrivial S
      inst✝⁴ : IsDedekindDomain S
      inst✝³ : Module.Free Int S
      inst✝² : Module.Finite Int S
      E : Type u_2
      e : E
      inst✝¹ : EquivLike E S (Subtype fun x => Membership.mem Bot.bot x)
      inst✝ : AddEquivClass E S (Subtype fun x => Membership.mem Bot.bot x)
      this : Ne 1 0
      ⊢ Eq (LinearMap.det ((↑Int (Submodule.subtype Bot.bot)).comp (↑e).toIntLinearM …
    -/
    have : (1 : S) = 0 := EquivLike.injective e (Subsingleton.elim _ _)
    /-
      case pos
      S : Type u_1
      inst✝⁶ : CommRing S
      inst✝⁵ : Nontrivial S
      inst✝⁴ : IsDedekindDomain S
      inst✝³ : Module.Free Int S
      inst✝² : Module.Finite Int S
      E : Type u_2
      e : E
      inst✝¹ : EquivLike E S (Subtype fun x => Membership.mem Bot.bot x)
      inst✝ : AddEquivClass E S (Subtype fun x => Membership.mem Bot.bot x)
      this✝ : Ne 1 0
      this : Eq 1 0
      ⊢ Eq (LinearMap.det ((↑Int (Submodule.subtype Bot.bot)).comp (↑e).toIntLinearM …
    -/
    contradiction
    /-
      🎉 no goals
    -/
  /-
    case neg
    S : Type u_1
    inst✝⁶ : CommRing S
    inst✝⁵ : Nontrivial S
    inst✝⁴ : IsDedekindDomain S
    inst✝³ : Module.Free Int S
    inst✝² : Module.Finite Int S
    I : Ideal S
    E : Type u_2
    inst✝¹ : EquivLike E S (Subtype fun x => Membership.mem I x)
    inst✝ : AddEquivClass E S (Subtype fun x => Membership.mem I x)
    e : E
    hI : Not (Eq I Bot.bot)
    ⊢ Eq (LinearMap.det ((↑Int (Submodule.subtype I)).comp (↑e).toIntLinearMap)).n …
  -/
  let ι := Module.Free.ChooseBasisIndex ℤ S
  /-
    case neg
    S : Type u_1
    inst✝⁶ : CommRing S
    inst✝⁵ : Nontrivial S
    inst✝⁴ : IsDedekindDomain S
    inst✝³ : Module.Free Int S
    inst✝² : Module.Finite Int S
    I : Ideal S
    E : Type u_2
    inst✝¹ : EquivLike E S (Subtype fun x => Membership.mem I x)
    inst✝ : AddEquivClass E S (Subtype fun x => Membership.mem I x)
    e : E
    hI : Not (Eq I Bot.bot)
    ι : Type u_1 := Module.Free.ChooseBasisIndex Int S
    ⊢ Eq (LinearMap.det ((↑Int (Submodule.subtype I)).comp (↑e).toIntLinearMap)).n …
  -/
  let b := Module.Free.chooseBasis ℤ S
  /-
    case neg
    S : Type u_1
    inst✝⁶ : CommRing S
    inst✝⁵ : Nontrivial S
    inst✝⁴ : IsDedekindDomain S
    inst✝³ : Module.Free Int S
    inst✝² : Module.Finite Int S
    I : Ideal S
    E : Type u_2
    inst✝¹ : EquivLike E S (Subtype fun x => Membership.mem I x)
    inst✝ : AddEquivClass E S (Subtype fun x => Membership.mem I x)
    e : E
    hI : Not (Eq I Bot.bot)
    ι : Type u_1 := Module.Free.ChooseBasisIndex Int S
    b : Basis (Module.Free.ChooseBasisIndex Int S) Int S := Module.Free.chooseBasi …
    ⊢ Eq (LinearMap.det ((↑Int (Submodule.subtype I)).comp (↑e).toIntLinearMap)).n …
  -/
  cases isEmpty_or_nonempty ι
    /-
      case neg.inl
      S : Type u_1
      inst✝⁶ : CommRing S
      inst✝⁵ : Nontrivial S
      inst✝⁴ : IsDedekindDomain S
      inst✝³ : Module.Free Int S
      inst✝² : Module.Finite Int S
      I : Ideal S
      E : Type u_2
      inst✝¹ : EquivLike E S (Subtype fun x => Membership.mem I x)
      inst✝ : AddEquivClass E S (Subtype fun x => Membership.mem I x)
      e : E
      hI : Not (Eq I Bot.bot)
      ι : Type u_1 := Module.Free.ChooseBasisIndex Int S
      b : Basis (Module.Free.ChooseBasisIndex Int S) Int S := Module.Free.chooseBasi …
      h✝ : IsEmpty ι
      ⊢ Eq (LinearMap.det ((↑Int (Submodule.subtype I)).comp (↑e).toIntLinearMap)).n …
    -/
  · nontriviality S
    exact (not_nontrivial_iff_subsingleton.mpr
      (Function.Surjective.subsingleton b.repr.toEquiv.symm.surjective) (by infer_instance)).elim
  -- Thus `(S ⧸ I)` is isomorphic to a product of `ZMod`s, so it is a fintype.
  /-
    case neg.inr
    S : Type u_1
    inst✝⁶ : CommRing S
    inst✝⁵ : Nontrivial S
    inst✝⁴ : IsDedekindDomain S
    inst✝³ : Module.Free Int S
    inst✝² : Module.Finite Int S
    I : Ideal S
    E : Type u_2
    inst✝¹ : EquivLike E S (Subtype fun x => Membership.mem I x)
    inst✝ : AddEquivClass E S (Subtype fun x => Membership.mem I x)
    e : E
    hI : Not (Eq I Bot.bot)
    ι : Type u_1 := Module.Free.ChooseBasisIndex Int S
    b : Basis (Module.Free.ChooseBasisIndex Int S) Int S := Module.Free.chooseBasi …
    h✝ : Nonempty ι
    ⊢ Eq (LinearMap.det ((↑Int (Submodule.subtype I)).comp (↑e).toIntLinearMap)).n …
  -/
  letI := Ideal.fintypeQuotientOfFreeOfNeBot I hI
  -- Use the Smith normal form to choose a nice basis for `I`.
  /-
    case neg.inr
    S : Type u_1
    inst✝⁶ : CommRing S
    inst✝⁵ : Nontrivial S
    inst✝⁴ : IsDedekindDomain S
    inst✝³ : Module.Free Int S
    inst✝² : Module.Finite Int S
    I : Ideal S
    E : Type u_2
    inst✝¹ : EquivLike E S (Subtype fun x => Membership.mem I x)
    inst✝ : AddEquivClass E S (Subtype fun x => Membership.mem I x)
    e : E
    hI : Not (Eq I Bot.bot)
    ι : Type u_1 := Module.Free.ChooseBasisIndex Int S
    b : Basis (Module.Free.ChooseBasisIndex Int S) Int S := Module.Free.chooseBasi …
    h✝ : Nonempty ι
    this : Fintype (HasQuotient.Quotient S I) := I.fintypeQuotientOfFreeOfNeBot hI
    ⊢ Eq (LinearMap.det ((↑Int (Submodule.subtype I)).comp (↑e).toIntLinearMap)).n …
  -/
  letI := Classical.decEq ι
  /-
    case neg.inr
    S : Type u_1
    inst✝⁶ : CommRing S
    inst✝⁵ : Nontrivial S
    inst✝⁴ : IsDedekindDomain S
    inst✝³ : Module.Free Int S
    inst✝² : Module.Finite Int S
    I : Ideal S
    E : Type u_2
    inst✝¹ : EquivLike E S (Subtype fun x => Membership.mem I x)
    inst✝ : AddEquivClass E S (Subtype fun x => Membership.mem I x)
    e : E
    hI : Not (Eq I Bot.bot)
    ι : Type u_1 := Module.Free.ChooseBasisIndex Int S
    b : Basis (Module.Free.ChooseBasisIndex Int S) Int S := Module.Free.chooseBasi …
    h✝ : Nonempty ι
    this✝ : Fintype (HasQuotient.Quotient S I) := I.fintypeQuotientOfFreeOfNeBot hI
    this : DecidableEq ι := Classical.decEq ι
    ⊢ Eq (LinearMap.det ((↑Int (Submodule.subtype I)).comp (↑e).toIntLinearMap)).n …
  -/
  let a := I.smithCoeffs b hI
  /-
    case neg.inr
    S : Type u_1
    inst✝⁶ : CommRing S
    inst✝⁵ : Nontrivial S
    inst✝⁴ : IsDedekindDomain S
    inst✝³ : Module.Free Int S
    inst✝² : Module.Finite Int S
    I : Ideal S
    E : Type u_2
    inst✝¹ : EquivLike E S (Subtype fun x => Membership.mem I x)
    inst✝ : AddEquivClass E S (Subtype fun x => Membership.mem I x)
    e : E
    hI : Not (Eq I Bot.bot)
    ι : Type u_1 := Module.Free.ChooseBasisIndex Int S
    b : Basis (Module.Free.ChooseBasisIndex Int S) Int S := Module.Free.chooseBasi …
    h✝ : Nonempty ι
    this✝ : Fintype (HasQuotient.Quotient S I) := I.fintypeQuotientOfFreeOfNeBot hI
    this : DecidableEq ι := Classical.decEq ι
    a : Module.Free.ChooseBasisIndex Int S → Int := Ideal.smithCoeffs b I hI
    ⊢ Eq (LinearMap.det ((↑Int (Submodule.subtype I)).comp (↑e).toIntLinearMap)).n …
  -/
  let b' := I.ringBasis b hI
  /-
    case neg.inr
    S : Type u_1
    inst✝⁶ : CommRing S
    inst✝⁵ : Nontrivial S
    inst✝⁴ : IsDedekindDomain S
    inst✝³ : Module.Free Int S
    inst✝² : Module.Finite Int S
    I : Ideal S
    E : Type u_2
    inst✝¹ : EquivLike E S (Subtype fun x => Membership.mem I x)
    inst✝ : AddEquivClass E S (Subtype fun x => Membership.mem I x)
    e : E
    hI : Not (Eq I Bot.bot)
    ι : Type u_1 := Module.Free.ChooseBasisIndex Int S
    b : Basis (Module.Free.ChooseBasisIndex Int S) Int S := Module.Free.chooseBasi …
    h✝ : Nonempty ι
    this✝ : Fintype (HasQuotient.Quotient S I) := I.fintypeQuotientOfFreeOfNeBot hI
    this : DecidableEq ι := Classical.decEq ι
    a : Module.Free.ChooseBasisIndex Int S → Int := Ideal.smithCoeffs b I hI
    b' : Basis (Module.Free.ChooseBasisIndex Int S) Int S := Ideal.ringBasis b I hI
    ⊢ Eq (LinearMap.det ((↑Int (Submodule.subtype I)).comp (↑e).toIntLinearMap)).n …
  -/
  let ab := I.selfBasis b hI
  /-
    case neg.inr
    S : Type u_1
    inst✝⁶ : CommRing S
    inst✝⁵ : Nontrivial S
    inst✝⁴ : IsDedekindDomain S
    inst✝³ : Module.Free Int S
    inst✝² : Module.Finite Int S
    I : Ideal S
    E : Type u_2
    inst✝¹ : EquivLike E S (Subtype fun x => Membership.mem I x)
    inst✝ : AddEquivClass E S (Subtype fun x => Membership.mem I x)
    e : E
    hI : Not (Eq I Bot.bot)
    ι : Type u_1 := Module.Free.ChooseBasisIndex Int S
    b : Basis (Module.Free.ChooseBasisIndex Int S) Int S := Module.Free.chooseBasi …
    h✝ : Nonempty ι
    this✝ : Fintype (HasQuotient.Quotient S I) := I.fintypeQuotientOfFreeOfNeBot hI
    this : DecidableEq ι := Classical.decEq ι
    a : Module.Free.ChooseBasisIndex Int S → Int := Ideal.smithCoeffs b I hI
    b' : Basis (Module.Free.ChooseBasisIndex Int S) Int S := Ideal.ringBasis b I hI
    ab : Basis (Module.Free.ChooseBasisIndex Int S) Int (Subtype fun x => Membersh …
    ⊢ Eq (LinearMap.det ((↑Int (Submodule.subtype I)).comp (↑e).toIntLinearMap)).n …
  -/
  have ab_eq := I.selfBasis_def b hI
  /-
    case neg.inr
    S : Type u_1
    inst✝⁶ : CommRing S
    inst✝⁵ : Nontrivial S
    inst✝⁴ : IsDedekindDomain S
    inst✝³ : Module.Free Int S
    inst✝² : Module.Finite Int S
    I : Ideal S
    E : Type u_2
    inst✝¹ : EquivLike E S (Subtype fun x => Membership.mem I x)
    inst✝ : AddEquivClass E S (Subtype fun x => Membership.mem I x)
    e : E
    hI : Not (Eq I Bot.bot)
    ι : Type u_1 := Module.Free.ChooseBasisIndex Int S
    b : Basis (Module.Free.ChooseBasisIndex Int S) Int S := Module.Free.chooseBasi …
    h✝ : Nonempty ι
    this✝ : Fintype (HasQuotient.Quotient S I) := I.fintypeQuotientOfFreeOfNeBot hI
    this : DecidableEq ι := Classical.decEq ι
    a : Module.Free.ChooseBasisIndex Int S → Int := Ideal.smithCoeffs b I hI
    b' : Basis (Module.Free.ChooseBasisIndex Int S) Int S := Ideal.ringBasis b I hI
    ab : Basis (Module.Free.ChooseBasisIndex Int S) Int (Subtype fun x => Membersh …
    ab_eq : ∀ (i : Module.Free.ChooseBasisIndex Int S), Eq (↑((Ideal.selfBasis b I …
    ⊢ Eq (LinearMap.det ((↑Int (Submodule.subtype I)).comp (↑e).toIntLinearMap)).n …
  -/
  let e' : S ≃ₗ[ℤ] I := b'.equiv ab (Equiv.refl _)
  /-
    case neg.inr
    S : Type u_1
    inst✝⁶ : CommRing S
    inst✝⁵ : Nontrivial S
    inst✝⁴ : IsDedekindDomain S
    inst✝³ : Module.Free Int S
    inst✝² : Module.Finite Int S
    I : Ideal S
    E : Type u_2
    inst✝¹ : EquivLike E S (Subtype fun x => Membership.mem I x)
    inst✝ : AddEquivClass E S (Subtype fun x => Membership.mem I x)
    e : E
    hI : Not (Eq I Bot.bot)
    ι : Type u_1 := Module.Free.ChooseBasisIndex Int S
    b : Basis (Module.Free.ChooseBasisIndex Int S) Int S := Module.Free.chooseBasi …
    h✝ : Nonempty ι
    this✝ : Fintype (HasQuotient.Quotient S I) := I.fintypeQuotientOfFreeOfNeBot hI
    this : DecidableEq ι := Classical.decEq ι
    a : Module.Free.ChooseBasisIndex Int S → Int := Ideal.smithCoeffs b I hI
    b' : Basis (Module.Free.ChooseBasisIndex Int S) Int S := Ideal.ringBasis b I hI
    ab : Basis (Module.Free.ChooseBasisIndex Int S) Int (Subtype fun x => Membersh …
    ab_eq : ∀ (i : Module.Free.ChooseBasisIndex Int S), Eq (↑((Ideal.selfBasis b I …
    e' : LinearEquiv (RingHom.id Int) S (Subtype fun x => Membership.mem I x) := b …
    ⊢ Eq (LinearMap.det ((↑Int (Submodule.subtype I)).comp (↑e).toIntLinearMap)).n …
  -/
  let f : S →ₗ[ℤ] S := (I.subtype.restrictScalars ℤ).comp (e' : S →ₗ[ℤ] I)
  /-
    case neg.inr
    S : Type u_1
    inst✝⁶ : CommRing S
    inst✝⁵ : Nontrivial S
    inst✝⁴ : IsDedekindDomain S
    inst✝³ : Module.Free Int S
    inst✝² : Module.Finite Int S
    I : Ideal S
    E : Type u_2
    inst✝¹ : EquivLike E S (Subtype fun x => Membership.mem I x)
    inst✝ : AddEquivClass E S (Subtype fun x => Membership.mem I x)
    e : E
    hI : Not (Eq I Bot.bot)
    ι : Type u_1 := Module.Free.ChooseBasisIndex Int S
    b : Basis (Module.Free.ChooseBasisIndex Int S) Int S := Module.Free.chooseBasi …
    h✝ : Nonempty ι
    this✝ : Fintype (HasQuotient.Quotient S I) := I.fintypeQuotientOfFreeOfNeBot hI
    this : DecidableEq ι := Classical.decEq ι
    a : Module.Free.ChooseBasisIndex Int S → Int := Ideal.smithCoeffs b I hI
    b' : Basis (Module.Free.ChooseBasisIndex Int S) Int S := Ideal.ringBasis b I hI
    ab : Basis (Module.Free.ChooseBasisIndex Int S) Int (Subtype fun x => Membersh …
    ab_eq : ∀ (i : Module.Free.ChooseBasisIndex Int S), Eq (↑((Ideal.selfBasis b I …
    e' : LinearEquiv (RingHom.id Int) S (Subtype fun x => Membership.mem I x) := b …
    f : LinearMap (RingHom.id Int) S S := (↑Int (Submodule.subtype I)).comp ↑e'
    ⊢ Eq (LinearMap.det ((↑Int (Submodule.subtype I)).comp (↑e).toIntLinearMap)).n …
  -/
  let f_apply : ∀ x, f x = b'.equiv ab (Equiv.refl _) x := fun x => rfl
  suffices (LinearMap.det f).natAbs = Ideal.absNorm I by
    calc
      _ = (LinearMap.det ((Submodule.subtype I).restrictScalars ℤ ∘ₗ
            (AddEquiv.toIntLinearEquiv e : S ≃ₗ[ℤ] I))).natAbs := rfl
      _ = (LinearMap.det ((Submodule.subtype I).restrictScalars ℤ ∘ₗ _)).natAbs :=
            Int.natAbs_eq_iff_associated.mpr (LinearMap.associated_det_comp_equiv _ _ _)
      _ = absNorm I := this
  have ha : ∀ i, f (b' i) = a i • b' i := by
    intro i; rw [f_apply, b'.equiv_apply, Equiv.refl_apply, ab_eq]
  -- `det f` is equal to `∏ i, a i`,
  /-
    case neg.inr
    S : Type u_1
    inst✝⁶ : CommRing S
    inst✝⁵ : Nontrivial S
    inst✝⁴ : IsDedekindDomain S
    inst✝³ : Module.Free Int S
    inst✝² : Module.Finite Int S
    I : Ideal S
    E : Type u_2
    inst✝¹ : EquivLike E S (Subtype fun x => Membership.mem I x)
    inst✝ : AddEquivClass E S (Subtype fun x => Membership.mem I x)
    e : E
    hI : Not (Eq I Bot.bot)
    ι : Type u_1 := Module.Free.ChooseBasisIndex Int S
    b : Basis (Module.Free.ChooseBasisIndex Int S) Int S := Module.Free.chooseBasi …
    h✝ : Nonempty ι
    this✝ : Fintype (HasQuotient.Quotient S I) := I.fintypeQuotientOfFreeOfNeBot hI
    this : DecidableEq ι := Classical.decEq ι
    a : Module.Free.ChooseBasisIndex Int S → Int := Ideal.smithCoeffs b I hI
    b' : Basis (Module.Free.ChooseBasisIndex Int S) Int S := Ideal.ringBasis b I hI
    ab : Basis (Module.Free.ChooseBasisIndex Int S) Int (Subtype fun x => Membersh …
    ab_eq : ∀ (i : Module.Free.ChooseBasisIndex Int S), Eq (↑((Ideal.selfBasis b I …
    e' : LinearEquiv (RingHom.id Int) S (Subtype fun x => Membership.mem I x) := b …
    f : LinearMap (RingHom.id Int) S S := (↑Int (Submodule.subtype I)).comp ↑e'
    f_apply : ∀ (x : S), Eq (f x) ↑((b'.equiv ab (Equiv.refl (Module.Free.ChooseBa …
    ha : ∀ (i : Module.Free.ChooseBasisIndex Int S), Eq (f (b' i)) (HSMul.hSMul (a …
    ⊢ Eq (LinearMap.det f).natAbs (Ideal.absNorm I)
  -/
  letI := Classical.decEq ι
  calc
    Int.natAbs (LinearMap.det f) = Int.natAbs (LinearMap.toMatrix b' b' f).det := by
      rw [LinearMap.det_toMatrix]
    _ = Int.natAbs (Matrix.diagonal a).det := ?_
    _ = Int.natAbs (∏ i, a i) := by rw [Matrix.det_diagonal]
    _ = ∏ i, Int.natAbs (a i) := map_prod Int.natAbsHom a Finset.univ
    _ = Nat.card (S ⧸ I) := ?_
    _ = absNorm I := (Submodule.cardQuot_apply _).symm
  -- since `LinearMap.toMatrix b' b' f` is the diagonal matrix with `a` along the diagonal.
    /-
      case neg.inr.calc_1
      S : Type u_1
      inst✝⁶ : CommRing S
      inst✝⁵ : Nontrivial S
      inst✝⁴ : IsDedekindDomain S
      inst✝³ : Module.Free Int S
      inst✝² : Module.Finite Int S
      I : Ideal S
      E : Type u_2
      inst✝¹ : EquivLike E S (Subtype fun x => Membership.mem I x)
      inst✝ : AddEquivClass E S (Subtype fun x => Membership.mem I x)
      e : E
      hI : Not (Eq I Bot.bot)
      ι : Type u_1 := Module.Free.ChooseBasisIndex Int S
      b : Basis (Module.Free.ChooseBasisIndex Int S) Int S := Module.Free.chooseBasi …
      h✝ : Nonempty ι
      this✝¹ : Fintype (HasQuotient.Quotient S I) := I.fintypeQuotientOfFreeOfNeBot hI
      this✝ : DecidableEq ι := Classical.decEq ι
      a : Module.Free.ChooseBasisIndex Int S → Int := Ideal.smithCoeffs b I hI
      b' : Basis (Module.Free.ChooseBasisIndex Int S) Int S := Ideal.ringBasis b I hI
      ab : Basis (Module.Free.ChooseBasisIndex Int S) Int (Subtype fun x => Membersh …
      ab_eq : ∀ (i : Module.Free.ChooseBasisIndex Int S), Eq (↑((Ideal.selfBasis b I …
      e' : LinearEquiv (RingHom.id Int) S (Subtype fun x => Membership.mem I x) := b …
      f : LinearMap (RingHom.id Int) S S := (↑Int (Submodule.subtype I)).comp ↑e'
      f_apply : ∀ (x : S), Eq (f x) ↑((b'.equiv ab (Equiv.refl (Module.Free.ChooseBa …
      ha : ∀ (i : Module.Free.ChooseBasisIndex Int S), Eq (f (b' i)) (HSMul.hSMul (a …
      this : DecidableEq ι := Classical.decEq ι
      ⊢ Eq ((LinearMap.toMatrix b' b') f).det.natAbs (Matrix.diagonal a).det.natAbs
    -/
  · congr 2; ext i j
    rw [LinearMap.toMatrix_apply, ha, LinearEquiv.map_smul, Basis.repr_self, Finsupp.smul_single,
      smul_eq_mul, mul_one]
    /-
      case neg.inr.calc_1.e_m.e_M.a
      S : Type u_1
      inst✝⁶ : CommRing S
      inst✝⁵ : Nontrivial S
      inst✝⁴ : IsDedekindDomain S
      inst✝³ : Module.Free Int S
      inst✝² : Module.Finite Int S
      I : Ideal S
      E : Type u_2
      inst✝¹ : EquivLike E S (Subtype fun x => Membership.mem I x)
      inst✝ : AddEquivClass E S (Subtype fun x => Membership.mem I x)
      e : E
      hI : Not (Eq I Bot.bot)
      ι : Type u_1 := Module.Free.ChooseBasisIndex Int S
      b : Basis (Module.Free.ChooseBasisIndex Int S) Int S := Module.Free.chooseBasi …
      h✝ : Nonempty ι
      this✝¹ : Fintype (HasQuotient.Quotient S I) := I.fintypeQuotientOfFreeOfNeBot hI
      this✝ : DecidableEq ι := Classical.decEq ι
      a : Module.Free.ChooseBasisIndex Int S → Int := Ideal.smithCoeffs b I hI
      b' : Basis (Module.Free.ChooseBasisIndex Int S) Int S := Ideal.ringBasis b I hI
      ab : Basis (Module.Free.ChooseBasisIndex Int S) Int (Subtype fun x => Membersh …
      ab_eq : ∀ (i : Module.Free.ChooseBasisIndex Int S), Eq (↑((Ideal.selfBasis b I …
      e' : LinearEquiv (RingHom.id Int) S (Subtype fun x => Membership.mem I x) := b …
      f : LinearMap (RingHom.id Int) S S := (↑Int (Submodule.subtype I)).comp ↑e'
      f_apply : ∀ (x : S), Eq (f x) ↑((b'.equiv ab (Equiv.refl (Module.Free.ChooseBa …
      ha : ∀ (i : Module.Free.ChooseBasisIndex Int S), Eq (f (b' i)) (HSMul.hSMul (a …
      this : DecidableEq ι := Classical.decEq ι
      i j : Module.Free.ChooseBasisIndex Int S
      ⊢ Eq ((Finsupp.single j (a j)) i) (Matrix.diagonal a i j)
    -/
    by_cases h : i = j
      /-
        case pos
        S : Type u_1
        inst✝⁶ : CommRing S
        inst✝⁵ : Nontrivial S
        inst✝⁴ : IsDedekindDomain S
        inst✝³ : Module.Free Int S
        inst✝² : Module.Finite Int S
        I : Ideal S
        E : Type u_2
        inst✝¹ : EquivLike E S (Subtype fun x => Membership.mem I x)
        inst✝ : AddEquivClass E S (Subtype fun x => Membership.mem I x)
        e : E
        hI : Not (Eq I Bot.bot)
        ι : Type u_1 := Module.Free.ChooseBasisIndex Int S
        b : Basis (Module.Free.ChooseBasisIndex Int S) Int S := Module.Free.chooseBasi …
        h✝ : Nonempty ι
        this✝¹ : Fintype (HasQuotient.Quotient S I) := I.fintypeQuotientOfFreeOfNeBot hI
        this✝ : DecidableEq ι := Classical.decEq ι
        a : Module.Free.ChooseBasisIndex Int S → Int := Ideal.smithCoeffs b I hI
        b' : Basis (Module.Free.ChooseBasisIndex Int S) Int S := Ideal.ringBasis b I hI
        ab : Basis (Module.Free.ChooseBasisIndex Int S) Int (Subtype fun x => Membersh …
        ab_eq : ∀ (i : Module.Free.ChooseBasisIndex Int S), Eq (↑((Ideal.selfBasis b I …
        e' : LinearEquiv (RingHom.id Int) S (Subtype fun x => Membership.mem I x) := b …
        f : LinearMap (RingHom.id Int) S S := (↑Int (Submodule.subtype I)).comp ↑e'
        f_apply : ∀ (x : S), Eq (f x) ↑((b'.equiv ab (Equiv.refl (Module.Free.ChooseBa …
        ha : ∀ (i : Module.Free.ChooseBasisIndex Int S), Eq (f (b' i)) (HSMul.hSMul (a …
        this : DecidableEq ι := Classical.decEq ι
        i j : Module.Free.ChooseBasisIndex Int S
        h : Eq i j
        ⊢ Eq ((Finsupp.single j (a j)) i) (Matrix.diagonal a i j)
      -/
    · rw [h, Matrix.diagonal_apply_eq, Finsupp.single_eq_same]
      /-
        🎉 no goals
      -/
      /-
        case neg
        S : Type u_1
        inst✝⁶ : CommRing S
        inst✝⁵ : Nontrivial S
        inst✝⁴ : IsDedekindDomain S
        inst✝³ : Module.Free Int S
        inst✝² : Module.Finite Int S
        I : Ideal S
        E : Type u_2
        inst✝¹ : EquivLike E S (Subtype fun x => Membership.mem I x)
        inst✝ : AddEquivClass E S (Subtype fun x => Membership.mem I x)
        e : E
        hI : Not (Eq I Bot.bot)
        ι : Type u_1 := Module.Free.ChooseBasisIndex Int S
        b : Basis (Module.Free.ChooseBasisIndex Int S) Int S := Module.Free.chooseBasi …
        h✝ : Nonempty ι
        this✝¹ : Fintype (HasQuotient.Quotient S I) := I.fintypeQuotientOfFreeOfNeBot hI
        this✝ : DecidableEq ι := Classical.decEq ι
        a : Module.Free.ChooseBasisIndex Int S → Int := Ideal.smithCoeffs b I hI
        b' : Basis (Module.Free.ChooseBasisIndex Int S) Int S := Ideal.ringBasis b I hI
        ab : Basis (Module.Free.ChooseBasisIndex Int S) Int (Subtype fun x => Membersh …
        ab_eq : ∀ (i : Module.Free.ChooseBasisIndex Int S), Eq (↑((Ideal.selfBasis b I …
        e' : LinearEquiv (RingHom.id Int) S (Subtype fun x => Membership.mem I x) := b …
        f : LinearMap (RingHom.id Int) S S := (↑Int (Submodule.subtype I)).comp ↑e'
        f_apply : ∀ (x : S), Eq (f x) ↑((b'.equiv ab (Equiv.refl (Module.Free.ChooseBa …
        ha : ∀ (i : Module.Free.ChooseBasisIndex Int S), Eq (f (b' i)) (HSMul.hSMul (a …
        this : DecidableEq ι := Classical.decEq ι
        i j : Module.Free.ChooseBasisIndex Int S
        h : Not (Eq i j)
        ⊢ Eq ((Finsupp.single j (a j)) i) (Matrix.diagonal a i j)
      -/
    · rw [Matrix.diagonal_apply_ne _ h, Finsupp.single_eq_of_ne (Ne.symm h)]
      /-
        🎉 no goals
      -/
  -- Now we map everything through the linear equiv `S ≃ₗ (ι → ℤ)`,
  -- which maps `(S ⧸ I)` to `Π i, ZMod (a i).nat_abs`.
  haveI : ∀ i, NeZero (a i).natAbs := fun i =>
    ⟨Int.natAbs_ne_zero.mpr (Ideal.smithCoeffs_ne_zero b I hI i)⟩
  simp_rw [Nat.card_congr (Ideal.quotientEquivPiZMod I b hI).toEquiv, Nat.card_pi,
    Nat.card_zmod]


/-- Let `b` be a basis for `S` over `ℤ` and `bI` a basis for `I` over `ℤ` of the same dimension.
Then an alternative way to compute the norm of `I` is given by taking the determinant of `bI`
over `b`. -/
theorem natAbs_det_basis_change {ι : Type*} [Fintype ι] [DecidableEq ι] (b : Basis ι ℤ S)
    (I : Ideal S) (bI : Basis ι ℤ I) : (b.det ((↑) ∘ bI)).natAbs = Ideal.absNorm I := by
  /-
    S : Type u_1
    inst✝⁶ : CommRing S
    inst✝⁵ : Nontrivial S
    inst✝⁴ : IsDedekindDomain S
    inst✝³ : Module.Free Int S
    inst✝² : Module.Finite Int S
    ι : Type u_2
    inst✝¹ : Fintype ι
    inst✝ : DecidableEq ι
    b : Basis ι Int S
    I : Ideal S
    bI : Basis ι Int (Subtype fun x => Membership.mem I x)
    ⊢ Eq (b.det (Function.comp Subtype.val ⇑bI)).natAbs (Ideal.absNorm I)
  -/
  let e := b.equiv bI (Equiv.refl _)
  calc
    (b.det ((Submodule.subtype I).restrictScalars ℤ ∘ bI)).natAbs =
        (LinearMap.det ((Submodule.subtype I).restrictScalars ℤ ∘ₗ (e : S →ₗ[ℤ] I))).natAbs := by
      rw [Basis.det_comp_basis]
    _ = _ := natAbs_det_equiv I e


@[simp]
theorem absNorm_span_singleton (r : S) :
    absNorm (span ({r} : Set S)) = (Algebra.norm ℤ r).natAbs := by
  /-
    S : Type u_1
    inst✝⁴ : CommRing S
    inst✝³ : Nontrivial S
    inst✝² : IsDedekindDomain S
    inst✝¹ : Module.Free Int S
    inst✝ : Module.Finite Int S
    r : S
    ⊢ Eq (Ideal.absNorm (Ideal.span (Singleton.singleton r))) ((Algebra.norm Int)  …
  -/
  rw [Algebra.norm_apply]
  /-
    S : Type u_1
    inst✝⁴ : CommRing S
    inst✝³ : Nontrivial S
    inst✝² : IsDedekindDomain S
    inst✝¹ : Module.Free Int S
    inst✝ : Module.Finite Int S
    r : S
    ⊢ Eq (Ideal.absNorm (Ideal.span (Singleton.singleton r))) (LinearMap.det ((Alg …
  -/
  by_cases hr : r = 0
  · simp only [hr, Ideal.span_zero, Algebra.coe_lmul_eq_mul, eq_self_iff_true, Ideal.absNorm_bot,
      LinearMap.det_zero'', Set.singleton_zero, _root_.map_zero, Int.natAbs_zero]
  /-
    case neg
    S : Type u_1
    inst✝⁴ : CommRing S
    inst✝³ : Nontrivial S
    inst✝² : IsDedekindDomain S
    inst✝¹ : Module.Free Int S
    inst✝ : Module.Finite Int S
    r : S
    hr : Not (Eq r 0)
    ⊢ Eq (Ideal.absNorm (Ideal.span (Singleton.singleton r))) (LinearMap.det ((Alg …
  -/
  letI := Ideal.fintypeQuotientOfFreeOfNeBot (span {r}) (mt span_singleton_eq_bot.mp hr)
  /-
    case neg
    S : Type u_1
    inst✝⁴ : CommRing S
    inst✝³ : Nontrivial S
    inst✝² : IsDedekindDomain S
    inst✝¹ : Module.Free Int S
    inst✝ : Module.Finite Int S
    r : S
    hr : Not (Eq r 0)
    this : Fintype (HasQuotient.Quotient S (Ideal.span (Singleton.singleton r))) : …
    ⊢ Eq (Ideal.absNorm (Ideal.span (Singleton.singleton r))) (LinearMap.det ((Alg …
  -/
  let b := Module.Free.chooseBasis ℤ S
  /-
    case neg
    S : Type u_1
    inst✝⁴ : CommRing S
    inst✝³ : Nontrivial S
    inst✝² : IsDedekindDomain S
    inst✝¹ : Module.Free Int S
    inst✝ : Module.Finite Int S
    r : S
    hr : Not (Eq r 0)
    this : Fintype (HasQuotient.Quotient S (Ideal.span (Singleton.singleton r))) : …
    b : Basis (Module.Free.ChooseBasisIndex Int S) Int S := Module.Free.chooseBasi …
    ⊢ Eq (Ideal.absNorm (Ideal.span (Singleton.singleton r))) (LinearMap.det ((Alg …
  -/
  rw [← natAbs_det_equiv _ (b.equiv (basisSpanSingleton b hr) (Equiv.refl _))]
  /-
    case neg
    S : Type u_1
    inst✝⁴ : CommRing S
    inst✝³ : Nontrivial S
    inst✝² : IsDedekindDomain S
    inst✝¹ : Module.Free Int S
    inst✝ : Module.Finite Int S
    r : S
    hr : Not (Eq r 0)
    this : Fintype (HasQuotient.Quotient S (Ideal.span (Singleton.singleton r))) : …
    b : Basis (Module.Free.ChooseBasisIndex Int S) Int S := Module.Free.chooseBasi …
    ⊢ Eq (LinearMap.det ((↑Int (Submodule.subtype (Ideal.span (Singleton.singleton …
  -/
  congr
  /-
    case neg.e_m.h.e_6.h
    S : Type u_1
    inst✝⁴ : CommRing S
    inst✝³ : Nontrivial S
    inst✝² : IsDedekindDomain S
    inst✝¹ : Module.Free Int S
    inst✝ : Module.Finite Int S
    r : S
    hr : Not (Eq r 0)
    this : Fintype (HasQuotient.Quotient S (Ideal.span (Singleton.singleton r))) : …
    b : Basis (Module.Free.ChooseBasisIndex Int S) Int S := Module.Free.chooseBasi …
    ⊢ Eq ((↑Int (Submodule.subtype (Ideal.span (Singleton.singleton r)))).comp (↑( …
  -/
  refine b.ext fun i => ?_
  /-
    case neg.e_m.h.e_6.h
    S : Type u_1
    inst✝⁴ : CommRing S
    inst✝³ : Nontrivial S
    inst✝² : IsDedekindDomain S
    inst✝¹ : Module.Free Int S
    inst✝ : Module.Finite Int S
    r : S
    hr : Not (Eq r 0)
    this : Fintype (HasQuotient.Quotient S (Ideal.span (Singleton.singleton r))) : …
    b : Basis (Module.Free.ChooseBasisIndex Int S) Int S := Module.Free.chooseBasi …
    i : Module.Free.ChooseBasisIndex Int S
    ⊢ Eq (((↑Int (Submodule.subtype (Ideal.span (Singleton.singleton r)))).comp (↑ …
  -/
  simp
  /-
    🎉 no goals
  -/


theorem absNorm_dvd_norm_of_mem {I : Ideal S} {x : S} (h : x ∈ I) :
    ↑(Ideal.absNorm I) ∣ Algebra.norm ℤ x := by
  /-
    S : Type u_1
    inst✝⁴ : CommRing S
    inst✝³ : Nontrivial S
    inst✝² : IsDedekindDomain S
    inst✝¹ : Module.Free Int S
    inst✝ : Module.Finite Int S
    I : Ideal S
    x : S
    h : Membership.mem I x
    ⊢ Dvd.dvd (↑(Ideal.absNorm I)) ((Algebra.norm Int) x)
  -/
  rw [← Int.dvd_natAbs, ← absNorm_span_singleton x, Int.natCast_dvd_natCast]
  /-
    S : Type u_1
    inst✝⁴ : CommRing S
    inst✝³ : Nontrivial S
    inst✝² : IsDedekindDomain S
    inst✝¹ : Module.Free Int S
    inst✝ : Module.Finite Int S
    I : Ideal S
    x : S
    h : Membership.mem I x
    ⊢ Dvd.dvd (Ideal.absNorm I) (Ideal.absNorm (Ideal.span (Singleton.singleton x)))
  -/
  exact absNorm_dvd_absNorm_of_le ((span_singleton_le_iff_mem _).mpr h)
  /-
    🎉 no goals
  -/


@[simp]
theorem absNorm_span_insert (r : S) (s : Set S) :
    absNorm (span (insert r s)) ∣ gcd (absNorm (span s)) (Algebra.norm ℤ r).natAbs :=
  (dvd_gcd_iff _ _ _).mpr
    ⟨absNorm_dvd_absNorm_of_le (span_mono (Set.subset_insert _ _)),
      _root_.trans
        (absNorm_dvd_absNorm_of_le (span_mono (Set.singleton_subset_iff.mpr (Set.mem_insert _ _))))
            /-
              S : Type u_1
              inst✝⁴ : CommRing S
              inst✝³ : Nontrivial S
              inst✝² : IsDedekindDomain S
              inst✝¹ : Module.Free Int S
              inst✝ : Module.Finite Int S
              r : S
              s : Set S
              ⊢ Dvd.dvd (Ideal.absNorm (Ideal.span (Singleton.singleton r))) ((Algebra.norm  …
            -/
        (by rw [absNorm_span_singleton])⟩
            /-
              🎉 no goals
            -/


theorem absNorm_eq_zero_iff {I : Ideal S} : Ideal.absNorm I = 0 ↔ I = ⊥ := by
  /-
    S : Type u_1
    inst✝⁴ : CommRing S
    inst✝³ : Nontrivial S
    inst✝² : IsDedekindDomain S
    inst✝¹ : Module.Free Int S
    inst✝ : Module.Finite Int S
    I : Ideal S
    ⊢ Iff (Eq (Ideal.absNorm I) 0) (Eq I Bot.bot)
  -/
  constructor
    /-
      case mp
      S : Type u_1
      inst✝⁴ : CommRing S
      inst✝³ : Nontrivial S
      inst✝² : IsDedekindDomain S
      inst✝¹ : Module.Free Int S
      inst✝ : Module.Finite Int S
      I : Ideal S
      ⊢ Eq (Ideal.absNorm I) 0 → Eq I Bot.bot
    -/
  · intro hI
    /-
      case mp
      S : Type u_1
      inst✝⁴ : CommRing S
      inst✝³ : Nontrivial S
      inst✝² : IsDedekindDomain S
      inst✝¹ : Module.Free Int S
      inst✝ : Module.Finite Int S
      I : Ideal S
      hI : Eq (Ideal.absNorm I) 0
      ⊢ Eq I Bot.bot
    -/
    rw [← le_bot_iff]
    /-
      case mp
      S : Type u_1
      inst✝⁴ : CommRing S
      inst✝³ : Nontrivial S
      inst✝² : IsDedekindDomain S
      inst✝¹ : Module.Free Int S
      inst✝ : Module.Finite Int S
      I : Ideal S
      hI : Eq (Ideal.absNorm I) 0
      ⊢ LE.le I Bot.bot
    -/
    intros x hx
    rw [mem_bot, ← Algebra.norm_eq_zero_iff (R := ℤ), ← Int.natAbs_eq_zero,
      ← Ideal.absNorm_span_singleton, ← zero_dvd_iff, ← hI]
    /-
      case mp
      S : Type u_1
      inst✝⁴ : CommRing S
      inst✝³ : Nontrivial S
      inst✝² : IsDedekindDomain S
      inst✝¹ : Module.Free Int S
      inst✝ : Module.Finite Int S
      I : Ideal S
      hI : Eq (Ideal.absNorm I) 0
      x : S
      hx : Membership.mem I x
      ⊢ Dvd.dvd (Ideal.absNorm I) (Ideal.absNorm (Ideal.span (Singleton.singleton x)))
    -/
    apply Ideal.absNorm_dvd_absNorm_of_le
    /-
      case mp.h
      S : Type u_1
      inst✝⁴ : CommRing S
      inst✝³ : Nontrivial S
      inst✝² : IsDedekindDomain S
      inst✝¹ : Module.Free Int S
      inst✝ : Module.Finite Int S
      I : Ideal S
      hI : Eq (Ideal.absNorm I) 0
      x : S
      hx : Membership.mem I x
      ⊢ LE.le (Ideal.span (Singleton.singleton x)) I
    -/
    rwa [Ideal.span_singleton_le_iff_mem]
    /-
      🎉 no goals
    -/
    /-
      case mpr
      S : Type u_1
      inst✝⁴ : CommRing S
      inst✝³ : Nontrivial S
      inst✝² : IsDedekindDomain S
      inst✝¹ : Module.Free Int S
      inst✝ : Module.Finite Int S
      I : Ideal S
      ⊢ Eq I Bot.bot → Eq (Ideal.absNorm I) 0
    -/
  · rintro rfl
    /-
      case mpr
      S : Type u_1
      inst✝⁴ : CommRing S
      inst✝³ : Nontrivial S
      inst✝² : IsDedekindDomain S
      inst✝¹ : Module.Free Int S
      inst✝ : Module.Finite Int S
      ⊢ Eq (Ideal.absNorm Bot.bot) 0
    -/
    exact absNorm_bot
    /-
      🎉 no goals
    -/


theorem absNorm_ne_zero_iff_mem_nonZeroDivisors {I : Ideal S} :
    absNorm I ≠ 0 ↔ I ∈ (Ideal S)⁰ := by
  /-
    S : Type u_1
    inst✝⁴ : CommRing S
    inst✝³ : Nontrivial S
    inst✝² : IsDedekindDomain S
    inst✝¹ : Module.Free Int S
    inst✝ : Module.Finite Int S
    I : Ideal S
    ⊢ Iff (Ne (Ideal.absNorm I) 0) (Membership.mem (nonZeroDivisors (Ideal S)) I)
  -/
  simp_rw [ne_eq, Ideal.absNorm_eq_zero_iff, mem_nonZeroDivisors_iff_ne_zero, Submodule.zero_eq_bot]
  /-
    🎉 no goals
  -/


theorem absNorm_pos_iff_mem_nonZeroDivisors {I : Ideal S} :
    0 < absNorm I ↔ I ∈ (Ideal S)⁰ := by
  /-
    S : Type u_1
    inst✝⁴ : CommRing S
    inst✝³ : Nontrivial S
    inst✝² : IsDedekindDomain S
    inst✝¹ : Module.Free Int S
    inst✝ : Module.Finite Int S
    I : Ideal S
    ⊢ Iff (LT.lt 0 (Ideal.absNorm I)) (Membership.mem (nonZeroDivisors (Ideal S)) I)
  -/
  rw [← absNorm_ne_zero_iff_mem_nonZeroDivisors, Nat.pos_iff_ne_zero]
  /-
    🎉 no goals
  -/


theorem absNorm_ne_zero_of_nonZeroDivisors (I : (Ideal S)⁰) : absNorm (I : Ideal S) ≠ 0 :=
  absNorm_ne_zero_iff_mem_nonZeroDivisors.mpr (SetLike.coe_mem I)


theorem absNorm_pos_of_nonZeroDivisors (I : (Ideal S)⁰) : 0 < absNorm (I : Ideal S) :=
  absNorm_pos_iff_mem_nonZeroDivisors.mpr (SetLike.coe_mem I)


theorem finite_setOf_absNorm_eq [CharZero S] (n : ℕ) :
    {I : Ideal S | Ideal.absNorm I = n}.Finite := by
  /-
    S : Type u_1
    inst✝⁵ : CommRing S
    inst✝⁴ : Nontrivial S
    inst✝³ : IsDedekindDomain S
    inst✝² : Module.Free Int S
    inst✝¹ : Module.Finite Int S
    inst✝ : CharZero S
    n : Nat
    ⊢ (setOf fun I => Eq (Ideal.absNorm I) n).Finite
  -/
  obtain hn | hn := Nat.eq_zero_or_pos n
    /-
      case inl
      S : Type u_1
      inst✝⁵ : CommRing S
      inst✝⁴ : Nontrivial S
      inst✝³ : IsDedekindDomain S
      inst✝² : Module.Free Int S
      inst✝¹ : Module.Finite Int S
      inst✝ : CharZero S
      n : Nat
      hn : Eq n 0
      ⊢ (setOf fun I => Eq (Ideal.absNorm I) n).Finite
    -/
  · simp only [hn, absNorm_eq_zero_iff, Set.setOf_eq_eq_singleton, Set.finite_singleton]
    /-
      🎉 no goals
    -/
    /-
      case inr
      S : Type u_1
      inst✝⁵ : CommRing S
      inst✝⁴ : Nontrivial S
      inst✝³ : IsDedekindDomain S
      inst✝² : Module.Free Int S
      inst✝¹ : Module.Finite Int S
      inst✝ : CharZero S
      n : Nat
      hn : GT.gt n 0
      ⊢ (setOf fun I => Eq (Ideal.absNorm I) n).Finite
    -/
  · let f := fun I : Ideal S => Ideal.map (Ideal.Quotient.mk (@Ideal.span S _ {↑n})) I
    /-
      case inr
      S : Type u_1
      inst✝⁵ : CommRing S
      inst✝⁴ : Nontrivial S
      inst✝³ : IsDedekindDomain S
      inst✝² : Module.Free Int S
      inst✝¹ : Module.Finite Int S
      inst✝ : CharZero S
      n : Nat
      hn : GT.gt n 0
      f : Ideal S → Ideal (HasQuotient.Quotient S (Ideal.span (Singleton.singleton ↑ …
      ⊢ (setOf fun I => Eq (Ideal.absNorm I) n).Finite
    -/
    refine Set.Finite.of_finite_image (f := f) ?_ ?_
    · suffices Finite (S ⧸ @Ideal.span S _ {↑n}) by
        let g := ((↑) : Ideal (S ⧸ @Ideal.span S _ {↑n}) → Set (S ⧸ @Ideal.span S _ {↑n}))
        refine Set.Finite.of_finite_image (f := g) ?_ SetLike.coe_injective.injOn
        exact Set.Finite.subset Set.finite_univ (Set.subset_univ _)
      /-
        case inr.refine_1
        S : Type u_1
        inst✝⁵ : CommRing S
        inst✝⁴ : Nontrivial S
        inst✝³ : IsDedekindDomain S
        inst✝² : Module.Free Int S
        inst✝¹ : Module.Finite Int S
        inst✝ : CharZero S
        n : Nat
        hn : GT.gt n 0
        f : Ideal S → Ideal (HasQuotient.Quotient S (Ideal.span (Singleton.singleton ↑ …
        ⊢ Finite (HasQuotient.Quotient S (Ideal.span (Singleton.singleton ↑n)))
      -/
      rw [← absNorm_ne_zero_iff, absNorm_span_singleton]
      simpa only [Ne, Int.natAbs_eq_zero, Algebra.norm_eq_zero_iff, Nat.cast_eq_zero] using
        ne_of_gt hn
      /-
        case inr.refine_2
        S : Type u_1
        inst✝⁵ : CommRing S
        inst✝⁴ : Nontrivial S
        inst✝³ : IsDedekindDomain S
        inst✝² : Module.Free Int S
        inst✝¹ : Module.Finite Int S
        inst✝ : CharZero S
        n : Nat
        hn : GT.gt n 0
        f : Ideal S → Ideal (HasQuotient.Quotient S (Ideal.span (Singleton.singleton ↑ …
        ⊢ Set.InjOn f (setOf fun I => Eq (Ideal.absNorm I) n)
      -/
    · intro I hI J hJ h
      rw [← comap_map_mk (span_singleton_absNorm_le I), ← hI.symm, ←
        comap_map_mk (span_singleton_absNorm_le J), ← hJ.symm]
      /-
        case inr.refine_2
        S : Type u_1
        inst✝⁵ : CommRing S
        inst✝⁴ : Nontrivial S
        inst✝³ : IsDedekindDomain S
        inst✝² : Module.Free Int S
        inst✝¹ : Module.Finite Int S
        inst✝ : CharZero S
        n : Nat
        hn : GT.gt n 0
        f : Ideal S → Ideal (HasQuotient.Quotient S (Ideal.span (Singleton.singleton ↑ …
        I : Ideal S
        hI : Membership.mem (setOf fun I => Eq (Ideal.absNorm I) n) I
        J : Ideal S
        hJ : Membership.mem (setOf fun I => Eq (Ideal.absNorm I) n) J
        h : Eq (f I) (f J)
        ⊢ Eq (Ideal.comap (Ideal.Quotient.mk (Ideal.span (Singleton.singleton ↑n))) (I …
      -/
      congr
      /-
        🎉 no goals
      -/


theorem finite_setOf_absNorm_le [CharZero S] (n : ℕ) :
    {I : Ideal S | Ideal.absNorm I ≤ n}.Finite := by
  rw [show {I : Ideal S | Ideal.absNorm I ≤ n} =
    (⋃ i ∈ Set.Icc 0 n, {I : Ideal S | Ideal.absNorm I = i}) by ext; simp]
  /-
    S : Type u_1
    inst✝⁵ : CommRing S
    inst✝⁴ : Nontrivial S
    inst✝³ : IsDedekindDomain S
    inst✝² : Module.Free Int S
    inst✝¹ : Module.Finite Int S
    inst✝ : CharZero S
    n : Nat
    ⊢ (Set.iUnion fun i => Set.iUnion fun h => setOf fun I => Eq (Ideal.absNorm I) …
  -/
  refine Set.Finite.biUnion (Set.finite_Icc 0 n) (fun i _ => Ideal.finite_setOf_absNorm_eq i)
  /-
    🎉 no goals
  -/


theorem card_norm_le_eq_card_norm_le_add_one (n : ℕ) [CharZero S] :
    Nat.card {I : Ideal S // absNorm I ≤ n} =
      Nat.card {I : (Ideal S)⁰ // absNorm (I : Ideal S) ≤ n} + 1 := by
  classical
  have : Finite {I : Ideal S // I ∈ (Ideal S)⁰ ∧ absNorm I ≤ n} :=
    (finite_setOf_absNorm_le n).subset fun _ ⟨_, h⟩ ↦ h
  have : Finite {I : Ideal S // I ∉ (Ideal S)⁰ ∧ absNorm I ≤ n} :=
    (finite_setOf_absNorm_le n).subset fun _ ⟨_, h⟩ ↦ h
  rw [Nat.card_congr (Equiv.subtypeSubtypeEquivSubtypeInter (fun I ↦ I ∈ (Ideal S)⁰)
    (fun I ↦ absNorm I ≤ n))]
  let e : {I : Ideal S // absNorm I ≤ n} ≃ {I : Ideal S // I ∈ (Ideal S)⁰ ∧ absNorm I ≤ n} ⊕
      {I : Ideal S // I ∉ (Ideal S)⁰ ∧ absNorm I ≤ n} := by
    refine (Equiv.subtypeEquivRight ?_).trans (subtypeOrEquiv _ _ ?_)
    · intro _
      simp_rw [← or_and_right, em, true_and]
    · exact Pi.disjoint_iff.mpr fun I ↦ Prop.disjoint_iff.mpr (by tauto)
  simp_rw [Nat.card_congr e, Nat.card_sum, add_right_inj]
  conv_lhs =>
    enter [1, 1, I]
    rw [← absNorm_ne_zero_iff_mem_nonZeroDivisors, ne_eq, not_not, and_iff_left_iff_imp.mpr
      (fun h ↦ by rw [h]; exact Nat.zero_le n), absNorm_eq_zero_iff]
  rw [Nat.card_unique]


theorem norm_dvd_iff {x : S} (hx : Prime (Algebra.norm ℤ x)) {y : ℤ} :
    Algebra.norm ℤ x ∣ y ↔ x ∣ y := by
  rw [← Ideal.mem_span_singleton (y := x), ← eq_intCast (algebraMap ℤ S), ← Ideal.mem_comap,
    ← Ideal.span_singleton_absNorm, Ideal.mem_span_singleton, Ideal.absNorm_span_singleton,
    Int.natAbs_dvd]
  /-
    S : Type u_1
    inst✝⁴ : CommRing S
    inst✝³ : Nontrivial S
    inst✝² : IsDedekindDomain S
    inst✝¹ : Module.Free Int S
    inst✝ : Module.Finite Int S
    x : S
    hx : Prime ((Algebra.norm Int) x)
    y : Int
    ⊢ Nat.Prime (Ideal.absNorm (Ideal.span (Singleton.singleton x)))
  -/
  rwa [Ideal.absNorm_span_singleton, ← Int.prime_iff_natAbs_prime]
  /-
    🎉 no goals
  -/


