/-- The ramification index of `P` over `p` is the largest exponent `n` such that
`p` is contained in `P^n`.

In particular, if `p` is not contained in `P^n`, then the ramification index is 0.

If there is no largest such `n` (e.g. because `p = ⊥`), then `ramificationIdx` is
defined to be 0.
-/
noncomputable def ramificationIdx : ℕ := sSup {n | map f p ≤ P ^ n}


theorem ramificationIdx_eq_find [DecidablePred fun n ↦ ∀ (k : ℕ), map f p ≤ P ^ k → k ≤ n]
    (h : ∃ n, ∀ k, map f p ≤ P ^ k → k ≤ n) :
    ramificationIdx f p P = Nat.find h := by
  /-
    R : Type u
    inst✝² : CommRing R
    S : Type v
    inst✝¹ : CommRing S
    f : RingHom R S
    p : Ideal R
    P : Ideal S
    inst✝ : DecidablePred fun n => ∀ (k : Nat), LE.le (Ideal.map f p) (HPow.hPow P …
    h : Exists fun n => ∀ (k : Nat), LE.le (Ideal.map f p) (HPow.hPow P k) → LE.le …
    ⊢ Eq (Ideal.ramificationIdx f p P) (Nat.find h)
  -/
  convert Nat.sSup_def h
  /-
    🎉 no goals
  -/


theorem ramificationIdx_eq_zero (h : ∀ n : ℕ, ∃ k, map f p ≤ P ^ k ∧ n < k) :
    ramificationIdx f p P = 0 :=
              /-
                R : Type u
                inst✝¹ : CommRing R
                S : Type v
                inst✝ : CommRing S
                f : RingHom R S
                p : Ideal R
                P : Ideal S
                h : ∀ (n : Nat), Exists fun k => And (LE.le (Ideal.map f p) (HPow.hPow P k)) ( …
                ⊢ Not (Exists fun n => ∀ (a : Nat), Membership.mem (setOf fun n => LE.le (Idea …
              -/
  dif_neg (by push_neg; exact h)
                        /-
                          🎉 no goals
                        -/


theorem ramificationIdx_spec {n : ℕ} (hle : map f p ≤ P ^ n) (hgt : ¬map f p ≤ P ^ (n + 1)) :
    ramificationIdx f p P = n := by
  classical
  let Q : ℕ → Prop := fun m => ∀ k : ℕ, map f p ≤ P ^ k → k ≤ m
  have : Q n := by
    intro k hk
    refine le_of_not_lt fun hnk => ?_
    exact hgt (hk.trans (Ideal.pow_le_pow_right hnk))
  rw [ramificationIdx_eq_find ⟨n, this⟩]
  refine le_antisymm (Nat.find_min' _ this) (le_of_not_gt fun h : Nat.find _ < n => ?_)
  obtain this' := Nat.find_spec ⟨n, this⟩
  exact h.not_le (this' _ hle)


theorem ramificationIdx_lt {n : ℕ} (hgt : ¬map f p ≤ P ^ n) : ramificationIdx f p P < n := by
  classical
  cases' n with n n
  · simp at hgt
  · rw [Nat.lt_succ_iff]
    have : ∀ k, map f p ≤ P ^ k → k ≤ n := by
      refine fun k hk => le_of_not_lt fun hnk => ?_
      exact hgt (hk.trans (Ideal.pow_le_pow_right hnk))
    rw [ramificationIdx_eq_find ⟨n, this⟩]
    exact Nat.find_min' ⟨n, this⟩ this


@[simp]
theorem ramificationIdx_bot : ramificationIdx f ⊥ P = 0 :=
                                                                        /-
                                                                          R : Type u
                                                                          inst✝¹ : CommRing R
                                                                          S : Type v
                                                                          inst✝ : CommRing S
                                                                          f : RingHom R S
                                                                          P : Ideal S
                                                                          n : Nat
                                                                          hn : ∀ (a : Nat), Membership.mem (setOf fun n => LE.le (Ideal.map f Bot.bot) ( …
                                                                          ⊢ Membership.mem (setOf fun n => LE.le (Ideal.map f Bot.bot) (HPow.hPow P n))  …
                                                                        -/
  dif_neg <| not_exists.mpr fun n hn => n.lt_succ_self.not_le (hn _ (by simp))
                                                                        /-
                                                                          🎉 no goals
                                                                        -/


@[simp]
theorem ramificationIdx_of_not_le (h : ¬map f p ≤ P) : ramificationIdx f p P = 0 :=
                           /-
                             R : Type u
                             inst✝¹ : CommRing R
                             S : Type v
                             inst✝ : CommRing S
                             f : RingHom R S
                             p : Ideal R
                             P : Ideal S
                             h : Not (LE.le (Ideal.map f p) P)
                             ⊢ LE.le (Ideal.map f p) (HPow.hPow P 0)
                           -/
                           /-
                             🎉 no goals
                           -/
  ramificationIdx_spec (by simp) (by simpa using h)
                                     /-
                                       🎉 no goals
                                     -/


theorem ramificationIdx_ne_zero {e : ℕ} (he : e ≠ 0) (hle : map f p ≤ P ^ e)
    (hnle : ¬map f p ≤ P ^ (e + 1)) : ramificationIdx f p P ≠ 0 := by
  /-
    R : Type u
    inst✝¹ : CommRing R
    S : Type v
    inst✝ : CommRing S
    f : RingHom R S
    p : Ideal R
    P : Ideal S
    e : Nat
    he : Ne e 0
    hle : LE.le (Ideal.map f p) (HPow.hPow P e)
    hnle : Not (LE.le (Ideal.map f p) (HPow.hPow P (HAdd.hAdd e 1)))
    ⊢ Ne (Ideal.ramificationIdx f p P) 0
  -/
  rwa [ramificationIdx_spec hle hnle]
  /-
    🎉 no goals
  -/


theorem le_pow_of_le_ramificationIdx {n : ℕ} (hn : n ≤ ramificationIdx f p P) :
    map f p ≤ P ^ n := by
  /-
    R : Type u
    inst✝¹ : CommRing R
    S : Type v
    inst✝ : CommRing S
    f : RingHom R S
    p : Ideal R
    P : Ideal S
    n : Nat
    hn : LE.le n (Ideal.ramificationIdx f p P)
    ⊢ LE.le (Ideal.map f p) (HPow.hPow P n)
  -/
  contrapose! hn
  /-
    R : Type u
    inst✝¹ : CommRing R
    S : Type v
    inst✝ : CommRing S
    f : RingHom R S
    p : Ideal R
    P : Ideal S
    n : Nat
    hn : Not (LE.le (Ideal.map f p) (HPow.hPow P n))
    ⊢ LT.lt (Ideal.ramificationIdx f p P) n
  -/
  exact ramificationIdx_lt hn
  /-
    🎉 no goals
  -/


theorem le_pow_ramificationIdx : map f p ≤ P ^ ramificationIdx f p P :=
  le_pow_of_le_ramificationIdx (le_refl _)


theorem le_comap_pow_ramificationIdx : p ≤ comap f (P ^ ramificationIdx f p P) :=
  map_le_iff_le_comap.mp le_pow_ramificationIdx


theorem le_comap_of_ramificationIdx_ne_zero (h : ramificationIdx f p P ≠ 0) : p ≤ comap f P :=
  Ideal.map_le_iff_le_comap.mp <| le_pow_ramificationIdx.trans <| Ideal.pow_le_self <| h


variable (p) in
lemma ramificationIdx_comap_eq [Algebra R S] (e : S ≃ₐ[R] S₁) (P : Ideal S₁) :
    ramificationIdx (algebraMap R S) p (P.comap e) = ramificationIdx (algebraMap R S₁) p P := by
  /-
    R : Type u
    inst✝⁴ : CommRing R
    S : Type v
    inst✝³ : CommRing S
    p : Ideal R
    S₁ : Type u_1
    inst✝² : CommRing S₁
    inst✝¹ : Algebra R S₁
    inst✝ : Algebra R S
    e : AlgEquiv R S S₁
    P : Ideal S₁
    ⊢ Eq (Ideal.ramificationIdx (algebraMap R S) p (Ideal.comap e P)) (Ideal.ramif …
  -/
  dsimp only [ramificationIdx]
  /-
    R : Type u
    inst✝⁴ : CommRing R
    S : Type v
    inst✝³ : CommRing S
    p : Ideal R
    S₁ : Type u_1
    inst✝² : CommRing S₁
    inst✝¹ : Algebra R S₁
    inst✝ : Algebra R S
    e : AlgEquiv R S S₁
    P : Ideal S₁
    ⊢ Eq (SupSet.sSup (setOf fun n => LE.le (Ideal.map (algebraMap R S) p) (HPow.h …
  -/
  congr
  /-
    case e_a
    R : Type u
    inst✝⁴ : CommRing R
    S : Type v
    inst✝³ : CommRing S
    p : Ideal R
    S₁ : Type u_1
    inst✝² : CommRing S₁
    inst✝¹ : Algebra R S₁
    inst✝ : Algebra R S
    e : AlgEquiv R S S₁
    P : Ideal S₁
    ⊢ Eq (setOf fun n => LE.le (Ideal.map (algebraMap R S) p) (HPow.hPow (Ideal.co …
  -/
  ext n
  /-
    case e_a.h
    R : Type u
    inst✝⁴ : CommRing R
    S : Type v
    inst✝³ : CommRing S
    p : Ideal R
    S₁ : Type u_1
    inst✝² : CommRing S₁
    inst✝¹ : Algebra R S₁
    inst✝ : Algebra R S
    e : AlgEquiv R S S₁
    P : Ideal S₁
    n : Nat
    ⊢ Iff (Membership.mem (setOf fun n => LE.le (Ideal.map (algebraMap R S) p) (HP …
  -/
  simp only [Set.mem_setOf_eq, Ideal.map_le_iff_le_comap]
  rw [← comap_coe e, ← e.toRingEquiv_toRingHom, comap_coe, ← RingEquiv.symm_symm (e : S ≃+* S₁),
    ← map_comap_of_equiv, ← Ideal.map_pow, map_comap_of_equiv, ← comap_coe (RingEquiv.symm _),
    comap_comap, RingEquiv.symm_symm, e.toRingEquiv_toRingHom, ← e.toAlgHom_toRingHom,
    AlgHom.comp_algebraMap]


variable (p) in
lemma ramificationIdx_map_eq [Algebra R S] {E : Type*} [EquivLike E S S₁] [AlgEquivClass E R S S₁]
    (P : Ideal S) (e : E) :
    ramificationIdx (algebraMap R S₁) p (P.map e) = ramificationIdx (algebraMap R S) p P := by
  /-
    R : Type u
    inst✝⁶ : CommRing R
    S : Type v
    inst✝⁵ : CommRing S
    p : Ideal R
    S₁ : Type u_1
    inst✝⁴ : CommRing S₁
    inst✝³ : Algebra R S₁
    inst✝² : Algebra R S
    E : Type u_2
    inst✝¹ : EquivLike E S S₁
    inst✝ : AlgEquivClass E R S S₁
    P : Ideal S
    e : E
    ⊢ Eq (Ideal.ramificationIdx (algebraMap R S₁) p (Ideal.map e P)) (Ideal.ramifi …
  -/
  rw [show P.map e = _ from P.map_comap_of_equiv (e : S ≃+* S₁)]
  /-
    R : Type u
    inst✝⁶ : CommRing R
    S : Type v
    inst✝⁵ : CommRing S
    p : Ideal R
    S₁ : Type u_1
    inst✝⁴ : CommRing S₁
    inst✝³ : Algebra R S₁
    inst✝² : Algebra R S
    E : Type u_2
    inst✝¹ : EquivLike E S S₁
    inst✝ : AlgEquivClass E R S S₁
    P : Ideal S
    e : E
    ⊢ Eq (Ideal.ramificationIdx (algebraMap R S₁) p (Ideal.comap (↑e).symm P)) (Id …
  -/
  exact p.ramificationIdx_comap_eq (e : S ≃ₐ[R] S₁).symm P
  /-
    🎉 no goals
  -/


theorem ramificationIdx_eq_normalizedFactors_count [DecidableEq (Ideal S)]
    (hp0 : map f p ≠ ⊥) (hP : P.IsPrime)
    (hP0 : P ≠ ⊥) : ramificationIdx f p P = (normalizedFactors (map f p)).count P := by
  /-
    R : Type u
    inst✝³ : CommRing R
    S : Type v
    inst✝² : CommRing S
    f : RingHom R S
    p : Ideal R
    P : Ideal S
    inst✝¹ : IsDedekindDomain S
    inst✝ : DecidableEq (Ideal S)
    hp0 : Ne (Ideal.map f p) Bot.bot
    hP : P.IsPrime
    hP0 : Ne P Bot.bot
    ⊢ Eq (Ideal.ramificationIdx f p P) (Multiset.count P (UniqueFactorizationMonoi …
  -/
  have hPirr := (Ideal.prime_of_isPrime hP0 hP).irreducible
  /-
    R : Type u
    inst✝³ : CommRing R
    S : Type v
    inst✝² : CommRing S
    f : RingHom R S
    p : Ideal R
    P : Ideal S
    inst✝¹ : IsDedekindDomain S
    inst✝ : DecidableEq (Ideal S)
    hp0 : Ne (Ideal.map f p) Bot.bot
    hP : P.IsPrime
    hP0 : Ne P Bot.bot
    hPirr : Irreducible P
    ⊢ Eq (Ideal.ramificationIdx f p P) (Multiset.count P (UniqueFactorizationMonoi …
  -/
  refine ramificationIdx_spec (Ideal.le_of_dvd ?_) (mt Ideal.dvd_iff_le.mpr ?_) <;>
    rw [dvd_iff_normalizedFactors_le_normalizedFactors (pow_ne_zero _ hP0) hp0,
      normalizedFactors_pow, normalizedFactors_irreducible hPirr, normalize_eq,
      Multiset.nsmul_singleton, ← Multiset.le_count_iff_replicate_le]
  /-
    case refine_2
    R : Type u
    inst✝³ : CommRing R
    S : Type v
    inst✝² : CommRing S
    f : RingHom R S
    p : Ideal R
    P : Ideal S
    inst✝¹ : IsDedekindDomain S
    inst✝ : DecidableEq (Ideal S)
    hp0 : Ne (Ideal.map f p) Bot.bot
    hP : P.IsPrime
    hP0 : Ne P Bot.bot
    hPirr : Irreducible P
    ⊢ Not (LE.le (HAdd.hAdd (Multiset.count P (UniqueFactorizationMonoid.normalize …
  -/
  exact (Nat.lt_succ_self _).not_le
  /-
    🎉 no goals
  -/


theorem ramificationIdx_eq_factors_count [DecidableEq (Ideal S)]
    (hp0 : map f p ≠ ⊥) (hP : P.IsPrime) (hP0 : P ≠ ⊥) :
    ramificationIdx f p P = (factors (map f p)).count P := by
  rw [IsDedekindDomain.ramificationIdx_eq_normalizedFactors_count hp0 hP hP0,
    factors_eq_normalizedFactors]


theorem ramificationIdx_ne_zero (hp0 : map f p ≠ ⊥) (hP : P.IsPrime) (le : map f p ≤ P) :
    ramificationIdx f p P ≠ 0 := by
  classical
  have hP0 : P ≠ ⊥ := by
    rintro rfl
    exact hp0 (le_bot_iff.mp le)
  have hPirr := (Ideal.prime_of_isPrime hP0 hP).irreducible
  rw [IsDedekindDomain.ramificationIdx_eq_normalizedFactors_count hp0 hP hP0]
  obtain ⟨P', hP', P'_eq⟩ :=
    exists_mem_normalizedFactors_of_dvd hp0 hPirr (Ideal.dvd_iff_le.mpr le)
  rwa [Multiset.count_ne_zero, associated_iff_eq.mp P'_eq]


local notation "f" => algebraMap R S


open Classical in
/-- The inertia degree of `P : Ideal S` lying over `p : Ideal R` is the degree of the
extension `(S / P) : (R / p)`.

We do not assume `P` lies over `p` in the definition; we return `0` instead.

See `inertiaDeg_algebraMap` for the common case where `f = algebraMap R S`
and there is an algebra structure `R / p → S / P`.
-/
noncomputable def inertiaDeg : ℕ :=
  if hPp : comap f P = p then
    letI : Algebra (R ⧸ p) (S ⧸ P) := Quotient.algebraQuotientOfLEComap hPp.ge
    finrank (R ⧸ p) (S ⧸ P)
  else 0

-- Useful for the `nontriviality` tactic using `comap_eq_of_scalar_tower_quotient`.

@[simp]
theorem inertiaDeg_of_subsingleton [hp : p.IsMaximal] [hQ : Subsingleton (S ⧸ P)] :
    inertiaDeg p P = 0 := by
  /-
    R : Type u
    inst✝² : CommRing R
    S : Type v
    inst✝¹ : CommRing S
    p : Ideal R
    P : Ideal S
    inst✝ : Algebra R S
    hp : p.IsMaximal
    hQ : Subsingleton (HasQuotient.Quotient S P)
    ⊢ Eq (p.inertiaDeg P) 0
  -/
  have := Ideal.Quotient.subsingleton_iff.mp hQ
  /-
    R : Type u
    inst✝² : CommRing R
    S : Type v
    inst✝¹ : CommRing S
    p : Ideal R
    P : Ideal S
    inst✝ : Algebra R S
    hp : p.IsMaximal
    hQ : Subsingleton (HasQuotient.Quotient S P)
    this : Eq P Top.top
    ⊢ Eq (p.inertiaDeg P) 0
  -/
  subst this
  /-
    R : Type u
    inst✝² : CommRing R
    S : Type v
    inst✝¹ : CommRing S
    p : Ideal R
    inst✝ : Algebra R S
    hp : p.IsMaximal
    hQ : Subsingleton (HasQuotient.Quotient S Top.top)
    ⊢ Eq (p.inertiaDeg Top.top) 0
  -/
  exact dif_neg fun h => hp.ne_top <| h.symm.trans comap_top
  /-
    🎉 no goals
  -/


@[simp]
theorem inertiaDeg_algebraMap [P.LiesOver p] [p.IsMaximal] :
    inertiaDeg p P = finrank (R ⧸ p) (S ⧸ P) := by
  /-
    R : Type u
    inst✝⁴ : CommRing R
    S : Type v
    inst✝³ : CommRing S
    p : Ideal R
    P : Ideal S
    inst✝² : Algebra R S
    inst✝¹ : P.LiesOver p
    inst✝ : p.IsMaximal
    ⊢ Eq (p.inertiaDeg P) (Module.finrank (HasQuotient.Quotient R p) (HasQuotient. …
  -/
  nontriviality S ⧸ P using inertiaDeg_of_subsingleton, finrank_zero_of_subsingleton
  /-
    R : Type u
    inst✝⁴ : CommRing R
    S : Type v
    inst✝³ : CommRing S
    p : Ideal R
    P : Ideal S
    inst✝² : Algebra R S
    inst✝¹ : P.LiesOver p
    inst✝ : p.IsMaximal
    a✝ : Nontrivial (HasQuotient.Quotient S P)
    ⊢ Eq (p.inertiaDeg P) (Module.finrank (HasQuotient.Quotient R p) (HasQuotient. …
  -/
  rw [inertiaDeg, dif_pos (over_def P p).symm]
  /-
    🎉 no goals
  -/


theorem inertiaDeg_pos [p.IsMaximal] [Module.Finite R S]
    [P.LiesOver p] : 0 < inertiaDeg p P :=
  haveI : Nontrivial (S ⧸ P) := Quotient.nontrivial_of_liesOver_of_isPrime P p
  finrank_pos.trans_eq (inertiaDeg_algebraMap p P).symm


lemma inertiaDeg_comap_eq (e : S ≃ₐ[R] S₁) (P : Ideal S₁) [p.IsMaximal] :
    inertiaDeg p (P.comap e) = inertiaDeg p P := by
  have he : (P.comap e).comap (algebraMap R S) = p ↔ P.comap (algebraMap R S₁) = p := by
    rw [← comap_coe e, comap_comap, ← e.toAlgHom_toRingHom, AlgHom.comp_algebraMap]
  /-
    R : Type u
    inst✝⁵ : CommRing R
    S : Type v
    inst✝⁴ : CommRing S
    p : Ideal R
    S₁ : Type u_1
    inst✝³ : CommRing S₁
    inst✝² : Algebra R S₁
    inst✝¹ : Algebra R S
    e : AlgEquiv R S S₁
    P : Ideal S₁
    inst✝ : p.IsMaximal
    he : Iff (Eq (Ideal.comap (algebraMap R S) (Ideal.comap e P)) p) (Eq (Ideal.co …
    ⊢ Eq (p.inertiaDeg (Ideal.comap e P)) (p.inertiaDeg P)
  -/
  by_cases h : P.LiesOver p
    /-
      case pos
      R : Type u
      inst✝⁵ : CommRing R
      S : Type v
      inst✝⁴ : CommRing S
      p : Ideal R
      S₁ : Type u_1
      inst✝³ : CommRing S₁
      inst✝² : Algebra R S₁
      inst✝¹ : Algebra R S
      e : AlgEquiv R S S₁
      P : Ideal S₁
      inst✝ : p.IsMaximal
      he : Iff (Eq (Ideal.comap (algebraMap R S) (Ideal.comap e P)) p) (Eq (Ideal.co …
      h : P.LiesOver p
      ⊢ Eq (p.inertiaDeg (Ideal.comap e P)) (p.inertiaDeg P)
    -/
  · rw [inertiaDeg_algebraMap, inertiaDeg_algebraMap]
    /-
      case pos
      R : Type u
      inst✝⁵ : CommRing R
      S : Type v
      inst✝⁴ : CommRing S
      p : Ideal R
      S₁ : Type u_1
      inst✝³ : CommRing S₁
      inst✝² : Algebra R S₁
      inst✝¹ : Algebra R S
      e : AlgEquiv R S S₁
      P : Ideal S₁
      inst✝ : p.IsMaximal
      he : Iff (Eq (Ideal.comap (algebraMap R S) (Ideal.comap e P)) p) (Eq (Ideal.co …
      h : P.LiesOver p
      ⊢ Eq (Module.finrank (HasQuotient.Quotient R p) (HasQuotient.Quotient S (Ideal …
    -/
    exact (Quotient.algEquivOfEqComap p e rfl).toLinearEquiv.finrank_eq
    /-
      🎉 no goals
    -/
    /-
      case neg
      R : Type u
      inst✝⁵ : CommRing R
      S : Type v
      inst✝⁴ : CommRing S
      p : Ideal R
      S₁ : Type u_1
      inst✝³ : CommRing S₁
      inst✝² : Algebra R S₁
      inst✝¹ : Algebra R S
      e : AlgEquiv R S S₁
      P : Ideal S₁
      inst✝ : p.IsMaximal
      he : Iff (Eq (Ideal.comap (algebraMap R S) (Ideal.comap e P)) p) (Eq (Ideal.co …
      h : Not (P.LiesOver p)
      ⊢ Eq (p.inertiaDeg (Ideal.comap e P)) (p.inertiaDeg P)
    -/
  · rw [inertiaDeg, dif_neg (fun eq => h ⟨(he.mp eq).symm⟩)]
    /-
      case neg
      R : Type u
      inst✝⁵ : CommRing R
      S : Type v
      inst✝⁴ : CommRing S
      p : Ideal R
      S₁ : Type u_1
      inst✝³ : CommRing S₁
      inst✝² : Algebra R S₁
      inst✝¹ : Algebra R S
      e : AlgEquiv R S S₁
      P : Ideal S₁
      inst✝ : p.IsMaximal
      he : Iff (Eq (Ideal.comap (algebraMap R S) (Ideal.comap e P)) p) (Eq (Ideal.co …
      h : Not (P.LiesOver p)
      ⊢ Eq 0 (p.inertiaDeg P)
    -/
    rw [inertiaDeg, dif_neg (fun eq => h ⟨eq.symm⟩)]
    /-
      🎉 no goals
    -/


lemma inertiaDeg_map_eq [p.IsMaximal] (P : Ideal S)
    {E : Type*} [EquivLike E S S₁] [AlgEquivClass E R S S₁] (e : E) :
    inertiaDeg p (P.map e) = inertiaDeg p P := by
  /-
    R : Type u
    inst✝⁷ : CommRing R
    S : Type v
    inst✝⁶ : CommRing S
    p : Ideal R
    S₁ : Type u_1
    inst✝⁵ : CommRing S₁
    inst✝⁴ : Algebra R S₁
    inst✝³ : Algebra R S
    inst✝² : p.IsMaximal
    P : Ideal S
    E : Type u_2
    inst✝¹ : EquivLike E S S₁
    inst✝ : AlgEquivClass E R S S₁
    e : E
    ⊢ Eq (p.inertiaDeg (Ideal.map e P)) (p.inertiaDeg P)
  -/
  rw [show P.map e = _ from map_comap_of_equiv (e : S ≃+* S₁)]
  /-
    R : Type u
    inst✝⁷ : CommRing R
    S : Type v
    inst✝⁶ : CommRing S
    p : Ideal R
    S₁ : Type u_1
    inst✝⁵ : CommRing S₁
    inst✝⁴ : Algebra R S₁
    inst✝³ : Algebra R S
    inst✝² : p.IsMaximal
    P : Ideal S
    E : Type u_2
    inst✝¹ : EquivLike E S S₁
    inst✝ : AlgEquivClass E R S S₁
    e : E
    ⊢ Eq (p.inertiaDeg (Ideal.comap (↑e).symm P)) (p.inertiaDeg P)
  -/
  exact p.inertiaDeg_comap_eq (e : S ≃ₐ[R] S₁).symm P
  /-
    🎉 no goals
  -/


/-- If `b` mod `p` spans `S/p` as `R/p`-space, then `b` itself spans `Frac(S)` as `K`-space.

Here,
 * `p` is an ideal of `R` such that `R / p` is nontrivial
 * `K` is a field that has an embedding of `R` (in particular we can take `K = Frac(R)`)
 * `L` is a field extension of `K`
 * `S` is the integral closure of `R` in `L`

More precisely, we avoid quotients in this statement and instead require that `b ∪ pS` spans `S`.
-/
theorem FinrankQuotientMap.span_eq_top [IsDomain R] [IsDomain S] [Algebra K L] [Module.Finite R S]
    [Algebra R L] [IsScalarTower R S L] [IsScalarTower R K L] [Algebra.IsAlgebraic R S]
    [NoZeroSMulDivisors R K] (hp : p ≠ ⊤) (b : Set S)
    (hb' : Submodule.span R b ⊔ (p.map (algebraMap R S)).restrictScalars R = ⊤) :
    Submodule.span K (algebraMap S L '' b) = ⊤ := by
  have hRL : Function.Injective (algebraMap R L) := by
    rw [IsScalarTower.algebraMap_eq R K L]
    exact (algebraMap K L).injective.comp (NoZeroSMulDivisors.algebraMap_injective R K)
  -- Let `M` be the `R`-module spanned by the proposed basis elements.
  /-
    R : Type u
    inst✝¹⁶ : CommRing R
    S : Type v
    inst✝¹⁵ : CommRing S
    p : Ideal R
    inst✝¹⁴ : Algebra R S
    K : Type u_1
    inst✝¹³ : Field K
    inst✝¹² : Algebra R K
    L : Type u_2
    inst✝¹¹ : Field L
    inst✝¹⁰ : Algebra S L
    inst✝⁹ : IsFractionRing S L
    inst✝⁸ : IsDomain R
    inst✝⁷ : IsDomain S
    inst✝⁶ : Algebra K L
    inst✝⁵ : Module.Finite R S
    inst✝⁴ : Algebra R L
    inst✝³ : IsScalarTower R S L
    inst✝² : IsScalarTower R K L
    inst✝¹ : Algebra.IsAlgebraic R S
    inst✝ : NoZeroSMulDivisors R K
    hp : Ne p Top.top
    b : Set S
    hb' : Eq (Max.max (Submodule.span R b) (Submodule.restrictScalars R (Ideal.map …
    hRL : Function.Injective ⇑(algebraMap R L)
    ⊢ Eq (Submodule.span K (Set.image (⇑(algebraMap S L)) b)) Top.top
  -/
  let M : Submodule R S := Submodule.span R b
  -- Then `S / M` is generated by some finite set of `n` vectors `a`.
  /-
    R : Type u
    inst✝¹⁶ : CommRing R
    S : Type v
    inst✝¹⁵ : CommRing S
    p : Ideal R
    inst✝¹⁴ : Algebra R S
    K : Type u_1
    inst✝¹³ : Field K
    inst✝¹² : Algebra R K
    L : Type u_2
    inst✝¹¹ : Field L
    inst✝¹⁰ : Algebra S L
    inst✝⁹ : IsFractionRing S L
    inst✝⁸ : IsDomain R
    inst✝⁷ : IsDomain S
    inst✝⁶ : Algebra K L
    inst✝⁵ : Module.Finite R S
    inst✝⁴ : Algebra R L
    inst✝³ : IsScalarTower R S L
    inst✝² : IsScalarTower R K L
    inst✝¹ : Algebra.IsAlgebraic R S
    inst✝ : NoZeroSMulDivisors R K
    hp : Ne p Top.top
    b : Set S
    hb' : Eq (Max.max (Submodule.span R b) (Submodule.restrictScalars R (Ideal.map …
    hRL : Function.Injective ⇑(algebraMap R L)
    M : Submodule R S := Submodule.span R b
    ⊢ Eq (Submodule.span K (Set.image (⇑(algebraMap S L)) b)) Top.top
  -/
  obtain ⟨n, a, ha⟩ := @Module.Finite.exists_fin R (S ⧸ M) _ _ _ _
  -- Because the image of `p` in `S / M` is `⊤`,
  have smul_top_eq : p • (⊤ : Submodule R (S ⧸ M)) = ⊤ := by
    calc
      p • ⊤ = Submodule.map M.mkQ (p • ⊤) := by
        rw [Submodule.map_smul'', Submodule.map_top, M.range_mkQ]
      _ = ⊤ := by rw [Ideal.smul_top_eq_map, (Submodule.map_mkQ_eq_top M _).mpr hb']
  -- we can write the elements of `a` as `p`-linear combinations of other elements of `a`.
  have exists_sum : ∀ x : S ⧸ M, ∃ a' : Fin n → R, (∀ i, a' i ∈ p) ∧ ∑ i, a' i • a i = x := by
    intro x
    obtain ⟨a'', ha'', hx⟩ := (Submodule.mem_ideal_smul_span_iff_exists_sum p a x).1
      (by { rw [ha, smul_top_eq]; exact Submodule.mem_top } :
        x ∈ p • Submodule.span R (Set.range a))
    · refine ⟨fun i => a'' i, fun i => ha'' _, ?_⟩
      rw [← hx, Finsupp.sum_fintype]
      exact fun _ => zero_smul _ _
  /-
    case intro.intro
    R : Type u
    inst✝¹⁶ : CommRing R
    S : Type v
    inst✝¹⁵ : CommRing S
    p : Ideal R
    inst✝¹⁴ : Algebra R S
    K : Type u_1
    inst✝¹³ : Field K
    inst✝¹² : Algebra R K
    L : Type u_2
    inst✝¹¹ : Field L
    inst✝¹⁰ : Algebra S L
    inst✝⁹ : IsFractionRing S L
    inst✝⁸ : IsDomain R
    inst✝⁷ : IsDomain S
    inst✝⁶ : Algebra K L
    inst✝⁵ : Module.Finite R S
    inst✝⁴ : Algebra R L
    inst✝³ : IsScalarTower R S L
    inst✝² : IsScalarTower R K L
    inst✝¹ : Algebra.IsAlgebraic R S
    inst✝ : NoZeroSMulDivisors R K
    hp : Ne p Top.top
    b : Set S
    hb' : Eq (Max.max (Submodule.span R b) (Submodule.restrictScalars R (Ideal.map …
    hRL : Function.Injective ⇑(algebraMap R L)
    M : Submodule R S := Submodule.span R b
    n : Nat
    a : Fin n → HasQuotient.Quotient S M
    ha : Eq (Submodule.span R (Set.range a)) Top.top
    smul_top_eq : Eq (HSMul.hSMul p Top.top) Top.top
    exists_sum : ∀ (x : HasQuotient.Quotient S M), Exists fun a' => And (∀ (i : Fi …
    ⊢ Eq (Submodule.span K (Set.image (⇑(algebraMap S L)) b)) Top.top
  -/
  choose A' hA'p hA' using fun i => exists_sum (a i)
  -- This gives us a(n invertible) matrix `A` such that `det A ∈ (M = span R b)`,
  /-
    case intro.intro
    R : Type u
    inst✝¹⁶ : CommRing R
    S : Type v
    inst✝¹⁵ : CommRing S
    p : Ideal R
    inst✝¹⁴ : Algebra R S
    K : Type u_1
    inst✝¹³ : Field K
    inst✝¹² : Algebra R K
    L : Type u_2
    inst✝¹¹ : Field L
    inst✝¹⁰ : Algebra S L
    inst✝⁹ : IsFractionRing S L
    inst✝⁸ : IsDomain R
    inst✝⁷ : IsDomain S
    inst✝⁶ : Algebra K L
    inst✝⁵ : Module.Finite R S
    inst✝⁴ : Algebra R L
    inst✝³ : IsScalarTower R S L
    inst✝² : IsScalarTower R K L
    inst✝¹ : Algebra.IsAlgebraic R S
    inst✝ : NoZeroSMulDivisors R K
    hp : Ne p Top.top
    b : Set S
    hb' : Eq (Max.max (Submodule.span R b) (Submodule.restrictScalars R (Ideal.map …
    hRL : Function.Injective ⇑(algebraMap R L)
    M : Submodule R S := Submodule.span R b
    n : Nat
    a : Fin n → HasQuotient.Quotient S M
    ha : Eq (Submodule.span R (Set.range a)) Top.top
    smul_top_eq : Eq (HSMul.hSMul p Top.top) Top.top
    exists_sum : ∀ (x : HasQuotient.Quotient S M), Exists fun a' => And (∀ (i : Fi …
    A' : Fin n → Fin n → R
    hA'p : ∀ (i i_1 : Fin n), Membership.mem p (A' i i_1)
    hA' : ∀ (i : Fin n), Eq (Finset.univ.sum fun i_1 => HSMul.hSMul (A' i i_1) (a  …
    ⊢ Eq (Submodule.span K (Set.image (⇑(algebraMap S L)) b)) Top.top
  -/
  let A : Matrix (Fin n) (Fin n) R := Matrix.of A' - 1
  /-
    case intro.intro
    R : Type u
    inst✝¹⁶ : CommRing R
    S : Type v
    inst✝¹⁵ : CommRing S
    p : Ideal R
    inst✝¹⁴ : Algebra R S
    K : Type u_1
    inst✝¹³ : Field K
    inst✝¹² : Algebra R K
    L : Type u_2
    inst✝¹¹ : Field L
    inst✝¹⁰ : Algebra S L
    inst✝⁹ : IsFractionRing S L
    inst✝⁸ : IsDomain R
    inst✝⁷ : IsDomain S
    inst✝⁶ : Algebra K L
    inst✝⁵ : Module.Finite R S
    inst✝⁴ : Algebra R L
    inst✝³ : IsScalarTower R S L
    inst✝² : IsScalarTower R K L
    inst✝¹ : Algebra.IsAlgebraic R S
    inst✝ : NoZeroSMulDivisors R K
    hp : Ne p Top.top
    b : Set S
    hb' : Eq (Max.max (Submodule.span R b) (Submodule.restrictScalars R (Ideal.map …
    hRL : Function.Injective ⇑(algebraMap R L)
    M : Submodule R S := Submodule.span R b
    n : Nat
    a : Fin n → HasQuotient.Quotient S M
    ha : Eq (Submodule.span R (Set.range a)) Top.top
    smul_top_eq : Eq (HSMul.hSMul p Top.top) Top.top
    exists_sum : ∀ (x : HasQuotient.Quotient S M), Exists fun a' => And (∀ (i : Fi …
    A' : Fin n → Fin n → R
    hA'p : ∀ (i i_1 : Fin n), Membership.mem p (A' i i_1)
    hA' : ∀ (i : Fin n), Eq (Finset.univ.sum fun i_1 => HSMul.hSMul (A' i i_1) (a  …
    A : Matrix (Fin n) (Fin n) R := HSub.hSub (Matrix.of A') 1
    ⊢ Eq (Submodule.span K (Set.image (⇑(algebraMap S L)) b)) Top.top
  -/
  let B := A.adjugate
  have A_smul : ∀ i, ∑ j, A i j • a j = 0 := by
    intros
    simp [A, Matrix.sub_apply, Matrix.of_apply, ne_eq, Matrix.one_apply, sub_smul,
      Finset.sum_sub_distrib, hA', sub_self]
  -- since `span S {det A} / M = 0`.
  have d_smul : ∀ i, A.det • a i = 0 := by
    intro i
    calc
      A.det • a i = ∑ j, (B * A) i j • a j := ?_
      _ = ∑ k, B i k • ∑ j, A k j • a j := ?_
      _ = 0 := Finset.sum_eq_zero fun k _ => ?_
    · simp only [B, Matrix.adjugate_mul, Matrix.smul_apply, Matrix.one_apply, smul_eq_mul, ite_true,
        mul_ite, mul_one, mul_zero, ite_smul, zero_smul, Finset.sum_ite_eq, Finset.mem_univ]
    · simp only [Matrix.mul_apply, Finset.smul_sum, Finset.sum_smul, smul_smul]
      rw [Finset.sum_comm]
    · rw [A_smul, smul_zero]
  -- In the rings of integers we have the desired inclusion.
  have span_d : (Submodule.span S ({algebraMap R S A.det} : Set S)).restrictScalars R ≤ M := by
    intro x hx
    rw [Submodule.restrictScalars_mem] at hx
    obtain ⟨x', rfl⟩ := Submodule.mem_span_singleton.mp hx
    rw [smul_eq_mul, mul_comm, ← Algebra.smul_def] at hx ⊢
    rw [← Submodule.Quotient.mk_eq_zero, Submodule.Quotient.mk_smul]
    obtain ⟨a', _, quot_x_eq⟩ := exists_sum (Submodule.Quotient.mk x')
    rw [← quot_x_eq, Finset.smul_sum]
    conv =>
      lhs; congr; next => skip
      intro x; rw [smul_comm A.det, d_smul, smul_zero]
    exact Finset.sum_const_zero
  refine top_le_iff.mp
      (calc
        ⊤ = (Ideal.span {algebraMap R L A.det}).restrictScalars K := ?_
        _ ≤ Submodule.span K (algebraMap S L '' b) := ?_)
  -- Because `det A ≠ 0`, we have `span L {det A} = ⊤`.
    /-
      case intro.intro.refine_1
      R : Type u
      inst✝¹⁶ : CommRing R
      S : Type v
      inst✝¹⁵ : CommRing S
      p : Ideal R
      inst✝¹⁴ : Algebra R S
      K : Type u_1
      inst✝¹³ : Field K
      inst✝¹² : Algebra R K
      L : Type u_2
      inst✝¹¹ : Field L
      inst✝¹⁰ : Algebra S L
      inst✝⁹ : IsFractionRing S L
      inst✝⁸ : IsDomain R
      inst✝⁷ : IsDomain S
      inst✝⁶ : Algebra K L
      inst✝⁵ : Module.Finite R S
      inst✝⁴ : Algebra R L
      inst✝³ : IsScalarTower R S L
      inst✝² : IsScalarTower R K L
      inst✝¹ : Algebra.IsAlgebraic R S
      inst✝ : NoZeroSMulDivisors R K
      hp : Ne p Top.top
      b : Set S
      hb' : Eq (Max.max (Submodule.span R b) (Submodule.restrictScalars R (Ideal.map …
      hRL : Function.Injective ⇑(algebraMap R L)
      M : Submodule R S := Submodule.span R b
      n : Nat
      a : Fin n → HasQuotient.Quotient S M
      ha : Eq (Submodule.span R (Set.range a)) Top.top
      smul_top_eq : Eq (HSMul.hSMul p Top.top) Top.top
      exists_sum : ∀ (x : HasQuotient.Quotient S M), Exists fun a' => And (∀ (i : Fi …
      A' : Fin n → Fin n → R
      hA'p : ∀ (i i_1 : Fin n), Membership.mem p (A' i i_1)
      hA' : ∀ (i : Fin n), Eq (Finset.univ.sum fun i_1 => HSMul.hSMul (A' i i_1) (a  …
      A : Matrix (Fin n) (Fin n) R := HSub.hSub (Matrix.of A') 1
      B : Matrix (Fin n) (Fin n) R := A.adjugate
      A_smul : ∀ (i : Fin n), Eq (Finset.univ.sum fun j => HSMul.hSMul (A i j) (a j) …
      d_smul : ∀ (i : Fin n), Eq (HSMul.hSMul A.det (a i)) 0
      span_d : LE.le (Submodule.restrictScalars R (Submodule.span S (Singleton.singl …
      ⊢ Eq Top.top (Submodule.restrictScalars K (Ideal.span (Singleton.singleton ((a …
    -/
  · rw [eq_comm, Submodule.restrictScalars_eq_top_iff, Ideal.span_singleton_eq_top]
    /-
      case intro.intro.refine_1
      R : Type u
      inst✝¹⁶ : CommRing R
      S : Type v
      inst✝¹⁵ : CommRing S
      p : Ideal R
      inst✝¹⁴ : Algebra R S
      K : Type u_1
      inst✝¹³ : Field K
      inst✝¹² : Algebra R K
      L : Type u_2
      inst✝¹¹ : Field L
      inst✝¹⁰ : Algebra S L
      inst✝⁹ : IsFractionRing S L
      inst✝⁸ : IsDomain R
      inst✝⁷ : IsDomain S
      inst✝⁶ : Algebra K L
      inst✝⁵ : Module.Finite R S
      inst✝⁴ : Algebra R L
      inst✝³ : IsScalarTower R S L
      inst✝² : IsScalarTower R K L
      inst✝¹ : Algebra.IsAlgebraic R S
      inst✝ : NoZeroSMulDivisors R K
      hp : Ne p Top.top
      b : Set S
      hb' : Eq (Max.max (Submodule.span R b) (Submodule.restrictScalars R (Ideal.map …
      hRL : Function.Injective ⇑(algebraMap R L)
      M : Submodule R S := Submodule.span R b
      n : Nat
      a : Fin n → HasQuotient.Quotient S M
      ha : Eq (Submodule.span R (Set.range a)) Top.top
      smul_top_eq : Eq (HSMul.hSMul p Top.top) Top.top
      exists_sum : ∀ (x : HasQuotient.Quotient S M), Exists fun a' => And (∀ (i : Fi …
      A' : Fin n → Fin n → R
      hA'p : ∀ (i i_1 : Fin n), Membership.mem p (A' i i_1)
      hA' : ∀ (i : Fin n), Eq (Finset.univ.sum fun i_1 => HSMul.hSMul (A' i i_1) (a  …
      A : Matrix (Fin n) (Fin n) R := HSub.hSub (Matrix.of A') 1
      B : Matrix (Fin n) (Fin n) R := A.adjugate
      A_smul : ∀ (i : Fin n), Eq (Finset.univ.sum fun j => HSMul.hSMul (A i j) (a j) …
      d_smul : ∀ (i : Fin n), Eq (HSMul.hSMul A.det (a i)) 0
      span_d : LE.le (Submodule.restrictScalars R (Submodule.span S (Singleton.singl …
      ⊢ IsUnit ((algebraMap R L) A.det)
    -/
    refine IsUnit.mk0 _ ((map_ne_zero_iff (algebraMap R L) hRL).mpr ?_)
    /-
      case intro.intro.refine_1
      R : Type u
      inst✝¹⁶ : CommRing R
      S : Type v
      inst✝¹⁵ : CommRing S
      p : Ideal R
      inst✝¹⁴ : Algebra R S
      K : Type u_1
      inst✝¹³ : Field K
      inst✝¹² : Algebra R K
      L : Type u_2
      inst✝¹¹ : Field L
      inst✝¹⁰ : Algebra S L
      inst✝⁹ : IsFractionRing S L
      inst✝⁸ : IsDomain R
      inst✝⁷ : IsDomain S
      inst✝⁶ : Algebra K L
      inst✝⁵ : Module.Finite R S
      inst✝⁴ : Algebra R L
      inst✝³ : IsScalarTower R S L
      inst✝² : IsScalarTower R K L
      inst✝¹ : Algebra.IsAlgebraic R S
      inst✝ : NoZeroSMulDivisors R K
      hp : Ne p Top.top
      b : Set S
      hb' : Eq (Max.max (Submodule.span R b) (Submodule.restrictScalars R (Ideal.map …
      hRL : Function.Injective ⇑(algebraMap R L)
      M : Submodule R S := Submodule.span R b
      n : Nat
      a : Fin n → HasQuotient.Quotient S M
      ha : Eq (Submodule.span R (Set.range a)) Top.top
      smul_top_eq : Eq (HSMul.hSMul p Top.top) Top.top
      exists_sum : ∀ (x : HasQuotient.Quotient S M), Exists fun a' => And (∀ (i : Fi …
      A' : Fin n → Fin n → R
      hA'p : ∀ (i i_1 : Fin n), Membership.mem p (A' i i_1)
      hA' : ∀ (i : Fin n), Eq (Finset.univ.sum fun i_1 => HSMul.hSMul (A' i i_1) (a  …
      A : Matrix (Fin n) (Fin n) R := HSub.hSub (Matrix.of A') 1
      B : Matrix (Fin n) (Fin n) R := A.adjugate
      A_smul : ∀ (i : Fin n), Eq (Finset.univ.sum fun j => HSMul.hSMul (A i j) (a j) …
      d_smul : ∀ (i : Fin n), Eq (HSMul.hSMul A.det (a i)) 0
      span_d : LE.le (Submodule.restrictScalars R (Submodule.span S (Singleton.singl …
      ⊢ Ne A.det 0
    -/
    refine ne_zero_of_map (f := Ideal.Quotient.mk p) ?_
    /-
      case intro.intro.refine_1
      R : Type u
      inst✝¹⁶ : CommRing R
      S : Type v
      inst✝¹⁵ : CommRing S
      p : Ideal R
      inst✝¹⁴ : Algebra R S
      K : Type u_1
      inst✝¹³ : Field K
      inst✝¹² : Algebra R K
      L : Type u_2
      inst✝¹¹ : Field L
      inst✝¹⁰ : Algebra S L
      inst✝⁹ : IsFractionRing S L
      inst✝⁸ : IsDomain R
      inst✝⁷ : IsDomain S
      inst✝⁶ : Algebra K L
      inst✝⁵ : Module.Finite R S
      inst✝⁴ : Algebra R L
      inst✝³ : IsScalarTower R S L
      inst✝² : IsScalarTower R K L
      inst✝¹ : Algebra.IsAlgebraic R S
      inst✝ : NoZeroSMulDivisors R K
      hp : Ne p Top.top
      b : Set S
      hb' : Eq (Max.max (Submodule.span R b) (Submodule.restrictScalars R (Ideal.map …
      hRL : Function.Injective ⇑(algebraMap R L)
      M : Submodule R S := Submodule.span R b
      n : Nat
      a : Fin n → HasQuotient.Quotient S M
      ha : Eq (Submodule.span R (Set.range a)) Top.top
      smul_top_eq : Eq (HSMul.hSMul p Top.top) Top.top
      exists_sum : ∀ (x : HasQuotient.Quotient S M), Exists fun a' => And (∀ (i : Fi …
      A' : Fin n → Fin n → R
      hA'p : ∀ (i i_1 : Fin n), Membership.mem p (A' i i_1)
      hA' : ∀ (i : Fin n), Eq (Finset.univ.sum fun i_1 => HSMul.hSMul (A' i i_1) (a  …
      A : Matrix (Fin n) (Fin n) R := HSub.hSub (Matrix.of A') 1
      B : Matrix (Fin n) (Fin n) R := A.adjugate
      A_smul : ∀ (i : Fin n), Eq (Finset.univ.sum fun j => HSMul.hSMul (A i j) (a j) …
      d_smul : ∀ (i : Fin n), Eq (HSMul.hSMul A.det (a i)) 0
      span_d : LE.le (Submodule.restrictScalars R (Submodule.span S (Singleton.singl …
      ⊢ Ne ((Ideal.Quotient.mk p) A.det) 0
    -/
    haveI := Ideal.Quotient.nontrivial hp
    calc
      Ideal.Quotient.mk p A.det = Matrix.det ((Ideal.Quotient.mk p).mapMatrix A) := by
        rw [RingHom.map_det]
      _ = Matrix.det ((Ideal.Quotient.mk p).mapMatrix (Matrix.of A' - 1)) := rfl
      _ = Matrix.det fun i j =>
          (Ideal.Quotient.mk p) (A' i j) - (1 : Matrix (Fin n) (Fin n) (R ⧸ p)) i j := ?_
      _ = Matrix.det (-1 : Matrix (Fin n) (Fin n) (R ⧸ p)) := ?_
      _ = (-1 : R ⧸ p) ^ n := by rw [Matrix.det_neg, Fintype.card_fin, Matrix.det_one, mul_one]
      _ ≠ 0 := IsUnit.ne_zero (isUnit_one.neg.pow _)
      /-
        case intro.intro.refine_1.calc_1
        R : Type u
        inst✝¹⁶ : CommRing R
        S : Type v
        inst✝¹⁵ : CommRing S
        p : Ideal R
        inst✝¹⁴ : Algebra R S
        K : Type u_1
        inst✝¹³ : Field K
        inst✝¹² : Algebra R K
        L : Type u_2
        inst✝¹¹ : Field L
        inst✝¹⁰ : Algebra S L
        inst✝⁹ : IsFractionRing S L
        inst✝⁸ : IsDomain R
        inst✝⁷ : IsDomain S
        inst✝⁶ : Algebra K L
        inst✝⁵ : Module.Finite R S
        inst✝⁴ : Algebra R L
        inst✝³ : IsScalarTower R S L
        inst✝² : IsScalarTower R K L
        inst✝¹ : Algebra.IsAlgebraic R S
        inst✝ : NoZeroSMulDivisors R K
        hp : Ne p Top.top
        b : Set S
        hb' : Eq (Max.max (Submodule.span R b) (Submodule.restrictScalars R (Ideal.map …
        hRL : Function.Injective ⇑(algebraMap R L)
        M : Submodule R S := Submodule.span R b
        n : Nat
        a : Fin n → HasQuotient.Quotient S M
        ha : Eq (Submodule.span R (Set.range a)) Top.top
        smul_top_eq : Eq (HSMul.hSMul p Top.top) Top.top
        exists_sum : ∀ (x : HasQuotient.Quotient S M), Exists fun a' => And (∀ (i : Fi …
        A' : Fin n → Fin n → R
        hA'p : ∀ (i i_1 : Fin n), Membership.mem p (A' i i_1)
        hA' : ∀ (i : Fin n), Eq (Finset.univ.sum fun i_1 => HSMul.hSMul (A' i i_1) (a  …
        A : Matrix (Fin n) (Fin n) R := HSub.hSub (Matrix.of A') 1
        B : Matrix (Fin n) (Fin n) R := A.adjugate
        A_smul : ∀ (i : Fin n), Eq (Finset.univ.sum fun j => HSMul.hSMul (A i j) (a j) …
        d_smul : ∀ (i : Fin n), Eq (HSMul.hSMul A.det (a i)) 0
        span_d : LE.le (Submodule.restrictScalars R (Submodule.span S (Singleton.singl …
        this : Nontrivial (HasQuotient.Quotient R p)
        ⊢ Eq ((Ideal.Quotient.mk p).mapMatrix (HSub.hSub (Matrix.of A') 1)).det (Matri …
      -/
    · refine congr_arg Matrix.det (Matrix.ext fun i j => ?_)
      /-
        case intro.intro.refine_1.calc_1
        R : Type u
        inst✝¹⁶ : CommRing R
        S : Type v
        inst✝¹⁵ : CommRing S
        p : Ideal R
        inst✝¹⁴ : Algebra R S
        K : Type u_1
        inst✝¹³ : Field K
        inst✝¹² : Algebra R K
        L : Type u_2
        inst✝¹¹ : Field L
        inst✝¹⁰ : Algebra S L
        inst✝⁹ : IsFractionRing S L
        inst✝⁸ : IsDomain R
        inst✝⁷ : IsDomain S
        inst✝⁶ : Algebra K L
        inst✝⁵ : Module.Finite R S
        inst✝⁴ : Algebra R L
        inst✝³ : IsScalarTower R S L
        inst✝² : IsScalarTower R K L
        inst✝¹ : Algebra.IsAlgebraic R S
        inst✝ : NoZeroSMulDivisors R K
        hp : Ne p Top.top
        b : Set S
        hb' : Eq (Max.max (Submodule.span R b) (Submodule.restrictScalars R (Ideal.map …
        hRL : Function.Injective ⇑(algebraMap R L)
        M : Submodule R S := Submodule.span R b
        n : Nat
        a : Fin n → HasQuotient.Quotient S M
        ha : Eq (Submodule.span R (Set.range a)) Top.top
        smul_top_eq : Eq (HSMul.hSMul p Top.top) Top.top
        exists_sum : ∀ (x : HasQuotient.Quotient S M), Exists fun a' => And (∀ (i : Fi …
        A' : Fin n → Fin n → R
        hA'p : ∀ (i i_1 : Fin n), Membership.mem p (A' i i_1)
        hA' : ∀ (i : Fin n), Eq (Finset.univ.sum fun i_1 => HSMul.hSMul (A' i i_1) (a  …
        A : Matrix (Fin n) (Fin n) R := HSub.hSub (Matrix.of A') 1
        B : Matrix (Fin n) (Fin n) R := A.adjugate
        A_smul : ∀ (i : Fin n), Eq (Finset.univ.sum fun j => HSMul.hSMul (A i j) (a j) …
        d_smul : ∀ (i : Fin n), Eq (HSMul.hSMul A.det (a i)) 0
        span_d : LE.le (Submodule.restrictScalars R (Submodule.span S (Singleton.singl …
        this : Nontrivial (HasQuotient.Quotient R p)
        i j : Fin n
        ⊢ Eq ((Ideal.Quotient.mk p).mapMatrix (HSub.hSub (Matrix.of A') 1) i j) (HSub. …
      -/
      rw [map_sub, RingHom.mapMatrix_apply, map_one]
      /-
        case intro.intro.refine_1.calc_1
        R : Type u
        inst✝¹⁶ : CommRing R
        S : Type v
        inst✝¹⁵ : CommRing S
        p : Ideal R
        inst✝¹⁴ : Algebra R S
        K : Type u_1
        inst✝¹³ : Field K
        inst✝¹² : Algebra R K
        L : Type u_2
        inst✝¹¹ : Field L
        inst✝¹⁰ : Algebra S L
        inst✝⁹ : IsFractionRing S L
        inst✝⁸ : IsDomain R
        inst✝⁷ : IsDomain S
        inst✝⁶ : Algebra K L
        inst✝⁵ : Module.Finite R S
        inst✝⁴ : Algebra R L
        inst✝³ : IsScalarTower R S L
        inst✝² : IsScalarTower R K L
        inst✝¹ : Algebra.IsAlgebraic R S
        inst✝ : NoZeroSMulDivisors R K
        hp : Ne p Top.top
        b : Set S
        hb' : Eq (Max.max (Submodule.span R b) (Submodule.restrictScalars R (Ideal.map …
        hRL : Function.Injective ⇑(algebraMap R L)
        M : Submodule R S := Submodule.span R b
        n : Nat
        a : Fin n → HasQuotient.Quotient S M
        ha : Eq (Submodule.span R (Set.range a)) Top.top
        smul_top_eq : Eq (HSMul.hSMul p Top.top) Top.top
        exists_sum : ∀ (x : HasQuotient.Quotient S M), Exists fun a' => And (∀ (i : Fi …
        A' : Fin n → Fin n → R
        hA'p : ∀ (i i_1 : Fin n), Membership.mem p (A' i i_1)
        hA' : ∀ (i : Fin n), Eq (Finset.univ.sum fun i_1 => HSMul.hSMul (A' i i_1) (a  …
        A : Matrix (Fin n) (Fin n) R := HSub.hSub (Matrix.of A') 1
        B : Matrix (Fin n) (Fin n) R := A.adjugate
        A_smul : ∀ (i : Fin n), Eq (Finset.univ.sum fun j => HSMul.hSMul (A i j) (a j) …
        d_smul : ∀ (i : Fin n), Eq (HSMul.hSMul A.det (a i)) 0
        span_d : LE.le (Submodule.restrictScalars R (Submodule.span S (Singleton.singl …
        this : Nontrivial (HasQuotient.Quotient R p)
        i j : Fin n
        ⊢ Eq (HSub.hSub ((Matrix.of A').map ⇑(Ideal.Quotient.mk p)) 1 i j) (HSub.hSub  …
      -/
      rfl
      /-
        🎉 no goals
      -/
      /-
        case intro.intro.refine_1.calc_2
        R : Type u
        inst✝¹⁶ : CommRing R
        S : Type v
        inst✝¹⁵ : CommRing S
        p : Ideal R
        inst✝¹⁴ : Algebra R S
        K : Type u_1
        inst✝¹³ : Field K
        inst✝¹² : Algebra R K
        L : Type u_2
        inst✝¹¹ : Field L
        inst✝¹⁰ : Algebra S L
        inst✝⁹ : IsFractionRing S L
        inst✝⁸ : IsDomain R
        inst✝⁷ : IsDomain S
        inst✝⁶ : Algebra K L
        inst✝⁵ : Module.Finite R S
        inst✝⁴ : Algebra R L
        inst✝³ : IsScalarTower R S L
        inst✝² : IsScalarTower R K L
        inst✝¹ : Algebra.IsAlgebraic R S
        inst✝ : NoZeroSMulDivisors R K
        hp : Ne p Top.top
        b : Set S
        hb' : Eq (Max.max (Submodule.span R b) (Submodule.restrictScalars R (Ideal.map …
        hRL : Function.Injective ⇑(algebraMap R L)
        M : Submodule R S := Submodule.span R b
        n : Nat
        a : Fin n → HasQuotient.Quotient S M
        ha : Eq (Submodule.span R (Set.range a)) Top.top
        smul_top_eq : Eq (HSMul.hSMul p Top.top) Top.top
        exists_sum : ∀ (x : HasQuotient.Quotient S M), Exists fun a' => And (∀ (i : Fi …
        A' : Fin n → Fin n → R
        hA'p : ∀ (i i_1 : Fin n), Membership.mem p (A' i i_1)
        hA' : ∀ (i : Fin n), Eq (Finset.univ.sum fun i_1 => HSMul.hSMul (A' i i_1) (a  …
        A : Matrix (Fin n) (Fin n) R := HSub.hSub (Matrix.of A') 1
        B : Matrix (Fin n) (Fin n) R := A.adjugate
        A_smul : ∀ (i : Fin n), Eq (Finset.univ.sum fun j => HSMul.hSMul (A i j) (a j) …
        d_smul : ∀ (i : Fin n), Eq (HSMul.hSMul A.det (a i)) 0
        span_d : LE.le (Submodule.restrictScalars R (Submodule.span S (Singleton.singl …
        this : Nontrivial (HasQuotient.Quotient R p)
        ⊢ Eq (Matrix.det fun i j => HSub.hSub ((Ideal.Quotient.mk p) (A' i j)) (1 i j) …
      -/
    · refine congr_arg Matrix.det (Matrix.ext fun i j => ?_)
      /-
        case intro.intro.refine_1.calc_2
        R : Type u
        inst✝¹⁶ : CommRing R
        S : Type v
        inst✝¹⁵ : CommRing S
        p : Ideal R
        inst✝¹⁴ : Algebra R S
        K : Type u_1
        inst✝¹³ : Field K
        inst✝¹² : Algebra R K
        L : Type u_2
        inst✝¹¹ : Field L
        inst✝¹⁰ : Algebra S L
        inst✝⁹ : IsFractionRing S L
        inst✝⁸ : IsDomain R
        inst✝⁷ : IsDomain S
        inst✝⁶ : Algebra K L
        inst✝⁵ : Module.Finite R S
        inst✝⁴ : Algebra R L
        inst✝³ : IsScalarTower R S L
        inst✝² : IsScalarTower R K L
        inst✝¹ : Algebra.IsAlgebraic R S
        inst✝ : NoZeroSMulDivisors R K
        hp : Ne p Top.top
        b : Set S
        hb' : Eq (Max.max (Submodule.span R b) (Submodule.restrictScalars R (Ideal.map …
        hRL : Function.Injective ⇑(algebraMap R L)
        M : Submodule R S := Submodule.span R b
        n : Nat
        a : Fin n → HasQuotient.Quotient S M
        ha : Eq (Submodule.span R (Set.range a)) Top.top
        smul_top_eq : Eq (HSMul.hSMul p Top.top) Top.top
        exists_sum : ∀ (x : HasQuotient.Quotient S M), Exists fun a' => And (∀ (i : Fi …
        A' : Fin n → Fin n → R
        hA'p : ∀ (i i_1 : Fin n), Membership.mem p (A' i i_1)
        hA' : ∀ (i : Fin n), Eq (Finset.univ.sum fun i_1 => HSMul.hSMul (A' i i_1) (a  …
        A : Matrix (Fin n) (Fin n) R := HSub.hSub (Matrix.of A') 1
        B : Matrix (Fin n) (Fin n) R := A.adjugate
        A_smul : ∀ (i : Fin n), Eq (Finset.univ.sum fun j => HSMul.hSMul (A i j) (a j) …
        d_smul : ∀ (i : Fin n), Eq (HSMul.hSMul A.det (a i)) 0
        span_d : LE.le (Submodule.restrictScalars R (Submodule.span S (Singleton.singl …
        this : Nontrivial (HasQuotient.Quotient R p)
        i j : Fin n
        ⊢ Eq (HSub.hSub ((Ideal.Quotient.mk p) (A' i j)) (1 i j)) (Neg.neg 1 i j)
      -/
      rw [Ideal.Quotient.eq_zero_iff_mem.mpr (hA'p i j), zero_sub]
      /-
        case intro.intro.refine_1.calc_2
        R : Type u
        inst✝¹⁶ : CommRing R
        S : Type v
        inst✝¹⁵ : CommRing S
        p : Ideal R
        inst✝¹⁴ : Algebra R S
        K : Type u_1
        inst✝¹³ : Field K
        inst✝¹² : Algebra R K
        L : Type u_2
        inst✝¹¹ : Field L
        inst✝¹⁰ : Algebra S L
        inst✝⁹ : IsFractionRing S L
        inst✝⁸ : IsDomain R
        inst✝⁷ : IsDomain S
        inst✝⁶ : Algebra K L
        inst✝⁵ : Module.Finite R S
        inst✝⁴ : Algebra R L
        inst✝³ : IsScalarTower R S L
        inst✝² : IsScalarTower R K L
        inst✝¹ : Algebra.IsAlgebraic R S
        inst✝ : NoZeroSMulDivisors R K
        hp : Ne p Top.top
        b : Set S
        hb' : Eq (Max.max (Submodule.span R b) (Submodule.restrictScalars R (Ideal.map …
        hRL : Function.Injective ⇑(algebraMap R L)
        M : Submodule R S := Submodule.span R b
        n : Nat
        a : Fin n → HasQuotient.Quotient S M
        ha : Eq (Submodule.span R (Set.range a)) Top.top
        smul_top_eq : Eq (HSMul.hSMul p Top.top) Top.top
        exists_sum : ∀ (x : HasQuotient.Quotient S M), Exists fun a' => And (∀ (i : Fi …
        A' : Fin n → Fin n → R
        hA'p : ∀ (i i_1 : Fin n), Membership.mem p (A' i i_1)
        hA' : ∀ (i : Fin n), Eq (Finset.univ.sum fun i_1 => HSMul.hSMul (A' i i_1) (a  …
        A : Matrix (Fin n) (Fin n) R := HSub.hSub (Matrix.of A') 1
        B : Matrix (Fin n) (Fin n) R := A.adjugate
        A_smul : ∀ (i : Fin n), Eq (Finset.univ.sum fun j => HSMul.hSMul (A i j) (a j) …
        d_smul : ∀ (i : Fin n), Eq (HSMul.hSMul A.det (a i)) 0
        span_d : LE.le (Submodule.restrictScalars R (Submodule.span S (Singleton.singl …
        this : Nontrivial (HasQuotient.Quotient R p)
        i j : Fin n
        ⊢ Eq (Neg.neg (1 i j)) (Neg.neg 1 i j)
      -/
      rfl
      /-
        🎉 no goals
      -/
  -- And we conclude `L = span L {det A} ≤ span K b`, so `span K b` spans everything.
    /-
      case intro.intro.refine_2
      R : Type u
      inst✝¹⁶ : CommRing R
      S : Type v
      inst✝¹⁵ : CommRing S
      p : Ideal R
      inst✝¹⁴ : Algebra R S
      K : Type u_1
      inst✝¹³ : Field K
      inst✝¹² : Algebra R K
      L : Type u_2
      inst✝¹¹ : Field L
      inst✝¹⁰ : Algebra S L
      inst✝⁹ : IsFractionRing S L
      inst✝⁸ : IsDomain R
      inst✝⁷ : IsDomain S
      inst✝⁶ : Algebra K L
      inst✝⁵ : Module.Finite R S
      inst✝⁴ : Algebra R L
      inst✝³ : IsScalarTower R S L
      inst✝² : IsScalarTower R K L
      inst✝¹ : Algebra.IsAlgebraic R S
      inst✝ : NoZeroSMulDivisors R K
      hp : Ne p Top.top
      b : Set S
      hb' : Eq (Max.max (Submodule.span R b) (Submodule.restrictScalars R (Ideal.map …
      hRL : Function.Injective ⇑(algebraMap R L)
      M : Submodule R S := Submodule.span R b
      n : Nat
      a : Fin n → HasQuotient.Quotient S M
      ha : Eq (Submodule.span R (Set.range a)) Top.top
      smul_top_eq : Eq (HSMul.hSMul p Top.top) Top.top
      exists_sum : ∀ (x : HasQuotient.Quotient S M), Exists fun a' => And (∀ (i : Fi …
      A' : Fin n → Fin n → R
      hA'p : ∀ (i i_1 : Fin n), Membership.mem p (A' i i_1)
      hA' : ∀ (i : Fin n), Eq (Finset.univ.sum fun i_1 => HSMul.hSMul (A' i i_1) (a  …
      A : Matrix (Fin n) (Fin n) R := HSub.hSub (Matrix.of A') 1
      B : Matrix (Fin n) (Fin n) R := A.adjugate
      A_smul : ∀ (i : Fin n), Eq (Finset.univ.sum fun j => HSMul.hSMul (A i j) (a j) …
      d_smul : ∀ (i : Fin n), Eq (HSMul.hSMul A.det (a i)) 0
      span_d : LE.le (Submodule.restrictScalars R (Submodule.span S (Singleton.singl …
      ⊢ LE.le (Submodule.restrictScalars K (Ideal.span (Singleton.singleton ((algebr …
    -/
  · intro x hx
    /-
      case intro.intro.refine_2
      R : Type u
      inst✝¹⁶ : CommRing R
      S : Type v
      inst✝¹⁵ : CommRing S
      p : Ideal R
      inst✝¹⁴ : Algebra R S
      K : Type u_1
      inst✝¹³ : Field K
      inst✝¹² : Algebra R K
      L : Type u_2
      inst✝¹¹ : Field L
      inst✝¹⁰ : Algebra S L
      inst✝⁹ : IsFractionRing S L
      inst✝⁸ : IsDomain R
      inst✝⁷ : IsDomain S
      inst✝⁶ : Algebra K L
      inst✝⁵ : Module.Finite R S
      inst✝⁴ : Algebra R L
      inst✝³ : IsScalarTower R S L
      inst✝² : IsScalarTower R K L
      inst✝¹ : Algebra.IsAlgebraic R S
      inst✝ : NoZeroSMulDivisors R K
      hp : Ne p Top.top
      b : Set S
      hb' : Eq (Max.max (Submodule.span R b) (Submodule.restrictScalars R (Ideal.map …
      hRL : Function.Injective ⇑(algebraMap R L)
      M : Submodule R S := Submodule.span R b
      n : Nat
      a : Fin n → HasQuotient.Quotient S M
      ha : Eq (Submodule.span R (Set.range a)) Top.top
      smul_top_eq : Eq (HSMul.hSMul p Top.top) Top.top
      exists_sum : ∀ (x : HasQuotient.Quotient S M), Exists fun a' => And (∀ (i : Fi …
      A' : Fin n → Fin n → R
      hA'p : ∀ (i i_1 : Fin n), Membership.mem p (A' i i_1)
      hA' : ∀ (i : Fin n), Eq (Finset.univ.sum fun i_1 => HSMul.hSMul (A' i i_1) (a  …
      A : Matrix (Fin n) (Fin n) R := HSub.hSub (Matrix.of A') 1
      B : Matrix (Fin n) (Fin n) R := A.adjugate
      A_smul : ∀ (i : Fin n), Eq (Finset.univ.sum fun j => HSMul.hSMul (A i j) (a j) …
      d_smul : ∀ (i : Fin n), Eq (HSMul.hSMul A.det (a i)) 0
      span_d : LE.le (Submodule.restrictScalars R (Submodule.span S (Singleton.singl …
      x : L
      hx : Membership.mem (Submodule.restrictScalars K (Ideal.span (Singleton.single …
      ⊢ Membership.mem (Submodule.span K (Set.image (⇑(algebraMap S L)) b)) x
    -/
    rw [Submodule.restrictScalars_mem, IsScalarTower.algebraMap_apply R S L] at hx
    /-
      case intro.intro.refine_2
      R : Type u
      inst✝¹⁶ : CommRing R
      S : Type v
      inst✝¹⁵ : CommRing S
      p : Ideal R
      inst✝¹⁴ : Algebra R S
      K : Type u_1
      inst✝¹³ : Field K
      inst✝¹² : Algebra R K
      L : Type u_2
      inst✝¹¹ : Field L
      inst✝¹⁰ : Algebra S L
      inst✝⁹ : IsFractionRing S L
      inst✝⁸ : IsDomain R
      inst✝⁷ : IsDomain S
      inst✝⁶ : Algebra K L
      inst✝⁵ : Module.Finite R S
      inst✝⁴ : Algebra R L
      inst✝³ : IsScalarTower R S L
      inst✝² : IsScalarTower R K L
      inst✝¹ : Algebra.IsAlgebraic R S
      inst✝ : NoZeroSMulDivisors R K
      hp : Ne p Top.top
      b : Set S
      hb' : Eq (Max.max (Submodule.span R b) (Submodule.restrictScalars R (Ideal.map …
      hRL : Function.Injective ⇑(algebraMap R L)
      M : Submodule R S := Submodule.span R b
      n : Nat
      a : Fin n → HasQuotient.Quotient S M
      ha : Eq (Submodule.span R (Set.range a)) Top.top
      smul_top_eq : Eq (HSMul.hSMul p Top.top) Top.top
      exists_sum : ∀ (x : HasQuotient.Quotient S M), Exists fun a' => And (∀ (i : Fi …
      A' : Fin n → Fin n → R
      hA'p : ∀ (i i_1 : Fin n), Membership.mem p (A' i i_1)
      hA' : ∀ (i : Fin n), Eq (Finset.univ.sum fun i_1 => HSMul.hSMul (A' i i_1) (a  …
      A : Matrix (Fin n) (Fin n) R := HSub.hSub (Matrix.of A') 1
      B : Matrix (Fin n) (Fin n) R := A.adjugate
      A_smul : ∀ (i : Fin n), Eq (Finset.univ.sum fun j => HSMul.hSMul (A i j) (a j) …
      d_smul : ∀ (i : Fin n), Eq (HSMul.hSMul A.det (a i)) 0
      span_d : LE.le (Submodule.restrictScalars R (Submodule.span S (Singleton.singl …
      x : L
      hx : Membership.mem (Ideal.span (Singleton.singleton ((algebraMap S L) ((algeb …
      ⊢ Membership.mem (Submodule.span K (Set.image (⇑(algebraMap S L)) b)) x
    -/
    exact IsFractionRing.ideal_span_singleton_map_subset R hRL span_d hx
    /-
      🎉 no goals
    -/


/-- Let `V` be a vector space over `K = Frac(R)`, `S / R` a ring extension
and `V'` a module over `S`. If `b`, in the intersection `V''` of `V` and `V'`,
is linear independent over `S` in `V'`, then it is linear independent over `R` in `V`.

The statement we prove is actually slightly more general:
 * it suffices that the inclusion `algebraMap R S : R → S` is nontrivial
 * the function `f' : V'' → V'` doesn't need to be injective
-/
theorem FinrankQuotientMap.linearIndependent_of_nontrivial [IsDedekindDomain R]
    (hRS : RingHom.ker (algebraMap R S) ≠ ⊤) (f : V'' →ₗ[R] V) (hf : Function.Injective f)
    (f' : V'' →ₗ[R] V') {ι : Type*} {b : ι → V''} (hb' : LinearIndependent S (f' ∘ b)) :
    LinearIndependent K (f ∘ b) := by
  /-
    R : Type u
    inst✝¹⁵ : CommRing R
    S : Type v
    inst✝¹⁴ : CommRing S
    inst✝¹³ : Algebra R S
    K : Type u_1
    inst✝¹² : Field K
    inst✝¹¹ : Algebra R K
    V : Type u_3
    V' : Type u_4
    V'' : Type u_5
    inst✝¹⁰ : AddCommGroup V
    inst✝⁹ : Module R V
    inst✝⁸ : Module K V
    inst✝⁷ : IsScalarTower R K V
    inst✝⁶ : AddCommGroup V'
    inst✝⁵ : Module R V'
    inst✝⁴ : Module S V'
    inst✝³ : IsScalarTower R S V'
    inst✝² : AddCommGroup V''
    inst✝¹ : Module R V''
    hRK : IsFractionRing R K
    inst✝ : IsDedekindDomain R
    hRS : Ne (RingHom.ker (algebraMap R S)) Top.top
    f : LinearMap (RingHom.id R) V'' V
    hf : Function.Injective ⇑f
    f' : LinearMap (RingHom.id R) V'' V'
    ι : Type u_6
    b : ι → V''
    hb' : LinearIndependent S (Function.comp (⇑f') b)
    ⊢ LinearIndependent K (Function.comp (⇑f) b)
  -/
  contrapose! hb' with hb
  -- Informally, if we have a nontrivial linear dependence with coefficients `g` in `K`,
  -- then we can find a linear dependence with coefficients `I.Quotient.mk g'` in `R/I`,
  -- where `I = ker (algebraMap R S)`.
  -- We make use of the same principle but stay in `R` everywhere.
  /-
    R : Type u
    inst✝¹⁵ : CommRing R
    S : Type v
    inst✝¹⁴ : CommRing S
    inst✝¹³ : Algebra R S
    K : Type u_1
    inst✝¹² : Field K
    inst✝¹¹ : Algebra R K
    V : Type u_3
    V' : Type u_4
    V'' : Type u_5
    inst✝¹⁰ : AddCommGroup V
    inst✝⁹ : Module R V
    inst✝⁸ : Module K V
    inst✝⁷ : IsScalarTower R K V
    inst✝⁶ : AddCommGroup V'
    inst✝⁵ : Module R V'
    inst✝⁴ : Module S V'
    inst✝³ : IsScalarTower R S V'
    inst✝² : AddCommGroup V''
    inst✝¹ : Module R V''
    hRK : IsFractionRing R K
    inst✝ : IsDedekindDomain R
    hRS : Ne (RingHom.ker (algebraMap R S)) Top.top
    f : LinearMap (RingHom.id R) V'' V
    hf : Function.Injective ⇑f
    f' : LinearMap (RingHom.id R) V'' V'
    ι : Type u_6
    b : ι → V''
    hb : Not (LinearIndependent K (Function.comp (⇑f) b))
    ⊢ Not (LinearIndependent S (Function.comp (⇑f') b))
  -/
  simp only [linearIndependent_iff', not_forall] at hb ⊢
  /-
    R : Type u
    inst✝¹⁵ : CommRing R
    S : Type v
    inst✝¹⁴ : CommRing S
    inst✝¹³ : Algebra R S
    K : Type u_1
    inst✝¹² : Field K
    inst✝¹¹ : Algebra R K
    V : Type u_3
    V' : Type u_4
    V'' : Type u_5
    inst✝¹⁰ : AddCommGroup V
    inst✝⁹ : Module R V
    inst✝⁸ : Module K V
    inst✝⁷ : IsScalarTower R K V
    inst✝⁶ : AddCommGroup V'
    inst✝⁵ : Module R V'
    inst✝⁴ : Module S V'
    inst✝³ : IsScalarTower R S V'
    inst✝² : AddCommGroup V''
    inst✝¹ : Module R V''
    hRK : IsFractionRing R K
    inst✝ : IsDedekindDomain R
    hRS : Ne (RingHom.ker (algebraMap R S)) Top.top
    f : LinearMap (RingHom.id R) V'' V
    hf : Function.Injective ⇑f
    f' : LinearMap (RingHom.id R) V'' V'
    ι : Type u_6
    b : ι → V''
    hb : Exists fun x => Exists fun x_1 => Exists fun h => Exists fun x_2 => Exist …
    ⊢ Exists fun x => Exists fun x_1 => Exists fun h => Exists fun x_2 => Exists f …
  -/
  obtain ⟨s, g, eq, j', hj's, hj'g⟩ := hb
  /-
    case intro.intro.intro.intro.intro
    R : Type u
    inst✝¹⁵ : CommRing R
    S : Type v
    inst✝¹⁴ : CommRing S
    inst✝¹³ : Algebra R S
    K : Type u_1
    inst✝¹² : Field K
    inst✝¹¹ : Algebra R K
    V : Type u_3
    V' : Type u_4
    V'' : Type u_5
    inst✝¹⁰ : AddCommGroup V
    inst✝⁹ : Module R V
    inst✝⁸ : Module K V
    inst✝⁷ : IsScalarTower R K V
    inst✝⁶ : AddCommGroup V'
    inst✝⁵ : Module R V'
    inst✝⁴ : Module S V'
    inst✝³ : IsScalarTower R S V'
    inst✝² : AddCommGroup V''
    inst✝¹ : Module R V''
    hRK : IsFractionRing R K
    inst✝ : IsDedekindDomain R
    hRS : Ne (RingHom.ker (algebraMap R S)) Top.top
    f : LinearMap (RingHom.id R) V'' V
    hf : Function.Injective ⇑f
    f' : LinearMap (RingHom.id R) V'' V'
    ι : Type u_6
    b : ι → V''
    s : Finset ι
    g : ι → K
    eq : Eq (s.sum fun i => HSMul.hSMul (g i) (Function.comp (⇑f) b i)) 0
    j' : ι
    hj's : Membership.mem s j'
    hj'g : Not (Eq (g j') 0)
    ⊢ Exists fun x => Exists fun x_1 => Exists fun h => Exists fun x_2 => Exists f …
  -/
  use s
  /-
    case h
    R : Type u
    inst✝¹⁵ : CommRing R
    S : Type v
    inst✝¹⁴ : CommRing S
    inst✝¹³ : Algebra R S
    K : Type u_1
    inst✝¹² : Field K
    inst✝¹¹ : Algebra R K
    V : Type u_3
    V' : Type u_4
    V'' : Type u_5
    inst✝¹⁰ : AddCommGroup V
    inst✝⁹ : Module R V
    inst✝⁸ : Module K V
    inst✝⁷ : IsScalarTower R K V
    inst✝⁶ : AddCommGroup V'
    inst✝⁵ : Module R V'
    inst✝⁴ : Module S V'
    inst✝³ : IsScalarTower R S V'
    inst✝² : AddCommGroup V''
    inst✝¹ : Module R V''
    hRK : IsFractionRing R K
    inst✝ : IsDedekindDomain R
    hRS : Ne (RingHom.ker (algebraMap R S)) Top.top
    f : LinearMap (RingHom.id R) V'' V
    hf : Function.Injective ⇑f
    f' : LinearMap (RingHom.id R) V'' V'
    ι : Type u_6
    b : ι → V''
    s : Finset ι
    g : ι → K
    eq : Eq (s.sum fun i => HSMul.hSMul (g i) (Function.comp (⇑f) b i)) 0
    j' : ι
    hj's : Membership.mem s j'
    hj'g : Not (Eq (g j') 0)
    ⊢ Exists fun x => Exists fun h => Exists fun x_1 => Exists fun x_2 => Not (Eq  …
  -/
  obtain ⟨a, hag, j, hjs, hgI⟩ := Ideal.exist_integer_multiples_not_mem hRS s g hj's hj'g
  /-
    case h.intro.intro.intro.intro
    R : Type u
    inst✝¹⁵ : CommRing R
    S : Type v
    inst✝¹⁴ : CommRing S
    inst✝¹³ : Algebra R S
    K : Type u_1
    inst✝¹² : Field K
    inst✝¹¹ : Algebra R K
    V : Type u_3
    V' : Type u_4
    V'' : Type u_5
    inst✝¹⁰ : AddCommGroup V
    inst✝⁹ : Module R V
    inst✝⁸ : Module K V
    inst✝⁷ : IsScalarTower R K V
    inst✝⁶ : AddCommGroup V'
    inst✝⁵ : Module R V'
    inst✝⁴ : Module S V'
    inst✝³ : IsScalarTower R S V'
    inst✝² : AddCommGroup V''
    inst✝¹ : Module R V''
    hRK : IsFractionRing R K
    inst✝ : IsDedekindDomain R
    hRS : Ne (RingHom.ker (algebraMap R S)) Top.top
    f : LinearMap (RingHom.id R) V'' V
    hf : Function.Injective ⇑f
    f' : LinearMap (RingHom.id R) V'' V'
    ι : Type u_6
    b : ι → V''
    s : Finset ι
    g : ι → K
    eq : Eq (s.sum fun i => HSMul.hSMul (g i) (Function.comp (⇑f) b i)) 0
    j' : ι
    hj's : Membership.mem s j'
    hj'g : Not (Eq (g j') 0)
    a : K
    hag : ∀ (i : ι), Membership.mem s i → IsLocalization.IsInteger R (HMul.hMul a  …
    j : ι
    hjs : Membership.mem s j
    hgI : Not (Membership.mem (↑(RingHom.ker (algebraMap R S))) (HMul.hMul a (g j)))
    ⊢ Exists fun x => Exists fun h => Exists fun x_1 => Exists fun x_2 => Not (Eq  …
  -/
  choose g'' hg'' using hag
  /-
    case h.intro.intro.intro.intro
    R : Type u
    inst✝¹⁵ : CommRing R
    S : Type v
    inst✝¹⁴ : CommRing S
    inst✝¹³ : Algebra R S
    K : Type u_1
    inst✝¹² : Field K
    inst✝¹¹ : Algebra R K
    V : Type u_3
    V' : Type u_4
    V'' : Type u_5
    inst✝¹⁰ : AddCommGroup V
    inst✝⁹ : Module R V
    inst✝⁸ : Module K V
    inst✝⁷ : IsScalarTower R K V
    inst✝⁶ : AddCommGroup V'
    inst✝⁵ : Module R V'
    inst✝⁴ : Module S V'
    inst✝³ : IsScalarTower R S V'
    inst✝² : AddCommGroup V''
    inst✝¹ : Module R V''
    hRK : IsFractionRing R K
    inst✝ : IsDedekindDomain R
    hRS : Ne (RingHom.ker (algebraMap R S)) Top.top
    f : LinearMap (RingHom.id R) V'' V
    hf : Function.Injective ⇑f
    f' : LinearMap (RingHom.id R) V'' V'
    ι : Type u_6
    b : ι → V''
    s : Finset ι
    g : ι → K
    eq : Eq (s.sum fun i => HSMul.hSMul (g i) (Function.comp (⇑f) b i)) 0
    j' : ι
    hj's : Membership.mem s j'
    hj'g : Not (Eq (g j') 0)
    a : K
    j : ι
    hjs : Membership.mem s j
    hgI : Not (Membership.mem (↑(RingHom.ker (algebraMap R S))) (HMul.hMul a (g j)))
    g'' : (i : ι) → Membership.mem s i → R
    hg'' : ∀ (i : ι) (a_1 : Membership.mem s i), Eq ((algebraMap R K) (g'' i a_1)) …
    ⊢ Exists fun x => Exists fun h => Exists fun x_1 => Exists fun x_2 => Not (Eq  …
  -/
  letI := Classical.propDecidable
  /-
    case h.intro.intro.intro.intro
    R : Type u
    inst✝¹⁵ : CommRing R
    S : Type v
    inst✝¹⁴ : CommRing S
    inst✝¹³ : Algebra R S
    K : Type u_1
    inst✝¹² : Field K
    inst✝¹¹ : Algebra R K
    V : Type u_3
    V' : Type u_4
    V'' : Type u_5
    inst✝¹⁰ : AddCommGroup V
    inst✝⁹ : Module R V
    inst✝⁸ : Module K V
    inst✝⁷ : IsScalarTower R K V
    inst✝⁶ : AddCommGroup V'
    inst✝⁵ : Module R V'
    inst✝⁴ : Module S V'
    inst✝³ : IsScalarTower R S V'
    inst✝² : AddCommGroup V''
    inst✝¹ : Module R V''
    hRK : IsFractionRing R K
    inst✝ : IsDedekindDomain R
    hRS : Ne (RingHom.ker (algebraMap R S)) Top.top
    f : LinearMap (RingHom.id R) V'' V
    hf : Function.Injective ⇑f
    f' : LinearMap (RingHom.id R) V'' V'
    ι : Type u_6
    b : ι → V''
    s : Finset ι
    g : ι → K
    eq : Eq (s.sum fun i => HSMul.hSMul (g i) (Function.comp (⇑f) b i)) 0
    j' : ι
    hj's : Membership.mem s j'
    hj'g : Not (Eq (g j') 0)
    a : K
    j : ι
    hjs : Membership.mem s j
    hgI : Not (Membership.mem (↑(RingHom.ker (algebraMap R S))) (HMul.hMul a (g j)))
    g'' : (i : ι) → Membership.mem s i → R
    hg'' : ∀ (i : ι) (a_1 : Membership.mem s i), Eq ((algebraMap R K) (g'' i a_1)) …
    this : (a : Prop) → Decidable a := Classical.propDecidable
    ⊢ Exists fun x => Exists fun h => Exists fun x_1 => Exists fun x_2 => Not (Eq  …
  -/
  let g' i := if h : i ∈ s then g'' i h else 0
  have hg' : ∀ i ∈ s, algebraMap _ _ (g' i) = a * g i := by
    intro i hi; exact (congr_arg _ (dif_pos hi)).trans (hg'' i hi)
  -- Because `R/I` is nontrivial, we can lift `g` to a nontrivial linear dependence in `S`.
  have hgI : algebraMap R S (g' j) ≠ 0 := by
    simp only [FractionalIdeal.mem_coeIdeal, not_exists, not_and'] at hgI
    exact hgI _ (hg' j hjs)
  /-
    case h.intro.intro.intro.intro
    R : Type u
    inst✝¹⁵ : CommRing R
    S : Type v
    inst✝¹⁴ : CommRing S
    inst✝¹³ : Algebra R S
    K : Type u_1
    inst✝¹² : Field K
    inst✝¹¹ : Algebra R K
    V : Type u_3
    V' : Type u_4
    V'' : Type u_5
    inst✝¹⁰ : AddCommGroup V
    inst✝⁹ : Module R V
    inst✝⁸ : Module K V
    inst✝⁷ : IsScalarTower R K V
    inst✝⁶ : AddCommGroup V'
    inst✝⁵ : Module R V'
    inst✝⁴ : Module S V'
    inst✝³ : IsScalarTower R S V'
    inst✝² : AddCommGroup V''
    inst✝¹ : Module R V''
    hRK : IsFractionRing R K
    inst✝ : IsDedekindDomain R
    hRS : Ne (RingHom.ker (algebraMap R S)) Top.top
    f : LinearMap (RingHom.id R) V'' V
    hf : Function.Injective ⇑f
    f' : LinearMap (RingHom.id R) V'' V'
    ι : Type u_6
    b : ι → V''
    s : Finset ι
    g : ι → K
    eq : Eq (s.sum fun i => HSMul.hSMul (g i) (Function.comp (⇑f) b i)) 0
    j' : ι
    hj's : Membership.mem s j'
    hj'g : Not (Eq (g j') 0)
    a : K
    j : ι
    hjs : Membership.mem s j
    hgI✝ : Not (Membership.mem (↑(RingHom.ker (algebraMap R S))) (HMul.hMul a (g j …
    g'' : (i : ι) → Membership.mem s i → R
    hg'' : ∀ (i : ι) (a_1 : Membership.mem s i), Eq ((algebraMap R K) (g'' i a_1)) …
    this : (a : Prop) → Decidable a := Classical.propDecidable
    g' : ι → R := fun i => dite (Membership.mem s i) (fun h => g'' i h) fun h => 0
    hg' : ∀ (i : ι), Membership.mem s i → Eq ((algebraMap R K) (g' i)) (HMul.hMul  …
    hgI : Ne ((algebraMap R S) (g' j)) 0
    ⊢ Exists fun x => Exists fun h => Exists fun x_1 => Exists fun x_2 => Not (Eq  …
  -/
  refine ⟨fun i => algebraMap R S (g' i), ?_, j, hjs, hgI⟩
  have eq : f (∑ i ∈ s, g' i • b i) = 0 := by
    rw [map_sum, ← smul_zero a, ← eq, Finset.smul_sum]
    refine Finset.sum_congr rfl ?_
    intro i hi
    rw [LinearMap.map_smul, ← IsScalarTower.algebraMap_smul K, hg' i hi, ← smul_assoc,
      smul_eq_mul, Function.comp_apply]
  simp only [IsScalarTower.algebraMap_smul, ← map_smul, ← map_sum,
    (f.map_eq_zero_iff hf).mp eq, LinearMap.map_zero, (· ∘ ·)]


/-- If `p` is a maximal ideal of `R`, and `S` is the integral closure of `R` in `L`,
then the dimension `[S/pS : R/p]` is equal to `[Frac(S) : Frac(R)]`. -/
theorem finrank_quotient_map [IsDomain S] [IsDedekindDomain R] [Algebra K L]
    [Algebra R L] [IsScalarTower R K L] [IsScalarTower R S L]
    [hp : p.IsMaximal] [Module.Finite R S] :
    finrank (R ⧸ p) (S ⧸ map (algebraMap R S) p) = finrank K L := by
  -- Choose an arbitrary basis `b` for `[S/pS : R/p]`.
  -- We'll use the previous results to turn it into a basis on `[Frac(S) : Frac(R)]`.
  /-
    R : Type u
    inst✝¹⁴ : CommRing R
    S : Type v
    inst✝¹³ : CommRing S
    p : Ideal R
    inst✝¹² : Algebra R S
    K : Type u_1
    inst✝¹¹ : Field K
    inst✝¹⁰ : Algebra R K
    L : Type u_2
    inst✝⁹ : Field L
    inst✝⁸ : Algebra S L
    inst✝⁷ : IsFractionRing S L
    hRK : IsFractionRing R K
    inst✝⁶ : IsDomain S
    inst✝⁵ : IsDedekindDomain R
    inst✝⁴ : Algebra K L
    inst✝³ : Algebra R L
    inst✝² : IsScalarTower R K L
    inst✝¹ : IsScalarTower R S L
    hp : p.IsMaximal
    inst✝ : Module.Finite R S
    ⊢ Eq (Module.finrank (HasQuotient.Quotient R p) (HasQuotient.Quotient S (Ideal …
  -/
  let ι := Module.Free.ChooseBasisIndex (R ⧸ p) (S ⧸ map (algebraMap R S) p)
  /-
    R : Type u
    inst✝¹⁴ : CommRing R
    S : Type v
    inst✝¹³ : CommRing S
    p : Ideal R
    inst✝¹² : Algebra R S
    K : Type u_1
    inst✝¹¹ : Field K
    inst✝¹⁰ : Algebra R K
    L : Type u_2
    inst✝⁹ : Field L
    inst✝⁸ : Algebra S L
    inst✝⁷ : IsFractionRing S L
    hRK : IsFractionRing R K
    inst✝⁶ : IsDomain S
    inst✝⁵ : IsDedekindDomain R
    inst✝⁴ : Algebra K L
    inst✝³ : Algebra R L
    inst✝² : IsScalarTower R K L
    inst✝¹ : IsScalarTower R S L
    hp : p.IsMaximal
    inst✝ : Module.Finite R S
    ι : Type v := Module.Free.ChooseBasisIndex (HasQuotient.Quotient R p) (HasQuot …
    ⊢ Eq (Module.finrank (HasQuotient.Quotient R p) (HasQuotient.Quotient S (Ideal …
  -/
  let b : Basis ι (R ⧸ p) (S ⧸ map (algebraMap R S) p) := Module.Free.chooseBasis _ _
  -- Namely, choose a representative `b' i : S` for each `b i : S / pS`.
  /-
    R : Type u
    inst✝¹⁴ : CommRing R
    S : Type v
    inst✝¹³ : CommRing S
    p : Ideal R
    inst✝¹² : Algebra R S
    K : Type u_1
    inst✝¹¹ : Field K
    inst✝¹⁰ : Algebra R K
    L : Type u_2
    inst✝⁹ : Field L
    inst✝⁸ : Algebra S L
    inst✝⁷ : IsFractionRing S L
    hRK : IsFractionRing R K
    inst✝⁶ : IsDomain S
    inst✝⁵ : IsDedekindDomain R
    inst✝⁴ : Algebra K L
    inst✝³ : Algebra R L
    inst✝² : IsScalarTower R K L
    inst✝¹ : IsScalarTower R S L
    hp : p.IsMaximal
    inst✝ : Module.Finite R S
    ι : Type v := Module.Free.ChooseBasisIndex (HasQuotient.Quotient R p) (HasQuot …
    b : Basis ι (HasQuotient.Quotient R p) (HasQuotient.Quotient S (Ideal.map (alg …
    ⊢ Eq (Module.finrank (HasQuotient.Quotient R p) (HasQuotient.Quotient S (Ideal …
  -/
  let b' : ι → S := fun i => (Ideal.Quotient.mk_surjective (b i)).choose
  have b_eq_b' : ⇑b = (Submodule.mkQ (map (algebraMap R S) p)).restrictScalars R ∘ b' :=
    funext fun i => (Ideal.Quotient.mk_surjective (b i)).choose_spec.symm
  -- We claim `b'` is a basis for `Frac(S)` over `Frac(R)` because it is linear independent
  -- and spans the whole of `Frac(S)`.
  /-
    R : Type u
    inst✝¹⁴ : CommRing R
    S : Type v
    inst✝¹³ : CommRing S
    p : Ideal R
    inst✝¹² : Algebra R S
    K : Type u_1
    inst✝¹¹ : Field K
    inst✝¹⁰ : Algebra R K
    L : Type u_2
    inst✝⁹ : Field L
    inst✝⁸ : Algebra S L
    inst✝⁷ : IsFractionRing S L
    hRK : IsFractionRing R K
    inst✝⁶ : IsDomain S
    inst✝⁵ : IsDedekindDomain R
    inst✝⁴ : Algebra K L
    inst✝³ : Algebra R L
    inst✝² : IsScalarTower R K L
    inst✝¹ : IsScalarTower R S L
    hp : p.IsMaximal
    inst✝ : Module.Finite R S
    ι : Type v := Module.Free.ChooseBasisIndex (HasQuotient.Quotient R p) (HasQuot …
    b : Basis ι (HasQuotient.Quotient R p) (HasQuotient.Quotient S (Ideal.map (alg …
    b' : ι → S := fun i => ⋯.choose
    b_eq_b' : Eq (⇑b) (Function.comp (⇑(↑R (Submodule.mkQ (Ideal.map (algebraMap R …
    ⊢ Eq (Module.finrank (HasQuotient.Quotient R p) (HasQuotient.Quotient S (Ideal …
  -/
  let b'' : ι → L := algebraMap S L ∘ b'
  /-
    R : Type u
    inst✝¹⁴ : CommRing R
    S : Type v
    inst✝¹³ : CommRing S
    p : Ideal R
    inst✝¹² : Algebra R S
    K : Type u_1
    inst✝¹¹ : Field K
    inst✝¹⁰ : Algebra R K
    L : Type u_2
    inst✝⁹ : Field L
    inst✝⁸ : Algebra S L
    inst✝⁷ : IsFractionRing S L
    hRK : IsFractionRing R K
    inst✝⁶ : IsDomain S
    inst✝⁵ : IsDedekindDomain R
    inst✝⁴ : Algebra K L
    inst✝³ : Algebra R L
    inst✝² : IsScalarTower R K L
    inst✝¹ : IsScalarTower R S L
    hp : p.IsMaximal
    inst✝ : Module.Finite R S
    ι : Type v := Module.Free.ChooseBasisIndex (HasQuotient.Quotient R p) (HasQuot …
    b : Basis ι (HasQuotient.Quotient R p) (HasQuotient.Quotient S (Ideal.map (alg …
    b' : ι → S := fun i => ⋯.choose
    b_eq_b' : Eq (⇑b) (Function.comp (⇑(↑R (Submodule.mkQ (Ideal.map (algebraMap R …
    b'' : ι → L := Function.comp (⇑(algebraMap S L)) b'
    ⊢ Eq (Module.finrank (HasQuotient.Quotient R p) (HasQuotient.Quotient S (Ideal …
  -/
  have b''_li : LinearIndependent K b'' := ?_
    /-
      case refine_2
      R : Type u
      inst✝¹⁴ : CommRing R
      S : Type v
      inst✝¹³ : CommRing S
      p : Ideal R
      inst✝¹² : Algebra R S
      K : Type u_1
      inst✝¹¹ : Field K
      inst✝¹⁰ : Algebra R K
      L : Type u_2
      inst✝⁹ : Field L
      inst✝⁸ : Algebra S L
      inst✝⁷ : IsFractionRing S L
      hRK : IsFractionRing R K
      inst✝⁶ : IsDomain S
      inst✝⁵ : IsDedekindDomain R
      inst✝⁴ : Algebra K L
      inst✝³ : Algebra R L
      inst✝² : IsScalarTower R K L
      inst✝¹ : IsScalarTower R S L
      hp : p.IsMaximal
      inst✝ : Module.Finite R S
      ι : Type v := Module.Free.ChooseBasisIndex (HasQuotient.Quotient R p) (HasQuot …
      b : Basis ι (HasQuotient.Quotient R p) (HasQuotient.Quotient S (Ideal.map (alg …
      b' : ι → S := fun i => ⋯.choose
      b_eq_b' : Eq (⇑b) (Function.comp (⇑(↑R (Submodule.mkQ (Ideal.map (algebraMap R …
      b'' : ι → L := Function.comp (⇑(algebraMap S L)) b'
      b''_li : LinearIndependent K b''
      ⊢ Eq (Module.finrank (HasQuotient.Quotient R p) (HasQuotient.Quotient S (Ideal …
    -/
  · have b''_sp : Submodule.span K (Set.range b'') = ⊤ := ?_
    -- Since the two bases have the same index set, the spaces have the same dimension.
      /-
        case refine_2.refine_2
        R : Type u
        inst✝¹⁴ : CommRing R
        S : Type v
        inst✝¹³ : CommRing S
        p : Ideal R
        inst✝¹² : Algebra R S
        K : Type u_1
        inst✝¹¹ : Field K
        inst✝¹⁰ : Algebra R K
        L : Type u_2
        inst✝⁹ : Field L
        inst✝⁸ : Algebra S L
        inst✝⁷ : IsFractionRing S L
        hRK : IsFractionRing R K
        inst✝⁶ : IsDomain S
        inst✝⁵ : IsDedekindDomain R
        inst✝⁴ : Algebra K L
        inst✝³ : Algebra R L
        inst✝² : IsScalarTower R K L
        inst✝¹ : IsScalarTower R S L
        hp : p.IsMaximal
        inst✝ : Module.Finite R S
        ι : Type v := Module.Free.ChooseBasisIndex (HasQuotient.Quotient R p) (HasQuot …
        b : Basis ι (HasQuotient.Quotient R p) (HasQuotient.Quotient S (Ideal.map (alg …
        b' : ι → S := fun i => ⋯.choose
        b_eq_b' : Eq (⇑b) (Function.comp (⇑(↑R (Submodule.mkQ (Ideal.map (algebraMap R …
        b'' : ι → L := Function.comp (⇑(algebraMap S L)) b'
        b''_li : LinearIndependent K b''
        b''_sp : Eq (Submodule.span K (Set.range b'')) Top.top
        ⊢ Eq (Module.finrank (HasQuotient.Quotient R p) (HasQuotient.Quotient S (Ideal …
      -/
    · let c : Basis ι K L := Basis.mk b''_li b''_sp.ge
      /-
        case refine_2.refine_2
        R : Type u
        inst✝¹⁴ : CommRing R
        S : Type v
        inst✝¹³ : CommRing S
        p : Ideal R
        inst✝¹² : Algebra R S
        K : Type u_1
        inst✝¹¹ : Field K
        inst✝¹⁰ : Algebra R K
        L : Type u_2
        inst✝⁹ : Field L
        inst✝⁸ : Algebra S L
        inst✝⁷ : IsFractionRing S L
        hRK : IsFractionRing R K
        inst✝⁶ : IsDomain S
        inst✝⁵ : IsDedekindDomain R
        inst✝⁴ : Algebra K L
        inst✝³ : Algebra R L
        inst✝² : IsScalarTower R K L
        inst✝¹ : IsScalarTower R S L
        hp : p.IsMaximal
        inst✝ : Module.Finite R S
        ι : Type v := Module.Free.ChooseBasisIndex (HasQuotient.Quotient R p) (HasQuot …
        b : Basis ι (HasQuotient.Quotient R p) (HasQuotient.Quotient S (Ideal.map (alg …
        b' : ι → S := fun i => ⋯.choose
        b_eq_b' : Eq (⇑b) (Function.comp (⇑(↑R (Submodule.mkQ (Ideal.map (algebraMap R …
        b'' : ι → L := Function.comp (⇑(algebraMap S L)) b'
        b''_li : LinearIndependent K b''
        b''_sp : Eq (Submodule.span K (Set.range b'')) Top.top
        c : Basis ι K L := Basis.mk b''_li ⋯
        ⊢ Eq (Module.finrank (HasQuotient.Quotient R p) (HasQuotient.Quotient S (Ideal …
      -/
      rw [finrank_eq_card_basis b, finrank_eq_card_basis c]
      /-
        🎉 no goals
      -/
    -- It remains to show that the basis is indeed linear independent and spans the whole space.
      /-
        case refine_2.refine_1
        R : Type u
        inst✝¹⁴ : CommRing R
        S : Type v
        inst✝¹³ : CommRing S
        p : Ideal R
        inst✝¹² : Algebra R S
        K : Type u_1
        inst✝¹¹ : Field K
        inst✝¹⁰ : Algebra R K
        L : Type u_2
        inst✝⁹ : Field L
        inst✝⁸ : Algebra S L
        inst✝⁷ : IsFractionRing S L
        hRK : IsFractionRing R K
        inst✝⁶ : IsDomain S
        inst✝⁵ : IsDedekindDomain R
        inst✝⁴ : Algebra K L
        inst✝³ : Algebra R L
        inst✝² : IsScalarTower R K L
        inst✝¹ : IsScalarTower R S L
        hp : p.IsMaximal
        inst✝ : Module.Finite R S
        ι : Type v := Module.Free.ChooseBasisIndex (HasQuotient.Quotient R p) (HasQuot …
        b : Basis ι (HasQuotient.Quotient R p) (HasQuotient.Quotient S (Ideal.map (alg …
        b' : ι → S := fun i => ⋯.choose
        b_eq_b' : Eq (⇑b) (Function.comp (⇑(↑R (Submodule.mkQ (Ideal.map (algebraMap R …
        b'' : ι → L := Function.comp (⇑(algebraMap S L)) b'
        b''_li : LinearIndependent K b''
        ⊢ Eq (Submodule.span K (Set.range b'')) Top.top
      -/
    · rw [Set.range_comp]
      /-
        case refine_2.refine_1
        R : Type u
        inst✝¹⁴ : CommRing R
        S : Type v
        inst✝¹³ : CommRing S
        p : Ideal R
        inst✝¹² : Algebra R S
        K : Type u_1
        inst✝¹¹ : Field K
        inst✝¹⁰ : Algebra R K
        L : Type u_2
        inst✝⁹ : Field L
        inst✝⁸ : Algebra S L
        inst✝⁷ : IsFractionRing S L
        hRK : IsFractionRing R K
        inst✝⁶ : IsDomain S
        inst✝⁵ : IsDedekindDomain R
        inst✝⁴ : Algebra K L
        inst✝³ : Algebra R L
        inst✝² : IsScalarTower R K L
        inst✝¹ : IsScalarTower R S L
        hp : p.IsMaximal
        inst✝ : Module.Finite R S
        ι : Type v := Module.Free.ChooseBasisIndex (HasQuotient.Quotient R p) (HasQuot …
        b : Basis ι (HasQuotient.Quotient R p) (HasQuotient.Quotient S (Ideal.map (alg …
        b' : ι → S := fun i => ⋯.choose
        b_eq_b' : Eq (⇑b) (Function.comp (⇑(↑R (Submodule.mkQ (Ideal.map (algebraMap R …
        b'' : ι → L := Function.comp (⇑(algebraMap S L)) b'
        b''_li : LinearIndependent K b''
        ⊢ Eq (Submodule.span K (Set.image (⇑(algebraMap S L)) (Set.range b'))) Top.top
      -/
      refine FinrankQuotientMap.span_eq_top p hp.ne_top _ (top_le_iff.mp ?_)
      -- The nicest way to show `S ≤ span b' ⊔ pS` is by reducing both sides modulo pS.
      -- However, this would imply distinguishing between `pS` as `S`-ideal,
      -- and `pS` as `R`-submodule, since they have different (non-defeq) quotients.
      -- Instead we'll lift `x mod pS ∈ span b` to `y ∈ span b'` for some `y - x ∈ pS`.
      /-
        case refine_2.refine_1
        R : Type u
        inst✝¹⁴ : CommRing R
        S : Type v
        inst✝¹³ : CommRing S
        p : Ideal R
        inst✝¹² : Algebra R S
        K : Type u_1
        inst✝¹¹ : Field K
        inst✝¹⁰ : Algebra R K
        L : Type u_2
        inst✝⁹ : Field L
        inst✝⁸ : Algebra S L
        inst✝⁷ : IsFractionRing S L
        hRK : IsFractionRing R K
        inst✝⁶ : IsDomain S
        inst✝⁵ : IsDedekindDomain R
        inst✝⁴ : Algebra K L
        inst✝³ : Algebra R L
        inst✝² : IsScalarTower R K L
        inst✝¹ : IsScalarTower R S L
        hp : p.IsMaximal
        inst✝ : Module.Finite R S
        ι : Type v := Module.Free.ChooseBasisIndex (HasQuotient.Quotient R p) (HasQuot …
        b : Basis ι (HasQuotient.Quotient R p) (HasQuotient.Quotient S (Ideal.map (alg …
        b' : ι → S := fun i => ⋯.choose
        b_eq_b' : Eq (⇑b) (Function.comp (⇑(↑R (Submodule.mkQ (Ideal.map (algebraMap R …
        b'' : ι → L := Function.comp (⇑(algebraMap S L)) b'
        b''_li : LinearIndependent K b''
        ⊢ LE.le Top.top (Max.max (Submodule.span R (Set.range b')) (Submodule.restrict …
      -/
      intro x _
      have mem_span_b : ((Submodule.mkQ (map (algebraMap R S) p)) x : S ⧸ map (algebraMap R S) p) ∈
          Submodule.span (R ⧸ p) (Set.range b) := b.mem_span _
      rw [← @Submodule.restrictScalars_mem R,
        Submodule.restrictScalars_span R (R ⧸ p) Ideal.Quotient.mk_surjective, b_eq_b',
        Set.range_comp, ← Submodule.map_span] at mem_span_b
      /-
        case refine_2.refine_1
        R : Type u
        inst✝¹⁴ : CommRing R
        S : Type v
        inst✝¹³ : CommRing S
        p : Ideal R
        inst✝¹² : Algebra R S
        K : Type u_1
        inst✝¹¹ : Field K
        inst✝¹⁰ : Algebra R K
        L : Type u_2
        inst✝⁹ : Field L
        inst✝⁸ : Algebra S L
        inst✝⁷ : IsFractionRing S L
        hRK : IsFractionRing R K
        inst✝⁶ : IsDomain S
        inst✝⁵ : IsDedekindDomain R
        inst✝⁴ : Algebra K L
        inst✝³ : Algebra R L
        inst✝² : IsScalarTower R K L
        inst✝¹ : IsScalarTower R S L
        hp : p.IsMaximal
        inst✝ : Module.Finite R S
        ι : Type v := Module.Free.ChooseBasisIndex (HasQuotient.Quotient R p) (HasQuot …
        b : Basis ι (HasQuotient.Quotient R p) (HasQuotient.Quotient S (Ideal.map (alg …
        b' : ι → S := fun i => ⋯.choose
        b_eq_b' : Eq (⇑b) (Function.comp (⇑(↑R (Submodule.mkQ (Ideal.map (algebraMap R …
        b'' : ι → L := Function.comp (⇑(algebraMap S L)) b'
        b''_li : LinearIndependent K b''
        x : S
        a✝ : Membership.mem Top.top x
        mem_span_b : Membership.mem (Submodule.map (↑R (Submodule.mkQ (Ideal.map (alge …
        ⊢ Membership.mem (Max.max (Submodule.span R (Set.range b')) (Submodule.restric …
      -/
      obtain ⟨y, y_mem, y_eq⟩ := Submodule.mem_map.mp mem_span_b
      /-
        case refine_2.refine_1.intro.intro
        R : Type u
        inst✝¹⁴ : CommRing R
        S : Type v
        inst✝¹³ : CommRing S
        p : Ideal R
        inst✝¹² : Algebra R S
        K : Type u_1
        inst✝¹¹ : Field K
        inst✝¹⁰ : Algebra R K
        L : Type u_2
        inst✝⁹ : Field L
        inst✝⁸ : Algebra S L
        inst✝⁷ : IsFractionRing S L
        hRK : IsFractionRing R K
        inst✝⁶ : IsDomain S
        inst✝⁵ : IsDedekindDomain R
        inst✝⁴ : Algebra K L
        inst✝³ : Algebra R L
        inst✝² : IsScalarTower R K L
        inst✝¹ : IsScalarTower R S L
        hp : p.IsMaximal
        inst✝ : Module.Finite R S
        ι : Type v := Module.Free.ChooseBasisIndex (HasQuotient.Quotient R p) (HasQuot …
        b : Basis ι (HasQuotient.Quotient R p) (HasQuotient.Quotient S (Ideal.map (alg …
        b' : ι → S := fun i => ⋯.choose
        b_eq_b' : Eq (⇑b) (Function.comp (⇑(↑R (Submodule.mkQ (Ideal.map (algebraMap R …
        b'' : ι → L := Function.comp (⇑(algebraMap S L)) b'
        b''_li : LinearIndependent K b''
        x : S
        a✝ : Membership.mem Top.top x
        mem_span_b : Membership.mem (Submodule.map (↑R (Submodule.mkQ (Ideal.map (alge …
        y : S
        y_mem : Membership.mem (Submodule.span R (Set.range b')) y
        y_eq : Eq ((↑R (Submodule.mkQ (Ideal.map (algebraMap R S) p))) y) ((Submodule. …
        ⊢ Membership.mem (Max.max (Submodule.span R (Set.range b')) (Submodule.restric …
      -/
      suffices y + -(y - x) ∈ _ by simpa
      rw [LinearMap.restrictScalars_apply, Submodule.mkQ_apply, Submodule.mkQ_apply,
        Submodule.Quotient.eq] at y_eq
      /-
        case refine_2.refine_1.intro.intro
        R : Type u
        inst✝¹⁴ : CommRing R
        S : Type v
        inst✝¹³ : CommRing S
        p : Ideal R
        inst✝¹² : Algebra R S
        K : Type u_1
        inst✝¹¹ : Field K
        inst✝¹⁰ : Algebra R K
        L : Type u_2
        inst✝⁹ : Field L
        inst✝⁸ : Algebra S L
        inst✝⁷ : IsFractionRing S L
        hRK : IsFractionRing R K
        inst✝⁶ : IsDomain S
        inst✝⁵ : IsDedekindDomain R
        inst✝⁴ : Algebra K L
        inst✝³ : Algebra R L
        inst✝² : IsScalarTower R K L
        inst✝¹ : IsScalarTower R S L
        hp : p.IsMaximal
        inst✝ : Module.Finite R S
        ι : Type v := Module.Free.ChooseBasisIndex (HasQuotient.Quotient R p) (HasQuot …
        b : Basis ι (HasQuotient.Quotient R p) (HasQuotient.Quotient S (Ideal.map (alg …
        b' : ι → S := fun i => ⋯.choose
        b_eq_b' : Eq (⇑b) (Function.comp (⇑(↑R (Submodule.mkQ (Ideal.map (algebraMap R …
        b'' : ι → L := Function.comp (⇑(algebraMap S L)) b'
        b''_li : LinearIndependent K b''
        x : S
        a✝ : Membership.mem Top.top x
        mem_span_b : Membership.mem (Submodule.map (↑R (Submodule.mkQ (Ideal.map (alge …
        y : S
        y_mem : Membership.mem (Submodule.span R (Set.range b')) y
        y_eq : Membership.mem (Ideal.map (algebraMap R S) p) (HSub.hSub y x)
        ⊢ Membership.mem (Max.max (Submodule.span R (Set.range b')) (Submodule.restric …
      -/
      exact add_mem (Submodule.mem_sup_left y_mem) (neg_mem <| Submodule.mem_sup_right y_eq)
      /-
        🎉 no goals
      -/
    /-
      case refine_1
      R : Type u
      inst✝¹⁴ : CommRing R
      S : Type v
      inst✝¹³ : CommRing S
      p : Ideal R
      inst✝¹² : Algebra R S
      K : Type u_1
      inst✝¹¹ : Field K
      inst✝¹⁰ : Algebra R K
      L : Type u_2
      inst✝⁹ : Field L
      inst✝⁸ : Algebra S L
      inst✝⁷ : IsFractionRing S L
      hRK : IsFractionRing R K
      inst✝⁶ : IsDomain S
      inst✝⁵ : IsDedekindDomain R
      inst✝⁴ : Algebra K L
      inst✝³ : Algebra R L
      inst✝² : IsScalarTower R K L
      inst✝¹ : IsScalarTower R S L
      hp : p.IsMaximal
      inst✝ : Module.Finite R S
      ι : Type v := Module.Free.ChooseBasisIndex (HasQuotient.Quotient R p) (HasQuot …
      b : Basis ι (HasQuotient.Quotient R p) (HasQuotient.Quotient S (Ideal.map (alg …
      b' : ι → S := fun i => ⋯.choose
      b_eq_b' : Eq (⇑b) (Function.comp (⇑(↑R (Submodule.mkQ (Ideal.map (algebraMap R …
      b'' : ι → L := Function.comp (⇑(algebraMap S L)) b'
      ⊢ LinearIndependent K b''
    -/
  · have := b.linearIndependent; rw [b_eq_b'] at this
    convert FinrankQuotientMap.linearIndependent_of_nontrivial K _
        ((Algebra.linearMap S L).restrictScalars R) _ ((Submodule.mkQ _).restrictScalars R) this
      /-
        case refine_1.convert_1
        R : Type u
        inst✝¹⁴ : CommRing R
        S : Type v
        inst✝¹³ : CommRing S
        p : Ideal R
        inst✝¹² : Algebra R S
        K : Type u_1
        inst✝¹¹ : Field K
        inst✝¹⁰ : Algebra R K
        L : Type u_2
        inst✝⁹ : Field L
        inst✝⁸ : Algebra S L
        inst✝⁷ : IsFractionRing S L
        hRK : IsFractionRing R K
        inst✝⁶ : IsDomain S
        inst✝⁵ : IsDedekindDomain R
        inst✝⁴ : Algebra K L
        inst✝³ : Algebra R L
        inst✝² : IsScalarTower R K L
        inst✝¹ : IsScalarTower R S L
        hp : p.IsMaximal
        inst✝ : Module.Finite R S
        ι : Type v := Module.Free.ChooseBasisIndex (HasQuotient.Quotient R p) (HasQuot …
        b : Basis ι (HasQuotient.Quotient R p) (HasQuotient.Quotient S (Ideal.map (alg …
        b' : ι → S := fun i => ⋯.choose
        b_eq_b' : Eq (⇑b) (Function.comp (⇑(↑R (Submodule.mkQ (Ideal.map (algebraMap R …
        b'' : ι → L := Function.comp (⇑(algebraMap S L)) b'
        this : LinearIndependent (HasQuotient.Quotient R p) (Function.comp (⇑(↑R (Subm …
        ⊢ Ne (RingHom.ker (algebraMap R (HasQuotient.Quotient R p))) Top.top
      -/
    · rw [Quotient.algebraMap_eq, Ideal.mk_ker]
      /-
        case refine_1.convert_1
        R : Type u
        inst✝¹⁴ : CommRing R
        S : Type v
        inst✝¹³ : CommRing S
        p : Ideal R
        inst✝¹² : Algebra R S
        K : Type u_1
        inst✝¹¹ : Field K
        inst✝¹⁰ : Algebra R K
        L : Type u_2
        inst✝⁹ : Field L
        inst✝⁸ : Algebra S L
        inst✝⁷ : IsFractionRing S L
        hRK : IsFractionRing R K
        inst✝⁶ : IsDomain S
        inst✝⁵ : IsDedekindDomain R
        inst✝⁴ : Algebra K L
        inst✝³ : Algebra R L
        inst✝² : IsScalarTower R K L
        inst✝¹ : IsScalarTower R S L
        hp : p.IsMaximal
        inst✝ : Module.Finite R S
        ι : Type v := Module.Free.ChooseBasisIndex (HasQuotient.Quotient R p) (HasQuot …
        b : Basis ι (HasQuotient.Quotient R p) (HasQuotient.Quotient S (Ideal.map (alg …
        b' : ι → S := fun i => ⋯.choose
        b_eq_b' : Eq (⇑b) (Function.comp (⇑(↑R (Submodule.mkQ (Ideal.map (algebraMap R …
        b'' : ι → L := Function.comp (⇑(algebraMap S L)) b'
        this : LinearIndependent (HasQuotient.Quotient R p) (Function.comp (⇑(↑R (Subm …
        ⊢ Ne p Top.top
      -/
      exact hp.ne_top
      /-
        🎉 no goals
      -/
      /-
        case refine_1.convert_2
        R : Type u
        inst✝¹⁴ : CommRing R
        S : Type v
        inst✝¹³ : CommRing S
        p : Ideal R
        inst✝¹² : Algebra R S
        K : Type u_1
        inst✝¹¹ : Field K
        inst✝¹⁰ : Algebra R K
        L : Type u_2
        inst✝⁹ : Field L
        inst✝⁸ : Algebra S L
        inst✝⁷ : IsFractionRing S L
        hRK : IsFractionRing R K
        inst✝⁶ : IsDomain S
        inst✝⁵ : IsDedekindDomain R
        inst✝⁴ : Algebra K L
        inst✝³ : Algebra R L
        inst✝² : IsScalarTower R K L
        inst✝¹ : IsScalarTower R S L
        hp : p.IsMaximal
        inst✝ : Module.Finite R S
        ι : Type v := Module.Free.ChooseBasisIndex (HasQuotient.Quotient R p) (HasQuot …
        b : Basis ι (HasQuotient.Quotient R p) (HasQuotient.Quotient S (Ideal.map (alg …
        b' : ι → S := fun i => ⋯.choose
        b_eq_b' : Eq (⇑b) (Function.comp (⇑(↑R (Submodule.mkQ (Ideal.map (algebraMap R …
        b'' : ι → L := Function.comp (⇑(algebraMap S L)) b'
        this : LinearIndependent (HasQuotient.Quotient R p) (Function.comp (⇑(↑R (Subm …
        ⊢ Function.Injective ⇑(↑R (Algebra.linearMap S L))
      -/
    · exact IsFractionRing.injective S L
      /-
        🎉 no goals
      -/


local notation "f" => algebraMap R S

local notation "e" => ramificationIdx f p P


/-- `R / p` has a canonical map to `S / (P ^ e)`, where `e` is the ramification index
of `P` over `p`. -/
noncomputable instance Quotient.algebraQuotientPowRamificationIdx : Algebra (R ⧸ p) (S ⧸ P ^ e) :=
  Quotient.algebraQuotientOfLEComap (Ideal.map_le_iff_le_comap.mp le_pow_ramificationIdx)


@[simp]
theorem Quotient.algebraMap_quotient_pow_ramificationIdx (x : R) :
    algebraMap (R ⧸ p) (S ⧸ P ^ e) (Ideal.Quotient.mk p x) = Ideal.Quotient.mk (P ^ e) (f x) := rfl


/-- If `P` lies over `p`, then `R / p` has a canonical map to `S / P`.

This can't be an instance since the map `f : R → S` is generally not inferable.
-/
def Quotient.algebraQuotientOfRamificationIdxNeZero [hfp : NeZero e] :
    Algebra (R ⧸ p) (S ⧸ P) :=
  Quotient.algebraQuotientOfLEComap (le_comap_of_ramificationIdx_ne_zero hfp.out)


@[simp]
theorem Quotient.algebraMap_quotient_of_ramificationIdx_neZero
    [NeZero e] (x : R) :
    algebraMap (R ⧸ p) (S ⧸ P) (Ideal.Quotient.mk p x) = Ideal.Quotient.mk P (f x) := rfl


/-- The inclusion `(P^(i + 1) / P^e) ⊂ (P^i / P^e)`. -/
@[simps]
def powQuotSuccInclusion (i : ℕ) :
    Ideal.map (Ideal.Quotient.mk (P ^ e)) (P ^ (i + 1)) →ₗ[R ⧸ p]
    Ideal.map (Ideal.Quotient.mk (P ^ e)) (P ^ i) where
  toFun x := ⟨x, Ideal.map_mono (Ideal.pow_le_pow_right i.le_succ) x.2⟩
  map_add' _ _ := rfl
  map_smul' _ _ := rfl


theorem powQuotSuccInclusion_injective (i : ℕ) :
    Function.Injective (powQuotSuccInclusion p P i) := by
  /-
    R : Type u
    inst✝² : CommRing R
    S : Type v
    inst✝¹ : CommRing S
    p : Ideal R
    P : Ideal S
    inst✝ : Algebra R S
    i : Nat
    ⊢ Function.Injective ⇑(p.powQuotSuccInclusion P i)
  -/
  rw [← LinearMap.ker_eq_bot, LinearMap.ker_eq_bot']
  /-
    R : Type u
    inst✝² : CommRing R
    S : Type v
    inst✝¹ : CommRing S
    p : Ideal R
    P : Ideal S
    inst✝ : Algebra R S
    i : Nat
    ⊢ ∀ (m : Subtype fun x => Membership.mem (Ideal.map (Ideal.Quotient.mk (HPow.h …
  -/
  rintro ⟨x, hx⟩ hx0
  /-
    case mk
    R : Type u
    inst✝² : CommRing R
    S : Type v
    inst✝¹ : CommRing S
    p : Ideal R
    P : Ideal S
    inst✝ : Algebra R S
    i : Nat
    x : HasQuotient.Quotient S (HPow.hPow P (Ideal.ramificationIdx (algebraMap R S …
    hx : Membership.mem (Ideal.map (Ideal.Quotient.mk (HPow.hPow P (Ideal.ramifica …
    hx0 : Eq ((p.powQuotSuccInclusion P i) ⟨x, hx⟩) 0
    ⊢ Eq ⟨x, hx⟩ 0
  -/
  rw [Subtype.ext_iff] at hx0 ⊢
  /-
    case mk
    R : Type u
    inst✝² : CommRing R
    S : Type v
    inst✝¹ : CommRing S
    p : Ideal R
    P : Ideal S
    inst✝ : Algebra R S
    i : Nat
    x : HasQuotient.Quotient S (HPow.hPow P (Ideal.ramificationIdx (algebraMap R S …
    hx : Membership.mem (Ideal.map (Ideal.Quotient.mk (HPow.hPow P (Ideal.ramifica …
    hx0 : Eq ↑((p.powQuotSuccInclusion P i) ⟨x, hx⟩) ↑0
    ⊢ Eq ↑⟨x, hx⟩ ↑0
  -/
  rwa [powQuotSuccInclusion_apply_coe] at hx0
  /-
    🎉 no goals
  -/


/-- `S ⧸ P` embeds into the quotient by `P^(i+1) ⧸ P^e` as a subspace of `P^i ⧸ P^e`.
See `quotientToQuotientRangePowQuotSucc` for this as a linear map,
and `quotientRangePowQuotSuccInclusionEquiv` for this as a linear equivalence.
-/
noncomputable def quotientToQuotientRangePowQuotSuccAux {i : ℕ} {a : S} (a_mem : a ∈ P ^ i) :
    S ⧸ P →
      (P ^ i).map (Ideal.Quotient.mk (P ^ e)) ⧸ LinearMap.range (powQuotSuccInclusion p P i) :=
  Quotient.map' (fun x : S => ⟨_, Ideal.mem_map_of_mem _ (Ideal.mul_mem_right x _ a_mem)⟩)
    fun x y h => by
    /-
      R : Type u
      inst✝² : CommRing R
      S : Type v
      inst✝¹ : CommRing S
      f : RingHom R S
      p : Ideal R
      P : Ideal S
      inst✝ : Algebra R S
      i : Nat
      a : S
      a_mem : Membership.mem (HPow.hPow P i) a
      x y : S
      h : (Submodule.quotientRel P) x y
      ⊢ (LinearMap.range (p.powQuotSuccInclusion P i)).quotientRel ((fun x => ⟨(Idea …
    -/
    rw [Submodule.quotientRel_def] at h ⊢
    /-
      R : Type u
      inst✝² : CommRing R
      S : Type v
      inst✝¹ : CommRing S
      f : RingHom R S
      p : Ideal R
      P : Ideal S
      inst✝ : Algebra R S
      i : Nat
      a : S
      a_mem : Membership.mem (HPow.hPow P i) a
      x y : S
      h : Membership.mem P (HSub.hSub x y)
      ⊢ Membership.mem (LinearMap.range (p.powQuotSuccInclusion P i)) (HSub.hSub ((f …
    -/
    simp only [_root_.map_mul, LinearMap.mem_range]
    /-
      R : Type u
      inst✝² : CommRing R
      S : Type v
      inst✝¹ : CommRing S
      f : RingHom R S
      p : Ideal R
      P : Ideal S
      inst✝ : Algebra R S
      i : Nat
      a : S
      a_mem : Membership.mem (HPow.hPow P i) a
      x y : S
      h : Membership.mem P (HSub.hSub x y)
      ⊢ Exists fun y_1 => Eq ((p.powQuotSuccInclusion P i) y_1) (HSub.hSub ⟨HMul.hMu …
    -/
    refine ⟨⟨_, Ideal.mem_map_of_mem _ (Ideal.mul_mem_mul a_mem h)⟩, ?_⟩
    /-
      R : Type u
      inst✝² : CommRing R
      S : Type v
      inst✝¹ : CommRing S
      f : RingHom R S
      p : Ideal R
      P : Ideal S
      inst✝ : Algebra R S
      i : Nat
      a : S
      a_mem : Membership.mem (HPow.hPow P i) a
      x y : S
      h : Membership.mem P (HSub.hSub x y)
      ⊢ Eq ((p.powQuotSuccInclusion P i) ⟨(Ideal.Quotient.mk (HPow.hPow P (Ideal.ram …
    -/
    ext
    rw [powQuotSuccInclusion_apply_coe, Subtype.coe_mk, Submodule.coe_sub, Subtype.coe_mk,
      Subtype.coe_mk, _root_.map_mul, map_sub, mul_sub]


theorem quotientToQuotientRangePowQuotSuccAux_mk {i : ℕ} {a : S} (a_mem : a ∈ P ^ i) (x : S) :
    quotientToQuotientRangePowQuotSuccAux p P a_mem (Submodule.Quotient.mk x) =
      Submodule.Quotient.mk ⟨_, Ideal.mem_map_of_mem _ (Ideal.mul_mem_right x _ a_mem)⟩ := by
  /-
    R : Type u
    inst✝² : CommRing R
    S : Type v
    inst✝¹ : CommRing S
    p : Ideal R
    P : Ideal S
    inst✝ : Algebra R S
    i : Nat
    a : S
    a_mem : Membership.mem (HPow.hPow P i) a
    x : S
    ⊢ Eq (p.quotientToQuotientRangePowQuotSuccAux P a_mem (Submodule.Quotient.mk x …
  -/
  apply Quotient.map'_mk''
  /-
    🎉 no goals
  -/


/-- `S ⧸ P` embeds into the quotient by `P^(i+1) ⧸ P^e` as a subspace of `P^i ⧸ P^e`. -/
noncomputable def quotientToQuotientRangePowQuotSucc
    {i : ℕ} {a : S} (a_mem : a ∈ P ^ i) :
    S ⧸ P →ₗ[R ⧸ p]
      (P ^ i).map (Ideal.Quotient.mk (P ^ e)) ⧸ LinearMap.range (powQuotSuccInclusion p P i) where
  toFun := quotientToQuotientRangePowQuotSuccAux p P a_mem
  map_add' := by
    /-
      R : Type u
      inst✝² : CommRing R
      S : Type v
      inst✝¹ : CommRing S
      f : RingHom R S
      p : Ideal R
      P : Ideal S
      inst✝ : Algebra R S
      hfp : NeZero (Ideal.ramificationIdx (algebraMap R S) p P)
      i : Nat
      a : S
      a_mem : Membership.mem (HPow.hPow P i) a
      ⊢ ∀ (x y : HasQuotient.Quotient S P), Eq (p.quotientToQuotientRangePowQuotSucc …
    -/
    intro x y; refine Quotient.inductionOn' x fun x => Quotient.inductionOn' y fun y => ?_
    simp only [Submodule.Quotient.mk''_eq_mk, ← Submodule.Quotient.mk_add,
      quotientToQuotientRangePowQuotSuccAux_mk, mul_add]
    /-
      R : Type u
      inst✝² : CommRing R
      S : Type v
      inst✝¹ : CommRing S
      f : RingHom R S
      p : Ideal R
      P : Ideal S
      inst✝ : Algebra R S
      hfp : NeZero (Ideal.ramificationIdx (algebraMap R S) p P)
      i : Nat
      a : S
      a_mem : Membership.mem (HPow.hPow P i) a
      x✝ y✝ : HasQuotient.Quotient S P
      x y : S
      ⊢ Eq (Submodule.Quotient.mk ⟨(Ideal.Quotient.mk (HPow.hPow P (Ideal.ramificati …
    -/
    exact congr_arg Submodule.Quotient.mk rfl
    /-
      🎉 no goals
    -/
  map_smul' := by
    /-
      R : Type u
      inst✝² : CommRing R
      S : Type v
      inst✝¹ : CommRing S
      f : RingHom R S
      p : Ideal R
      P : Ideal S
      inst✝ : Algebra R S
      hfp : NeZero (Ideal.ramificationIdx (algebraMap R S) p P)
      i : Nat
      a : S
      a_mem : Membership.mem (HPow.hPow P i) a
      ⊢ ∀ (m : HasQuotient.Quotient R p) (x : HasQuotient.Quotient S P), Eq ({ toFun …
    -/
    intro x y; refine Quotient.inductionOn' x fun x => Quotient.inductionOn' y fun y => ?_
    simp only [Submodule.Quotient.mk''_eq_mk, RingHom.id_apply,
      quotientToQuotientRangePowQuotSuccAux_mk]
    /-
      R : Type u
      inst✝² : CommRing R
      S : Type v
      inst✝¹ : CommRing S
      f : RingHom R S
      p : Ideal R
      P : Ideal S
      inst✝ : Algebra R S
      hfp : NeZero (Ideal.ramificationIdx (algebraMap R S) p P)
      i : Nat
      a : S
      a_mem : Membership.mem (HPow.hPow P i) a
      x✝ : HasQuotient.Quotient R p
      y✝ : HasQuotient.Quotient S P
      x : R
      y : S
      ⊢ Eq (p.quotientToQuotientRangePowQuotSuccAux P a_mem (HSMul.hSMul (Submodule. …
    -/
    refine congr_arg Submodule.Quotient.mk ?_
    /-
      R : Type u
      inst✝² : CommRing R
      S : Type v
      inst✝¹ : CommRing S
      f : RingHom R S
      p : Ideal R
      P : Ideal S
      inst✝ : Algebra R S
      hfp : NeZero (Ideal.ramificationIdx (algebraMap R S) p P)
      i : Nat
      a : S
      a_mem : Membership.mem (HPow.hPow P i) a
      x✝ : HasQuotient.Quotient R p
      y✝ : HasQuotient.Quotient S P
      x : R
      y : S
      ⊢ Eq ((fun x => ⟨(Ideal.Quotient.mk (HPow.hPow P (Ideal.ramificationIdx (algeb …
    -/
    ext
    simp only [mul_assoc, _root_.map_mul, Quotient.mk_eq_mk, Submodule.coe_smul_of_tower,
      Algebra.smul_def, Quotient.algebraMap_quotient_pow_ramificationIdx]
    /-
      case a
      R : Type u
      inst✝² : CommRing R
      S : Type v
      inst✝¹ : CommRing S
      f : RingHom R S
      p : Ideal R
      P : Ideal S
      inst✝ : Algebra R S
      hfp : NeZero (Ideal.ramificationIdx (algebraMap R S) p P)
      i : Nat
      a : S
      a_mem : Membership.mem (HPow.hPow P i) a
      x✝ : HasQuotient.Quotient R p
      y✝ : HasQuotient.Quotient S P
      x : R
      y : S
      ⊢ Eq (HMul.hMul ((Ideal.Quotient.mk (HPow.hPow P (Ideal.ramificationIdx (algeb …
    -/
    ring
    /-
      🎉 no goals
    -/


theorem quotientToQuotientRangePowQuotSucc_mk {i : ℕ} {a : S} (a_mem : a ∈ P ^ i) (x : S) :
    quotientToQuotientRangePowQuotSucc p P a_mem (Submodule.Quotient.mk x) =
      Submodule.Quotient.mk ⟨_, Ideal.mem_map_of_mem _ (Ideal.mul_mem_right x _ a_mem)⟩ :=
  quotientToQuotientRangePowQuotSuccAux_mk p P a_mem x


theorem quotientToQuotientRangePowQuotSucc_injective [IsDedekindDomain S] [P.IsPrime]
    {i : ℕ} (hi : i < e) {a : S} (a_mem : a ∈ P ^ i) (a_not_mem : a ∉ P ^ (i + 1)) :
    Function.Injective (quotientToQuotientRangePowQuotSucc p P a_mem) := fun x =>
  Quotient.inductionOn' x fun x y =>
    Quotient.inductionOn' y fun y h => by
      /-
        R : Type u
        inst✝⁴ : CommRing R
        S : Type v
        inst✝³ : CommRing S
        p : Ideal R
        P : Ideal S
        inst✝² : Algebra R S
        hfp : NeZero (Ideal.ramificationIdx (algebraMap R S) p P)
        inst✝¹ : IsDedekindDomain S
        inst✝ : P.IsPrime
        i : Nat
        hi : LT.lt i (Ideal.ramificationIdx (algebraMap R S) p P)
        a : S
        a_mem : Membership.mem (HPow.hPow P i) a
        a_not_mem : Not (Membership.mem (HPow.hPow P (HAdd.hAdd i 1)) a)
        x✝ : HasQuotient.Quotient S P
        x : S
        y✝ : HasQuotient.Quotient S P
        y : S
        h : Eq ((p.quotientToQuotientRangePowQuotSucc P a_mem) (Quotient.mk'' x)) ((p. …
        ⊢ Eq (Quotient.mk'' x) (Quotient.mk'' y)
      -/
      have Pe_le_Pi1 : P ^ e ≤ P ^ (i + 1) := Ideal.pow_le_pow_right hi
      simp only [Submodule.Quotient.mk''_eq_mk, quotientToQuotientRangePowQuotSucc_mk,
        Submodule.Quotient.eq, LinearMap.mem_range, Subtype.ext_iff, Subtype.coe_mk,
        Submodule.coe_sub] at h ⊢
      /-
        R : Type u
        inst✝⁴ : CommRing R
        S : Type v
        inst✝³ : CommRing S
        p : Ideal R
        P : Ideal S
        inst✝² : Algebra R S
        hfp : NeZero (Ideal.ramificationIdx (algebraMap R S) p P)
        inst✝¹ : IsDedekindDomain S
        inst✝ : P.IsPrime
        i : Nat
        hi : LT.lt i (Ideal.ramificationIdx (algebraMap R S) p P)
        a : S
        a_mem : Membership.mem (HPow.hPow P i) a
        a_not_mem : Not (Membership.mem (HPow.hPow P (HAdd.hAdd i 1)) a)
        x✝ : HasQuotient.Quotient S P
        x : S
        y✝ : HasQuotient.Quotient S P
        y : S
        Pe_le_Pi1 : LE.le (HPow.hPow P (Ideal.ramificationIdx (algebraMap R S) p P)) ( …
        h : Exists fun y_1 => Eq (↑((p.powQuotSuccInclusion P i) y_1)) (HSub.hSub ((Id …
        ⊢ Membership.mem P (HSub.hSub x y)
      -/
      rcases h with ⟨⟨⟨z⟩, hz⟩, h⟩
      rw [Submodule.Quotient.quot_mk_eq_mk, Ideal.Quotient.mk_eq_mk, Ideal.mem_quotient_iff_mem_sup,
        sup_eq_left.mpr Pe_le_Pi1] at hz
      rw [powQuotSuccInclusion_apply_coe, Subtype.coe_mk, Submodule.Quotient.quot_mk_eq_mk,
        Ideal.Quotient.mk_eq_mk, ← map_sub, Ideal.Quotient.eq, ← mul_sub] at h
      exact
        (Ideal.IsPrime.mem_pow_mul _
              ((Submodule.sub_mem_iff_right _ hz).mp (Pe_le_Pi1 h))).resolve_left
          a_not_mem


theorem quotientToQuotientRangePowQuotSucc_surjective [IsDedekindDomain S]
    (hP0 : P ≠ ⊥) [hP : P.IsPrime] {i : ℕ} (hi : i < e) {a : S} (a_mem : a ∈ P ^ i)
    (a_not_mem : a ∉ P ^ (i + 1)) :
    Function.Surjective (quotientToQuotientRangePowQuotSucc p P a_mem) := by
  /-
    R : Type u
    inst✝³ : CommRing R
    S : Type v
    inst✝² : CommRing S
    p : Ideal R
    P : Ideal S
    inst✝¹ : Algebra R S
    hfp : NeZero (Ideal.ramificationIdx (algebraMap R S) p P)
    inst✝ : IsDedekindDomain S
    hP0 : Ne P Bot.bot
    hP : P.IsPrime
    i : Nat
    hi : LT.lt i (Ideal.ramificationIdx (algebraMap R S) p P)
    a : S
    a_mem : Membership.mem (HPow.hPow P i) a
    a_not_mem : Not (Membership.mem (HPow.hPow P (HAdd.hAdd i 1)) a)
    ⊢ Function.Surjective ⇑(p.quotientToQuotientRangePowQuotSucc P a_mem)
  -/
  rintro ⟨⟨⟨x⟩, hx⟩⟩
  /-
    case mk.mk.mk
    R : Type u
    inst✝³ : CommRing R
    S : Type v
    inst✝² : CommRing S
    p : Ideal R
    P : Ideal S
    inst✝¹ : Algebra R S
    hfp : NeZero (Ideal.ramificationIdx (algebraMap R S) p P)
    inst✝ : IsDedekindDomain S
    hP0 : Ne P Bot.bot
    hP : P.IsPrime
    i : Nat
    hi : LT.lt i (Ideal.ramificationIdx (algebraMap R S) p P)
    a : S
    a_mem : Membership.mem (HPow.hPow P i) a
    a_not_mem : Not (Membership.mem (HPow.hPow P (HAdd.hAdd i 1)) a)
    b✝ : HasQuotient.Quotient (Subtype fun x => Membership.mem (Ideal.map (Ideal.Q …
    val✝ : HasQuotient.Quotient S (HPow.hPow P (Ideal.ramificationIdx (algebraMap  …
    x : S
    hx : Membership.mem (Ideal.map (Ideal.Quotient.mk (HPow.hPow P (Ideal.ramifica …
    ⊢ Exists fun a_1 => Eq ((p.quotientToQuotientRangePowQuotSucc P a_mem) a_1) (Q …
  -/
  have Pe_le_Pi : P ^ e ≤ P ^ i := Ideal.pow_le_pow_right hi.le
  rw [Submodule.Quotient.quot_mk_eq_mk, Ideal.Quotient.mk_eq_mk, Ideal.mem_quotient_iff_mem_sup,
    sup_eq_left.mpr Pe_le_Pi] at hx
  suffices hx' : x ∈ Ideal.span {a} ⊔ P ^ (i + 1) by
    obtain ⟨y', hy', z, hz, rfl⟩ := Submodule.mem_sup.mp hx'
    obtain ⟨y, rfl⟩ := Ideal.mem_span_singleton.mp hy'
    refine ⟨Submodule.Quotient.mk y, ?_⟩
    simp only [Submodule.Quotient.quot_mk_eq_mk, quotientToQuotientRangePowQuotSucc_mk,
      Submodule.Quotient.eq, LinearMap.mem_range, Subtype.ext_iff, Subtype.coe_mk,
      Submodule.coe_sub]
    refine ⟨⟨_, Ideal.mem_map_of_mem _ (Submodule.neg_mem _ hz)⟩, ?_⟩
    rw [powQuotSuccInclusion_apply_coe, Subtype.coe_mk, Ideal.Quotient.mk_eq_mk, map_add,
      sub_add_cancel_left, map_neg]
  /-
    case mk.mk.mk
    R : Type u
    inst✝³ : CommRing R
    S : Type v
    inst✝² : CommRing S
    p : Ideal R
    P : Ideal S
    inst✝¹ : Algebra R S
    hfp : NeZero (Ideal.ramificationIdx (algebraMap R S) p P)
    inst✝ : IsDedekindDomain S
    hP0 : Ne P Bot.bot
    hP : P.IsPrime
    i : Nat
    hi : LT.lt i (Ideal.ramificationIdx (algebraMap R S) p P)
    a : S
    a_mem : Membership.mem (HPow.hPow P i) a
    a_not_mem : Not (Membership.mem (HPow.hPow P (HAdd.hAdd i 1)) a)
    b✝ : HasQuotient.Quotient (Subtype fun x => Membership.mem (Ideal.map (Ideal.Q …
    val✝ : HasQuotient.Quotient S (HPow.hPow P (Ideal.ramificationIdx (algebraMap  …
    x : S
    hx✝ : Membership.mem (Ideal.map (Ideal.Quotient.mk (HPow.hPow P (Ideal.ramific …
    hx : Membership.mem (HPow.hPow P i) x
    Pe_le_Pi : LE.le (HPow.hPow P (Ideal.ramificationIdx (algebraMap R S) p P)) (H …
    ⊢ Membership.mem (Max.max (Ideal.span (Singleton.singleton a)) (HPow.hPow P (H …
  -/
  letI := Classical.decEq (Ideal S)
  rw [sup_eq_prod_inf_factors _ (pow_ne_zero _ hP0), normalizedFactors_pow,
    normalizedFactors_irreducible ((Ideal.prime_iff_isPrime hP0).mpr hP).irreducible, normalize_eq,
    Multiset.nsmul_singleton, Multiset.inter_replicate, Multiset.prod_replicate]
    /-
      case mk.mk.mk
      R : Type u
      inst✝³ : CommRing R
      S : Type v
      inst✝² : CommRing S
      p : Ideal R
      P : Ideal S
      inst✝¹ : Algebra R S
      hfp : NeZero (Ideal.ramificationIdx (algebraMap R S) p P)
      inst✝ : IsDedekindDomain S
      hP0 : Ne P Bot.bot
      hP : P.IsPrime
      i : Nat
      hi : LT.lt i (Ideal.ramificationIdx (algebraMap R S) p P)
      a : S
      a_mem : Membership.mem (HPow.hPow P i) a
      a_not_mem : Not (Membership.mem (HPow.hPow P (HAdd.hAdd i 1)) a)
      b✝ : HasQuotient.Quotient (Subtype fun x => Membership.mem (Ideal.map (Ideal.Q …
      val✝ : HasQuotient.Quotient S (HPow.hPow P (Ideal.ramificationIdx (algebraMap  …
      x : S
      hx✝ : Membership.mem (Ideal.map (Ideal.Quotient.mk (HPow.hPow P (Ideal.ramific …
      hx : Membership.mem (HPow.hPow P i) x
      Pe_le_Pi : LE.le (HPow.hPow P (Ideal.ramificationIdx (algebraMap R S) p P)) (H …
      this : DecidableEq (Ideal S) := Classical.decEq (Ideal S)
      ⊢ Membership.mem (HPow.hPow P (Min.min (Multiset.count P (UniqueFactorizationM …
    -/
  · rw [← Submodule.span_singleton_le_iff_mem, Ideal.submodule_span_eq] at a_mem a_not_mem
    /-
      case mk.mk.mk
      R : Type u
      inst✝³ : CommRing R
      S : Type v
      inst✝² : CommRing S
      p : Ideal R
      P : Ideal S
      inst✝¹ : Algebra R S
      hfp : NeZero (Ideal.ramificationIdx (algebraMap R S) p P)
      inst✝ : IsDedekindDomain S
      hP0 : Ne P Bot.bot
      hP : P.IsPrime
      i : Nat
      hi : LT.lt i (Ideal.ramificationIdx (algebraMap R S) p P)
      a : S
      a_mem : LE.le (Ideal.span (Singleton.singleton a)) (HPow.hPow P i)
      a_not_mem : Not (LE.le (Ideal.span (Singleton.singleton a)) (HPow.hPow P (HAdd …
      b✝ : HasQuotient.Quotient (Subtype fun x => Membership.mem (Ideal.map (Ideal.Q …
      val✝ : HasQuotient.Quotient S (HPow.hPow P (Ideal.ramificationIdx (algebraMap  …
      x : S
      hx✝ : Membership.mem (Ideal.map (Ideal.Quotient.mk (HPow.hPow P (Ideal.ramific …
      hx : Membership.mem (HPow.hPow P i) x
      Pe_le_Pi : LE.le (HPow.hPow P (Ideal.ramificationIdx (algebraMap R S) p P)) (H …
      this : DecidableEq (Ideal S) := Classical.decEq (Ideal S)
      ⊢ Membership.mem (HPow.hPow P (Min.min (Multiset.count P (UniqueFactorizationM …
    -/
    rwa [Ideal.count_normalizedFactors_eq a_mem a_not_mem, min_eq_left i.le_succ]
    /-
      🎉 no goals
    -/
    /-
      R : Type u
      inst✝³ : CommRing R
      S : Type v
      inst✝² : CommRing S
      p : Ideal R
      P : Ideal S
      inst✝¹ : Algebra R S
      hfp : NeZero (Ideal.ramificationIdx (algebraMap R S) p P)
      inst✝ : IsDedekindDomain S
      hP0 : Ne P Bot.bot
      hP : P.IsPrime
      i : Nat
      hi : LT.lt i (Ideal.ramificationIdx (algebraMap R S) p P)
      a : S
      a_mem : Membership.mem (HPow.hPow P i) a
      a_not_mem : Not (Membership.mem (HPow.hPow P (HAdd.hAdd i 1)) a)
      b✝ : HasQuotient.Quotient (Subtype fun x => Membership.mem (Ideal.map (Ideal.Q …
      val✝ : HasQuotient.Quotient S (HPow.hPow P (Ideal.ramificationIdx (algebraMap  …
      x : S
      hx✝ : Membership.mem (Ideal.map (Ideal.Quotient.mk (HPow.hPow P (Ideal.ramific …
      hx : Membership.mem (HPow.hPow P i) x
      Pe_le_Pi : LE.le (HPow.hPow P (Ideal.ramificationIdx (algebraMap R S) p P)) (H …
      this : DecidableEq (Ideal S) := Classical.decEq (Ideal S)
      ⊢ Ne (Ideal.span (Singleton.singleton a)) Bot.bot
    -/
  · intro ha
    /-
      R : Type u
      inst✝³ : CommRing R
      S : Type v
      inst✝² : CommRing S
      p : Ideal R
      P : Ideal S
      inst✝¹ : Algebra R S
      hfp : NeZero (Ideal.ramificationIdx (algebraMap R S) p P)
      inst✝ : IsDedekindDomain S
      hP0 : Ne P Bot.bot
      hP : P.IsPrime
      i : Nat
      hi : LT.lt i (Ideal.ramificationIdx (algebraMap R S) p P)
      a : S
      a_mem : Membership.mem (HPow.hPow P i) a
      a_not_mem : Not (Membership.mem (HPow.hPow P (HAdd.hAdd i 1)) a)
      b✝ : HasQuotient.Quotient (Subtype fun x => Membership.mem (Ideal.map (Ideal.Q …
      val✝ : HasQuotient.Quotient S (HPow.hPow P (Ideal.ramificationIdx (algebraMap  …
      x : S
      hx✝ : Membership.mem (Ideal.map (Ideal.Quotient.mk (HPow.hPow P (Ideal.ramific …
      hx : Membership.mem (HPow.hPow P i) x
      Pe_le_Pi : LE.le (HPow.hPow P (Ideal.ramificationIdx (algebraMap R S) p P)) (H …
      this : DecidableEq (Ideal S) := Classical.decEq (Ideal S)
      ha : Eq (Ideal.span (Singleton.singleton a)) Bot.bot
      ⊢ False
    -/
    rw [Ideal.span_singleton_eq_bot.mp ha] at a_not_mem
    /-
      R : Type u
      inst✝³ : CommRing R
      S : Type v
      inst✝² : CommRing S
      p : Ideal R
      P : Ideal S
      inst✝¹ : Algebra R S
      hfp : NeZero (Ideal.ramificationIdx (algebraMap R S) p P)
      inst✝ : IsDedekindDomain S
      hP0 : Ne P Bot.bot
      hP : P.IsPrime
      i : Nat
      hi : LT.lt i (Ideal.ramificationIdx (algebraMap R S) p P)
      a : S
      a_mem : Membership.mem (HPow.hPow P i) a
      a_not_mem : Not (Membership.mem (HPow.hPow P (HAdd.hAdd i 1)) 0)
      b✝ : HasQuotient.Quotient (Subtype fun x => Membership.mem (Ideal.map (Ideal.Q …
      val✝ : HasQuotient.Quotient S (HPow.hPow P (Ideal.ramificationIdx (algebraMap  …
      x : S
      hx✝ : Membership.mem (Ideal.map (Ideal.Quotient.mk (HPow.hPow P (Ideal.ramific …
      hx : Membership.mem (HPow.hPow P i) x
      Pe_le_Pi : LE.le (HPow.hPow P (Ideal.ramificationIdx (algebraMap R S) p P)) (H …
      this : DecidableEq (Ideal S) := Classical.decEq (Ideal S)
      ha : Eq (Ideal.span (Singleton.singleton a)) Bot.bot
      ⊢ False
    -/
    have := (P ^ (i + 1)).zero_mem
    /-
      R : Type u
      inst✝³ : CommRing R
      S : Type v
      inst✝² : CommRing S
      p : Ideal R
      P : Ideal S
      inst✝¹ : Algebra R S
      hfp : NeZero (Ideal.ramificationIdx (algebraMap R S) p P)
      inst✝ : IsDedekindDomain S
      hP0 : Ne P Bot.bot
      hP : P.IsPrime
      i : Nat
      hi : LT.lt i (Ideal.ramificationIdx (algebraMap R S) p P)
      a : S
      a_mem : Membership.mem (HPow.hPow P i) a
      a_not_mem : Not (Membership.mem (HPow.hPow P (HAdd.hAdd i 1)) 0)
      b✝ : HasQuotient.Quotient (Subtype fun x => Membership.mem (Ideal.map (Ideal.Q …
      val✝ : HasQuotient.Quotient S (HPow.hPow P (Ideal.ramificationIdx (algebraMap  …
      x : S
      hx✝ : Membership.mem (Ideal.map (Ideal.Quotient.mk (HPow.hPow P (Ideal.ramific …
      hx : Membership.mem (HPow.hPow P i) x
      Pe_le_Pi : LE.le (HPow.hPow P (Ideal.ramificationIdx (algebraMap R S) p P)) (H …
      this✝ : DecidableEq (Ideal S) := Classical.decEq (Ideal S)
      ha : Eq (Ideal.span (Singleton.singleton a)) Bot.bot
      this : Membership.mem (HPow.hPow P (HAdd.hAdd i 1)) 0
      ⊢ False
    -/
    contradiction
    /-
      🎉 no goals
    -/


/-- Quotienting `P^i / P^e` by its subspace `P^(i+1) ⧸ P^e` is
`R ⧸ p`-linearly isomorphic to `S ⧸ P`. -/
noncomputable def quotientRangePowQuotSuccInclusionEquiv [IsDedekindDomain S]
    [P.IsPrime] (hP : P ≠ ⊥) {i : ℕ} (hi : i < e) :
    ((P ^ i).map (Ideal.Quotient.mk (P ^ e)) ⧸ LinearMap.range (powQuotSuccInclusion p P i))
      ≃ₗ[R ⧸ p] S ⧸ P := by
  choose a a_mem a_not_mem using
    SetLike.exists_of_lt
      (Ideal.pow_right_strictAnti P hP (Ideal.IsPrime.ne_top inferInstance) (le_refl i.succ))
  /-
    R : Type u
    inst✝⁴ : CommRing R
    S : Type v
    inst✝³ : CommRing S
    f : RingHom R S
    p : Ideal R
    P : Ideal S
    inst✝² : Algebra R S
    hfp : NeZero (Ideal.ramificationIdx (algebraMap R S) p P)
    inst✝¹ : IsDedekindDomain S
    inst✝ : P.IsPrime
    hP : Ne P Bot.bot
    i : Nat
    hi : LT.lt i (Ideal.ramificationIdx (algebraMap R S) p P)
    a : S
    a_mem : Membership.mem ((fun x => HPow.hPow P x) i) a
    a_not_mem : Not (Membership.mem ((fun x => HPow.hPow P x) i.succ) a)
    ⊢ LinearEquiv (RingHom.id (HasQuotient.Quotient R p)) (HasQuotient.Quotient (S …
  -/
  refine (LinearEquiv.ofBijective ?_ ⟨?_, ?_⟩).symm
    /-
      case refine_1
      R : Type u
      inst✝⁴ : CommRing R
      S : Type v
      inst✝³ : CommRing S
      f : RingHom R S
      p : Ideal R
      P : Ideal S
      inst✝² : Algebra R S
      hfp : NeZero (Ideal.ramificationIdx (algebraMap R S) p P)
      inst✝¹ : IsDedekindDomain S
      inst✝ : P.IsPrime
      hP : Ne P Bot.bot
      i : Nat
      hi : LT.lt i (Ideal.ramificationIdx (algebraMap R S) p P)
      a : S
      a_mem : Membership.mem ((fun x => HPow.hPow P x) i) a
      a_not_mem : Not (Membership.mem ((fun x => HPow.hPow P x) i.succ) a)
      ⊢ LinearMap (RingHom.id (HasQuotient.Quotient R p)) (HasQuotient.Quotient S P) …
    -/
  · exact quotientToQuotientRangePowQuotSucc p P a_mem
    /-
      🎉 no goals
    -/
    /-
      case refine_2
      R : Type u
      inst✝⁴ : CommRing R
      S : Type v
      inst✝³ : CommRing S
      f : RingHom R S
      p : Ideal R
      P : Ideal S
      inst✝² : Algebra R S
      hfp : NeZero (Ideal.ramificationIdx (algebraMap R S) p P)
      inst✝¹ : IsDedekindDomain S
      inst✝ : P.IsPrime
      hP : Ne P Bot.bot
      i : Nat
      hi : LT.lt i (Ideal.ramificationIdx (algebraMap R S) p P)
      a : S
      a_mem : Membership.mem ((fun x => HPow.hPow P x) i) a
      a_not_mem : Not (Membership.mem ((fun x => HPow.hPow P x) i.succ) a)
      ⊢ Function.Injective ⇑(p.quotientToQuotientRangePowQuotSucc P a_mem)
    -/
  · exact quotientToQuotientRangePowQuotSucc_injective p P hi a_mem a_not_mem
    /-
      🎉 no goals
    -/
    /-
      case refine_3
      R : Type u
      inst✝⁴ : CommRing R
      S : Type v
      inst✝³ : CommRing S
      f : RingHom R S
      p : Ideal R
      P : Ideal S
      inst✝² : Algebra R S
      hfp : NeZero (Ideal.ramificationIdx (algebraMap R S) p P)
      inst✝¹ : IsDedekindDomain S
      inst✝ : P.IsPrime
      hP : Ne P Bot.bot
      i : Nat
      hi : LT.lt i (Ideal.ramificationIdx (algebraMap R S) p P)
      a : S
      a_mem : Membership.mem ((fun x => HPow.hPow P x) i) a
      a_not_mem : Not (Membership.mem ((fun x => HPow.hPow P x) i.succ) a)
      ⊢ Function.Surjective ⇑(p.quotientToQuotientRangePowQuotSucc P a_mem)
    -/
  · exact quotientToQuotientRangePowQuotSucc_surjective p P hP hi a_mem a_not_mem
    /-
      🎉 no goals
    -/


/-- Since the inclusion `(P^(i + 1) / P^e) ⊂ (P^i / P^e)` has a kernel isomorphic to `P / S`,
`[P^i / P^e : R / p] = [P^(i+1) / P^e : R / p] + [P / S : R / p]` -/
theorem rank_pow_quot_aux [IsDedekindDomain S] [p.IsMaximal] [P.IsPrime] (hP0 : P ≠ ⊥)
    {i : ℕ} (hi : i < e) :
    Module.rank (R ⧸ p) (Ideal.map (Ideal.Quotient.mk (P ^ e)) (P ^ i)) =
      Module.rank (R ⧸ p) (S ⧸ P) +
        Module.rank (R ⧸ p) (Ideal.map (Ideal.Quotient.mk (P ^ e)) (P ^ (i + 1))) := by
  rw [← rank_range_of_injective _ (powQuotSuccInclusion_injective p P i),
    (quotientRangePowQuotSuccInclusionEquiv p P hP0 hi).symm.rank_eq]
  /-
    R : Type u
    inst✝⁵ : CommRing R
    S : Type v
    inst✝⁴ : CommRing S
    p : Ideal R
    P : Ideal S
    inst✝³ : Algebra R S
    hfp : NeZero (Ideal.ramificationIdx (algebraMap R S) p P)
    inst✝² : IsDedekindDomain S
    inst✝¹ : p.IsMaximal
    inst✝ : P.IsPrime
    hP0 : Ne P Bot.bot
    i : Nat
    hi : LT.lt i (Ideal.ramificationIdx (algebraMap R S) p P)
    ⊢ Eq (Module.rank (HasQuotient.Quotient R p) (Subtype fun x => Membership.mem  …
  -/
  exact (Submodule.rank_quotient_add_rank (LinearMap.range (powQuotSuccInclusion p P i))).symm
  /-
    🎉 no goals
  -/


theorem rank_pow_quot [IsDedekindDomain S] [p.IsMaximal] [P.IsPrime] (hP0 : P ≠ ⊥)
    (i : ℕ) (hi : i ≤ e) :
    Module.rank (R ⧸ p) (Ideal.map (Ideal.Quotient.mk (P ^ e)) (P ^ i)) =
      (e - i) • Module.rank (R ⧸ p) (S ⧸ P) := by
-- Porting note: Lean cannot figure out what to prove by itself
  let Q : ℕ → Prop :=
    fun i => Module.rank (R ⧸ p) { x // x ∈ map (Quotient.mk (P ^ e)) (P ^ i) }
      = (e - i) • Module.rank (R ⧸ p) (S ⧸ P)
  /-
    R : Type u
    inst✝⁵ : CommRing R
    S : Type v
    inst✝⁴ : CommRing S
    p : Ideal R
    P : Ideal S
    inst✝³ : Algebra R S
    hfp : NeZero (Ideal.ramificationIdx (algebraMap R S) p P)
    inst✝² : IsDedekindDomain S
    inst✝¹ : p.IsMaximal
    inst✝ : P.IsPrime
    hP0 : Ne P Bot.bot
    i : Nat
    hi : LE.le i (Ideal.ramificationIdx (algebraMap R S) p P)
    Q : Nat → Prop := fun i => Eq (Module.rank (HasQuotient.Quotient R p) (Subtype …
    ⊢ Eq (Module.rank (HasQuotient.Quotient R p) (Subtype fun x => Membership.mem  …
  -/
  refine Nat.decreasingInduction' (P := Q) (fun j lt_e _le_j ih => ?_) hi ?_
    /-
      case refine_1
      R : Type u
      inst✝⁵ : CommRing R
      S : Type v
      inst✝⁴ : CommRing S
      p : Ideal R
      P : Ideal S
      inst✝³ : Algebra R S
      hfp : NeZero (Ideal.ramificationIdx (algebraMap R S) p P)
      inst✝² : IsDedekindDomain S
      inst✝¹ : p.IsMaximal
      inst✝ : P.IsPrime
      hP0 : Ne P Bot.bot
      i : Nat
      hi : LE.le i (Ideal.ramificationIdx (algebraMap R S) p P)
      Q : Nat → Prop := fun i => Eq (Module.rank (HasQuotient.Quotient R p) (Subtype …
      j : Nat
      lt_e : LT.lt j (Ideal.ramificationIdx (algebraMap R S) p P)
      _le_j : LE.le i j
      ih : Q (HAdd.hAdd j 1)
      ⊢ Q j
    -/
  · dsimp only [Q]
    rw [rank_pow_quot_aux p P _ lt_e, ih, ← succ_nsmul', Nat.sub_succ, ← Nat.succ_eq_add_one,
      Nat.succ_pred_eq_of_pos (Nat.sub_pos_of_lt lt_e)]
    /-
      R : Type u
      inst✝⁵ : CommRing R
      S : Type v
      inst✝⁴ : CommRing S
      p : Ideal R
      P : Ideal S
      inst✝³ : Algebra R S
      hfp : NeZero (Ideal.ramificationIdx (algebraMap R S) p P)
      inst✝² : IsDedekindDomain S
      inst✝¹ : p.IsMaximal
      inst✝ : P.IsPrime
      hP0 : Ne P Bot.bot
      i : Nat
      hi : LE.le i (Ideal.ramificationIdx (algebraMap R S) p P)
      Q : Nat → Prop := fun i => Eq (Module.rank (HasQuotient.Quotient R p) (Subtype …
      j : Nat
      lt_e : LT.lt j (Ideal.ramificationIdx (algebraMap R S) p P)
      _le_j : LE.le i j
      ih : Q (HAdd.hAdd j 1)
      ⊢ Ne P Bot.bot
    -/
    assumption
    /-
      🎉 no goals
    -/
    /-
      case refine_2
      R : Type u
      inst✝⁵ : CommRing R
      S : Type v
      inst✝⁴ : CommRing S
      p : Ideal R
      P : Ideal S
      inst✝³ : Algebra R S
      hfp : NeZero (Ideal.ramificationIdx (algebraMap R S) p P)
      inst✝² : IsDedekindDomain S
      inst✝¹ : p.IsMaximal
      inst✝ : P.IsPrime
      hP0 : Ne P Bot.bot
      i : Nat
      hi : LE.le i (Ideal.ramificationIdx (algebraMap R S) p P)
      Q : Nat → Prop := fun i => Eq (Module.rank (HasQuotient.Quotient R p) (Subtype …
      ⊢ Q (Ideal.ramificationIdx (algebraMap R S) p P)
    -/
  · dsimp only [Q]
    /-
      case refine_2
      R : Type u
      inst✝⁵ : CommRing R
      S : Type v
      inst✝⁴ : CommRing S
      p : Ideal R
      P : Ideal S
      inst✝³ : Algebra R S
      hfp : NeZero (Ideal.ramificationIdx (algebraMap R S) p P)
      inst✝² : IsDedekindDomain S
      inst✝¹ : p.IsMaximal
      inst✝ : P.IsPrime
      hP0 : Ne P Bot.bot
      i : Nat
      hi : LE.le i (Ideal.ramificationIdx (algebraMap R S) p P)
      Q : Nat → Prop := fun i => Eq (Module.rank (HasQuotient.Quotient R p) (Subtype …
      ⊢ Eq (Module.rank (HasQuotient.Quotient R p) (Subtype fun x => Membership.mem  …
    -/
    rw [Nat.sub_self, zero_nsmul, map_quotient_self]
    /-
      case refine_2
      R : Type u
      inst✝⁵ : CommRing R
      S : Type v
      inst✝⁴ : CommRing S
      p : Ideal R
      P : Ideal S
      inst✝³ : Algebra R S
      hfp : NeZero (Ideal.ramificationIdx (algebraMap R S) p P)
      inst✝² : IsDedekindDomain S
      inst✝¹ : p.IsMaximal
      inst✝ : P.IsPrime
      hP0 : Ne P Bot.bot
      i : Nat
      hi : LE.le i (Ideal.ramificationIdx (algebraMap R S) p P)
      Q : Nat → Prop := fun i => Eq (Module.rank (HasQuotient.Quotient R p) (Subtype …
      ⊢ Eq (Module.rank (HasQuotient.Quotient R p) (Subtype fun x => Membership.mem  …
    -/
    exact rank_bot (R ⧸ p) (S ⧸ P ^ e)
    /-
      🎉 no goals
    -/


/-- If `p` is a maximal ideal of `R`, `S` extends `R` and `P^e` lies over `p`,
then the dimension `[S/(P^e) : R/p]` is equal to `e * [S/P : R/p]`. -/
theorem rank_prime_pow_ramificationIdx [IsDedekindDomain S] [p.IsMaximal] [P.IsPrime]
    (hP0 : P ≠ ⊥) (he : e ≠ 0) :
    Module.rank (R ⧸ p) (S ⧸ P ^ e) =
      e •
        @Module.rank (R ⧸ p) (S ⧸ P) _ _
          (@Algebra.toModule _ _ _ _ <|
            @Quotient.algebraQuotientOfRamificationIdxNeZero _ _ _ _ _ _ _ ⟨he⟩) := by
  /-
    R : Type u
    inst✝⁵ : CommRing R
    S : Type v
    inst✝⁴ : CommRing S
    p : Ideal R
    P : Ideal S
    inst✝³ : Algebra R S
    inst✝² : IsDedekindDomain S
    inst✝¹ : p.IsMaximal
    inst✝ : P.IsPrime
    hP0 : Ne P Bot.bot
    he : Ne (Ideal.ramificationIdx (algebraMap R S) p P) 0
    ⊢ Eq (Module.rank (HasQuotient.Quotient R p) (HasQuotient.Quotient S (HPow.hPo …
  -/
  letI : NeZero e := ⟨he⟩
  /-
    R : Type u
    inst✝⁵ : CommRing R
    S : Type v
    inst✝⁴ : CommRing S
    p : Ideal R
    P : Ideal S
    inst✝³ : Algebra R S
    inst✝² : IsDedekindDomain S
    inst✝¹ : p.IsMaximal
    inst✝ : P.IsPrime
    hP0 : Ne P Bot.bot
    he : Ne (Ideal.ramificationIdx (algebraMap R S) p P) 0
    this : NeZero (Ideal.ramificationIdx (algebraMap R S) p P) := { out := he }
    ⊢ Eq (Module.rank (HasQuotient.Quotient R p) (HasQuotient.Quotient S (HPow.hPo …
  -/
  have := rank_pow_quot p P hP0 0 (Nat.zero_le e)
  /-
    R : Type u
    inst✝⁵ : CommRing R
    S : Type v
    inst✝⁴ : CommRing S
    p : Ideal R
    P : Ideal S
    inst✝³ : Algebra R S
    inst✝² : IsDedekindDomain S
    inst✝¹ : p.IsMaximal
    inst✝ : P.IsPrime
    hP0 : Ne P Bot.bot
    he : Ne (Ideal.ramificationIdx (algebraMap R S) p P) 0
    this✝ : NeZero (Ideal.ramificationIdx (algebraMap R S) p P) := { out := he }
    this : Eq (Module.rank (HasQuotient.Quotient R p) (Subtype fun x => Membership …
    ⊢ Eq (Module.rank (HasQuotient.Quotient R p) (HasQuotient.Quotient S (HPow.hPo …
  -/
  rw [pow_zero, Nat.sub_zero, Ideal.one_eq_top, Ideal.map_top] at this
  /-
    R : Type u
    inst✝⁵ : CommRing R
    S : Type v
    inst✝⁴ : CommRing S
    p : Ideal R
    P : Ideal S
    inst✝³ : Algebra R S
    inst✝² : IsDedekindDomain S
    inst✝¹ : p.IsMaximal
    inst✝ : P.IsPrime
    hP0 : Ne P Bot.bot
    he : Ne (Ideal.ramificationIdx (algebraMap R S) p P) 0
    this✝ : NeZero (Ideal.ramificationIdx (algebraMap R S) p P) := { out := he }
    this : Eq (Module.rank (HasQuotient.Quotient R p) (Subtype fun x => Membership …
    ⊢ Eq (Module.rank (HasQuotient.Quotient R p) (HasQuotient.Quotient S (HPow.hPo …
  -/
  exact (rank_top (R ⧸ p) _).symm.trans this
  /-
    🎉 no goals
  -/


/-- If `p` is a maximal ideal of `R`, `S` extends `R` and `P^e` lies over `p`,
then the dimension `[S/(P^e) : R/p]`, as a natural number, is equal to `e * [S/P : R/p]`. -/
theorem finrank_prime_pow_ramificationIdx [IsDedekindDomain S] (hP0 : P ≠ ⊥)
    [p.IsMaximal] [P.IsPrime] (he : e ≠ 0) :
    finrank (R ⧸ p) (S ⧸ P ^ e) =
      e *
        @finrank (R ⧸ p) (S ⧸ P) _ _
          (@Algebra.toModule _ _ _ _ <|
            @Quotient.algebraQuotientOfRamificationIdxNeZero _ _ _ _ _ _ _ ⟨he⟩) := by
  /-
    R : Type u
    inst✝⁵ : CommRing R
    S : Type v
    inst✝⁴ : CommRing S
    p : Ideal R
    P : Ideal S
    inst✝³ : Algebra R S
    inst✝² : IsDedekindDomain S
    hP0 : Ne P Bot.bot
    inst✝¹ : p.IsMaximal
    inst✝ : P.IsPrime
    he : Ne (Ideal.ramificationIdx (algebraMap R S) p P) 0
    ⊢ Eq (Module.finrank (HasQuotient.Quotient R p) (HasQuotient.Quotient S (HPow. …
  -/
  letI : NeZero e := ⟨he⟩
  /-
    R : Type u
    inst✝⁵ : CommRing R
    S : Type v
    inst✝⁴ : CommRing S
    p : Ideal R
    P : Ideal S
    inst✝³ : Algebra R S
    inst✝² : IsDedekindDomain S
    hP0 : Ne P Bot.bot
    inst✝¹ : p.IsMaximal
    inst✝ : P.IsPrime
    he : Ne (Ideal.ramificationIdx (algebraMap R S) p P) 0
    this : NeZero (Ideal.ramificationIdx (algebraMap R S) p P) := { out := he }
    ⊢ Eq (Module.finrank (HasQuotient.Quotient R p) (HasQuotient.Quotient S (HPow. …
  -/
  letI : Algebra (R ⧸ p) (S ⧸ P) := Quotient.algebraQuotientOfRamificationIdxNeZero p P
  /-
    R : Type u
    inst✝⁵ : CommRing R
    S : Type v
    inst✝⁴ : CommRing S
    p : Ideal R
    P : Ideal S
    inst✝³ : Algebra R S
    inst✝² : IsDedekindDomain S
    hP0 : Ne P Bot.bot
    inst✝¹ : p.IsMaximal
    inst✝ : P.IsPrime
    he : Ne (Ideal.ramificationIdx (algebraMap R S) p P) 0
    this✝ : NeZero (Ideal.ramificationIdx (algebraMap R S) p P) := { out := he }
    this : Algebra (HasQuotient.Quotient R p) (HasQuotient.Quotient S P) := Ideal. …
    ⊢ Eq (Module.finrank (HasQuotient.Quotient R p) (HasQuotient.Quotient S (HPow. …
  -/
  have hdim := rank_prime_pow_ramificationIdx _ _ hP0 he
  /-
    R : Type u
    inst✝⁵ : CommRing R
    S : Type v
    inst✝⁴ : CommRing S
    p : Ideal R
    P : Ideal S
    inst✝³ : Algebra R S
    inst✝² : IsDedekindDomain S
    hP0 : Ne P Bot.bot
    inst✝¹ : p.IsMaximal
    inst✝ : P.IsPrime
    he : Ne (Ideal.ramificationIdx (algebraMap R S) p P) 0
    this✝ : NeZero (Ideal.ramificationIdx (algebraMap R S) p P) := { out := he }
    this : Algebra (HasQuotient.Quotient R p) (HasQuotient.Quotient S P) := Ideal. …
    hdim : Eq (Module.rank (HasQuotient.Quotient R p) (HasQuotient.Quotient S (HPo …
    ⊢ Eq (Module.finrank (HasQuotient.Quotient R p) (HasQuotient.Quotient S (HPow. …
  -/
  by_cases hP : FiniteDimensional (R ⧸ p) (S ⧸ P)
    /-
      case pos
      R : Type u
      inst✝⁵ : CommRing R
      S : Type v
      inst✝⁴ : CommRing S
      p : Ideal R
      P : Ideal S
      inst✝³ : Algebra R S
      inst✝² : IsDedekindDomain S
      hP0 : Ne P Bot.bot
      inst✝¹ : p.IsMaximal
      inst✝ : P.IsPrime
      he : Ne (Ideal.ramificationIdx (algebraMap R S) p P) 0
      this✝ : NeZero (Ideal.ramificationIdx (algebraMap R S) p P) := { out := he }
      this : Algebra (HasQuotient.Quotient R p) (HasQuotient.Quotient S P) := Ideal. …
      hdim : Eq (Module.rank (HasQuotient.Quotient R p) (HasQuotient.Quotient S (HPo …
      hP : FiniteDimensional (HasQuotient.Quotient R p) (HasQuotient.Quotient S P)
      ⊢ Eq (Module.finrank (HasQuotient.Quotient R p) (HasQuotient.Quotient S (HPow. …
    -/
  · haveI := hP
    /-
      case pos
      R : Type u
      inst✝⁵ : CommRing R
      S : Type v
      inst✝⁴ : CommRing S
      p : Ideal R
      P : Ideal S
      inst✝³ : Algebra R S
      inst✝² : IsDedekindDomain S
      hP0 : Ne P Bot.bot
      inst✝¹ : p.IsMaximal
      inst✝ : P.IsPrime
      he : Ne (Ideal.ramificationIdx (algebraMap R S) p P) 0
      this✝¹ : NeZero (Ideal.ramificationIdx (algebraMap R S) p P) := { out := he }
      this✝ : Algebra (HasQuotient.Quotient R p) (HasQuotient.Quotient S P) := Ideal …
      hdim : Eq (Module.rank (HasQuotient.Quotient R p) (HasQuotient.Quotient S (HPo …
      hP this : FiniteDimensional (HasQuotient.Quotient R p) (HasQuotient.Quotient S …
      ⊢ Eq (Module.finrank (HasQuotient.Quotient R p) (HasQuotient.Quotient S (HPow. …
    -/
    haveI := (finiteDimensional_iff_of_rank_eq_nsmul he hdim).mpr hP
    /-
      case pos
      R : Type u
      inst✝⁵ : CommRing R
      S : Type v
      inst✝⁴ : CommRing S
      p : Ideal R
      P : Ideal S
      inst✝³ : Algebra R S
      inst✝² : IsDedekindDomain S
      hP0 : Ne P Bot.bot
      inst✝¹ : p.IsMaximal
      inst✝ : P.IsPrime
      he : Ne (Ideal.ramificationIdx (algebraMap R S) p P) 0
      this✝² : NeZero (Ideal.ramificationIdx (algebraMap R S) p P) := { out := he }
      this✝¹ : Algebra (HasQuotient.Quotient R p) (HasQuotient.Quotient S P) := Idea …
      hdim : Eq (Module.rank (HasQuotient.Quotient R p) (HasQuotient.Quotient S (HPo …
      hP this✝ : FiniteDimensional (HasQuotient.Quotient R p) (HasQuotient.Quotient  …
      this : FiniteDimensional (HasQuotient.Quotient R p) (HasQuotient.Quotient S (H …
      ⊢ Eq (Module.finrank (HasQuotient.Quotient R p) (HasQuotient.Quotient S (HPow. …
    -/
    apply @Nat.cast_injective Cardinal
    /-
      case pos.a
      R : Type u
      inst✝⁵ : CommRing R
      S : Type v
      inst✝⁴ : CommRing S
      p : Ideal R
      P : Ideal S
      inst✝³ : Algebra R S
      inst✝² : IsDedekindDomain S
      hP0 : Ne P Bot.bot
      inst✝¹ : p.IsMaximal
      inst✝ : P.IsPrime
      he : Ne (Ideal.ramificationIdx (algebraMap R S) p P) 0
      this✝² : NeZero (Ideal.ramificationIdx (algebraMap R S) p P) := { out := he }
      this✝¹ : Algebra (HasQuotient.Quotient R p) (HasQuotient.Quotient S P) := Idea …
      hdim : Eq (Module.rank (HasQuotient.Quotient R p) (HasQuotient.Quotient S (HPo …
      hP this✝ : FiniteDimensional (HasQuotient.Quotient R p) (HasQuotient.Quotient  …
      this : FiniteDimensional (HasQuotient.Quotient R p) (HasQuotient.Quotient S (H …
      ⊢ Eq ↑(Module.finrank (HasQuotient.Quotient R p) (HasQuotient.Quotient S (HPow …
    -/
    rw [finrank_eq_rank', Nat.cast_mul, finrank_eq_rank', hdim, nsmul_eq_mul]
    /-
      🎉 no goals
    -/
  /-
    case neg
    R : Type u
    inst✝⁵ : CommRing R
    S : Type v
    inst✝⁴ : CommRing S
    p : Ideal R
    P : Ideal S
    inst✝³ : Algebra R S
    inst✝² : IsDedekindDomain S
    hP0 : Ne P Bot.bot
    inst✝¹ : p.IsMaximal
    inst✝ : P.IsPrime
    he : Ne (Ideal.ramificationIdx (algebraMap R S) p P) 0
    this✝ : NeZero (Ideal.ramificationIdx (algebraMap R S) p P) := { out := he }
    this : Algebra (HasQuotient.Quotient R p) (HasQuotient.Quotient S P) := Ideal. …
    hdim : Eq (Module.rank (HasQuotient.Quotient R p) (HasQuotient.Quotient S (HPo …
    hP : Not (FiniteDimensional (HasQuotient.Quotient R p) (HasQuotient.Quotient S …
    ⊢ Eq (Module.finrank (HasQuotient.Quotient R p) (HasQuotient.Quotient S (HPow. …
  -/
  have hPe := mt (finiteDimensional_iff_of_rank_eq_nsmul he hdim).mp hP
  simp only [finrank_of_infinite_dimensional hP, finrank_of_infinite_dimensional hPe,
    mul_zero]


theorem Factors.ne_bot (P : (factors (map (algebraMap R S) p)).toFinset) : (P : Ideal S) ≠ ⊥ :=
  (prime_of_factor _ (Multiset.mem_toFinset.mp P.2)).ne_zero


instance Factors.isPrime (P : (factors (map (algebraMap R S) p)).toFinset) :
    IsPrime (P : Ideal S) :=
  Ideal.isPrime_of_prime (prime_of_factor _ (Multiset.mem_toFinset.mp P.2))


theorem Factors.ramificationIdx_ne_zero (P : (factors (map (algebraMap R S) p)).toFinset) :
    ramificationIdx (algebraMap R S) p P ≠ 0 :=
  IsDedekindDomain.ramificationIdx_ne_zero (ne_zero_of_mem_factors (Multiset.mem_toFinset.mp P.2))
    (Factors.isPrime p P) (Ideal.le_of_dvd (dvd_of_mem_factors (Multiset.mem_toFinset.mp P.2)))


instance Factors.fact_ramificationIdx_neZero (P : (factors (map (algebraMap R S) p)).toFinset) :
    NeZero (ramificationIdx (algebraMap R S) p P) :=
  ⟨Factors.ramificationIdx_ne_zero p P⟩


instance Factors.isScalarTower (P : (factors (map (algebraMap R S) p)).toFinset) :
    IsScalarTower R (R ⧸ p) (S ⧸ (P : Ideal S)) :=
  IsScalarTower.of_algebraMap_eq' rfl


instance Factors.liesOver [p.IsMaximal] (P : (factors (map (algebraMap R S) p)).toFinset) :
    P.1.LiesOver p :=
  ⟨(comap_eq_of_scalar_tower_quotient (algebraMap (R ⧸ p) (S ⧸ P.1)).injective).symm⟩


theorem Factors.finrank_pow_ramificationIdx [p.IsMaximal]
    (P : (factors (map (algebraMap R S) p)).toFinset) :
    finrank (R ⧸ p) (S ⧸ (P : Ideal S) ^ ramificationIdx (algebraMap R S) p P) =
      ramificationIdx (algebraMap R S) p P * inertiaDeg p (P : Ideal S) := by
  /-
    R : Type u
    inst✝⁴ : CommRing R
    S : Type v
    inst✝³ : CommRing S
    p : Ideal R
    inst✝² : IsDedekindDomain S
    inst✝¹ : Algebra R S
    inst✝ : p.IsMaximal
    P : Subtype fun x => Membership.mem (UniqueFactorizationMonoid.factors (Ideal. …
    ⊢ Eq (Module.finrank (HasQuotient.Quotient R p) (HasQuotient.Quotient S (HPow. …
  -/
  rw [finrank_prime_pow_ramificationIdx, inertiaDeg_algebraMap]
  /-
    case hP0
    R : Type u
    inst✝⁴ : CommRing R
    S : Type v
    inst✝³ : CommRing S
    p : Ideal R
    inst✝² : IsDedekindDomain S
    inst✝¹ : Algebra R S
    inst✝ : p.IsMaximal
    P : Subtype fun x => Membership.mem (UniqueFactorizationMonoid.factors (Ideal. …
    ⊢ Ne (↑P) Bot.bot
  -/
  exacts [Factors.ne_bot p P, NeZero.ne _]
  /-
    🎉 no goals
  -/


instance Factors.finiteDimensional_quotient_pow [Module.Finite R S] [p.IsMaximal]
    (P : (factors (map (algebraMap R S) p)).toFinset) :
    FiniteDimensional (R ⧸ p) (S ⧸ (P : Ideal S) ^ ramificationIdx (algebraMap R S) p P) := by
  /-
    R : Type u
    inst✝⁵ : CommRing R
    S : Type v
    inst✝⁴ : CommRing S
    f : RingHom R S
    p : Ideal R
    P✝ : Ideal S
    inst✝³ : IsDedekindDomain S
    inst✝² : Algebra R S
    inst✝¹ : Module.Finite R S
    inst✝ : p.IsMaximal
    P : Subtype fun x => Membership.mem (UniqueFactorizationMonoid.factors (Ideal. …
    ⊢ FiniteDimensional (HasQuotient.Quotient R p) (HasQuotient.Quotient S (HPow.h …
  -/
  refine .of_finrank_pos ?_
  /-
    R : Type u
    inst✝⁵ : CommRing R
    S : Type v
    inst✝⁴ : CommRing S
    f : RingHom R S
    p : Ideal R
    P✝ : Ideal S
    inst✝³ : IsDedekindDomain S
    inst✝² : Algebra R S
    inst✝¹ : Module.Finite R S
    inst✝ : p.IsMaximal
    P : Subtype fun x => Membership.mem (UniqueFactorizationMonoid.factors (Ideal. …
    ⊢ LT.lt 0 (Module.finrank (HasQuotient.Quotient R p) (HasQuotient.Quotient S ( …
  -/
  rw [pos_iff_ne_zero, Factors.finrank_pow_ramificationIdx]
  /-
    R : Type u
    inst✝⁵ : CommRing R
    S : Type v
    inst✝⁴ : CommRing S
    f : RingHom R S
    p : Ideal R
    P✝ : Ideal S
    inst✝³ : IsDedekindDomain S
    inst✝² : Algebra R S
    inst✝¹ : Module.Finite R S
    inst✝ : p.IsMaximal
    P : Subtype fun x => Membership.mem (UniqueFactorizationMonoid.factors (Ideal. …
    ⊢ Ne (HMul.hMul (Ideal.ramificationIdx (algebraMap R S) p ↑P) (p.inertiaDeg ↑P …
  -/
  exact mul_ne_zero (Factors.ramificationIdx_ne_zero p P) (inertiaDeg_pos p P.1).ne'
  /-
    🎉 no goals
  -/


/-- **Chinese remainder theorem** for a ring of integers: if the prime ideal `p : Ideal R`
factors in `S` as `∏ i, P i ^ e i`, then `S ⧸ I` factors as `Π i, R ⧸ (P i ^ e i)`. -/
noncomputable def Factors.piQuotientEquiv (p : Ideal R) (hp : map (algebraMap R S) p ≠ ⊥) :
    S ⧸ map (algebraMap R S) p ≃+*
      ∀ P : (factors (map (algebraMap R S) p)).toFinset,
        S ⧸ (P : Ideal S) ^ ramificationIdx (algebraMap R S) p P :=
  (IsDedekindDomain.quotientEquivPiFactors hp).trans <|
    @RingEquiv.piCongrRight (factors (map (algebraMap R S) p)).toFinset
      (fun P => S ⧸ (P : Ideal S) ^ (factors (map (algebraMap R S) p)).count (P : Ideal S))
      (fun P => S ⧸ (P : Ideal S) ^ ramificationIdx (algebraMap R S) p P) _ _
      fun P : (factors (map (algebraMap R S) p)).toFinset =>
      Ideal.quotEquivOfEq <| by
        rw [IsDedekindDomain.ramificationIdx_eq_factors_count hp (Factors.isPrime p P)
            (Factors.ne_bot p P)]


@[simp]
theorem Factors.piQuotientEquiv_mk (p : Ideal R) (hp : map (algebraMap R S) p ≠ ⊥) (x : S) :
    Factors.piQuotientEquiv p hp (Ideal.Quotient.mk _ x) = fun _ => Ideal.Quotient.mk _ x := rfl


@[simp]
theorem Factors.piQuotientEquiv_map (p : Ideal R) (hp : map (algebraMap R S) p ≠ ⊥) (x : R) :
    Factors.piQuotientEquiv p hp (algebraMap _ _ x) = fun _ =>
      Ideal.Quotient.mk _ (algebraMap _ _ x) := rfl


/-- **Chinese remainder theorem** for a ring of integers: if the prime ideal `p : Ideal R`
factors in `S` as `∏ i, P i ^ e i`,
then `S ⧸ I` factors `R ⧸ I`-linearly as `Π i, R ⧸ (P i ^ e i)`. -/
noncomputable def Factors.piQuotientLinearEquiv (p : Ideal R) (hp : map (algebraMap R S) p ≠ ⊥) :
    (S ⧸ map (algebraMap R S) p) ≃ₗ[R ⧸ p]
      ∀ P : (factors (map (algebraMap R S) p)).toFinset,
        S ⧸ (P : Ideal S) ^ ramificationIdx (algebraMap R S) p P :=
  { Factors.piQuotientEquiv p hp with
    map_smul' := by
      /-
        R : Type u
        inst✝³ : CommRing R
        S : Type v
        inst✝² : CommRing S
        f : RingHom R S
        p✝ : Ideal R
        P : Ideal S
        inst✝¹ : IsDedekindDomain S
        inst✝ : Algebra R S
        p : Ideal R
        hp : Ne (Ideal.map (algebraMap R S) p) Bot.bot
        ⊢ ∀ (m : HasQuotient.Quotient R p) (x : HasQuotient.Quotient S (Ideal.map (alg …
      -/
      rintro ⟨c⟩ ⟨x⟩; ext P
      simp only [Submodule.Quotient.quot_mk_eq_mk, Quotient.mk_eq_mk, Algebra.smul_def,
        Quotient.algebraMap_quotient_map_quotient, Quotient.mk_algebraMap,
        RingHomCompTriple.comp_apply, Pi.mul_apply, Pi.algebraMap_apply]
      /-
        case mk.mk.h
        R : Type u
        inst✝³ : CommRing R
        S : Type v
        inst✝² : CommRing S
        f : RingHom R S
        p✝ : Ideal R
        P✝ : Ideal S
        inst✝¹ : IsDedekindDomain S
        inst✝ : Algebra R S
        p : Ideal R
        hp : Ne (Ideal.map (algebraMap R S) p) Bot.bot
        m✝ : HasQuotient.Quotient R p
        c : R
        x✝ : HasQuotient.Quotient S (Ideal.map (algebraMap R S) p)
        x : S
        P : Subtype fun x => Membership.mem (UniqueFactorizationMonoid.factors (Ideal. …
        ⊢ Eq ((Ideal.Factors.piQuotientEquiv p hp).toFun (HMul.hMul ((algebraMap R (Ha …
      -/
      congr }
      /-
        🎉 no goals
      -/


/-- The **fundamental identity** of ramification index `e` and inertia degree `f`:
for `P` ranging over the primes lying over `p`, `∑ P, e P * f P = [Frac(S) : Frac(R)]`;
here `S` is a finite `R`-module (and thus `Frac(S) : Frac(R)` is a finite extension) and `p`
is maximal. -/
theorem sum_ramification_inertia (K L : Type*) [Field K] [Field L] [IsDedekindDomain R]
    [Algebra R K] [IsFractionRing R K] [Algebra S L] [IsFractionRing S L] [Algebra K L]
    [Algebra R L] [IsScalarTower R S L] [IsScalarTower R K L] [Module.Finite R S]
    [p.IsMaximal] (hp0 : p ≠ ⊥) :
    (∑ P ∈ (factors (map (algebraMap R S) p)).toFinset,
        ramificationIdx (algebraMap R S) p P * inertiaDeg p P) =
      finrank K L := by
  /-
    R : Type u
    inst✝¹⁶ : CommRing R
    S : Type v
    inst✝¹⁵ : CommRing S
    p : Ideal R
    inst✝¹⁴ : IsDedekindDomain S
    inst✝¹³ : Algebra R S
    K : Type u_1
    L : Type u_2
    inst✝¹² : Field K
    inst✝¹¹ : Field L
    inst✝¹⁰ : IsDedekindDomain R
    inst✝⁹ : Algebra R K
    inst✝⁸ : IsFractionRing R K
    inst✝⁷ : Algebra S L
    inst✝⁶ : IsFractionRing S L
    inst✝⁵ : Algebra K L
    inst✝⁴ : Algebra R L
    inst✝³ : IsScalarTower R S L
    inst✝² : IsScalarTower R K L
    inst✝¹ : Module.Finite R S
    inst✝ : p.IsMaximal
    hp0 : Ne p Bot.bot
    ⊢ Eq ((UniqueFactorizationMonoid.factors (Ideal.map (algebraMap R S) p)).toFin …
  -/
  set e := ramificationIdx (algebraMap R S) p
  /-
    R : Type u
    inst✝¹⁶ : CommRing R
    S : Type v
    inst✝¹⁵ : CommRing S
    p : Ideal R
    inst✝¹⁴ : IsDedekindDomain S
    inst✝¹³ : Algebra R S
    K : Type u_1
    L : Type u_2
    inst✝¹² : Field K
    inst✝¹¹ : Field L
    inst✝¹⁰ : IsDedekindDomain R
    inst✝⁹ : Algebra R K
    inst✝⁸ : IsFractionRing R K
    inst✝⁷ : Algebra S L
    inst✝⁶ : IsFractionRing S L
    inst✝⁵ : Algebra K L
    inst✝⁴ : Algebra R L
    inst✝³ : IsScalarTower R S L
    inst✝² : IsScalarTower R K L
    inst✝¹ : Module.Finite R S
    inst✝ : p.IsMaximal
    hp0 : Ne p Bot.bot
    e : Ideal S → Nat := Ideal.ramificationIdx (algebraMap R S) p
    ⊢ Eq ((UniqueFactorizationMonoid.factors (Ideal.map (algebraMap R S) p)).toFin …
  -/
  set f := inertiaDeg p (S := S)
  calc
    (∑ P ∈ (factors (map (algebraMap R S) p)).toFinset, e P * f P) =
        ∑ P ∈ (factors (map (algebraMap R S) p)).toFinset.attach,
          finrank (R ⧸ p) (S ⧸ (P : Ideal S) ^ e P) := ?_
    _ = finrank (R ⧸ p)
          (∀ P : (factors (map (algebraMap R S) p)).toFinset, S ⧸ (P : Ideal S) ^ e P) :=
      (finrank_pi_fintype (R ⧸ p)).symm
    _ = finrank (R ⧸ p) (S ⧸ map (algebraMap R S) p) := ?_
    _ = finrank K L := ?_
    /-
      case calc_1
      R : Type u
      inst✝¹⁶ : CommRing R
      S : Type v
      inst✝¹⁵ : CommRing S
      p : Ideal R
      inst✝¹⁴ : IsDedekindDomain S
      inst✝¹³ : Algebra R S
      K : Type u_1
      L : Type u_2
      inst✝¹² : Field K
      inst✝¹¹ : Field L
      inst✝¹⁰ : IsDedekindDomain R
      inst✝⁹ : Algebra R K
      inst✝⁸ : IsFractionRing R K
      inst✝⁷ : Algebra S L
      inst✝⁶ : IsFractionRing S L
      inst✝⁵ : Algebra K L
      inst✝⁴ : Algebra R L
      inst✝³ : IsScalarTower R S L
      inst✝² : IsScalarTower R K L
      inst✝¹ : Module.Finite R S
      inst✝ : p.IsMaximal
      hp0 : Ne p Bot.bot
      e : Ideal S → Nat := Ideal.ramificationIdx (algebraMap R S) p
      f : Ideal S → [inst : Algebra R S] → Nat := p.inertiaDeg
      ⊢ Eq ((UniqueFactorizationMonoid.factors (Ideal.map (algebraMap R S) p)).toFin …
    -/
  · rw [← Finset.sum_attach]
    /-
      case calc_1
      R : Type u
      inst✝¹⁶ : CommRing R
      S : Type v
      inst✝¹⁵ : CommRing S
      p : Ideal R
      inst✝¹⁴ : IsDedekindDomain S
      inst✝¹³ : Algebra R S
      K : Type u_1
      L : Type u_2
      inst✝¹² : Field K
      inst✝¹¹ : Field L
      inst✝¹⁰ : IsDedekindDomain R
      inst✝⁹ : Algebra R K
      inst✝⁸ : IsFractionRing R K
      inst✝⁷ : Algebra S L
      inst✝⁶ : IsFractionRing S L
      inst✝⁵ : Algebra K L
      inst✝⁴ : Algebra R L
      inst✝³ : IsScalarTower R S L
      inst✝² : IsScalarTower R K L
      inst✝¹ : Module.Finite R S
      inst✝ : p.IsMaximal
      hp0 : Ne p Bot.bot
      e : Ideal S → Nat := Ideal.ramificationIdx (algebraMap R S) p
      f : Ideal S → [inst : Algebra R S] → Nat := p.inertiaDeg
      ⊢ Eq ((UniqueFactorizationMonoid.factors (Ideal.map (algebraMap R S) p)).toFin …
    -/
    refine Finset.sum_congr rfl fun P _ => ?_
    /-
      case calc_1
      R : Type u
      inst✝¹⁶ : CommRing R
      S : Type v
      inst✝¹⁵ : CommRing S
      p : Ideal R
      inst✝¹⁴ : IsDedekindDomain S
      inst✝¹³ : Algebra R S
      K : Type u_1
      L : Type u_2
      inst✝¹² : Field K
      inst✝¹¹ : Field L
      inst✝¹⁰ : IsDedekindDomain R
      inst✝⁹ : Algebra R K
      inst✝⁸ : IsFractionRing R K
      inst✝⁷ : Algebra S L
      inst✝⁶ : IsFractionRing S L
      inst✝⁵ : Algebra K L
      inst✝⁴ : Algebra R L
      inst✝³ : IsScalarTower R S L
      inst✝² : IsScalarTower R K L
      inst✝¹ : Module.Finite R S
      inst✝ : p.IsMaximal
      hp0 : Ne p Bot.bot
      e : Ideal S → Nat := Ideal.ramificationIdx (algebraMap R S) p
      f : Ideal S → [inst : Algebra R S] → Nat := p.inertiaDeg
      P : Subtype fun x => Membership.mem (UniqueFactorizationMonoid.factors (Ideal. …
      x✝ : Membership.mem (UniqueFactorizationMonoid.factors (Ideal.map (algebraMap  …
      ⊢ Eq (HMul.hMul (e ↑P) (f ↑P)) (Module.finrank (HasQuotient.Quotient R p) (Has …
    -/
    rw [Factors.finrank_pow_ramificationIdx]
    /-
      🎉 no goals
    -/
    /-
      case calc_2
      R : Type u
      inst✝¹⁶ : CommRing R
      S : Type v
      inst✝¹⁵ : CommRing S
      p : Ideal R
      inst✝¹⁴ : IsDedekindDomain S
      inst✝¹³ : Algebra R S
      K : Type u_1
      L : Type u_2
      inst✝¹² : Field K
      inst✝¹¹ : Field L
      inst✝¹⁰ : IsDedekindDomain R
      inst✝⁹ : Algebra R K
      inst✝⁸ : IsFractionRing R K
      inst✝⁷ : Algebra S L
      inst✝⁶ : IsFractionRing S L
      inst✝⁵ : Algebra K L
      inst✝⁴ : Algebra R L
      inst✝³ : IsScalarTower R S L
      inst✝² : IsScalarTower R K L
      inst✝¹ : Module.Finite R S
      inst✝ : p.IsMaximal
      hp0 : Ne p Bot.bot
      e : Ideal S → Nat := Ideal.ramificationIdx (algebraMap R S) p
      f : Ideal S → [inst : Algebra R S] → Nat := p.inertiaDeg
      ⊢ Eq (Module.finrank (HasQuotient.Quotient R p) ((P : Subtype fun x => Members …
    -/
  · refine LinearEquiv.finrank_eq (Factors.piQuotientLinearEquiv S p ?_).symm
    rwa [Ne, Ideal.map_eq_bot_iff_le_ker, (RingHom.injective_iff_ker_eq_bot _).mp <|
      algebraMap_injective_of_field_isFractionRing R S K L, le_bot_iff]
    /-
      case calc_3
      R : Type u
      inst✝¹⁶ : CommRing R
      S : Type v
      inst✝¹⁵ : CommRing S
      p : Ideal R
      inst✝¹⁴ : IsDedekindDomain S
      inst✝¹³ : Algebra R S
      K : Type u_1
      L : Type u_2
      inst✝¹² : Field K
      inst✝¹¹ : Field L
      inst✝¹⁰ : IsDedekindDomain R
      inst✝⁹ : Algebra R K
      inst✝⁸ : IsFractionRing R K
      inst✝⁷ : Algebra S L
      inst✝⁶ : IsFractionRing S L
      inst✝⁵ : Algebra K L
      inst✝⁴ : Algebra R L
      inst✝³ : IsScalarTower R S L
      inst✝² : IsScalarTower R K L
      inst✝¹ : Module.Finite R S
      inst✝ : p.IsMaximal
      hp0 : Ne p Bot.bot
      e : Ideal S → Nat := Ideal.ramificationIdx (algebraMap R S) p
      f : Ideal S → [inst : Algebra R S] → Nat := p.inertiaDeg
      ⊢ Eq (Module.finrank (HasQuotient.Quotient R p) (HasQuotient.Quotient S (Ideal …
    -/
  · exact finrank_quotient_map p K L
    /-
      🎉 no goals
    -/


theorem ramificationIdx_tower [IsDedekindDomain S] [IsDedekindDomain T] {f : R →+* S} {g : S →+* T}
    {p : Ideal R} {P : Ideal S} {Q : Ideal T} [hpm : P.IsPrime] [hqm : Q.IsPrime]
    (hg0 : map g P ≠ ⊥) (hfg : map (g.comp f) p ≠ ⊥) (hg : map g P ≤ Q) :
    ramificationIdx (g.comp f) p Q = ramificationIdx f p P * ramificationIdx g P Q := by
  classical
  have hf0 : map f p ≠ ⊥ :=
    ne_bot_of_map_ne_bot (Eq.mp (congrArg (fun I ↦ I ≠ ⊥) (map_map f g).symm) hfg)
  have hp0 : P ≠ ⊥ := ne_bot_of_map_ne_bot hg0
  have hq0 : Q ≠ ⊥ := ne_bot_of_le_ne_bot hg0 hg
  letI : P.IsMaximal := Ring.DimensionLEOne.maximalOfPrime hp0 hpm
  rw [IsDedekindDomain.ramificationIdx_eq_normalizedFactors_count hf0 hpm hp0,
    IsDedekindDomain.ramificationIdx_eq_normalizedFactors_count hg0 hqm hq0,
    IsDedekindDomain.ramificationIdx_eq_normalizedFactors_count hfg hqm hq0, ← map_map]
  rcases eq_prime_pow_mul_coprime hf0 P with ⟨I, hcp, heq⟩
  have hcp : ⊤ = map g P ⊔ map g I := by rw [← map_sup, hcp, map_top g]
  have hntq : ¬ ⊤ ≤ Q := fun ht ↦ IsPrime.ne_top hqm (Iff.mpr (eq_top_iff_one Q) (ht trivial))
  nth_rw 1 [heq, map_mul, Ideal.map_pow, normalizedFactors_mul (pow_ne_zero _ hg0) <| by
    by_contra h
    simp only [h, Submodule.zero_eq_bot, bot_le, sup_of_le_left] at hcp
    exact hntq (hcp.trans_le hg), Multiset.count_add, normalizedFactors_pow, Multiset.count_nsmul]
  exact add_right_eq_self.mpr <| Decidable.byContradiction fun h ↦ hntq <| hcp.trans_le <|
    sup_le hg <| le_of_dvd <| dvd_of_mem_normalizedFactors <| Multiset.count_ne_zero.mp h


/-- Let `T / S / R` be a tower of algebras, `p, P, Q` be ideals in `R, S, T` respectively,
  and `P` and `Q` are prime. If `P = Q ∩ S`, then `e (Q | p) = e (P | p) * e (Q | P)`. -/
theorem ramificationIdx_algebra_tower [IsDedekindDomain S] [IsDedekindDomain T]
    {p : Ideal R} {P : Ideal S} {Q : Ideal T} [hpm : P.IsPrime] [hqm : Q.IsPrime]
    (hg0 : map (algebraMap S T) P ≠ ⊥)
    (hfg : map (algebraMap R T) p ≠ ⊥) (hg : map (algebraMap S T) P ≤ Q) :
    ramificationIdx (algebraMap R T) p Q =
    ramificationIdx (algebraMap R S) p P * ramificationIdx (algebraMap S T) P Q := by
  /-
    R : Type u_1
    S : Type u_2
    T : Type u_3
    inst✝⁸ : CommRing R
    inst✝⁷ : CommRing S
    inst✝⁶ : CommRing T
    inst✝⁵ : Algebra R S
    inst✝⁴ : Algebra S T
    inst✝³ : Algebra R T
    inst✝² : IsScalarTower R S T
    inst✝¹ : IsDedekindDomain S
    inst✝ : IsDedekindDomain T
    p : Ideal R
    P : Ideal S
    Q : Ideal T
    hpm : P.IsPrime
    hqm : Q.IsPrime
    hg0 : Ne (Ideal.map (algebraMap S T) P) Bot.bot
    hfg : Ne (Ideal.map (algebraMap R T) p) Bot.bot
    hg : LE.le (Ideal.map (algebraMap S T) P) Q
    ⊢ Eq (Ideal.ramificationIdx (algebraMap R T) p Q) (HMul.hMul (Ideal.ramificati …
  -/
  rw [IsScalarTower.algebraMap_eq R S T] at hfg ⊢
  /-
    R : Type u_1
    S : Type u_2
    T : Type u_3
    inst✝⁸ : CommRing R
    inst✝⁷ : CommRing S
    inst✝⁶ : CommRing T
    inst✝⁵ : Algebra R S
    inst✝⁴ : Algebra S T
    inst✝³ : Algebra R T
    inst✝² : IsScalarTower R S T
    inst✝¹ : IsDedekindDomain S
    inst✝ : IsDedekindDomain T
    p : Ideal R
    P : Ideal S
    Q : Ideal T
    hpm : P.IsPrime
    hqm : Q.IsPrime
    hg0 : Ne (Ideal.map (algebraMap S T) P) Bot.bot
    hfg : Ne (Ideal.map ((algebraMap S T).comp (algebraMap R S)) p) Bot.bot
    hg : LE.le (Ideal.map (algebraMap S T) P) Q
    ⊢ Eq (Ideal.ramificationIdx ((algebraMap S T).comp (algebraMap R S)) p Q) (HMu …
  -/
  exact ramificationIdx_tower hg0 hfg hg
  /-
    🎉 no goals
  -/


/-- Let `T / S / R` be a tower of algebras, `p, P, I` be ideals in `R, S, T`, respectively,
  and `p` and `P` are maximal. If `p = P ∩ S` and `P = I ∩ S`,
  then `f (I | p) = f (P | p) * f (I | P)`. -/
theorem inertiaDeg_algebra_tower (p : Ideal R) (P : Ideal S) (I : Ideal T) [p.IsMaximal]
    [P.IsMaximal] [P.LiesOver p] [I.LiesOver P] : inertiaDeg p I =
    inertiaDeg p P * inertiaDeg P I := by
  /-
    R : Type u_1
    S : Type u_2
    T : Type u_3
    inst✝¹⁰ : CommRing R
    inst✝⁹ : CommRing S
    inst✝⁸ : CommRing T
    inst✝⁷ : Algebra R S
    inst✝⁶ : Algebra S T
    inst✝⁵ : Algebra R T
    inst✝⁴ : IsScalarTower R S T
    p : Ideal R
    P : Ideal S
    I : Ideal T
    inst✝³ : p.IsMaximal
    inst✝² : P.IsMaximal
    inst✝¹ : P.LiesOver p
    inst✝ : I.LiesOver P
    ⊢ Eq (p.inertiaDeg I) (HMul.hMul (p.inertiaDeg P) (P.inertiaDeg I))
  -/
  have h₁ := P.over_def p
  /-
    R : Type u_1
    S : Type u_2
    T : Type u_3
    inst✝¹⁰ : CommRing R
    inst✝⁹ : CommRing S
    inst✝⁸ : CommRing T
    inst✝⁷ : Algebra R S
    inst✝⁶ : Algebra S T
    inst✝⁵ : Algebra R T
    inst✝⁴ : IsScalarTower R S T
    p : Ideal R
    P : Ideal S
    I : Ideal T
    inst✝³ : p.IsMaximal
    inst✝² : P.IsMaximal
    inst✝¹ : P.LiesOver p
    inst✝ : I.LiesOver P
    h₁ : Eq p (Ideal.under R P)
    ⊢ Eq (p.inertiaDeg I) (HMul.hMul (p.inertiaDeg P) (P.inertiaDeg I))
  -/
  have h₂ := I.over_def P
  /-
    R : Type u_1
    S : Type u_2
    T : Type u_3
    inst✝¹⁰ : CommRing R
    inst✝⁹ : CommRing S
    inst✝⁸ : CommRing T
    inst✝⁷ : Algebra R S
    inst✝⁶ : Algebra S T
    inst✝⁵ : Algebra R T
    inst✝⁴ : IsScalarTower R S T
    p : Ideal R
    P : Ideal S
    I : Ideal T
    inst✝³ : p.IsMaximal
    inst✝² : P.IsMaximal
    inst✝¹ : P.LiesOver p
    inst✝ : I.LiesOver P
    h₁ : Eq p (Ideal.under R P)
    h₂ : Eq P (Ideal.under S I)
    ⊢ Eq (p.inertiaDeg I) (HMul.hMul (p.inertiaDeg P) (P.inertiaDeg I))
  -/
  have h₃ := (LiesOver.trans I P p).over
  /-
    R : Type u_1
    S : Type u_2
    T : Type u_3
    inst✝¹⁰ : CommRing R
    inst✝⁹ : CommRing S
    inst✝⁸ : CommRing T
    inst✝⁷ : Algebra R S
    inst✝⁶ : Algebra S T
    inst✝⁵ : Algebra R T
    inst✝⁴ : IsScalarTower R S T
    p : Ideal R
    P : Ideal S
    I : Ideal T
    inst✝³ : p.IsMaximal
    inst✝² : P.IsMaximal
    inst✝¹ : P.LiesOver p
    inst✝ : I.LiesOver P
    h₁ : Eq p (Ideal.under R P)
    h₂ : Eq P (Ideal.under S I)
    h₃ : Eq p (Ideal.under R I)
    ⊢ Eq (p.inertiaDeg I) (HMul.hMul (p.inertiaDeg P) (P.inertiaDeg I))
  -/
  simp only [inertiaDeg, dif_pos h₁.symm, dif_pos h₂.symm, dif_pos h₃.symm]
  /-
    R : Type u_1
    S : Type u_2
    T : Type u_3
    inst✝¹⁰ : CommRing R
    inst✝⁹ : CommRing S
    inst✝⁸ : CommRing T
    inst✝⁷ : Algebra R S
    inst✝⁶ : Algebra S T
    inst✝⁵ : Algebra R T
    inst✝⁴ : IsScalarTower R S T
    p : Ideal R
    P : Ideal S
    I : Ideal T
    inst✝³ : p.IsMaximal
    inst✝² : P.IsMaximal
    inst✝¹ : P.LiesOver p
    inst✝ : I.LiesOver P
    h₁ : Eq p (Ideal.under R P)
    h₂ : Eq P (Ideal.under S I)
    h₃ : Eq p (Ideal.under R I)
    ⊢ Eq (Module.finrank (HasQuotient.Quotient R p) (HasQuotient.Quotient T I)) (H …
  -/
  letI : Algebra (R ⧸ p) (S ⧸ P) := Ideal.Quotient.algebraQuotientOfLEComap h₁.le
  /-
    R : Type u_1
    S : Type u_2
    T : Type u_3
    inst✝¹⁰ : CommRing R
    inst✝⁹ : CommRing S
    inst✝⁸ : CommRing T
    inst✝⁷ : Algebra R S
    inst✝⁶ : Algebra S T
    inst✝⁵ : Algebra R T
    inst✝⁴ : IsScalarTower R S T
    p : Ideal R
    P : Ideal S
    I : Ideal T
    inst✝³ : p.IsMaximal
    inst✝² : P.IsMaximal
    inst✝¹ : P.LiesOver p
    inst✝ : I.LiesOver P
    h₁ : Eq p (Ideal.under R P)
    h₂ : Eq P (Ideal.under S I)
    h₃ : Eq p (Ideal.under R I)
    this : Algebra (HasQuotient.Quotient R p) (HasQuotient.Quotient S P) := Ideal. …
    ⊢ Eq (Module.finrank (HasQuotient.Quotient R p) (HasQuotient.Quotient T I)) (H …
  -/
  letI : Algebra (S ⧸ P) (T ⧸ I) := Ideal.Quotient.algebraQuotientOfLEComap h₂.le
  /-
    R : Type u_1
    S : Type u_2
    T : Type u_3
    inst✝¹⁰ : CommRing R
    inst✝⁹ : CommRing S
    inst✝⁸ : CommRing T
    inst✝⁷ : Algebra R S
    inst✝⁶ : Algebra S T
    inst✝⁵ : Algebra R T
    inst✝⁴ : IsScalarTower R S T
    p : Ideal R
    P : Ideal S
    I : Ideal T
    inst✝³ : p.IsMaximal
    inst✝² : P.IsMaximal
    inst✝¹ : P.LiesOver p
    inst✝ : I.LiesOver P
    h₁ : Eq p (Ideal.under R P)
    h₂ : Eq P (Ideal.under S I)
    h₃ : Eq p (Ideal.under R I)
    this✝ : Algebra (HasQuotient.Quotient R p) (HasQuotient.Quotient S P) := Ideal …
    this : Algebra (HasQuotient.Quotient S P) (HasQuotient.Quotient T I) := Ideal. …
    ⊢ Eq (Module.finrank (HasQuotient.Quotient R p) (HasQuotient.Quotient T I)) (H …
  -/
  letI : Algebra (R ⧸ p) (T ⧸ I) := Ideal.Quotient.algebraQuotientOfLEComap h₃.le
  letI : IsScalarTower (R ⧸ p) (S ⧸ P) (T ⧸ I) := IsScalarTower.of_algebraMap_eq <| by
    rintro ⟨x⟩; exact congr_arg _ (IsScalarTower.algebraMap_apply R S T x)
  /-
    R : Type u_1
    S : Type u_2
    T : Type u_3
    inst✝¹⁰ : CommRing R
    inst✝⁹ : CommRing S
    inst✝⁸ : CommRing T
    inst✝⁷ : Algebra R S
    inst✝⁶ : Algebra S T
    inst✝⁵ : Algebra R T
    inst✝⁴ : IsScalarTower R S T
    p : Ideal R
    P : Ideal S
    I : Ideal T
    inst✝³ : p.IsMaximal
    inst✝² : P.IsMaximal
    inst✝¹ : P.LiesOver p
    inst✝ : I.LiesOver P
    h₁ : Eq p (Ideal.under R P)
    h₂ : Eq P (Ideal.under S I)
    h₃ : Eq p (Ideal.under R I)
    this✝² : Algebra (HasQuotient.Quotient R p) (HasQuotient.Quotient S P) := Idea …
    this✝¹ : Algebra (HasQuotient.Quotient S P) (HasQuotient.Quotient T I) := Idea …
    this✝ : Algebra (HasQuotient.Quotient R p) (HasQuotient.Quotient T I) := Ideal …
    this : IsScalarTower (HasQuotient.Quotient R p) (HasQuotient.Quotient S P) (Ha …
    ⊢ Eq (Module.finrank (HasQuotient.Quotient R p) (HasQuotient.Quotient T I)) (H …
  -/
  exact (finrank_mul_finrank (R ⧸ p) (S ⧸ P) (T ⧸ I)).symm
  /-
    🎉 no goals
  -/


@[deprecated (since := "2024-12-09")] alias inertiaDeg_tower := inertiaDeg_algebra_tower


