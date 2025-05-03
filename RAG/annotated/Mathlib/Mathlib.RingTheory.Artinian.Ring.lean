@[stacks 00J8]
theorem isNilpotent_jacobson_bot : IsNilpotent (Ideal.jacobson (⊥ : Ideal R)) := by
  /-
    R : Type u_1
    inst✝¹ : CommRing R
    inst✝ : IsArtinianRing R
    ⊢ IsNilpotent Bot.bot.jacobson
  -/
  let Jac := Ideal.jacobson (⊥ : Ideal R)
  /-
    R : Type u_1
    inst✝¹ : CommRing R
    inst✝ : IsArtinianRing R
    Jac : Ideal R := Bot.bot.jacobson
    ⊢ IsNilpotent Bot.bot.jacobson
  -/
  let f : ℕ →o (Ideal R)ᵒᵈ := ⟨fun n => Jac ^ n, fun _ _ h => Ideal.pow_le_pow_right h⟩
  /-
    R : Type u_1
    inst✝¹ : CommRing R
    inst✝ : IsArtinianRing R
    Jac : Ideal R := Bot.bot.jacobson
    f : OrderHom Nat (OrderDual (Ideal R)) := { toFun := fun n => HPow.hPow Jac n, …
    ⊢ IsNilpotent Bot.bot.jacobson
  -/
  obtain ⟨n, hn⟩ : ∃ n, ∀ m, n ≤ m → Jac ^ n = Jac ^ m := IsArtinian.monotone_stabilizes f
  /-
    case intro
    R : Type u_1
    inst✝¹ : CommRing R
    inst✝ : IsArtinianRing R
    Jac : Ideal R := Bot.bot.jacobson
    f : OrderHom Nat (OrderDual (Ideal R)) := { toFun := fun n => HPow.hPow Jac n, …
    n : Nat
    hn : ∀ (m : Nat), LE.le n m → Eq (HPow.hPow Jac n) (HPow.hPow Jac m)
    ⊢ IsNilpotent Bot.bot.jacobson
  -/
  refine ⟨n, ?_⟩
  /-
    case intro
    R : Type u_1
    inst✝¹ : CommRing R
    inst✝ : IsArtinianRing R
    Jac : Ideal R := Bot.bot.jacobson
    f : OrderHom Nat (OrderDual (Ideal R)) := { toFun := fun n => HPow.hPow Jac n, …
    n : Nat
    hn : ∀ (m : Nat), LE.le n m → Eq (HPow.hPow Jac n) (HPow.hPow Jac m)
    ⊢ Eq (HPow.hPow Bot.bot.jacobson n) 0
  -/
  let J : Ideal R := annihilator (Jac ^ n)
  suffices J = ⊤ by
    have hJ : J • Jac ^ n = ⊥ := annihilator_smul (Jac ^ n)
    simpa only [this, top_smul, Ideal.zero_eq_bot] using hJ
  /-
    case intro
    R : Type u_1
    inst✝¹ : CommRing R
    inst✝ : IsArtinianRing R
    Jac : Ideal R := Bot.bot.jacobson
    f : OrderHom Nat (OrderDual (Ideal R)) := { toFun := fun n => HPow.hPow Jac n, …
    n : Nat
    hn : ∀ (m : Nat), LE.le n m → Eq (HPow.hPow Jac n) (HPow.hPow Jac m)
    J : Ideal R := Submodule.annihilator (HPow.hPow Jac n)
    ⊢ Eq J Top.top
  -/
  by_contra hJ
  /-
    case intro
    R : Type u_1
    inst✝¹ : CommRing R
    inst✝ : IsArtinianRing R
    Jac : Ideal R := Bot.bot.jacobson
    f : OrderHom Nat (OrderDual (Ideal R)) := { toFun := fun n => HPow.hPow Jac n, …
    n : Nat
    hn : ∀ (m : Nat), LE.le n m → Eq (HPow.hPow Jac n) (HPow.hPow Jac m)
    J : Ideal R := Submodule.annihilator (HPow.hPow Jac n)
    hJ : Not (Eq J Top.top)
    ⊢ False
  -/
  change J ≠ ⊤ at hJ
  rcases IsArtinian.set_has_minimal { J' : Ideal R | J < J' } ⟨⊤, hJ.lt_top⟩ with
    ⟨J', hJJ' : J < J', hJ' : ∀ I, J < I → ¬I < J'⟩
  /-
    case intro.intro.intro
    R : Type u_1
    inst✝¹ : CommRing R
    inst✝ : IsArtinianRing R
    Jac : Ideal R := Bot.bot.jacobson
    f : OrderHom Nat (OrderDual (Ideal R)) := { toFun := fun n => HPow.hPow Jac n, …
    n : Nat
    hn : ∀ (m : Nat), LE.le n m → Eq (HPow.hPow Jac n) (HPow.hPow Jac m)
    J : Ideal R := Submodule.annihilator (HPow.hPow Jac n)
    hJ : Ne J Top.top
    J' : Submodule R R
    hJJ' : LT.lt J J'
    hJ' : ∀ (I : Ideal R), LT.lt J I → Not (LT.lt I J')
    ⊢ False
  -/
  rcases SetLike.exists_of_lt hJJ' with ⟨x, hxJ', hxJ⟩
  obtain rfl : J ⊔ Ideal.span {x} = J' := by
    apply eq_of_le_of_not_lt _ (hJ' (J ⊔ Ideal.span {x}) _)
    · exact sup_le hJJ'.le (span_le.2 (singleton_subset_iff.2 hxJ'))
    · rw [SetLike.lt_iff_le_and_exists]
      exact ⟨le_sup_left, ⟨x, mem_sup_right (mem_span_singleton_self x), hxJ⟩⟩
  have : J ⊔ Jac • Ideal.span {x} ≤ J ⊔ Ideal.span {x} :=
    sup_le_sup_left (smul_le.2 fun _ _ _ => Submodule.smul_mem _ _) _
  have : Jac * Ideal.span {x} ≤ J := by -- Need version 4 of Nakayama's lemma on Stacks
    by_contra H
    refine H (Ideal.mul_le_left.trans (le_of_le_smul_of_le_jacobson_bot (fg_span_singleton _) le_rfl
      (le_sup_right.trans_eq (this.eq_of_not_lt (hJ' _ ?_)).symm)))
    exact lt_of_le_of_ne le_sup_left fun h => H <| h.symm ▸ le_sup_right
  have : Ideal.span {x} * Jac ^ (n + 1) ≤ ⊥ := calc
    Ideal.span {x} * Jac ^ (n + 1) = Ideal.span {x} * Jac * Jac ^ n := by
      rw [pow_succ', ← mul_assoc]
    _ ≤ J * Jac ^ n := mul_le_mul (by rwa [mul_comm]) le_rfl
    _ = ⊥ := by simp [J]
  /-
    case intro.intro.intro.intro.intro
    R : Type u_1
    inst✝¹ : CommRing R
    inst✝ : IsArtinianRing R
    Jac : Ideal R := Bot.bot.jacobson
    f : OrderHom Nat (OrderDual (Ideal R)) := { toFun := fun n => HPow.hPow Jac n, …
    n : Nat
    hn : ∀ (m : Nat), LE.le n m → Eq (HPow.hPow Jac n) (HPow.hPow Jac m)
    J : Ideal R := Submodule.annihilator (HPow.hPow Jac n)
    hJ : Ne J Top.top
    x : R
    hxJ : Not (Membership.mem J x)
    hJJ' : LT.lt J (Max.max J (Ideal.span (Singleton.singleton x)))
    hJ' : ∀ (I : Ideal R), LT.lt J I → Not (LT.lt I (Max.max J (Ideal.span (Single …
    hxJ' : Membership.mem (Max.max J (Ideal.span (Singleton.singleton x))) x
    this✝¹ : LE.le (Max.max J (HSMul.hSMul Jac (Ideal.span (Singleton.singleton x) …
    this✝ : LE.le (HMul.hMul Jac (Ideal.span (Singleton.singleton x))) J
    this : LE.le (HMul.hMul (Ideal.span (Singleton.singleton x)) (HPow.hPow Jac (H …
    ⊢ False
  -/
  refine hxJ (mem_annihilator.2 fun y hy => (mem_bot R).1 ?_)
  /-
    case intro.intro.intro.intro.intro
    R : Type u_1
    inst✝¹ : CommRing R
    inst✝ : IsArtinianRing R
    Jac : Ideal R := Bot.bot.jacobson
    f : OrderHom Nat (OrderDual (Ideal R)) := { toFun := fun n => HPow.hPow Jac n, …
    n : Nat
    hn : ∀ (m : Nat), LE.le n m → Eq (HPow.hPow Jac n) (HPow.hPow Jac m)
    J : Ideal R := Submodule.annihilator (HPow.hPow Jac n)
    hJ : Ne J Top.top
    x : R
    hxJ : Not (Membership.mem J x)
    hJJ' : LT.lt J (Max.max J (Ideal.span (Singleton.singleton x)))
    hJ' : ∀ (I : Ideal R), LT.lt J I → Not (LT.lt I (Max.max J (Ideal.span (Single …
    hxJ' : Membership.mem (Max.max J (Ideal.span (Singleton.singleton x))) x
    this✝¹ : LE.le (Max.max J (HSMul.hSMul Jac (Ideal.span (Singleton.singleton x) …
    this✝ : LE.le (HMul.hMul Jac (Ideal.span (Singleton.singleton x))) J
    this : LE.le (HMul.hMul (Ideal.span (Singleton.singleton x)) (HPow.hPow Jac (H …
    y : R
    hy : Membership.mem (HPow.hPow Jac n) y
    ⊢ Membership.mem Bot.bot (HSMul.hSMul x y)
  -/
  refine this (mul_mem_mul (mem_span_singleton_self x) ?_)
  /-
    case intro.intro.intro.intro.intro
    R : Type u_1
    inst✝¹ : CommRing R
    inst✝ : IsArtinianRing R
    Jac : Ideal R := Bot.bot.jacobson
    f : OrderHom Nat (OrderDual (Ideal R)) := { toFun := fun n => HPow.hPow Jac n, …
    n : Nat
    hn : ∀ (m : Nat), LE.le n m → Eq (HPow.hPow Jac n) (HPow.hPow Jac m)
    J : Ideal R := Submodule.annihilator (HPow.hPow Jac n)
    hJ : Ne J Top.top
    x : R
    hxJ : Not (Membership.mem J x)
    hJJ' : LT.lt J (Max.max J (Ideal.span (Singleton.singleton x)))
    hJ' : ∀ (I : Ideal R), LT.lt J I → Not (LT.lt I (Max.max J (Ideal.span (Single …
    hxJ' : Membership.mem (Max.max J (Ideal.span (Singleton.singleton x))) x
    this✝¹ : LE.le (Max.max J (HSMul.hSMul Jac (Ideal.span (Singleton.singleton x) …
    this✝ : LE.le (HMul.hMul Jac (Ideal.span (Singleton.singleton x))) J
    this : LE.le (HMul.hMul (Ideal.span (Singleton.singleton x)) (HPow.hPow Jac (H …
    y : R
    hy : Membership.mem (HPow.hPow Jac n) y
    ⊢ Membership.mem (HPow.hPow Jac (HAdd.hAdd n 1)) y
  -/
  rwa [← hn (n + 1) (Nat.le_succ _)]
  /-
    🎉 no goals
  -/


/-- Localizing an artinian ring can only reduce the amount of elements. -/
theorem localization_surjective : Function.Surjective (algebraMap R L) := by
  /-
    R : Type u_1
    inst✝⁴ : CommRing R
    inst✝³ : IsArtinianRing R
    S : Submonoid R
    L : Type u_2
    inst✝² : CommRing L
    inst✝¹ : Algebra R L
    inst✝ : IsLocalization S L
    ⊢ Function.Surjective ⇑(algebraMap R L)
  -/
  intro r'
  /-
    R : Type u_1
    inst✝⁴ : CommRing R
    inst✝³ : IsArtinianRing R
    S : Submonoid R
    L : Type u_2
    inst✝² : CommRing L
    inst✝¹ : Algebra R L
    inst✝ : IsLocalization S L
    r' : L
    ⊢ Exists fun a => Eq ((algebraMap R L) a) r'
  -/
  obtain ⟨r₁, s, rfl⟩ := IsLocalization.mk'_surjective S r'
  -- TODO: can `rsuffices` be used to move the `exact` below before the proof of this `obtain`?
  obtain ⟨r₂, h⟩ : ∃ r : R, IsLocalization.mk' L 1 s = algebraMap R L r := by
    obtain ⟨n, r, hr⟩ := IsArtinian.exists_pow_succ_smul_dvd (s : R) (1 : R)
    use r
    rw [smul_eq_mul, smul_eq_mul, pow_succ, mul_assoc] at hr
    apply_fun algebraMap R L at hr
    simp only [map_mul] at hr
    rw [← IsLocalization.mk'_one (M := S) L, IsLocalization.mk'_eq_iff_eq, mul_one,
      Submonoid.coe_one, ← (IsLocalization.map_units L (s ^ n)).mul_left_cancel hr, map_mul]
  /-
    case intro.intro.intro
    R : Type u_1
    inst✝⁴ : CommRing R
    inst✝³ : IsArtinianRing R
    S : Submonoid R
    L : Type u_2
    inst✝² : CommRing L
    inst✝¹ : Algebra R L
    inst✝ : IsLocalization S L
    r₁ : R
    s : Subtype fun x => Membership.mem S x
    r₂ : R
    h : Eq (IsLocalization.mk' L 1 s) ((algebraMap R L) r₂)
    ⊢ Exists fun a => Eq ((algebraMap R L) a) (IsLocalization.mk' L r₁ s)
  -/
  exact ⟨r₁ * r₂, by rw [IsLocalization.mk'_eq_mul_mk'_one, map_mul, h]⟩
  /-
    🎉 no goals
  -/


theorem localization_artinian : IsArtinianRing L :=
  (localization_surjective S L).isArtinianRing


/-- `IsArtinianRing.localization_artinian` can't be made an instance, as it would make `S` + `R`
into metavariables. However, this is safe. -/
instance : IsArtinianRing (Localization S) :=
  localization_artinian S _


