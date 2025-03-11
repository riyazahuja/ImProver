lemma eventually_genWeightSpace_smul_add_eq_bot :
    ∀ᶠ (k : ℕ) in Filter.atTop, genWeightSpace M (k • χ₁ + χ₂) = ⊥ := by
  /-
    R : Type u_1
    L : Type u_2
    inst✝¹⁰ : CommRing R
    inst✝⁹ : LieRing L
    inst✝⁸ : LieAlgebra R L
    M : Type u_3
    inst✝⁷ : AddCommGroup M
    inst✝⁶ : Module R M
    inst✝⁵ : LieRingModule L M
    inst✝⁴ : LieModule R L M
    inst✝³ : LieAlgebra.IsNilpotent R L
    χ₁ χ₂ : L → R
    inst✝² : NoZeroSMulDivisors Int R
    inst✝¹ : NoZeroSMulDivisors R M
    inst✝ : IsNoetherian R M
    hχ₁ : Ne χ₁ 0
    ⊢ Filter.Eventually (fun k => Eq (LieModule.genWeightSpace M (HAdd.hAdd (HSMul …
  -/
  let f : ℕ → L → R := fun k ↦ k • χ₁ + χ₂
  suffices Injective f by
    rw [← Nat.cofinite_eq_atTop, Filter.eventually_cofinite, ← finite_image_iff this.injOn]
    apply (finite_genWeightSpace_ne_bot R L M).subset
    simp [f]
  /-
    R : Type u_1
    L : Type u_2
    inst✝¹⁰ : CommRing R
    inst✝⁹ : LieRing L
    inst✝⁸ : LieAlgebra R L
    M : Type u_3
    inst✝⁷ : AddCommGroup M
    inst✝⁶ : Module R M
    inst✝⁵ : LieRingModule L M
    inst✝⁴ : LieModule R L M
    inst✝³ : LieAlgebra.IsNilpotent R L
    χ₁ χ₂ : L → R
    inst✝² : NoZeroSMulDivisors Int R
    inst✝¹ : NoZeroSMulDivisors R M
    inst✝ : IsNoetherian R M
    hχ₁ : Ne χ₁ 0
    f : Nat → L → R := fun k => HAdd.hAdd (HSMul.hSMul k χ₁) χ₂
    ⊢ Function.Injective f
  -/
  intro k l hkl
  replace hkl : (k : ℤ) • χ₁ = (l : ℤ) • χ₁ := by
    simpa only [f, add_left_inj, natCast_zsmul] using hkl
  /-
    R : Type u_1
    L : Type u_2
    inst✝¹⁰ : CommRing R
    inst✝⁹ : LieRing L
    inst✝⁸ : LieAlgebra R L
    M : Type u_3
    inst✝⁷ : AddCommGroup M
    inst✝⁶ : Module R M
    inst✝⁵ : LieRingModule L M
    inst✝⁴ : LieModule R L M
    inst✝³ : LieAlgebra.IsNilpotent R L
    χ₁ χ₂ : L → R
    inst✝² : NoZeroSMulDivisors Int R
    inst✝¹ : NoZeroSMulDivisors R M
    inst✝ : IsNoetherian R M
    hχ₁ : Ne χ₁ 0
    f : Nat → L → R := fun k => HAdd.hAdd (HSMul.hSMul k χ₁) χ₂
    k l : Nat
    hkl : Eq (HSMul.hSMul (↑k) χ₁) (HSMul.hSMul (↑l) χ₁)
    ⊢ Eq k l
  -/
  exact Nat.cast_inj.mp <| smul_left_injective ℤ hχ₁ hkl
  /-
    🎉 no goals
  -/


lemma exists_genWeightSpace_smul_add_eq_bot :
    ∃ k > 0, genWeightSpace M (k • χ₁ + χ₂) = ⊥ :=
  (Nat.eventually_pos.and <| eventually_genWeightSpace_smul_add_eq_bot M χ₁ χ₂ hχ₁).exists


lemma exists₂_genWeightSpace_smul_add_eq_bot :
    ∃ᵉ (p < (0 : ℤ)) (q > (0 : ℤ)),
      genWeightSpace M (p • χ₁ + χ₂) = ⊥ ∧
      genWeightSpace M (q • χ₁ + χ₂) = ⊥ := by
  /-
    R : Type u_1
    L : Type u_2
    inst✝¹⁰ : CommRing R
    inst✝⁹ : LieRing L
    inst✝⁸ : LieAlgebra R L
    M : Type u_3
    inst✝⁷ : AddCommGroup M
    inst✝⁶ : Module R M
    inst✝⁵ : LieRingModule L M
    inst✝⁴ : LieModule R L M
    inst✝³ : LieAlgebra.IsNilpotent R L
    χ₁ χ₂ : L → R
    inst✝² : NoZeroSMulDivisors Int R
    inst✝¹ : NoZeroSMulDivisors R M
    inst✝ : IsNoetherian R M
    hχ₁ : Ne χ₁ 0
    ⊢ Exists fun p => And (LT.lt p 0) (Exists fun q => And (GT.gt q 0) (And (Eq (L …
  -/
  obtain ⟨q, hq₀, hq⟩ := exists_genWeightSpace_smul_add_eq_bot M χ₁ χ₂ hχ₁
  /-
    case intro.intro
    R : Type u_1
    L : Type u_2
    inst✝¹⁰ : CommRing R
    inst✝⁹ : LieRing L
    inst✝⁸ : LieAlgebra R L
    M : Type u_3
    inst✝⁷ : AddCommGroup M
    inst✝⁶ : Module R M
    inst✝⁵ : LieRingModule L M
    inst✝⁴ : LieModule R L M
    inst✝³ : LieAlgebra.IsNilpotent R L
    χ₁ χ₂ : L → R
    inst✝² : NoZeroSMulDivisors Int R
    inst✝¹ : NoZeroSMulDivisors R M
    inst✝ : IsNoetherian R M
    hχ₁ : Ne χ₁ 0
    q : Nat
    hq₀ : GT.gt q 0
    hq : Eq (LieModule.genWeightSpace M (HAdd.hAdd (HSMul.hSMul q χ₁) χ₂)) Bot.bot
    ⊢ Exists fun p => And (LT.lt p 0) (Exists fun q => And (GT.gt q 0) (And (Eq (L …
  -/
  obtain ⟨p, hp₀, hp⟩ := exists_genWeightSpace_smul_add_eq_bot M (-χ₁) χ₂ (neg_ne_zero.mpr hχ₁)
  /-
    case intro.intro.intro.intro
    R : Type u_1
    L : Type u_2
    inst✝¹⁰ : CommRing R
    inst✝⁹ : LieRing L
    inst✝⁸ : LieAlgebra R L
    M : Type u_3
    inst✝⁷ : AddCommGroup M
    inst✝⁶ : Module R M
    inst✝⁵ : LieRingModule L M
    inst✝⁴ : LieModule R L M
    inst✝³ : LieAlgebra.IsNilpotent R L
    χ₁ χ₂ : L → R
    inst✝² : NoZeroSMulDivisors Int R
    inst✝¹ : NoZeroSMulDivisors R M
    inst✝ : IsNoetherian R M
    hχ₁ : Ne χ₁ 0
    q : Nat
    hq₀ : GT.gt q 0
    hq : Eq (LieModule.genWeightSpace M (HAdd.hAdd (HSMul.hSMul q χ₁) χ₂)) Bot.bot
    p : Nat
    hp₀ : GT.gt p 0
    hp : Eq (LieModule.genWeightSpace M (HAdd.hAdd (HSMul.hSMul p (Neg.neg χ₁)) χ₂ …
    ⊢ Exists fun p => And (LT.lt p 0) (Exists fun q => And (GT.gt q 0) (And (Eq (L …
  -/
  refine ⟨-(p : ℤ), by simpa, q, by simpa, ?_, ?_⟩
    /-
      case intro.intro.intro.intro.refine_1
      R : Type u_1
      L : Type u_2
      inst✝¹⁰ : CommRing R
      inst✝⁹ : LieRing L
      inst✝⁸ : LieAlgebra R L
      M : Type u_3
      inst✝⁷ : AddCommGroup M
      inst✝⁶ : Module R M
      inst✝⁵ : LieRingModule L M
      inst✝⁴ : LieModule R L M
      inst✝³ : LieAlgebra.IsNilpotent R L
      χ₁ χ₂ : L → R
      inst✝² : NoZeroSMulDivisors Int R
      inst✝¹ : NoZeroSMulDivisors R M
      inst✝ : IsNoetherian R M
      hχ₁ : Ne χ₁ 0
      q : Nat
      hq₀ : GT.gt q 0
      hq : Eq (LieModule.genWeightSpace M (HAdd.hAdd (HSMul.hSMul q χ₁) χ₂)) Bot.bot
      p : Nat
      hp₀ : GT.gt p 0
      hp : Eq (LieModule.genWeightSpace M (HAdd.hAdd (HSMul.hSMul p (Neg.neg χ₁)) χ₂ …
      ⊢ Eq (LieModule.genWeightSpace M (HAdd.hAdd (HSMul.hSMul (Neg.neg ↑p) χ₁) χ₂)) …
    -/
  · rw [neg_smul, ← smul_neg, natCast_zsmul]
    /-
      case intro.intro.intro.intro.refine_1
      R : Type u_1
      L : Type u_2
      inst✝¹⁰ : CommRing R
      inst✝⁹ : LieRing L
      inst✝⁸ : LieAlgebra R L
      M : Type u_3
      inst✝⁷ : AddCommGroup M
      inst✝⁶ : Module R M
      inst✝⁵ : LieRingModule L M
      inst✝⁴ : LieModule R L M
      inst✝³ : LieAlgebra.IsNilpotent R L
      χ₁ χ₂ : L → R
      inst✝² : NoZeroSMulDivisors Int R
      inst✝¹ : NoZeroSMulDivisors R M
      inst✝ : IsNoetherian R M
      hχ₁ : Ne χ₁ 0
      q : Nat
      hq₀ : GT.gt q 0
      hq : Eq (LieModule.genWeightSpace M (HAdd.hAdd (HSMul.hSMul q χ₁) χ₂)) Bot.bot
      p : Nat
      hp₀ : GT.gt p 0
      hp : Eq (LieModule.genWeightSpace M (HAdd.hAdd (HSMul.hSMul p (Neg.neg χ₁)) χ₂ …
      ⊢ Eq (LieModule.genWeightSpace M (HAdd.hAdd (HSMul.hSMul p (Neg.neg χ₁)) χ₂))  …
    -/
    exact hp
    /-
      🎉 no goals
    -/
    /-
      case intro.intro.intro.intro.refine_2
      R : Type u_1
      L : Type u_2
      inst✝¹⁰ : CommRing R
      inst✝⁹ : LieRing L
      inst✝⁸ : LieAlgebra R L
      M : Type u_3
      inst✝⁷ : AddCommGroup M
      inst✝⁶ : Module R M
      inst✝⁵ : LieRingModule L M
      inst✝⁴ : LieModule R L M
      inst✝³ : LieAlgebra.IsNilpotent R L
      χ₁ χ₂ : L → R
      inst✝² : NoZeroSMulDivisors Int R
      inst✝¹ : NoZeroSMulDivisors R M
      inst✝ : IsNoetherian R M
      hχ₁ : Ne χ₁ 0
      q : Nat
      hq₀ : GT.gt q 0
      hq : Eq (LieModule.genWeightSpace M (HAdd.hAdd (HSMul.hSMul q χ₁) χ₂)) Bot.bot
      p : Nat
      hp₀ : GT.gt p 0
      hp : Eq (LieModule.genWeightSpace M (HAdd.hAdd (HSMul.hSMul p (Neg.neg χ₁)) χ₂ …
      ⊢ Eq (LieModule.genWeightSpace M (HAdd.hAdd (HSMul.hSMul (↑q) χ₁) χ₂)) Bot.bot
    -/
  · rw [natCast_zsmul]
    /-
      case intro.intro.intro.intro.refine_2
      R : Type u_1
      L : Type u_2
      inst✝¹⁰ : CommRing R
      inst✝⁹ : LieRing L
      inst✝⁸ : LieAlgebra R L
      M : Type u_3
      inst✝⁷ : AddCommGroup M
      inst✝⁶ : Module R M
      inst✝⁵ : LieRingModule L M
      inst✝⁴ : LieModule R L M
      inst✝³ : LieAlgebra.IsNilpotent R L
      χ₁ χ₂ : L → R
      inst✝² : NoZeroSMulDivisors Int R
      inst✝¹ : NoZeroSMulDivisors R M
      inst✝ : IsNoetherian R M
      hχ₁ : Ne χ₁ 0
      q : Nat
      hq₀ : GT.gt q 0
      hq : Eq (LieModule.genWeightSpace M (HAdd.hAdd (HSMul.hSMul q χ₁) χ₂)) Bot.bot
      p : Nat
      hp₀ : GT.gt p 0
      hp : Eq (LieModule.genWeightSpace M (HAdd.hAdd (HSMul.hSMul p (Neg.neg χ₁)) χ₂ …
      ⊢ Eq (LieModule.genWeightSpace M (HAdd.hAdd (HSMul.hSMul q χ₁) χ₂)) Bot.bot
    -/
    exact hq
    /-
      🎉 no goals
    -/


/-- Given two (potential) weights `χ₁` and `χ₂` together with integers `p` and `q`, it is often
useful to study the sum of weight spaces associated to the family of weights `k • χ₁ + χ₂` for
`p < k < q`. -/
def genWeightSpaceChain : LieSubmodule R L M :=
  ⨆ k ∈ Ioo p q, genWeightSpace M (k • χ₁ + χ₂)


lemma genWeightSpaceChain_def :
    genWeightSpaceChain M χ₁ χ₂ p q = ⨆ k ∈ Ioo p q, genWeightSpace M (k • χ₁ + χ₂) :=
  rfl


lemma genWeightSpaceChain_def' :
    genWeightSpaceChain M χ₁ χ₂ p q = ⨆ k ∈ Finset.Ioo p q, genWeightSpace M (k • χ₁ + χ₂) := by
  /-
    R : Type u_1
    L : Type u_2
    inst✝⁷ : CommRing R
    inst✝⁶ : LieRing L
    inst✝⁵ : LieAlgebra R L
    M : Type u_3
    inst✝⁴ : AddCommGroup M
    inst✝³ : Module R M
    inst✝² : LieRingModule L M
    inst✝¹ : LieModule R L M
    inst✝ : LieAlgebra.IsNilpotent R L
    χ₁ χ₂ : L → R
    p q : Int
    ⊢ Eq (LieModule.genWeightSpaceChain M χ₁ χ₂ p q) (iSup fun k => iSup fun h =>  …
  -/
  have : ∀ (k : ℤ), k ∈ Ioo p q ↔ k ∈ Finset.Ioo p q := by simp
  /-
    R : Type u_1
    L : Type u_2
    inst✝⁷ : CommRing R
    inst✝⁶ : LieRing L
    inst✝⁵ : LieAlgebra R L
    M : Type u_3
    inst✝⁴ : AddCommGroup M
    inst✝³ : Module R M
    inst✝² : LieRingModule L M
    inst✝¹ : LieModule R L M
    inst✝ : LieAlgebra.IsNilpotent R L
    χ₁ χ₂ : L → R
    p q : Int
    this : ∀ (k : Int), Iff (Membership.mem (Set.Ioo p q) k) (Membership.mem (Fins …
    ⊢ Eq (LieModule.genWeightSpaceChain M χ₁ χ₂ p q) (iSup fun k => iSup fun h =>  …
  -/
  simp_rw [genWeightSpaceChain_def, this]
  /-
    🎉 no goals
  -/


@[simp]
lemma genWeightSpaceChain_neg :
    genWeightSpaceChain M (-χ₁) χ₂ (-q) (-p) = genWeightSpaceChain M χ₁ χ₂ p q := by
  /-
    R : Type u_1
    L : Type u_2
    inst✝⁷ : CommRing R
    inst✝⁶ : LieRing L
    inst✝⁵ : LieAlgebra R L
    M : Type u_3
    inst✝⁴ : AddCommGroup M
    inst✝³ : Module R M
    inst✝² : LieRingModule L M
    inst✝¹ : LieModule R L M
    inst✝ : LieAlgebra.IsNilpotent R L
    χ₁ χ₂ : L → R
    p q : Int
    ⊢ Eq (LieModule.genWeightSpaceChain M (Neg.neg χ₁) χ₂ (Neg.neg q) (Neg.neg p)) …
  -/
  let e : ℤ ≃ ℤ := neg_involutive.toPerm
  /-
    R : Type u_1
    L : Type u_2
    inst✝⁷ : CommRing R
    inst✝⁶ : LieRing L
    inst✝⁵ : LieAlgebra R L
    M : Type u_3
    inst✝⁴ : AddCommGroup M
    inst✝³ : Module R M
    inst✝² : LieRingModule L M
    inst✝¹ : LieModule R L M
    inst✝ : LieAlgebra.IsNilpotent R L
    χ₁ χ₂ : L → R
    p q : Int
    e : Equiv Int Int := Function.Involutive.toPerm Neg.neg ⋯
    ⊢ Eq (LieModule.genWeightSpaceChain M (Neg.neg χ₁) χ₂ (Neg.neg q) (Neg.neg p)) …
  -/
  simp_rw [genWeightSpaceChain, ← e.biSup_comp (Ioo p q)]
  /-
    R : Type u_1
    L : Type u_2
    inst✝⁷ : CommRing R
    inst✝⁶ : LieRing L
    inst✝⁵ : LieAlgebra R L
    M : Type u_3
    inst✝⁴ : AddCommGroup M
    inst✝³ : Module R M
    inst✝² : LieRingModule L M
    inst✝¹ : LieModule R L M
    inst✝ : LieAlgebra.IsNilpotent R L
    χ₁ χ₂ : L → R
    p q : Int
    e : Equiv Int Int := Function.Involutive.toPerm Neg.neg ⋯
    ⊢ Eq (iSup fun k => iSup fun h => LieModule.genWeightSpace M (HAdd.hAdd (HSMul …
  -/
  simp [e, -mem_Ioo, neg_mem_Ioo_iff]
  /-
    🎉 no goals
  -/


lemma genWeightSpace_le_genWeightSpaceChain {k : ℤ} (hk : k ∈ Ioo p q) :
    genWeightSpace M (k • χ₁ + χ₂) ≤ genWeightSpaceChain M χ₁ χ₂ p q :=
  le_biSup (fun i ↦ genWeightSpace M (i • χ₁ + χ₂)) hk


lemma lie_mem_genWeightSpaceChain_of_genWeightSpace_eq_bot_right [LieAlgebra.IsNilpotent R H]
    (hq : genWeightSpace M (q • α + χ) = ⊥)
    {x : L} (hx : x ∈ rootSpace H α)
    {y : M} (hy : y ∈ genWeightSpaceChain M α χ p q) :
    ⁅x, y⁆ ∈ genWeightSpaceChain M α χ p q := by
  /-
    R : Type u_1
    L : Type u_2
    inst✝⁷ : CommRing R
    inst✝⁶ : LieRing L
    inst✝⁵ : LieAlgebra R L
    M : Type u_3
    inst✝⁴ : AddCommGroup M
    inst✝³ : Module R M
    inst✝² : LieRingModule L M
    inst✝¹ : LieModule R L M
    H : LieSubalgebra R L
    α χ : (Subtype fun x => Membership.mem H x) → R
    p q : Int
    inst✝ : LieAlgebra.IsNilpotent R (Subtype fun x => Membership.mem H x)
    hq : Eq (LieModule.genWeightSpace M (HAdd.hAdd (HSMul.hSMul q α) χ)) Bot.bot
    x : L
    hx : Membership.mem (LieAlgebra.rootSpace H α) x
    y : M
    hy : Membership.mem (LieModule.genWeightSpaceChain M α χ p q) y
    ⊢ Membership.mem (LieModule.genWeightSpaceChain M α χ p q) (Bracket.bracket x y)
  -/
  rw [genWeightSpaceChain, iSup_subtype'] at hy
  induction hy using LieSubmodule.iSup_induction' with
  | hN k z hz =>
    obtain ⟨k, hk⟩ := k
    suffices genWeightSpace M ((k + 1) • α + χ) ≤ genWeightSpaceChain M α χ p q by
      apply this
      -- was `simpa using [...]` and very slow
      -- (https://github.com/leanprover-community/mathlib4/issues/19751)
      simpa only [zsmul_eq_mul, Int.cast_add, Pi.intCast_def, Int.cast_one] using
        (rootSpaceWeightSpaceProduct R L H M α (k • α + χ) ((k + 1) • α + χ)
            (by rw [add_smul]; abel) (⟨x, hx⟩ ⊗ₜ ⟨z, hz⟩)).property
    rw [genWeightSpaceChain]
    rcases eq_or_ne (k + 1) q with rfl | hk'; · simp only [hq, bot_le]
    replace hk' : k + 1 ∈ Ioo p q := ⟨by linarith [hk.1], lt_of_le_of_ne hk.2 hk'⟩
    exact le_biSup (fun k ↦ genWeightSpace M (k • α + χ)) hk'
  | h0 => simp
  | hadd _ _ _ _ hz₁ hz₂ => rw [lie_add]; exact add_mem hz₁ hz₂


lemma lie_mem_genWeightSpaceChain_of_genWeightSpace_eq_bot_left [LieAlgebra.IsNilpotent R H]
    (hp : genWeightSpace M (p • α + χ) = ⊥)
    {x : L} (hx : x ∈ rootSpace H (-α))
    {y : M} (hy : y ∈ genWeightSpaceChain M α χ p q) :
    ⁅x, y⁆ ∈ genWeightSpaceChain M α χ p q := by
  /-
    R : Type u_1
    L : Type u_2
    inst✝⁷ : CommRing R
    inst✝⁶ : LieRing L
    inst✝⁵ : LieAlgebra R L
    M : Type u_3
    inst✝⁴ : AddCommGroup M
    inst✝³ : Module R M
    inst✝² : LieRingModule L M
    inst✝¹ : LieModule R L M
    H : LieSubalgebra R L
    α χ : (Subtype fun x => Membership.mem H x) → R
    p q : Int
    inst✝ : LieAlgebra.IsNilpotent R (Subtype fun x => Membership.mem H x)
    hp : Eq (LieModule.genWeightSpace M (HAdd.hAdd (HSMul.hSMul p α) χ)) Bot.bot
    x : L
    hx : Membership.mem (LieAlgebra.rootSpace H (Neg.neg α)) x
    y : M
    hy : Membership.mem (LieModule.genWeightSpaceChain M α χ p q) y
    ⊢ Membership.mem (LieModule.genWeightSpaceChain M α χ p q) (Bracket.bracket x y)
  -/
  replace hp : genWeightSpace M ((-p) • (-α) + χ) = ⊥ := by rwa [smul_neg, neg_smul, neg_neg]
  /-
    R : Type u_1
    L : Type u_2
    inst✝⁷ : CommRing R
    inst✝⁶ : LieRing L
    inst✝⁵ : LieAlgebra R L
    M : Type u_3
    inst✝⁴ : AddCommGroup M
    inst✝³ : Module R M
    inst✝² : LieRingModule L M
    inst✝¹ : LieModule R L M
    H : LieSubalgebra R L
    α χ : (Subtype fun x => Membership.mem H x) → R
    p q : Int
    inst✝ : LieAlgebra.IsNilpotent R (Subtype fun x => Membership.mem H x)
    x : L
    hx : Membership.mem (LieAlgebra.rootSpace H (Neg.neg α)) x
    y : M
    hy : Membership.mem (LieModule.genWeightSpaceChain M α χ p q) y
    hp : Eq (LieModule.genWeightSpace M (HAdd.hAdd (HSMul.hSMul (Neg.neg p) (Neg.n …
    ⊢ Membership.mem (LieModule.genWeightSpaceChain M α χ p q) (Bracket.bracket x y)
  -/
  rw [← genWeightSpaceChain_neg] at hy ⊢
  /-
    R : Type u_1
    L : Type u_2
    inst✝⁷ : CommRing R
    inst✝⁶ : LieRing L
    inst✝⁵ : LieAlgebra R L
    M : Type u_3
    inst✝⁴ : AddCommGroup M
    inst✝³ : Module R M
    inst✝² : LieRingModule L M
    inst✝¹ : LieModule R L M
    H : LieSubalgebra R L
    α χ : (Subtype fun x => Membership.mem H x) → R
    p q : Int
    inst✝ : LieAlgebra.IsNilpotent R (Subtype fun x => Membership.mem H x)
    x : L
    hx : Membership.mem (LieAlgebra.rootSpace H (Neg.neg α)) x
    y : M
    hy : Membership.mem (LieModule.genWeightSpaceChain M (Neg.neg α) χ (Neg.neg q) …
    hp : Eq (LieModule.genWeightSpace M (HAdd.hAdd (HSMul.hSMul (Neg.neg p) (Neg.n …
    ⊢ Membership.mem (LieModule.genWeightSpaceChain M (Neg.neg α) χ (Neg.neg q) (N …
  -/
  exact lie_mem_genWeightSpaceChain_of_genWeightSpace_eq_bot_right M (-α) χ (-q) (-p) hp hx hy
  /-
    🎉 no goals
  -/


lemma trace_toEnd_genWeightSpaceChain_eq_zero
    (hp : genWeightSpace M (p • α + χ) = ⊥)
    (hq : genWeightSpace M (q • α + χ) = ⊥)
    {x : H} (hx : x ∈ corootSpace α) :
    LinearMap.trace R _ (toEnd R H (genWeightSpaceChain M α χ p q) x) = 0 := by
  /-
    R : Type u_1
    L : Type u_2
    inst✝⁸ : CommRing R
    inst✝⁷ : LieRing L
    inst✝⁶ : LieAlgebra R L
    M : Type u_3
    inst✝⁵ : AddCommGroup M
    inst✝⁴ : Module R M
    inst✝³ : LieRingModule L M
    inst✝² : LieModule R L M
    H : LieSubalgebra R L
    α χ : (Subtype fun x => Membership.mem H x) → R
    p q : Int
    inst✝¹ : H.IsCartanSubalgebra
    inst✝ : IsNoetherian R L
    hp : Eq (LieModule.genWeightSpace M (HAdd.hAdd (HSMul.hSMul p α) χ)) Bot.bot
    hq : Eq (LieModule.genWeightSpace M (HAdd.hAdd (HSMul.hSMul q α) χ)) Bot.bot
    x : Subtype fun x => Membership.mem H x
    hx : Membership.mem (LieAlgebra.corootSpace α) x
    ⊢ Eq ((LinearMap.trace R (Subtype fun x => Membership.mem (LieModule.genWeight …
  -/
  rw [LieAlgebra.mem_corootSpace'] at hx
  /-
    R : Type u_1
    L : Type u_2
    inst✝⁸ : CommRing R
    inst✝⁷ : LieRing L
    inst✝⁶ : LieAlgebra R L
    M : Type u_3
    inst✝⁵ : AddCommGroup M
    inst✝⁴ : Module R M
    inst✝³ : LieRingModule L M
    inst✝² : LieModule R L M
    H : LieSubalgebra R L
    α χ : (Subtype fun x => Membership.mem H x) → R
    p q : Int
    inst✝¹ : H.IsCartanSubalgebra
    inst✝ : IsNoetherian R L
    hp : Eq (LieModule.genWeightSpace M (HAdd.hAdd (HSMul.hSMul p α) χ)) Bot.bot
    hq : Eq (LieModule.genWeightSpace M (HAdd.hAdd (HSMul.hSMul q α) χ)) Bot.bot
    x : Subtype fun x => Membership.mem H x
    hx : Membership.mem (Submodule.span R (setOf fun x => Exists fun y => And (Mem …
    ⊢ Eq ((LinearMap.trace R (Subtype fun x => Membership.mem (LieModule.genWeight …
  -/
  induction hx using Submodule.span_induction
  · next u hu =>
    obtain ⟨y, hy, z, hz, hyz⟩ := hu
    let f : Module.End R (genWeightSpaceChain M α χ p q) :=
      { toFun := fun ⟨m, hm⟩ ↦ ⟨⁅(y : L), m⁆,
          lie_mem_genWeightSpaceChain_of_genWeightSpace_eq_bot_right M α χ p q hq hy hm⟩
        map_add' := fun _ _ ↦ by simp
        map_smul' := fun t m ↦ by simp }
    let g : Module.End R (genWeightSpaceChain M α χ p q) :=
      { toFun := fun ⟨m, hm⟩ ↦ ⟨⁅(z : L), m⁆,
          lie_mem_genWeightSpaceChain_of_genWeightSpace_eq_bot_left M α χ p q hp hz hm⟩
        map_add' := fun _ _ ↦ by simp
        map_smul' := fun t m ↦ by simp }
    have hfg : toEnd R H _ u = ⁅f, g⁆ := by
      ext
      rw [toEnd_apply_apply, LieSubmodule.coe_bracket, LieSubalgebra.coe_bracket_of_module, ← hyz]
      simp only [lie_lie, LieHom.lie_apply, LinearMap.coe_mk, AddHom.coe_mk, Module.End.lie_apply,
      AddSubgroupClass.coe_sub, f, g]
    simp [hfg]
    /-
      case zero
      R : Type u_1
      L : Type u_2
      inst✝⁸ : CommRing R
      inst✝⁷ : LieRing L
      inst✝⁶ : LieAlgebra R L
      M : Type u_3
      inst✝⁵ : AddCommGroup M
      inst✝⁴ : Module R M
      inst✝³ : LieRingModule L M
      inst✝² : LieModule R L M
      H : LieSubalgebra R L
      α χ : (Subtype fun x => Membership.mem H x) → R
      p q : Int
      inst✝¹ : H.IsCartanSubalgebra
      inst✝ : IsNoetherian R L
      hp : Eq (LieModule.genWeightSpace M (HAdd.hAdd (HSMul.hSMul p α) χ)) Bot.bot
      hq : Eq (LieModule.genWeightSpace M (HAdd.hAdd (HSMul.hSMul q α) χ)) Bot.bot
      x : Subtype fun x => Membership.mem H x
      ⊢ Eq ((LinearMap.trace R (Subtype fun x => Membership.mem (LieModule.genWeight …
    -/
  · simp
    /-
      🎉 no goals
    -/
    /-
      case add
      R : Type u_1
      L : Type u_2
      inst✝⁸ : CommRing R
      inst✝⁷ : LieRing L
      inst✝⁶ : LieAlgebra R L
      M : Type u_3
      inst✝⁵ : AddCommGroup M
      inst✝⁴ : Module R M
      inst✝³ : LieRingModule L M
      inst✝² : LieModule R L M
      H : LieSubalgebra R L
      α χ : (Subtype fun x => Membership.mem H x) → R
      p q : Int
      inst✝¹ : H.IsCartanSubalgebra
      inst✝ : IsNoetherian R L
      hp : Eq (LieModule.genWeightSpace M (HAdd.hAdd (HSMul.hSMul p α) χ)) Bot.bot
      hq : Eq (LieModule.genWeightSpace M (HAdd.hAdd (HSMul.hSMul q α) χ)) Bot.bot
      x x✝ y✝ : Subtype fun x => Membership.mem H x
      hx✝ : Membership.mem (Submodule.span R (setOf fun x => Exists fun y => And (Me …
      hy✝ : Membership.mem (Submodule.span R (setOf fun x => Exists fun y => And (Me …
      a✝¹ : Eq ((LinearMap.trace R (Subtype fun x => Membership.mem (LieModule.genWe …
      a✝ : Eq ((LinearMap.trace R (Subtype fun x => Membership.mem (LieModule.genWei …
      ⊢ Eq ((LinearMap.trace R (Subtype fun x => Membership.mem (LieModule.genWeight …
    -/
  · simp_all
    /-
      🎉 no goals
    -/
    /-
      case smul
      R : Type u_1
      L : Type u_2
      inst✝⁸ : CommRing R
      inst✝⁷ : LieRing L
      inst✝⁶ : LieAlgebra R L
      M : Type u_3
      inst✝⁵ : AddCommGroup M
      inst✝⁴ : Module R M
      inst✝³ : LieRingModule L M
      inst✝² : LieModule R L M
      H : LieSubalgebra R L
      α χ : (Subtype fun x => Membership.mem H x) → R
      p q : Int
      inst✝¹ : H.IsCartanSubalgebra
      inst✝ : IsNoetherian R L
      hp : Eq (LieModule.genWeightSpace M (HAdd.hAdd (HSMul.hSMul p α) χ)) Bot.bot
      hq : Eq (LieModule.genWeightSpace M (HAdd.hAdd (HSMul.hSMul q α) χ)) Bot.bot
      x : Subtype fun x => Membership.mem H x
      a✝¹ : R
      x✝ : Subtype fun x => Membership.mem H x
      hx✝ : Membership.mem (Submodule.span R (setOf fun x => Exists fun y => And (Me …
      a✝ : Eq ((LinearMap.trace R (Subtype fun x => Membership.mem (LieModule.genWei …
      ⊢ Eq ((LinearMap.trace R (Subtype fun x => Membership.mem (LieModule.genWeight …
    -/
  · simp_all
    /-
      🎉 no goals
    -/


/-- Given a (potential) root `α` relative to a Cartan subalgebra `H`, if we restrict to the ideal
`I = corootSpace α` of `H` (informally, `I = ⁅H(α), H(-α)⁆`), we may find an
integral linear combination between `α` and any weight `χ` of a representation.

This is Proposition 4.4 from [carter2005] and is a key step in the proof that the roots of a
semisimple Lie algebra form a root system. It shows that the restriction of `α` to `I` vanishes iff
the restriction of every root to `I` vanishes (which cannot happen in a semisimple Lie algebra). -/
lemma exists_forall_mem_corootSpace_smul_add_eq_zero
    [IsDomain R] [IsPrincipalIdealRing R] [CharZero R] [NoZeroSMulDivisors R M] [IsNoetherian R M]
    (hα : α ≠ 0) (hχ : genWeightSpace M χ ≠ ⊥) :
    ∃ a b : ℤ, 0 < b ∧ ∀ x ∈ corootSpace α, (a • α + b • χ) x = 0 := by
  /-
    R : Type u_1
    L : Type u_2
    inst✝¹³ : CommRing R
    inst✝¹² : LieRing L
    inst✝¹¹ : LieAlgebra R L
    M : Type u_3
    inst✝¹⁰ : AddCommGroup M
    inst✝⁹ : Module R M
    inst✝⁸ : LieRingModule L M
    inst✝⁷ : LieModule R L M
    H : LieSubalgebra R L
    α χ : (Subtype fun x => Membership.mem H x) → R
    inst✝⁶ : H.IsCartanSubalgebra
    inst✝⁵ : IsNoetherian R L
    inst✝⁴ : IsDomain R
    inst✝³ : IsPrincipalIdealRing R
    inst✝² : CharZero R
    inst✝¹ : NoZeroSMulDivisors R M
    inst✝ : IsNoetherian R M
    hα : Ne α 0
    hχ : Ne (LieModule.genWeightSpace M χ) Bot.bot
    ⊢ Exists fun a => Exists fun b => And (LT.lt 0 b) (∀ (x : Subtype fun x => Mem …
  -/
  obtain ⟨p, hp₀, q, hq₀, hp, hq⟩ := exists₂_genWeightSpace_smul_add_eq_bot M α χ hα
  /-
    case intro.intro.intro.intro.intro
    R : Type u_1
    L : Type u_2
    inst✝¹³ : CommRing R
    inst✝¹² : LieRing L
    inst✝¹¹ : LieAlgebra R L
    M : Type u_3
    inst✝¹⁰ : AddCommGroup M
    inst✝⁹ : Module R M
    inst✝⁸ : LieRingModule L M
    inst✝⁷ : LieModule R L M
    H : LieSubalgebra R L
    α χ : (Subtype fun x => Membership.mem H x) → R
    inst✝⁶ : H.IsCartanSubalgebra
    inst✝⁵ : IsNoetherian R L
    inst✝⁴ : IsDomain R
    inst✝³ : IsPrincipalIdealRing R
    inst✝² : CharZero R
    inst✝¹ : NoZeroSMulDivisors R M
    inst✝ : IsNoetherian R M
    hα : Ne α 0
    hχ : Ne (LieModule.genWeightSpace M χ) Bot.bot
    p : Int
    hp₀ : LT.lt p 0
    q : Int
    hq₀ : GT.gt q 0
    hp : Eq (LieModule.genWeightSpace M (HAdd.hAdd (HSMul.hSMul p α) χ)) Bot.bot
    hq : Eq (LieModule.genWeightSpace M (HAdd.hAdd (HSMul.hSMul q α) χ)) Bot.bot
    ⊢ Exists fun a => Exists fun b => And (LT.lt 0 b) (∀ (x : Subtype fun x => Mem …
  -/
  let a := ∑ i ∈ Finset.Ioo p q, finrank R (genWeightSpace M (i • α + χ)) • i
  /-
    case intro.intro.intro.intro.intro
    R : Type u_1
    L : Type u_2
    inst✝¹³ : CommRing R
    inst✝¹² : LieRing L
    inst✝¹¹ : LieAlgebra R L
    M : Type u_3
    inst✝¹⁰ : AddCommGroup M
    inst✝⁹ : Module R M
    inst✝⁸ : LieRingModule L M
    inst✝⁷ : LieModule R L M
    H : LieSubalgebra R L
    α χ : (Subtype fun x => Membership.mem H x) → R
    inst✝⁶ : H.IsCartanSubalgebra
    inst✝⁵ : IsNoetherian R L
    inst✝⁴ : IsDomain R
    inst✝³ : IsPrincipalIdealRing R
    inst✝² : CharZero R
    inst✝¹ : NoZeroSMulDivisors R M
    inst✝ : IsNoetherian R M
    hα : Ne α 0
    hχ : Ne (LieModule.genWeightSpace M χ) Bot.bot
    p : Int
    hp₀ : LT.lt p 0
    q : Int
    hq₀ : GT.gt q 0
    hp : Eq (LieModule.genWeightSpace M (HAdd.hAdd (HSMul.hSMul p α) χ)) Bot.bot
    hq : Eq (LieModule.genWeightSpace M (HAdd.hAdd (HSMul.hSMul q α) χ)) Bot.bot
    a : Int := (Finset.Ioo p q).sum fun i => HSMul.hSMul (Module.finrank R (Subtyp …
    ⊢ Exists fun a => Exists fun b => And (LT.lt 0 b) (∀ (x : Subtype fun x => Mem …
  -/
  let b := ∑ i ∈ Finset.Ioo p q, finrank R (genWeightSpace M (i • α + χ))
  have hb : 0 < b := by
    replace hχ : Nontrivial (genWeightSpace M χ) := by rwa [LieSubmodule.nontrivial_iff_ne_bot]
    refine Finset.sum_pos' (fun _ _ ↦ zero_le _) ⟨0, Finset.mem_Ioo.mpr ⟨hp₀, hq₀⟩, ?_⟩
    rw [zero_smul, zero_add]
    exact finrank_pos
  /-
    case intro.intro.intro.intro.intro
    R : Type u_1
    L : Type u_2
    inst✝¹³ : CommRing R
    inst✝¹² : LieRing L
    inst✝¹¹ : LieAlgebra R L
    M : Type u_3
    inst✝¹⁰ : AddCommGroup M
    inst✝⁹ : Module R M
    inst✝⁸ : LieRingModule L M
    inst✝⁷ : LieModule R L M
    H : LieSubalgebra R L
    α χ : (Subtype fun x => Membership.mem H x) → R
    inst✝⁶ : H.IsCartanSubalgebra
    inst✝⁵ : IsNoetherian R L
    inst✝⁴ : IsDomain R
    inst✝³ : IsPrincipalIdealRing R
    inst✝² : CharZero R
    inst✝¹ : NoZeroSMulDivisors R M
    inst✝ : IsNoetherian R M
    hα : Ne α 0
    hχ : Ne (LieModule.genWeightSpace M χ) Bot.bot
    p : Int
    hp₀ : LT.lt p 0
    q : Int
    hq₀ : GT.gt q 0
    hp : Eq (LieModule.genWeightSpace M (HAdd.hAdd (HSMul.hSMul p α) χ)) Bot.bot
    hq : Eq (LieModule.genWeightSpace M (HAdd.hAdd (HSMul.hSMul q α) χ)) Bot.bot
    a : Int := (Finset.Ioo p q).sum fun i => HSMul.hSMul (Module.finrank R (Subtyp …
    b : Nat := (Finset.Ioo p q).sum fun i => Module.finrank R (Subtype fun x => Me …
    hb : LT.lt 0 b
    ⊢ Exists fun a => Exists fun b => And (LT.lt 0 b) (∀ (x : Subtype fun x => Mem …
  -/
  refine ⟨a, b, Int.ofNat_pos.mpr hb, fun x hx ↦ ?_⟩
  /-
    case intro.intro.intro.intro.intro
    R : Type u_1
    L : Type u_2
    inst✝¹³ : CommRing R
    inst✝¹² : LieRing L
    inst✝¹¹ : LieAlgebra R L
    M : Type u_3
    inst✝¹⁰ : AddCommGroup M
    inst✝⁹ : Module R M
    inst✝⁸ : LieRingModule L M
    inst✝⁷ : LieModule R L M
    H : LieSubalgebra R L
    α χ : (Subtype fun x => Membership.mem H x) → R
    inst✝⁶ : H.IsCartanSubalgebra
    inst✝⁵ : IsNoetherian R L
    inst✝⁴ : IsDomain R
    inst✝³ : IsPrincipalIdealRing R
    inst✝² : CharZero R
    inst✝¹ : NoZeroSMulDivisors R M
    inst✝ : IsNoetherian R M
    hα : Ne α 0
    hχ : Ne (LieModule.genWeightSpace M χ) Bot.bot
    p : Int
    hp₀ : LT.lt p 0
    q : Int
    hq₀ : GT.gt q 0
    hp : Eq (LieModule.genWeightSpace M (HAdd.hAdd (HSMul.hSMul p α) χ)) Bot.bot
    hq : Eq (LieModule.genWeightSpace M (HAdd.hAdd (HSMul.hSMul q α) χ)) Bot.bot
    a : Int := (Finset.Ioo p q).sum fun i => HSMul.hSMul (Module.finrank R (Subtyp …
    b : Nat := (Finset.Ioo p q).sum fun i => Module.finrank R (Subtype fun x => Me …
    hb : LT.lt 0 b
    x : Subtype fun x => Membership.mem H x
    hx : Membership.mem (LieAlgebra.corootSpace α) x
    ⊢ Eq (HAdd.hAdd (HSMul.hSMul a α) (HSMul.hSMul (↑b) χ) x) 0
  -/
  let N : ℤ → Submodule R M := fun k ↦ genWeightSpace M (k • α + χ)
  have h₁ : iSupIndep fun (i : Finset.Ioo p q) ↦ N i := by
    rw [← LieSubmodule.iSupIndep_iff_toSubmodule]
    refine (iSupIndep_genWeightSpace R H M).comp fun i j hij ↦ ?_
    exact SetCoe.ext <| smul_left_injective ℤ hα <| by rwa [add_left_inj] at hij
  /-
    case intro.intro.intro.intro.intro
    R : Type u_1
    L : Type u_2
    inst✝¹³ : CommRing R
    inst✝¹² : LieRing L
    inst✝¹¹ : LieAlgebra R L
    M : Type u_3
    inst✝¹⁰ : AddCommGroup M
    inst✝⁹ : Module R M
    inst✝⁸ : LieRingModule L M
    inst✝⁷ : LieModule R L M
    H : LieSubalgebra R L
    α χ : (Subtype fun x => Membership.mem H x) → R
    inst✝⁶ : H.IsCartanSubalgebra
    inst✝⁵ : IsNoetherian R L
    inst✝⁴ : IsDomain R
    inst✝³ : IsPrincipalIdealRing R
    inst✝² : CharZero R
    inst✝¹ : NoZeroSMulDivisors R M
    inst✝ : IsNoetherian R M
    hα : Ne α 0
    hχ : Ne (LieModule.genWeightSpace M χ) Bot.bot
    p : Int
    hp₀ : LT.lt p 0
    q : Int
    hq₀ : GT.gt q 0
    hp : Eq (LieModule.genWeightSpace M (HAdd.hAdd (HSMul.hSMul p α) χ)) Bot.bot
    hq : Eq (LieModule.genWeightSpace M (HAdd.hAdd (HSMul.hSMul q α) χ)) Bot.bot
    a : Int := (Finset.Ioo p q).sum fun i => HSMul.hSMul (Module.finrank R (Subtyp …
    b : Nat := (Finset.Ioo p q).sum fun i => Module.finrank R (Subtype fun x => Me …
    hb : LT.lt 0 b
    x : Subtype fun x => Membership.mem H x
    hx : Membership.mem (LieAlgebra.corootSpace α) x
    N : Int → Submodule R M := fun k => ↑(LieModule.genWeightSpace M (HAdd.hAdd (H …
    h₁ : iSupIndep fun i => N ↑i
    ⊢ Eq (HAdd.hAdd (HSMul.hSMul a α) (HSMul.hSMul (↑b) χ) x) 0
  -/
  have h₂ : ∀ i, MapsTo (toEnd R H M x) ↑(N i) ↑(N i) := fun _ _ ↦ LieSubmodule.lie_mem _
  have h₃ : genWeightSpaceChain M α χ p q = ⨆ i ∈ Finset.Ioo p q, N i := by
    simp_rw [N, genWeightSpaceChain_def', LieSubmodule.iSup_toSubmodule]
  rw [← trace_toEnd_genWeightSpaceChain_eq_zero M α χ p q hp hq hx,
    ← LieSubmodule.toEnd_restrict_eq_toEnd]
  -- The lines below illustrate the cost of treating `LieSubmodule` as both a
  -- `Submodule` and a `LieSubmodule` simultaneously.
  /-
    case intro.intro.intro.intro.intro
    R : Type u_1
    L : Type u_2
    inst✝¹³ : CommRing R
    inst✝¹² : LieRing L
    inst✝¹¹ : LieAlgebra R L
    M : Type u_3
    inst✝¹⁰ : AddCommGroup M
    inst✝⁹ : Module R M
    inst✝⁸ : LieRingModule L M
    inst✝⁷ : LieModule R L M
    H : LieSubalgebra R L
    α χ : (Subtype fun x => Membership.mem H x) → R
    inst✝⁶ : H.IsCartanSubalgebra
    inst✝⁵ : IsNoetherian R L
    inst✝⁴ : IsDomain R
    inst✝³ : IsPrincipalIdealRing R
    inst✝² : CharZero R
    inst✝¹ : NoZeroSMulDivisors R M
    inst✝ : IsNoetherian R M
    hα : Ne α 0
    hχ : Ne (LieModule.genWeightSpace M χ) Bot.bot
    p : Int
    hp₀ : LT.lt p 0
    q : Int
    hq₀ : GT.gt q 0
    hp : Eq (LieModule.genWeightSpace M (HAdd.hAdd (HSMul.hSMul p α) χ)) Bot.bot
    hq : Eq (LieModule.genWeightSpace M (HAdd.hAdd (HSMul.hSMul q α) χ)) Bot.bot
    a : Int := (Finset.Ioo p q).sum fun i => HSMul.hSMul (Module.finrank R (Subtyp …
    b : Nat := (Finset.Ioo p q).sum fun i => Module.finrank R (Subtype fun x => Me …
    hb : LT.lt 0 b
    x : Subtype fun x => Membership.mem H x
    hx : Membership.mem (LieAlgebra.corootSpace α) x
    N : Int → Submodule R M := fun k => ↑(LieModule.genWeightSpace M (HAdd.hAdd (H …
    h₁ : iSupIndep fun i => N ↑i
    h₂ : ∀ (i : Int), Set.MapsTo ⇑((LieModule.toEnd R (Subtype fun x => Membership …
    h₃ : Eq (↑(LieModule.genWeightSpaceChain M α χ p q)) (iSup fun i => iSup fun h …
    ⊢ Eq (HAdd.hAdd (HSMul.hSMul a α) (HSMul.hSMul (↑b) χ) x) ((LinearMap.trace R  …
  -/
  erw [LinearMap.trace_eq_sum_trace_restrict_of_eq_biSup _ h₁ h₂ (genWeightSpaceChain M α χ p q) h₃]
  /-
    case intro.intro.intro.intro.intro
    R : Type u_1
    L : Type u_2
    inst✝¹³ : CommRing R
    inst✝¹² : LieRing L
    inst✝¹¹ : LieAlgebra R L
    M : Type u_3
    inst✝¹⁰ : AddCommGroup M
    inst✝⁹ : Module R M
    inst✝⁸ : LieRingModule L M
    inst✝⁷ : LieModule R L M
    H : LieSubalgebra R L
    α χ : (Subtype fun x => Membership.mem H x) → R
    inst✝⁶ : H.IsCartanSubalgebra
    inst✝⁵ : IsNoetherian R L
    inst✝⁴ : IsDomain R
    inst✝³ : IsPrincipalIdealRing R
    inst✝² : CharZero R
    inst✝¹ : NoZeroSMulDivisors R M
    inst✝ : IsNoetherian R M
    hα : Ne α 0
    hχ : Ne (LieModule.genWeightSpace M χ) Bot.bot
    p : Int
    hp₀ : LT.lt p 0
    q : Int
    hq₀ : GT.gt q 0
    hp : Eq (LieModule.genWeightSpace M (HAdd.hAdd (HSMul.hSMul p α) χ)) Bot.bot
    hq : Eq (LieModule.genWeightSpace M (HAdd.hAdd (HSMul.hSMul q α) χ)) Bot.bot
    a : Int := (Finset.Ioo p q).sum fun i => HSMul.hSMul (Module.finrank R (Subtyp …
    b : Nat := (Finset.Ioo p q).sum fun i => Module.finrank R (Subtype fun x => Me …
    hb : LT.lt 0 b
    x : Subtype fun x => Membership.mem H x
    hx : Membership.mem (LieAlgebra.corootSpace α) x
    N : Int → Submodule R M := fun k => ↑(LieModule.genWeightSpace M (HAdd.hAdd (H …
    h₁ : iSupIndep fun i => N ↑i
    h₂ : ∀ (i : Int), Set.MapsTo ⇑((LieModule.toEnd R (Subtype fun x => Membership …
    h₃ : Eq (↑(LieModule.genWeightSpaceChain M α χ p q)) (iSup fun i => iSup fun h …
    ⊢ Eq (HAdd.hAdd (HSMul.hSMul a α) (HSMul.hSMul (↑b) χ) x) ((Finset.Ioo p q).su …
  -/
  simp_rw [N, LieSubmodule.toEnd_restrict_eq_toEnd]
  /-
    case intro.intro.intro.intro.intro
    R : Type u_1
    L : Type u_2
    inst✝¹³ : CommRing R
    inst✝¹² : LieRing L
    inst✝¹¹ : LieAlgebra R L
    M : Type u_3
    inst✝¹⁰ : AddCommGroup M
    inst✝⁹ : Module R M
    inst✝⁸ : LieRingModule L M
    inst✝⁷ : LieModule R L M
    H : LieSubalgebra R L
    α χ : (Subtype fun x => Membership.mem H x) → R
    inst✝⁶ : H.IsCartanSubalgebra
    inst✝⁵ : IsNoetherian R L
    inst✝⁴ : IsDomain R
    inst✝³ : IsPrincipalIdealRing R
    inst✝² : CharZero R
    inst✝¹ : NoZeroSMulDivisors R M
    inst✝ : IsNoetherian R M
    hα : Ne α 0
    hχ : Ne (LieModule.genWeightSpace M χ) Bot.bot
    p : Int
    hp₀ : LT.lt p 0
    q : Int
    hq₀ : GT.gt q 0
    hp : Eq (LieModule.genWeightSpace M (HAdd.hAdd (HSMul.hSMul p α) χ)) Bot.bot
    hq : Eq (LieModule.genWeightSpace M (HAdd.hAdd (HSMul.hSMul q α) χ)) Bot.bot
    a : Int := (Finset.Ioo p q).sum fun i => HSMul.hSMul (Module.finrank R (Subtyp …
    b : Nat := (Finset.Ioo p q).sum fun i => Module.finrank R (Subtype fun x => Me …
    hb : LT.lt 0 b
    x : Subtype fun x => Membership.mem H x
    hx : Membership.mem (LieAlgebra.corootSpace α) x
    N : Int → Submodule R M := fun k => ↑(LieModule.genWeightSpace M (HAdd.hAdd (H …
    h₁ : iSupIndep fun i => N ↑i
    h₂ : ∀ (i : Int), Set.MapsTo ⇑((LieModule.toEnd R (Subtype fun x => Membership …
    h₃ : Eq (↑(LieModule.genWeightSpaceChain M α χ p q)) (iSup fun i => iSup fun h …
    ⊢ Eq (HAdd.hAdd (HSMul.hSMul a α) (HSMul.hSMul (↑b) χ) x) ((Finset.Ioo p q).su …
  -/
  dsimp [N]
  convert_to _ =
    ∑ k ∈ Finset.Ioo p q, (LinearMap.trace R { x // x ∈ (genWeightSpace M (k • α + χ)) })
      ((toEnd R { x // x ∈ H } { x // x ∈ genWeightSpace M (k • α + χ) }) x)
  simp_rw [a, b, trace_toEnd_genWeightSpace, Pi.add_apply, Pi.smul_apply, smul_add,
    ← smul_assoc, Finset.sum_add_distrib, ← Finset.sum_smul, natCast_zsmul]


/-- This is the largest `n : ℕ` such that `i • α + β` is a weight for all `0 ≤ i ≤ n`. -/
noncomputable
def chainTopCoeff : ℕ :=
  letI := Classical.propDecidable
  if hα : α = 0 then 0 else
  Nat.pred <| Nat.find (show ∃ n, genWeightSpace M (n • α + β : L → R) = ⊥ from
    (eventually_genWeightSpace_smul_add_eq_bot M α β hα).exists)


/-- This is the largest `n : ℕ` such that `-i • α + β` is a weight for all `0 ≤ i ≤ n`. -/
noncomputable
def chainBotCoeff : ℕ := chainTopCoeff (-α) β


@[simp] lemma chainTopCoeff_neg : chainTopCoeff (-α) β = chainBotCoeff α β := rfl

@[simp] lemma chainBotCoeff_neg : chainBotCoeff (-α) β = chainTopCoeff α β := by
  /-
    R : Type u_1
    L : Type u_2
    inst✝¹⁰ : CommRing R
    inst✝⁹ : LieRing L
    inst✝⁸ : LieAlgebra R L
    M : Type u_3
    inst✝⁷ : AddCommGroup M
    inst✝⁶ : Module R M
    inst✝⁵ : LieRingModule L M
    inst✝⁴ : LieModule R L M
    inst✝³ : LieAlgebra.IsNilpotent R L
    inst✝² : NoZeroSMulDivisors Int R
    inst✝¹ : NoZeroSMulDivisors R M
    inst✝ : IsNoetherian R M
    α : L → R
    β : LieModule.Weight R L M
    ⊢ Eq (LieModule.chainBotCoeff (Neg.neg α) β) (LieModule.chainTopCoeff α β)
  -/
  rw [← chainTopCoeff_neg, neg_neg]
  /-
    🎉 no goals
  -/


@[simp] lemma chainTopCoeff_zero : chainTopCoeff 0 β = 0 := dif_pos rfl

@[simp] lemma chainBotCoeff_zero : chainBotCoeff 0 β = 0 := dif_pos neg_zero


lemma chainTopCoeff_add_one :
    letI := Classical.propDecidable
    chainTopCoeff α β + 1 =
      Nat.find (eventually_genWeightSpace_smul_add_eq_bot M α β hα).exists := by
  classical
  rw [chainTopCoeff, dif_neg hα]
  apply Nat.succ_pred_eq_of_pos
  rw [zero_lt_iff]
  intro e
  have : genWeightSpace M (0 • α + β : L → R) = ⊥ := by
    rw [← e]
    exact Nat.find_spec (eventually_genWeightSpace_smul_add_eq_bot M α β hα).exists
  exact β.genWeightSpace_ne_bot _ (by simpa only [zero_smul, zero_add] using this)


lemma genWeightSpace_chainTopCoeff_add_one_nsmul_add :
    genWeightSpace M ((chainTopCoeff α β + 1) • α + β : L → R) = ⊥ := by
  classical
  rw [chainTopCoeff_add_one _ _ hα]
  exact Nat.find_spec (eventually_genWeightSpace_smul_add_eq_bot M α β hα).exists


lemma genWeightSpace_chainTopCoeff_add_one_zsmul_add :
    genWeightSpace M ((chainTopCoeff α β + 1 : ℤ) • α + β : L → R) = ⊥ := by
  rw [← genWeightSpace_chainTopCoeff_add_one_nsmul_add α β hα, ← Nat.cast_smul_eq_nsmul ℤ,
    Nat.cast_add, Nat.cast_one]


lemma genWeightSpace_chainBotCoeff_sub_one_zsmul_sub :
    genWeightSpace M ((-chainBotCoeff α β - 1 : ℤ) • α + β : L → R) = ⊥ := by
  rw [sub_eq_add_neg, ← neg_add, neg_smul, ← smul_neg, chainBotCoeff,
    genWeightSpace_chainTopCoeff_add_one_zsmul_add _ _ (by simpa using hα)]


lemma genWeightSpace_nsmul_add_ne_bot_of_le {n} (hn : n ≤ chainTopCoeff α β) :
    genWeightSpace M (n • α + β : L → R) ≠ ⊥ := by
  /-
    R : Type u_1
    L : Type u_2
    inst✝¹⁰ : CommRing R
    inst✝⁹ : LieRing L
    inst✝⁸ : LieAlgebra R L
    M : Type u_3
    inst✝⁷ : AddCommGroup M
    inst✝⁶ : Module R M
    inst✝⁵ : LieRingModule L M
    inst✝⁴ : LieModule R L M
    inst✝³ : LieAlgebra.IsNilpotent R L
    inst✝² : NoZeroSMulDivisors Int R
    inst✝¹ : NoZeroSMulDivisors R M
    inst✝ : IsNoetherian R M
    α : L → R
    β : LieModule.Weight R L M
    n : Nat
    hn : LE.le n (LieModule.chainTopCoeff α β)
    ⊢ Ne (LieModule.genWeightSpace M (HAdd.hAdd (HSMul.hSMul n α) ⇑β)) Bot.bot
  -/
  by_cases hα : α = 0
    /-
      case pos
      R : Type u_1
      L : Type u_2
      inst✝¹⁰ : CommRing R
      inst✝⁹ : LieRing L
      inst✝⁸ : LieAlgebra R L
      M : Type u_3
      inst✝⁷ : AddCommGroup M
      inst✝⁶ : Module R M
      inst✝⁵ : LieRingModule L M
      inst✝⁴ : LieModule R L M
      inst✝³ : LieAlgebra.IsNilpotent R L
      inst✝² : NoZeroSMulDivisors Int R
      inst✝¹ : NoZeroSMulDivisors R M
      inst✝ : IsNoetherian R M
      α : L → R
      β : LieModule.Weight R L M
      n : Nat
      hn : LE.le n (LieModule.chainTopCoeff α β)
      hα : Eq α 0
      ⊢ Ne (LieModule.genWeightSpace M (HAdd.hAdd (HSMul.hSMul n α) ⇑β)) Bot.bot
    -/
  · rw [hα, smul_zero, zero_add]; exact β.genWeightSpace_ne_bot
                                  /-
                                    🎉 no goals
                                  -/
  classical
  rw [← Nat.lt_succ, Nat.succ_eq_add_one, chainTopCoeff_add_one _ _ hα] at hn
  exact Nat.find_min (eventually_genWeightSpace_smul_add_eq_bot M α β hα).exists hn


lemma genWeightSpace_zsmul_add_ne_bot {n : ℤ}
    (hn : -chainBotCoeff α β ≤ n) (hn' : n ≤ chainTopCoeff α β) :
      genWeightSpace M (n • α + β : L → R) ≠ ⊥ := by
  /-
    R : Type u_1
    L : Type u_2
    inst✝¹⁰ : CommRing R
    inst✝⁹ : LieRing L
    inst✝⁸ : LieAlgebra R L
    M : Type u_3
    inst✝⁷ : AddCommGroup M
    inst✝⁶ : Module R M
    inst✝⁵ : LieRingModule L M
    inst✝⁴ : LieModule R L M
    inst✝³ : LieAlgebra.IsNilpotent R L
    inst✝² : NoZeroSMulDivisors Int R
    inst✝¹ : NoZeroSMulDivisors R M
    inst✝ : IsNoetherian R M
    α : L → R
    β : LieModule.Weight R L M
    n : Int
    hn : LE.le (Neg.neg ↑(LieModule.chainBotCoeff α β)) n
    hn' : LE.le n ↑(LieModule.chainTopCoeff α β)
    ⊢ Ne (LieModule.genWeightSpace M (HAdd.hAdd (HSMul.hSMul n α) ⇑β)) Bot.bot
  -/
  rcases n with (n | n)
    /-
      case ofNat
      R : Type u_1
      L : Type u_2
      inst✝¹⁰ : CommRing R
      inst✝⁹ : LieRing L
      inst✝⁸ : LieAlgebra R L
      M : Type u_3
      inst✝⁷ : AddCommGroup M
      inst✝⁶ : Module R M
      inst✝⁵ : LieRingModule L M
      inst✝⁴ : LieModule R L M
      inst✝³ : LieAlgebra.IsNilpotent R L
      inst✝² : NoZeroSMulDivisors Int R
      inst✝¹ : NoZeroSMulDivisors R M
      inst✝ : IsNoetherian R M
      α : L → R
      β : LieModule.Weight R L M
      n : Nat
      hn : LE.le (Neg.neg ↑(LieModule.chainBotCoeff α β)) (Int.ofNat n)
      hn' : LE.le (Int.ofNat n) ↑(LieModule.chainTopCoeff α β)
      ⊢ Ne (LieModule.genWeightSpace M (HAdd.hAdd (HSMul.hSMul (Int.ofNat n) α) ⇑β)) …
    -/
  · simp only [Int.ofNat_eq_coe, Nat.cast_le, Nat.cast_smul_eq_nsmul] at hn' ⊢
    /-
      case ofNat
      R : Type u_1
      L : Type u_2
      inst✝¹⁰ : CommRing R
      inst✝⁹ : LieRing L
      inst✝⁸ : LieAlgebra R L
      M : Type u_3
      inst✝⁷ : AddCommGroup M
      inst✝⁶ : Module R M
      inst✝⁵ : LieRingModule L M
      inst✝⁴ : LieModule R L M
      inst✝³ : LieAlgebra.IsNilpotent R L
      inst✝² : NoZeroSMulDivisors Int R
      inst✝¹ : NoZeroSMulDivisors R M
      inst✝ : IsNoetherian R M
      α : L → R
      β : LieModule.Weight R L M
      n : Nat
      hn : LE.le (Neg.neg ↑(LieModule.chainBotCoeff α β)) (Int.ofNat n)
      hn' : LE.le n (LieModule.chainTopCoeff α β)
      ⊢ Ne (LieModule.genWeightSpace M (HAdd.hAdd (HSMul.hSMul n α) ⇑β)) Bot.bot
    -/
    exact genWeightSpace_nsmul_add_ne_bot_of_le α β hn'
    /-
      🎉 no goals
    -/
    /-
      case negSucc
      R : Type u_1
      L : Type u_2
      inst✝¹⁰ : CommRing R
      inst✝⁹ : LieRing L
      inst✝⁸ : LieAlgebra R L
      M : Type u_3
      inst✝⁷ : AddCommGroup M
      inst✝⁶ : Module R M
      inst✝⁵ : LieRingModule L M
      inst✝⁴ : LieModule R L M
      inst✝³ : LieAlgebra.IsNilpotent R L
      inst✝² : NoZeroSMulDivisors Int R
      inst✝¹ : NoZeroSMulDivisors R M
      inst✝ : IsNoetherian R M
      α : L → R
      β : LieModule.Weight R L M
      n : Nat
      hn : LE.le (Neg.neg ↑(LieModule.chainBotCoeff α β)) (Int.negSucc n)
      hn' : LE.le (Int.negSucc n) ↑(LieModule.chainTopCoeff α β)
      ⊢ Ne (LieModule.genWeightSpace M (HAdd.hAdd (HSMul.hSMul (Int.negSucc n) α) ⇑β …
    -/
  · simp only [Int.negSucc_eq, ← Nat.cast_succ, neg_le_neg_iff, Nat.cast_le] at hn ⊢
    /-
      case negSucc
      R : Type u_1
      L : Type u_2
      inst✝¹⁰ : CommRing R
      inst✝⁹ : LieRing L
      inst✝⁸ : LieAlgebra R L
      M : Type u_3
      inst✝⁷ : AddCommGroup M
      inst✝⁶ : Module R M
      inst✝⁵ : LieRingModule L M
      inst✝⁴ : LieModule R L M
      inst✝³ : LieAlgebra.IsNilpotent R L
      inst✝² : NoZeroSMulDivisors Int R
      inst✝¹ : NoZeroSMulDivisors R M
      inst✝ : IsNoetherian R M
      α : L → R
      β : LieModule.Weight R L M
      n : Nat
      hn' : LE.le (Int.negSucc n) ↑(LieModule.chainTopCoeff α β)
      hn : LE.le n.succ (LieModule.chainBotCoeff α β)
      ⊢ Ne (LieModule.genWeightSpace M (HAdd.hAdd (HSMul.hSMul (Neg.neg ↑n.succ) α)  …
    -/
    rw [neg_smul, ← smul_neg, Nat.cast_smul_eq_nsmul]
    /-
      case negSucc
      R : Type u_1
      L : Type u_2
      inst✝¹⁰ : CommRing R
      inst✝⁹ : LieRing L
      inst✝⁸ : LieAlgebra R L
      M : Type u_3
      inst✝⁷ : AddCommGroup M
      inst✝⁶ : Module R M
      inst✝⁵ : LieRingModule L M
      inst✝⁴ : LieModule R L M
      inst✝³ : LieAlgebra.IsNilpotent R L
      inst✝² : NoZeroSMulDivisors Int R
      inst✝¹ : NoZeroSMulDivisors R M
      inst✝ : IsNoetherian R M
      α : L → R
      β : LieModule.Weight R L M
      n : Nat
      hn' : LE.le (Int.negSucc n) ↑(LieModule.chainTopCoeff α β)
      hn : LE.le n.succ (LieModule.chainBotCoeff α β)
      ⊢ Ne (LieModule.genWeightSpace M (HAdd.hAdd (HSMul.hSMul n.succ (Neg.neg α)) ⇑ …
    -/
    exact genWeightSpace_nsmul_add_ne_bot_of_le (-α) β hn
    /-
      🎉 no goals
    -/


lemma genWeightSpace_neg_zsmul_add_ne_bot {n : ℕ} (hn : n ≤ chainBotCoeff α β) :
    genWeightSpace M ((-n : ℤ) • α + β : L → R) ≠ ⊥ := by
  /-
    R : Type u_1
    L : Type u_2
    inst✝¹⁰ : CommRing R
    inst✝⁹ : LieRing L
    inst✝⁸ : LieAlgebra R L
    M : Type u_3
    inst✝⁷ : AddCommGroup M
    inst✝⁶ : Module R M
    inst✝⁵ : LieRingModule L M
    inst✝⁴ : LieModule R L M
    inst✝³ : LieAlgebra.IsNilpotent R L
    inst✝² : NoZeroSMulDivisors Int R
    inst✝¹ : NoZeroSMulDivisors R M
    inst✝ : IsNoetherian R M
    α : L → R
    β : LieModule.Weight R L M
    n : Nat
    hn : LE.le n (LieModule.chainBotCoeff α β)
    ⊢ Ne (LieModule.genWeightSpace M (HAdd.hAdd (HSMul.hSMul (Neg.neg ↑n) α) ⇑β))  …
  -/
                                                /-
                                                  🎉 no goals
                                                -/
  apply genWeightSpace_zsmul_add_ne_bot α β <;> omega
                                                /-
                                                  🎉 no goals
                                                -/


/-- The last weight in an `α`-chain through `β`. -/
noncomputable
def chainTop (α : L → R) (β : Weight R L M) : Weight R L M :=
  ⟨chainTopCoeff α β • α + β, genWeightSpace_nsmul_add_ne_bot_of_le α β le_rfl⟩


/-- The first weight in an `α`-chain through `β`. -/
noncomputable
def chainBot (α : L → R) (β : Weight R L M) : Weight R L M :=
  ⟨(- chainBotCoeff α β : ℤ) • α + β, genWeightSpace_neg_zsmul_add_ne_bot α β le_rfl⟩


lemma coe_chainTop' : (chainTop α β : L → R) = chainTopCoeff α β • α + β := rfl


@[simp] lemma coe_chainTop : (chainTop α β : L → R) = (chainTopCoeff α β : ℤ) • α + β := by
  /-
    R : Type u_1
    L : Type u_2
    inst✝¹⁰ : CommRing R
    inst✝⁹ : LieRing L
    inst✝⁸ : LieAlgebra R L
    M : Type u_3
    inst✝⁷ : AddCommGroup M
    inst✝⁶ : Module R M
    inst✝⁵ : LieRingModule L M
    inst✝⁴ : LieModule R L M
    inst✝³ : LieAlgebra.IsNilpotent R L
    inst✝² : NoZeroSMulDivisors Int R
    inst✝¹ : NoZeroSMulDivisors R M
    inst✝ : IsNoetherian R M
    α : L → R
    β : LieModule.Weight R L M
    ⊢ Eq (⇑(LieModule.chainTop α β)) (HAdd.hAdd (HSMul.hSMul (↑(LieModule.chainTop …
  -/
  rw [Nat.cast_smul_eq_nsmul ℤ]; rfl
                                 /-
                                   🎉 no goals
                                 -/

@[simp] lemma coe_chainBot : (chainBot α β : L → R) = (-chainBotCoeff α β : ℤ) • α + β := rfl


                                                                  /-
                                                                    R : Type u_1
                                                                    L : Type u_2
                                                                    inst✝¹⁰ : CommRing R
                                                                    inst✝⁹ : LieRing L
                                                                    inst✝⁸ : LieAlgebra R L
                                                                    M : Type u_3
                                                                    inst✝⁷ : AddCommGroup M
                                                                    inst✝⁶ : Module R M
                                                                    inst✝⁵ : LieRingModule L M
                                                                    inst✝⁴ : LieModule R L M
                                                                    inst✝³ : LieAlgebra.IsNilpotent R L
                                                                    inst✝² : NoZeroSMulDivisors Int R
                                                                    inst✝¹ : NoZeroSMulDivisors R M
                                                                    inst✝ : IsNoetherian R M
                                                                    α : L → R
                                                                    β : LieModule.Weight R L M
                                                                    ⊢ Eq (LieModule.chainTop (Neg.neg α) β) (LieModule.chainBot α β)
                                                                  -/
@[simp] lemma chainTop_neg : chainTop (-α) β = chainBot α β := by ext; simp
                                                                       /-
                                                                         🎉 no goals
                                                                       -/

                                                                  /-
                                                                    R : Type u_1
                                                                    L : Type u_2
                                                                    inst✝¹⁰ : CommRing R
                                                                    inst✝⁹ : LieRing L
                                                                    inst✝⁸ : LieAlgebra R L
                                                                    M : Type u_3
                                                                    inst✝⁷ : AddCommGroup M
                                                                    inst✝⁶ : Module R M
                                                                    inst✝⁵ : LieRingModule L M
                                                                    inst✝⁴ : LieModule R L M
                                                                    inst✝³ : LieAlgebra.IsNilpotent R L
                                                                    inst✝² : NoZeroSMulDivisors Int R
                                                                    inst✝¹ : NoZeroSMulDivisors R M
                                                                    inst✝ : IsNoetherian R M
                                                                    α : L → R
                                                                    β : LieModule.Weight R L M
                                                                    ⊢ Eq (LieModule.chainBot (Neg.neg α) β) (LieModule.chainTop α β)
                                                                  -/
@[simp] lemma chainBot_neg : chainBot (-α) β = chainTop α β := by ext; simp
                                                                       /-
                                                                         🎉 no goals
                                                                       -/


                                                     /-
                                                       R : Type u_1
                                                       L : Type u_2
                                                       inst✝¹⁰ : CommRing R
                                                       inst✝⁹ : LieRing L
                                                       inst✝⁸ : LieAlgebra R L
                                                       M : Type u_3
                                                       inst✝⁷ : AddCommGroup M
                                                       inst✝⁶ : Module R M
                                                       inst✝⁵ : LieRingModule L M
                                                       inst✝⁴ : LieModule R L M
                                                       inst✝³ : LieAlgebra.IsNilpotent R L
                                                       inst✝² : NoZeroSMulDivisors Int R
                                                       inst✝¹ : NoZeroSMulDivisors R M
                                                       inst✝ : IsNoetherian R M
                                                       β : LieModule.Weight R L M
                                                       ⊢ Eq (LieModule.chainTop 0 β) β
                                                     -/
@[simp] lemma chainTop_zero : chainTop 0 β = β := by ext; simp
                                                          /-
                                                            🎉 no goals
                                                          -/

                                                     /-
                                                       R : Type u_1
                                                       L : Type u_2
                                                       inst✝¹⁰ : CommRing R
                                                       inst✝⁹ : LieRing L
                                                       inst✝⁸ : LieAlgebra R L
                                                       M : Type u_3
                                                       inst✝⁷ : AddCommGroup M
                                                       inst✝⁶ : Module R M
                                                       inst✝⁵ : LieRingModule L M
                                                       inst✝⁴ : LieModule R L M
                                                       inst✝³ : LieAlgebra.IsNilpotent R L
                                                       inst✝² : NoZeroSMulDivisors Int R
                                                       inst✝¹ : NoZeroSMulDivisors R M
                                                       inst✝ : IsNoetherian R M
                                                       β : LieModule.Weight R L M
                                                       ⊢ Eq (LieModule.chainBot 0 β) β
                                                     -/
@[simp] lemma chainBot_zero : chainBot 0 β = β := by ext; simp
                                                          /-
                                                            🎉 no goals
                                                          -/


lemma genWeightSpace_add_chainTop :
    genWeightSpace M (α + chainTop α β : L → R) = ⊥ := by
  rw [coe_chainTop', ← add_assoc, ← succ_nsmul',
    genWeightSpace_chainTopCoeff_add_one_nsmul_add _ _ hα]


lemma genWeightSpace_neg_add_chainBot :
    genWeightSpace M (-α + chainBot α β : L → R) = ⊥ := by
  /-
    R : Type u_1
    L : Type u_2
    inst✝¹⁰ : CommRing R
    inst✝⁹ : LieRing L
    inst✝⁸ : LieAlgebra R L
    M : Type u_3
    inst✝⁷ : AddCommGroup M
    inst✝⁶ : Module R M
    inst✝⁵ : LieRingModule L M
    inst✝⁴ : LieModule R L M
    inst✝³ : LieAlgebra.IsNilpotent R L
    inst✝² : NoZeroSMulDivisors Int R
    inst✝¹ : NoZeroSMulDivisors R M
    inst✝ : IsNoetherian R M
    α : L → R
    β : LieModule.Weight R L M
    hα : Ne α 0
    ⊢ Eq (LieModule.genWeightSpace M (HAdd.hAdd (Neg.neg α) ⇑(LieModule.chainBot α …
  -/
  rw [← chainTop_neg, genWeightSpace_add_chainTop _ _ (by simpa using hα)]
  /-
    🎉 no goals
  -/


lemma chainTop_isNonZero' (hα' : genWeightSpace M α ≠ ⊥) :
    (chainTop α β).IsNonZero := by
  /-
    R : Type u_1
    L : Type u_2
    inst✝¹⁰ : CommRing R
    inst✝⁹ : LieRing L
    inst✝⁸ : LieAlgebra R L
    M : Type u_3
    inst✝⁷ : AddCommGroup M
    inst✝⁶ : Module R M
    inst✝⁵ : LieRingModule L M
    inst✝⁴ : LieModule R L M
    inst✝³ : LieAlgebra.IsNilpotent R L
    inst✝² : NoZeroSMulDivisors Int R
    inst✝¹ : NoZeroSMulDivisors R M
    inst✝ : IsNoetherian R M
    α : L → R
    β : LieModule.Weight R L M
    hα : Ne α 0
    hα' : Ne (LieModule.genWeightSpace M α) Bot.bot
    ⊢ (LieModule.chainTop α β).IsNonZero
  -/
  by_contra e
  /-
    R : Type u_1
    L : Type u_2
    inst✝¹⁰ : CommRing R
    inst✝⁹ : LieRing L
    inst✝⁸ : LieAlgebra R L
    M : Type u_3
    inst✝⁷ : AddCommGroup M
    inst✝⁶ : Module R M
    inst✝⁵ : LieRingModule L M
    inst✝⁴ : LieModule R L M
    inst✝³ : LieAlgebra.IsNilpotent R L
    inst✝² : NoZeroSMulDivisors Int R
    inst✝¹ : NoZeroSMulDivisors R M
    inst✝ : IsNoetherian R M
    α : L → R
    β : LieModule.Weight R L M
    hα : Ne α 0
    hα' : Ne (LieModule.genWeightSpace M α) Bot.bot
    e : (LieModule.chainTop α β).IsZero
    ⊢ False
  -/
  apply hα'
  /-
    R : Type u_1
    L : Type u_2
    inst✝¹⁰ : CommRing R
    inst✝⁹ : LieRing L
    inst✝⁸ : LieAlgebra R L
    M : Type u_3
    inst✝⁷ : AddCommGroup M
    inst✝⁶ : Module R M
    inst✝⁵ : LieRingModule L M
    inst✝⁴ : LieModule R L M
    inst✝³ : LieAlgebra.IsNilpotent R L
    inst✝² : NoZeroSMulDivisors Int R
    inst✝¹ : NoZeroSMulDivisors R M
    inst✝ : IsNoetherian R M
    α : L → R
    β : LieModule.Weight R L M
    hα : Ne α 0
    hα' : Ne (LieModule.genWeightSpace M α) Bot.bot
    e : (LieModule.chainTop α β).IsZero
    ⊢ Eq (LieModule.genWeightSpace M α) Bot.bot
  -/
  rw [← add_zero (α : L → R), ← e, genWeightSpace_add_chainTop _ _ hα]
  /-
    🎉 no goals
  -/


lemma chainTop_isNonZero (α β : Weight R L M) (hα : α.IsNonZero) :
    (chainTop α β).IsNonZero :=
  chainTop_isNonZero' α β hα α.2


