/-- A local notation for the set of (Lie) derivations on `L`. -/
local notation "𝔻" => (LieDerivation R L L)


/-- A local notation for the range of `ad`. -/
local notation "𝕀" => (LieHom.range (ad R L))


/-- A local notation for the Killing complement of the ideal range of `ad`. -/
local notation "𝕀ᗮ" => LinearMap.BilinForm.orthogonal (killingForm R 𝔻) 𝕀


lemma killingForm_restrict_range_ad [Module.Finite R L] :
    (killingForm R 𝔻).restrict 𝕀 = killingForm R 𝕀 := by
  /-
    R : Type u_1
    L : Type u_2
    inst✝³ : Field R
    inst✝² : LieRing L
    inst✝¹ : LieAlgebra R L
    inst✝ : Module.Finite R L
    ⊢ Eq ((killingForm R (LieDerivation R L L)).restrict (LieDerivation.ad R L).ra …
  -/
  rw [← (ad_isIdealMorphism R L).eq, ← LieIdeal.killingForm_eq]
  /-
    R : Type u_1
    L : Type u_2
    inst✝³ : Field R
    inst✝² : LieRing L
    inst✝¹ : LieAlgebra R L
    inst✝ : Module.Finite R L
    ⊢ Eq (killingForm R (Subtype fun x => Membership.mem (LieDerivation.ad R L).id …
  -/
  rfl
  /-
    🎉 no goals
  -/


/-- The orthogonal complement of the inner derivations is a Lie submodule of all derivations. -/
@[simps!] noncomputable def rangeAdOrthogonal : LieSubmodule R L (LieDerivation R L L) where
  __ := 𝕀ᗮ
  lie_mem := by
    /-
      R : Type u_1
      L : Type u_2
      inst✝² : Field R
      inst✝¹ : LieRing L
      inst✝ : LieAlgebra R L
      ⊢ ∀ {x : L} {m : LieDerivation R L L}, Membership.mem __spread✝⁻⁰.carrier m →  …
    -/
    intro x D hD
    /-
      R : Type u_1
      L : Type u_2
      inst✝² : Field R
      inst✝¹ : LieRing L
      inst✝ : LieAlgebra R L
      x : L
      D : LieDerivation R L L
      hD : Membership.mem __spread✝⁻⁰.carrier D
      ⊢ Membership.mem __spread✝⁻⁰.carrier (Bracket.bracket x D)
    -/
    have : 𝕀ᗮ = (ad R L).idealRange.killingCompl := by simp [← (ad_isIdealMorphism R L).eq]
    /-
      R : Type u_1
      L : Type u_2
      inst✝² : Field R
      inst✝¹ : LieRing L
      inst✝ : LieAlgebra R L
      x : L
      D : LieDerivation R L L
      hD : Membership.mem __spread✝⁻⁰.carrier D
      this : Eq ((killingForm R (LieDerivation R L L)).orthogonal (LieDerivation.ad  …
      ⊢ Membership.mem __spread✝⁻⁰.carrier (Bracket.bracket x D)
    -/
    change D ∈ 𝕀ᗮ at hD
    /-
      R : Type u_1
      L : Type u_2
      inst✝² : Field R
      inst✝¹ : LieRing L
      inst✝ : LieAlgebra R L
      x : L
      D : LieDerivation R L L
      this : Eq ((killingForm R (LieDerivation R L L)).orthogonal (LieDerivation.ad  …
      hD : Membership.mem ((killingForm R (LieDerivation R L L)).orthogonal (LieDeri …
      ⊢ Membership.mem __spread✝⁻⁰.carrier (Bracket.bracket x D)
    -/
    change ⁅x, D⁆ ∈ 𝕀ᗮ
    /-
      R : Type u_1
      L : Type u_2
      inst✝² : Field R
      inst✝¹ : LieRing L
      inst✝ : LieAlgebra R L
      x : L
      D : LieDerivation R L L
      this : Eq ((killingForm R (LieDerivation R L L)).orthogonal (LieDerivation.ad  …
      hD : Membership.mem ((killingForm R (LieDerivation R L L)).orthogonal (LieDeri …
      ⊢ Membership.mem ((killingForm R (LieDerivation R L L)).orthogonal (LieDerivat …
    -/
    rw [this] at hD ⊢
    /-
      R : Type u_1
      L : Type u_2
      inst✝² : Field R
      inst✝¹ : LieRing L
      inst✝ : LieAlgebra R L
      x : L
      D : LieDerivation R L L
      this : Eq ((killingForm R (LieDerivation R L L)).orthogonal (LieDerivation.ad  …
      hD : Membership.mem (LieIdeal.toLieSubalgebra R (LieDerivation R L L) (LieIdea …
      ⊢ Membership.mem (LieIdeal.toLieSubalgebra R (LieDerivation R L L) (LieIdeal.k …
    -/
    rw [← lie_ad]
    /-
      R : Type u_1
      L : Type u_2
      inst✝² : Field R
      inst✝¹ : LieRing L
      inst✝ : LieAlgebra R L
      x : L
      D : LieDerivation R L L
      this : Eq ((killingForm R (LieDerivation R L L)).orthogonal (LieDerivation.ad  …
      hD : Membership.mem (LieIdeal.toLieSubalgebra R (LieDerivation R L L) (LieIdea …
      ⊢ Membership.mem (LieIdeal.toLieSubalgebra R (LieDerivation R L L) (LieIdeal.k …
    -/
    exact lie_mem_right _ _ (ad R L).idealRange.killingCompl _ _ hD
    /-
      🎉 no goals
    -/


/-- If a derivation `D` is in the Killing orthogonal of the range of the adjoint action, then, for
any `x : L`, `ad (D x)` is also in this orthogonal. -/
lemma ad_mem_orthogonal_of_mem_orthogonal {D : LieDerivation R L L} (hD : D ∈ 𝕀ᗮ) (x : L) :
    ad R L (D x) ∈ 𝕀ᗮ := by
  /-
    R : Type u_1
    L : Type u_2
    inst✝² : Field R
    inst✝¹ : LieRing L
    inst✝ : LieAlgebra R L
    D : LieDerivation R L L
    hD : Membership.mem ((killingForm R (LieDerivation R L L)).orthogonal (LieDeri …
    x : L
    ⊢ Membership.mem ((killingForm R (LieDerivation R L L)).orthogonal (LieDerivat …
  -/
  simp only [ad_apply_lieDerivation, LieHom.range_toSubmodule, neg_mem_iff]
  /-
    R : Type u_1
    L : Type u_2
    inst✝² : Field R
    inst✝¹ : LieRing L
    inst✝ : LieAlgebra R L
    D : LieDerivation R L L
    hD : Membership.mem ((killingForm R (LieDerivation R L L)).orthogonal (LieDeri …
    x : L
    ⊢ Membership.mem ((killingForm R (LieDerivation R L L)).orthogonal (LinearMap. …
  -/
  exact (rangeAdOrthogonal R L).lie_mem hD
  /-
    🎉 no goals
  -/


lemma ad_mem_ker_killingForm_ad_range_of_mem_orthogonal
    {D : LieDerivation R L L} (hD : D ∈ 𝕀ᗮ) (x : L) :
    ad R L (D x) ∈ (LinearMap.ker (killingForm R 𝕀)).map (LieHom.range (ad R L)).subtype := by
  /-
    R : Type u_1
    L : Type u_2
    inst✝³ : Field R
    inst✝² : LieRing L
    inst✝¹ : LieAlgebra R L
    inst✝ : Module.Finite R L
    D : LieDerivation R L L
    hD : Membership.mem ((killingForm R (LieDerivation R L L)).orthogonal (LieDeri …
    x : L
    ⊢ Membership.mem (Submodule.map (LieDerivation.ad R L).range.subtype (LinearMa …
  -/
  rw [← killingForm_restrict_range_ad]
  exact LinearMap.BilinForm.inf_orthogonal_self_le_ker_restrict
    (LieModule.traceForm_isSymm R 𝔻 𝔻).isRefl ⟨by simp, ad_mem_orthogonal_of_mem_orthogonal hD x⟩


@[simp] lemma ad_apply_eq_zero_iff (x : L) : ad R L x = 0 ↔ x = 0 := by
  /-
    R : Type u_1
    L : Type u_2
    inst✝⁴ : Field R
    inst✝³ : LieRing L
    inst✝² : LieAlgebra R L
    inst✝¹ : Module.Finite R L
    inst✝ : LieAlgebra.IsKilling R L
    x : L
    ⊢ Iff (Eq ((LieDerivation.ad R L) x) 0) (Eq x 0)
  -/
  refine ⟨fun h ↦ ?_, fun h ↦ by simp [h]⟩
  rwa [← LieHom.mem_ker, ad_ker_eq_center, LieAlgebra.HasTrivialRadical.center_eq_bot,
    LieSubmodule.mem_bot] at h


instance instIsKilling_range_ad : LieAlgebra.IsKilling R 𝕀 :=
                                                                      /-
                                                                        R : Type u_1
                                                                        L : Type u_2
                                                                        inst✝⁴ : Field R
                                                                        inst✝³ : LieRing L
                                                                        inst✝² : LieAlgebra R L
                                                                        inst✝¹ : Module.Finite R L
                                                                        inst✝ : LieAlgebra.IsKilling R L
                                                                        ⊢ Eq (LieAlgebra.center R L) Bot.bot
                                                                      -/
  (LieEquiv.ofInjective (ad R L) (injective_ad_of_center_eq_bot <| by simp)).isKilling
                                                                      /-
                                                                        🎉 no goals
                                                                      -/


/-- The restriction of the Killing form of a finite-dimensional Killing Lie algebra to the range of
the adjoint action is nondegenerate. -/
lemma killingForm_restrict_range_ad_nondegenerate :
    ((killingForm R 𝔻).restrict 𝕀).Nondegenerate := by
  /-
    R : Type u_1
    L : Type u_2
    inst✝⁴ : Field R
    inst✝³ : LieRing L
    inst✝² : LieAlgebra R L
    inst✝¹ : Module.Finite R L
    inst✝ : LieAlgebra.IsKilling R L
    ⊢ ((killingForm R (LieDerivation R L L)).restrict (LieDerivation.ad R L).range …
  -/
  convert LieAlgebra.IsKilling.killingForm_nondegenerate R 𝕀
  /-
    case h.e'_6.h.h.h
    R : Type u_1
    L : Type u_2
    inst✝⁴ : Field R
    inst✝³ : LieRing L
    inst✝² : LieAlgebra R L
    inst✝¹ : Module.Finite R L
    inst✝ : LieAlgebra.IsKilling R L
    e_2✝ : Eq (Subtype fun x => Membership.mem (LieDerivation.ad R L).range.toSubm …
    he✝¹ : Eq (LieDerivation.ad R L).range.addCommMonoid AddCommGroup.toAddCommMon …
    he✝ : Eq (LieDerivation.ad R L).range.module LieAlgebra.toModule
    ⊢ Eq ((killingForm R (LieDerivation R L L)).restrict (LieDerivation.ad R L).ra …
  -/
  exact killingForm_restrict_range_ad R L
  /-
    🎉 no goals
  -/


/-- The range of the adjoint action on a finite-dimensional Killing Lie algebra is full. -/
@[simp]
lemma range_ad_eq_top : 𝕀 = ⊤ := by
  /-
    R : Type u_1
    L : Type u_2
    inst✝⁴ : Field R
    inst✝³ : LieRing L
    inst✝² : LieAlgebra R L
    inst✝¹ : Module.Finite R L
    inst✝ : LieAlgebra.IsKilling R L
    ⊢ Eq (LieDerivation.ad R L).range Top.top
  -/
  rw [← LieSubalgebra.toSubmodule_inj]
  apply LinearMap.BilinForm.eq_top_of_restrict_nondegenerate_of_orthogonal_eq_bot
    (LieModule.traceForm_isSymm R 𝔻 𝔻).isRefl (killingForm_restrict_range_ad_nondegenerate R L)
  /-
    R : Type u_1
    L : Type u_2
    inst✝⁴ : Field R
    inst✝³ : LieRing L
    inst✝² : LieAlgebra R L
    inst✝¹ : Module.Finite R L
    inst✝ : LieAlgebra.IsKilling R L
    ⊢ Eq ((LieModule.traceForm R (LieDerivation R L L) (LieDerivation R L L)).orth …
  -/
  refine (Submodule.eq_bot_iff _).mpr fun D hD ↦ ext fun x ↦ ?_
  /-
    R : Type u_1
    L : Type u_2
    inst✝⁴ : Field R
    inst✝³ : LieRing L
    inst✝² : LieAlgebra R L
    inst✝¹ : Module.Finite R L
    inst✝ : LieAlgebra.IsKilling R L
    D : LieDerivation R L L
    hD : Membership.mem ((LieModule.traceForm R (LieDerivation R L L) (LieDerivati …
    x : L
    ⊢ Eq (D x) (0 x)
  -/
  simpa using ad_mem_ker_killingForm_ad_range_of_mem_orthogonal hD x
  /-
    🎉 no goals
  -/


variable {R L} in
/-- Every derivation of a finite-dimensional Killing Lie algebra is an inner derivation. -/
lemma exists_eq_ad (D : 𝔻) : ∃ x, ad R L x = D := by
  /-
    R : Type u_1
    L : Type u_2
    inst✝⁴ : Field R
    inst✝³ : LieRing L
    inst✝² : LieAlgebra R L
    inst✝¹ : Module.Finite R L
    inst✝ : LieAlgebra.IsKilling R L
    D : LieDerivation R L L
    ⊢ Exists fun x => Eq ((LieDerivation.ad R L) x) D
  -/
  change D ∈ 𝕀
  /-
    R : Type u_1
    L : Type u_2
    inst✝⁴ : Field R
    inst✝³ : LieRing L
    inst✝² : LieAlgebra R L
    inst✝¹ : Module.Finite R L
    inst✝ : LieAlgebra.IsKilling R L
    D : LieDerivation R L L
    ⊢ Membership.mem (LieDerivation.ad R L).range D
  -/
  rw [range_ad_eq_top R L]
  /-
    R : Type u_1
    L : Type u_2
    inst✝⁴ : Field R
    inst✝³ : LieRing L
    inst✝² : LieAlgebra R L
    inst✝¹ : Module.Finite R L
    inst✝ : LieAlgebra.IsKilling R L
    D : LieDerivation R L L
    ⊢ Membership.mem Top.top D
  -/
  exact Submodule.mem_top
  /-
    🎉 no goals
  -/


