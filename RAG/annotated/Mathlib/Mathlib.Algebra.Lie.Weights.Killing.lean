lemma restrict_killingForm (H : LieSubalgebra R L) :
    (killingForm R L).restrict H = LieModule.traceForm R H L :=
  rfl


/-- If the Killing form of a Lie algebra is non-singular, it remains non-singular when restricted
to a Cartan subalgebra. -/
lemma ker_restrict_eq_bot_of_isCartanSubalgebra
    [IsNoetherian R L] [IsArtinian R L] (H : LieSubalgebra R L) [H.IsCartanSubalgebra] :
    LinearMap.ker ((killingForm R L).restrict H) = ⊥ := by
  have h : Codisjoint (rootSpace H 0) (LieModule.posFittingComp R H L) :=
    (LieModule.isCompl_genWeightSpace_zero_posFittingComp R H L).codisjoint
  replace h : Codisjoint (H : Submodule R L) (LieModule.posFittingComp R H L : Submodule R L) := by
    rwa [codisjoint_iff, ← LieSubmodule.toSubmodule_inj, LieSubmodule.sup_toSubmodule,
      LieSubmodule.top_toSubmodule, rootSpace_zero_eq R L H, LieSubalgebra.coe_toLieSubmodule,
      ← codisjoint_iff] at h
  suffices this : ∀ m₀ ∈ H, ∀ m₁ ∈ LieModule.posFittingComp R H L, killingForm R L m₀ m₁ = 0 by
    simp [LinearMap.BilinForm.ker_restrict_eq_of_codisjoint h this]
  /-
    R : Type u_1
    L : Type u_3
    inst✝⁶ : CommRing R
    inst✝⁵ : LieRing L
    inst✝⁴ : LieAlgebra R L
    inst✝³ : LieAlgebra.IsKilling R L
    inst✝² : IsNoetherian R L
    inst✝¹ : IsArtinian R L
    H : LieSubalgebra R L
    inst✝ : H.IsCartanSubalgebra
    h : Codisjoint H.toSubmodule ↑(LieModule.posFittingComp R (Subtype fun x => Me …
    ⊢ ∀ (m₀ : L), Membership.mem H m₀ → ∀ (m₁ : L), Membership.mem (LieModule.posF …
  -/
  intro m₀ h₀ m₁ h₁
  /-
    R : Type u_1
    L : Type u_3
    inst✝⁶ : CommRing R
    inst✝⁵ : LieRing L
    inst✝⁴ : LieAlgebra R L
    inst✝³ : LieAlgebra.IsKilling R L
    inst✝² : IsNoetherian R L
    inst✝¹ : IsArtinian R L
    H : LieSubalgebra R L
    inst✝ : H.IsCartanSubalgebra
    h : Codisjoint H.toSubmodule ↑(LieModule.posFittingComp R (Subtype fun x => Me …
    m₀ : L
    h₀ : Membership.mem H m₀
    m₁ : L
    h₁ : Membership.mem (LieModule.posFittingComp R (Subtype fun x => Membership.m …
    ⊢ Eq (((killingForm R L) m₀) m₁) 0
  -/
  exact killingForm_eq_zero_of_mem_zeroRoot_mem_posFitting R L H (le_zeroRootSubalgebra R L H h₀) h₁
  /-
    🎉 no goals
  -/


@[simp] lemma ker_traceForm_eq_bot_of_isCartanSubalgebra
    [IsNoetherian R L] [IsArtinian R L] (H : LieSubalgebra R L) [H.IsCartanSubalgebra] :
    LinearMap.ker (LieModule.traceForm R H L) = ⊥ :=
  ker_restrict_eq_bot_of_isCartanSubalgebra R L H


lemma traceForm_cartan_nondegenerate
    [IsNoetherian R L] [IsArtinian R L] (H : LieSubalgebra R L) [H.IsCartanSubalgebra] :
    (LieModule.traceForm R H L).Nondegenerate := by
  /-
    R : Type u_1
    L : Type u_3
    inst✝⁶ : CommRing R
    inst✝⁵ : LieRing L
    inst✝⁴ : LieAlgebra R L
    inst✝³ : LieAlgebra.IsKilling R L
    inst✝² : IsNoetherian R L
    inst✝¹ : IsArtinian R L
    H : LieSubalgebra R L
    inst✝ : H.IsCartanSubalgebra
    ⊢ (LieModule.traceForm R (Subtype fun x => Membership.mem H x) L).Nondegenerate
  -/
  simp [LinearMap.BilinForm.nondegenerate_iff_ker_eq_bot]
  /-
    🎉 no goals
  -/


instance instIsLieAbelianOfIsCartanSubalgebra
    [IsDomain R] [IsPrincipalIdealRing R] [IsArtinian R L]
    (H : LieSubalgebra R L) [H.IsCartanSubalgebra] :
    IsLieAbelian H :=
  LieModule.isLieAbelian_of_ker_traceForm_eq_bot R H L <|
    ker_restrict_eq_bot_of_isCartanSubalgebra R L H


/-- For any `α` and `β`, the corresponding root spaces are orthogonal with respect to the Killing
form, provided `α + β ≠ 0`. -/
lemma killingForm_apply_eq_zero_of_mem_rootSpace_of_add_ne_zero {α β : H → K} {x y : L}
    (hx : x ∈ rootSpace H α) (hy : y ∈ rootSpace H β) (hαβ : α + β ≠ 0) :
    killingForm K L x y = 0 := by
  /- If `ad R L z` is semisimple for all `z ∈ H` then writing `⟪x, y⟫ = killingForm K L x y`, there
  is a slick proof of this lemma that requires only invariance of the Killing form as follows.
  For any `z ∈ H`, we have:
  `α z • ⟪x, y⟫ = ⟪α z • x, y⟫ = ⟪⁅z, x⁆, y⟫ = - ⟪x, ⁅z, y⁆⟫ = - ⟪x, β z • y⟫ = - β z • ⟪x, y⟫`.
  Since this is true for any `z`, we thus have: `(α + β) • ⟪x, y⟫ = 0`, and hence the result.
  However the semisimplicity of `ad R L z` is (a) non-trivial and (b) requires the assumption
  that `K` is a perfect field and `L` has non-degenerate Killing form. -/
  /-
    K : Type u_2
    L : Type u_3
    inst✝⁵ : LieRing L
    inst✝⁴ : Field K
    inst✝³ : LieAlgebra K L
    inst✝² : FiniteDimensional K L
    H : LieSubalgebra K L
    inst✝¹ : H.IsCartanSubalgebra
    inst✝ : LieModule.IsTriangularizable K (Subtype fun x => Membership.mem H x) L
    α β : (Subtype fun x => Membership.mem H x) → K
    x y : L
    hx : Membership.mem (LieAlgebra.rootSpace H α) x
    hy : Membership.mem (LieAlgebra.rootSpace H β) y
    hαβ : Ne (HAdd.hAdd α β) 0
    ⊢ Eq (((killingForm K L) x) y) 0
  -/
  let σ : (H → K) → (H → K) := fun γ ↦ α + (β + γ)
  /-
    K : Type u_2
    L : Type u_3
    inst✝⁵ : LieRing L
    inst✝⁴ : Field K
    inst✝³ : LieAlgebra K L
    inst✝² : FiniteDimensional K L
    H : LieSubalgebra K L
    inst✝¹ : H.IsCartanSubalgebra
    inst✝ : LieModule.IsTriangularizable K (Subtype fun x => Membership.mem H x) L
    α β : (Subtype fun x => Membership.mem H x) → K
    x y : L
    hx : Membership.mem (LieAlgebra.rootSpace H α) x
    hy : Membership.mem (LieAlgebra.rootSpace H β) y
    hαβ : Ne (HAdd.hAdd α β) 0
    σ : ((Subtype fun x => Membership.mem H x) → K) → (Subtype fun x => Membership …
    ⊢ Eq (((killingForm K L) x) y) 0
  -/
  have hσ : ∀ γ, σ γ ≠ γ := fun γ ↦ by simpa only [σ, ← add_assoc] using add_left_ne_self.mpr hαβ
  /-
    K : Type u_2
    L : Type u_3
    inst✝⁵ : LieRing L
    inst✝⁴ : Field K
    inst✝³ : LieAlgebra K L
    inst✝² : FiniteDimensional K L
    H : LieSubalgebra K L
    inst✝¹ : H.IsCartanSubalgebra
    inst✝ : LieModule.IsTriangularizable K (Subtype fun x => Membership.mem H x) L
    α β : (Subtype fun x => Membership.mem H x) → K
    x y : L
    hx : Membership.mem (LieAlgebra.rootSpace H α) x
    hy : Membership.mem (LieAlgebra.rootSpace H β) y
    hαβ : Ne (HAdd.hAdd α β) 0
    σ : ((Subtype fun x => Membership.mem H x) → K) → (Subtype fun x => Membership …
    hσ : ∀ (γ : (Subtype fun x => Membership.mem H x) → K), Ne (σ γ) γ
    ⊢ Eq (((killingForm K L) x) y) 0
  -/
  let f : Module.End K L := (ad K L x) ∘ₗ (ad K L y)
  have hf : ∀ γ, MapsTo f (rootSpace H γ) (rootSpace H (σ γ)) := fun γ ↦
    (mapsTo_toEnd_genWeightSpace_add_of_mem_rootSpace K L H L α (β + γ) hx).comp <|
      mapsTo_toEnd_genWeightSpace_add_of_mem_rootSpace K L H L β γ hy
  classical
  have hds := DirectSum.isInternal_submodule_of_iSupIndep_of_iSup_eq_top
    (LieSubmodule.iSupIndep_iff_toSubmodule.mp <| iSupIndep_genWeightSpace K H L)
    (LieSubmodule.iSup_eq_top_iff_toSubmodule.mp <| iSup_genWeightSpace_eq_top K H L)
  exact LinearMap.trace_eq_zero_of_mapsTo_ne hds σ hσ hf


/-- Elements of the `α` root space which are Killing-orthogonal to the `-α` root space are
Killing-orthogonal to all of `L`. -/
lemma mem_ker_killingForm_of_mem_rootSpace_of_forall_rootSpace_neg
    {α : H → K} {x : L} (hx : x ∈ rootSpace H α)
    (hx' : ∀ y ∈ rootSpace H (-α), killingForm K L x y = 0) :
    x ∈ LinearMap.ker (killingForm K L) := by
  /-
    K : Type u_2
    L : Type u_3
    inst✝⁵ : LieRing L
    inst✝⁴ : Field K
    inst✝³ : LieAlgebra K L
    inst✝² : FiniteDimensional K L
    H : LieSubalgebra K L
    inst✝¹ : H.IsCartanSubalgebra
    inst✝ : LieModule.IsTriangularizable K (Subtype fun x => Membership.mem H x) L
    α : (Subtype fun x => Membership.mem H x) → K
    x : L
    hx : Membership.mem (LieAlgebra.rootSpace H α) x
    hx' : ∀ (y : L), Membership.mem (LieAlgebra.rootSpace H (Neg.neg α)) y → Eq (( …
    ⊢ Membership.mem (LinearMap.ker (killingForm K L)) x
  -/
  rw [LinearMap.mem_ker]
  /-
    K : Type u_2
    L : Type u_3
    inst✝⁵ : LieRing L
    inst✝⁴ : Field K
    inst✝³ : LieAlgebra K L
    inst✝² : FiniteDimensional K L
    H : LieSubalgebra K L
    inst✝¹ : H.IsCartanSubalgebra
    inst✝ : LieModule.IsTriangularizable K (Subtype fun x => Membership.mem H x) L
    α : (Subtype fun x => Membership.mem H x) → K
    x : L
    hx : Membership.mem (LieAlgebra.rootSpace H α) x
    hx' : ∀ (y : L), Membership.mem (LieAlgebra.rootSpace H (Neg.neg α)) y → Eq (( …
    ⊢ Eq ((killingForm K L) x) 0
  -/
  ext y
  /-
    case h
    K : Type u_2
    L : Type u_3
    inst✝⁵ : LieRing L
    inst✝⁴ : Field K
    inst✝³ : LieAlgebra K L
    inst✝² : FiniteDimensional K L
    H : LieSubalgebra K L
    inst✝¹ : H.IsCartanSubalgebra
    inst✝ : LieModule.IsTriangularizable K (Subtype fun x => Membership.mem H x) L
    α : (Subtype fun x => Membership.mem H x) → K
    x : L
    hx : Membership.mem (LieAlgebra.rootSpace H α) x
    hx' : ∀ (y : L), Membership.mem (LieAlgebra.rootSpace H (Neg.neg α)) y → Eq (( …
    y : L
    ⊢ Eq (((killingForm K L) x) y) (0 y)
  -/
  have hy : y ∈ ⨆ β, rootSpace H β := by simp [iSup_genWeightSpace_eq_top K H L]
  induction hy using LieSubmodule.iSup_induction' with
  | hN β y hy =>
    by_cases hαβ : α + β = 0
    · exact hx' _ (add_eq_zero_iff_neg_eq.mp hαβ ▸ hy)
    · exact killingForm_apply_eq_zero_of_mem_rootSpace_of_add_ne_zero K L H hx hy hαβ
  | h0 => simp
  | hadd => simp_all

/-- If a Lie algebra `L` has non-degenerate Killing form, the only element of a Cartan subalgebra
whose adjoint action on `L` is nilpotent, is the zero element.

Over a perfect field a much stronger result is true, see
`LieAlgebra.IsKilling.isSemisimple_ad_of_mem_isCartanSubalgebra`. -/
lemma eq_zero_of_isNilpotent_ad_of_mem_isCartanSubalgebra {x : L} (hx : x ∈ H)
    (hx' : _root_.IsNilpotent (ad K L x)) : x = 0 := by
  suffices ⟨x, hx⟩ ∈ LinearMap.ker (traceForm K H L) by
    simp at this
    exact (AddSubmonoid.mk_eq_zero H.toAddSubmonoid).mp this
  /-
    K : Type u_2
    L : Type u_3
    inst✝⁵ : LieRing L
    inst✝⁴ : Field K
    inst✝³ : LieAlgebra K L
    inst✝² : FiniteDimensional K L
    H : LieSubalgebra K L
    inst✝¹ : H.IsCartanSubalgebra
    inst✝ : LieAlgebra.IsKilling K L
    x : L
    hx : Membership.mem H x
    hx' : _root_.IsNilpotent ((LieAlgebra.ad K L) x)
    ⊢ Membership.mem (LinearMap.ker (LieModule.traceForm K (Subtype fun x => Membe …
  -/
  simp only [LinearMap.mem_ker]
  /-
    K : Type u_2
    L : Type u_3
    inst✝⁵ : LieRing L
    inst✝⁴ : Field K
    inst✝³ : LieAlgebra K L
    inst✝² : FiniteDimensional K L
    H : LieSubalgebra K L
    inst✝¹ : H.IsCartanSubalgebra
    inst✝ : LieAlgebra.IsKilling K L
    x : L
    hx : Membership.mem H x
    hx' : _root_.IsNilpotent ((LieAlgebra.ad K L) x)
    ⊢ Eq ((LieModule.traceForm K (Subtype fun x => Membership.mem H x) L) ⟨x, hx⟩) 0
  -/
  ext y
  have comm : Commute (toEnd K H L ⟨x, hx⟩) (toEnd K H L y) := by
    rw [commute_iff_lie_eq, ← LieHom.map_lie, trivial_lie_zero, LieHom.map_zero]
  /-
    case h
    K : Type u_2
    L : Type u_3
    inst✝⁵ : LieRing L
    inst✝⁴ : Field K
    inst✝³ : LieAlgebra K L
    inst✝² : FiniteDimensional K L
    H : LieSubalgebra K L
    inst✝¹ : H.IsCartanSubalgebra
    inst✝ : LieAlgebra.IsKilling K L
    x : L
    hx : Membership.mem H x
    hx' : _root_.IsNilpotent ((LieAlgebra.ad K L) x)
    y : Subtype fun x => Membership.mem H x
    comm : Commute ((LieModule.toEnd K (Subtype fun x => Membership.mem H x) L) ⟨x …
    ⊢ Eq (((LieModule.traceForm K (Subtype fun x => Membership.mem H x) L) ⟨x, hx⟩ …
  -/
  rw [traceForm_apply_apply, ← LinearMap.mul_eq_comp, LinearMap.zero_apply]
  /-
    case h
    K : Type u_2
    L : Type u_3
    inst✝⁵ : LieRing L
    inst✝⁴ : Field K
    inst✝³ : LieAlgebra K L
    inst✝² : FiniteDimensional K L
    H : LieSubalgebra K L
    inst✝¹ : H.IsCartanSubalgebra
    inst✝ : LieAlgebra.IsKilling K L
    x : L
    hx : Membership.mem H x
    hx' : _root_.IsNilpotent ((LieAlgebra.ad K L) x)
    y : Subtype fun x => Membership.mem H x
    comm : Commute ((LieModule.toEnd K (Subtype fun x => Membership.mem H x) L) ⟨x …
    ⊢ Eq ((LinearMap.trace K L) (HMul.hMul ((LieModule.toEnd K (Subtype fun x => M …
  -/
  exact (LinearMap.isNilpotent_trace_of_isNilpotent (comm.isNilpotent_mul_left hx')).eq_zero
  /-
    🎉 no goals
  -/


@[simp]
lemma corootSpace_zero_eq_bot :
    corootSpace (0 : H → K) = ⊥ := by
  /-
    K : Type u_2
    L : Type u_3
    inst✝⁵ : LieRing L
    inst✝⁴ : Field K
    inst✝³ : LieAlgebra K L
    inst✝² : FiniteDimensional K L
    H : LieSubalgebra K L
    inst✝¹ : H.IsCartanSubalgebra
    inst✝ : LieAlgebra.IsKilling K L
    ⊢ Eq (LieAlgebra.corootSpace 0) Bot.bot
  -/
  refine eq_bot_iff.mpr fun x hx ↦ ?_
  /-
    K : Type u_2
    L : Type u_3
    inst✝⁵ : LieRing L
    inst✝⁴ : Field K
    inst✝³ : LieAlgebra K L
    inst✝² : FiniteDimensional K L
    H : LieSubalgebra K L
    inst✝¹ : H.IsCartanSubalgebra
    inst✝ : LieAlgebra.IsKilling K L
    x : Subtype fun x => Membership.mem H x
    hx : Membership.mem (LieAlgebra.corootSpace 0) x
    ⊢ Membership.mem Bot.bot x
  -/
  suffices {x | ∃ y ∈ H, ∃ z ∈ H, ⁅y, z⁆ = x} = {0} by simpa [mem_corootSpace, this] using hx
  /-
    K : Type u_2
    L : Type u_3
    inst✝⁵ : LieRing L
    inst✝⁴ : Field K
    inst✝³ : LieAlgebra K L
    inst✝² : FiniteDimensional K L
    H : LieSubalgebra K L
    inst✝¹ : H.IsCartanSubalgebra
    inst✝ : LieAlgebra.IsKilling K L
    x : Subtype fun x => Membership.mem H x
    hx : Membership.mem (LieAlgebra.corootSpace 0) x
    ⊢ Eq (setOf fun x => Exists fun y => And (Membership.mem H y) (Exists fun z => …
  -/
  refine eq_singleton_iff_unique_mem.mpr ⟨⟨0, H.zero_mem, 0, H.zero_mem, zero_lie 0⟩, ?_⟩
  /-
    K : Type u_2
    L : Type u_3
    inst✝⁵ : LieRing L
    inst✝⁴ : Field K
    inst✝³ : LieAlgebra K L
    inst✝² : FiniteDimensional K L
    H : LieSubalgebra K L
    inst✝¹ : H.IsCartanSubalgebra
    inst✝ : LieAlgebra.IsKilling K L
    x : Subtype fun x => Membership.mem H x
    hx : Membership.mem (LieAlgebra.corootSpace 0) x
    ⊢ ∀ (x : L), Membership.mem (setOf fun x => Exists fun y => And (Membership.me …
  -/
  rintro - ⟨y, hy, z, hz, rfl⟩
  suffices ⁅(⟨y, hy⟩ : H), (⟨z, hz⟩ : H)⁆ = 0 by
    simpa only [Subtype.ext_iff, LieSubalgebra.coe_bracket, ZeroMemClass.coe_zero] using this
  /-
    case intro.intro.intro.intro
    K : Type u_2
    L : Type u_3
    inst✝⁵ : LieRing L
    inst✝⁴ : Field K
    inst✝³ : LieAlgebra K L
    inst✝² : FiniteDimensional K L
    H : LieSubalgebra K L
    inst✝¹ : H.IsCartanSubalgebra
    inst✝ : LieAlgebra.IsKilling K L
    x : Subtype fun x => Membership.mem H x
    hx : Membership.mem (LieAlgebra.corootSpace 0) x
    y : L
    hy : Membership.mem H y
    z : L
    hz : Membership.mem H z
    ⊢ Eq (Bracket.bracket ⟨y, hy⟩ ⟨z, hz⟩) 0
  -/
  simp
  /-
    🎉 no goals
  -/


variable {K L} in
/-- The restriction of the Killing form to a Cartan subalgebra, as a linear equivalence to the
dual. -/
@[simps! apply_apply]
noncomputable def cartanEquivDual :
    H ≃ₗ[K] Module.Dual K H :=
  (traceForm K H L).toDual <| traceForm_cartan_nondegenerate K L H


/-- The coroot corresponding to a root. -/
noncomputable def coroot (α : Weight K H L) : H :=
  2 • (α <| (cartanEquivDual H).symm α)⁻¹ • (cartanEquivDual H).symm α


lemma traceForm_coroot (α : Weight K H L) (x : H) :
    traceForm K H L (coroot α) x = 2 • (α <| (cartanEquivDual H).symm α)⁻¹ • α x := by
  have : cartanEquivDual H ((cartanEquivDual H).symm α) x = α x := by
    rw [LinearEquiv.apply_symm_apply, Weight.toLinear_apply]
  /-
    K : Type u_2
    L : Type u_3
    inst✝⁵ : LieRing L
    inst✝⁴ : Field K
    inst✝³ : LieAlgebra K L
    inst✝² : FiniteDimensional K L
    H : LieSubalgebra K L
    inst✝¹ : H.IsCartanSubalgebra
    inst✝ : LieAlgebra.IsKilling K L
    α : LieModule.Weight K (Subtype fun x => Membership.mem H x) L
    x : Subtype fun x => Membership.mem H x
    this : Eq (((LieAlgebra.IsKilling.cartanEquivDual H) ((LieAlgebra.IsKilling.ca …
    ⊢ Eq (((LieModule.traceForm K (Subtype fun x => Membership.mem H x) L) (LieAlg …
  -/
  rw [coroot, map_nsmul, map_smul, LinearMap.smul_apply, LinearMap.smul_apply]
  /-
    K : Type u_2
    L : Type u_3
    inst✝⁵ : LieRing L
    inst✝⁴ : Field K
    inst✝³ : LieAlgebra K L
    inst✝² : FiniteDimensional K L
    H : LieSubalgebra K L
    inst✝¹ : H.IsCartanSubalgebra
    inst✝ : LieAlgebra.IsKilling K L
    α : LieModule.Weight K (Subtype fun x => Membership.mem H x) L
    x : Subtype fun x => Membership.mem H x
    this : Eq (((LieAlgebra.IsKilling.cartanEquivDual H) ((LieAlgebra.IsKilling.ca …
    ⊢ Eq (HSMul.hSMul 2 (HSMul.hSMul (Inv.inv (α ((LieAlgebra.IsKilling.cartanEqui …
  -/
  congr 2
  /-
    🎉 no goals
  -/


lemma lie_eq_killingForm_smul_of_mem_rootSpace_of_mem_rootSpace_neg_aux
    {α : Weight K H L} {e f : L} (heα : e ∈ rootSpace H α) (hfα : f ∈ rootSpace H (-α))
    (aux : ∀ (h : H), ⁅h, e⁆ = α h • e) :
    ⁅e, f⁆ = killingForm K L e f • (cartanEquivDual H).symm α := by
  /-
    K : Type u_2
    L : Type u_3
    inst✝⁶ : LieRing L
    inst✝⁵ : Field K
    inst✝⁴ : LieAlgebra K L
    inst✝³ : FiniteDimensional K L
    H : LieSubalgebra K L
    inst✝² : H.IsCartanSubalgebra
    inst✝¹ : LieAlgebra.IsKilling K L
    inst✝ : LieModule.IsTriangularizable K (Subtype fun x => Membership.mem H x) L
    α : LieModule.Weight K (Subtype fun x => Membership.mem H x) L
    e f : L
    heα : Membership.mem (LieAlgebra.rootSpace H ⇑α) e
    hfα : Membership.mem (LieAlgebra.rootSpace H (Neg.neg ⇑α)) f
    aux : ∀ (h : Subtype fun x => Membership.mem H x), Eq (Bracket.bracket h e) (H …
    ⊢ Eq (Bracket.bracket e f) (HSMul.hSMul (((killingForm K L) e) f) ↑((LieAlgebr …
  -/
  set α' := (cartanEquivDual H).symm α
  /-
    K : Type u_2
    L : Type u_3
    inst✝⁶ : LieRing L
    inst✝⁵ : Field K
    inst✝⁴ : LieAlgebra K L
    inst✝³ : FiniteDimensional K L
    H : LieSubalgebra K L
    inst✝² : H.IsCartanSubalgebra
    inst✝¹ : LieAlgebra.IsKilling K L
    inst✝ : LieModule.IsTriangularizable K (Subtype fun x => Membership.mem H x) L
    α : LieModule.Weight K (Subtype fun x => Membership.mem H x) L
    e f : L
    heα : Membership.mem (LieAlgebra.rootSpace H ⇑α) e
    hfα : Membership.mem (LieAlgebra.rootSpace H (Neg.neg ⇑α)) f
    aux : ∀ (h : Subtype fun x => Membership.mem H x), Eq (Bracket.bracket h e) (H …
    α' : Subtype fun x => Membership.mem H x := (LieAlgebra.IsKilling.cartanEquivD …
    ⊢ Eq (Bracket.bracket e f) (HSMul.hSMul (((killingForm K L) e) f) ↑α')
  -/
  rw [← sub_eq_zero, ← Submodule.mem_bot (R := K), ← ker_killingForm_eq_bot]
  /-
    K : Type u_2
    L : Type u_3
    inst✝⁶ : LieRing L
    inst✝⁵ : Field K
    inst✝⁴ : LieAlgebra K L
    inst✝³ : FiniteDimensional K L
    H : LieSubalgebra K L
    inst✝² : H.IsCartanSubalgebra
    inst✝¹ : LieAlgebra.IsKilling K L
    inst✝ : LieModule.IsTriangularizable K (Subtype fun x => Membership.mem H x) L
    α : LieModule.Weight K (Subtype fun x => Membership.mem H x) L
    e f : L
    heα : Membership.mem (LieAlgebra.rootSpace H ⇑α) e
    hfα : Membership.mem (LieAlgebra.rootSpace H (Neg.neg ⇑α)) f
    aux : ∀ (h : Subtype fun x => Membership.mem H x), Eq (Bracket.bracket h e) (H …
    α' : Subtype fun x => Membership.mem H x := (LieAlgebra.IsKilling.cartanEquivD …
    ⊢ Membership.mem (LinearMap.ker (killingForm K L)) (HSub.hSub (Bracket.bracket …
  -/
  apply mem_ker_killingForm_of_mem_rootSpace_of_forall_rootSpace_neg (α := (0 : H → K))
    /-
      case hx
      K : Type u_2
      L : Type u_3
      inst✝⁶ : LieRing L
      inst✝⁵ : Field K
      inst✝⁴ : LieAlgebra K L
      inst✝³ : FiniteDimensional K L
      H : LieSubalgebra K L
      inst✝² : H.IsCartanSubalgebra
      inst✝¹ : LieAlgebra.IsKilling K L
      inst✝ : LieModule.IsTriangularizable K (Subtype fun x => Membership.mem H x) L
      α : LieModule.Weight K (Subtype fun x => Membership.mem H x) L
      e f : L
      heα : Membership.mem (LieAlgebra.rootSpace H ⇑α) e
      hfα : Membership.mem (LieAlgebra.rootSpace H (Neg.neg ⇑α)) f
      aux : ∀ (h : Subtype fun x => Membership.mem H x), Eq (Bracket.bracket h e) (H …
      α' : Subtype fun x => Membership.mem H x := (LieAlgebra.IsKilling.cartanEquivD …
      ⊢ Membership.mem (LieAlgebra.rootSpace H 0) (HSub.hSub (Bracket.bracket e f) ( …
    -/
  · simp only [rootSpace_zero_eq, LieSubalgebra.mem_toLieSubmodule]
    /-
      case hx
      K : Type u_2
      L : Type u_3
      inst✝⁶ : LieRing L
      inst✝⁵ : Field K
      inst✝⁴ : LieAlgebra K L
      inst✝³ : FiniteDimensional K L
      H : LieSubalgebra K L
      inst✝² : H.IsCartanSubalgebra
      inst✝¹ : LieAlgebra.IsKilling K L
      inst✝ : LieModule.IsTriangularizable K (Subtype fun x => Membership.mem H x) L
      α : LieModule.Weight K (Subtype fun x => Membership.mem H x) L
      e f : L
      heα : Membership.mem (LieAlgebra.rootSpace H ⇑α) e
      hfα : Membership.mem (LieAlgebra.rootSpace H (Neg.neg ⇑α)) f
      aux : ∀ (h : Subtype fun x => Membership.mem H x), Eq (Bracket.bracket h e) (H …
      α' : Subtype fun x => Membership.mem H x := (LieAlgebra.IsKilling.cartanEquivD …
      ⊢ Membership.mem H (HSub.hSub (Bracket.bracket e f) (HSMul.hSMul (((killingFor …
    -/
    refine sub_mem ?_ (H.smul_mem _ α'.property)
    /-
      case hx
      K : Type u_2
      L : Type u_3
      inst✝⁶ : LieRing L
      inst✝⁵ : Field K
      inst✝⁴ : LieAlgebra K L
      inst✝³ : FiniteDimensional K L
      H : LieSubalgebra K L
      inst✝² : H.IsCartanSubalgebra
      inst✝¹ : LieAlgebra.IsKilling K L
      inst✝ : LieModule.IsTriangularizable K (Subtype fun x => Membership.mem H x) L
      α : LieModule.Weight K (Subtype fun x => Membership.mem H x) L
      e f : L
      heα : Membership.mem (LieAlgebra.rootSpace H ⇑α) e
      hfα : Membership.mem (LieAlgebra.rootSpace H (Neg.neg ⇑α)) f
      aux : ∀ (h : Subtype fun x => Membership.mem H x), Eq (Bracket.bracket h e) (H …
      α' : Subtype fun x => Membership.mem H x := (LieAlgebra.IsKilling.cartanEquivD …
      ⊢ Membership.mem H (Bracket.bracket e f)
    -/
    simpa using mapsTo_toEnd_genWeightSpace_add_of_mem_rootSpace K L H L α (-α) heα hfα
    /-
      🎉 no goals
    -/
    /-
      case hx'
      K : Type u_2
      L : Type u_3
      inst✝⁶ : LieRing L
      inst✝⁵ : Field K
      inst✝⁴ : LieAlgebra K L
      inst✝³ : FiniteDimensional K L
      H : LieSubalgebra K L
      inst✝² : H.IsCartanSubalgebra
      inst✝¹ : LieAlgebra.IsKilling K L
      inst✝ : LieModule.IsTriangularizable K (Subtype fun x => Membership.mem H x) L
      α : LieModule.Weight K (Subtype fun x => Membership.mem H x) L
      e f : L
      heα : Membership.mem (LieAlgebra.rootSpace H ⇑α) e
      hfα : Membership.mem (LieAlgebra.rootSpace H (Neg.neg ⇑α)) f
      aux : ∀ (h : Subtype fun x => Membership.mem H x), Eq (Bracket.bracket h e) (H …
      α' : Subtype fun x => Membership.mem H x := (LieAlgebra.IsKilling.cartanEquivD …
      ⊢ ∀ (y : L), Membership.mem (LieAlgebra.rootSpace H (-0)) y → Eq (((killingFor …
    -/
  · intro z hz
    /-
      case hx'
      K : Type u_2
      L : Type u_3
      inst✝⁶ : LieRing L
      inst✝⁵ : Field K
      inst✝⁴ : LieAlgebra K L
      inst✝³ : FiniteDimensional K L
      H : LieSubalgebra K L
      inst✝² : H.IsCartanSubalgebra
      inst✝¹ : LieAlgebra.IsKilling K L
      inst✝ : LieModule.IsTriangularizable K (Subtype fun x => Membership.mem H x) L
      α : LieModule.Weight K (Subtype fun x => Membership.mem H x) L
      e f : L
      heα : Membership.mem (LieAlgebra.rootSpace H ⇑α) e
      hfα : Membership.mem (LieAlgebra.rootSpace H (Neg.neg ⇑α)) f
      aux : ∀ (h : Subtype fun x => Membership.mem H x), Eq (Bracket.bracket h e) (H …
      α' : Subtype fun x => Membership.mem H x := (LieAlgebra.IsKilling.cartanEquivD …
      z : L
      hz : Membership.mem (LieAlgebra.rootSpace H (-0)) z
      ⊢ Eq (((killingForm K L) (HSub.hSub (Bracket.bracket e f) (HSMul.hSMul (((kill …
    -/
    replace hz : z ∈ H := by simpa using hz
    /-
      case hx'
      K : Type u_2
      L : Type u_3
      inst✝⁶ : LieRing L
      inst✝⁵ : Field K
      inst✝⁴ : LieAlgebra K L
      inst✝³ : FiniteDimensional K L
      H : LieSubalgebra K L
      inst✝² : H.IsCartanSubalgebra
      inst✝¹ : LieAlgebra.IsKilling K L
      inst✝ : LieModule.IsTriangularizable K (Subtype fun x => Membership.mem H x) L
      α : LieModule.Weight K (Subtype fun x => Membership.mem H x) L
      e f : L
      heα : Membership.mem (LieAlgebra.rootSpace H ⇑α) e
      hfα : Membership.mem (LieAlgebra.rootSpace H (Neg.neg ⇑α)) f
      aux : ∀ (h : Subtype fun x => Membership.mem H x), Eq (Bracket.bracket h e) (H …
      α' : Subtype fun x => Membership.mem H x := (LieAlgebra.IsKilling.cartanEquivD …
      z : L
      hz : Membership.mem H z
      ⊢ Eq (((killingForm K L) (HSub.hSub (Bracket.bracket e f) (HSMul.hSMul (((kill …
    -/
    have he : ⁅z, e⁆ = α ⟨z, hz⟩ • e := aux ⟨z, hz⟩
    have hαz : killingForm K L α' (⟨z, hz⟩ : H) = α ⟨z, hz⟩ :=
      LinearMap.BilinForm.apply_toDual_symm_apply (hB := traceForm_cartan_nondegenerate K L H) _ _
    /-
      case hx'
      K : Type u_2
      L : Type u_3
      inst✝⁶ : LieRing L
      inst✝⁵ : Field K
      inst✝⁴ : LieAlgebra K L
      inst✝³ : FiniteDimensional K L
      H : LieSubalgebra K L
      inst✝² : H.IsCartanSubalgebra
      inst✝¹ : LieAlgebra.IsKilling K L
      inst✝ : LieModule.IsTriangularizable K (Subtype fun x => Membership.mem H x) L
      α : LieModule.Weight K (Subtype fun x => Membership.mem H x) L
      e f : L
      heα : Membership.mem (LieAlgebra.rootSpace H ⇑α) e
      hfα : Membership.mem (LieAlgebra.rootSpace H (Neg.neg ⇑α)) f
      aux : ∀ (h : Subtype fun x => Membership.mem H x), Eq (Bracket.bracket h e) (H …
      α' : Subtype fun x => Membership.mem H x := (LieAlgebra.IsKilling.cartanEquivD …
      z : L
      hz : Membership.mem H z
      he : Eq (Bracket.bracket z e) (HSMul.hSMul (α ⟨z, hz⟩) e)
      hαz : Eq (((killingForm K L) ↑α') ↑⟨z, hz⟩) (α ⟨z, hz⟩)
      ⊢ Eq (((killingForm K L) (HSub.hSub (Bracket.bracket e f) (HSMul.hSMul (((kill …
    -/
    simp [traceForm_comm K L L ⁅e, f⁆, ← traceForm_apply_lie_apply, he, mul_comm _ (α ⟨z, hz⟩), hαz]
    /-
      🎉 no goals
    -/


/-- This is Proposition 4.18 from [carter2005] except that we use
`LieModule.exists_forall_lie_eq_smul` instead of Lie's theorem (and so avoid
assuming `K` has characteristic zero). -/
lemma cartanEquivDual_symm_apply_mem_corootSpace (α : Weight K H L) :
    (cartanEquivDual H).symm α ∈ corootSpace α := by
  /-
    K : Type u_2
    L : Type u_3
    inst✝⁶ : LieRing L
    inst✝⁵ : Field K
    inst✝⁴ : LieAlgebra K L
    inst✝³ : FiniteDimensional K L
    H : LieSubalgebra K L
    inst✝² : H.IsCartanSubalgebra
    inst✝¹ : LieAlgebra.IsKilling K L
    inst✝ : LieModule.IsTriangularizable K (Subtype fun x => Membership.mem H x) L
    α : LieModule.Weight K (Subtype fun x => Membership.mem H x) L
    ⊢ Membership.mem (LieAlgebra.corootSpace ⇑α) ((LieAlgebra.IsKilling.cartanEqui …
  -/
  obtain ⟨e : L, he₀ : e ≠ 0, he : ∀ x, ⁅x, e⁆ = α x • e⟩ := exists_forall_lie_eq_smul K H L α
  /-
    case intro.intro
    K : Type u_2
    L : Type u_3
    inst✝⁶ : LieRing L
    inst✝⁵ : Field K
    inst✝⁴ : LieAlgebra K L
    inst✝³ : FiniteDimensional K L
    H : LieSubalgebra K L
    inst✝² : H.IsCartanSubalgebra
    inst✝¹ : LieAlgebra.IsKilling K L
    inst✝ : LieModule.IsTriangularizable K (Subtype fun x => Membership.mem H x) L
    α : LieModule.Weight K (Subtype fun x => Membership.mem H x) L
    e : L
    he₀ : Ne e 0
    he : ∀ (x : Subtype fun x => Membership.mem H x), Eq (Bracket.bracket x e) (HS …
    ⊢ Membership.mem (LieAlgebra.corootSpace ⇑α) ((LieAlgebra.IsKilling.cartanEqui …
  -/
  have heα : e ∈ rootSpace H α := (mem_genWeightSpace L α e).mpr fun x ↦ ⟨1, by simp [← he x]⟩
  obtain ⟨f, hfα, hf⟩ : ∃ f ∈ rootSpace H (-α), killingForm K L e f ≠ 0 := by
    contrapose! he₀
    simpa using mem_ker_killingForm_of_mem_rootSpace_of_forall_rootSpace_neg K L H heα he₀
  suffices ⁅e, f⁆ = killingForm K L e f • ((cartanEquivDual H).symm α : L) from
    (mem_corootSpace α).mpr <| Submodule.subset_span ⟨(killingForm K L e f)⁻¹ • e,
      Submodule.smul_mem _ _ heα, f, hfα, by simpa [inv_smul_eq_iff₀ hf]⟩
  /-
    case intro.intro.intro.intro
    K : Type u_2
    L : Type u_3
    inst✝⁶ : LieRing L
    inst✝⁵ : Field K
    inst✝⁴ : LieAlgebra K L
    inst✝³ : FiniteDimensional K L
    H : LieSubalgebra K L
    inst✝² : H.IsCartanSubalgebra
    inst✝¹ : LieAlgebra.IsKilling K L
    inst✝ : LieModule.IsTriangularizable K (Subtype fun x => Membership.mem H x) L
    α : LieModule.Weight K (Subtype fun x => Membership.mem H x) L
    e : L
    he₀ : Ne e 0
    he : ∀ (x : Subtype fun x => Membership.mem H x), Eq (Bracket.bracket x e) (HS …
    heα : Membership.mem (LieAlgebra.rootSpace H ⇑α) e
    f : L
    hfα : Membership.mem (LieAlgebra.rootSpace H (Neg.neg ⇑α)) f
    hf : Ne (((killingForm K L) e) f) 0
    ⊢ Eq (Bracket.bracket e f) (HSMul.hSMul (((killingForm K L) e) f) ↑((LieAlgebr …
  -/
  exact lie_eq_killingForm_smul_of_mem_rootSpace_of_mem_rootSpace_neg_aux heα hfα he
  /-
    🎉 no goals
  -/


/-- Given a splitting Cartan subalgebra `H` of a finite-dimensional Lie algebra with non-singular
Killing form, the corresponding roots span the dual space of `H`. -/
@[simp]
lemma span_weight_eq_top :
    span K (range (Weight.toLinear K H L)) = ⊤ := by
  /-
    K : Type u_2
    L : Type u_3
    inst✝⁶ : LieRing L
    inst✝⁵ : Field K
    inst✝⁴ : LieAlgebra K L
    inst✝³ : FiniteDimensional K L
    H : LieSubalgebra K L
    inst✝² : H.IsCartanSubalgebra
    inst✝¹ : LieAlgebra.IsKilling K L
    inst✝ : LieModule.IsTriangularizable K (Subtype fun x => Membership.mem H x) L
    ⊢ Eq (Submodule.span K (Set.range (LieModule.Weight.toLinear K (Subtype fun x  …
  -/
  refine eq_top_iff.mpr (le_trans ?_ (LieModule.range_traceForm_le_span_weight K H L))
  rw [← traceForm_flip K H L, ← LinearMap.dualAnnihilator_ker_eq_range_flip,
    ker_traceForm_eq_bot_of_isCartanSubalgebra, Submodule.dualAnnihilator_bot]


variable (K L H) in
@[simp]
lemma span_weight_isNonZero_eq_top :
    span K ({α : Weight K H L | α.IsNonZero}.image (Weight.toLinear K H L)) = ⊤ := by
  /-
    K : Type u_2
    L : Type u_3
    inst✝⁶ : LieRing L
    inst✝⁵ : Field K
    inst✝⁴ : LieAlgebra K L
    inst✝³ : FiniteDimensional K L
    H : LieSubalgebra K L
    inst✝² : H.IsCartanSubalgebra
    inst✝¹ : LieAlgebra.IsKilling K L
    inst✝ : LieModule.IsTriangularizable K (Subtype fun x => Membership.mem H x) L
    ⊢ Eq (Submodule.span K (Set.image (LieModule.Weight.toLinear K (Subtype fun x  …
  -/
  rw [← span_weight_eq_top]
  /-
    K : Type u_2
    L : Type u_3
    inst✝⁶ : LieRing L
    inst✝⁵ : Field K
    inst✝⁴ : LieAlgebra K L
    inst✝³ : FiniteDimensional K L
    H : LieSubalgebra K L
    inst✝² : H.IsCartanSubalgebra
    inst✝¹ : LieAlgebra.IsKilling K L
    inst✝ : LieModule.IsTriangularizable K (Subtype fun x => Membership.mem H x) L
    ⊢ Eq (Submodule.span K (Set.image (LieModule.Weight.toLinear K (Subtype fun x  …
  -/
  refine le_antisymm (Submodule.span_mono <| by simp) ?_
  suffices range (Weight.toLinear K H L) ⊆
    insert 0 ({α : Weight K H L | α.IsNonZero}.image (Weight.toLinear K H L)) by
    simpa only [Submodule.span_insert_zero] using Submodule.span_mono this
  /-
    K : Type u_2
    L : Type u_3
    inst✝⁶ : LieRing L
    inst✝⁵ : Field K
    inst✝⁴ : LieAlgebra K L
    inst✝³ : FiniteDimensional K L
    H : LieSubalgebra K L
    inst✝² : H.IsCartanSubalgebra
    inst✝¹ : LieAlgebra.IsKilling K L
    inst✝ : LieModule.IsTriangularizable K (Subtype fun x => Membership.mem H x) L
    ⊢ HasSubset.Subset (Set.range (LieModule.Weight.toLinear K (Subtype fun x => M …
  -/
  rintro - ⟨α, rfl⟩
  /-
    case intro
    K : Type u_2
    L : Type u_3
    inst✝⁶ : LieRing L
    inst✝⁵ : Field K
    inst✝⁴ : LieAlgebra K L
    inst✝³ : FiniteDimensional K L
    H : LieSubalgebra K L
    inst✝² : H.IsCartanSubalgebra
    inst✝¹ : LieAlgebra.IsKilling K L
    inst✝ : LieModule.IsTriangularizable K (Subtype fun x => Membership.mem H x) L
    α : LieModule.Weight K (Subtype fun x => Membership.mem H x) L
    ⊢ Membership.mem (Insert.insert 0 (Set.image (LieModule.Weight.toLinear K (Sub …
  -/
  simp only [mem_insert_iff, Weight.coe_toLinear_eq_zero_iff, mem_image, mem_setOf_eq]
  /-
    case intro
    K : Type u_2
    L : Type u_3
    inst✝⁶ : LieRing L
    inst✝⁵ : Field K
    inst✝⁴ : LieAlgebra K L
    inst✝³ : FiniteDimensional K L
    H : LieSubalgebra K L
    inst✝² : H.IsCartanSubalgebra
    inst✝¹ : LieAlgebra.IsKilling K L
    inst✝ : LieModule.IsTriangularizable K (Subtype fun x => Membership.mem H x) L
    α : LieModule.Weight K (Subtype fun x => Membership.mem H x) L
    ⊢ Or α.IsZero (Exists fun x => And x.IsNonZero (Eq (LieModule.Weight.toLinear  …
  -/
  tauto
  /-
    🎉 no goals
  -/


@[simp]
lemma iInf_ker_weight_eq_bot :
    ⨅ α : Weight K H L, α.ker = ⊥ := by
  rw [← Subspace.dualAnnihilator_inj, Subspace.dualAnnihilator_iInf_eq,
    Submodule.dualAnnihilator_bot]
  /-
    K : Type u_2
    L : Type u_3
    inst✝⁶ : LieRing L
    inst✝⁵ : Field K
    inst✝⁴ : LieAlgebra K L
    inst✝³ : FiniteDimensional K L
    H : LieSubalgebra K L
    inst✝² : H.IsCartanSubalgebra
    inst✝¹ : LieAlgebra.IsKilling K L
    inst✝ : LieModule.IsTriangularizable K (Subtype fun x => Membership.mem H x) L
    ⊢ Eq (iSup fun i => LieModule.Weight.ker.dualAnnihilator) Top.top
  -/
  simp [← LinearMap.range_dualMap_eq_dualAnnihilator_ker, ← Submodule.span_range_eq_iSup]
  /-
    🎉 no goals
  -/


open Module.End in
lemma isSemisimple_ad_of_mem_isCartanSubalgebra {x : L} (hx : x ∈ H) :
    (ad K L x).IsSemisimple := by
  /- Using Jordan-Chevalley, write `ad K L x` as a sum of its semisimple and nilpotent parts. -/
  /-
    K : Type u_2
    L : Type u_3
    inst✝⁷ : LieRing L
    inst✝⁶ : Field K
    inst✝⁵ : LieAlgebra K L
    inst✝⁴ : FiniteDimensional K L
    H : LieSubalgebra K L
    inst✝³ : H.IsCartanSubalgebra
    inst✝² : LieAlgebra.IsKilling K L
    inst✝¹ : LieModule.IsTriangularizable K (Subtype fun x => Membership.mem H x) L
    inst✝ : PerfectField K
    x : L
    hx : Membership.mem H x
    ⊢ ((LieAlgebra.ad K L) x).IsSemisimple
  -/
  obtain ⟨N, -, S, hS₀, hN, hS, hSN⟩ := (ad K L x).exists_isNilpotent_isSemisimple
  /-
    case intro.intro.intro.intro.intro.intro
    K : Type u_2
    L : Type u_3
    inst✝⁷ : LieRing L
    inst✝⁶ : Field K
    inst✝⁵ : LieAlgebra K L
    inst✝⁴ : FiniteDimensional K L
    H : LieSubalgebra K L
    inst✝³ : H.IsCartanSubalgebra
    inst✝² : LieAlgebra.IsKilling K L
    inst✝¹ : LieModule.IsTriangularizable K (Subtype fun x => Membership.mem H x) L
    inst✝ : PerfectField K
    x : L
    hx : Membership.mem H x
    N S : Module.End K L
    hS₀ : Membership.mem (Algebra.adjoin K (Singleton.singleton ((LieAlgebra.ad K  …
    hN : _root_.IsNilpotent N
    hS : S.IsSemisimple
    hSN : Eq ((LieAlgebra.ad K L) x) (HAdd.hAdd N S)
    ⊢ ((LieAlgebra.ad K L) x).IsSemisimple
  -/
  replace hS₀ : Commute (ad K L x) S := Algebra.commute_of_mem_adjoin_self hS₀
  /-
    case intro.intro.intro.intro.intro.intro
    K : Type u_2
    L : Type u_3
    inst✝⁷ : LieRing L
    inst✝⁶ : Field K
    inst✝⁵ : LieAlgebra K L
    inst✝⁴ : FiniteDimensional K L
    H : LieSubalgebra K L
    inst✝³ : H.IsCartanSubalgebra
    inst✝² : LieAlgebra.IsKilling K L
    inst✝¹ : LieModule.IsTriangularizable K (Subtype fun x => Membership.mem H x) L
    inst✝ : PerfectField K
    x : L
    hx : Membership.mem H x
    N S : Module.End K L
    hN : _root_.IsNilpotent N
    hS : S.IsSemisimple
    hSN : Eq ((LieAlgebra.ad K L) x) (HAdd.hAdd N S)
    hS₀ : Commute ((LieAlgebra.ad K L) x) S
    ⊢ ((LieAlgebra.ad K L) x).IsSemisimple
  -/
  set x' : H := ⟨x, hx⟩
  /-
    case intro.intro.intro.intro.intro.intro
    K : Type u_2
    L : Type u_3
    inst✝⁷ : LieRing L
    inst✝⁶ : Field K
    inst✝⁵ : LieAlgebra K L
    inst✝⁴ : FiniteDimensional K L
    H : LieSubalgebra K L
    inst✝³ : H.IsCartanSubalgebra
    inst✝² : LieAlgebra.IsKilling K L
    inst✝¹ : LieModule.IsTriangularizable K (Subtype fun x => Membership.mem H x) L
    inst✝ : PerfectField K
    x : L
    hx : Membership.mem H x
    N S : Module.End K L
    hN : _root_.IsNilpotent N
    hS : S.IsSemisimple
    hSN : Eq ((LieAlgebra.ad K L) x) (HAdd.hAdd N S)
    hS₀ : Commute ((LieAlgebra.ad K L) x) S
    x' : Subtype fun x => Membership.mem H x := ⟨x, hx⟩
    ⊢ ((LieAlgebra.ad K L) x).IsSemisimple
  -/
  rw [eq_sub_of_add_eq hSN.symm] at hN
  /- Note that the semisimple part `S` is just a scalar action on each root space. -/
  have aux {α : H → K} {y : L} (hy : y ∈ rootSpace H α) : S y = α x' • y := by
    replace hy : y ∈ (ad K L x).maxGenEigenspace (α x') :=
      (genWeightSpace_le_genWeightSpaceOf L x' α) hy
    rw [maxGenEigenspace_eq] at hy
    set k := maxGenEigenspaceIndex (ad K L x) (α x')
    rw [apply_eq_of_mem_of_comm_of_isFinitelySemisimple_of_isNil hy hS₀ hS.isFinitelySemisimple hN]
  /- So `S` obeys the derivation axiom if we restrict to root spaces. -/
  have h_der (y z : L) (α β : H → K) (hy : y ∈ rootSpace H α) (hz : z ∈ rootSpace H β) :
      S ⁅y, z⁆ = ⁅S y, z⁆ + ⁅y, S z⁆ := by
    have hyz : ⁅y, z⁆ ∈ rootSpace H (α + β) :=
      mapsTo_toEnd_genWeightSpace_add_of_mem_rootSpace K L H L α β hy hz
    rw [aux hy, aux hz, aux hyz, smul_lie, lie_smul, ← add_smul, ← Pi.add_apply]
  /- Thus `S` is a derivation since root spaces span. -/
  replace h_der (y z : L) : S ⁅y, z⁆ = ⁅S y, z⁆ + ⁅y, S z⁆ := by
    have hy : y ∈ ⨆ α : H → K, rootSpace H α := by simp [iSup_genWeightSpace_eq_top]
    have hz : z ∈ ⨆ α : H → K, rootSpace H α := by simp [iSup_genWeightSpace_eq_top]
    induction hy using LieSubmodule.iSup_induction' with
    | hN α y hy =>
      induction hz using LieSubmodule.iSup_induction' with
      | hN β z hz => exact h_der y z α β hy hz
      | h0 => simp
      | hadd _ _ _ _ h h' => simp only [lie_add, map_add, h, h']; abel
    | h0 => simp
    | hadd _ _ _ _ h h' => simp only [add_lie, map_add, h, h']; abel
  /- An equivalent form of the derivation axiom used in `LieDerivation`. -/
  replace h_der : ∀ y z : L, S ⁅y, z⁆ = ⁅y, S z⁆ - ⁅z, S y⁆ := by
    simp_rw [← lie_skew (S _) _, add_comm, ← sub_eq_add_neg] at h_der; assumption
  /- Bundle `S` as a `LieDerivation`. -/
  /-
    case intro.intro.intro.intro.intro.intro
    K : Type u_2
    L : Type u_3
    inst✝⁷ : LieRing L
    inst✝⁶ : Field K
    inst✝⁵ : LieAlgebra K L
    inst✝⁴ : FiniteDimensional K L
    H : LieSubalgebra K L
    inst✝³ : H.IsCartanSubalgebra
    inst✝² : LieAlgebra.IsKilling K L
    inst✝¹ : LieModule.IsTriangularizable K (Subtype fun x => Membership.mem H x) L
    inst✝ : PerfectField K
    x : L
    hx : Membership.mem H x
    N S : Module.End K L
    hN : _root_.IsNilpotent (HSub.hSub ((LieAlgebra.ad K L) x) S)
    hS : S.IsSemisimple
    hSN : Eq ((LieAlgebra.ad K L) x) (HAdd.hAdd N S)
    hS₀ : Commute ((LieAlgebra.ad K L) x) S
    x' : Subtype fun x => Membership.mem H x := ⟨x, hx⟩
    aux : ∀ {α : (Subtype fun x => Membership.mem H x) → K} {y : L}, Membership.me …
    h_der : ∀ (y z : L), Eq (S (Bracket.bracket y z)) (HSub.hSub (Bracket.bracket  …
    ⊢ ((LieAlgebra.ad K L) x).IsSemisimple
  -/
  let S' : LieDerivation K L L := ⟨S, h_der⟩
  /- Since `L` has non-degenerate Killing form, `S` must be inner, corresponding to some `y : L`. -/
  /-
    case intro.intro.intro.intro.intro.intro
    K : Type u_2
    L : Type u_3
    inst✝⁷ : LieRing L
    inst✝⁶ : Field K
    inst✝⁵ : LieAlgebra K L
    inst✝⁴ : FiniteDimensional K L
    H : LieSubalgebra K L
    inst✝³ : H.IsCartanSubalgebra
    inst✝² : LieAlgebra.IsKilling K L
    inst✝¹ : LieModule.IsTriangularizable K (Subtype fun x => Membership.mem H x) L
    inst✝ : PerfectField K
    x : L
    hx : Membership.mem H x
    N S : Module.End K L
    hN : _root_.IsNilpotent (HSub.hSub ((LieAlgebra.ad K L) x) S)
    hS : S.IsSemisimple
    hSN : Eq ((LieAlgebra.ad K L) x) (HAdd.hAdd N S)
    hS₀ : Commute ((LieAlgebra.ad K L) x) S
    x' : Subtype fun x => Membership.mem H x := ⟨x, hx⟩
    aux : ∀ {α : (Subtype fun x => Membership.mem H x) → K} {y : L}, Membership.me …
    h_der : ∀ (y z : L), Eq (S (Bracket.bracket y z)) (HSub.hSub (Bracket.bracket  …
    S' : LieDerivation K L L := { toLinearMap := S, leibniz' := h_der }
    ⊢ ((LieAlgebra.ad K L) x).IsSemisimple
  -/
  obtain ⟨y, hy⟩ := LieDerivation.IsKilling.exists_eq_ad S'
  /- `y` commutes with all elements of `H` because `S` has eigenvalue 0 on `H`, `S = ad K L y`. -/
  have hy' (z : L) (hz : z ∈ H) : ⁅y, z⁆ = 0 := by
    rw [← LieSubalgebra.mem_toLieSubmodule, ← rootSpace_zero_eq] at hz
    simp [S', ← ad_apply (R := K), ← LieDerivation.coe_ad_apply_eq_ad_apply, hy, aux hz]
  /- Thus `y` belongs to `H` since `H` is self-normalizing. -/
  replace hy' : y ∈ H := by
    suffices y ∈ H.normalizer by rwa [LieSubalgebra.IsCartanSubalgebra.self_normalizing] at this
    exact (H.mem_normalizer_iff y).mpr fun z hz ↦ hy' z hz ▸ LieSubalgebra.zero_mem H
  /- It suffices to show `x = y` since `S = ad K L y` is semisimple. -/
  /-
    case intro.intro.intro.intro.intro.intro.intro
    K : Type u_2
    L : Type u_3
    inst✝⁷ : LieRing L
    inst✝⁶ : Field K
    inst✝⁵ : LieAlgebra K L
    inst✝⁴ : FiniteDimensional K L
    H : LieSubalgebra K L
    inst✝³ : H.IsCartanSubalgebra
    inst✝² : LieAlgebra.IsKilling K L
    inst✝¹ : LieModule.IsTriangularizable K (Subtype fun x => Membership.mem H x) L
    inst✝ : PerfectField K
    x : L
    hx : Membership.mem H x
    N S : Module.End K L
    hN : _root_.IsNilpotent (HSub.hSub ((LieAlgebra.ad K L) x) S)
    hS : S.IsSemisimple
    hSN : Eq ((LieAlgebra.ad K L) x) (HAdd.hAdd N S)
    hS₀ : Commute ((LieAlgebra.ad K L) x) S
    x' : Subtype fun x => Membership.mem H x := ⟨x, hx⟩
    aux : ∀ {α : (Subtype fun x => Membership.mem H x) → K} {y : L}, Membership.me …
    h_der : ∀ (y z : L), Eq (S (Bracket.bracket y z)) (HSub.hSub (Bracket.bracket  …
    S' : LieDerivation K L L := { toLinearMap := S, leibniz' := h_der }
    y : L
    hy : Eq ((LieDerivation.ad K L) y) S'
    hy' : Membership.mem H y
    ⊢ ((LieAlgebra.ad K L) x).IsSemisimple
  -/
  suffices x = y by rwa [this, ← LieDerivation.coe_ad_apply_eq_ad_apply y, hy]
  /-
    case intro.intro.intro.intro.intro.intro.intro
    K : Type u_2
    L : Type u_3
    inst✝⁷ : LieRing L
    inst✝⁶ : Field K
    inst✝⁵ : LieAlgebra K L
    inst✝⁴ : FiniteDimensional K L
    H : LieSubalgebra K L
    inst✝³ : H.IsCartanSubalgebra
    inst✝² : LieAlgebra.IsKilling K L
    inst✝¹ : LieModule.IsTriangularizable K (Subtype fun x => Membership.mem H x) L
    inst✝ : PerfectField K
    x : L
    hx : Membership.mem H x
    N S : Module.End K L
    hN : _root_.IsNilpotent (HSub.hSub ((LieAlgebra.ad K L) x) S)
    hS : S.IsSemisimple
    hSN : Eq ((LieAlgebra.ad K L) x) (HAdd.hAdd N S)
    hS₀ : Commute ((LieAlgebra.ad K L) x) S
    x' : Subtype fun x => Membership.mem H x := ⟨x, hx⟩
    aux : ∀ {α : (Subtype fun x => Membership.mem H x) → K} {y : L}, Membership.me …
    h_der : ∀ (y z : L), Eq (S (Bracket.bracket y z)) (HSub.hSub (Bracket.bracket  …
    S' : LieDerivation K L L := { toLinearMap := S, leibniz' := h_der }
    y : L
    hy : Eq ((LieDerivation.ad K L) y) S'
    hy' : Membership.mem H y
    ⊢ Eq x y
  -/
  rw [← sub_eq_zero]
  /- This will follow if we can show that `ad K L (x - y)` is nilpotent. -/
  /-
    case intro.intro.intro.intro.intro.intro.intro
    K : Type u_2
    L : Type u_3
    inst✝⁷ : LieRing L
    inst✝⁶ : Field K
    inst✝⁵ : LieAlgebra K L
    inst✝⁴ : FiniteDimensional K L
    H : LieSubalgebra K L
    inst✝³ : H.IsCartanSubalgebra
    inst✝² : LieAlgebra.IsKilling K L
    inst✝¹ : LieModule.IsTriangularizable K (Subtype fun x => Membership.mem H x) L
    inst✝ : PerfectField K
    x : L
    hx : Membership.mem H x
    N S : Module.End K L
    hN : _root_.IsNilpotent (HSub.hSub ((LieAlgebra.ad K L) x) S)
    hS : S.IsSemisimple
    hSN : Eq ((LieAlgebra.ad K L) x) (HAdd.hAdd N S)
    hS₀ : Commute ((LieAlgebra.ad K L) x) S
    x' : Subtype fun x => Membership.mem H x := ⟨x, hx⟩
    aux : ∀ {α : (Subtype fun x => Membership.mem H x) → K} {y : L}, Membership.me …
    h_der : ∀ (y z : L), Eq (S (Bracket.bracket y z)) (HSub.hSub (Bracket.bracket  …
    S' : LieDerivation K L L := { toLinearMap := S, leibniz' := h_der }
    y : L
    hy : Eq ((LieDerivation.ad K L) y) S'
    hy' : Membership.mem H y
    ⊢ Eq (HSub.hSub x y) 0
  -/
  apply eq_zero_of_isNilpotent_ad_of_mem_isCartanSubalgebra K L H (H.sub_mem hx hy')
  /- Which is true because `ad K L (x - y) = N`. -/
  /-
    case intro.intro.intro.intro.intro.intro.intro
    K : Type u_2
    L : Type u_3
    inst✝⁷ : LieRing L
    inst✝⁶ : Field K
    inst✝⁵ : LieAlgebra K L
    inst✝⁴ : FiniteDimensional K L
    H : LieSubalgebra K L
    inst✝³ : H.IsCartanSubalgebra
    inst✝² : LieAlgebra.IsKilling K L
    inst✝¹ : LieModule.IsTriangularizable K (Subtype fun x => Membership.mem H x) L
    inst✝ : PerfectField K
    x : L
    hx : Membership.mem H x
    N S : Module.End K L
    hN : _root_.IsNilpotent (HSub.hSub ((LieAlgebra.ad K L) x) S)
    hS : S.IsSemisimple
    hSN : Eq ((LieAlgebra.ad K L) x) (HAdd.hAdd N S)
    hS₀ : Commute ((LieAlgebra.ad K L) x) S
    x' : Subtype fun x => Membership.mem H x := ⟨x, hx⟩
    aux : ∀ {α : (Subtype fun x => Membership.mem H x) → K} {y : L}, Membership.me …
    h_der : ∀ (y z : L), Eq (S (Bracket.bracket y z)) (HSub.hSub (Bracket.bracket  …
    S' : LieDerivation K L L := { toLinearMap := S, leibniz' := h_der }
    y : L
    hy : Eq ((LieDerivation.ad K L) y) S'
    hy' : Membership.mem H y
    ⊢ _root_.IsNilpotent ((LieAlgebra.ad K L) (HSub.hSub x y))
  -/
  replace hy : S = ad K L y := by rw [← LieDerivation.coe_ad_apply_eq_ad_apply y, hy]
  /-
    case intro.intro.intro.intro.intro.intro.intro
    K : Type u_2
    L : Type u_3
    inst✝⁷ : LieRing L
    inst✝⁶ : Field K
    inst✝⁵ : LieAlgebra K L
    inst✝⁴ : FiniteDimensional K L
    H : LieSubalgebra K L
    inst✝³ : H.IsCartanSubalgebra
    inst✝² : LieAlgebra.IsKilling K L
    inst✝¹ : LieModule.IsTriangularizable K (Subtype fun x => Membership.mem H x) L
    inst✝ : PerfectField K
    x : L
    hx : Membership.mem H x
    N S : Module.End K L
    hN : _root_.IsNilpotent (HSub.hSub ((LieAlgebra.ad K L) x) S)
    hS : S.IsSemisimple
    hSN : Eq ((LieAlgebra.ad K L) x) (HAdd.hAdd N S)
    hS₀ : Commute ((LieAlgebra.ad K L) x) S
    x' : Subtype fun x => Membership.mem H x := ⟨x, hx⟩
    aux : ∀ {α : (Subtype fun x => Membership.mem H x) → K} {y : L}, Membership.me …
    h_der : ∀ (y z : L), Eq (S (Bracket.bracket y z)) (HSub.hSub (Bracket.bracket  …
    S' : LieDerivation K L L := { toLinearMap := S, leibniz' := h_der }
    y : L
    hy' : Membership.mem H y
    hy : Eq S ((LieAlgebra.ad K L) y)
    ⊢ _root_.IsNilpotent ((LieAlgebra.ad K L) (HSub.hSub x y))
  -/
  rwa [LieHom.map_sub, hSN, hy, add_sub_cancel_right, eq_sub_of_add_eq hSN.symm]
  /-
    🎉 no goals
  -/


lemma lie_eq_smul_of_mem_rootSpace {α : H → K} {x : L} (hx : x ∈ rootSpace H α) (h : H) :
    ⁅h, x⁆ = α h • x := by
  replace hx : x ∈ (ad K L h).maxGenEigenspace (α h) :=
    genWeightSpace_le_genWeightSpaceOf L h α hx
  rw [(isSemisimple_ad_of_mem_isCartanSubalgebra
    h.property).isFinitelySemisimple.maxGenEigenspace_eq_eigenspace,
    Module.End.mem_eigenspace_iff] at hx
  /-
    K : Type u_2
    L : Type u_3
    inst✝⁷ : LieRing L
    inst✝⁶ : Field K
    inst✝⁵ : LieAlgebra K L
    inst✝⁴ : FiniteDimensional K L
    H : LieSubalgebra K L
    inst✝³ : H.IsCartanSubalgebra
    inst✝² : LieAlgebra.IsKilling K L
    inst✝¹ : LieModule.IsTriangularizable K (Subtype fun x => Membership.mem H x) L
    inst✝ : PerfectField K
    α : (Subtype fun x => Membership.mem H x) → K
    x : L
    h : Subtype fun x => Membership.mem H x
    hx : Eq (((LieAlgebra.ad K L) ↑h) x) (HSMul.hSMul (α h) x)
    ⊢ Eq (Bracket.bracket h x) (HSMul.hSMul (α h) x)
  -/
  simpa using hx
  /-
    🎉 no goals
  -/


lemma lie_eq_killingForm_smul_of_mem_rootSpace_of_mem_rootSpace_neg
    {α : Weight K H L} {e f : L} (heα : e ∈ rootSpace H α) (hfα : f ∈ rootSpace H (-α)) :
    ⁅e, f⁆ = killingForm K L e f • (cartanEquivDual H).symm α := by
  /-
    K : Type u_2
    L : Type u_3
    inst✝⁷ : LieRing L
    inst✝⁶ : Field K
    inst✝⁵ : LieAlgebra K L
    inst✝⁴ : FiniteDimensional K L
    H : LieSubalgebra K L
    inst✝³ : H.IsCartanSubalgebra
    inst✝² : LieAlgebra.IsKilling K L
    inst✝¹ : LieModule.IsTriangularizable K (Subtype fun x => Membership.mem H x) L
    inst✝ : PerfectField K
    α : LieModule.Weight K (Subtype fun x => Membership.mem H x) L
    e f : L
    heα : Membership.mem (LieAlgebra.rootSpace H ⇑α) e
    hfα : Membership.mem (LieAlgebra.rootSpace H (Neg.neg ⇑α)) f
    ⊢ Eq (Bracket.bracket e f) (HSMul.hSMul (((killingForm K L) e) f) ↑((LieAlgebr …
  -/
  apply lie_eq_killingForm_smul_of_mem_rootSpace_of_mem_rootSpace_neg_aux heα hfα
  /-
    K : Type u_2
    L : Type u_3
    inst✝⁷ : LieRing L
    inst✝⁶ : Field K
    inst✝⁵ : LieAlgebra K L
    inst✝⁴ : FiniteDimensional K L
    H : LieSubalgebra K L
    inst✝³ : H.IsCartanSubalgebra
    inst✝² : LieAlgebra.IsKilling K L
    inst✝¹ : LieModule.IsTriangularizable K (Subtype fun x => Membership.mem H x) L
    inst✝ : PerfectField K
    α : LieModule.Weight K (Subtype fun x => Membership.mem H x) L
    e f : L
    heα : Membership.mem (LieAlgebra.rootSpace H ⇑α) e
    hfα : Membership.mem (LieAlgebra.rootSpace H (Neg.neg ⇑α)) f
    ⊢ ∀ (h : Subtype fun x => Membership.mem H x), Eq (Bracket.bracket h e) (HSMul …
  -/
  exact lie_eq_smul_of_mem_rootSpace heα
  /-
    🎉 no goals
  -/


lemma coe_corootSpace_eq_span_singleton' (α : Weight K H L) :
    (corootSpace α).toSubmodule = K ∙ (cartanEquivDual H).symm α := by
  /-
    K : Type u_2
    L : Type u_3
    inst✝⁷ : LieRing L
    inst✝⁶ : Field K
    inst✝⁵ : LieAlgebra K L
    inst✝⁴ : FiniteDimensional K L
    H : LieSubalgebra K L
    inst✝³ : H.IsCartanSubalgebra
    inst✝² : LieAlgebra.IsKilling K L
    inst✝¹ : LieModule.IsTriangularizable K (Subtype fun x => Membership.mem H x) L
    inst✝ : PerfectField K
    α : LieModule.Weight K (Subtype fun x => Membership.mem H x) L
    ⊢ Eq (↑(LieAlgebra.corootSpace ⇑α)) (Submodule.span K (Singleton.singleton ((L …
  -/
  refine le_antisymm ?_ ?_
    /-
      case refine_1
      K : Type u_2
      L : Type u_3
      inst✝⁷ : LieRing L
      inst✝⁶ : Field K
      inst✝⁵ : LieAlgebra K L
      inst✝⁴ : FiniteDimensional K L
      H : LieSubalgebra K L
      inst✝³ : H.IsCartanSubalgebra
      inst✝² : LieAlgebra.IsKilling K L
      inst✝¹ : LieModule.IsTriangularizable K (Subtype fun x => Membership.mem H x) L
      inst✝ : PerfectField K
      α : LieModule.Weight K (Subtype fun x => Membership.mem H x) L
      ⊢ LE.le (↑(LieAlgebra.corootSpace ⇑α)) (Submodule.span K (Singleton.singleton  …
    -/
  · intro ⟨x, hx⟩ hx'
    have : {⁅y, z⁆ | (y ∈ rootSpace H α) (z ∈ rootSpace H (-α))} ⊆
        K ∙ ((cartanEquivDual H).symm α : L) := by
      rintro - ⟨e, heα, f, hfα, rfl⟩
      rw [lie_eq_killingForm_smul_of_mem_rootSpace_of_mem_rootSpace_neg heα hfα, SetLike.mem_coe,
        Submodule.mem_span_singleton]
      exact ⟨killingForm K L e f, rfl⟩
    /-
      case refine_1
      K : Type u_2
      L : Type u_3
      inst✝⁷ : LieRing L
      inst✝⁶ : Field K
      inst✝⁵ : LieAlgebra K L
      inst✝⁴ : FiniteDimensional K L
      H : LieSubalgebra K L
      inst✝³ : H.IsCartanSubalgebra
      inst✝² : LieAlgebra.IsKilling K L
      inst✝¹ : LieModule.IsTriangularizable K (Subtype fun x => Membership.mem H x) L
      inst✝ : PerfectField K
      α : LieModule.Weight K (Subtype fun x => Membership.mem H x) L
      x : L
      hx : Membership.mem H x
      hx' : Membership.mem ↑(LieAlgebra.corootSpace ⇑α) ⟨x, hx⟩
      this : HasSubset.Subset (setOf fun x => Exists fun y => And (Membership.mem (L …
      ⊢ Membership.mem (Submodule.span K (Singleton.singleton ((LieAlgebra.IsKilling …
    -/
    simp only [LieSubmodule.mem_toSubmodule, mem_corootSpace] at hx'
    /-
      case refine_1
      K : Type u_2
      L : Type u_3
      inst✝⁷ : LieRing L
      inst✝⁶ : Field K
      inst✝⁵ : LieAlgebra K L
      inst✝⁴ : FiniteDimensional K L
      H : LieSubalgebra K L
      inst✝³ : H.IsCartanSubalgebra
      inst✝² : LieAlgebra.IsKilling K L
      inst✝¹ : LieModule.IsTriangularizable K (Subtype fun x => Membership.mem H x) L
      inst✝ : PerfectField K
      α : LieModule.Weight K (Subtype fun x => Membership.mem H x) L
      x : L
      hx : Membership.mem H x
      this : HasSubset.Subset (setOf fun x => Exists fun y => And (Membership.mem (L …
      hx' : Membership.mem (Submodule.span K (setOf fun x => Exists fun y => And (Me …
      ⊢ Membership.mem (Submodule.span K (Singleton.singleton ((LieAlgebra.IsKilling …
    -/
    replace this := Submodule.span_mono this hx'
    /-
      case refine_1
      K : Type u_2
      L : Type u_3
      inst✝⁷ : LieRing L
      inst✝⁶ : Field K
      inst✝⁵ : LieAlgebra K L
      inst✝⁴ : FiniteDimensional K L
      H : LieSubalgebra K L
      inst✝³ : H.IsCartanSubalgebra
      inst✝² : LieAlgebra.IsKilling K L
      inst✝¹ : LieModule.IsTriangularizable K (Subtype fun x => Membership.mem H x) L
      inst✝ : PerfectField K
      α : LieModule.Weight K (Subtype fun x => Membership.mem H x) L
      x : L
      hx : Membership.mem H x
      hx' : Membership.mem (Submodule.span K (setOf fun x => Exists fun y => And (Me …
      this : Membership.mem (Submodule.span K ↑(Submodule.span K (Singleton.singleto …
      ⊢ Membership.mem (Submodule.span K (Singleton.singleton ((LieAlgebra.IsKilling …
    -/
    rw [Submodule.span_span] at this
    /-
      case refine_1
      K : Type u_2
      L : Type u_3
      inst✝⁷ : LieRing L
      inst✝⁶ : Field K
      inst✝⁵ : LieAlgebra K L
      inst✝⁴ : FiniteDimensional K L
      H : LieSubalgebra K L
      inst✝³ : H.IsCartanSubalgebra
      inst✝² : LieAlgebra.IsKilling K L
      inst✝¹ : LieModule.IsTriangularizable K (Subtype fun x => Membership.mem H x) L
      inst✝ : PerfectField K
      α : LieModule.Weight K (Subtype fun x => Membership.mem H x) L
      x : L
      hx : Membership.mem H x
      hx' : Membership.mem (Submodule.span K (setOf fun x => Exists fun y => And (Me …
      this : Membership.mem (Submodule.span K (Singleton.singleton ↑((LieAlgebra.IsK …
      ⊢ Membership.mem (Submodule.span K (Singleton.singleton ((LieAlgebra.IsKilling …
    -/
    rw [Submodule.mem_span_singleton] at this ⊢
    /-
      case refine_1
      K : Type u_2
      L : Type u_3
      inst✝⁷ : LieRing L
      inst✝⁶ : Field K
      inst✝⁵ : LieAlgebra K L
      inst✝⁴ : FiniteDimensional K L
      H : LieSubalgebra K L
      inst✝³ : H.IsCartanSubalgebra
      inst✝² : LieAlgebra.IsKilling K L
      inst✝¹ : LieModule.IsTriangularizable K (Subtype fun x => Membership.mem H x) L
      inst✝ : PerfectField K
      α : LieModule.Weight K (Subtype fun x => Membership.mem H x) L
      x : L
      hx : Membership.mem H x
      hx' : Membership.mem (Submodule.span K (setOf fun x => Exists fun y => And (Me …
      this : Exists fun a => Eq (HSMul.hSMul a ↑((LieAlgebra.IsKilling.cartanEquivDu …
      ⊢ Exists fun a => Eq (HSMul.hSMul a ((LieAlgebra.IsKilling.cartanEquivDual H). …
    -/
    obtain ⟨t, rfl⟩ := this
    /-
      case refine_1.intro
      K : Type u_2
      L : Type u_3
      inst✝⁷ : LieRing L
      inst✝⁶ : Field K
      inst✝⁵ : LieAlgebra K L
      inst✝⁴ : FiniteDimensional K L
      H : LieSubalgebra K L
      inst✝³ : H.IsCartanSubalgebra
      inst✝² : LieAlgebra.IsKilling K L
      inst✝¹ : LieModule.IsTriangularizable K (Subtype fun x => Membership.mem H x) L
      inst✝ : PerfectField K
      α : LieModule.Weight K (Subtype fun x => Membership.mem H x) L
      t : K
      hx : Membership.mem H (HSMul.hSMul t ↑((LieAlgebra.IsKilling.cartanEquivDual H …
      hx' : Membership.mem (Submodule.span K (setOf fun x => Exists fun y => And (Me …
      ⊢ Exists fun a => Eq (HSMul.hSMul a ((LieAlgebra.IsKilling.cartanEquivDual H). …
    -/
    use t
    /-
      case h
      K : Type u_2
      L : Type u_3
      inst✝⁷ : LieRing L
      inst✝⁶ : Field K
      inst✝⁵ : LieAlgebra K L
      inst✝⁴ : FiniteDimensional K L
      H : LieSubalgebra K L
      inst✝³ : H.IsCartanSubalgebra
      inst✝² : LieAlgebra.IsKilling K L
      inst✝¹ : LieModule.IsTriangularizable K (Subtype fun x => Membership.mem H x) L
      inst✝ : PerfectField K
      α : LieModule.Weight K (Subtype fun x => Membership.mem H x) L
      t : K
      hx : Membership.mem H (HSMul.hSMul t ↑((LieAlgebra.IsKilling.cartanEquivDual H …
      hx' : Membership.mem (Submodule.span K (setOf fun x => Exists fun y => And (Me …
      ⊢ Eq (HSMul.hSMul t ((LieAlgebra.IsKilling.cartanEquivDual H).symm (LieModule. …
    -/
    simp only [Subtype.ext_iff]
    /-
      case h
      K : Type u_2
      L : Type u_3
      inst✝⁷ : LieRing L
      inst✝⁶ : Field K
      inst✝⁵ : LieAlgebra K L
      inst✝⁴ : FiniteDimensional K L
      H : LieSubalgebra K L
      inst✝³ : H.IsCartanSubalgebra
      inst✝² : LieAlgebra.IsKilling K L
      inst✝¹ : LieModule.IsTriangularizable K (Subtype fun x => Membership.mem H x) L
      inst✝ : PerfectField K
      α : LieModule.Weight K (Subtype fun x => Membership.mem H x) L
      t : K
      hx : Membership.mem H (HSMul.hSMul t ↑((LieAlgebra.IsKilling.cartanEquivDual H …
      hx' : Membership.mem (Submodule.span K (setOf fun x => Exists fun y => And (Me …
      ⊢ Eq (↑(HSMul.hSMul t ((LieAlgebra.IsKilling.cartanEquivDual H).symm (LieModul …
    -/
    rw [Submodule.coe_smul_of_tower]
    /-
      🎉 no goals
    -/
    /-
      case refine_2
      K : Type u_2
      L : Type u_3
      inst✝⁷ : LieRing L
      inst✝⁶ : Field K
      inst✝⁵ : LieAlgebra K L
      inst✝⁴ : FiniteDimensional K L
      H : LieSubalgebra K L
      inst✝³ : H.IsCartanSubalgebra
      inst✝² : LieAlgebra.IsKilling K L
      inst✝¹ : LieModule.IsTriangularizable K (Subtype fun x => Membership.mem H x) L
      inst✝ : PerfectField K
      α : LieModule.Weight K (Subtype fun x => Membership.mem H x) L
      ⊢ LE.le (Submodule.span K (Singleton.singleton ((LieAlgebra.IsKilling.cartanEq …
    -/
  · simp only [Submodule.span_singleton_le_iff_mem, LieSubmodule.mem_toSubmodule]
    /-
      case refine_2
      K : Type u_2
      L : Type u_3
      inst✝⁷ : LieRing L
      inst✝⁶ : Field K
      inst✝⁵ : LieAlgebra K L
      inst✝⁴ : FiniteDimensional K L
      H : LieSubalgebra K L
      inst✝³ : H.IsCartanSubalgebra
      inst✝² : LieAlgebra.IsKilling K L
      inst✝¹ : LieModule.IsTriangularizable K (Subtype fun x => Membership.mem H x) L
      inst✝ : PerfectField K
      α : LieModule.Weight K (Subtype fun x => Membership.mem H x) L
      ⊢ Membership.mem (LieAlgebra.corootSpace ⇑α) ((LieAlgebra.IsKilling.cartanEqui …
    -/
    exact cartanEquivDual_symm_apply_mem_corootSpace α
    /-
      🎉 no goals
    -/


/-- The contrapositive of this result is very useful, taking `x` to be the element of `H`
corresponding to a root `α` under the identification between `H` and `H^*` provided by the Killing
form. -/
lemma eq_zero_of_apply_eq_zero_of_mem_corootSpace
    (x : H) (α : H → K) (hαx : α x = 0) (hx : x ∈ corootSpace α) :
    x = 0 := by
  /-
    K : Type u_2
    L : Type u_3
    inst✝⁷ : LieRing L
    inst✝⁶ : Field K
    inst✝⁵ : LieAlgebra K L
    inst✝⁴ : FiniteDimensional K L
    H : LieSubalgebra K L
    inst✝³ : H.IsCartanSubalgebra
    inst✝² : LieAlgebra.IsKilling K L
    inst✝¹ : LieModule.IsTriangularizable K (Subtype fun x => Membership.mem H x) L
    inst✝ : CharZero K
    x : Subtype fun x => Membership.mem H x
    α : (Subtype fun x => Membership.mem H x) → K
    hαx : Eq (α x) 0
    hx : Membership.mem (LieAlgebra.corootSpace α) x
    ⊢ Eq x 0
  -/
  rcases eq_or_ne α 0 with rfl | hα; · simpa using hx
                                       /-
                                         🎉 no goals
                                       -/
  replace hx : x ∈ ⨅ β : Weight K H L, β.ker := by
    refine (Submodule.mem_iInf _).mpr fun β ↦ ?_
    obtain ⟨a, b, hb, hab⟩ :=
      exists_forall_mem_corootSpace_smul_add_eq_zero L α β hα β.genWeightSpace_ne_bot
    simpa [hαx, hb.ne'] using hab _ hx
  /-
    case inr
    K : Type u_2
    L : Type u_3
    inst✝⁷ : LieRing L
    inst✝⁶ : Field K
    inst✝⁵ : LieAlgebra K L
    inst✝⁴ : FiniteDimensional K L
    H : LieSubalgebra K L
    inst✝³ : H.IsCartanSubalgebra
    inst✝² : LieAlgebra.IsKilling K L
    inst✝¹ : LieModule.IsTriangularizable K (Subtype fun x => Membership.mem H x) L
    inst✝ : CharZero K
    x : Subtype fun x => Membership.mem H x
    α : (Subtype fun x => Membership.mem H x) → K
    hαx : Eq (α x) 0
    hα : Ne α 0
    hx : Membership.mem (iInf fun β => LieModule.Weight.ker) x
    ⊢ Eq x 0
  -/
  simpa using hx
  /-
    🎉 no goals
  -/


lemma disjoint_ker_weight_corootSpace (α : Weight K H L) :
    Disjoint α.ker (corootSpace α) := by
  /-
    K : Type u_2
    L : Type u_3
    inst✝⁷ : LieRing L
    inst✝⁶ : Field K
    inst✝⁵ : LieAlgebra K L
    inst✝⁴ : FiniteDimensional K L
    H : LieSubalgebra K L
    inst✝³ : H.IsCartanSubalgebra
    inst✝² : LieAlgebra.IsKilling K L
    inst✝¹ : LieModule.IsTriangularizable K (Subtype fun x => Membership.mem H x) L
    inst✝ : CharZero K
    α : LieModule.Weight K (Subtype fun x => Membership.mem H x) L
    ⊢ Disjoint LieModule.Weight.ker (LieIdeal.toLieSubalgebra K (Subtype fun x =>  …
  -/
  rw [disjoint_iff]
  /-
    K : Type u_2
    L : Type u_3
    inst✝⁷ : LieRing L
    inst✝⁶ : Field K
    inst✝⁵ : LieAlgebra K L
    inst✝⁴ : FiniteDimensional K L
    H : LieSubalgebra K L
    inst✝³ : H.IsCartanSubalgebra
    inst✝² : LieAlgebra.IsKilling K L
    inst✝¹ : LieModule.IsTriangularizable K (Subtype fun x => Membership.mem H x) L
    inst✝ : CharZero K
    α : LieModule.Weight K (Subtype fun x => Membership.mem H x) L
    ⊢ Eq (Min.min LieModule.Weight.ker (LieIdeal.toLieSubalgebra K (Subtype fun x  …
  -/
  refine (Submodule.eq_bot_iff _).mpr fun x ⟨hαx, hx⟩ ↦ ?_
  /-
    K : Type u_2
    L : Type u_3
    inst✝⁷ : LieRing L
    inst✝⁶ : Field K
    inst✝⁵ : LieAlgebra K L
    inst✝⁴ : FiniteDimensional K L
    H : LieSubalgebra K L
    inst✝³ : H.IsCartanSubalgebra
    inst✝² : LieAlgebra.IsKilling K L
    inst✝¹ : LieModule.IsTriangularizable K (Subtype fun x => Membership.mem H x) L
    inst✝ : CharZero K
    α : LieModule.Weight K (Subtype fun x => Membership.mem H x) L
    x : Subtype fun x => Membership.mem H x
    x✝ : Membership.mem (Min.min LieModule.Weight.ker (LieIdeal.toLieSubalgebra K  …
    hαx : Membership.mem (↑LieModule.Weight.ker) x
    hx : Membership.mem (↑(LieIdeal.toLieSubalgebra K (Subtype fun x => Membership …
    ⊢ Eq x 0
  -/
  replace hαx : α x = 0 := by simpa using hαx
  /-
    K : Type u_2
    L : Type u_3
    inst✝⁷ : LieRing L
    inst✝⁶ : Field K
    inst✝⁵ : LieAlgebra K L
    inst✝⁴ : FiniteDimensional K L
    H : LieSubalgebra K L
    inst✝³ : H.IsCartanSubalgebra
    inst✝² : LieAlgebra.IsKilling K L
    inst✝¹ : LieModule.IsTriangularizable K (Subtype fun x => Membership.mem H x) L
    inst✝ : CharZero K
    α : LieModule.Weight K (Subtype fun x => Membership.mem H x) L
    x : Subtype fun x => Membership.mem H x
    x✝ : Membership.mem (Min.min LieModule.Weight.ker (LieIdeal.toLieSubalgebra K  …
    hx : Membership.mem (↑(LieIdeal.toLieSubalgebra K (Subtype fun x => Membership …
    hαx : Eq (α x) 0
    ⊢ Eq x 0
  -/
  exact eq_zero_of_apply_eq_zero_of_mem_corootSpace x α hαx hx
  /-
    🎉 no goals
  -/


lemma root_apply_cartanEquivDual_symm_ne_zero {α : Weight K H L} (hα : α.IsNonZero) :
    α ((cartanEquivDual H).symm α) ≠ 0 := by
  /-
    K : Type u_2
    L : Type u_3
    inst✝⁷ : LieRing L
    inst✝⁶ : Field K
    inst✝⁵ : LieAlgebra K L
    inst✝⁴ : FiniteDimensional K L
    H : LieSubalgebra K L
    inst✝³ : H.IsCartanSubalgebra
    inst✝² : LieAlgebra.IsKilling K L
    inst✝¹ : LieModule.IsTriangularizable K (Subtype fun x => Membership.mem H x) L
    inst✝ : CharZero K
    α : LieModule.Weight K (Subtype fun x => Membership.mem H x) L
    hα : α.IsNonZero
    ⊢ Ne (α ((LieAlgebra.IsKilling.cartanEquivDual H).symm (LieModule.Weight.toLin …
  -/
  contrapose! hα
  suffices (cartanEquivDual H).symm α ∈ α.ker ⊓ corootSpace α by
    rw [(disjoint_ker_weight_corootSpace α).eq_bot] at this
    simpa using this
  /-
    K : Type u_2
    L : Type u_3
    inst✝⁷ : LieRing L
    inst✝⁶ : Field K
    inst✝⁵ : LieAlgebra K L
    inst✝⁴ : FiniteDimensional K L
    H : LieSubalgebra K L
    inst✝³ : H.IsCartanSubalgebra
    inst✝² : LieAlgebra.IsKilling K L
    inst✝¹ : LieModule.IsTriangularizable K (Subtype fun x => Membership.mem H x) L
    inst✝ : CharZero K
    α : LieModule.Weight K (Subtype fun x => Membership.mem H x) L
    hα : Eq (α ((LieAlgebra.IsKilling.cartanEquivDual H).symm (LieModule.Weight.to …
    ⊢ Membership.mem (Min.min LieModule.Weight.ker (LieIdeal.toLieSubalgebra K (Su …
  -/
  exact Submodule.mem_inf.mp ⟨hα, cartanEquivDual_symm_apply_mem_corootSpace α⟩
  /-
    🎉 no goals
  -/


lemma root_apply_coroot {α : Weight K H L} (hα : α.IsNonZero) :
    α (coroot α) = 2 := by
  /-
    K : Type u_2
    L : Type u_3
    inst✝⁷ : LieRing L
    inst✝⁶ : Field K
    inst✝⁵ : LieAlgebra K L
    inst✝⁴ : FiniteDimensional K L
    H : LieSubalgebra K L
    inst✝³ : H.IsCartanSubalgebra
    inst✝² : LieAlgebra.IsKilling K L
    inst✝¹ : LieModule.IsTriangularizable K (Subtype fun x => Membership.mem H x) L
    inst✝ : CharZero K
    α : LieModule.Weight K (Subtype fun x => Membership.mem H x) L
    hα : α.IsNonZero
    ⊢ Eq (α (LieAlgebra.IsKilling.coroot α)) 2
  -/
  rw [← Weight.coe_coe]
  /-
    K : Type u_2
    L : Type u_3
    inst✝⁷ : LieRing L
    inst✝⁶ : Field K
    inst✝⁵ : LieAlgebra K L
    inst✝⁴ : FiniteDimensional K L
    H : LieSubalgebra K L
    inst✝³ : H.IsCartanSubalgebra
    inst✝² : LieAlgebra.IsKilling K L
    inst✝¹ : LieModule.IsTriangularizable K (Subtype fun x => Membership.mem H x) L
    inst✝ : CharZero K
    α : LieModule.Weight K (Subtype fun x => Membership.mem H x) L
    hα : α.IsNonZero
    ⊢ Eq ((LieModule.Weight.toLinear K (Subtype fun x => Membership.mem H x) L α)  …
  -/
  simpa [coroot] using inv_mul_cancel₀ (root_apply_cartanEquivDual_symm_ne_zero hα)
  /-
    🎉 no goals
  -/


@[simp] lemma coroot_eq_zero_iff {α : Weight K H L} :
    coroot α = 0 ↔ α.IsZero := by
  /-
    K : Type u_2
    L : Type u_3
    inst✝⁷ : LieRing L
    inst✝⁶ : Field K
    inst✝⁵ : LieAlgebra K L
    inst✝⁴ : FiniteDimensional K L
    H : LieSubalgebra K L
    inst✝³ : H.IsCartanSubalgebra
    inst✝² : LieAlgebra.IsKilling K L
    inst✝¹ : LieModule.IsTriangularizable K (Subtype fun x => Membership.mem H x) L
    inst✝ : CharZero K
    α : LieModule.Weight K (Subtype fun x => Membership.mem H x) L
    ⊢ Iff (Eq (LieAlgebra.IsKilling.coroot α) 0) α.IsZero
  -/
  refine ⟨fun hα ↦ ?_, fun hα ↦ ?_⟩
    /-
      case refine_1
      K : Type u_2
      L : Type u_3
      inst✝⁷ : LieRing L
      inst✝⁶ : Field K
      inst✝⁵ : LieAlgebra K L
      inst✝⁴ : FiniteDimensional K L
      H : LieSubalgebra K L
      inst✝³ : H.IsCartanSubalgebra
      inst✝² : LieAlgebra.IsKilling K L
      inst✝¹ : LieModule.IsTriangularizable K (Subtype fun x => Membership.mem H x) L
      inst✝ : CharZero K
      α : LieModule.Weight K (Subtype fun x => Membership.mem H x) L
      hα : Eq (LieAlgebra.IsKilling.coroot α) 0
      ⊢ α.IsZero
    -/
  · by_contra contra
    /-
      case refine_1
      K : Type u_2
      L : Type u_3
      inst✝⁷ : LieRing L
      inst✝⁶ : Field K
      inst✝⁵ : LieAlgebra K L
      inst✝⁴ : FiniteDimensional K L
      H : LieSubalgebra K L
      inst✝³ : H.IsCartanSubalgebra
      inst✝² : LieAlgebra.IsKilling K L
      inst✝¹ : LieModule.IsTriangularizable K (Subtype fun x => Membership.mem H x) L
      inst✝ : CharZero K
      α : LieModule.Weight K (Subtype fun x => Membership.mem H x) L
      hα : Eq (LieAlgebra.IsKilling.coroot α) 0
      contra : Not α.IsZero
      ⊢ False
    -/
    simpa [hα, ← α.coe_coe, map_zero] using root_apply_coroot contra
    /-
      🎉 no goals
    -/
    /-
      case refine_2
      K : Type u_2
      L : Type u_3
      inst✝⁷ : LieRing L
      inst✝⁶ : Field K
      inst✝⁵ : LieAlgebra K L
      inst✝⁴ : FiniteDimensional K L
      H : LieSubalgebra K L
      inst✝³ : H.IsCartanSubalgebra
      inst✝² : LieAlgebra.IsKilling K L
      inst✝¹ : LieModule.IsTriangularizable K (Subtype fun x => Membership.mem H x) L
      inst✝ : CharZero K
      α : LieModule.Weight K (Subtype fun x => Membership.mem H x) L
      hα : α.IsZero
      ⊢ Eq (LieAlgebra.IsKilling.coroot α) 0
    -/
  · simp [coroot, Weight.coe_toLinear_eq_zero_iff.mpr hα]
    /-
      🎉 no goals
    -/


@[simp]
                                                                       /-
                                                                         K : Type u_2
                                                                         L : Type u_3
                                                                         inst✝⁸ : LieRing L
                                                                         inst✝⁷ : Field K
                                                                         inst✝⁶ : LieAlgebra K L
                                                                         inst✝⁵ : FiniteDimensional K L
                                                                         H : LieSubalgebra K L
                                                                         inst✝⁴ : H.IsCartanSubalgebra
                                                                         inst✝³ : LieAlgebra.IsKilling K L
                                                                         inst✝² : LieModule.IsTriangularizable K (Subtype fun x => Membership.mem H x) L
                                                                         inst✝¹ : CharZero K
                                                                         inst✝ : Nontrivial L
                                                                         ⊢ Eq (LieAlgebra.IsKilling.coroot 0) 0
                                                                       -/
lemma coroot_zero [Nontrivial L] : coroot (0 : Weight K H L) = 0 := by simp [Weight.isZero_zero]
                                                                       /-
                                                                         🎉 no goals
                                                                       -/


lemma coe_corootSpace_eq_span_singleton (α : Weight K H L) :
    (corootSpace α).toSubmodule = K ∙ coroot α := by
  if hα : α.IsZero then
    simp [hα.eq, coroot_eq_zero_iff.mpr hα]
  else
    set α' := (cartanEquivDual H).symm α
    suffices (K ∙ coroot α) = K ∙ α' by rw [coe_corootSpace_eq_span_singleton']; exact this.symm
    have : IsUnit (2 * (α α')⁻¹) := by simpa using root_apply_cartanEquivDual_symm_ne_zero hα
    change (K ∙ (2 • (α α')⁻¹ • α')) = _
    simpa [← Nat.cast_smul_eq_nsmul K, smul_smul] using Submodule.span_singleton_smul_eq this _


@[simp]
lemma corootSpace_eq_bot_iff {α : Weight K H L} :
    corootSpace α = ⊥ ↔ α.IsZero := by
  /-
    K : Type u_2
    L : Type u_3
    inst✝⁷ : LieRing L
    inst✝⁶ : Field K
    inst✝⁵ : LieAlgebra K L
    inst✝⁴ : FiniteDimensional K L
    H : LieSubalgebra K L
    inst✝³ : H.IsCartanSubalgebra
    inst✝² : LieAlgebra.IsKilling K L
    inst✝¹ : LieModule.IsTriangularizable K (Subtype fun x => Membership.mem H x) L
    inst✝ : CharZero K
    α : LieModule.Weight K (Subtype fun x => Membership.mem H x) L
    ⊢ Iff (Eq (LieAlgebra.corootSpace ⇑α) Bot.bot) α.IsZero
  -/
  simp [← LieSubmodule.toSubmodule_eq_bot, coe_corootSpace_eq_span_singleton α]
  /-
    🎉 no goals
  -/


lemma isCompl_ker_weight_span_coroot (α : Weight K H L) :
    IsCompl α.ker (K ∙ coroot α) := by
  if hα : α.IsZero then
    simpa [Weight.coe_toLinear_eq_zero_iff.mpr hα, coroot_eq_zero_iff.mpr hα, Weight.ker]
      using isCompl_top_bot
  else
    rw [← coe_corootSpace_eq_span_singleton]
    apply Module.Dual.isCompl_ker_of_disjoint_of_ne_bot (by aesop)
      (disjoint_ker_weight_corootSpace α)
    replace hα : corootSpace α ≠ ⊥ := by simpa using hα
    rwa [ne_eq, ← LieSubmodule.toSubmodule_inj] at hα


lemma traceForm_eq_zero_of_mem_ker_of_mem_span_coroot {α : Weight K H L} {x y : H}
    (hx : x ∈ α.ker) (hy : y ∈ K ∙ coroot α) :
    traceForm K H L x y = 0 := by
  /-
    K : Type u_2
    L : Type u_3
    inst✝⁷ : LieRing L
    inst✝⁶ : Field K
    inst✝⁵ : LieAlgebra K L
    inst✝⁴ : FiniteDimensional K L
    H : LieSubalgebra K L
    inst✝³ : H.IsCartanSubalgebra
    inst✝² : LieAlgebra.IsKilling K L
    inst✝¹ : LieModule.IsTriangularizable K (Subtype fun x => Membership.mem H x) L
    inst✝ : CharZero K
    α : LieModule.Weight K (Subtype fun x => Membership.mem H x) L
    x y : Subtype fun x => Membership.mem H x
    hx : Membership.mem LieModule.Weight.ker x
    hy : Membership.mem (Submodule.span K (Singleton.singleton (LieAlgebra.IsKilli …
    ⊢ Eq (((LieModule.traceForm K (Subtype fun x => Membership.mem H x) L) x) y) 0
  -/
  rw [← coe_corootSpace_eq_span_singleton, LieSubmodule.mem_toSubmodule, mem_corootSpace'] at hy
  induction hy using Submodule.span_induction with
  | mem z hz =>
    obtain ⟨u, hu, v, -, huv⟩ := hz
    change killingForm K L (x : L) (z : L) = 0
    replace hx : α x = 0 := by simpa using hx
    rw [← huv, ← traceForm_apply_lie_apply, ← LieSubalgebra.coe_bracket_of_module,
      lie_eq_smul_of_mem_rootSpace hu, hx, zero_smul, map_zero, LinearMap.zero_apply]
  | zero => simp
  | add _ _ _ _ hx hy => simp [hx, hy]
  | smul _ _ _ hz => simp [hz]


@[simp] lemma orthogonal_span_coroot_eq_ker (α : Weight K H L) :
    (traceForm K H L).orthogonal (K ∙ coroot α) = α.ker := by
  if hα : α.IsZero then
    have hα' : coroot α = 0 := by simpa
    replace hα : α.ker = ⊤ := by ext; simp [hα]
    simp [hα, hα']
  else
    refine le_antisymm (fun x hx ↦ ?_) (fun x hx y hy ↦ ?_)
    · simp only [LinearMap.BilinForm.mem_orthogonal_iff] at hx
      specialize hx (coroot α) (Submodule.mem_span_singleton_self _)
      simp only [LinearMap.BilinForm.isOrtho_def, traceForm_coroot, smul_eq_mul, nsmul_eq_mul,
        Nat.cast_ofNat, mul_eq_zero, OfNat.ofNat_ne_zero, inv_eq_zero, false_or] at hx
      simpa using hx.resolve_left (root_apply_cartanEquivDual_symm_ne_zero hα)
    · have := traceForm_eq_zero_of_mem_ker_of_mem_span_coroot hx hy
      rwa [traceForm_comm] at this


@[simp] lemma coroot_eq_iff (α β : Weight K H L) :
    coroot α = coroot β ↔ α = β := by
  /-
    K : Type u_2
    L : Type u_3
    inst✝⁷ : LieRing L
    inst✝⁶ : Field K
    inst✝⁵ : LieAlgebra K L
    inst✝⁴ : FiniteDimensional K L
    H : LieSubalgebra K L
    inst✝³ : H.IsCartanSubalgebra
    inst✝² : LieAlgebra.IsKilling K L
    inst✝¹ : LieModule.IsTriangularizable K (Subtype fun x => Membership.mem H x) L
    inst✝ : CharZero K
    α β : LieModule.Weight K (Subtype fun x => Membership.mem H x) L
    ⊢ Iff (Eq (LieAlgebra.IsKilling.coroot α) (LieAlgebra.IsKilling.coroot β)) (Eq …
  -/
  refine ⟨fun hyp ↦ ?_, fun h ↦ by rw [h]⟩
  if hα : α.IsZero then
    have hβ : β.IsZero := by
      rw [← coroot_eq_zero_iff] at hα ⊢
      rwa [← hyp]
    ext
    simp [hα.eq, hβ.eq]
  else
    have hβ : β.IsNonZero := by
      contrapose! hα
      simp only [not_not, ← coroot_eq_zero_iff] at hα ⊢
      rwa [hyp]
    have : α.ker = β.ker := by
      rw [← orthogonal_span_coroot_eq_ker α, hyp, orthogonal_span_coroot_eq_ker]
    suffices (α : H →ₗ[K] K) = β by ext x; simpa using LinearMap.congr_fun this x
    apply Module.Dual.eq_of_ker_eq_of_apply_eq (coroot α) this
    · rw [Weight.toLinear_apply, root_apply_coroot hα, hyp, Weight.toLinear_apply,
        root_apply_coroot hβ]
    · simp [root_apply_coroot hα]


lemma exists_isSl2Triple_of_weight_isNonZero {α : Weight K H L} (hα : α.IsNonZero) :
    ∃ h e f : L, IsSl2Triple h e f ∧ e ∈ rootSpace H α ∧ f ∈ rootSpace H (- α) := by
  /-
    K : Type u_2
    L : Type u_3
    inst✝⁷ : LieRing L
    inst✝⁶ : Field K
    inst✝⁵ : LieAlgebra K L
    inst✝⁴ : FiniteDimensional K L
    H : LieSubalgebra K L
    inst✝³ : H.IsCartanSubalgebra
    inst✝² : LieAlgebra.IsKilling K L
    inst✝¹ : LieModule.IsTriangularizable K (Subtype fun x => Membership.mem H x) L
    inst✝ : CharZero K
    α : LieModule.Weight K (Subtype fun x => Membership.mem H x) L
    hα : α.IsNonZero
    ⊢ Exists fun h => Exists fun e => Exists fun f => And (IsSl2Triple h e f) (And …
  -/
  obtain ⟨e, heα : e ∈ rootSpace H α, he₀ : e ≠ 0⟩ := α.exists_ne_zero
  obtain ⟨f', hfα, hf⟩ : ∃ f ∈ rootSpace H (-α), killingForm K L e f ≠ 0 := by
    contrapose! he₀
    simpa using mem_ker_killingForm_of_mem_rootSpace_of_forall_rootSpace_neg K L H heα he₀
  /-
    case intro.intro.intro.intro
    K : Type u_2
    L : Type u_3
    inst✝⁷ : LieRing L
    inst✝⁶ : Field K
    inst✝⁵ : LieAlgebra K L
    inst✝⁴ : FiniteDimensional K L
    H : LieSubalgebra K L
    inst✝³ : H.IsCartanSubalgebra
    inst✝² : LieAlgebra.IsKilling K L
    inst✝¹ : LieModule.IsTriangularizable K (Subtype fun x => Membership.mem H x) L
    inst✝ : CharZero K
    α : LieModule.Weight K (Subtype fun x => Membership.mem H x) L
    hα : α.IsNonZero
    e : L
    heα : Membership.mem (LieAlgebra.rootSpace H ⇑α) e
    he₀ : Ne e 0
    f' : L
    hfα : Membership.mem (LieAlgebra.rootSpace H (Neg.neg ⇑α)) f'
    hf : Ne (((killingForm K L) e) f') 0
    ⊢ Exists fun h => Exists fun e => Exists fun f => And (IsSl2Triple h e f) (And …
  -/
  have hef := lie_eq_killingForm_smul_of_mem_rootSpace_of_mem_rootSpace_neg heα hfα
  /-
    case intro.intro.intro.intro
    K : Type u_2
    L : Type u_3
    inst✝⁷ : LieRing L
    inst✝⁶ : Field K
    inst✝⁵ : LieAlgebra K L
    inst✝⁴ : FiniteDimensional K L
    H : LieSubalgebra K L
    inst✝³ : H.IsCartanSubalgebra
    inst✝² : LieAlgebra.IsKilling K L
    inst✝¹ : LieModule.IsTriangularizable K (Subtype fun x => Membership.mem H x) L
    inst✝ : CharZero K
    α : LieModule.Weight K (Subtype fun x => Membership.mem H x) L
    hα : α.IsNonZero
    e : L
    heα : Membership.mem (LieAlgebra.rootSpace H ⇑α) e
    he₀ : Ne e 0
    f' : L
    hfα : Membership.mem (LieAlgebra.rootSpace H (Neg.neg ⇑α)) f'
    hf : Ne (((killingForm K L) e) f') 0
    hef : Eq (Bracket.bracket e f') (HSMul.hSMul (((killingForm K L) e) f') ↑((Lie …
    ⊢ Exists fun h => Exists fun e => Exists fun f => And (IsSl2Triple h e f) (And …
  -/
  let h : H := ⟨⁅e, f'⁆, hef ▸ Submodule.smul_mem _ _ (Submodule.coe_mem _)⟩
  have hh : α h ≠ 0 := by
    have : h = killingForm K L e f' • (cartanEquivDual H).symm α := by
      simp only [h, Subtype.ext_iff, hef]
      rw [Submodule.coe_smul_of_tower]
    rw [this, map_smul, smul_eq_mul, ne_eq, mul_eq_zero, not_or]
    exact ⟨hf, root_apply_cartanEquivDual_symm_ne_zero hα⟩
  /-
    case intro.intro.intro.intro
    K : Type u_2
    L : Type u_3
    inst✝⁷ : LieRing L
    inst✝⁶ : Field K
    inst✝⁵ : LieAlgebra K L
    inst✝⁴ : FiniteDimensional K L
    H : LieSubalgebra K L
    inst✝³ : H.IsCartanSubalgebra
    inst✝² : LieAlgebra.IsKilling K L
    inst✝¹ : LieModule.IsTriangularizable K (Subtype fun x => Membership.mem H x) L
    inst✝ : CharZero K
    α : LieModule.Weight K (Subtype fun x => Membership.mem H x) L
    hα : α.IsNonZero
    e : L
    heα : Membership.mem (LieAlgebra.rootSpace H ⇑α) e
    he₀ : Ne e 0
    f' : L
    hfα : Membership.mem (LieAlgebra.rootSpace H (Neg.neg ⇑α)) f'
    hf : Ne (((killingForm K L) e) f') 0
    hef : Eq (Bracket.bracket e f') (HSMul.hSMul (((killingForm K L) e) f') ↑((Lie …
    h : Subtype fun x => Membership.mem H x := ⟨Bracket.bracket e f', ⋯⟩
    hh : Ne (α h) 0
    ⊢ Exists fun h => Exists fun e => Exists fun f => And (IsSl2Triple h e f) (And …
  -/
  let f := (2 * (α h)⁻¹) • f'
  replace hef : ⁅⁅e, f⁆, e⁆ = 2 • e := by
    have : ⁅⁅e, f'⁆, e⁆ = α h • e := lie_eq_smul_of_mem_rootSpace heα h
    rw [lie_smul, smul_lie, this, ← smul_assoc, smul_eq_mul, mul_assoc, inv_mul_cancel₀ hh,
      mul_one, two_smul, two_smul]
  /-
    case intro.intro.intro.intro
    K : Type u_2
    L : Type u_3
    inst✝⁷ : LieRing L
    inst✝⁶ : Field K
    inst✝⁵ : LieAlgebra K L
    inst✝⁴ : FiniteDimensional K L
    H : LieSubalgebra K L
    inst✝³ : H.IsCartanSubalgebra
    inst✝² : LieAlgebra.IsKilling K L
    inst✝¹ : LieModule.IsTriangularizable K (Subtype fun x => Membership.mem H x) L
    inst✝ : CharZero K
    α : LieModule.Weight K (Subtype fun x => Membership.mem H x) L
    hα : α.IsNonZero
    e : L
    heα : Membership.mem (LieAlgebra.rootSpace H ⇑α) e
    he₀ : Ne e 0
    f' : L
    hfα : Membership.mem (LieAlgebra.rootSpace H (Neg.neg ⇑α)) f'
    hf : Ne (((killingForm K L) e) f') 0
    hef✝ : Eq (Bracket.bracket e f') (HSMul.hSMul (((killingForm K L) e) f') ↑((Li …
    h : Subtype fun x => Membership.mem H x := ⟨Bracket.bracket e f', ⋯⟩
    hh : Ne (α h) 0
    f : L := HSMul.hSMul (HMul.hMul 2 (Inv.inv (α h))) f'
    hef : Eq (Bracket.bracket (Bracket.bracket e f) e) (HSMul.hSMul 2 e)
    ⊢ Exists fun h => Exists fun e => Exists fun f => And (IsSl2Triple h e f) (And …
  -/
  refine ⟨⁅e, f⁆, e, f, ⟨fun contra ↦ ?_, rfl, hef, ?_⟩, heα, Submodule.smul_mem _ _ hfα⟩
    /-
      case intro.intro.intro.intro.refine_1
      K : Type u_2
      L : Type u_3
      inst✝⁷ : LieRing L
      inst✝⁶ : Field K
      inst✝⁵ : LieAlgebra K L
      inst✝⁴ : FiniteDimensional K L
      H : LieSubalgebra K L
      inst✝³ : H.IsCartanSubalgebra
      inst✝² : LieAlgebra.IsKilling K L
      inst✝¹ : LieModule.IsTriangularizable K (Subtype fun x => Membership.mem H x) L
      inst✝ : CharZero K
      α : LieModule.Weight K (Subtype fun x => Membership.mem H x) L
      hα : α.IsNonZero
      e : L
      heα : Membership.mem (LieAlgebra.rootSpace H ⇑α) e
      he₀ : Ne e 0
      f' : L
      hfα : Membership.mem (LieAlgebra.rootSpace H (Neg.neg ⇑α)) f'
      hf : Ne (((killingForm K L) e) f') 0
      hef✝ : Eq (Bracket.bracket e f') (HSMul.hSMul (((killingForm K L) e) f') ↑((Li …
      h : Subtype fun x => Membership.mem H x := ⟨Bracket.bracket e f', ⋯⟩
      hh : Ne (α h) 0
      f : L := HSMul.hSMul (HMul.hMul 2 (Inv.inv (α h))) f'
      hef : Eq (Bracket.bracket (Bracket.bracket e f) e) (HSMul.hSMul 2 e)
      contra : Eq (Bracket.bracket e f) 0
      ⊢ False
    -/
  · rw [contra] at hef
    /-
      case intro.intro.intro.intro.refine_1
      K : Type u_2
      L : Type u_3
      inst✝⁷ : LieRing L
      inst✝⁶ : Field K
      inst✝⁵ : LieAlgebra K L
      inst✝⁴ : FiniteDimensional K L
      H : LieSubalgebra K L
      inst✝³ : H.IsCartanSubalgebra
      inst✝² : LieAlgebra.IsKilling K L
      inst✝¹ : LieModule.IsTriangularizable K (Subtype fun x => Membership.mem H x) L
      inst✝ : CharZero K
      α : LieModule.Weight K (Subtype fun x => Membership.mem H x) L
      hα : α.IsNonZero
      e : L
      heα : Membership.mem (LieAlgebra.rootSpace H ⇑α) e
      he₀ : Ne e 0
      f' : L
      hfα : Membership.mem (LieAlgebra.rootSpace H (Neg.neg ⇑α)) f'
      hf : Ne (((killingForm K L) e) f') 0
      hef✝ : Eq (Bracket.bracket e f') (HSMul.hSMul (((killingForm K L) e) f') ↑((Li …
      h : Subtype fun x => Membership.mem H x := ⟨Bracket.bracket e f', ⋯⟩
      hh : Ne (α h) 0
      f : L := HSMul.hSMul (HMul.hMul 2 (Inv.inv (α h))) f'
      hef : Eq (Bracket.bracket 0 e) (HSMul.hSMul 2 e)
      contra : Eq (Bracket.bracket e f) 0
      ⊢ False
    -/
    have _i : NoZeroSMulDivisors ℤ L := NoZeroSMulDivisors.int_of_charZero K L
    /-
      case intro.intro.intro.intro.refine_1
      K : Type u_2
      L : Type u_3
      inst✝⁷ : LieRing L
      inst✝⁶ : Field K
      inst✝⁵ : LieAlgebra K L
      inst✝⁴ : FiniteDimensional K L
      H : LieSubalgebra K L
      inst✝³ : H.IsCartanSubalgebra
      inst✝² : LieAlgebra.IsKilling K L
      inst✝¹ : LieModule.IsTriangularizable K (Subtype fun x => Membership.mem H x) L
      inst✝ : CharZero K
      α : LieModule.Weight K (Subtype fun x => Membership.mem H x) L
      hα : α.IsNonZero
      e : L
      heα : Membership.mem (LieAlgebra.rootSpace H ⇑α) e
      he₀ : Ne e 0
      f' : L
      hfα : Membership.mem (LieAlgebra.rootSpace H (Neg.neg ⇑α)) f'
      hf : Ne (((killingForm K L) e) f') 0
      hef✝ : Eq (Bracket.bracket e f') (HSMul.hSMul (((killingForm K L) e) f') ↑((Li …
      h : Subtype fun x => Membership.mem H x := ⟨Bracket.bracket e f', ⋯⟩
      hh : Ne (α h) 0
      f : L := HSMul.hSMul (HMul.hMul 2 (Inv.inv (α h))) f'
      hef : Eq (Bracket.bracket 0 e) (HSMul.hSMul 2 e)
      contra : Eq (Bracket.bracket e f) 0
      _i : NoZeroSMulDivisors Int L
      ⊢ False
    -/
    simp only [zero_lie, eq_comm (a := (0 : L)), smul_eq_zero, OfNat.ofNat_ne_zero, false_or] at hef
    /-
      case intro.intro.intro.intro.refine_1
      K : Type u_2
      L : Type u_3
      inst✝⁷ : LieRing L
      inst✝⁶ : Field K
      inst✝⁵ : LieAlgebra K L
      inst✝⁴ : FiniteDimensional K L
      H : LieSubalgebra K L
      inst✝³ : H.IsCartanSubalgebra
      inst✝² : LieAlgebra.IsKilling K L
      inst✝¹ : LieModule.IsTriangularizable K (Subtype fun x => Membership.mem H x) L
      inst✝ : CharZero K
      α : LieModule.Weight K (Subtype fun x => Membership.mem H x) L
      hα : α.IsNonZero
      e : L
      heα : Membership.mem (LieAlgebra.rootSpace H ⇑α) e
      he₀ : Ne e 0
      f' : L
      hfα : Membership.mem (LieAlgebra.rootSpace H (Neg.neg ⇑α)) f'
      hf : Ne (((killingForm K L) e) f') 0
      hef✝ : Eq (Bracket.bracket e f') (HSMul.hSMul (((killingForm K L) e) f') ↑((Li …
      h : Subtype fun x => Membership.mem H x := ⟨Bracket.bracket e f', ⋯⟩
      hh : Ne (α h) 0
      f : L := HSMul.hSMul (HMul.hMul 2 (Inv.inv (α h))) f'
      contra : Eq (Bracket.bracket e f) 0
      _i : NoZeroSMulDivisors Int L
      hef : Eq e 0
      ⊢ False
    -/
    contradiction
    /-
      🎉 no goals
    -/
    /-
      case intro.intro.intro.intro.refine_2
      K : Type u_2
      L : Type u_3
      inst✝⁷ : LieRing L
      inst✝⁶ : Field K
      inst✝⁵ : LieAlgebra K L
      inst✝⁴ : FiniteDimensional K L
      H : LieSubalgebra K L
      inst✝³ : H.IsCartanSubalgebra
      inst✝² : LieAlgebra.IsKilling K L
      inst✝¹ : LieModule.IsTriangularizable K (Subtype fun x => Membership.mem H x) L
      inst✝ : CharZero K
      α : LieModule.Weight K (Subtype fun x => Membership.mem H x) L
      hα : α.IsNonZero
      e : L
      heα : Membership.mem (LieAlgebra.rootSpace H ⇑α) e
      he₀ : Ne e 0
      f' : L
      hfα : Membership.mem (LieAlgebra.rootSpace H (Neg.neg ⇑α)) f'
      hf : Ne (((killingForm K L) e) f') 0
      hef✝ : Eq (Bracket.bracket e f') (HSMul.hSMul (((killingForm K L) e) f') ↑((Li …
      h : Subtype fun x => Membership.mem H x := ⟨Bracket.bracket e f', ⋯⟩
      hh : Ne (α h) 0
      f : L := HSMul.hSMul (HMul.hMul 2 (Inv.inv (α h))) f'
      hef : Eq (Bracket.bracket (Bracket.bracket e f) e) (HSMul.hSMul 2 e)
      ⊢ Eq (Bracket.bracket (Bracket.bracket e f) f) (Neg.neg (HSMul.hSMul 2 f))
    -/
  · have : ⁅⁅e, f'⁆, f'⁆ = - α h • f' := lie_eq_smul_of_mem_rootSpace hfα h
    /-
      case intro.intro.intro.intro.refine_2
      K : Type u_2
      L : Type u_3
      inst✝⁷ : LieRing L
      inst✝⁶ : Field K
      inst✝⁵ : LieAlgebra K L
      inst✝⁴ : FiniteDimensional K L
      H : LieSubalgebra K L
      inst✝³ : H.IsCartanSubalgebra
      inst✝² : LieAlgebra.IsKilling K L
      inst✝¹ : LieModule.IsTriangularizable K (Subtype fun x => Membership.mem H x) L
      inst✝ : CharZero K
      α : LieModule.Weight K (Subtype fun x => Membership.mem H x) L
      hα : α.IsNonZero
      e : L
      heα : Membership.mem (LieAlgebra.rootSpace H ⇑α) e
      he₀ : Ne e 0
      f' : L
      hfα : Membership.mem (LieAlgebra.rootSpace H (Neg.neg ⇑α)) f'
      hf : Ne (((killingForm K L) e) f') 0
      hef✝ : Eq (Bracket.bracket e f') (HSMul.hSMul (((killingForm K L) e) f') ↑((Li …
      h : Subtype fun x => Membership.mem H x := ⟨Bracket.bracket e f', ⋯⟩
      hh : Ne (α h) 0
      f : L := HSMul.hSMul (HMul.hMul 2 (Inv.inv (α h))) f'
      hef : Eq (Bracket.bracket (Bracket.bracket e f) e) (HSMul.hSMul 2 e)
      this : Eq (Bracket.bracket (Bracket.bracket e f') f') (HSMul.hSMul (Neg.neg (α …
      ⊢ Eq (Bracket.bracket (Bracket.bracket e f) f) (Neg.neg (HSMul.hSMul 2 f))
    -/
    rw [lie_smul, lie_smul, smul_lie, this]
    /-
      case intro.intro.intro.intro.refine_2
      K : Type u_2
      L : Type u_3
      inst✝⁷ : LieRing L
      inst✝⁶ : Field K
      inst✝⁵ : LieAlgebra K L
      inst✝⁴ : FiniteDimensional K L
      H : LieSubalgebra K L
      inst✝³ : H.IsCartanSubalgebra
      inst✝² : LieAlgebra.IsKilling K L
      inst✝¹ : LieModule.IsTriangularizable K (Subtype fun x => Membership.mem H x) L
      inst✝ : CharZero K
      α : LieModule.Weight K (Subtype fun x => Membership.mem H x) L
      hα : α.IsNonZero
      e : L
      heα : Membership.mem (LieAlgebra.rootSpace H ⇑α) e
      he₀ : Ne e 0
      f' : L
      hfα : Membership.mem (LieAlgebra.rootSpace H (Neg.neg ⇑α)) f'
      hf : Ne (((killingForm K L) e) f') 0
      hef✝ : Eq (Bracket.bracket e f') (HSMul.hSMul (((killingForm K L) e) f') ↑((Li …
      h : Subtype fun x => Membership.mem H x := ⟨Bracket.bracket e f', ⋯⟩
      hh : Ne (α h) 0
      f : L := HSMul.hSMul (HMul.hMul 2 (Inv.inv (α h))) f'
      hef : Eq (Bracket.bracket (Bracket.bracket e f) e) (HSMul.hSMul 2 e)
      this : Eq (Bracket.bracket (Bracket.bracket e f') f') (HSMul.hSMul (Neg.neg (α …
      ⊢ Eq (HSMul.hSMul (HMul.hMul 2 (Inv.inv (α h))) (HSMul.hSMul (HMul.hMul 2 (Inv …
    -/
    simp [← smul_assoc, f, hh, mul_comm _ (2 * (α h)⁻¹)]
    /-
      🎉 no goals
    -/


lemma _root_.IsSl2Triple.h_eq_coroot {α : Weight K H L} (hα : α.IsNonZero)
    {h e f : L} (ht : IsSl2Triple h e f) (heα : e ∈ rootSpace H α) (hfα : f ∈ rootSpace H (- α)) :
    h = coroot α := by
  /-
    K : Type u_2
    L : Type u_3
    inst✝⁷ : LieRing L
    inst✝⁶ : Field K
    inst✝⁵ : LieAlgebra K L
    inst✝⁴ : FiniteDimensional K L
    H : LieSubalgebra K L
    inst✝³ : H.IsCartanSubalgebra
    inst✝² : LieAlgebra.IsKilling K L
    inst✝¹ : LieModule.IsTriangularizable K (Subtype fun x => Membership.mem H x) L
    inst✝ : CharZero K
    α : LieModule.Weight K (Subtype fun x => Membership.mem H x) L
    hα : α.IsNonZero
    h e f : L
    ht : IsSl2Triple h e f
    heα : Membership.mem (LieAlgebra.rootSpace H ⇑α) e
    hfα : Membership.mem (LieAlgebra.rootSpace H (Neg.neg ⇑α)) f
    ⊢ Eq h ↑(LieAlgebra.IsKilling.coroot α)
  -/
  have hef := lie_eq_killingForm_smul_of_mem_rootSpace_of_mem_rootSpace_neg heα hfα
  /-
    K : Type u_2
    L : Type u_3
    inst✝⁷ : LieRing L
    inst✝⁶ : Field K
    inst✝⁵ : LieAlgebra K L
    inst✝⁴ : FiniteDimensional K L
    H : LieSubalgebra K L
    inst✝³ : H.IsCartanSubalgebra
    inst✝² : LieAlgebra.IsKilling K L
    inst✝¹ : LieModule.IsTriangularizable K (Subtype fun x => Membership.mem H x) L
    inst✝ : CharZero K
    α : LieModule.Weight K (Subtype fun x => Membership.mem H x) L
    hα : α.IsNonZero
    h e f : L
    ht : IsSl2Triple h e f
    heα : Membership.mem (LieAlgebra.rootSpace H ⇑α) e
    hfα : Membership.mem (LieAlgebra.rootSpace H (Neg.neg ⇑α)) f
    hef : Eq (Bracket.bracket e f) (HSMul.hSMul (((killingForm K L) e) f) ↑((LieAl …
    ⊢ Eq h ↑(LieAlgebra.IsKilling.coroot α)
  -/
  lift h to H using by simpa only [← ht.lie_e_f, hef] using H.smul_mem _ (Submodule.coe_mem _)
  /-
    case intro
    K : Type u_2
    L : Type u_3
    inst✝⁷ : LieRing L
    inst✝⁶ : Field K
    inst✝⁵ : LieAlgebra K L
    inst✝⁴ : FiniteDimensional K L
    H : LieSubalgebra K L
    inst✝³ : H.IsCartanSubalgebra
    inst✝² : LieAlgebra.IsKilling K L
    inst✝¹ : LieModule.IsTriangularizable K (Subtype fun x => Membership.mem H x) L
    inst✝ : CharZero K
    α : LieModule.Weight K (Subtype fun x => Membership.mem H x) L
    hα : α.IsNonZero
    e f : L
    heα : Membership.mem (LieAlgebra.rootSpace H ⇑α) e
    hfα : Membership.mem (LieAlgebra.rootSpace H (Neg.neg ⇑α)) f
    hef : Eq (Bracket.bracket e f) (HSMul.hSMul (((killingForm K L) e) f) ↑((LieAl …
    h : Subtype fun x => Membership.mem H x
    ht : IsSl2Triple (↑h) e f
    ⊢ Eq ↑h ↑(LieAlgebra.IsKilling.coroot α)
  -/
  congr 1
  have key : α h = 2 := by
    have := lie_eq_smul_of_mem_rootSpace heα h
    rw [LieSubalgebra.coe_bracket_of_module, ht.lie_h_e_smul K] at this
    exact smul_left_injective K ht.e_ne_zero this.symm
  suffices ∃ s : K, s • h = coroot α by
    obtain ⟨s, hs⟩ := this
    replace this : s = 1 := by simpa [root_apply_coroot hα, key] using congr_arg α hs
    rwa [this, one_smul] at hs
  /-
    case intro.e_self
    K : Type u_2
    L : Type u_3
    inst✝⁷ : LieRing L
    inst✝⁶ : Field K
    inst✝⁵ : LieAlgebra K L
    inst✝⁴ : FiniteDimensional K L
    H : LieSubalgebra K L
    inst✝³ : H.IsCartanSubalgebra
    inst✝² : LieAlgebra.IsKilling K L
    inst✝¹ : LieModule.IsTriangularizable K (Subtype fun x => Membership.mem H x) L
    inst✝ : CharZero K
    α : LieModule.Weight K (Subtype fun x => Membership.mem H x) L
    hα : α.IsNonZero
    e f : L
    heα : Membership.mem (LieAlgebra.rootSpace H ⇑α) e
    hfα : Membership.mem (LieAlgebra.rootSpace H (Neg.neg ⇑α)) f
    hef : Eq (Bracket.bracket e f) (HSMul.hSMul (((killingForm K L) e) f) ↑((LieAl …
    h : Subtype fun x => Membership.mem H x
    ht : IsSl2Triple (↑h) e f
    key : Eq (α h) 2
    ⊢ Exists fun s => Eq (HSMul.hSMul s h) (LieAlgebra.IsKilling.coroot α)
  -/
  set α' := (cartanEquivDual H).symm α with hα'
  have h_eq : h = killingForm K L e f • α' := by
    simp only [hα', Subtype.ext_iff, ← ht.lie_e_f, hef]
    rw [Submodule.coe_smul_of_tower]
  /-
    case intro.e_self
    K : Type u_2
    L : Type u_3
    inst✝⁷ : LieRing L
    inst✝⁶ : Field K
    inst✝⁵ : LieAlgebra K L
    inst✝⁴ : FiniteDimensional K L
    H : LieSubalgebra K L
    inst✝³ : H.IsCartanSubalgebra
    inst✝² : LieAlgebra.IsKilling K L
    inst✝¹ : LieModule.IsTriangularizable K (Subtype fun x => Membership.mem H x) L
    inst✝ : CharZero K
    α : LieModule.Weight K (Subtype fun x => Membership.mem H x) L
    hα : α.IsNonZero
    e f : L
    heα : Membership.mem (LieAlgebra.rootSpace H ⇑α) e
    hfα : Membership.mem (LieAlgebra.rootSpace H (Neg.neg ⇑α)) f
    h : Subtype fun x => Membership.mem H x
    ht : IsSl2Triple (↑h) e f
    key : Eq (α h) 2
    α' : Subtype fun x => Membership.mem H x := (LieAlgebra.IsKilling.cartanEquivD …
    hef : Eq (Bracket.bracket e f) (HSMul.hSMul (((killingForm K L) e) f) ↑α')
    hα' : Eq α' ((LieAlgebra.IsKilling.cartanEquivDual H).symm (LieModule.Weight.t …
    h_eq : Eq h (HSMul.hSMul (((killingForm K L) e) f) α')
    ⊢ Exists fun s => Eq (HSMul.hSMul s h) (LieAlgebra.IsKilling.coroot α)
  -/
  use (2 • (α α')⁻¹) * (killingForm K L e f)⁻¹
  have hef₀ : killingForm K L e f ≠ 0 := by
    have := ht.h_ne_zero
    contrapose! this
    simpa [this] using h_eq
  /-
    case h
    K : Type u_2
    L : Type u_3
    inst✝⁷ : LieRing L
    inst✝⁶ : Field K
    inst✝⁵ : LieAlgebra K L
    inst✝⁴ : FiniteDimensional K L
    H : LieSubalgebra K L
    inst✝³ : H.IsCartanSubalgebra
    inst✝² : LieAlgebra.IsKilling K L
    inst✝¹ : LieModule.IsTriangularizable K (Subtype fun x => Membership.mem H x) L
    inst✝ : CharZero K
    α : LieModule.Weight K (Subtype fun x => Membership.mem H x) L
    hα : α.IsNonZero
    e f : L
    heα : Membership.mem (LieAlgebra.rootSpace H ⇑α) e
    hfα : Membership.mem (LieAlgebra.rootSpace H (Neg.neg ⇑α)) f
    h : Subtype fun x => Membership.mem H x
    ht : IsSl2Triple (↑h) e f
    key : Eq (α h) 2
    α' : Subtype fun x => Membership.mem H x := (LieAlgebra.IsKilling.cartanEquivD …
    hef : Eq (Bracket.bracket e f) (HSMul.hSMul (((killingForm K L) e) f) ↑α')
    hα' : Eq α' ((LieAlgebra.IsKilling.cartanEquivDual H).symm (LieModule.Weight.t …
    h_eq : Eq h (HSMul.hSMul (((killingForm K L) e) f) α')
    hef₀ : Ne (((killingForm K L) e) f) 0
    ⊢ Eq (HSMul.hSMul (HMul.hMul (HSMul.hSMul 2 (Inv.inv (α α'))) (Inv.inv (((kill …
  -/
  rw [h_eq, smul_smul, mul_assoc, inv_mul_cancel₀ hef₀, mul_one, smul_assoc, coroot]
  /-
    🎉 no goals
  -/


lemma finrank_rootSpace_eq_one (α : Weight K H L) (hα : α.IsNonZero) :
    finrank K (rootSpace H α) = 1 := by
  suffices ¬ 1 < finrank K (rootSpace H α) by
    have h₀ : finrank K (rootSpace H α) ≠ 0 := by
      convert_to finrank K (rootSpace H α).toSubmodule ≠ 0
      simpa using α.genWeightSpace_ne_bot
    omega
  /-
    K : Type u_2
    L : Type u_3
    inst✝⁷ : LieRing L
    inst✝⁶ : Field K
    inst✝⁵ : LieAlgebra K L
    inst✝⁴ : FiniteDimensional K L
    H : LieSubalgebra K L
    inst✝³ : H.IsCartanSubalgebra
    inst✝² : LieAlgebra.IsKilling K L
    inst✝¹ : LieModule.IsTriangularizable K (Subtype fun x => Membership.mem H x) L
    inst✝ : CharZero K
    α : LieModule.Weight K (Subtype fun x => Membership.mem H x) L
    hα : α.IsNonZero
    ⊢ Not (LT.lt 1 (Module.finrank K (Subtype fun x => Membership.mem (LieAlgebra. …
  -/
  intro contra
  /-
    K : Type u_2
    L : Type u_3
    inst✝⁷ : LieRing L
    inst✝⁶ : Field K
    inst✝⁵ : LieAlgebra K L
    inst✝⁴ : FiniteDimensional K L
    H : LieSubalgebra K L
    inst✝³ : H.IsCartanSubalgebra
    inst✝² : LieAlgebra.IsKilling K L
    inst✝¹ : LieModule.IsTriangularizable K (Subtype fun x => Membership.mem H x) L
    inst✝ : CharZero K
    α : LieModule.Weight K (Subtype fun x => Membership.mem H x) L
    hα : α.IsNonZero
    contra : LT.lt 1 (Module.finrank K (Subtype fun x => Membership.mem (LieAlgebr …
    ⊢ False
  -/
  obtain ⟨h, e, f, ht, heα, hfα⟩ := exists_isSl2Triple_of_weight_isNonZero hα
  /-
    case intro.intro.intro.intro.intro
    K : Type u_2
    L : Type u_3
    inst✝⁷ : LieRing L
    inst✝⁶ : Field K
    inst✝⁵ : LieAlgebra K L
    inst✝⁴ : FiniteDimensional K L
    H : LieSubalgebra K L
    inst✝³ : H.IsCartanSubalgebra
    inst✝² : LieAlgebra.IsKilling K L
    inst✝¹ : LieModule.IsTriangularizable K (Subtype fun x => Membership.mem H x) L
    inst✝ : CharZero K
    α : LieModule.Weight K (Subtype fun x => Membership.mem H x) L
    hα : α.IsNonZero
    contra : LT.lt 1 (Module.finrank K (Subtype fun x => Membership.mem (LieAlgebr …
    h e f : L
    ht : IsSl2Triple h e f
    heα : Membership.mem (LieAlgebra.rootSpace H ⇑α) e
    hfα : Membership.mem (LieAlgebra.rootSpace H (Neg.neg ⇑α)) f
    ⊢ False
  -/
  let F : rootSpace H α →ₗ[K] K := killingForm K L f ∘ₗ (rootSpace H α).subtype
  /-
    case intro.intro.intro.intro.intro
    K : Type u_2
    L : Type u_3
    inst✝⁷ : LieRing L
    inst✝⁶ : Field K
    inst✝⁵ : LieAlgebra K L
    inst✝⁴ : FiniteDimensional K L
    H : LieSubalgebra K L
    inst✝³ : H.IsCartanSubalgebra
    inst✝² : LieAlgebra.IsKilling K L
    inst✝¹ : LieModule.IsTriangularizable K (Subtype fun x => Membership.mem H x) L
    inst✝ : CharZero K
    α : LieModule.Weight K (Subtype fun x => Membership.mem H x) L
    hα : α.IsNonZero
    contra : LT.lt 1 (Module.finrank K (Subtype fun x => Membership.mem (LieAlgebr …
    h e f : L
    ht : IsSl2Triple h e f
    heα : Membership.mem (LieAlgebra.rootSpace H ⇑α) e
    hfα : Membership.mem (LieAlgebra.rootSpace H (Neg.neg ⇑α)) f
    F : LinearMap (RingHom.id K) (Subtype fun x => Membership.mem (LieAlgebra.root …
    ⊢ False
  -/
  have hF : LinearMap.ker F ≠ ⊥ := F.ker_ne_bot_of_finrank_lt <| by rwa [finrank_self]
  /-
    case intro.intro.intro.intro.intro
    K : Type u_2
    L : Type u_3
    inst✝⁷ : LieRing L
    inst✝⁶ : Field K
    inst✝⁵ : LieAlgebra K L
    inst✝⁴ : FiniteDimensional K L
    H : LieSubalgebra K L
    inst✝³ : H.IsCartanSubalgebra
    inst✝² : LieAlgebra.IsKilling K L
    inst✝¹ : LieModule.IsTriangularizable K (Subtype fun x => Membership.mem H x) L
    inst✝ : CharZero K
    α : LieModule.Weight K (Subtype fun x => Membership.mem H x) L
    hα : α.IsNonZero
    contra : LT.lt 1 (Module.finrank K (Subtype fun x => Membership.mem (LieAlgebr …
    h e f : L
    ht : IsSl2Triple h e f
    heα : Membership.mem (LieAlgebra.rootSpace H ⇑α) e
    hfα : Membership.mem (LieAlgebra.rootSpace H (Neg.neg ⇑α)) f
    F : LinearMap (RingHom.id K) (Subtype fun x => Membership.mem (LieAlgebra.root …
    hF : Ne (LinearMap.ker F) Bot.bot
    ⊢ False
  -/
  obtain ⟨⟨y, hyα⟩, hy, hy₀⟩ := (Submodule.ne_bot_iff _).mp hF
  replace hy : ⁅y, f⁆ = 0 := by
    have : killingForm K L y f = 0 := by simpa [F, traceForm_comm] using hy
    simpa [this] using lie_eq_killingForm_smul_of_mem_rootSpace_of_mem_rootSpace_neg hyα hfα
  have P : ht.symm.HasPrimitiveVectorWith y (-2 : K) :=
    { ne_zero := by simpa [LieSubmodule.mk_eq_zero] using hy₀
      lie_h := by simp only [neg_smul, neg_lie, neg_inj, ht.h_eq_coroot hα heα hfα,
        ← H.coe_bracket_of_module, lie_eq_smul_of_mem_rootSpace hyα (coroot α),
        root_apply_coroot hα]
      lie_e := by rw [← lie_skew, hy, neg_zero] }
  /-
    case intro.intro.intro.intro.intro.intro.mk.intro
    K : Type u_2
    L : Type u_3
    inst✝⁷ : LieRing L
    inst✝⁶ : Field K
    inst✝⁵ : LieAlgebra K L
    inst✝⁴ : FiniteDimensional K L
    H : LieSubalgebra K L
    inst✝³ : H.IsCartanSubalgebra
    inst✝² : LieAlgebra.IsKilling K L
    inst✝¹ : LieModule.IsTriangularizable K (Subtype fun x => Membership.mem H x) L
    inst✝ : CharZero K
    α : LieModule.Weight K (Subtype fun x => Membership.mem H x) L
    hα : α.IsNonZero
    contra : LT.lt 1 (Module.finrank K (Subtype fun x => Membership.mem (LieAlgebr …
    h e f : L
    ht : IsSl2Triple h e f
    heα : Membership.mem (LieAlgebra.rootSpace H ⇑α) e
    hfα : Membership.mem (LieAlgebra.rootSpace H (Neg.neg ⇑α)) f
    F : LinearMap (RingHom.id K) (Subtype fun x => Membership.mem (LieAlgebra.root …
    hF : Ne (LinearMap.ker F) Bot.bot
    y : L
    hyα : Membership.mem (LieAlgebra.rootSpace H ⇑α) y
    hy₀ : Ne ⟨y, hyα⟩ 0
    hy : Eq (Bracket.bracket y f) 0
    P : ⋯.HasPrimitiveVectorWith y (-2)
    ⊢ False
  -/
  obtain ⟨n, hn⟩ := P.exists_nat
  /-
    case intro.intro.intro.intro.intro.intro.mk.intro.intro
    K : Type u_2
    L : Type u_3
    inst✝⁷ : LieRing L
    inst✝⁶ : Field K
    inst✝⁵ : LieAlgebra K L
    inst✝⁴ : FiniteDimensional K L
    H : LieSubalgebra K L
    inst✝³ : H.IsCartanSubalgebra
    inst✝² : LieAlgebra.IsKilling K L
    inst✝¹ : LieModule.IsTriangularizable K (Subtype fun x => Membership.mem H x) L
    inst✝ : CharZero K
    α : LieModule.Weight K (Subtype fun x => Membership.mem H x) L
    hα : α.IsNonZero
    contra : LT.lt 1 (Module.finrank K (Subtype fun x => Membership.mem (LieAlgebr …
    h e f : L
    ht : IsSl2Triple h e f
    heα : Membership.mem (LieAlgebra.rootSpace H ⇑α) e
    hfα : Membership.mem (LieAlgebra.rootSpace H (Neg.neg ⇑α)) f
    F : LinearMap (RingHom.id K) (Subtype fun x => Membership.mem (LieAlgebra.root …
    hF : Ne (LinearMap.ker F) Bot.bot
    y : L
    hyα : Membership.mem (LieAlgebra.rootSpace H ⇑α) y
    hy₀ : Ne ⟨y, hyα⟩ 0
    hy : Eq (Bracket.bracket y f) 0
    P : ⋯.HasPrimitiveVectorWith y (-2)
    n : Nat
    hn : Eq (-2) ↑n
    ⊢ False
  -/
  replace hn : -2 = (n : ℤ) := by norm_cast at hn
  /-
    case intro.intro.intro.intro.intro.intro.mk.intro.intro
    K : Type u_2
    L : Type u_3
    inst✝⁷ : LieRing L
    inst✝⁶ : Field K
    inst✝⁵ : LieAlgebra K L
    inst✝⁴ : FiniteDimensional K L
    H : LieSubalgebra K L
    inst✝³ : H.IsCartanSubalgebra
    inst✝² : LieAlgebra.IsKilling K L
    inst✝¹ : LieModule.IsTriangularizable K (Subtype fun x => Membership.mem H x) L
    inst✝ : CharZero K
    α : LieModule.Weight K (Subtype fun x => Membership.mem H x) L
    hα : α.IsNonZero
    contra : LT.lt 1 (Module.finrank K (Subtype fun x => Membership.mem (LieAlgebr …
    h e f : L
    ht : IsSl2Triple h e f
    heα : Membership.mem (LieAlgebra.rootSpace H ⇑α) e
    hfα : Membership.mem (LieAlgebra.rootSpace H (Neg.neg ⇑α)) f
    F : LinearMap (RingHom.id K) (Subtype fun x => Membership.mem (LieAlgebra.root …
    hF : Ne (LinearMap.ker F) Bot.bot
    y : L
    hyα : Membership.mem (LieAlgebra.rootSpace H ⇑α) y
    hy₀ : Ne ⟨y, hyα⟩ 0
    hy : Eq (Bracket.bracket y f) 0
    P : ⋯.HasPrimitiveVectorWith y (-2)
    n : Nat
    hn : Eq (-2) ↑n
    ⊢ False
  -/
  omega
  /-
    🎉 no goals
  -/


/-- The collection of roots as a `Finset`. -/
noncomputable abbrev _root_.LieSubalgebra.root : Finset (Weight K H L) := {α | α.IsNonZero}


lemma restrict_killingForm_eq_sum :
    (killingForm K L).restrict H = ∑ α in H.root, (α : H →ₗ[K] K).smulRight (α : H →ₗ[K] K) := by
  /-
    K : Type u_2
    L : Type u_3
    inst✝⁷ : LieRing L
    inst✝⁶ : Field K
    inst✝⁵ : LieAlgebra K L
    inst✝⁴ : FiniteDimensional K L
    H : LieSubalgebra K L
    inst✝³ : H.IsCartanSubalgebra
    inst✝² : LieAlgebra.IsKilling K L
    inst✝¹ : LieModule.IsTriangularizable K (Subtype fun x => Membership.mem H x) L
    inst✝ : CharZero K
    ⊢ Eq ((killingForm K L).restrict H.toSubmodule) (LieSubalgebra.root.sum fun α  …
  -/
  rw [restrict_killingForm, traceForm_eq_sum_finrank_nsmul' K H L]
  /-
    K : Type u_2
    L : Type u_3
    inst✝⁷ : LieRing L
    inst✝⁶ : Field K
    inst✝⁵ : LieAlgebra K L
    inst✝⁴ : FiniteDimensional K L
    H : LieSubalgebra K L
    inst✝³ : H.IsCartanSubalgebra
    inst✝² : LieAlgebra.IsKilling K L
    inst✝¹ : LieModule.IsTriangularizable K (Subtype fun x => Membership.mem H x) L
    inst✝ : CharZero K
    ⊢ Eq ((Finset.filter (fun χ => χ.IsNonZero) Finset.univ).sum fun χ => HSMul.hS …
  -/
  refine Finset.sum_congr rfl fun χ hχ ↦ ?_
  /-
    K : Type u_2
    L : Type u_3
    inst✝⁷ : LieRing L
    inst✝⁶ : Field K
    inst✝⁵ : LieAlgebra K L
    inst✝⁴ : FiniteDimensional K L
    H : LieSubalgebra K L
    inst✝³ : H.IsCartanSubalgebra
    inst✝² : LieAlgebra.IsKilling K L
    inst✝¹ : LieModule.IsTriangularizable K (Subtype fun x => Membership.mem H x) L
    inst✝ : CharZero K
    χ : LieModule.Weight K (Subtype fun x => Membership.mem H x) L
    hχ : Membership.mem LieSubalgebra.root χ
    ⊢ Eq (HSMul.hSMul (Module.finrank K (Subtype fun x => Membership.mem (LieModul …
  -/
  replace hχ : χ.IsNonZero := by simpa [LieSubalgebra.root] using hχ
  /-
    K : Type u_2
    L : Type u_3
    inst✝⁷ : LieRing L
    inst✝⁶ : Field K
    inst✝⁵ : LieAlgebra K L
    inst✝⁴ : FiniteDimensional K L
    H : LieSubalgebra K L
    inst✝³ : H.IsCartanSubalgebra
    inst✝² : LieAlgebra.IsKilling K L
    inst✝¹ : LieModule.IsTriangularizable K (Subtype fun x => Membership.mem H x) L
    inst✝ : CharZero K
    χ : LieModule.Weight K (Subtype fun x => Membership.mem H x) L
    hχ : χ.IsNonZero
    ⊢ Eq (HSMul.hSMul (Module.finrank K (Subtype fun x => Membership.mem (LieModul …
  -/
  simp [finrank_rootSpace_eq_one _ hχ]
  /-
    🎉 no goals
  -/


instance : InvolutiveNeg (Weight K H L) where
  neg α := ⟨-α, by
    /-
      R : Type u_1
      K : Type u_2
      L : Type u_3
      inst✝⁸ : CommRing R
      inst✝⁷ : LieRing L
      inst✝⁶ : LieAlgebra R L
      inst✝⁵ : Field K
      inst✝⁴ : LieAlgebra K L
      inst✝³ : FiniteDimensional K L
      inst✝² : LieAlgebra.IsKilling K L
      H : LieSubalgebra K L
      inst✝¹ : H.IsCartanSubalgebra
      inst✝ : LieModule.IsTriangularizable K (Subtype fun x => Membership.mem H x) L
      α✝ α : LieModule.Weight K (Subtype fun x => Membership.mem H x) L
      ⊢ Ne (LieModule.genWeightSpace L (Neg.neg ⇑α)) Bot.bot
    -/
    by_cases hα : α.IsZero
      /-
        case pos
        R : Type u_1
        K : Type u_2
        L : Type u_3
        inst✝⁸ : CommRing R
        inst✝⁷ : LieRing L
        inst✝⁶ : LieAlgebra R L
        inst✝⁵ : Field K
        inst✝⁴ : LieAlgebra K L
        inst✝³ : FiniteDimensional K L
        inst✝² : LieAlgebra.IsKilling K L
        H : LieSubalgebra K L
        inst✝¹ : H.IsCartanSubalgebra
        inst✝ : LieModule.IsTriangularizable K (Subtype fun x => Membership.mem H x) L
        α✝ α : LieModule.Weight K (Subtype fun x => Membership.mem H x) L
        hα : α.IsZero
        ⊢ Ne (LieModule.genWeightSpace L (Neg.neg ⇑α)) Bot.bot
      -/
    · convert α.genWeightSpace_ne_bot; rw [hα, neg_zero]
                                       /-
                                         🎉 no goals
                                       -/
      /-
        case neg
        R : Type u_1
        K : Type u_2
        L : Type u_3
        inst✝⁸ : CommRing R
        inst✝⁷ : LieRing L
        inst✝⁶ : LieAlgebra R L
        inst✝⁵ : Field K
        inst✝⁴ : LieAlgebra K L
        inst✝³ : FiniteDimensional K L
        inst✝² : LieAlgebra.IsKilling K L
        H : LieSubalgebra K L
        inst✝¹ : H.IsCartanSubalgebra
        inst✝ : LieModule.IsTriangularizable K (Subtype fun x => Membership.mem H x) L
        α✝ α : LieModule.Weight K (Subtype fun x => Membership.mem H x) L
        hα : Not α.IsZero
        ⊢ Ne (LieModule.genWeightSpace L (Neg.neg ⇑α)) Bot.bot
      -/
    · intro e
      /-
        case neg
        R : Type u_1
        K : Type u_2
        L : Type u_3
        inst✝⁸ : CommRing R
        inst✝⁷ : LieRing L
        inst✝⁶ : LieAlgebra R L
        inst✝⁵ : Field K
        inst✝⁴ : LieAlgebra K L
        inst✝³ : FiniteDimensional K L
        inst✝² : LieAlgebra.IsKilling K L
        H : LieSubalgebra K L
        inst✝¹ : H.IsCartanSubalgebra
        inst✝ : LieModule.IsTriangularizable K (Subtype fun x => Membership.mem H x) L
        α✝ α : LieModule.Weight K (Subtype fun x => Membership.mem H x) L
        hα : Not α.IsZero
        e : Eq (LieModule.genWeightSpace L (Neg.neg ⇑α)) Bot.bot
        ⊢ False
      -/
      obtain ⟨x, hx, x_ne0⟩ := α.exists_ne_zero
      have := mem_ker_killingForm_of_mem_rootSpace_of_forall_rootSpace_neg K L H hx
        (fun y hy ↦ by rw [rootSpace, e] at hy; rw [hy, map_zero])
      /-
        case neg.intro.intro
        R : Type u_1
        K : Type u_2
        L : Type u_3
        inst✝⁸ : CommRing R
        inst✝⁷ : LieRing L
        inst✝⁶ : LieAlgebra R L
        inst✝⁵ : Field K
        inst✝⁴ : LieAlgebra K L
        inst✝³ : FiniteDimensional K L
        inst✝² : LieAlgebra.IsKilling K L
        H : LieSubalgebra K L
        inst✝¹ : H.IsCartanSubalgebra
        inst✝ : LieModule.IsTriangularizable K (Subtype fun x => Membership.mem H x) L
        α✝ α : LieModule.Weight K (Subtype fun x => Membership.mem H x) L
        hα : Not α.IsZero
        e : Eq (LieModule.genWeightSpace L (Neg.neg ⇑α)) Bot.bot
        x : L
        hx : Membership.mem (LieModule.genWeightSpace L ⇑α) x
        x_ne0 : Ne x 0
        this : Membership.mem (LinearMap.ker (killingForm K L)) x
        ⊢ False
      -/
      rw [ker_killingForm_eq_bot] at this
      /-
        case neg.intro.intro
        R : Type u_1
        K : Type u_2
        L : Type u_3
        inst✝⁸ : CommRing R
        inst✝⁷ : LieRing L
        inst✝⁶ : LieAlgebra R L
        inst✝⁵ : Field K
        inst✝⁴ : LieAlgebra K L
        inst✝³ : FiniteDimensional K L
        inst✝² : LieAlgebra.IsKilling K L
        H : LieSubalgebra K L
        inst✝¹ : H.IsCartanSubalgebra
        inst✝ : LieModule.IsTriangularizable K (Subtype fun x => Membership.mem H x) L
        α✝ α : LieModule.Weight K (Subtype fun x => Membership.mem H x) L
        hα : Not α.IsZero
        e : Eq (LieModule.genWeightSpace L (Neg.neg ⇑α)) Bot.bot
        x : L
        hx : Membership.mem (LieModule.genWeightSpace L ⇑α) x
        x_ne0 : Ne x 0
        this : Membership.mem Bot.bot x
        ⊢ False
      -/
      exact x_ne0 this⟩
      /-
        🎉 no goals
      -/
                  /-
                    R : Type u_1
                    K : Type u_2
                    L : Type u_3
                    inst✝⁸ : CommRing R
                    inst✝⁷ : LieRing L
                    inst✝⁶ : LieAlgebra R L
                    inst✝⁵ : Field K
                    inst✝⁴ : LieAlgebra K L
                    inst✝³ : FiniteDimensional K L
                    inst✝² : LieAlgebra.IsKilling K L
                    H : LieSubalgebra K L
                    inst✝¹ : H.IsCartanSubalgebra
                    inst✝ : LieModule.IsTriangularizable K (Subtype fun x => Membership.mem H x) L
                    α✝ α : LieModule.Weight K (Subtype fun x => Membership.mem H x) L
                    ⊢ Eq (Neg.neg (Neg.neg α)) α
                  -/
  neg_neg α := by ext; simp
                       /-
                         🎉 no goals
                       -/


@[simp] lemma coe_neg : ((-α : Weight K H L) : H → K) = -α := rfl


                                                    /-
                                                      K : Type u_2
                                                      L : Type u_3
                                                      inst✝⁶ : LieRing L
                                                      inst✝⁵ : Field K
                                                      inst✝⁴ : LieAlgebra K L
                                                      inst✝³ : FiniteDimensional K L
                                                      inst✝² : LieAlgebra.IsKilling K L
                                                      H : LieSubalgebra K L
                                                      inst✝¹ : H.IsCartanSubalgebra
                                                      inst✝ : LieModule.IsTriangularizable K (Subtype fun x => Membership.mem H x) L
                                                      α : LieModule.Weight K (Subtype fun x => Membership.mem H x) L
                                                      h : α.IsZero
                                                      ⊢ (Neg.neg α).IsZero
                                                    -/
lemma IsZero.neg (h : α.IsZero) : (-α).IsZero := by ext; rw [coe_neg, h, neg_zero]
                                                         /-
                                                           🎉 no goals
                                                         -/


@[simp] lemma isZero_neg : (-α).IsZero ↔ α.IsZero := ⟨fun h ↦ neg_neg α ▸ h.neg, fun h ↦ h.neg⟩


                                                                        /-
                                                                          K : Type u_2
                                                                          L : Type u_3
                                                                          inst✝⁶ : LieRing L
                                                                          inst✝⁵ : Field K
                                                                          inst✝⁴ : LieAlgebra K L
                                                                          inst✝³ : FiniteDimensional K L
                                                                          inst✝² : LieAlgebra.IsKilling K L
                                                                          H : LieSubalgebra K L
                                                                          inst✝¹ : H.IsCartanSubalgebra
                                                                          inst✝ : LieModule.IsTriangularizable K (Subtype fun x => Membership.mem H x) L
                                                                          α : LieModule.Weight K (Subtype fun x => Membership.mem H x) L
                                                                          h : α.IsNonZero
                                                                          e : (Neg.neg α).IsZero
                                                                          ⊢ α.IsZero
                                                                        -/
lemma IsNonZero.neg (h : α.IsNonZero) : (-α).IsNonZero := fun e ↦ h (by simpa using e.neg)
                                                                        /-
                                                                          🎉 no goals
                                                                        -/


@[simp] lemma isNonZero_neg {α : Weight K H L} : (-α).IsNonZero ↔ α.IsNonZero := isZero_neg.not


@[simp] lemma toLinear_neg {α : Weight K H L} : (-α).toLinear = -α.toLinear := rfl


@[simp]
lemma _root_.LieAlgebra.IsKilling.coroot_neg (α : Weight K H L) : coroot (-α) = -coroot α := by
  /-
    K : Type u_2
    L : Type u_3
    inst✝⁷ : LieRing L
    inst✝⁶ : Field K
    inst✝⁵ : LieAlgebra K L
    inst✝⁴ : FiniteDimensional K L
    inst✝³ : LieAlgebra.IsKilling K L
    H : LieSubalgebra K L
    inst✝² : H.IsCartanSubalgebra
    inst✝¹ : LieModule.IsTriangularizable K (Subtype fun x => Membership.mem H x) L
    inst✝ : CharZero K
    α : LieModule.Weight K (Subtype fun x => Membership.mem H x) L
    ⊢ Eq (LieAlgebra.IsKilling.coroot (Neg.neg α)) (Neg.neg (LieAlgebra.IsKilling. …
  -/
  simp [coroot]
  /-
    🎉 no goals
  -/


