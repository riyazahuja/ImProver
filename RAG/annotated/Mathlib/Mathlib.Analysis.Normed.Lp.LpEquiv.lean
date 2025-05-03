/-- When `α` is `Finite`, every `f : PreLp E p` satisfies `Memℓp f p`. -/
theorem Memℓp.all (f : ∀ i, E i) : Memℓp f p := by
  /-
    α : Type u_1
    E : α → Type u_2
    inst✝¹ : (i : α) → NormedAddCommGroup (E i)
    p : ENNReal
    inst✝ : Finite α
    f : (i : α) → E i
    ⊢ Memℓp f p
  -/
  rcases p.trichotomy with (rfl | rfl | _h)
    /-
      case inl
      α : Type u_1
      E : α → Type u_2
      inst✝¹ : (i : α) → NormedAddCommGroup (E i)
      inst✝ : Finite α
      f : (i : α) → E i
      ⊢ Memℓp f 0
    -/
  · exact memℓp_zero_iff.mpr { i : α | f i ≠ 0 }.toFinite
    /-
      🎉 no goals
    -/
    /-
      case inr.inl
      α : Type u_1
      E : α → Type u_2
      inst✝¹ : (i : α) → NormedAddCommGroup (E i)
      inst✝ : Finite α
      f : (i : α) → E i
      ⊢ Memℓp f Top.top
    -/
  · exact memℓp_infty_iff.mpr (Set.Finite.bddAbove (Set.range fun i : α ↦ ‖f i‖).toFinite)
    /-
      🎉 no goals
    -/
    /-
      case inr.inr
      α : Type u_1
      E : α → Type u_2
      inst✝¹ : (i : α) → NormedAddCommGroup (E i)
      p : ENNReal
      inst✝ : Finite α
      f : (i : α) → E i
      _h : LT.lt 0 p.toReal
      ⊢ Memℓp f p
    -/
  · cases nonempty_fintype α; exact memℓp_gen ⟨Finset.univ.sum _, hasSum_fintype _⟩
                              /-
                                🎉 no goals
                              -/


/-- The canonical `Equiv` between `lp E p ≃ PiLp p E` when `E : α → Type u` with `[Finite α]`. -/
def Equiv.lpPiLp : lp E p ≃ PiLp p E where
  toFun f := ⇑f
  invFun f := ⟨f, Memℓp.all f⟩
  left_inv _f := rfl
  right_inv _f := rfl


theorem coe_equiv_lpPiLp (f : lp E p) : Equiv.lpPiLp f = ⇑f :=
  rfl


theorem coe_equiv_lpPiLp_symm (f : PiLp p E) : (Equiv.lpPiLp.symm f : ∀ i, E i) = f :=
  rfl


/-- The canonical `AddEquiv` between `lp E p` and `PiLp p E` when `E : α → Type u` with
`[Fintype α]`. -/
def AddEquiv.lpPiLp : lp E p ≃+ PiLp p E :=
  { Equiv.lpPiLp with map_add' := fun _f _g ↦ rfl }


theorem coe_addEquiv_lpPiLp (f : lp E p) : AddEquiv.lpPiLp f = ⇑f :=
  rfl


theorem coe_addEquiv_lpPiLp_symm (f : PiLp p E) :
    (AddEquiv.lpPiLp.symm f : ∀ i, E i) = f :=
  rfl


theorem equiv_lpPiLp_norm [Fintype α] (f : lp E p) : ‖Equiv.lpPiLp f‖ = ‖f‖ := by
  /-
    α : Type u_1
    E : α → Type u_2
    inst✝¹ : (i : α) → NormedAddCommGroup (E i)
    p : ENNReal
    inst✝ : Fintype α
    f : Subtype fun x => Membership.mem (lp E p) x
    ⊢ Eq (Norm.norm (Equiv.lpPiLp f)) (Norm.norm f)
  -/
  rcases p.trichotomy with (rfl | rfl | h)
    /-
      case inl
      α : Type u_1
      E : α → Type u_2
      inst✝¹ : (i : α) → NormedAddCommGroup (E i)
      inst✝ : Fintype α
      f : Subtype fun x => Membership.mem (lp E 0) x
      ⊢ Eq (Norm.norm (Equiv.lpPiLp f)) (Norm.norm f)
    -/
  · simp [Equiv.lpPiLp, PiLp.norm_eq_card, lp.norm_eq_card_dsupport]
    /-
      🎉 no goals
    -/
    /-
      case inr.inl
      α : Type u_1
      E : α → Type u_2
      inst✝¹ : (i : α) → NormedAddCommGroup (E i)
      inst✝ : Fintype α
      f : Subtype fun x => Membership.mem (lp E Top.top) x
      ⊢ Eq (Norm.norm (Equiv.lpPiLp f)) (Norm.norm f)
    -/
  · rw [PiLp.norm_eq_ciSup, lp.norm_eq_ciSup]; rfl
                                               /-
                                                 🎉 no goals
                                               -/
    /-
      case inr.inr
      α : Type u_1
      E : α → Type u_2
      inst✝¹ : (i : α) → NormedAddCommGroup (E i)
      p : ENNReal
      inst✝ : Fintype α
      f : Subtype fun x => Membership.mem (lp E p) x
      h : LT.lt 0 p.toReal
      ⊢ Eq (Norm.norm (Equiv.lpPiLp f)) (Norm.norm f)
    -/
  · rw [PiLp.norm_eq_sum h, lp.norm_eq_tsum_rpow h, tsum_fintype]; rfl
                                                                   /-
                                                                     🎉 no goals
                                                                   -/


/-- The canonical `LinearIsometryEquiv` between `lp E p` and `PiLp p E` when `E : α → Type u`
with `[Fintype α]` and `[Fact (1 ≤ p)]`. -/
noncomputable def lpPiLpₗᵢ [Fact (1 ≤ p)] : lp E p ≃ₗᵢ[𝕜] PiLp p E :=
  { AddEquiv.lpPiLp with
    map_smul' := fun _k _f ↦ rfl
    norm_map' := equiv_lpPiLp_norm }


theorem coe_lpPiLpₗᵢ [Fact (1 ≤ p)] (f : lp E p) : (lpPiLpₗᵢ E 𝕜 f : ∀ i, E i) = ⇑f :=
  rfl


theorem coe_lpPiLpₗᵢ_symm [Fact (1 ≤ p)] (f : PiLp p E) :
    ((lpPiLpₗᵢ E 𝕜).symm f : ∀ i, E i) = f :=
  rfl


/-- The canonical map between `lp (fun _ : α ↦ E) ∞` and `α →ᵇ E` as an `AddEquiv`. -/
noncomputable def AddEquiv.lpBCF : lp (fun _ : α ↦ E) ∞ ≃+ (α →ᵇ E) where
  toFun f := ofNormedAddCommGroupDiscrete f ‖f‖ <| le_ciSup (memℓp_infty_iff.mp f.prop)
  invFun f := ⟨⇑f, f.bddAbove_range_norm_comp⟩
  left_inv _f := lp.ext rfl
  right_inv _f := rfl
  map_add' _f _g := rfl


@[deprecated (since := "2024-03-16")] alias AddEquiv.lpBcf := AddEquiv.lpBCF


theorem coe_addEquiv_lpBCF (f : lp (fun _ : α ↦ E) ∞) : (AddEquiv.lpBCF f : α → E) = f :=
  rfl


theorem coe_addEquiv_lpBCF_symm (f : α →ᵇ E) : (AddEquiv.lpBCF.symm f : α → E) = f :=
  rfl


/-- The canonical map between `lp (fun _ : α ↦ E) ∞` and `α →ᵇ E` as a `LinearIsometryEquiv`. -/
noncomputable def lpBCFₗᵢ : lp (fun _ : α ↦ E) ∞ ≃ₗᵢ[𝕜] α →ᵇ E :=
  { AddEquiv.lpBCF with
    map_smul' := fun _ _ ↦ rfl
                            /-
                              α : Type u_1
                              E : Type u_2
                              R : Type u_3
                              A : Type u_4
                              𝕜 : Type u_5
                              inst✝⁸ : TopologicalSpace α
                              inst✝⁷ : DiscreteTopology α
                              inst✝⁶ : NormedRing A
                              inst✝⁵ : NormOneClass A
                              inst✝⁴ : NontriviallyNormedField 𝕜
                              inst✝³ : NormedAlgebra 𝕜 A
                              inst✝² : NormedAddCommGroup E
                              inst✝¹ : NormedSpace 𝕜 E
                              inst✝ : NonUnitalNormedRing R
                              f : Subtype fun x => Membership.mem (lp (fun x => E) Top.top) x
                              ⊢ Eq (Norm.norm ({ toFun := __src✝.toFun, map_add' := ⋯, map_smul' := ⋯, invFu …
                            -/
    norm_map' := fun f ↦ by simp only [norm_eq_iSup_norm, lp.norm_eq_ciSup]; rfl }
                                                                             /-
                                                                               🎉 no goals
                                                                             -/


@[deprecated (since := "2024-03-16")] alias lpBcfₗᵢ := lpBCFₗᵢ


theorem coe_lpBCFₗᵢ (f : lp (fun _ : α ↦ E) ∞) : (lpBCFₗᵢ E 𝕜 f : α → E) = f :=
  rfl


theorem coe_lpBCFₗᵢ_symm (f : α →ᵇ E) : ((lpBCFₗᵢ E 𝕜).symm f : α → E) = f :=
  rfl


/-- The canonical map between `lp (fun _ : α ↦ R) ∞` and `α →ᵇ R` as a `RingEquiv`. -/
noncomputable def RingEquiv.lpBCF : lp (fun _ : α ↦ R) ∞ ≃+* (α →ᵇ R) :=
  { @AddEquiv.lpBCF _ R _ _ _ with
    map_mul' := fun _f _g => rfl }


@[deprecated (since := "2024-03-16")] alias RingEquiv.lpBcf := RingEquiv.lpBCF


theorem coe_ringEquiv_lpBCF (f : lp (fun _ : α ↦ R) ∞) : (RingEquiv.lpBCF R f : α → R) = f :=
  rfl


theorem coe_ringEquiv_lpBCF_symm (f : α →ᵇ R) : ((RingEquiv.lpBCF R).symm f : α → R) = f :=
  rfl


/-- The canonical map between `lp (fun _ : α ↦ A) ∞` and `α →ᵇ A` as an `AlgEquiv`. -/
noncomputable def AlgEquiv.lpBCF : lp (fun _ : α ↦ A) ∞ ≃ₐ[𝕜] α →ᵇ A :=
  { RingEquiv.lpBCF A with commutes' := fun _k ↦ rfl }


@[deprecated (since := "2024-03-16")] alias AlgEquiv.lpBcf := AlgEquiv.lpBCF


theorem coe_algEquiv_lpBCF (f : lp (fun _ : α ↦ A) ∞) : (AlgEquiv.lpBCF α A 𝕜 f : α → A) = f :=
  rfl


theorem coe_algEquiv_lpBCF_symm (f : α →ᵇ A) : ((AlgEquiv.lpBCF α A 𝕜).symm f : α → A) = f :=
  rfl


