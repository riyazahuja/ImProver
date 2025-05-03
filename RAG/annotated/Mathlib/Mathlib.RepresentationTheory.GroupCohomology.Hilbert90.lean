/-- Given `f : Aut_K(L) → Lˣ`, the sum `∑ f(φ) • φ` for `φ ∈ Aut_K(L)`, as a function `L → L`. -/
noncomputable def aux (f : (L ≃ₐ[K] L) → Lˣ) : L → L :=
  Finsupp.linearCombination L (fun φ : L ≃ₐ[K] L ↦ (φ : L → L))
    (Finsupp.equivFunOnFinite.symm (fun φ => (f φ : L)))


theorem aux_ne_zero (f : (L ≃ₐ[K] L) → Lˣ) : aux f ≠ 0 :=
/- the set `Aut_K(L)` is linearly independent in the `L`-vector space `L → L`, by Dedekind's
linear independence of characters -/
  have : LinearIndependent L (fun (f : L ≃ₐ[K] L) => (f : L → L)) :=
    LinearIndependent.comp (ι' := L ≃ₐ[K] L)
      (linearIndependent_monoidHom L L) (fun f => f)
                       /-
                         K : Type u_1
                         L : Type u_2
                         inst✝³ : Field K
                         inst✝² : Field L
                         inst✝¹ : Algebra K L
                         inst✝ : FiniteDimensional K L
                         f : AlgEquiv K L L → Units L
                         x y : AlgEquiv K L L
                         h : Eq ((fun f => ↑f) x) ((fun f => ↑f) y)
                         ⊢ Eq x y
                       -/
      (fun x y h => by ext; exact DFunLike.ext_iff.1 h _)
                            /-
                              🎉 no goals
                            -/
  have h := linearIndependent_iff.1 this
    (Finsupp.equivFunOnFinite.symm (fun φ => (f φ : L)))
  fun H => Units.ne_zero (f 1) (DFunLike.ext_iff.1 (h H) 1)


/-- Noether's generalization of Hilbert's Theorem 90: given a finite extension of fields and a
function `f : Aut_K(L) → Lˣ` satisfying `f(gh) = g(f(h)) * f(g)` for all `g, h : Aut_K(L)`, there
exists `β : Lˣ` such that `g(β)/β = f(g)` for all `g : Aut_K(L).` -/
theorem isMulOneCoboundary_of_isMulOneCocycle_of_aut_to_units
    (f : (L ≃ₐ[K] L) → Lˣ) (hf : IsMulOneCocycle f) :
    IsMulOneCoboundary f := by
/- Let `z : L` be such that `∑ f(h) * h(z) ≠ 0`, for `h ∈ Aut_K(L)` -/
  obtain ⟨z, hz⟩ : ∃ z, aux f z ≠ 0 :=
    not_forall.1 (fun H => aux_ne_zero f <| funext <| fun x => H x)
  /-
    case intro
    K : Type u_1
    L : Type u_2
    inst✝³ : Field K
    inst✝² : Field L
    inst✝¹ : Algebra K L
    inst✝ : FiniteDimensional K L
    f : AlgEquiv K L L → Units L
    hf : groupCohomology.IsMulOneCocycle f
    z : L
    hz : Ne (groupCohomology.Hilbert90.aux f z) 0
    ⊢ groupCohomology.IsMulOneCoboundary f
  -/
  have : aux f z = ∑ h, f h * h z := by simp [aux, Finsupp.linearCombination, Finsupp.sum_fintype]
/- Let `β = (∑ f(h) * h(z))⁻¹.` -/
  /-
    case intro
    K : Type u_1
    L : Type u_2
    inst✝³ : Field K
    inst✝² : Field L
    inst✝¹ : Algebra K L
    inst✝ : FiniteDimensional K L
    f : AlgEquiv K L L → Units L
    hf : groupCohomology.IsMulOneCocycle f
    z : L
    hz : Ne (groupCohomology.Hilbert90.aux f z) 0
    this : Eq (groupCohomology.Hilbert90.aux f z) (Finset.univ.sum fun h => HMul.h …
    ⊢ groupCohomology.IsMulOneCoboundary f
  -/
  use (Units.mk0 (aux f z) hz)⁻¹
  /-
    case h
    K : Type u_1
    L : Type u_2
    inst✝³ : Field K
    inst✝² : Field L
    inst✝¹ : Algebra K L
    inst✝ : FiniteDimensional K L
    f : AlgEquiv K L L → Units L
    hf : groupCohomology.IsMulOneCocycle f
    z : L
    hz : Ne (groupCohomology.Hilbert90.aux f z) 0
    this : Eq (groupCohomology.Hilbert90.aux f z) (Finset.univ.sum fun h => HMul.h …
    ⊢ ∀ (g : AlgEquiv K L L), Eq (HDiv.hDiv (HSMul.hSMul g (Inv.inv (Units.mk0 (gr …
  -/
  intro g
/- Then the equality follows from the hypothesis that `f` is a 1-cocycle. -/
  simp only [IsMulOneCocycle, IsMulOneCoboundary, AlgEquiv.smul_units_def,
    map_inv, div_inv_eq_mul, inv_mul_eq_iff_eq_mul, Units.ext_iff, this,
    Units.val_mul, Units.coe_map, Units.val_mk0, MonoidHom.coe_coe] at hf ⊢
  /-
    case h
    K : Type u_1
    L : Type u_2
    inst✝³ : Field K
    inst✝² : Field L
    inst✝¹ : Algebra K L
    inst✝ : FiniteDimensional K L
    f : AlgEquiv K L L → Units L
    z : L
    hz : Ne (groupCohomology.Hilbert90.aux f z) 0
    this : Eq (groupCohomology.Hilbert90.aux f z) (Finset.univ.sum fun h => HMul.h …
    g : AlgEquiv K L L
    hf : ∀ (g h : AlgEquiv K L L), Eq (↑(f (HMul.hMul g h))) (HMul.hMul (g ↑(f h)) …
    ⊢ Eq (Finset.univ.sum fun h => HMul.hMul (↑(f h)) (h z)) (HMul.hMul (g (Finset …
  -/
  simp_rw [map_sum, map_mul, Finset.sum_mul, mul_assoc, mul_comm _ (f _ : L), ← mul_assoc, ← hf g]
  exact eq_comm.1 (Fintype.sum_bijective (fun i => g * i)
    (Group.mulLeft_bijective g) _ _ (fun i => rfl))


/-- Noether's generalization of Hilbert's Theorem 90: given a finite extension of fields `L/K`, the
first group cohomology `H¹(Aut_K(L), Lˣ)` is trivial. -/
noncomputable instance H1ofAutOnUnitsUnique : Unique (H1 (Rep.ofAlgebraAutOnUnits K L)) where
  default := 0
  uniq := fun a => Quotient.inductionOn' a fun x => (Submodule.Quotient.mk_eq_zero _).2 <| by
    /-
      K L : Type
      inst✝³ : Field K
      inst✝² : Field L
      inst✝¹ : Algebra K L
      inst✝ : FiniteDimensional K L
      a : groupCohomology.H1 (Rep.ofAlgebraAutOnUnits K L)
      x : Subtype fun x => Membership.mem (groupCohomology.oneCocycles (Rep.ofAlgebr …
      ⊢ Membership.mem (groupCohomology.oneCoboundaries (Rep.ofAlgebraAutOnUnits K L …
    -/
    refine (oneCoboundariesOfIsMulOneCoboundary ?_).2
    rcases isMulOneCoboundary_of_isMulOneCocycle_of_aut_to_units x.1
      (isMulOneCocycle_of_oneCocycles x) with ⟨β, hβ⟩
    /-
      case intro
      K L : Type
      inst✝³ : Field K
      inst✝² : Field L
      inst✝¹ : Algebra K L
      inst✝ : FiniteDimensional K L
      a : groupCohomology.H1 (Rep.ofAlgebraAutOnUnits K L)
      x : Subtype fun x => Membership.mem (groupCohomology.oneCocycles (Rep.ofAlgebr …
      β : Units L
      hβ : ∀ (g : AlgEquiv K L L), Eq (HDiv.hDiv (HSMul.hSMul g β) β) (↑x g)
      ⊢ groupCohomology.IsMulOneCoboundary x.1
    -/
    use β
    /-
      🎉 no goals
    -/


