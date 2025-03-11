/-- The linear equivalence between `(α ⊕ β) →₀ M` and `(α →₀ M) × (β →₀ M)`.

This is the `LinearEquiv` version of `Finsupp.sumFinsuppEquivProdFinsupp`. -/
@[simps apply symm_apply]
def sumFinsuppLEquivProdFinsupp {α β : Type*} : (α ⊕ β →₀ M) ≃ₗ[R] (α →₀ M) × (β →₀ M) :=
  { sumFinsuppAddEquivProdFinsupp with
    map_smul' := by
      /-
        α✝ : Type u_1
        M : Type u_2
        N : Type u_3
        P : Type u_4
        R : Type u_5
        S : Type u_6
        inst✝⁷ : Semiring R
        inst✝⁶ : Semiring S
        inst✝⁵ : AddCommMonoid M
        inst✝⁴ : Module R M
        inst✝³ : AddCommMonoid N
        inst✝² : Module R N
        inst✝¹ : AddCommMonoid P
        inst✝ : Module R P
        α : Type u_7
        β : Type u_8
        ⊢ ∀ (m : R) (x : Finsupp (Sum α β) M), Eq ({ toFun := __src✝.toFun, map_add' : …
      -/
      intros
      /-
        α✝ : Type u_1
        M : Type u_2
        N : Type u_3
        P : Type u_4
        R : Type u_5
        S : Type u_6
        inst✝⁷ : Semiring R
        inst✝⁶ : Semiring S
        inst✝⁵ : AddCommMonoid M
        inst✝⁴ : Module R M
        inst✝³ : AddCommMonoid N
        inst✝² : Module R N
        inst✝¹ : AddCommMonoid P
        inst✝ : Module R P
        α : Type u_7
        β : Type u_8
        m✝ : R
        x✝ : Finsupp (Sum α β) M
        ⊢ Eq ({ toFun := __src✝.toFun, map_add' := ⋯ }.toFun (HSMul.hSMul m✝ x✝)) (HSM …
      -/
      ext <;>
        -- Porting note: `add_equiv.to_fun_eq_coe` →
        --               `Equiv.toFun_as_coe` & `AddEquiv.toEquiv_eq_coe` & `AddEquiv.coe_toEquiv`
        simp only [Equiv.toFun_as_coe, AddEquiv.toEquiv_eq_coe, AddEquiv.coe_toEquiv, Prod.smul_fst,
          Prod.smul_snd, smul_apply,
          snd_sumFinsuppAddEquivProdFinsupp, fst_sumFinsuppAddEquivProdFinsupp,
          RingHom.id_apply] }


theorem fst_sumFinsuppLEquivProdFinsupp {α β : Type*} (f : α ⊕ β →₀ M) (x : α) :
    (sumFinsuppLEquivProdFinsupp R f).1 x = f (Sum.inl x) :=
  rfl


theorem snd_sumFinsuppLEquivProdFinsupp {α β : Type*} (f : α ⊕ β →₀ M) (y : β) :
    (sumFinsuppLEquivProdFinsupp R f).2 y = f (Sum.inr y) :=
  rfl


theorem sumFinsuppLEquivProdFinsupp_symm_inl {α β : Type*} (fg : (α →₀ M) × (β →₀ M)) (x : α) :
    ((sumFinsuppLEquivProdFinsupp R).symm fg) (Sum.inl x) = fg.1 x :=
  rfl


theorem sumFinsuppLEquivProdFinsupp_symm_inr {α β : Type*} (fg : (α →₀ M) × (β →₀ M)) (y : β) :
    ((sumFinsuppLEquivProdFinsupp R).symm fg) (Sum.inr y) = fg.2 y :=
  rfl


/-- On a `Fintype η`, `Finsupp.split` is a linear equivalence between
`(Σ (j : η), ιs j) →₀ M` and `(j : η) → (ιs j →₀ M)`.

This is the `LinearEquiv` version of `Finsupp.sigmaFinsuppAddEquivPiFinsupp`. -/
noncomputable def sigmaFinsuppLEquivPiFinsupp {M : Type*} {ιs : η → Type*} [AddCommMonoid M]
    [Module R M] : ((Σ j, ιs j) →₀ M) ≃ₗ[R] (j : _) → (ιs j →₀ M) :=
  -- Porting note: `ιs` should be specified.
  { sigmaFinsuppAddEquivPiFinsupp (ιs := ιs) with
    map_smul' := fun c f => by
      /-
        α : Type u_1
        M✝ : Type u_2
        N : Type u_3
        P : Type u_4
        R : Type u_5
        S : Type u_6
        inst✝¹¹ : Semiring R
        inst✝¹⁰ : Semiring S
        inst✝⁹ : AddCommMonoid M✝
        inst✝⁸ : Module R M✝
        inst✝⁷ : AddCommMonoid N
        inst✝⁶ : Module R N
        inst✝⁵ : AddCommMonoid P
        inst✝⁴ : Module R P
        η : Type u_7
        inst✝³ : Fintype η
        ιs✝ : η → Type u_8
        inst✝² : Zero α
        M : Type u_9
        ιs : η → Type u_10
        inst✝¹ : AddCommMonoid M
        inst✝ : Module R M
        c : R
        f : Finsupp (Sigma fun j => ιs j) M
        ⊢ Eq ({ toFun := __src✝.toFun, map_add' := ⋯ }.toFun (HSMul.hSMul c f)) (HSMul …
      -/
      ext
      /-
        case h.h
        α : Type u_1
        M✝ : Type u_2
        N : Type u_3
        P : Type u_4
        R : Type u_5
        S : Type u_6
        inst✝¹¹ : Semiring R
        inst✝¹⁰ : Semiring S
        inst✝⁹ : AddCommMonoid M✝
        inst✝⁸ : Module R M✝
        inst✝⁷ : AddCommMonoid N
        inst✝⁶ : Module R N
        inst✝⁵ : AddCommMonoid P
        inst✝⁴ : Module R P
        η : Type u_7
        inst✝³ : Fintype η
        ιs✝ : η → Type u_8
        inst✝² : Zero α
        M : Type u_9
        ιs : η → Type u_10
        inst✝¹ : AddCommMonoid M
        inst✝ : Module R M
        c : R
        f : Finsupp (Sigma fun j => ιs j) M
        x✝ : η
        a✝ : ιs x✝
        ⊢ Eq (({ toFun := __src✝.toFun, map_add' := ⋯ }.toFun (HSMul.hSMul c f) x✝) a✝ …
      -/
      simp }
      /-
        🎉 no goals
      -/


@[simp]
theorem sigmaFinsuppLEquivPiFinsupp_apply {M : Type*} {ιs : η → Type*} [AddCommMonoid M]
    [Module R M] (f : (Σj, ιs j) →₀ M) (j i) : sigmaFinsuppLEquivPiFinsupp R f j i = f ⟨j, i⟩ :=
  rfl


@[simp]
theorem sigmaFinsuppLEquivPiFinsupp_symm_apply {M : Type*} {ιs : η → Type*} [AddCommMonoid M]
    [Module R M] (f : (j : _) → (ιs j →₀ M)) (ji) :
    (Finsupp.sigmaFinsuppLEquivPiFinsupp R).symm f ji = f ji.1 ji.2 :=
  rfl


