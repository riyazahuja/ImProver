/-- An isometric equivalence between two quadratic spaces `M₁, Q₁` and `M₂, Q₂` over a ring `R`,
is a linear equivalence between `M₁` and `M₂` that commutes with the quadratic forms. -/
-- Porting note (https://github.com/leanprover-community/mathlib4/issues/5171): linter not ported yet @[nolint has_nonempty_instance]
structure IsometryEquiv (Q₁ : QuadraticMap R M₁ N) (Q₂ : QuadraticMap R M₂ N)
    extends M₁ ≃ₗ[R] M₂ where
  map_app' : ∀ m, Q₂ (toFun m) = Q₁ m


/-- Two quadratic forms over a ring `R` are equivalent
if there exists an isometric equivalence between them:
a linear equivalence that transforms one quadratic form into the other. -/
def Equivalent (Q₁ : QuadraticMap R M₁ N) (Q₂ : QuadraticMap R M₂ N) : Prop :=
  Nonempty (Q₁.IsometryEquiv Q₂)


instance : EquivLike (Q₁.IsometryEquiv Q₂) M₁ M₂ where
  coe f := f.toLinearEquiv
  inv f := f.toLinearEquiv.symm
  left_inv f := f.toLinearEquiv.left_inv
  right_inv f := f.toLinearEquiv.right_inv
                           /-
                             ι : Type u_1
                             R : Type u_2
                             K : Type u_3
                             M : Type u_4
                             M₁ : Type u_5
                             M₂ : Type u_6
                             M₃ : Type u_7
                             V : Type u_8
                             N : Type u_9
                             inst✝¹⁰ : CommSemiring R
                             inst✝⁹ : AddCommMonoid M
                             inst✝⁸ : AddCommMonoid M₁
                             inst✝⁷ : AddCommMonoid M₂
                             inst✝⁶ : AddCommMonoid M₃
                             inst✝⁵ : AddCommMonoid N
                             inst✝⁴ : Module R M
                             inst✝³ : Module R M₁
                             inst✝² : Module R M₂
                             inst✝¹ : Module R M₃
                             inst✝ : Module R N
                             Q₁ : QuadraticMap R M₁ N
                             Q₂ : QuadraticMap R M₂ N
                             Q₃ : QuadraticMap R M₃ N
                             f g : Q₁.IsometryEquiv Q₂
                             ⊢ Eq ((fun f => ⇑f.toLinearEquiv) f) ((fun f => ⇑f.toLinearEquiv) g) → Eq ((fu …
                           -/
  coe_injective' f g := by cases f; cases g; simp (config := {contextual := true})
                                             /-
                                               🎉 no goals
                                             -/


instance : LinearEquivClass (Q₁.IsometryEquiv Q₂) R M₁ M₂ where
  map_add f := map_add f.toLinearEquiv
  map_smulₛₗ f := map_smulₛₗ f.toLinearEquiv

-- Porting note: was `Coe`

instance : CoeOut (Q₁.IsometryEquiv Q₂) (M₁ ≃ₗ[R] M₂) :=
  ⟨IsometryEquiv.toLinearEquiv⟩

-- Porting note: syntaut


@[simp]
theorem coe_toLinearEquiv (f : Q₁.IsometryEquiv Q₂) : ⇑(f : M₁ ≃ₗ[R] M₂) = f :=
  rfl


@[simp]
theorem map_app (f : Q₁.IsometryEquiv Q₂) (m : M₁) : Q₂ (f m) = Q₁ m :=
  f.map_app' m


/-- The identity isometric equivalence between a quadratic form and itself. -/
@[refl]
def refl (Q : QuadraticMap R M N) : Q.IsometryEquiv Q :=
  { LinearEquiv.refl R M with map_app' := fun _ => rfl }


/-- The inverse isometric equivalence of an isometric equivalence between two quadratic forms. -/
@[symm]
def symm (f : Q₁.IsometryEquiv Q₂) : Q₂.IsometryEquiv Q₁ :=
  { (f : M₁ ≃ₗ[R] M₂).symm with
                   /-
                     ι : Type u_1
                     R : Type u_2
                     K : Type u_3
                     M : Type u_4
                     M₁ : Type u_5
                     M₂ : Type u_6
                     M₃ : Type u_7
                     V : Type u_8
                     N : Type u_9
                     inst✝¹⁰ : CommSemiring R
                     inst✝⁹ : AddCommMonoid M
                     inst✝⁸ : AddCommMonoid M₁
                     inst✝⁷ : AddCommMonoid M₂
                     inst✝⁶ : AddCommMonoid M₃
                     inst✝⁵ : AddCommMonoid N
                     inst✝⁴ : Module R M
                     inst✝³ : Module R M₁
                     inst✝² : Module R M₂
                     inst✝¹ : Module R M₃
                     inst✝ : Module R N
                     Q₁ : QuadraticMap R M₁ N
                     Q₂ : QuadraticMap R M₂ N
                     Q₃ : QuadraticMap R M₃ N
                     f : Q₁.IsometryEquiv Q₂
                     ⊢ ∀ (m : M₂), Eq (Q₁ ((↑__src✝).toFun m)) (Q₂ m)
                   -/
    map_app' := by intro m; rw [← f.map_app]; congr; exact f.toLinearEquiv.apply_symm_apply m }
                                                     /-
                                                       🎉 no goals
                                                     -/


/-- The composition of two isometric equivalences between quadratic forms. -/
@[trans]
def trans (f : Q₁.IsometryEquiv Q₂) (g : Q₂.IsometryEquiv Q₃) : Q₁.IsometryEquiv Q₃ :=
  { (f : M₁ ≃ₗ[R] M₂).trans (g : M₂ ≃ₗ[R] M₃) with
                   /-
                     ι : Type u_1
                     R : Type u_2
                     K : Type u_3
                     M : Type u_4
                     M₁ : Type u_5
                     M₂ : Type u_6
                     M₃ : Type u_7
                     V : Type u_8
                     N : Type u_9
                     inst✝¹⁰ : CommSemiring R
                     inst✝⁹ : AddCommMonoid M
                     inst✝⁸ : AddCommMonoid M₁
                     inst✝⁷ : AddCommMonoid M₂
                     inst✝⁶ : AddCommMonoid M₃
                     inst✝⁵ : AddCommMonoid N
                     inst✝⁴ : Module R M
                     inst✝³ : Module R M₁
                     inst✝² : Module R M₂
                     inst✝¹ : Module R M₃
                     inst✝ : Module R N
                     Q₁ : QuadraticMap R M₁ N
                     Q₂ : QuadraticMap R M₂ N
                     Q₃ : QuadraticMap R M₃ N
                     f : Q₁.IsometryEquiv Q₂
                     g : Q₂.IsometryEquiv Q₃
                     ⊢ ∀ (m : M₁), Eq (Q₃ ((↑__src✝).toFun m)) (Q₁ m)
                   -/
    map_app' := by intro m; rw [← f.map_app, ← g.map_app]; rfl }
                                                           /-
                                                             🎉 no goals
                                                           -/


/-- Isometric equivalences are isometric maps -/
@[simps]
def toIsometry (g : Q₁.IsometryEquiv Q₂) : Q₁ →qᵢ Q₂ where
  toFun x := g x
  __ := g


@[refl]
theorem refl (Q : QuadraticMap R M N) : Q.Equivalent Q :=
  ⟨IsometryEquiv.refl Q⟩


@[symm]
theorem symm (h : Q₁.Equivalent Q₂) : Q₂.Equivalent Q₁ :=
  h.elim fun f => ⟨f.symm⟩


@[trans]
theorem trans (h : Q₁.Equivalent Q₂) (h' : Q₂.Equivalent Q₃) : Q₁.Equivalent Q₃ :=
  h'.elim <| h.elim fun f g => ⟨f.trans g⟩


/-- A quadratic form composed with a `LinearEquiv` is isometric to itself. -/
def isometryEquivOfCompLinearEquiv (Q : QuadraticMap R M N) (f : M₁ ≃ₗ[R] M) :
    Q.IsometryEquiv (Q.comp (f : M₁ →ₗ[R] M)) :=
  { f.symm with
    map_app' := by
      /-
        ι : Type u_1
        R : Type u_2
        K : Type u_3
        M : Type u_4
        M₁ : Type u_5
        M₂ : Type u_6
        M₃ : Type u_7
        V : Type u_8
        N : Type u_9
        inst✝¹⁰ : CommSemiring R
        inst✝⁹ : AddCommMonoid M
        inst✝⁸ : AddCommMonoid M₁
        inst✝⁷ : AddCommMonoid M₂
        inst✝⁶ : AddCommMonoid M₃
        inst✝⁵ : AddCommMonoid N
        inst✝⁴ : Module R M
        inst✝³ : Module R M₁
        inst✝² : Module R M₂
        inst✝¹ : Module R M₃
        inst✝ : Module R N
        Q : QuadraticMap R M N
        f : LinearEquiv (RingHom.id R) M₁ M
        ⊢ ∀ (m : M), Eq ((Q.comp ↑f) ((↑__src✝).toFun m)) (Q m)
      -/
      intro
      simp only [comp_apply, LinearEquiv.coe_coe, LinearEquiv.toFun_eq_coe,
        LinearEquiv.apply_symm_apply, f.apply_symm_apply] }


/-- A quadratic form is isometrically equivalent to its bases representations. -/
noncomputable def isometryEquivBasisRepr (Q : QuadraticMap R M N) (v : Basis ι R M) :
    IsometryEquiv Q (Q.basisRepr v) :=
  isometryEquivOfCompLinearEquiv Q v.equivFun.symm


/-- Given an orthogonal basis, a quadratic form is isometrically equivalent with a weighted sum of
squares. -/
noncomputable def isometryEquivWeightedSumSquares (Q : QuadraticForm K V)
    (v : Basis (Fin (Module.finrank K V)) K V)
    (hv₁ : (associated (R := K) Q).IsOrthoᵢ v) :
    Q.IsometryEquiv (weightedSumSquares K fun i => Q (v i)) := by
  /-
    ι : Type u_1
    R : Type u_2
    K : Type u_3
    M : Type u_4
    M₁ : Type u_5
    M₂ : Type u_6
    M₃ : Type u_7
    V : Type u_8
    N : Type u_9
    inst✝³ : Field K
    inst✝² : Invertible 2
    inst✝¹ : AddCommGroup V
    inst✝ : Module K V
    Q : QuadraticForm K V
    v : Basis (Fin (Module.finrank K V)) K V
    hv₁ : LinearMap.IsOrthoᵢ (QuadraticMap.associated Q) ⇑v
    ⊢ QuadraticMap.IsometryEquiv Q (QuadraticMap.weightedSumSquares K fun i => Q ( …
  -/
  let iso := Q.isometryEquivBasisRepr v
  /-
    ι : Type u_1
    R : Type u_2
    K : Type u_3
    M : Type u_4
    M₁ : Type u_5
    M₂ : Type u_6
    M₃ : Type u_7
    V : Type u_8
    N : Type u_9
    inst✝³ : Field K
    inst✝² : Invertible 2
    inst✝¹ : AddCommGroup V
    inst✝ : Module K V
    Q : QuadraticForm K V
    v : Basis (Fin (Module.finrank K V)) K V
    hv₁ : LinearMap.IsOrthoᵢ (QuadraticMap.associated Q) ⇑v
    iso : QuadraticMap.IsometryEquiv Q (QuadraticMap.basisRepr Q v) := QuadraticMa …
    ⊢ QuadraticMap.IsometryEquiv Q (QuadraticMap.weightedSumSquares K fun i => Q ( …
  -/
  refine ⟨iso, fun m => ?_⟩
  /-
    ι : Type u_1
    R : Type u_2
    K : Type u_3
    M : Type u_4
    M₁ : Type u_5
    M₂ : Type u_6
    M₃ : Type u_7
    V : Type u_8
    N : Type u_9
    inst✝³ : Field K
    inst✝² : Invertible 2
    inst✝¹ : AddCommGroup V
    inst✝ : Module K V
    Q : QuadraticForm K V
    v : Basis (Fin (Module.finrank K V)) K V
    hv₁ : LinearMap.IsOrthoᵢ (QuadraticMap.associated Q) ⇑v
    iso : QuadraticMap.IsometryEquiv Q (QuadraticMap.basisRepr Q v) := QuadraticMa …
    m : V
    ⊢ Eq ((QuadraticMap.weightedSumSquares K fun i => Q (v i)) ((↑iso.toLinearEqui …
  -/
  convert iso.map_app m
  /-
    case h.e'_2.h.e'_5
    ι : Type u_1
    R : Type u_2
    K : Type u_3
    M : Type u_4
    M₁ : Type u_5
    M₂ : Type u_6
    M₃ : Type u_7
    V : Type u_8
    N : Type u_9
    inst✝³ : Field K
    inst✝² : Invertible 2
    inst✝¹ : AddCommGroup V
    inst✝ : Module K V
    Q : QuadraticForm K V
    v : Basis (Fin (Module.finrank K V)) K V
    hv₁ : LinearMap.IsOrthoᵢ (QuadraticMap.associated Q) ⇑v
    iso : QuadraticMap.IsometryEquiv Q (QuadraticMap.basisRepr Q v) := QuadraticMa …
    m : V
    ⊢ Eq (QuadraticMap.weightedSumSquares K fun i => Q (v i)) (QuadraticMap.basisR …
  -/
  rw [basisRepr_eq_of_iIsOrtho _ _ hv₁]
  /-
    🎉 no goals
  -/


theorem equivalent_weightedSumSquares (Q : QuadraticForm K V) :
    ∃ w : Fin (Module.finrank K V) → K, Equivalent Q (weightedSumSquares K w) :=
  let ⟨v, hv₁⟩ := exists_orthogonal_basis (associated_isSymm _ Q)
  ⟨_, ⟨Q.isometryEquivWeightedSumSquares v hv₁⟩⟩


theorem equivalent_weightedSumSquares_units_of_nondegenerate' (Q : QuadraticForm K V)
    (hQ : (associated (R := K) Q).SeparatingLeft) :
    ∃ w : Fin (Module.finrank K V) → Kˣ, Equivalent Q (weightedSumSquares K w) := by
  /-
    K : Type u_3
    V : Type u_8
    inst✝⁴ : Field K
    inst✝³ : Invertible 2
    inst✝² : AddCommGroup V
    inst✝¹ : Module K V
    inst✝ : FiniteDimensional K V
    Q : QuadraticForm K V
    hQ : LinearMap.SeparatingLeft (QuadraticMap.associated Q)
    ⊢ Exists fun w => QuadraticMap.Equivalent Q (QuadraticMap.weightedSumSquares K …
  -/
  obtain ⟨v, hv₁⟩ := exists_orthogonal_basis (associated_isSymm K Q)
  /-
    case intro
    K : Type u_3
    V : Type u_8
    inst✝⁴ : Field K
    inst✝³ : Invertible 2
    inst✝² : AddCommGroup V
    inst✝¹ : Module K V
    inst✝ : FiniteDimensional K V
    Q : QuadraticForm K V
    hQ : LinearMap.SeparatingLeft (QuadraticMap.associated Q)
    v : Basis (Fin (Module.finrank K V)) K V
    hv₁ : LinearMap.IsOrthoᵢ ((QuadraticMap.associatedHom K) Q) ⇑v
    ⊢ Exists fun w => QuadraticMap.Equivalent Q (QuadraticMap.weightedSumSquares K …
  -/
  have hv₂ := hv₁.not_isOrtho_basis_self_of_separatingLeft hQ
  /-
    case intro
    K : Type u_3
    V : Type u_8
    inst✝⁴ : Field K
    inst✝³ : Invertible 2
    inst✝² : AddCommGroup V
    inst✝¹ : Module K V
    inst✝ : FiniteDimensional K V
    Q : QuadraticForm K V
    hQ : LinearMap.SeparatingLeft (QuadraticMap.associated Q)
    v : Basis (Fin (Module.finrank K V)) K V
    hv₁ : LinearMap.IsOrthoᵢ ((QuadraticMap.associatedHom K) Q) ⇑v
    hv₂ : ∀ (i : Fin (Module.finrank K V)), Not (LinearMap.IsOrtho ((QuadraticMap. …
    ⊢ Exists fun w => QuadraticMap.Equivalent Q (QuadraticMap.weightedSumSquares K …
  -/
  simp_rw [LinearMap.IsOrtho, associated_eq_self_apply] at hv₂
  /-
    case intro
    K : Type u_3
    V : Type u_8
    inst✝⁴ : Field K
    inst✝³ : Invertible 2
    inst✝² : AddCommGroup V
    inst✝¹ : Module K V
    inst✝ : FiniteDimensional K V
    Q : QuadraticForm K V
    hQ : LinearMap.SeparatingLeft (QuadraticMap.associated Q)
    v : Basis (Fin (Module.finrank K V)) K V
    hv₁ : LinearMap.IsOrthoᵢ ((QuadraticMap.associatedHom K) Q) ⇑v
    hv₂ : ∀ (i : Fin (Module.finrank K V)), Not (Eq (Q (v i)) 0)
    ⊢ Exists fun w => QuadraticMap.Equivalent Q (QuadraticMap.weightedSumSquares K …
  -/
  exact ⟨fun i => Units.mk0 _ (hv₂ i), ⟨Q.isometryEquivWeightedSumSquares v hv₁⟩⟩
  /-
    🎉 no goals
  -/


