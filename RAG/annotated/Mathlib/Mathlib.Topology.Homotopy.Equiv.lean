/-- A homotopy equivalence between topological spaces `X` and `Y` are a pair of functions
`toFun : C(X, Y)` and `invFun : C(Y, X)` such that `toFun.comp invFun` and `invFun.comp toFun`
are both homotopic to corresponding identity maps.
-/
@[ext]
structure HomotopyEquiv (X : Type u) (Y : Type v) [TopologicalSpace X] [TopologicalSpace Y] where
  toFun : C(X, Y)
  invFun : C(Y, X)
  left_inv : (invFun.comp toFun).Homotopic (ContinuousMap.id X)
  right_inv : (toFun.comp invFun).Homotopic (ContinuousMap.id Y)


@[inherit_doc] scoped infixl:25 " ≃ₕ " => ContinuousMap.HomotopyEquiv


/-- Coercion of a `HomotopyEquiv` to function. While the Lean 4 way is to unfold coercions, this
auxiliary definition will make porting of Lean 3 code easier.

Porting note (https://github.com/leanprover-community/mathlib4/issues/11215): TODO: drop this definition. -/
@[coe] def toFun' (e : X ≃ₕ Y) : X → Y := e.toFun


instance : CoeFun (X ≃ₕ Y) fun _ => X → Y := ⟨toFun'⟩


@[simp]
theorem toFun_eq_coe (h : HomotopyEquiv X Y) : (h.toFun : X → Y) = h :=
  rfl


@[continuity]
theorem continuous (h : HomotopyEquiv X Y) : Continuous h :=
  h.toFun.continuous


/-- Any homeomorphism is a homotopy equivalence.
-/
def toHomotopyEquiv (h : X ≃ₜ Y) : X ≃ₕ Y where
  toFun := h
  invFun := h.symm
                 /-
                   X : Type u
                   Y : Type v
                   Z : Type w
                   Z' : Type x
                   inst✝³ : TopologicalSpace X
                   inst✝² : TopologicalSpace Y
                   inst✝¹ : TopologicalSpace Z
                   inst✝ : TopologicalSpace Z'
                   h : Homeomorph X Y
                   ⊢ ((↑h.symm).comp ↑h).Homotopic (ContinuousMap.id X)
                 -/
  left_inv := by rw [symm_comp_toContinuousMap]
                 /-
                   🎉 no goals
                 -/
                  /-
                    X : Type u
                    Y : Type v
                    Z : Type w
                    Z' : Type x
                    inst✝³ : TopologicalSpace X
                    inst✝² : TopologicalSpace Y
                    inst✝¹ : TopologicalSpace Z
                    inst✝ : TopologicalSpace Z'
                    h : Homeomorph X Y
                    ⊢ ((↑h).comp ↑h.symm).Homotopic (ContinuousMap.id Y)
                  -/
  right_inv := by rw [toContinuousMap_comp_symm]
                  /-
                    🎉 no goals
                  -/


@[simp]
theorem coe_toHomotopyEquiv (h : X ≃ₜ Y) : (h.toHomotopyEquiv : X → Y) = h :=
  rfl


/-- If `X` is homotopy equivalent to `Y`, then `Y` is homotopy equivalent to `X`.
-/
def symm (h : X ≃ₕ Y) : Y ≃ₕ X where
  toFun := h.invFun
  invFun := h.toFun
  left_inv := h.right_inv
  right_inv := h.left_inv


@[simp]
theorem coe_invFun (h : HomotopyEquiv X Y) : (⇑h.invFun : Y → X) = ⇑h.symm :=
  rfl


/-- See Note [custom simps projection]. We need to specify this projection explicitly in this case,
because it is a composition of multiple projections. -/
def Simps.apply (h : X ≃ₕ Y) : X → Y :=
  h


/-- See Note [custom simps projection]. We need to specify this projection explicitly in this case,
because it is a composition of multiple projections. -/
def Simps.symm_apply (h : X ≃ₕ Y) : Y → X :=
  h.symm


/-- Any topological space is homotopy equivalent to itself.
-/
@[simps!]
def refl (X : Type u) [TopologicalSpace X] : X ≃ₕ X :=
  (Homeomorph.refl X).toHomotopyEquiv


instance : Inhabited (HomotopyEquiv Unit Unit) :=
  ⟨refl Unit⟩


/--
If `X` is homotopy equivalent to `Y`, and `Y` is homotopy equivalent to `Z`, then `X` is homotopy
equivalent to `Z`.
-/
@[simps!]
def trans (h₁ : X ≃ₕ Y) (h₂ : Y ≃ₕ Z) : X ≃ₕ Z where
  toFun := h₂.toFun.comp h₁.toFun
  invFun := h₁.invFun.comp h₂.invFun
  left_inv := by
    /-
      X : Type u
      Y : Type v
      Z : Type w
      Z' : Type x
      inst✝³ : TopologicalSpace X
      inst✝² : TopologicalSpace Y
      inst✝¹ : TopologicalSpace Z
      inst✝ : TopologicalSpace Z'
      h₁ : ContinuousMap.HomotopyEquiv X Y
      h₂ : ContinuousMap.HomotopyEquiv Y Z
      ⊢ ((h₁.invFun.comp h₂.invFun).comp (h₂.toFun.comp h₁.toFun)).Homotopic (Contin …
    -/
    refine Homotopic.trans ?_ h₁.left_inv
    /-
      X : Type u
      Y : Type v
      Z : Type w
      Z' : Type x
      inst✝³ : TopologicalSpace X
      inst✝² : TopologicalSpace Y
      inst✝¹ : TopologicalSpace Z
      inst✝ : TopologicalSpace Z'
      h₁ : ContinuousMap.HomotopyEquiv X Y
      h₂ : ContinuousMap.HomotopyEquiv Y Z
      ⊢ ((h₁.invFun.comp h₂.invFun).comp (h₂.toFun.comp h₁.toFun)).Homotopic (h₁.inv …
    -/
    exact ((Homotopic.refl _).hcomp h₂.left_inv).hcomp (Homotopic.refl _)
    /-
      🎉 no goals
    -/
  right_inv := by
    /-
      X : Type u
      Y : Type v
      Z : Type w
      Z' : Type x
      inst✝³ : TopologicalSpace X
      inst✝² : TopologicalSpace Y
      inst✝¹ : TopologicalSpace Z
      inst✝ : TopologicalSpace Z'
      h₁ : ContinuousMap.HomotopyEquiv X Y
      h₂ : ContinuousMap.HomotopyEquiv Y Z
      ⊢ ((h₂.toFun.comp h₁.toFun).comp (h₁.invFun.comp h₂.invFun)).Homotopic (Contin …
    -/
    refine Homotopic.trans ?_ h₂.right_inv
    /-
      X : Type u
      Y : Type v
      Z : Type w
      Z' : Type x
      inst✝³ : TopologicalSpace X
      inst✝² : TopologicalSpace Y
      inst✝¹ : TopologicalSpace Z
      inst✝ : TopologicalSpace Z'
      h₁ : ContinuousMap.HomotopyEquiv X Y
      h₂ : ContinuousMap.HomotopyEquiv Y Z
      ⊢ ((h₂.toFun.comp h₁.toFun).comp (h₁.invFun.comp h₂.invFun)).Homotopic (h₂.toF …
    -/
    exact ((Homotopic.refl _).hcomp h₁.right_inv).hcomp (Homotopic.refl _)
    /-
      🎉 no goals
    -/


theorem symm_trans (h₁ : X ≃ₕ Y) (h₂ : Y ≃ₕ Z) : (h₁.trans h₂).symm = h₂.symm.trans h₁.symm := rfl


/-- If `X` is homotopy equivalent to `Y` and `Z` is homotopy equivalent to `Z'`, then `X × Z` is
homotopy equivalent to `Z × Z'`. -/
def prodCongr (h₁ : X ≃ₕ Y) (h₂ : Z ≃ₕ Z') : (X × Z) ≃ₕ (Y × Z') where
  toFun := h₁.toFun.prodMap h₂.toFun
  invFun := h₁.invFun.prodMap h₂.invFun
  left_inv := h₁.left_inv.prodMap h₂.left_inv
  right_inv := h₁.right_inv.prodMap h₂.right_inv


/-- If `X i` is homotopy equivalent to `Y i` for each `i`, then the space of functions (a.k.a. the
indexed product) `∀ i, X i` is homotopy equivalent to `∀ i, Y i`. -/
def piCongrRight {ι : Type*} {X Y : ι → Type*} [∀ i, TopologicalSpace (X i)]
    [∀ i, TopologicalSpace (Y i)] (h : ∀ i, X i ≃ₕ Y i) :
    (∀ i, X i) ≃ₕ (∀ i, Y i) where
  toFun := .piMap fun i ↦ (h i).toFun
  invFun := .piMap fun i ↦ (h i).invFun
  left_inv := .piMap fun i ↦ (h i).left_inv
  right_inv := .piMap fun i ↦ (h i).right_inv


@[simp]
theorem refl_toHomotopyEquiv (X : Type u) [TopologicalSpace X] :
    (Homeomorph.refl X).toHomotopyEquiv = HomotopyEquiv.refl X :=
  rfl


@[simp]
theorem symm_toHomotopyEquiv (h : X ≃ₜ Y) : h.symm.toHomotopyEquiv = h.toHomotopyEquiv.symm :=
  rfl


@[simp]
theorem trans_toHomotopyEquiv (h₀ : X ≃ₜ Y) (h₁ : Y ≃ₜ Z) :
    (h₀.trans h₁).toHomotopyEquiv = h₀.toHomotopyEquiv.trans h₁.toHomotopyEquiv :=
  rfl


