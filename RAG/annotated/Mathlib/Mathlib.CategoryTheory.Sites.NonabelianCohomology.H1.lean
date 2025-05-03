/-- A zero cochain consists of a family of sections. -/
def ZeroCochain := ∀ (i : I), G.obj (Opposite.op (U i))


instance : Group (ZeroCochain G U) := Pi.group


@[simp, nolint simpNF]
lemma one_apply (i : I) : (1 : ZeroCochain G U) i = 1 := rfl


@[simp]
lemma inv_apply (γ : ZeroCochain G U) (i : I) : γ⁻¹ i = (γ i)⁻¹ := rfl


@[simp]
lemma mul_apply (γ₁ γ₂ : ZeroCochain G U) (i : I) : (γ₁ * γ₂) i = γ₁ i * γ₂ i := rfl


/-- A 1-cochain of a presheaf of groups `G : Cᵒᵖ ⥤ Grp` on a family `U : I → C` of objects
consists of the data of an element in `G.obj (Opposite.op T)` whenever we have elements
`i` and `j` in `I` and maps `a : T ⟶ U i` and `b : T ⟶ U j`, and it must satisfy a compatibility
with respect to precomposition. (When the binary product of `U i` and `U j` exists, this
data for all `T`, `a` and `b` corresponds to the data of a section of `G` on this product.) -/
@[ext]
structure OneCochain where
  /-- the data involved in a 1-cochain -/
  ev (i j : I) ⦃T : C⦄ (a : T ⟶ U i) (b : T ⟶ U j) : G.obj (Opposite.op T)
  ev_precomp (i j : I) ⦃T T' : C⦄ (φ : T ⟶ T') (a : T' ⟶ U i) (b : T' ⟶ U j) :
    G.map φ.op (ev i j a b) = ev i j (φ ≫ a) (φ ≫ b) := by aesop


instance : One (OneCochain G U) where
  one := { ev := fun _ _ _ _ _ ↦ 1 }


@[simp]
lemma one_ev (i j : I) {T : C} (a : T ⟶ U i) (b : T ⟶ U j) :
    (1 : OneCochain G U).ev i j a b = 1 := rfl


instance : Mul (OneCochain G U) where
  mul γ₁ γ₂ := { ev := fun i j _ a b ↦ γ₁.ev i j a b * γ₂.ev i j a b }


@[simp]
lemma mul_ev (γ₁ γ₂ : OneCochain G U) (i j : I) {T : C} (a : T ⟶ U i) (b : T ⟶ U j) :
    (γ₁ * γ₂).ev i j a b = γ₁.ev i j a b * γ₂.ev i j a b := rfl


instance : Inv (OneCochain G U) where
  inv γ := { ev := fun i j _ a b ↦ (γ.ev i j a b) ⁻¹}


@[simp]
lemma inv_ev (γ : OneCochain G U) (i j : I) {T : C} (a : T ⟶ U i) (b : T ⟶ U j) :
    (γ⁻¹).ev i j a b = (γ.ev i j a b)⁻¹ := rfl


instance : Group (OneCochain G U) where
                        /-
                          C : Type u
                          inst✝ : CategoryTheory.Category.{v, u} C
                          G : CategoryTheory.Functor (Opposite C) Grp
                          I : Type w'
                          U : I → C
                          x✝² x✝¹ x✝ : CategoryTheory.PresheafOfGroups.OneCochain G U
                          ⊢ Eq (HMul.hMul (HMul.hMul x✝² x✝¹) x✝) (HMul.hMul x✝² (HMul.hMul x✝¹ x✝))
                        -/
  mul_assoc _ _ _ := by ext; apply mul_assoc
                             /-
                               🎉 no goals
                             -/
                  /-
                    C : Type u
                    inst✝ : CategoryTheory.Category.{v, u} C
                    G : CategoryTheory.Functor (Opposite C) Grp
                    I : Type w'
                    U : I → C
                    x✝ : CategoryTheory.PresheafOfGroups.OneCochain G U
                    ⊢ Eq (HMul.hMul 1 x✝) x✝
                  -/
  one_mul _ := by ext; apply one_mul
                       /-
                         🎉 no goals
                       -/
                  /-
                    C : Type u
                    inst✝ : CategoryTheory.Category.{v, u} C
                    G : CategoryTheory.Functor (Opposite C) Grp
                    I : Type w'
                    U : I → C
                    x✝ : CategoryTheory.PresheafOfGroups.OneCochain G U
                    ⊢ Eq (HMul.hMul x✝ 1) x✝
                  -/
  mul_one _ := by ext; apply mul_one
                       /-
                         🎉 no goals
                       -/
                         /-
                           C : Type u
                           inst✝ : CategoryTheory.Category.{v, u} C
                           G : CategoryTheory.Functor (Opposite C) Grp
                           I : Type w'
                           U : I → C
                           x✝ : CategoryTheory.PresheafOfGroups.OneCochain G U
                           ⊢ Eq (HMul.hMul (Inv.inv x✝) x✝) 1
                         -/
  inv_mul_cancel _ := by ext; apply inv_mul_cancel
                              /-
                                🎉 no goals
                              -/


/-- A 1-cocycle is a 1-cochain which satisfies the cocycle condition. -/
structure OneCocycle extends OneCochain G U where
  ev_trans (i j k : I) ⦃T : C⦄ (a : T ⟶ U i) (b : T ⟶ U j) (c : T ⟶ U k) :
      ev i j a b * ev j k b c = ev i k a c := by aesop


instance : One (OneCocycle G U) where
         /-
           C : Type u
           inst✝ : CategoryTheory.Category.{v, u} C
           G : CategoryTheory.Functor (Opposite C) Grp
           I : Type w'
           U : I → C
           ⊢ ∀ (i j k : I) ⦃T : C⦄ (a : Quiver.Hom T (U i)) (b : Quiver.Hom T (U j)) (c : …
         -/
  one := OneCocycle.mk 1
         /-
           🎉 no goals
         -/


@[simp]
lemma one_toOneCochain : (1 : OneCocycle G U).toOneCochain = 1 := rfl


@[simp]
lemma ev_refl (γ : OneCocycle G U) (i : I) ⦃T : C⦄ (a : T ⟶ U i) :
    γ.ev i i a a = 1 := by
  /-
    C : Type u
    inst✝ : CategoryTheory.Category.{v, u} C
    G : CategoryTheory.Functor (Opposite C) Grp
    I : Type w'
    U : I → C
    γ : CategoryTheory.PresheafOfGroups.OneCocycle G U
    i : I
    T : C
    a : Quiver.Hom T (U i)
    ⊢ Eq (γ.ev i i a a) 1
  -/
  simpa using γ.ev_trans i i i a a a
  /-
    🎉 no goals
  -/


lemma ev_symm (γ : OneCocycle G U) (i j : I) ⦃T : C⦄ (a : T ⟶ U i) (b : T ⟶ U j) :
    γ.ev i j a b = (γ.ev j i b a)⁻¹ := by
  rw [← mul_left_inj (γ.ev j i b a), γ.ev_trans i j i a b a,
    ev_refl, inv_mul_cancel]


/-- The assertion that two cochains in `OneCochain G U` are cohomologous via
an explicit zero-cochain. -/
def OneCohomologyRelation (γ₁ γ₂ : OneCochain G U) (α : ZeroCochain G U) : Prop :=
  ∀ (i j : I) ⦃T : C⦄ (a : T ⟶ U i) (b : T ⟶ U j),
    G.map a.op (α i) * γ₁.ev i j a b = γ₂.ev i j a b * G.map b.op (α j)


                                                                                    /-
                                                                                      C : Type u
                                                                                      inst✝ : CategoryTheory.Category.{v, u} C
                                                                                      G : CategoryTheory.Functor (Opposite C) Grp
                                                                                      I : Type w'
                                                                                      U : I → C
                                                                                      γ : CategoryTheory.PresheafOfGroups.OneCochain G U
                                                                                      x✝⁴ x✝³ : I
                                                                                      x✝² : C
                                                                                      x✝¹ : Quiver.Hom x✝² (U x✝⁴)
                                                                                      x✝ : Quiver.Hom x✝² (U x✝³)
                                                                                      ⊢ Eq (HMul.hMul ((G.map x✝¹.op) (1 x✝⁴)) (γ.ev x✝⁴ x✝³ x✝¹ x✝)) (HMul.hMul (γ. …
                                                                                    -/
lemma refl (γ : OneCochain G U) : OneCohomologyRelation γ γ 1 := fun _ _ _ _ _ ↦ by simp
                                                                                    /-
                                                                                      🎉 no goals
                                                                                    -/


lemma symm {γ₁ γ₂ : OneCochain G U} {α : ZeroCochain G U} (h : OneCohomologyRelation γ₁ γ₂ α) :
    OneCohomologyRelation γ₂ γ₁ α⁻¹ := fun i j T a b ↦ by
  rw [← mul_left_inj (G.map b.op (α j)), mul_assoc, ← h i j a b,
    mul_assoc, Cochain₀.inv_apply, map_inv, inv_mul_cancel_left,
    Cochain₀.inv_apply, map_inv, inv_mul_cancel, mul_one]


lemma trans {γ₁ γ₂ γ₃ : OneCochain G U} {α β : ZeroCochain G U}
    (h₁₂ : OneCohomologyRelation γ₁ γ₂ α) (h₂₃ : OneCohomologyRelation γ₂ γ₃ β) :
    OneCohomologyRelation γ₁ γ₃ (β * α) := fun i j T a b ↦ by
  /-
    C : Type u
    inst✝ : CategoryTheory.Category.{v, u} C
    G : CategoryTheory.Functor (Opposite C) Grp
    I : Type w'
    U : I → C
    γ₁ γ₂ γ₃ : CategoryTheory.PresheafOfGroups.OneCochain G U
    α β : CategoryTheory.PresheafOfGroups.ZeroCochain G U
    h₁₂ : CategoryTheory.PresheafOfGroups.OneCohomologyRelation γ₁ γ₂ α
    h₂₃ : CategoryTheory.PresheafOfGroups.OneCohomologyRelation γ₂ γ₃ β
    i j : I
    T : C
    a : Quiver.Hom T (U i)
    b : Quiver.Hom T (U j)
    ⊢ Eq (HMul.hMul ((G.map a.op) (HMul.hMul β α i)) (γ₁.ev i j a b)) (HMul.hMul ( …
  -/
  dsimp
  rw [map_mul, map_mul, mul_assoc, h₁₂ i j a b, ← mul_assoc,
    h₂₃ i j a b, mul_assoc]


/-- The cohomology (equivalence) relation on 1-cocycles. -/
def IsCohomologous (γ₁ γ₂ : OneCocycle G U) : Prop :=
  ∃ (α : ZeroCochain G U), OneCohomologyRelation γ₁.toOneCochain γ₂.toOneCochain α


lemma equivalence_isCohomologous :
    _root_.Equivalence (IsCohomologous (G := G) (U := U)) where
  refl γ := ⟨_, OneCohomologyRelation.refl γ.toOneCochain⟩
  symm := by
    /-
      C : Type u
      inst✝ : CategoryTheory.Category.{v, u} C
      G : CategoryTheory.Functor (Opposite C) Grp
      I : Type w'
      U : I → C
      ⊢ ∀ {x y : CategoryTheory.PresheafOfGroups.OneCocycle G U}, x.IsCohomologous y …
    -/
    rintro γ₁ γ₂ ⟨α, h⟩
    /-
      case intro
      C : Type u
      inst✝ : CategoryTheory.Category.{v, u} C
      G : CategoryTheory.Functor (Opposite C) Grp
      I : Type w'
      U : I → C
      γ₁ γ₂ : CategoryTheory.PresheafOfGroups.OneCocycle G U
      α : CategoryTheory.PresheafOfGroups.ZeroCochain G U
      h : CategoryTheory.PresheafOfGroups.OneCohomologyRelation γ₁.toOneCochain γ₂.t …
      ⊢ γ₂.IsCohomologous γ₁
    -/
    exact ⟨_, h.symm⟩
    /-
      🎉 no goals
    -/
  trans := by
    /-
      C : Type u
      inst✝ : CategoryTheory.Category.{v, u} C
      G : CategoryTheory.Functor (Opposite C) Grp
      I : Type w'
      U : I → C
      ⊢ ∀ {x y z : CategoryTheory.PresheafOfGroups.OneCocycle G U}, x.IsCohomologous …
    -/
    rintro γ₁ γ₂ γ₂ ⟨α, h⟩ ⟨β, h'⟩
    /-
      case intro.intro
      C : Type u
      inst✝ : CategoryTheory.Category.{v, u} C
      G : CategoryTheory.Functor (Opposite C) Grp
      I : Type w'
      U : I → C
      γ₁ γ₂✝ γ₂ : CategoryTheory.PresheafOfGroups.OneCocycle G U
      α : CategoryTheory.PresheafOfGroups.ZeroCochain G U
      h : CategoryTheory.PresheafOfGroups.OneCohomologyRelation γ₁.toOneCochain γ₂✝. …
      β : CategoryTheory.PresheafOfGroups.ZeroCochain G U
      h' : CategoryTheory.PresheafOfGroups.OneCohomologyRelation γ₂✝.toOneCochain γ₂ …
      ⊢ γ₁.IsCohomologous γ₂
    -/
    exact ⟨_, h.trans h'⟩
    /-
      🎉 no goals
    -/


variable (G U) in
/-- The cohomology in degree 1 of a presheaf of groups
`G : Cᵒᵖ ⥤ Grp` on a family of objects `U : I → C`. -/
def H1 := Quot (OneCocycle.IsCohomologous (G := G) (U := U))


/-- The cohomology class of a 1-cocycle. -/
def OneCocycle.class (γ : OneCocycle G U) : H1 G U := Quot.mk _ γ


instance : One (H1 G U) where
  one := OneCocycle.class 1


lemma OneCocycle.class_eq_iff (γ₁ γ₂ : OneCocycle G U) :
    γ₁.class = γ₂.class ↔ γ₁.IsCohomologous γ₂ :=
  (equivalence_isCohomologous _ _ ).quot_mk_eq_iff _ _


lemma OneCocycle.IsCohomologous.class_eq {γ₁ γ₂ : OneCocycle G U} (h : γ₁.IsCohomologous γ₂) :
    γ₁.class = γ₂.class :=
  Quot.sound h


