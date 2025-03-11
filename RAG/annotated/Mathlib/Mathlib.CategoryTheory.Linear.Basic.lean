/-- A category is called `R`-linear if `P ⟶ Q` is an `R`-module such that composition is
    `R`-linear in both variables. -/
class Linear (R : Type w) [Semiring R] (C : Type u) [Category.{v} C] [Preadditive C] where
  homModule : ∀ X Y : C, Module R (X ⟶ Y) := by infer_instance
  /-- compatibility of the scalar multiplication with the post-composition -/
  smul_comp : ∀ (X Y Z : C) (r : R) (f : X ⟶ Y) (g : Y ⟶ Z), (r • f) ≫ g = r • f ≫ g := by
    aesop_cat
  /-- compatibility of the scalar multiplication with the pre-composition -/
  comp_smul : ∀ (X Y Z : C) (f : X ⟶ Y) (r : R) (g : Y ⟶ Z), f ≫ (r • g) = r • f ≫ g := by
    aesop_cat


instance preadditiveNatLinear : Linear ℕ C where
                              /-
                                C : Type u
                                inst✝¹ : CategoryTheory.Category.{v, u} C
                                inst✝ : CategoryTheory.Preadditive C
                                X Y Z : C
                                r : Nat
                                f : Quiver.Hom X Y
                                g : Quiver.Hom Y Z
                                ⊢ Eq (CategoryTheory.CategoryStruct.comp (HSMul.hSMul r f) g) (HSMul.hSMul r ( …
                              -/
  smul_comp X Y Z r f g := by exact (Preadditive.rightComp X g).map_nsmul f r
                              /-
                                🎉 no goals
                              -/
                              /-
                                C : Type u
                                inst✝¹ : CategoryTheory.Category.{v, u} C
                                inst✝ : CategoryTheory.Preadditive C
                                X Y Z : C
                                f : Quiver.Hom X Y
                                r : Nat
                                g : Quiver.Hom Y Z
                                ⊢ Eq (CategoryTheory.CategoryStruct.comp f (HSMul.hSMul r g)) (HSMul.hSMul r ( …
                              -/
  comp_smul X Y Z f r g := by exact (Preadditive.leftComp Z f).map_nsmul g r
                              /-
                                🎉 no goals
                              -/


instance preadditiveIntLinear : Linear ℤ C where
                              /-
                                C : Type u
                                inst✝¹ : CategoryTheory.Category.{v, u} C
                                inst✝ : CategoryTheory.Preadditive C
                                X Y Z : C
                                r : Int
                                f : Quiver.Hom X Y
                                g : Quiver.Hom Y Z
                                ⊢ Eq (CategoryTheory.CategoryStruct.comp (HSMul.hSMul r f) g) (HSMul.hSMul r ( …
                              -/
  smul_comp X Y Z r f g := by exact (Preadditive.rightComp X g).map_zsmul f r
                              /-
                                🎉 no goals
                              -/
                              /-
                                C : Type u
                                inst✝¹ : CategoryTheory.Category.{v, u} C
                                inst✝ : CategoryTheory.Preadditive C
                                X Y Z : C
                                f : Quiver.Hom X Y
                                r : Int
                                g : Quiver.Hom Y Z
                                ⊢ Eq (CategoryTheory.CategoryStruct.comp f (HSMul.hSMul r g)) (HSMul.hSMul r ( …
                              -/
  comp_smul X Y Z f r g := by exact (Preadditive.leftComp Z f).map_zsmul g r
                              /-
                                🎉 no goals
                              -/


instance [Semiring R] [Linear R C] (X : C) : Module R (End X) := by
  /-
    C : Type u
    inst✝³ : CategoryTheory.Category.{v, u} C
    inst✝² : CategoryTheory.Preadditive C
    R : Type w
    inst✝¹ : Semiring R
    inst✝ : CategoryTheory.Linear R C
    X : C
    ⊢ Module R (CategoryTheory.End X)
  -/
  dsimp [End]
  /-
    C : Type u
    inst✝³ : CategoryTheory.Category.{v, u} C
    inst✝² : CategoryTheory.Preadditive C
    R : Type w
    inst✝¹ : Semiring R
    inst✝ : CategoryTheory.Linear R C
    X : C
    ⊢ Module R (Quiver.Hom X X)
  -/
  infer_instance
  /-
    🎉 no goals
  -/


instance [CommSemiring R] [Linear R C] (X : C) : Algebra R (End X) :=
  Algebra.ofModule (fun _ _ _ => comp_smul _ _ _ _ _ _) fun _ _ _ => smul_comp _ _ _ _ _ _


instance inducedCategory : Linear.{w, v} R (InducedCategory C F) where
  homModule X Y := @Linear.homModule R _ C _ _ _ (F X) (F Y)
  smul_comp _ _ _ _ _ _ := smul_comp _ _ _ _ _ _
  comp_smul _ _ _ _ _ _ := comp_smul _ _ _ _ _ _


instance fullSubcategory (Z : C → Prop) : Linear.{w, v} R (FullSubcategory Z) where
  homModule X Y := @Linear.homModule R _ C _ _ _ X.obj Y.obj
  smul_comp _ _ _ _ _ _ := smul_comp _ _ _ _ _ _
  comp_smul _ _ _ _ _ _ := comp_smul _ _ _ _ _ _


/-- Composition by a fixed left argument as an `R`-linear map. -/
@[simps]
def leftComp {X Y : C} (Z : C) (f : X ⟶ Y) : (Y ⟶ Z) →ₗ[R] X ⟶ Z where
  toFun g := f ≫ g
                 /-
                   C : Type u
                   inst✝³ : CategoryTheory.Category.{v, u} C
                   inst✝² : CategoryTheory.Preadditive C
                   R : Type w
                   inst✝¹ : Semiring R
                   inst✝ : CategoryTheory.Linear R C
                   X Y Z : C
                   f : Quiver.Hom X Y
                   ⊢ ∀ (x y : Quiver.Hom Y Z), Eq ((fun g => CategoryTheory.CategoryStruct.comp f …
                 -/
  map_add' := by simp
                 /-
                   🎉 no goals
                 -/
                  /-
                    C : Type u
                    inst✝³ : CategoryTheory.Category.{v, u} C
                    inst✝² : CategoryTheory.Preadditive C
                    R : Type w
                    inst✝¹ : Semiring R
                    inst✝ : CategoryTheory.Linear R C
                    X Y Z : C
                    f : Quiver.Hom X Y
                    ⊢ ∀ (m : R) (x : Quiver.Hom Y Z), Eq ({ toFun := fun g => CategoryTheory.Categ …
                  -/
  map_smul' := by simp
                  /-
                    🎉 no goals
                  -/


/-- Composition by a fixed right argument as an `R`-linear map. -/
@[simps]
def rightComp (X : C) {Y Z : C} (g : Y ⟶ Z) : (X ⟶ Y) →ₗ[R] X ⟶ Z where
  toFun f := f ≫ g
                 /-
                   C : Type u
                   inst✝³ : CategoryTheory.Category.{v, u} C
                   inst✝² : CategoryTheory.Preadditive C
                   R : Type w
                   inst✝¹ : Semiring R
                   inst✝ : CategoryTheory.Linear R C
                   X Y Z : C
                   g : Quiver.Hom Y Z
                   ⊢ ∀ (x y : Quiver.Hom X Y), Eq ((fun f => CategoryTheory.CategoryStruct.comp f …
                 -/
  map_add' := by simp
                 /-
                   🎉 no goals
                 -/
                  /-
                    C : Type u
                    inst✝³ : CategoryTheory.Category.{v, u} C
                    inst✝² : CategoryTheory.Preadditive C
                    R : Type w
                    inst✝¹ : Semiring R
                    inst✝ : CategoryTheory.Linear R C
                    X Y Z : C
                    g : Quiver.Hom Y Z
                    ⊢ ∀ (m : R) (x : Quiver.Hom X Y), Eq ({ toFun := fun f => CategoryTheory.Categ …
                  -/
  map_smul' := by simp
                  /-
                    🎉 no goals
                  -/


instance {X Y : C} (f : X ⟶ Y) [Epi f] (r : R) [Invertible r] : Epi (r • f) :=
  ⟨fun g g' H => by
    /-
      C : Type u
      inst✝⁵ : CategoryTheory.Category.{v, u} C
      inst✝⁴ : CategoryTheory.Preadditive C
      R : Type w
      inst✝³ : Semiring R
      inst✝² : CategoryTheory.Linear R C
      X Y : C
      f : Quiver.Hom X Y
      inst✝¹ : CategoryTheory.Epi f
      r : R
      inst✝ : Invertible r
      Z✝ : C
      g g' : Quiver.Hom Y Z✝
      H : Eq (CategoryTheory.CategoryStruct.comp (HSMul.hSMul r f) g) (CategoryTheor …
      ⊢ Eq g g'
    -/
    rw [smul_comp, smul_comp, ← comp_smul, ← comp_smul, cancel_epi] at H
    /-
      C : Type u
      inst✝⁵ : CategoryTheory.Category.{v, u} C
      inst✝⁴ : CategoryTheory.Preadditive C
      R : Type w
      inst✝³ : Semiring R
      inst✝² : CategoryTheory.Linear R C
      X Y : C
      f : Quiver.Hom X Y
      inst✝¹ : CategoryTheory.Epi f
      r : R
      inst✝ : Invertible r
      Z✝ : C
      g g' : Quiver.Hom Y Z✝
      H : Eq (HSMul.hSMul r g) (HSMul.hSMul r g')
      ⊢ Eq g g'
    -/
    simpa [smul_smul] using congr_arg (fun f => ⅟ r • f) H⟩
    /-
      🎉 no goals
    -/


instance {X Y : C} (f : X ⟶ Y) [Mono f] (r : R) [Invertible r] : Mono (r • f) :=
  ⟨fun g g' H => by
    /-
      C : Type u
      inst✝⁵ : CategoryTheory.Category.{v, u} C
      inst✝⁴ : CategoryTheory.Preadditive C
      R : Type w
      inst✝³ : Semiring R
      inst✝² : CategoryTheory.Linear R C
      X Y : C
      f : Quiver.Hom X Y
      inst✝¹ : CategoryTheory.Mono f
      r : R
      inst✝ : Invertible r
      Z✝ : C
      g g' : Quiver.Hom Z✝ X
      H : Eq (CategoryTheory.CategoryStruct.comp g (HSMul.hSMul r f)) (CategoryTheor …
      ⊢ Eq g g'
    -/
    rw [comp_smul, comp_smul, ← smul_comp, ← smul_comp, cancel_mono] at H
    /-
      C : Type u
      inst✝⁵ : CategoryTheory.Category.{v, u} C
      inst✝⁴ : CategoryTheory.Preadditive C
      R : Type w
      inst✝³ : Semiring R
      inst✝² : CategoryTheory.Linear R C
      X Y : C
      f : Quiver.Hom X Y
      inst✝¹ : CategoryTheory.Mono f
      r : R
      inst✝ : Invertible r
      Z✝ : C
      g g' : Quiver.Hom Z✝ X
      H : Eq (HSMul.hSMul r g) (HSMul.hSMul r g')
      ⊢ Eq g g'
    -/
    simpa [smul_smul] using congr_arg (fun f => ⅟ r • f) H⟩
    /-
      🎉 no goals
    -/


/-- Given isomorphic objects `X ≅ Y, W ≅ Z` in a `k`-linear category, we have a `k`-linear
isomorphism between `Hom(X, W)` and `Hom(Y, Z).` -/
def homCongr (k : Type*) {C : Type*} [Category C] [Semiring k] [Preadditive C] [Linear k C]
    {X Y W Z : C} (f₁ : X ≅ Y) (f₂ : W ≅ Z) : (X ⟶ W) ≃ₗ[k] Y ⟶ Z :=
  {
    (rightComp k Y f₂.hom).comp
      (leftComp k W
        f₁.symm.hom) with
    invFun := (leftComp k W f₁.hom).comp (rightComp k Y f₂.symm.hom)
    left_inv := fun x => by
      simp only [Iso.symm_hom, LinearMap.toFun_eq_coe, LinearMap.coe_comp, Function.comp_apply,
        leftComp_apply, rightComp_apply, Category.assoc, Iso.hom_inv_id, Category.comp_id,
        Iso.hom_inv_id_assoc]
    right_inv := fun x => by
      simp only [Iso.symm_hom, LinearMap.coe_comp, Function.comp_apply, rightComp_apply,
        leftComp_apply, LinearMap.toFun_eq_coe, Iso.inv_hom_id_assoc, Category.assoc,
        Iso.inv_hom_id, Category.comp_id] }


theorem homCongr_apply (k : Type*) {C : Type*} [Category C] [Semiring k] [Preadditive C]
    [Linear k C] {X Y W Z : C} (f₁ : X ≅ Y) (f₂ : W ≅ Z) (f : X ⟶ W) :
    homCongr k f₁ f₂ f = (f₁.inv ≫ f) ≫ f₂.hom :=
  rfl


theorem homCongr_symm_apply (k : Type*) {C : Type*} [Category C] [Semiring k] [Preadditive C]
    [Linear k C] {X Y W Z : C} (f₁ : X ≅ Y) (f₂ : W ≅ Z) (f : Y ⟶ Z) :
    (homCongr k f₁ f₂).symm f = f₁.hom ≫ f ≫ f₂.inv :=
  rfl


@[simp]
lemma units_smul_comp {X Y Z : C} (r : Rˣ) (f : X ⟶ Y) (g : Y ⟶ Z) :
    (r • f) ≫ g = r • f ≫ g := by
  /-
    C : Type u
    inst✝³ : CategoryTheory.Category.{v, u} C
    inst✝² : CategoryTheory.Preadditive C
    R : Type w
    inst✝¹ : Semiring R
    inst✝ : CategoryTheory.Linear R C
    X Y Z : C
    r : Units R
    f : Quiver.Hom X Y
    g : Quiver.Hom Y Z
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (HSMul.hSMul r f) g) (HSMul.hSMul r ( …
  -/
  apply Linear.smul_comp
  /-
    🎉 no goals
  -/


@[simp]
lemma comp_units_smul {X Y Z : C} (f : X ⟶ Y) (r : Rˣ) (g : Y ⟶ Z) :
    f ≫ (r • g) = r • f ≫ g := by
  /-
    C : Type u
    inst✝³ : CategoryTheory.Category.{v, u} C
    inst✝² : CategoryTheory.Preadditive C
    R : Type w
    inst✝¹ : Semiring R
    inst✝ : CategoryTheory.Linear R C
    X Y Z : C
    f : Quiver.Hom X Y
    r : Units R
    g : Quiver.Hom Y Z
    ⊢ Eq (CategoryTheory.CategoryStruct.comp f (HSMul.hSMul r g)) (HSMul.hSMul r ( …
  -/
  apply Linear.comp_smul
  /-
    🎉 no goals
  -/


/-- Composition as a bilinear map. -/
@[simps]
def comp (X Y Z : C) : (X ⟶ Y) →ₗ[S] (Y ⟶ Z) →ₗ[S] X ⟶ Z where
  toFun f := leftComp S Z f
  map_add' := by
    /-
      C : Type u
      inst✝³ : CategoryTheory.Category.{v, u} C
      inst✝² : CategoryTheory.Preadditive C
      S : Type w
      inst✝¹ : CommSemiring S
      inst✝ : CategoryTheory.Linear S C
      X Y Z : C
      ⊢ ∀ (x y : Quiver.Hom X Y), Eq ((fun f => CategoryTheory.Linear.leftComp S Z f …
    -/
    intros
    /-
      C : Type u
      inst✝³ : CategoryTheory.Category.{v, u} C
      inst✝² : CategoryTheory.Preadditive C
      S : Type w
      inst✝¹ : CommSemiring S
      inst✝ : CategoryTheory.Linear S C
      X Y Z : C
      x✝ y✝ : Quiver.Hom X Y
      ⊢ Eq ((fun f => CategoryTheory.Linear.leftComp S Z f) (HAdd.hAdd x✝ y✝)) (HAdd …
    -/
    ext
    /-
      case h
      C : Type u
      inst✝³ : CategoryTheory.Category.{v, u} C
      inst✝² : CategoryTheory.Preadditive C
      S : Type w
      inst✝¹ : CommSemiring S
      inst✝ : CategoryTheory.Linear S C
      X Y Z : C
      x✝¹ y✝ : Quiver.Hom X Y
      x✝ : Quiver.Hom Y Z
      ⊢ Eq (((fun f => CategoryTheory.Linear.leftComp S Z f) (HAdd.hAdd x✝¹ y✝)) x✝) …
    -/
    simp
    /-
      🎉 no goals
    -/
  map_smul' := by
    /-
      C : Type u
      inst✝³ : CategoryTheory.Category.{v, u} C
      inst✝² : CategoryTheory.Preadditive C
      S : Type w
      inst✝¹ : CommSemiring S
      inst✝ : CategoryTheory.Linear S C
      X Y Z : C
      ⊢ ∀ (m : S) (x : Quiver.Hom X Y), Eq ({ toFun := fun f => CategoryTheory.Linea …
    -/
    intros
    /-
      C : Type u
      inst✝³ : CategoryTheory.Category.{v, u} C
      inst✝² : CategoryTheory.Preadditive C
      S : Type w
      inst✝¹ : CommSemiring S
      inst✝ : CategoryTheory.Linear S C
      X Y Z : C
      m✝ : S
      x✝ : Quiver.Hom X Y
      ⊢ Eq ({ toFun := fun f => CategoryTheory.Linear.leftComp S Z f, map_add' := ⋯  …
    -/
    ext
    /-
      case h
      C : Type u
      inst✝³ : CategoryTheory.Category.{v, u} C
      inst✝² : CategoryTheory.Preadditive C
      S : Type w
      inst✝¹ : CommSemiring S
      inst✝ : CategoryTheory.Linear S C
      X Y Z : C
      m✝ : S
      x✝¹ : Quiver.Hom X Y
      x✝ : Quiver.Hom Y Z
      ⊢ Eq (({ toFun := fun f => CategoryTheory.Linear.leftComp S Z f, map_add' := ⋯ …
    -/
    simp
    /-
      🎉 no goals
    -/


