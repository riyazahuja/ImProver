/-- The datum of an extra degeneracy is a technical condition on
augmented simplicial objects. The morphisms `s'` and `s n` of the
structure formally behave like extra degeneracies `σ (-1)`. -/
@[ext]
structure ExtraDegeneracy (X : SimplicialObject.Augmented C) where
  s' : point.obj X ⟶ drop.obj X _[0]
  s : ∀ n : ℕ, drop.obj X _[n] ⟶ drop.obj X _[n + 1]
  s'_comp_ε : s' ≫ X.hom.app (op [0]) = 𝟙 _
  s₀_comp_δ₁ : s 0 ≫ X.left.δ 1 = X.hom.app (op [0]) ≫ s'
  s_comp_δ₀ : ∀ n : ℕ, s n ≫ X.left.δ 0 = 𝟙 _
  s_comp_δ :
    ∀ (n : ℕ) (i : Fin (n + 2)), s (n + 1) ≫ X.left.δ i.succ = X.left.δ i ≫ s n
  s_comp_σ :
    ∀ (n : ℕ) (i : Fin (n + 1)), s n ≫ X.left.σ i.succ = X.left.σ i ≫ s (n + 1)


attribute [reassoc] s₀_comp_δ₁ s_comp_δ s_comp_σ

attribute [reassoc (attr := simp)] s'_comp_ε s_comp_δ₀


/-- If `ed` is an extra degeneracy for `X : SimplicialObject.Augmented C` and
`F : C ⥤ D` is a functor, then `ed.map F` is an extra degeneracy for the
augmented simplicial object in `D` obtained by applying `F` to `X`. -/
def map {D : Type*} [Category D] {X : SimplicialObject.Augmented C} (ed : ExtraDegeneracy X)
    (F : C ⥤ D) : ExtraDegeneracy (((whiskering _ _).obj F).obj X) where
  s' := F.map ed.s'
  s n := F.map (ed.s n)
  s'_comp_ε := by
    /-
      C : Type u_1
      inst✝¹ : CategoryTheory.Category.{?u.14494, u_1} C
      D : Type u_2
      inst✝ : CategoryTheory.Category.{?u.14501, u_2} D
      X : CategoryTheory.SimplicialObject.Augmented C
      ed : SimplicialObject.Augmented.ExtraDegeneracy X
      F : CategoryTheory.Functor C D
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (F.map ed.s') ((((CategoryTheory.Simp …
    -/
    dsimp
    /-
      C : Type u_1
      inst✝¹ : CategoryTheory.Category.{?u.14494, u_1} C
      D : Type u_2
      inst✝ : CategoryTheory.Category.{?u.14501, u_2} D
      X : CategoryTheory.SimplicialObject.Augmented C
      ed : SimplicialObject.Augmented.ExtraDegeneracy X
      F : CategoryTheory.Functor C D
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (F.map ed.s') (CategoryTheory.Categor …
    -/
    erw [comp_id, ← F.map_comp, ed.s'_comp_ε, F.map_id]
    /-
      🎉 no goals
    -/
  s₀_comp_δ₁ := by
    /-
      C : Type u_1
      inst✝¹ : CategoryTheory.Category.{?u.14494, u_1} C
      D : Type u_2
      inst✝ : CategoryTheory.Category.{?u.14501, u_2} D
      X : CategoryTheory.SimplicialObject.Augmented C
      ed : SimplicialObject.Augmented.ExtraDegeneracy X
      F : CategoryTheory.Functor C D
      ⊢ Eq (CategoryTheory.CategoryStruct.comp ((fun n => F.map (ed.s n)) 0) ((((Cat …
    -/
    dsimp
    /-
      C : Type u_1
      inst✝¹ : CategoryTheory.Category.{?u.14494, u_1} C
      D : Type u_2
      inst✝ : CategoryTheory.Category.{?u.14501, u_2} D
      X : CategoryTheory.SimplicialObject.Augmented C
      ed : SimplicialObject.Augmented.ExtraDegeneracy X
      F : CategoryTheory.Functor C D
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (F.map (ed.s 0)) ((((CategoryTheory.S …
    -/
    erw [comp_id, ← F.map_comp, ← F.map_comp, ed.s₀_comp_δ₁]
    /-
      🎉 no goals
    -/
  s_comp_δ₀ n := by
    /-
      C : Type u_1
      inst✝¹ : CategoryTheory.Category.{?u.14494, u_1} C
      D : Type u_2
      inst✝ : CategoryTheory.Category.{?u.14501, u_2} D
      X : CategoryTheory.SimplicialObject.Augmented C
      ed : SimplicialObject.Augmented.ExtraDegeneracy X
      F : CategoryTheory.Functor C D
      n : Nat
      ⊢ Eq (CategoryTheory.CategoryStruct.comp ((fun n => F.map (ed.s n)) n) ((((Cat …
    -/
    dsimp
    /-
      C : Type u_1
      inst✝¹ : CategoryTheory.Category.{?u.14494, u_1} C
      D : Type u_2
      inst✝ : CategoryTheory.Category.{?u.14501, u_2} D
      X : CategoryTheory.SimplicialObject.Augmented C
      ed : SimplicialObject.Augmented.ExtraDegeneracy X
      F : CategoryTheory.Functor C D
      n : Nat
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (F.map (ed.s n)) ((((CategoryTheory.S …
    -/
    erw [← F.map_comp, ed.s_comp_δ₀, F.map_id]
    /-
      🎉 no goals
    -/
  s_comp_δ n i := by
    /-
      C : Type u_1
      inst✝¹ : CategoryTheory.Category.{?u.14494, u_1} C
      D : Type u_2
      inst✝ : CategoryTheory.Category.{?u.14501, u_2} D
      X : CategoryTheory.SimplicialObject.Augmented C
      ed : SimplicialObject.Augmented.ExtraDegeneracy X
      F : CategoryTheory.Functor C D
      n : Nat
      i : Fin (HAdd.hAdd n 2)
      ⊢ Eq (CategoryTheory.CategoryStruct.comp ((fun n => F.map (ed.s n)) (HAdd.hAdd …
    -/
    dsimp
    /-
      C : Type u_1
      inst✝¹ : CategoryTheory.Category.{?u.14494, u_1} C
      D : Type u_2
      inst✝ : CategoryTheory.Category.{?u.14501, u_2} D
      X : CategoryTheory.SimplicialObject.Augmented C
      ed : SimplicialObject.Augmented.ExtraDegeneracy X
      F : CategoryTheory.Functor C D
      n : Nat
      i : Fin (HAdd.hAdd n 2)
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (F.map (ed.s (HAdd.hAdd n 1))) ((((Ca …
    -/
    erw [← F.map_comp, ← F.map_comp, ed.s_comp_δ]
    /-
      C : Type u_1
      inst✝¹ : CategoryTheory.Category.{?u.14494, u_1} C
      D : Type u_2
      inst✝ : CategoryTheory.Category.{?u.14501, u_2} D
      X : CategoryTheory.SimplicialObject.Augmented C
      ed : SimplicialObject.Augmented.ExtraDegeneracy X
      F : CategoryTheory.Functor C D
      n : Nat
      i : Fin (HAdd.hAdd n 2)
      ⊢ Eq (F.map (CategoryTheory.CategoryStruct.comp (X.left.δ i) (ed.s n))) (F.map …
    -/
    rfl
    /-
      🎉 no goals
    -/
  s_comp_σ n i := by
    /-
      C : Type u_1
      inst✝¹ : CategoryTheory.Category.{?u.14494, u_1} C
      D : Type u_2
      inst✝ : CategoryTheory.Category.{?u.14501, u_2} D
      X : CategoryTheory.SimplicialObject.Augmented C
      ed : SimplicialObject.Augmented.ExtraDegeneracy X
      F : CategoryTheory.Functor C D
      n : Nat
      i : Fin (HAdd.hAdd n 1)
      ⊢ Eq (CategoryTheory.CategoryStruct.comp ((fun n => F.map (ed.s n)) n) ((((Cat …
    -/
    dsimp
    /-
      C : Type u_1
      inst✝¹ : CategoryTheory.Category.{?u.14494, u_1} C
      D : Type u_2
      inst✝ : CategoryTheory.Category.{?u.14501, u_2} D
      X : CategoryTheory.SimplicialObject.Augmented C
      ed : SimplicialObject.Augmented.ExtraDegeneracy X
      F : CategoryTheory.Functor C D
      n : Nat
      i : Fin (HAdd.hAdd n 1)
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (F.map (ed.s n)) ((((CategoryTheory.S …
    -/
    erw [← F.map_comp, ← F.map_comp, ed.s_comp_σ]
    /-
      C : Type u_1
      inst✝¹ : CategoryTheory.Category.{?u.14494, u_1} C
      D : Type u_2
      inst✝ : CategoryTheory.Category.{?u.14501, u_2} D
      X : CategoryTheory.SimplicialObject.Augmented C
      ed : SimplicialObject.Augmented.ExtraDegeneracy X
      F : CategoryTheory.Functor C D
      n : Nat
      i : Fin (HAdd.hAdd n 1)
      ⊢ Eq (F.map (CategoryTheory.CategoryStruct.comp (X.left.σ i) (ed.s (HAdd.hAdd  …
    -/
    rfl
    /-
      🎉 no goals
    -/


/-- If `X` and `Y` are isomorphic augmented simplicial objects, then an extra
degeneracy for `X` gives also an extra degeneracy for `Y` -/
def ofIso {X Y : SimplicialObject.Augmented C} (e : X ≅ Y) (ed : ExtraDegeneracy X) :
    ExtraDegeneracy Y where
  s' := (point.mapIso e).inv ≫ ed.s' ≫ (drop.mapIso e).hom.app (op [0])
  s n := (drop.mapIso e).inv.app (op [n]) ≫ ed.s n ≫ (drop.mapIso e).hom.app (op [n + 1])
  s'_comp_ε := by
    /-
      C : Type u_1
      inst✝ : CategoryTheory.Category.{?u.23685, u_1} C
      X Y : CategoryTheory.SimplicialObject.Augmented C
      e : CategoryTheory.Iso X Y
      ed : SimplicialObject.Augmented.ExtraDegeneracy X
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.comp ( …
    -/
    simpa only [Functor.mapIso, assoc, w₀, ed.s'_comp_ε_assoc] using (point.mapIso e).inv_hom_id
    /-
      🎉 no goals
    -/
  s₀_comp_δ₁ := by
    /-
      C : Type u_1
      inst✝ : CategoryTheory.Category.{?u.23685, u_1} C
      X Y : CategoryTheory.SimplicialObject.Augmented C
      e : CategoryTheory.Iso X Y
      ed : SimplicialObject.Augmented.ExtraDegeneracy X
      ⊢ Eq (CategoryTheory.CategoryStruct.comp ((fun n => CategoryTheory.CategoryStr …
    -/
    have h := w₀ e.inv
    /-
      C : Type u_1
      inst✝ : CategoryTheory.Category.{?u.23685, u_1} C
      X Y : CategoryTheory.SimplicialObject.Augmented C
      e : CategoryTheory.Iso X Y
      ed : SimplicialObject.Augmented.ExtraDegeneracy X
      h : Eq (CategoryTheory.CategoryStruct.comp ((CategoryTheory.SimplicialObject.A …
      ⊢ Eq (CategoryTheory.CategoryStruct.comp ((fun n => CategoryTheory.CategoryStr …
    -/
    dsimp at h ⊢
    /-
      C : Type u_1
      inst✝ : CategoryTheory.Category.{?u.23685, u_1} C
      X Y : CategoryTheory.SimplicialObject.Augmented C
      e : CategoryTheory.Iso X Y
      ed : SimplicialObject.Augmented.ExtraDegeneracy X
      h : Eq (CategoryTheory.CategoryStruct.comp (e.inv.left.app { unop := SimplexCa …
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.comp ( …
    -/
    simp only [assoc, ← SimplicialObject.δ_naturality, ed.s₀_comp_δ₁_assoc, reassoc_of% h]
    /-
      🎉 no goals
    -/
  s_comp_δ₀ n := by
    /-
      C : Type u_1
      inst✝ : CategoryTheory.Category.{?u.23685, u_1} C
      X Y : CategoryTheory.SimplicialObject.Augmented C
      e : CategoryTheory.Iso X Y
      ed : SimplicialObject.Augmented.ExtraDegeneracy X
      n : Nat
      ⊢ Eq (CategoryTheory.CategoryStruct.comp ((fun n => CategoryTheory.CategoryStr …
    -/
    have h := ed.s_comp_δ₀
    /-
      C : Type u_1
      inst✝ : CategoryTheory.Category.{?u.23685, u_1} C
      X Y : CategoryTheory.SimplicialObject.Augmented C
      e : CategoryTheory.Iso X Y
      ed : SimplicialObject.Augmented.ExtraDegeneracy X
      n : Nat
      h : ∀ (n : Nat), Eq (CategoryTheory.CategoryStruct.comp (ed.s n) (X.left.δ 0)) …
      ⊢ Eq (CategoryTheory.CategoryStruct.comp ((fun n => CategoryTheory.CategoryStr …
    -/
    dsimp at h ⊢
    simpa only [assoc, ← SimplicialObject.δ_naturality, reassoc_of% h] using
      congr_app (drop.mapIso e).inv_hom_id (op [n])
  s_comp_δ n i := by
    /-
      C : Type u_1
      inst✝ : CategoryTheory.Category.{?u.23685, u_1} C
      X Y : CategoryTheory.SimplicialObject.Augmented C
      e : CategoryTheory.Iso X Y
      ed : SimplicialObject.Augmented.ExtraDegeneracy X
      n : Nat
      i : Fin (HAdd.hAdd n 2)
      ⊢ Eq (CategoryTheory.CategoryStruct.comp ((fun n => CategoryTheory.CategoryStr …
    -/
    have h := ed.s_comp_δ n i
    /-
      C : Type u_1
      inst✝ : CategoryTheory.Category.{?u.23685, u_1} C
      X Y : CategoryTheory.SimplicialObject.Augmented C
      e : CategoryTheory.Iso X Y
      ed : SimplicialObject.Augmented.ExtraDegeneracy X
      n : Nat
      i : Fin (HAdd.hAdd n 2)
      h : Eq (CategoryTheory.CategoryStruct.comp (ed.s (HAdd.hAdd n 1)) (X.left.δ i. …
      ⊢ Eq (CategoryTheory.CategoryStruct.comp ((fun n => CategoryTheory.CategoryStr …
    -/
    dsimp at h ⊢
    simp only [assoc, ← SimplicialObject.δ_naturality, reassoc_of% h,
      ← SimplicialObject.δ_naturality_assoc]
  s_comp_σ n i := by
    /-
      C : Type u_1
      inst✝ : CategoryTheory.Category.{?u.23685, u_1} C
      X Y : CategoryTheory.SimplicialObject.Augmented C
      e : CategoryTheory.Iso X Y
      ed : SimplicialObject.Augmented.ExtraDegeneracy X
      n : Nat
      i : Fin (HAdd.hAdd n 1)
      ⊢ Eq (CategoryTheory.CategoryStruct.comp ((fun n => CategoryTheory.CategoryStr …
    -/
    have h := ed.s_comp_σ n i
    /-
      C : Type u_1
      inst✝ : CategoryTheory.Category.{?u.23685, u_1} C
      X Y : CategoryTheory.SimplicialObject.Augmented C
      e : CategoryTheory.Iso X Y
      ed : SimplicialObject.Augmented.ExtraDegeneracy X
      n : Nat
      i : Fin (HAdd.hAdd n 1)
      h : Eq (CategoryTheory.CategoryStruct.comp (ed.s n) (X.left.σ i.succ)) (Catego …
      ⊢ Eq (CategoryTheory.CategoryStruct.comp ((fun n => CategoryTheory.CategoryStr …
    -/
    dsimp at h ⊢
    simp only [assoc, ← SimplicialObject.σ_naturality, reassoc_of% h,
      ← SimplicialObject.σ_naturality_assoc]


/-- When `[HasZero X]`, the shift of a map `f : Fin n → X`
is a map `Fin (n+1) → X` which sends `0` to `0` and `i.succ` to `f i`. -/
def shiftFun {n : ℕ} {X : Type*} [Zero X] (f : Fin n → X) (i : Fin (n + 1)) : X :=
  dite (i = 0) (fun _ => 0) fun h => f (i.pred h)


@[simp]
theorem shiftFun_0 {n : ℕ} {X : Type*} [Zero X] (f : Fin n → X) : shiftFun f 0 = 0 :=
  rfl


@[simp]
theorem shiftFun_succ {n : ℕ} {X : Type*} [Zero X] (f : Fin n → X) (i : Fin n) :
    shiftFun f i.succ = f i := by
  /-
    n : Nat
    X : Type u_1
    inst✝ : Zero X
    f : Fin n → X
    i : Fin n
    ⊢ Eq (SSet.Augmented.StandardSimplex.shiftFun f i.succ) (f i)
  -/
  dsimp [shiftFun]
  /-
    n : Nat
    X : Type u_1
    inst✝ : Zero X
    f : Fin n → X
    i : Fin n
    ⊢ Eq (dite (Eq i.succ 0) (fun x => 0) fun h => f (i.succ.pred h)) (f i)
  -/
  split_ifs with h
    /-
      case pos
      n : Nat
      X : Type u_1
      inst✝ : Zero X
      f : Fin n → X
      i : Fin n
      h : Eq i.succ 0
      ⊢ Eq 0 (f i)
    -/
  · exfalso
    /-
      case pos
      n : Nat
      X : Type u_1
      inst✝ : Zero X
      f : Fin n → X
      i : Fin n
      h : Eq i.succ 0
      ⊢ False
    -/
    simp only [Fin.ext_iff, Fin.val_succ, Fin.val_zero, add_eq_zero, and_false, reduceCtorEq] at h
    /-
      🎉 no goals
    -/
    /-
      case neg
      n : Nat
      X : Type u_1
      inst✝ : Zero X
      f : Fin n → X
      i : Fin n
      h : Not (Eq i.succ 0)
      ⊢ Eq (f (i.succ.pred h)) (f i)
    -/
  · simp only [Fin.pred_succ]
    /-
      🎉 no goals
    -/


/-- The shift of a morphism `f : [n] → Δ` in `SimplexCategory` corresponds to
the monotone map which sends `0` to `0` and `i.succ` to `f.toOrderHom i`. -/
@[simp]
def shift {n : ℕ} {Δ : SimplexCategory}
    (f : ([n] : SimplexCategory) ⟶ Δ) : ([n + 1] : SimplexCategory) ⟶ Δ :=
  SimplexCategory.Hom.mk
    { toFun := shiftFun f.toOrderHom
      monotone' := fun i₁ i₂ hi => by
        /-
          n : Nat
          Δ : SimplexCategory
          f : Quiver.Hom (SimplexCategory.mk n) Δ
          i₁ i₂ : Fin (HAdd.hAdd (SimplexCategory.mk (HAdd.hAdd n 1)).len 1)
          hi : LE.le i₁ i₂
          ⊢ LE.le (SSet.Augmented.StandardSimplex.shiftFun (⇑(SimplexCategory.Hom.toOrde …
        -/
        by_cases h₁ : i₁ = 0
          /-
            case pos
            n : Nat
            Δ : SimplexCategory
            f : Quiver.Hom (SimplexCategory.mk n) Δ
            i₁ i₂ : Fin (HAdd.hAdd (SimplexCategory.mk (HAdd.hAdd n 1)).len 1)
            hi : LE.le i₁ i₂
            h₁ : Eq i₁ 0
            ⊢ LE.le (SSet.Augmented.StandardSimplex.shiftFun (⇑(SimplexCategory.Hom.toOrde …
          -/
        · subst h₁
          /-
            case pos
            n : Nat
            Δ : SimplexCategory
            f : Quiver.Hom (SimplexCategory.mk n) Δ
            i₂ : Fin (HAdd.hAdd (SimplexCategory.mk (HAdd.hAdd n 1)).len 1)
            hi : LE.le 0 i₂
            ⊢ LE.le (SSet.Augmented.StandardSimplex.shiftFun (⇑(SimplexCategory.Hom.toOrde …
          -/
          simp only [shiftFun_0, Fin.zero_le]
          /-
            🎉 no goals
          -/
        · have h₂ : i₂ ≠ 0 := by
            intro h₂
            subst h₂
            exact h₁ (le_antisymm hi (Fin.zero_le _))
          /-
            case neg
            n : Nat
            Δ : SimplexCategory
            f : Quiver.Hom (SimplexCategory.mk n) Δ
            i₁ i₂ : Fin (HAdd.hAdd (SimplexCategory.mk (HAdd.hAdd n 1)).len 1)
            hi : LE.le i₁ i₂
            h₁ : Not (Eq i₁ 0)
            h₂ : Ne i₂ 0
            ⊢ LE.le (SSet.Augmented.StandardSimplex.shiftFun (⇑(SimplexCategory.Hom.toOrde …
          -/
          cases' Fin.eq_succ_of_ne_zero h₁ with j₁ hj₁
          /-
            case neg.intro
            n : Nat
            Δ : SimplexCategory
            f : Quiver.Hom (SimplexCategory.mk n) Δ
            i₁ i₂ : Fin (HAdd.hAdd (SimplexCategory.mk (HAdd.hAdd n 1)).len 1)
            hi : LE.le i₁ i₂
            h₁ : Not (Eq i₁ 0)
            h₂ : Ne i₂ 0
            j₁ : Fin (SimplexCategory.mk (HAdd.hAdd n 1)).len
            hj₁ : Eq i₁ j₁.succ
            ⊢ LE.le (SSet.Augmented.StandardSimplex.shiftFun (⇑(SimplexCategory.Hom.toOrde …
          -/
          cases' Fin.eq_succ_of_ne_zero h₂ with j₂ hj₂
          /-
            case neg.intro.intro
            n : Nat
            Δ : SimplexCategory
            f : Quiver.Hom (SimplexCategory.mk n) Δ
            i₁ i₂ : Fin (HAdd.hAdd (SimplexCategory.mk (HAdd.hAdd n 1)).len 1)
            hi : LE.le i₁ i₂
            h₁ : Not (Eq i₁ 0)
            h₂ : Ne i₂ 0
            j₁ : Fin (SimplexCategory.mk (HAdd.hAdd n 1)).len
            hj₁ : Eq i₁ j₁.succ
            j₂ : Fin (SimplexCategory.mk (HAdd.hAdd n 1)).len
            hj₂ : Eq i₂ j₂.succ
            ⊢ LE.le (SSet.Augmented.StandardSimplex.shiftFun (⇑(SimplexCategory.Hom.toOrde …
          -/
          substs hj₁ hj₂
          /-
            case neg.intro.intro
            n : Nat
            Δ : SimplexCategory
            f : Quiver.Hom (SimplexCategory.mk n) Δ
            j₁ j₂ : Fin (SimplexCategory.mk (HAdd.hAdd n 1)).len
            h₁ : Not (Eq j₁.succ 0)
            h₂ : Ne j₂.succ 0
            hi : LE.le j₁.succ j₂.succ
            ⊢ LE.le (SSet.Augmented.StandardSimplex.shiftFun (⇑(SimplexCategory.Hom.toOrde …
          -/
          simpa only [shiftFun_succ] using f.toOrderHom.monotone (Fin.succ_le_succ_iff.mp hi) }
          /-
            🎉 no goals
          -/


open SSet.standardSimplex in
/-- The obvious extra degeneracy on the standard simplex. -/
protected noncomputable def extraDegeneracy (Δ : SimplexCategory) :
    SimplicialObject.Augmented.ExtraDegeneracy (standardSimplex.obj Δ) where
  s' _ := objMk (OrderHom.const _ 0)
  s  _ f := (objEquiv _ _).symm
    (shift (objEquiv _ _ f))
  s'_comp_ε := by
    /-
      Δ : SimplexCategory
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (fun x => SSet.standardSimplex.objMk  …
    -/
    dsimp
    /-
      Δ : SimplexCategory
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (fun x => SSet.standardSimplex.objMk  …
    -/
    subsingleton
    /-
      🎉 no goals
    -/
  s₀_comp_δ₁ := by
    /-
      Δ : SimplexCategory
      ⊢ Eq (CategoryTheory.CategoryStruct.comp ((fun x f => (SSet.standardSimplex.ob …
    -/
    dsimp
    /-
      Δ : SimplexCategory
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (fun f => (SSet.standardSimplex.objEq …
    -/
    ext1 x
    /-
      case h
      Δ : SimplexCategory
      x : (SSet.standardSimplex.obj Δ).obj { unop := SimplexCategory.mk 0 }
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (fun f => (SSet.standardSimplex.objEq …
    -/
    apply (objEquiv _ _).injective
    /-
      case h.a
      Δ : SimplexCategory
      x : (SSet.standardSimplex.obj Δ).obj { unop := SimplexCategory.mk 0 }
      ⊢ Eq ((SSet.standardSimplex.objEquiv Δ { unop := SimplexCategory.mk 0 }) (Cate …
    -/
    ext j
    /-
      case h.a.a.h.h.h
      Δ : SimplexCategory
      x : (SSet.standardSimplex.obj Δ).obj { unop := SimplexCategory.mk 0 }
      j : Fin (HAdd.hAdd (Opposite.unop { unop := SimplexCategory.mk 0 }).len 1)
      ⊢ Eq ↑((SimplexCategory.Hom.toOrderHom ((SSet.standardSimplex.objEquiv Δ { uno …
    -/
    fin_cases j
    /-
      case h.a.a.h.h.h.«0»
      Δ : SimplexCategory
      x : (SSet.standardSimplex.obj Δ).obj { unop := SimplexCategory.mk 0 }
      ⊢ Eq ↑((SimplexCategory.Hom.toOrderHom ((SSet.standardSimplex.objEquiv Δ { uno …
    -/
    rfl
    /-
      🎉 no goals
    -/
  s_comp_δ₀ n := by
    /-
      Δ : SimplexCategory
      n : Nat
      ⊢ Eq (CategoryTheory.CategoryStruct.comp ((fun x f => (SSet.standardSimplex.ob …
    -/
    ext1 φ
    /-
      case h
      Δ : SimplexCategory
      n : Nat
      φ : (CategoryTheory.SimplicialObject.Augmented.drop.obj (SSet.Augmented.standa …
      ⊢ Eq (CategoryTheory.CategoryStruct.comp ((fun x f => (SSet.standardSimplex.ob …
    -/
    apply (objEquiv _ _).injective
    /-
      case h.a
      Δ : SimplexCategory
      n : Nat
      φ : (CategoryTheory.SimplicialObject.Augmented.drop.obj (SSet.Augmented.standa …
      ⊢ Eq ((SSet.standardSimplex.objEquiv Δ { unop := SimplexCategory.mk n }) (Cate …
    -/
    apply SimplexCategory.Hom.ext
    /-
      case h.a.a
      Δ : SimplexCategory
      n : Nat
      φ : (CategoryTheory.SimplicialObject.Augmented.drop.obj (SSet.Augmented.standa …
      ⊢ Eq (SimplexCategory.Hom.toOrderHom ((SSet.standardSimplex.objEquiv Δ { unop  …
    -/
    ext i : 2
    dsimp [SimplicialObject.δ, SimplexCategory.δ, SSet.standardSimplex,
      objEquiv, Equiv.ulift, uliftFunctor]
    /-
      case h.a.a.h.h
      Δ : SimplexCategory
      n : Nat
      φ : (CategoryTheory.SimplicialObject.Augmented.drop.obj (SSet.Augmented.standa …
      i : Fin (HAdd.hAdd (Opposite.unop { unop := SimplexCategory.mk n }).len 1)
      ⊢ Eq (SSet.Augmented.StandardSimplex.shiftFun (⇑(SimplexCategory.Hom.toOrderHo …
    -/
    simp only [shiftFun_succ]
    /-
      🎉 no goals
    -/
  s_comp_δ n i := by
    /-
      Δ : SimplexCategory
      n : Nat
      i : Fin (HAdd.hAdd n 2)
      ⊢ Eq (CategoryTheory.CategoryStruct.comp ((fun x f => (SSet.standardSimplex.ob …
    -/
    ext1 φ
    /-
      case h
      Δ : SimplexCategory
      n : Nat
      i : Fin (HAdd.hAdd n 2)
      φ : (CategoryTheory.SimplicialObject.Augmented.drop.obj (SSet.Augmented.standa …
      ⊢ Eq (CategoryTheory.CategoryStruct.comp ((fun x f => (SSet.standardSimplex.ob …
    -/
    apply (objEquiv _ _).injective
    /-
      case h.a
      Δ : SimplexCategory
      n : Nat
      i : Fin (HAdd.hAdd n 2)
      φ : (CategoryTheory.SimplicialObject.Augmented.drop.obj (SSet.Augmented.standa …
      ⊢ Eq ((SSet.standardSimplex.objEquiv Δ { unop := SimplexCategory.mk (HAdd.hAdd …
    -/
    apply SimplexCategory.Hom.ext
    /-
      case h.a.a
      Δ : SimplexCategory
      n : Nat
      i : Fin (HAdd.hAdd n 2)
      φ : (CategoryTheory.SimplicialObject.Augmented.drop.obj (SSet.Augmented.standa …
      ⊢ Eq (SimplexCategory.Hom.toOrderHom ((SSet.standardSimplex.objEquiv Δ { unop  …
    -/
    ext j : 2
    dsimp [SimplicialObject.δ, SimplexCategory.δ, SSet.standardSimplex,
      objEquiv, Equiv.ulift, uliftFunctor]
    /-
      case h.a.a.h.h
      Δ : SimplexCategory
      n : Nat
      i : Fin (HAdd.hAdd n 2)
      φ : (CategoryTheory.SimplicialObject.Augmented.drop.obj (SSet.Augmented.standa …
      j : Fin (HAdd.hAdd (Opposite.unop { unop := SimplexCategory.mk (HAdd.hAdd n 1) …
      ⊢ Eq (SSet.Augmented.StandardSimplex.shiftFun (⇑(SimplexCategory.Hom.toOrderHo …
    -/
    by_cases h : j = 0
      /-
        case pos
        Δ : SimplexCategory
        n : Nat
        i : Fin (HAdd.hAdd n 2)
        φ : (CategoryTheory.SimplicialObject.Augmented.drop.obj (SSet.Augmented.standa …
        j : Fin (HAdd.hAdd (Opposite.unop { unop := SimplexCategory.mk (HAdd.hAdd n 1) …
        h : Eq j 0
        ⊢ Eq (SSet.Augmented.StandardSimplex.shiftFun (⇑(SimplexCategory.Hom.toOrderHo …
      -/
    · subst h
      /-
        case pos
        Δ : SimplexCategory
        n : Nat
        i : Fin (HAdd.hAdd n 2)
        φ : (CategoryTheory.SimplicialObject.Augmented.drop.obj (SSet.Augmented.standa …
        ⊢ Eq (SSet.Augmented.StandardSimplex.shiftFun (⇑(SimplexCategory.Hom.toOrderHo …
      -/
      simp only [Fin.succ_succAbove_zero, shiftFun_0]
      /-
        🎉 no goals
      -/
      /-
        case neg
        Δ : SimplexCategory
        n : Nat
        i : Fin (HAdd.hAdd n 2)
        φ : (CategoryTheory.SimplicialObject.Augmented.drop.obj (SSet.Augmented.standa …
        j : Fin (HAdd.hAdd (Opposite.unop { unop := SimplexCategory.mk (HAdd.hAdd n 1) …
        h : Not (Eq j 0)
        ⊢ Eq (SSet.Augmented.StandardSimplex.shiftFun (⇑(SimplexCategory.Hom.toOrderHo …
      -/
    · obtain ⟨_, rfl⟩ := Fin.eq_succ_of_ne_zero <| h
      simp only [Fin.succ_succAbove_succ, shiftFun_succ, Function.comp_apply,
        Fin.succAboveOrderEmb_apply]
  s_comp_σ n i := by
    /-
      Δ : SimplexCategory
      n : Nat
      i : Fin (HAdd.hAdd n 1)
      ⊢ Eq (CategoryTheory.CategoryStruct.comp ((fun x f => (SSet.standardSimplex.ob …
    -/
    ext1 φ
    /-
      case h
      Δ : SimplexCategory
      n : Nat
      i : Fin (HAdd.hAdd n 1)
      φ : (CategoryTheory.SimplicialObject.Augmented.drop.obj (SSet.Augmented.standa …
      ⊢ Eq (CategoryTheory.CategoryStruct.comp ((fun x f => (SSet.standardSimplex.ob …
    -/
    apply (objEquiv _ _).injective
    /-
      case h.a
      Δ : SimplexCategory
      n : Nat
      i : Fin (HAdd.hAdd n 1)
      φ : (CategoryTheory.SimplicialObject.Augmented.drop.obj (SSet.Augmented.standa …
      ⊢ Eq ((SSet.standardSimplex.objEquiv Δ { unop := SimplexCategory.mk (HAdd.hAdd …
    -/
    apply SimplexCategory.Hom.ext
    /-
      case h.a.a
      Δ : SimplexCategory
      n : Nat
      i : Fin (HAdd.hAdd n 1)
      φ : (CategoryTheory.SimplicialObject.Augmented.drop.obj (SSet.Augmented.standa …
      ⊢ Eq (SimplexCategory.Hom.toOrderHom ((SSet.standardSimplex.objEquiv Δ { unop  …
    -/
    ext j : 2
    dsimp [SimplicialObject.σ, SimplexCategory.σ, SSet.standardSimplex,
      objEquiv, Equiv.ulift, uliftFunctor]
    /-
      case h.a.a.h.h
      Δ : SimplexCategory
      n : Nat
      i : Fin (HAdd.hAdd n 1)
      φ : (CategoryTheory.SimplicialObject.Augmented.drop.obj (SSet.Augmented.standa …
      j : Fin (HAdd.hAdd (Opposite.unop { unop := SimplexCategory.mk (HAdd.hAdd (HAd …
      ⊢ Eq (SSet.Augmented.StandardSimplex.shiftFun (⇑(SimplexCategory.Hom.toOrderHo …
    -/
    by_cases h : j = 0
      /-
        case pos
        Δ : SimplexCategory
        n : Nat
        i : Fin (HAdd.hAdd n 1)
        φ : (CategoryTheory.SimplicialObject.Augmented.drop.obj (SSet.Augmented.standa …
        j : Fin (HAdd.hAdd (Opposite.unop { unop := SimplexCategory.mk (HAdd.hAdd (HAd …
        h : Eq j 0
        ⊢ Eq (SSet.Augmented.StandardSimplex.shiftFun (⇑(SimplexCategory.Hom.toOrderHo …
      -/
    · subst h
      /-
        case pos
        Δ : SimplexCategory
        n : Nat
        i : Fin (HAdd.hAdd n 1)
        φ : (CategoryTheory.SimplicialObject.Augmented.drop.obj (SSet.Augmented.standa …
        ⊢ Eq (SSet.Augmented.StandardSimplex.shiftFun (⇑(SimplexCategory.Hom.toOrderHo …
      -/
      rfl
      /-
        🎉 no goals
      -/
      /-
        case neg
        Δ : SimplexCategory
        n : Nat
        i : Fin (HAdd.hAdd n 1)
        φ : (CategoryTheory.SimplicialObject.Augmented.drop.obj (SSet.Augmented.standa …
        j : Fin (HAdd.hAdd (Opposite.unop { unop := SimplexCategory.mk (HAdd.hAdd (HAd …
        h : Not (Eq j 0)
        ⊢ Eq (SSet.Augmented.StandardSimplex.shiftFun (⇑(SimplexCategory.Hom.toOrderHo …
      -/
    · obtain ⟨_, rfl⟩ := Fin.eq_succ_of_ne_zero h
      /-
        case neg.intro
        Δ : SimplexCategory
        n : Nat
        i : Fin (HAdd.hAdd n 1)
        φ : (CategoryTheory.SimplicialObject.Augmented.drop.obj (SSet.Augmented.standa …
        w✝ : Fin (Opposite.unop { unop := SimplexCategory.mk (HAdd.hAdd (HAdd.hAdd n 1 …
        h : Not (Eq w✝.succ 0)
        ⊢ Eq (SSet.Augmented.StandardSimplex.shiftFun (⇑(SimplexCategory.Hom.toOrderHo …
      -/
      simp only [Fin.succ_predAbove_succ, shiftFun_succ, Function.comp_apply]
      /-
        🎉 no goals
      -/


instance nonempty_extraDegeneracy_standardSimplex (Δ : SimplexCategory) :
    Nonempty (SimplicialObject.Augmented.ExtraDegeneracy (standardSimplex.obj Δ)) :=
  ⟨StandardSimplex.extraDegeneracy Δ⟩


/-- The extra degeneracy map on the Čech nerve of a split epi. It is
given on the `0`-projection by the given section of the split epi,
and by shifting the indices on the other projections. -/
noncomputable def ExtraDegeneracy.s (n : ℕ) :
    f.cechNerve.obj (op [n]) ⟶ f.cechNerve.obj (op [n + 1]) :=
  WidePullback.lift (WidePullback.base _)
    (fun i =>
      dite (i = 0)
        (fun _ => WidePullback.base _ ≫ S.section_)
        (fun h => WidePullback.π _ (i.pred h)))
    fun i => by
      /-
        C : Type u_1
        inst✝¹ : CategoryTheory.Category.{?u.50211, u_1} C
        f : CategoryTheory.Arrow C
        inst✝ : ∀ (n : Nat), CategoryTheory.Limits.HasWidePullback f.right (fun x => f …
        S : CategoryTheory.SplitEpi f.hom
        n : Nat
        i : Fin (HAdd.hAdd (Opposite.unop { unop := SimplexCategory.mk (HAdd.hAdd n 1) …
        ⊢ Eq (CategoryTheory.CategoryStruct.comp ((fun i => dite (Eq i 0) (fun x => Ca …
      -/
      dsimp
      /-
        C : Type u_1
        inst✝¹ : CategoryTheory.Category.{?u.50211, u_1} C
        f : CategoryTheory.Arrow C
        inst✝ : ∀ (n : Nat), CategoryTheory.Limits.HasWidePullback f.right (fun x => f …
        S : CategoryTheory.SplitEpi f.hom
        n : Nat
        i : Fin (HAdd.hAdd (Opposite.unop { unop := SimplexCategory.mk (HAdd.hAdd n 1) …
        ⊢ Eq (CategoryTheory.CategoryStruct.comp (dite (Eq i 0) (fun x => CategoryTheo …
      -/
      split_ifs with h
        /-
          case pos
          C : Type u_1
          inst✝¹ : CategoryTheory.Category.{?u.50211, u_1} C
          f : CategoryTheory.Arrow C
          inst✝ : ∀ (n : Nat), CategoryTheory.Limits.HasWidePullback f.right (fun x => f …
          S : CategoryTheory.SplitEpi f.hom
          n : Nat
          i : Fin (HAdd.hAdd (Opposite.unop { unop := SimplexCategory.mk (HAdd.hAdd n 1) …
          h : Eq i 0
          ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.comp ( …
        -/
      · subst h
        /-
          case pos
          C : Type u_1
          inst✝¹ : CategoryTheory.Category.{?u.50211, u_1} C
          f : CategoryTheory.Arrow C
          inst✝ : ∀ (n : Nat), CategoryTheory.Limits.HasWidePullback f.right (fun x => f …
          S : CategoryTheory.SplitEpi f.hom
          n : Nat
          ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.comp ( …
        -/
        simp only [assoc, SplitEpi.id, comp_id]
        /-
          🎉 no goals
        -/
        /-
          case neg
          C : Type u_1
          inst✝¹ : CategoryTheory.Category.{?u.50211, u_1} C
          f : CategoryTheory.Arrow C
          inst✝ : ∀ (n : Nat), CategoryTheory.Limits.HasWidePullback f.right (fun x => f …
          S : CategoryTheory.SplitEpi f.hom
          n : Nat
          i : Fin (HAdd.hAdd (Opposite.unop { unop := SimplexCategory.mk (HAdd.hAdd n 1) …
          h : Not (Eq i 0)
          ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.WidePullback.π …
        -/
      · simp only [WidePullback.π_arrow]
        /-
          🎉 no goals
        -/

-- Porting note (https://github.com/leanprover-community/mathlib4/issues/11119): @[simp] removed as the linter complains the LHS is not in normal form

theorem ExtraDegeneracy.s_comp_π_0 (n : ℕ) :
    ExtraDegeneracy.s f S n ≫ WidePullback.π _ 0 =
      @WidePullback.base _ _ _ f.right (fun _ : Fin (n + 1) => f.left) (fun _ => f.hom) _ ≫
        S.section_ := by
  /-
    C : Type u_1
    inst✝¹ : CategoryTheory.Category.{u_2, u_1} C
    f : CategoryTheory.Arrow C
    inst✝ : ∀ (n : Nat), CategoryTheory.Limits.HasWidePullback f.right (fun x => f …
    S : CategoryTheory.SplitEpi f.hom
    n : Nat
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Arrow.AugmentedCechNe …
  -/
  dsimp [ExtraDegeneracy.s]
  /-
    C : Type u_1
    inst✝¹ : CategoryTheory.Category.{u_2, u_1} C
    f : CategoryTheory.Arrow C
    inst✝ : ∀ (n : Nat), CategoryTheory.Limits.HasWidePullback f.right (fun x => f …
    S : CategoryTheory.SplitEpi f.hom
    n : Nat
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.WidePullback.l …
  -/
  simp only [WidePullback.lift_π]
  /-
    C : Type u_1
    inst✝¹ : CategoryTheory.Category.{u_2, u_1} C
    f : CategoryTheory.Arrow C
    inst✝ : ∀ (n : Nat), CategoryTheory.Limits.HasWidePullback f.right (fun x => f …
    S : CategoryTheory.SplitEpi f.hom
    n : Nat
    ⊢ Eq (dite True (fun h => CategoryTheory.CategoryStruct.comp (CategoryTheory.L …
  -/
  rfl
  /-
    🎉 no goals
  -/

-- Porting note (https://github.com/leanprover-community/mathlib4/issues/11119): @[simp] removed as the linter complains the LHS is not in normal form

theorem ExtraDegeneracy.s_comp_π_succ (n : ℕ) (i : Fin (n + 1)) :
    ExtraDegeneracy.s f S n ≫ WidePullback.π _ i.succ =
      @WidePullback.π _ _ _ f.right (fun _ : Fin (n + 1) => f.left) (fun _ => f.hom) _ i := by
  /-
    C : Type u_1
    inst✝¹ : CategoryTheory.Category.{u_2, u_1} C
    f : CategoryTheory.Arrow C
    inst✝ : ∀ (n : Nat), CategoryTheory.Limits.HasWidePullback f.right (fun x => f …
    S : CategoryTheory.SplitEpi f.hom
    n : Nat
    i : Fin (HAdd.hAdd n 1)
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Arrow.AugmentedCechNe …
  -/
  dsimp [ExtraDegeneracy.s]
  /-
    C : Type u_1
    inst✝¹ : CategoryTheory.Category.{u_2, u_1} C
    f : CategoryTheory.Arrow C
    inst✝ : ∀ (n : Nat), CategoryTheory.Limits.HasWidePullback f.right (fun x => f …
    S : CategoryTheory.SplitEpi f.hom
    n : Nat
    i : Fin (HAdd.hAdd n 1)
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.WidePullback.l …
  -/
  simp only [WidePullback.lift_π]
  /-
    C : Type u_1
    inst✝¹ : CategoryTheory.Category.{u_2, u_1} C
    f : CategoryTheory.Arrow C
    inst✝ : ∀ (n : Nat), CategoryTheory.Limits.HasWidePullback f.right (fun x => f …
    S : CategoryTheory.SplitEpi f.hom
    n : Nat
    i : Fin (HAdd.hAdd n 1)
    ⊢ Eq (dite (Eq i.succ 0) (fun x => CategoryTheory.CategoryStruct.comp (Categor …
  -/
  split_ifs with h
    /-
      case pos
      C : Type u_1
      inst✝¹ : CategoryTheory.Category.{u_2, u_1} C
      f : CategoryTheory.Arrow C
      inst✝ : ∀ (n : Nat), CategoryTheory.Limits.HasWidePullback f.right (fun x => f …
      S : CategoryTheory.SplitEpi f.hom
      n : Nat
      i : Fin (HAdd.hAdd n 1)
      h : Eq i.succ 0
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.WidePullback.b …
    -/
  · simp only [Fin.ext_iff, Fin.val_succ, Fin.val_zero, add_eq_zero, and_false, reduceCtorEq] at h
    /-
      🎉 no goals
    -/
    /-
      case neg
      C : Type u_1
      inst✝¹ : CategoryTheory.Category.{u_2, u_1} C
      f : CategoryTheory.Arrow C
      inst✝ : ∀ (n : Nat), CategoryTheory.Limits.HasWidePullback f.right (fun x => f …
      S : CategoryTheory.SplitEpi f.hom
      n : Nat
      i : Fin (HAdd.hAdd n 1)
      h : Not (Eq i.succ 0)
      ⊢ Eq (CategoryTheory.Limits.WidePullback.π (fun x => f.hom) (i.succ.pred h)) ( …
    -/
  · simp only [Fin.pred_succ]
    /-
      🎉 no goals
    -/

-- Porting note (https://github.com/leanprover-community/mathlib4/issues/11119): @[simp] removed as the linter complains the LHS is not in normal form

theorem ExtraDegeneracy.s_comp_base (n : ℕ) :
    ExtraDegeneracy.s f S n ≫ WidePullback.base _ = WidePullback.base _ := by
  /-
    C : Type u_1
    inst✝¹ : CategoryTheory.Category.{u_2, u_1} C
    f : CategoryTheory.Arrow C
    inst✝ : ∀ (n : Nat), CategoryTheory.Limits.HasWidePullback f.right (fun x => f …
    S : CategoryTheory.SplitEpi f.hom
    n : Nat
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Arrow.AugmentedCechNe …
  -/
  apply WidePullback.lift_base
  /-
    🎉 no goals
  -/


/-- The augmented Čech nerve associated to a split epimorphism has an extra degeneracy. -/
noncomputable def extraDegeneracy :
    SimplicialObject.Augmented.ExtraDegeneracy f.augmentedCechNerve where
                                                                        /-
                                                                          C : Type u_1
                                                                          inst✝¹ : CategoryTheory.Category.{?u.61848, u_1} C
                                                                          f : CategoryTheory.Arrow C
                                                                          inst✝ : ∀ (n : Nat), CategoryTheory.Limits.HasWidePullback f.right (fun x => f …
                                                                          S : CategoryTheory.SplitEpi f.hom
                                                                          i : Fin (HAdd.hAdd (Opposite.unop { unop := SimplexCategory.mk 0 }).len 1)
                                                                          ⊢ Eq (CategoryTheory.CategoryStruct.comp ((fun x => CategoryTheory.CategoryStr …
                                                                        -/
  s' := S.section_ ≫ WidePullback.lift f.hom (fun _ => 𝟙 _) fun i => by rw [id_comp]
                                                                        /-
                                                                          🎉 no goals
                                                                        -/
  s n := ExtraDegeneracy.s f S n
  s'_comp_ε := by
    /-
      C : Type u_1
      inst✝¹ : CategoryTheory.Category.{?u.61848, u_1} C
      f : CategoryTheory.Arrow C
      inst✝ : ∀ (n : Nat), CategoryTheory.Limits.HasWidePullback f.right (fun x => f …
      S : CategoryTheory.SplitEpi f.hom
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.comp S …
    -/
    dsimp
    /-
      C : Type u_1
      inst✝¹ : CategoryTheory.Category.{?u.61848, u_1} C
      f : CategoryTheory.Arrow C
      inst✝ : ∀ (n : Nat), CategoryTheory.Limits.HasWidePullback f.right (fun x => f …
      S : CategoryTheory.SplitEpi f.hom
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.comp S …
    -/
    simp only [augmentedCechNerve_hom_app, assoc, WidePullback.lift_base, SplitEpi.id]
    /-
      🎉 no goals
    -/
  s₀_comp_δ₁ := by
    /-
      C : Type u_1
      inst✝¹ : CategoryTheory.Category.{?u.61848, u_1} C
      f : CategoryTheory.Arrow C
      inst✝ : ∀ (n : Nat), CategoryTheory.Limits.HasWidePullback f.right (fun x => f …
      S : CategoryTheory.SplitEpi f.hom
      ⊢ Eq (CategoryTheory.CategoryStruct.comp ((fun n => CategoryTheory.Arrow.Augme …
    -/
    dsimp [cechNerve, SimplicialObject.δ, SimplexCategory.δ]
    /-
      C : Type u_1
      inst✝¹ : CategoryTheory.Category.{?u.61848, u_1} C
      f : CategoryTheory.Arrow C
      inst✝ : ∀ (n : Nat), CategoryTheory.Limits.HasWidePullback f.right (fun x => f …
      S : CategoryTheory.SplitEpi f.hom
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Arrow.AugmentedCechNe …
    -/
    ext j
      /-
        case a
        C : Type u_1
        inst✝¹ : CategoryTheory.Category.{?u.61848, u_1} C
        f : CategoryTheory.Arrow C
        inst✝ : ∀ (n : Nat), CategoryTheory.Limits.HasWidePullback f.right (fun x => f …
        S : CategoryTheory.SplitEpi f.hom
        j : Fin 1
        ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.comp ( …
      -/
    · fin_cases j
      /-
        case a.«_@».Mathlib.CategoryTheory.Limits.Shapes.WidePullbacks._hyg.4251.«0»
        C : Type u_1
        inst✝¹ : CategoryTheory.Category.{?u.61848, u_1} C
        f : CategoryTheory.Arrow C
        inst✝ : ∀ (n : Nat), CategoryTheory.Limits.HasWidePullback f.right (fun x => f …
        S : CategoryTheory.SplitEpi f.hom
        ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.comp ( …
      -/
      simpa only [assoc, WidePullback.lift_π, comp_id] using ExtraDegeneracy.s_comp_π_0 f S 0
      /-
        🎉 no goals
      -/
    · simpa only [assoc, WidePullback.lift_base, SplitEpi.id, comp_id] using
        ExtraDegeneracy.s_comp_base f S 0
  s_comp_δ₀ n := by
    /-
      C : Type u_1
      inst✝¹ : CategoryTheory.Category.{?u.61848, u_1} C
      f : CategoryTheory.Arrow C
      inst✝ : ∀ (n : Nat), CategoryTheory.Limits.HasWidePullback f.right (fun x => f …
      S : CategoryTheory.SplitEpi f.hom
      n : Nat
      ⊢ Eq (CategoryTheory.CategoryStruct.comp ((fun n => CategoryTheory.Arrow.Augme …
    -/
    dsimp [cechNerve, SimplicialObject.δ, SimplexCategory.δ]
    /-
      C : Type u_1
      inst✝¹ : CategoryTheory.Category.{?u.61848, u_1} C
      f : CategoryTheory.Arrow C
      inst✝ : ∀ (n : Nat), CategoryTheory.Limits.HasWidePullback f.right (fun x => f …
      S : CategoryTheory.SplitEpi f.hom
      n : Nat
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Arrow.AugmentedCechNe …
    -/
    ext j
      /-
        case a
        C : Type u_1
        inst✝¹ : CategoryTheory.Category.{?u.61848, u_1} C
        f : CategoryTheory.Arrow C
        inst✝ : ∀ (n : Nat), CategoryTheory.Limits.HasWidePullback f.right (fun x => f …
        S : CategoryTheory.SplitEpi f.hom
        n : Nat
        j : Fin (HAdd.hAdd n 1)
        ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.comp ( …
      -/
    · simpa only [assoc, WidePullback.lift_π, id_comp] using ExtraDegeneracy.s_comp_π_succ f S n j
      /-
        🎉 no goals
      -/
      /-
        case a
        C : Type u_1
        inst✝¹ : CategoryTheory.Category.{?u.61848, u_1} C
        f : CategoryTheory.Arrow C
        inst✝ : ∀ (n : Nat), CategoryTheory.Limits.HasWidePullback f.right (fun x => f …
        S : CategoryTheory.SplitEpi f.hom
        n : Nat
        ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.comp ( …
      -/
    · simpa only [assoc, WidePullback.lift_base, id_comp] using ExtraDegeneracy.s_comp_base f S n
      /-
        🎉 no goals
      -/
  s_comp_δ n i := by
    /-
      C : Type u_1
      inst✝¹ : CategoryTheory.Category.{?u.61848, u_1} C
      f : CategoryTheory.Arrow C
      inst✝ : ∀ (n : Nat), CategoryTheory.Limits.HasWidePullback f.right (fun x => f …
      S : CategoryTheory.SplitEpi f.hom
      n : Nat
      i : Fin (HAdd.hAdd n 2)
      ⊢ Eq (CategoryTheory.CategoryStruct.comp ((fun n => CategoryTheory.Arrow.Augme …
    -/
    dsimp [cechNerve, SimplicialObject.δ, SimplexCategory.δ]
    /-
      C : Type u_1
      inst✝¹ : CategoryTheory.Category.{?u.61848, u_1} C
      f : CategoryTheory.Arrow C
      inst✝ : ∀ (n : Nat), CategoryTheory.Limits.HasWidePullback f.right (fun x => f …
      S : CategoryTheory.SplitEpi f.hom
      n : Nat
      i : Fin (HAdd.hAdd n 2)
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Arrow.AugmentedCechNe …
    -/
    ext j
      /-
        case a
        C : Type u_1
        inst✝¹ : CategoryTheory.Category.{?u.61848, u_1} C
        f : CategoryTheory.Arrow C
        inst✝ : ∀ (n : Nat), CategoryTheory.Limits.HasWidePullback f.right (fun x => f …
        S : CategoryTheory.SplitEpi f.hom
        n : Nat
        i : Fin (HAdd.hAdd n 2)
        j : Fin (HAdd.hAdd (HAdd.hAdd n 1) 1)
        ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.comp ( …
      -/
    · simp only [assoc, WidePullback.lift_π]
      /-
        case a
        C : Type u_1
        inst✝¹ : CategoryTheory.Category.{?u.61848, u_1} C
        f : CategoryTheory.Arrow C
        inst✝ : ∀ (n : Nat), CategoryTheory.Limits.HasWidePullback f.right (fun x => f …
        S : CategoryTheory.SplitEpi f.hom
        n : Nat
        i : Fin (HAdd.hAdd n 2)
        j : Fin (HAdd.hAdd (HAdd.hAdd n 1) 1)
        ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Arrow.AugmentedCechNe …
      -/
      by_cases h : j = 0
        /-
          case pos
          C : Type u_1
          inst✝¹ : CategoryTheory.Category.{?u.61848, u_1} C
          f : CategoryTheory.Arrow C
          inst✝ : ∀ (n : Nat), CategoryTheory.Limits.HasWidePullback f.right (fun x => f …
          S : CategoryTheory.SplitEpi f.hom
          n : Nat
          i : Fin (HAdd.hAdd n 2)
          j : Fin (HAdd.hAdd (HAdd.hAdd n 1) 1)
          h : Eq j 0
          ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Arrow.AugmentedCechNe …
        -/
      · subst h
        /-
          case pos
          C : Type u_1
          inst✝¹ : CategoryTheory.Category.{?u.61848, u_1} C
          f : CategoryTheory.Arrow C
          inst✝ : ∀ (n : Nat), CategoryTheory.Limits.HasWidePullback f.right (fun x => f …
          S : CategoryTheory.SplitEpi f.hom
          n : Nat
          i : Fin (HAdd.hAdd n 2)
          ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Arrow.AugmentedCechNe …
        -/
        erw [Fin.succ_succAbove_zero, ExtraDegeneracy.s_comp_π_0, ExtraDegeneracy.s_comp_π_0]
        /-
          case pos
          C : Type u_1
          inst✝¹ : CategoryTheory.Category.{?u.61848, u_1} C
          f : CategoryTheory.Arrow C
          inst✝ : ∀ (n : Nat), CategoryTheory.Limits.HasWidePullback f.right (fun x => f …
          S : CategoryTheory.SplitEpi f.hom
          n : Nat
          i : Fin (HAdd.hAdd n 2)
          ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.WidePullback.b …
        -/
        dsimp
        /-
          case pos
          C : Type u_1
          inst✝¹ : CategoryTheory.Category.{?u.61848, u_1} C
          f : CategoryTheory.Arrow C
          inst✝ : ∀ (n : Nat), CategoryTheory.Limits.HasWidePullback f.right (fun x => f …
          S : CategoryTheory.SplitEpi f.hom
          n : Nat
          i : Fin (HAdd.hAdd n 2)
          ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.WidePullback.b …
        -/
        simp only [WidePullback.lift_base_assoc]
        /-
          🎉 no goals
        -/
        /-
          case neg
          C : Type u_1
          inst✝¹ : CategoryTheory.Category.{?u.61848, u_1} C
          f : CategoryTheory.Arrow C
          inst✝ : ∀ (n : Nat), CategoryTheory.Limits.HasWidePullback f.right (fun x => f …
          S : CategoryTheory.SplitEpi f.hom
          n : Nat
          i : Fin (HAdd.hAdd n 2)
          j : Fin (HAdd.hAdd (HAdd.hAdd n 1) 1)
          h : Not (Eq j 0)
          ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Arrow.AugmentedCechNe …
        -/
      · cases' Fin.eq_succ_of_ne_zero h with k hk
        /-
          case neg.intro
          C : Type u_1
          inst✝¹ : CategoryTheory.Category.{?u.61848, u_1} C
          f : CategoryTheory.Arrow C
          inst✝ : ∀ (n : Nat), CategoryTheory.Limits.HasWidePullback f.right (fun x => f …
          S : CategoryTheory.SplitEpi f.hom
          n : Nat
          i : Fin (HAdd.hAdd n 2)
          j : Fin (HAdd.hAdd (HAdd.hAdd n 1) 1)
          h : Not (Eq j 0)
          k : Fin (HAdd.hAdd n 1)
          hk : Eq j k.succ
          ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Arrow.AugmentedCechNe …
        -/
        subst hk
        erw [Fin.succ_succAbove_succ, ExtraDegeneracy.s_comp_π_succ,
          ExtraDegeneracy.s_comp_π_succ]
        /-
          case neg.intro
          C : Type u_1
          inst✝¹ : CategoryTheory.Category.{?u.61848, u_1} C
          f : CategoryTheory.Arrow C
          inst✝ : ∀ (n : Nat), CategoryTheory.Limits.HasWidePullback f.right (fun x => f …
          S : CategoryTheory.SplitEpi f.hom
          n : Nat
          i : Fin (HAdd.hAdd n 2)
          k : Fin (HAdd.hAdd n 1)
          h : Not (Eq k.succ 0)
          ⊢ Eq (CategoryTheory.Limits.WidePullback.π (fun x => f.hom) (i.succAbove k)) ( …
        -/
        simp only [WidePullback.lift_π]
        /-
          🎉 no goals
        -/
      /-
        case a
        C : Type u_1
        inst✝¹ : CategoryTheory.Category.{?u.61848, u_1} C
        f : CategoryTheory.Arrow C
        inst✝ : ∀ (n : Nat), CategoryTheory.Limits.HasWidePullback f.right (fun x => f …
        S : CategoryTheory.SplitEpi f.hom
        n : Nat
        i : Fin (HAdd.hAdd n 2)
        ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.comp ( …
      -/
    · simp only [assoc, WidePullback.lift_base]
      /-
        case a
        C : Type u_1
        inst✝¹ : CategoryTheory.Category.{?u.61848, u_1} C
        f : CategoryTheory.Arrow C
        inst✝ : ∀ (n : Nat), CategoryTheory.Limits.HasWidePullback f.right (fun x => f …
        S : CategoryTheory.SplitEpi f.hom
        n : Nat
        i : Fin (HAdd.hAdd n 2)
        ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Arrow.AugmentedCechNe …
      -/
      erw [ExtraDegeneracy.s_comp_base, ExtraDegeneracy.s_comp_base]
      /-
        case a
        C : Type u_1
        inst✝¹ : CategoryTheory.Category.{?u.61848, u_1} C
        f : CategoryTheory.Arrow C
        inst✝ : ∀ (n : Nat), CategoryTheory.Limits.HasWidePullback f.right (fun x => f …
        S : CategoryTheory.SplitEpi f.hom
        n : Nat
        i : Fin (HAdd.hAdd n 2)
        ⊢ Eq (CategoryTheory.Limits.WidePullback.base fun x => f.hom) (CategoryTheory. …
      -/
      dsimp
      /-
        case a
        C : Type u_1
        inst✝¹ : CategoryTheory.Category.{?u.61848, u_1} C
        f : CategoryTheory.Arrow C
        inst✝ : ∀ (n : Nat), CategoryTheory.Limits.HasWidePullback f.right (fun x => f …
        S : CategoryTheory.SplitEpi f.hom
        n : Nat
        i : Fin (HAdd.hAdd n 2)
        ⊢ Eq (CategoryTheory.Limits.WidePullback.base fun x => f.hom) (CategoryTheory. …
      -/
      simp only [WidePullback.lift_base]
      /-
        🎉 no goals
      -/
  s_comp_σ n i := by
    /-
      C : Type u_1
      inst✝¹ : CategoryTheory.Category.{?u.61848, u_1} C
      f : CategoryTheory.Arrow C
      inst✝ : ∀ (n : Nat), CategoryTheory.Limits.HasWidePullback f.right (fun x => f …
      S : CategoryTheory.SplitEpi f.hom
      n : Nat
      i : Fin (HAdd.hAdd n 1)
      ⊢ Eq (CategoryTheory.CategoryStruct.comp ((fun n => CategoryTheory.Arrow.Augme …
    -/
    dsimp [cechNerve, SimplicialObject.σ, SimplexCategory.σ]
    /-
      C : Type u_1
      inst✝¹ : CategoryTheory.Category.{?u.61848, u_1} C
      f : CategoryTheory.Arrow C
      inst✝ : ∀ (n : Nat), CategoryTheory.Limits.HasWidePullback f.right (fun x => f …
      S : CategoryTheory.SplitEpi f.hom
      n : Nat
      i : Fin (HAdd.hAdd n 1)
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Arrow.AugmentedCechNe …
    -/
    ext j
      /-
        case a
        C : Type u_1
        inst✝¹ : CategoryTheory.Category.{?u.61848, u_1} C
        f : CategoryTheory.Arrow C
        inst✝ : ∀ (n : Nat), CategoryTheory.Limits.HasWidePullback f.right (fun x => f …
        S : CategoryTheory.SplitEpi f.hom
        n : Nat
        i : Fin (HAdd.hAdd n 1)
        j : Fin (HAdd.hAdd (HAdd.hAdd (HAdd.hAdd n 1) 1) 1)
        ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.comp ( …
      -/
    · simp only [assoc, WidePullback.lift_π]
      /-
        case a
        C : Type u_1
        inst✝¹ : CategoryTheory.Category.{?u.61848, u_1} C
        f : CategoryTheory.Arrow C
        inst✝ : ∀ (n : Nat), CategoryTheory.Limits.HasWidePullback f.right (fun x => f …
        S : CategoryTheory.SplitEpi f.hom
        n : Nat
        i : Fin (HAdd.hAdd n 1)
        j : Fin (HAdd.hAdd (HAdd.hAdd (HAdd.hAdd n 1) 1) 1)
        ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Arrow.AugmentedCechNe …
      -/
      by_cases h : j = 0
        /-
          case pos
          C : Type u_1
          inst✝¹ : CategoryTheory.Category.{?u.61848, u_1} C
          f : CategoryTheory.Arrow C
          inst✝ : ∀ (n : Nat), CategoryTheory.Limits.HasWidePullback f.right (fun x => f …
          S : CategoryTheory.SplitEpi f.hom
          n : Nat
          i : Fin (HAdd.hAdd n 1)
          j : Fin (HAdd.hAdd (HAdd.hAdd (HAdd.hAdd n 1) 1) 1)
          h : Eq j 0
          ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Arrow.AugmentedCechNe …
        -/
      · subst h
        /-
          case pos
          C : Type u_1
          inst✝¹ : CategoryTheory.Category.{?u.61848, u_1} C
          f : CategoryTheory.Arrow C
          inst✝ : ∀ (n : Nat), CategoryTheory.Limits.HasWidePullback f.right (fun x => f …
          S : CategoryTheory.SplitEpi f.hom
          n : Nat
          i : Fin (HAdd.hAdd n 1)
          ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Arrow.AugmentedCechNe …
        -/
        erw [ExtraDegeneracy.s_comp_π_0, ExtraDegeneracy.s_comp_π_0]
        /-
          case pos
          C : Type u_1
          inst✝¹ : CategoryTheory.Category.{?u.61848, u_1} C
          f : CategoryTheory.Arrow C
          inst✝ : ∀ (n : Nat), CategoryTheory.Limits.HasWidePullback f.right (fun x => f …
          S : CategoryTheory.SplitEpi f.hom
          n : Nat
          i : Fin (HAdd.hAdd n 1)
          ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.WidePullback.b …
        -/
        dsimp
        /-
          case pos
          C : Type u_1
          inst✝¹ : CategoryTheory.Category.{?u.61848, u_1} C
          f : CategoryTheory.Arrow C
          inst✝ : ∀ (n : Nat), CategoryTheory.Limits.HasWidePullback f.right (fun x => f …
          S : CategoryTheory.SplitEpi f.hom
          n : Nat
          i : Fin (HAdd.hAdd n 1)
          ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.WidePullback.b …
        -/
        simp only [WidePullback.lift_base_assoc]
        /-
          🎉 no goals
        -/
        /-
          case neg
          C : Type u_1
          inst✝¹ : CategoryTheory.Category.{?u.61848, u_1} C
          f : CategoryTheory.Arrow C
          inst✝ : ∀ (n : Nat), CategoryTheory.Limits.HasWidePullback f.right (fun x => f …
          S : CategoryTheory.SplitEpi f.hom
          n : Nat
          i : Fin (HAdd.hAdd n 1)
          j : Fin (HAdd.hAdd (HAdd.hAdd (HAdd.hAdd n 1) 1) 1)
          h : Not (Eq j 0)
          ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Arrow.AugmentedCechNe …
        -/
      · cases' Fin.eq_succ_of_ne_zero h with k hk
        /-
          case neg.intro
          C : Type u_1
          inst✝¹ : CategoryTheory.Category.{?u.61848, u_1} C
          f : CategoryTheory.Arrow C
          inst✝ : ∀ (n : Nat), CategoryTheory.Limits.HasWidePullback f.right (fun x => f …
          S : CategoryTheory.SplitEpi f.hom
          n : Nat
          i : Fin (HAdd.hAdd n 1)
          j : Fin (HAdd.hAdd (HAdd.hAdd (HAdd.hAdd n 1) 1) 1)
          h : Not (Eq j 0)
          k : Fin (HAdd.hAdd n 2)
          hk : Eq j k.succ
          ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Arrow.AugmentedCechNe …
        -/
        subst hk
        erw [Fin.succ_predAbove_succ, ExtraDegeneracy.s_comp_π_succ,
          ExtraDegeneracy.s_comp_π_succ]
        /-
          case neg.intro
          C : Type u_1
          inst✝¹ : CategoryTheory.Category.{?u.61848, u_1} C
          f : CategoryTheory.Arrow C
          inst✝ : ∀ (n : Nat), CategoryTheory.Limits.HasWidePullback f.right (fun x => f …
          S : CategoryTheory.SplitEpi f.hom
          n : Nat
          i : Fin (HAdd.hAdd n 1)
          k : Fin (HAdd.hAdd n 2)
          h : Not (Eq k.succ 0)
          ⊢ Eq (CategoryTheory.Limits.WidePullback.π (fun x => f.hom) (i.predAbove k)) ( …
        -/
        simp only [WidePullback.lift_π]
        /-
          🎉 no goals
        -/
      /-
        case a
        C : Type u_1
        inst✝¹ : CategoryTheory.Category.{?u.61848, u_1} C
        f : CategoryTheory.Arrow C
        inst✝ : ∀ (n : Nat), CategoryTheory.Limits.HasWidePullback f.right (fun x => f …
        S : CategoryTheory.SplitEpi f.hom
        n : Nat
        i : Fin (HAdd.hAdd n 1)
        ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.comp ( …
      -/
    · simp only [assoc, WidePullback.lift_base]
      /-
        case a
        C : Type u_1
        inst✝¹ : CategoryTheory.Category.{?u.61848, u_1} C
        f : CategoryTheory.Arrow C
        inst✝ : ∀ (n : Nat), CategoryTheory.Limits.HasWidePullback f.right (fun x => f …
        S : CategoryTheory.SplitEpi f.hom
        n : Nat
        i : Fin (HAdd.hAdd n 1)
        ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Arrow.AugmentedCechNe …
      -/
      erw [ExtraDegeneracy.s_comp_base, ExtraDegeneracy.s_comp_base]
      /-
        case a
        C : Type u_1
        inst✝¹ : CategoryTheory.Category.{?u.61848, u_1} C
        f : CategoryTheory.Arrow C
        inst✝ : ∀ (n : Nat), CategoryTheory.Limits.HasWidePullback f.right (fun x => f …
        S : CategoryTheory.SplitEpi f.hom
        n : Nat
        i : Fin (HAdd.hAdd n 1)
        ⊢ Eq (CategoryTheory.Limits.WidePullback.base fun x => f.hom) (CategoryTheory. …
      -/
      dsimp
      /-
        case a
        C : Type u_1
        inst✝¹ : CategoryTheory.Category.{?u.61848, u_1} C
        f : CategoryTheory.Arrow C
        inst✝ : ∀ (n : Nat), CategoryTheory.Limits.HasWidePullback f.right (fun x => f …
        S : CategoryTheory.SplitEpi f.hom
        n : Nat
        i : Fin (HAdd.hAdd n 1)
        ⊢ Eq (CategoryTheory.Limits.WidePullback.base fun x => f.hom) (CategoryTheory. …
      -/
      simp only [WidePullback.lift_base]
      /-
        🎉 no goals
      -/


/-- If `C` is a preadditive category and `X` is an augmented simplicial object
in `C` that has an extra degeneracy, then the augmentation on the alternating
face map complex of `X` is a homotopy equivalence. -/
noncomputable def homotopyEquiv {C : Type*} [Category C] [Preadditive C] [HasZeroObject C]
    {X : SimplicialObject.Augmented C} (ed : ExtraDegeneracy X) :
    HomotopyEquiv (AlgebraicTopology.AlternatingFaceMapComplex.obj (drop.obj X))
      ((ChainComplex.single₀ C).obj (point.obj X)) where
  hom := AlternatingFaceMapComplex.ε.app X
                                                      /-
                                                        C : Type u_1
                                                        inst✝² : CategoryTheory.Category.{?u.94271, u_1} C
                                                        inst✝¹ : CategoryTheory.Preadditive C
                                                        inst✝ : CategoryTheory.Limits.HasZeroObject C
                                                        X : CategoryTheory.SimplicialObject.Augmented C
                                                        ed : SimplicialObject.Augmented.ExtraDegeneracy X
                                                        ⊢ Quiver.Hom (CategoryTheory.SimplicialObject.Augmented.point.obj X) ((Algebra …
                                                      -/
  inv := (ChainComplex.fromSingle₀Equiv _ _).symm (by exact ed.s')
                                                      /-
                                                        🎉 no goals
                                                      -/
  homotopyInvHomId := Homotopy.ofEq (by
    /-
      C : Type u_1
      inst✝² : CategoryTheory.Category.{?u.94271, u_1} C
      inst✝¹ : CategoryTheory.Preadditive C
      inst✝ : CategoryTheory.Limits.HasZeroObject C
      X : CategoryTheory.SimplicialObject.Augmented C
      ed : SimplicialObject.Augmented.ExtraDegeneracy X
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (((AlgebraicTopology.AlternatingFaceM …
    -/
    ext
    /-
      case hfg
      C : Type u_1
      inst✝² : CategoryTheory.Category.{?u.94271, u_1} C
      inst✝¹ : CategoryTheory.Preadditive C
      inst✝ : CategoryTheory.Limits.HasZeroObject C
      X : CategoryTheory.SimplicialObject.Augmented C
      ed : SimplicialObject.Augmented.ExtraDegeneracy X
      ⊢ Eq ((CategoryTheory.CategoryStruct.comp (((AlgebraicTopology.AlternatingFace …
    -/
    dsimp
    erw [AlternatingFaceMapComplex.ε_app_f_zero,
      ChainComplex.fromSingle₀Equiv_symm_apply_f_zero, s'_comp_ε]
    /-
      case hfg
      C : Type u_1
      inst✝² : CategoryTheory.Category.{?u.94271, u_1} C
      inst✝¹ : CategoryTheory.Preadditive C
      inst✝ : CategoryTheory.Limits.HasZeroObject C
      X : CategoryTheory.SimplicialObject.Augmented C
      ed : SimplicialObject.Augmented.ExtraDegeneracy X
      ⊢ Eq (CategoryTheory.CategoryStruct.id (CategoryTheory.SimplicialObject.Augmen …
    -/
        /-
          C : Type u_1
          inst✝² : CategoryTheory.Category.{?u.94271, u_1} C
          inst✝¹ : CategoryTheory.Preadditive C
          inst✝ : CategoryTheory.Limits.HasZeroObject C
          X : CategoryTheory.SimplicialObject.Augmented C
          ed : SimplicialObject.Augmented.ExtraDegeneracy X
          i j : Nat
          ⊢ Quiver.Hom ((AlgebraicTopology.AlternatingFaceMapComplex.obj (CategoryTheory …
        -/
    rfl)
          /-
            case pos
            C : Type u_1
            inst✝² : CategoryTheory.Category.{?u.94271, u_1} C
            inst✝¹ : CategoryTheory.Preadditive C
            inst✝ : CategoryTheory.Limits.HasZeroObject C
            X : CategoryTheory.SimplicialObject.Augmented C
            ed : SimplicialObject.Augmented.ExtraDegeneracy X
            i j : Nat
            h✝ : Eq (HAdd.hAdd i 1) j
            ⊢ Quiver.Hom ((AlgebraicTopology.AlternatingFaceMapComplex.obj (CategoryTheory …
          -/
    /-
      🎉 no goals
    -/
          /-
            🎉 no goals
          -/
          /-
            case neg
            C : Type u_1
            inst✝² : CategoryTheory.Category.{?u.94271, u_1} C
            inst✝¹ : CategoryTheory.Preadditive C
            inst✝ : CategoryTheory.Limits.HasZeroObject C
            X : CategoryTheory.SimplicialObject.Augmented C
            ed : SimplicialObject.Augmented.ExtraDegeneracy X
            i j : Nat
            h✝ : Not (Eq (HAdd.hAdd i 1) j)
            ⊢ Quiver.Hom ((AlgebraicTopology.AlternatingFaceMapComplex.obj (CategoryTheory …
          -/
  homotopyHomInvId :=
          /-
            🎉 no goals
          -/
    { hom := fun i j => by
        /-
          C : Type u_1
          inst✝² : CategoryTheory.Category.{?u.94271, u_1} C
          inst✝¹ : CategoryTheory.Preadditive C
          inst✝ : CategoryTheory.Limits.HasZeroObject C
          X : CategoryTheory.SimplicialObject.Augmented C
          ed : SimplicialObject.Augmented.ExtraDegeneracy X
          i j : Nat
          hij : Not ((ComplexShape.down Nat).Rel j i)
          ⊢ Eq ((fun i j => dite (Eq (HAdd.hAdd i 1) j) (fun h => CategoryTheory.Categor …
        -/
        by_cases i + 1 = j
        /-
          C : Type u_1
          inst✝² : CategoryTheory.Category.{?u.94271, u_1} C
          inst✝¹ : CategoryTheory.Preadditive C
          inst✝ : CategoryTheory.Limits.HasZeroObject C
          X : CategoryTheory.SimplicialObject.Augmented C
          ed : SimplicialObject.Augmented.ExtraDegeneracy X
          i j : Nat
          hij : Not ((ComplexShape.down Nat).Rel j i)
          ⊢ Eq (dite (Eq (HAdd.hAdd i 1) j) (fun h => CategoryTheory.CategoryStruct.comp …
        -/
        · exact (-ed.s i) ≫ eqToHom (by congr)
          /-
            case pos
            C : Type u_1
            inst✝² : CategoryTheory.Category.{?u.94271, u_1} C
            inst✝¹ : CategoryTheory.Preadditive C
            inst✝ : CategoryTheory.Limits.HasZeroObject C
            X : CategoryTheory.SimplicialObject.Augmented C
            ed : SimplicialObject.Augmented.ExtraDegeneracy X
            i j : Nat
            hij : Not ((ComplexShape.down Nat).Rel j i)
            h : Eq (HAdd.hAdd i 1) j
            ⊢ Eq (CategoryTheory.CategoryStruct.comp (Neg.neg (ed.s i)) (CategoryTheory.eq …
          -/
        · exact 0
          /-
            case pos
            C : Type u_1
            inst✝² : CategoryTheory.Category.{?u.94271, u_1} C
            inst✝¹ : CategoryTheory.Preadditive C
            inst✝ : CategoryTheory.Limits.HasZeroObject C
            X : CategoryTheory.SimplicialObject.Augmented C
            ed : SimplicialObject.Augmented.ExtraDegeneracy X
            i j : Nat
            hij : Not ((ComplexShape.down Nat).Rel j i)
            h : Eq (HAdd.hAdd i 1) j
            ⊢ False
          -/
      zero := fun i j hij => by
          /-
            🎉 no goals
          -/
          /-
            case neg
            C : Type u_1
            inst✝² : CategoryTheory.Category.{?u.94271, u_1} C
            inst✝¹ : CategoryTheory.Preadditive C
            inst✝ : CategoryTheory.Limits.HasZeroObject C
            X : CategoryTheory.SimplicialObject.Augmented C
            ed : SimplicialObject.Augmented.ExtraDegeneracy X
            i j : Nat
            hij : Not ((ComplexShape.down Nat).Rel j i)
            h : Not (Eq (HAdd.hAdd i 1) j)
            ⊢ Eq 0 0
          -/
        dsimp
          /-
            🎉 no goals
          -/
        split_ifs with h
        /-
          C : Type u_1
          inst✝² : CategoryTheory.Category.{?u.94271, u_1} C
          inst✝¹ : CategoryTheory.Preadditive C
          inst✝ : CategoryTheory.Limits.HasZeroObject C
          X : CategoryTheory.SimplicialObject.Augmented C
          ed : SimplicialObject.Augmented.ExtraDegeneracy X
          i : Nat
          ⊢ Eq ((CategoryTheory.CategoryStruct.comp (AlgebraicTopology.AlternatingFaceMa …
        -/
        · exfalso
          /-
            case zero
            C : Type u_1
            inst✝² : CategoryTheory.Category.{?u.94271, u_1} C
            inst✝¹ : CategoryTheory.Preadditive C
            inst✝ : CategoryTheory.Limits.HasZeroObject C
            X : CategoryTheory.SimplicialObject.Augmented C
            ed : SimplicialObject.Augmented.ExtraDegeneracy X
            ⊢ Eq ((CategoryTheory.CategoryStruct.comp (AlgebraicTopology.AlternatingFaceMa …
          -/
          exact hij h
          /-
            case zero
            C : Type u_1
            inst✝² : CategoryTheory.Category.{?u.94271, u_1} C
            inst✝¹ : CategoryTheory.Preadditive C
            inst✝ : CategoryTheory.Limits.HasZeroObject C
            X : CategoryTheory.SimplicialObject.Augmented C
            ed : SimplicialObject.Augmented.ExtraDegeneracy X
            ⊢ Eq ((CategoryTheory.CategoryStruct.comp (AlgebraicTopology.AlternatingFaceMa …
          -/
        · simp only [eq_self_iff_true]
          /-
            case zero
            C : Type u_1
            inst✝² : CategoryTheory.Category.{?u.94271, u_1} C
            inst✝¹ : CategoryTheory.Preadditive C
            inst✝ : CategoryTheory.Limits.HasZeroObject C
            X : CategoryTheory.SimplicialObject.Augmented C
            ed : SimplicialObject.Augmented.ExtraDegeneracy X
            ⊢ Eq (CategoryTheory.CategoryStruct.comp ((AlgebraicTopology.AlternatingFaceMa …
          -/
      comm := fun i => by
        rcases i with _|i
        · rw [Homotopy.prevD_chainComplex, Homotopy.dNext_zero_chainComplex, zero_add]
          /-
            case zero
            C : Type u_1
            inst✝² : CategoryTheory.Category.{?u.94271, u_1} C
            inst✝¹ : CategoryTheory.Preadditive C
            inst✝ : CategoryTheory.Limits.HasZeroObject C
            X : CategoryTheory.SimplicialObject.Augmented C
            ed : SimplicialObject.Augmented.ExtraDegeneracy X
            ⊢ Eq (CategoryTheory.CategoryStruct.comp ((AlgebraicTopology.AlternatingFaceMa …
          -/
          dsimp
          erw [ChainComplex.fromSingle₀Equiv_symm_apply_f_zero]
          simp only [comp_id, ite_true, zero_add, ComplexShape.down_Rel, not_true,
            AlternatingFaceMapComplex.obj_d_eq, Preadditive.neg_comp]
          rw [Fin.sum_univ_two]
          /-
            case succ
            C : Type u_1
            inst✝² : CategoryTheory.Category.{?u.94271, u_1} C
            inst✝¹ : CategoryTheory.Preadditive C
            inst✝ : CategoryTheory.Limits.HasZeroObject C
            X : CategoryTheory.SimplicialObject.Augmented C
            ed : SimplicialObject.Augmented.ExtraDegeneracy X
            i : Nat
            ⊢ Eq ((CategoryTheory.CategoryStruct.comp (AlgebraicTopology.AlternatingFaceMa …
          -/
          simp only [Fin.val_zero, pow_zero, one_smul, Fin.val_one, pow_one, neg_smul,
          /-
            case succ
            C : Type u_1
            inst✝² : CategoryTheory.Category.{?u.94271, u_1} C
            inst✝¹ : CategoryTheory.Preadditive C
            inst✝ : CategoryTheory.Limits.HasZeroObject C
            X : CategoryTheory.SimplicialObject.Augmented C
            ed : SimplicialObject.Augmented.ExtraDegeneracy X
            i : Nat
            ⊢ Eq ((CategoryTheory.CategoryStruct.comp (AlgebraicTopology.AlternatingFaceMa …
          -/
            Preadditive.comp_add, s_comp_δ₀, drop_obj, Preadditive.comp_neg, neg_add_rev,
            neg_neg, neg_add_cancel_right, s₀_comp_δ₁,
            AlternatingFaceMapComplex.ε_app_f_zero]
        · rw [Homotopy.prevD_chainComplex, Homotopy.dNext_succ_chainComplex]
          dsimp
          simp only [Preadditive.neg_comp,
            AlternatingFaceMapComplex.obj_d_eq, comp_id, ite_true, Preadditive.comp_neg,
            @Fin.sum_univ_succ _ _ (i + 2), Fin.val_zero, pow_zero, one_smul, Fin.val_succ,
            Preadditive.comp_add, drop_obj, s_comp_δ₀, Preadditive.sum_comp,
            Preadditive.zsmul_comp, Preadditive.comp_sum, Preadditive.comp_zsmul,
            zsmul_neg, s_comp_δ, pow_add, pow_one, mul_neg, mul_one, neg_zsmul, neg_neg,
            neg_add_cancel_comm_assoc, neg_add_cancel, zero_comp] }


