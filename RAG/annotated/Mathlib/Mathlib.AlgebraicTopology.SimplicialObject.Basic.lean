/-- The category of simplicial objects valued in a category `C`.
This is the category of contravariant functors from `SimplexCategory` to `C`. -/
def SimplicialObject :=
  SimplexCategoryᵒᵖ ⥤ C


@[simps!]
instance : Category (SimplicialObject C) := by
  /-
    C : Type u
    inst✝ : CategoryTheory.Category.{v, u} C
    ⊢ CategoryTheory.Category.{?u.63, max u v} (CategoryTheory.SimplicialObject C)
  -/
  dsimp only [SimplicialObject]
  /-
    C : Type u
    inst✝ : CategoryTheory.Category.{v, u} C
    ⊢ CategoryTheory.Category.{?u.63, max u v} (CategoryTheory.Functor (Opposite S …
  -/
  infer_instance
  /-
    🎉 no goals
  -/


set_option quotPrecheck false in
/-- `X _[n]` denotes the `n`th-term of the simplicial object X -/
scoped[Simplicial]
  notation3:1000 X " _[" n "]" =>
      (X : CategoryTheory.SimplicialObject _).obj (Opposite.op (SimplexCategory.mk n))


instance {J : Type v} [SmallCategory J] [HasLimitsOfShape J C] :
    HasLimitsOfShape J (SimplicialObject C) := by
  /-
    C : Type u
    inst✝² : CategoryTheory.Category.{v, u} C
    J : Type v
    inst✝¹ : CategoryTheory.SmallCategory J
    inst✝ : CategoryTheory.Limits.HasLimitsOfShape J C
    ⊢ CategoryTheory.Limits.HasLimitsOfShape J (CategoryTheory.SimplicialObject C)
  -/
  dsimp [SimplicialObject]
  /-
    C : Type u
    inst✝² : CategoryTheory.Category.{v, u} C
    J : Type v
    inst✝¹ : CategoryTheory.SmallCategory J
    inst✝ : CategoryTheory.Limits.HasLimitsOfShape J C
    ⊢ CategoryTheory.Limits.HasLimitsOfShape J (CategoryTheory.Functor (Opposite S …
  -/
  infer_instance
  /-
    🎉 no goals
  -/


instance [HasLimits C] : HasLimits (SimplicialObject C) :=
  ⟨inferInstance⟩


instance {J : Type v} [SmallCategory J] [HasColimitsOfShape J C] :
    HasColimitsOfShape J (SimplicialObject C) := by
  /-
    C : Type u
    inst✝² : CategoryTheory.Category.{v, u} C
    J : Type v
    inst✝¹ : CategoryTheory.SmallCategory J
    inst✝ : CategoryTheory.Limits.HasColimitsOfShape J C
    ⊢ CategoryTheory.Limits.HasColimitsOfShape J (CategoryTheory.SimplicialObject C)
  -/
  dsimp [SimplicialObject]
  /-
    C : Type u
    inst✝² : CategoryTheory.Category.{v, u} C
    J : Type v
    inst✝¹ : CategoryTheory.SmallCategory J
    inst✝ : CategoryTheory.Limits.HasColimitsOfShape J C
    ⊢ CategoryTheory.Limits.HasColimitsOfShape J (CategoryTheory.Functor (Opposite …
  -/
  infer_instance
  /-
    🎉 no goals
  -/


instance [HasColimits C] : HasColimits (SimplicialObject C) :=
  ⟨inferInstance⟩


@[ext]
lemma hom_ext {X Y : SimplicialObject C} (f g : X ⟶ Y)
    (h : ∀ (n : SimplexCategoryᵒᵖ), f.app n = g.app n) : f = g :=
                   /-
                     C : Type u
                     inst✝ : CategoryTheory.Category.{v, u} C
                     X Y : CategoryTheory.SimplicialObject C
                     f g : Quiver.Hom X Y
                     h : ∀ (n : Opposite SimplexCategory), Eq (f.app n) (g.app n)
                     ⊢ Eq f.app g.app
                   -/
  NatTrans.ext (by ext; apply h)
                        /-
                          🎉 no goals
                        -/


/-- Face maps for a simplicial object. -/
def δ {n} (i : Fin (n + 2)) : X _[n + 1] ⟶ X _[n] :=
  X.map (SimplexCategory.δ i).op


/-- Degeneracy maps for a simplicial object. -/
def σ {n} (i : Fin (n + 1)) : X _[n] ⟶ X _[n + 1] :=
  X.map (SimplexCategory.σ i).op


/-- The diagonal of a simplex is the long edge of the simplex.-/
def diagonal {n : ℕ} : X _[n] ⟶ X _[1] := X.map ((SimplexCategory.diag n).op)


/-- Isomorphisms from identities in ℕ. -/
def eqToIso {n m : ℕ} (h : n = m) : X _[n] ≅ X _[m] :=
                                       /-
                                         C : Type u
                                         inst✝ : CategoryTheory.Category.{v, u} C
                                         X : CategoryTheory.SimplicialObject C
                                         n m : Nat
                                         h : Eq n m
                                         ⊢ Eq { unop := SimplexCategory.mk n } { unop := SimplexCategory.mk m }
                                       -/
  X.mapIso (CategoryTheory.eqToIso (by congr))
                                       /-
                                         🎉 no goals
                                       -/


@[simp]
theorem eqToIso_refl {n : ℕ} (h : n = n) : X.eqToIso h = Iso.refl _ := by
  /-
    C : Type u
    inst✝ : CategoryTheory.Category.{v, u} C
    X : CategoryTheory.SimplicialObject C
    n : Nat
    h : Eq n n
    ⊢ Eq (X.eqToIso h) (CategoryTheory.Iso.refl (X.obj { unop := SimplexCategory.m …
  -/
  ext
  /-
    case w
    C : Type u
    inst✝ : CategoryTheory.Category.{v, u} C
    X : CategoryTheory.SimplicialObject C
    n : Nat
    h : Eq n n
    ⊢ Eq (X.eqToIso h).hom (CategoryTheory.Iso.refl (X.obj { unop := SimplexCatego …
  -/
  simp [eqToIso]
  /-
    🎉 no goals
  -/


/-- The generic case of the first simplicial identity -/
@[reassoc]
theorem δ_comp_δ {n} {i j : Fin (n + 2)} (H : i ≤ j) :
    X.δ j.succ ≫ X.δ i = X.δ (Fin.castSucc i) ≫ X.δ j := by
  /-
    C : Type u
    inst✝ : CategoryTheory.Category.{v, u} C
    X : CategoryTheory.SimplicialObject C
    n : Nat
    i j : Fin (HAdd.hAdd n 2)
    H : LE.le i j
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (X.δ j.succ) (X.δ i)) (CategoryTheory …
  -/
  dsimp [δ]
  /-
    C : Type u
    inst✝ : CategoryTheory.Category.{v, u} C
    X : CategoryTheory.SimplicialObject C
    n : Nat
    i j : Fin (HAdd.hAdd n 2)
    H : LE.le i j
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (X.map (SimplexCategory.δ j.succ).op) …
  -/
  simp only [← X.map_comp, ← op_comp, SimplexCategory.δ_comp_δ H]
  /-
    🎉 no goals
  -/


@[reassoc]
theorem δ_comp_δ' {n} {i : Fin (n + 2)} {j : Fin (n + 3)} (H : Fin.castSucc i < j) :
    X.δ j ≫ X.δ i =
      X.δ (Fin.castSucc i) ≫
                                           /-
                                             C : Type u
                                             inst✝ : CategoryTheory.Category.{v, u} C
                                             X : CategoryTheory.SimplicialObject C
                                             n : Nat
                                             i : Fin (HAdd.hAdd n 2)
                                             j : Fin (HAdd.hAdd n 3)
                                             H : LT.lt i.castSucc j
                                             hj : Eq j 0
                                             ⊢ False
                                           -/
        X.δ (j.pred fun (hj : j = 0) => by simp [hj, Fin.not_lt_zero] at H) := by
                                           /-
                                             🎉 no goals
                                           -/
  /-
    C : Type u
    inst✝ : CategoryTheory.Category.{v, u} C
    X : CategoryTheory.SimplicialObject C
    n : Nat
    i : Fin (HAdd.hAdd n 2)
    j : Fin (HAdd.hAdd n 3)
    H : LT.lt i.castSucc j
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (X.δ j) (X.δ i)) (CategoryTheory.Cate …
  -/
  dsimp [δ]
  /-
    C : Type u
    inst✝ : CategoryTheory.Category.{v, u} C
    X : CategoryTheory.SimplicialObject C
    n : Nat
    i : Fin (HAdd.hAdd n 2)
    j : Fin (HAdd.hAdd n 3)
    H : LT.lt i.castSucc j
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (X.map (SimplexCategory.δ j).op) (X.m …
  -/
  simp only [← X.map_comp, ← op_comp, SimplexCategory.δ_comp_δ' H]
  /-
    🎉 no goals
  -/

@[reassoc]
theorem δ_comp_δ'' {n} {i : Fin (n + 3)} {j : Fin (n + 2)} (H : i ≤ Fin.castSucc j) :
    X.δ j.succ ≫ X.δ (i.castLT (Nat.lt_of_le_of_lt (Fin.le_iff_val_le_val.mp H) j.is_lt)) =
      X.δ i ≫ X.δ j := by
  /-
    C : Type u
    inst✝ : CategoryTheory.Category.{v, u} C
    X : CategoryTheory.SimplicialObject C
    n : Nat
    i : Fin (HAdd.hAdd n 3)
    j : Fin (HAdd.hAdd n 2)
    H : LE.le i j.castSucc
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (X.δ j.succ) (X.δ (i.castLT ⋯))) (Cat …
  -/
  dsimp [δ]
  /-
    C : Type u
    inst✝ : CategoryTheory.Category.{v, u} C
    X : CategoryTheory.SimplicialObject C
    n : Nat
    i : Fin (HAdd.hAdd n 3)
    j : Fin (HAdd.hAdd n 2)
    H : LE.le i j.castSucc
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (X.map (SimplexCategory.δ j.succ).op) …
  -/
  simp only [← X.map_comp, ← op_comp, SimplexCategory.δ_comp_δ'' H]
  /-
    🎉 no goals
  -/


/-- The special case of the first simplicial identity -/
@[reassoc]
theorem δ_comp_δ_self {n} {i : Fin (n + 2)} :
    X.δ (Fin.castSucc i) ≫ X.δ i = X.δ i.succ ≫ X.δ i := by
  /-
    C : Type u
    inst✝ : CategoryTheory.Category.{v, u} C
    X : CategoryTheory.SimplicialObject C
    n : Nat
    i : Fin (HAdd.hAdd n 2)
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (X.δ i.castSucc) (X.δ i)) (CategoryTh …
  -/
  dsimp [δ]
  /-
    C : Type u
    inst✝ : CategoryTheory.Category.{v, u} C
    X : CategoryTheory.SimplicialObject C
    n : Nat
    i : Fin (HAdd.hAdd n 2)
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (X.map (SimplexCategory.δ i.castSucc) …
  -/
  simp only [← X.map_comp, ← op_comp, SimplexCategory.δ_comp_δ_self]
  /-
    🎉 no goals
  -/


@[reassoc]
theorem δ_comp_δ_self' {n} {j : Fin (n + 3)} {i : Fin (n + 2)} (H : j = Fin.castSucc i) :
    X.δ j ≫ X.δ i = X.δ i.succ ≫ X.δ i := by
  /-
    C : Type u
    inst✝ : CategoryTheory.Category.{v, u} C
    X : CategoryTheory.SimplicialObject C
    n : Nat
    j : Fin (HAdd.hAdd n 3)
    i : Fin (HAdd.hAdd n 2)
    H : Eq j i.castSucc
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (X.δ j) (X.δ i)) (CategoryTheory.Cate …
  -/
  subst H
  /-
    C : Type u
    inst✝ : CategoryTheory.Category.{v, u} C
    X : CategoryTheory.SimplicialObject C
    n : Nat
    i : Fin (HAdd.hAdd n 2)
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (X.δ i.castSucc) (X.δ i)) (CategoryTh …
  -/
  rw [δ_comp_δ_self]
  /-
    🎉 no goals
  -/


/-- The second simplicial identity -/
@[reassoc]
theorem δ_comp_σ_of_le {n} {i : Fin (n + 2)} {j : Fin (n + 1)} (H : i ≤ Fin.castSucc j) :
    X.σ j.succ ≫ X.δ (Fin.castSucc i) = X.δ i ≫ X.σ j := by
  /-
    C : Type u
    inst✝ : CategoryTheory.Category.{v, u} C
    X : CategoryTheory.SimplicialObject C
    n : Nat
    i : Fin (HAdd.hAdd n 2)
    j : Fin (HAdd.hAdd n 1)
    H : LE.le i j.castSucc
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (X.σ j.succ) (X.δ i.castSucc)) (Categ …
  -/
  dsimp [δ, σ]
  /-
    C : Type u
    inst✝ : CategoryTheory.Category.{v, u} C
    X : CategoryTheory.SimplicialObject C
    n : Nat
    i : Fin (HAdd.hAdd n 2)
    j : Fin (HAdd.hAdd n 1)
    H : LE.le i j.castSucc
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (X.map (SimplexCategory.σ j.succ).op) …
  -/
  simp only [← X.map_comp, ← op_comp, SimplexCategory.δ_comp_σ_of_le H]
  /-
    🎉 no goals
  -/


/-- The first part of the third simplicial identity -/
@[reassoc]
theorem δ_comp_σ_self {n} {i : Fin (n + 1)} : X.σ i ≫ X.δ (Fin.castSucc i) = 𝟙 _ := by
  /-
    C : Type u
    inst✝ : CategoryTheory.Category.{v, u} C
    X : CategoryTheory.SimplicialObject C
    n : Nat
    i : Fin (HAdd.hAdd n 1)
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (X.σ i) (X.δ i.castSucc)) (CategoryTh …
  -/
  dsimp [δ, σ]
  /-
    C : Type u
    inst✝ : CategoryTheory.Category.{v, u} C
    X : CategoryTheory.SimplicialObject C
    n : Nat
    i : Fin (HAdd.hAdd n 1)
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (X.map (SimplexCategory.σ i).op) (X.m …
  -/
  simp only [← X.map_comp, ← op_comp, SimplexCategory.δ_comp_σ_self, op_id, X.map_id]
  /-
    🎉 no goals
  -/


@[reassoc]
theorem δ_comp_σ_self' {n} {j : Fin (n + 2)} {i : Fin (n + 1)} (H : j = Fin.castSucc i) :
    X.σ i ≫ X.δ j = 𝟙 _ := by
  /-
    C : Type u
    inst✝ : CategoryTheory.Category.{v, u} C
    X : CategoryTheory.SimplicialObject C
    n : Nat
    j : Fin (HAdd.hAdd n 2)
    i : Fin (HAdd.hAdd n 1)
    H : Eq j i.castSucc
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (X.σ i) (X.δ j)) (CategoryTheory.Cate …
  -/
  subst H
  /-
    C : Type u
    inst✝ : CategoryTheory.Category.{v, u} C
    X : CategoryTheory.SimplicialObject C
    n : Nat
    i : Fin (HAdd.hAdd n 1)
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (X.σ i) (X.δ i.castSucc)) (CategoryTh …
  -/
  rw [δ_comp_σ_self]
  /-
    🎉 no goals
  -/


/-- The second part of the third simplicial identity -/
@[reassoc]
theorem δ_comp_σ_succ {n} {i : Fin (n + 1)} : X.σ i ≫ X.δ i.succ = 𝟙 _ := by
  /-
    C : Type u
    inst✝ : CategoryTheory.Category.{v, u} C
    X : CategoryTheory.SimplicialObject C
    n : Nat
    i : Fin (HAdd.hAdd n 1)
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (X.σ i) (X.δ i.succ)) (CategoryTheory …
  -/
  dsimp [δ, σ]
  /-
    C : Type u
    inst✝ : CategoryTheory.Category.{v, u} C
    X : CategoryTheory.SimplicialObject C
    n : Nat
    i : Fin (HAdd.hAdd n 1)
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (X.map (SimplexCategory.σ i).op) (X.m …
  -/
  simp only [← X.map_comp, ← op_comp, SimplexCategory.δ_comp_σ_succ, op_id, X.map_id]
  /-
    🎉 no goals
  -/


@[reassoc]
theorem δ_comp_σ_succ' {n} {j : Fin (n + 2)} {i : Fin (n + 1)} (H : j = i.succ) :
    X.σ i ≫ X.δ j = 𝟙 _ := by
  /-
    C : Type u
    inst✝ : CategoryTheory.Category.{v, u} C
    X : CategoryTheory.SimplicialObject C
    n : Nat
    j : Fin (HAdd.hAdd n 2)
    i : Fin (HAdd.hAdd n 1)
    H : Eq j i.succ
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (X.σ i) (X.δ j)) (CategoryTheory.Cate …
  -/
  subst H
  /-
    C : Type u
    inst✝ : CategoryTheory.Category.{v, u} C
    X : CategoryTheory.SimplicialObject C
    n : Nat
    i : Fin (HAdd.hAdd n 1)
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (X.σ i) (X.δ i.succ)) (CategoryTheory …
  -/
  rw [δ_comp_σ_succ]
  /-
    🎉 no goals
  -/


/-- The fourth simplicial identity -/
@[reassoc]
theorem δ_comp_σ_of_gt {n} {i : Fin (n + 2)} {j : Fin (n + 1)} (H : Fin.castSucc j < i) :
    X.σ (Fin.castSucc j) ≫ X.δ i.succ = X.δ i ≫ X.σ j := by
  /-
    C : Type u
    inst✝ : CategoryTheory.Category.{v, u} C
    X : CategoryTheory.SimplicialObject C
    n : Nat
    i : Fin (HAdd.hAdd n 2)
    j : Fin (HAdd.hAdd n 1)
    H : LT.lt j.castSucc i
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (X.σ j.castSucc) (X.δ i.succ)) (Categ …
  -/
  dsimp [δ, σ]
  /-
    C : Type u
    inst✝ : CategoryTheory.Category.{v, u} C
    X : CategoryTheory.SimplicialObject C
    n : Nat
    i : Fin (HAdd.hAdd n 2)
    j : Fin (HAdd.hAdd n 1)
    H : LT.lt j.castSucc i
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (X.map (SimplexCategory.σ j.castSucc) …
  -/
  simp only [← X.map_comp, ← op_comp, SimplexCategory.δ_comp_σ_of_gt H]
  /-
    🎉 no goals
  -/


@[reassoc]
theorem δ_comp_σ_of_gt' {n} {i : Fin (n + 3)} {j : Fin (n + 2)} (H : j.succ < i) :
    X.σ j ≫ X.δ i =
                                         /-
                                           C : Type u
                                           inst✝ : CategoryTheory.Category.{v, u} C
                                           X : CategoryTheory.SimplicialObject C
                                           n : Nat
                                           i : Fin (HAdd.hAdd n 3)
                                           j : Fin (HAdd.hAdd n 2)
                                           H : LT.lt j.succ i
                                           hi : Eq i 0
                                           ⊢ False
                                         -/
      X.δ (i.pred fun (hi : i = 0) => by simp only [Fin.not_lt_zero, hi] at H) ≫
                                         /-
                                           🎉 no goals
                                         -/
        X.σ (j.castLT ((add_lt_add_iff_right 1).mp (lt_of_lt_of_le H i.is_le))) := by
  /-
    C : Type u
    inst✝ : CategoryTheory.Category.{v, u} C
    X : CategoryTheory.SimplicialObject C
    n : Nat
    i : Fin (HAdd.hAdd n 3)
    j : Fin (HAdd.hAdd n 2)
    H : LT.lt j.succ i
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (X.σ j) (X.δ i)) (CategoryTheory.Cate …
  -/
  dsimp [δ, σ]
  /-
    C : Type u
    inst✝ : CategoryTheory.Category.{v, u} C
    X : CategoryTheory.SimplicialObject C
    n : Nat
    i : Fin (HAdd.hAdd n 3)
    j : Fin (HAdd.hAdd n 2)
    H : LT.lt j.succ i
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (X.map (SimplexCategory.σ j).op) (X.m …
  -/
  simp only [← X.map_comp, ← op_comp, SimplexCategory.δ_comp_σ_of_gt' H]
  /-
    🎉 no goals
  -/


/-- The fifth simplicial identity -/
@[reassoc]
theorem σ_comp_σ {n} {i j : Fin (n + 1)} (H : i ≤ j) :
    X.σ j ≫ X.σ (Fin.castSucc i) = X.σ i ≫ X.σ j.succ := by
  /-
    C : Type u
    inst✝ : CategoryTheory.Category.{v, u} C
    X : CategoryTheory.SimplicialObject C
    n : Nat
    i j : Fin (HAdd.hAdd n 1)
    H : LE.le i j
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (X.σ j) (X.σ i.castSucc)) (CategoryTh …
  -/
  dsimp [δ, σ]
  /-
    C : Type u
    inst✝ : CategoryTheory.Category.{v, u} C
    X : CategoryTheory.SimplicialObject C
    n : Nat
    i j : Fin (HAdd.hAdd n 1)
    H : LE.le i j
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (X.map (SimplexCategory.σ j).op) (X.m …
  -/
  simp only [← X.map_comp, ← op_comp, SimplexCategory.σ_comp_σ H]
  /-
    🎉 no goals
  -/


@[reassoc (attr := simp)]
theorem δ_naturality {X' X : SimplicialObject C} (f : X ⟶ X') {n : ℕ} (i : Fin (n + 2)) :
    X.δ i ≫ f.app (op [n]) = f.app (op [n + 1]) ≫ X'.δ i :=
  f.naturality _


@[reassoc (attr := simp)]
theorem σ_naturality {X' X : SimplicialObject C} (f : X ⟶ X') {n : ℕ} (i : Fin (n + 1)) :
    X.σ i ≫ f.app (op [n + 1]) = f.app (op [n]) ≫ X'.σ i :=
  f.naturality _


/-- Functor composition induces a functor on simplicial objects. -/
@[simps!]
def whiskering (D : Type*) [Category D] : (C ⥤ D) ⥤ SimplicialObject C ⥤ SimplicialObject D :=
  whiskeringRight _ _ _

-- Porting note (https://github.com/leanprover-community/mathlib4/issues/5171): removed @[nolint has_nonempty_instance]

/-- Truncated simplicial objects. -/
def Truncated (n : ℕ) :=
  (SimplexCategory.Truncated n)ᵒᵖ ⥤ C


instance {n : ℕ} : Category (Truncated C n) := by
  /-
    C : Type u
    inst✝ : CategoryTheory.Category.{v, u} C
    X : CategoryTheory.SimplicialObject C
    n : Nat
    ⊢ CategoryTheory.Category.{?u.78925, max u v} (CategoryTheory.SimplicialObject …
  -/
  dsimp [Truncated]
  /-
    C : Type u
    inst✝ : CategoryTheory.Category.{v, u} C
    X : CategoryTheory.SimplicialObject C
    n : Nat
    ⊢ CategoryTheory.Category.{?u.78925, max u v} (CategoryTheory.Functor (Opposit …
  -/
  infer_instance
  /-
    🎉 no goals
  -/


instance {n} {J : Type v} [SmallCategory J] [HasLimitsOfShape J C] :
    HasLimitsOfShape J (SimplicialObject.Truncated C n) := by
  /-
    C : Type u
    inst✝² : CategoryTheory.Category.{v, u} C
    X : CategoryTheory.SimplicialObject C
    n : Nat
    J : Type v
    inst✝¹ : CategoryTheory.SmallCategory J
    inst✝ : CategoryTheory.Limits.HasLimitsOfShape J C
    ⊢ CategoryTheory.Limits.HasLimitsOfShape J (CategoryTheory.SimplicialObject.Tr …
  -/
  dsimp [Truncated]
  /-
    C : Type u
    inst✝² : CategoryTheory.Category.{v, u} C
    X : CategoryTheory.SimplicialObject C
    n : Nat
    J : Type v
    inst✝¹ : CategoryTheory.SmallCategory J
    inst✝ : CategoryTheory.Limits.HasLimitsOfShape J C
    ⊢ CategoryTheory.Limits.HasLimitsOfShape J (CategoryTheory.Functor (Opposite ( …
  -/
  infer_instance
  /-
    🎉 no goals
  -/


instance {n} [HasLimits C] : HasLimits (SimplicialObject.Truncated C n) :=
  ⟨inferInstance⟩


instance {n} {J : Type v} [SmallCategory J] [HasColimitsOfShape J C] :
    HasColimitsOfShape J (SimplicialObject.Truncated C n) := by
  /-
    C : Type u
    inst✝² : CategoryTheory.Category.{v, u} C
    X : CategoryTheory.SimplicialObject C
    n : Nat
    J : Type v
    inst✝¹ : CategoryTheory.SmallCategory J
    inst✝ : CategoryTheory.Limits.HasColimitsOfShape J C
    ⊢ CategoryTheory.Limits.HasColimitsOfShape J (CategoryTheory.SimplicialObject. …
  -/
  dsimp [Truncated]
  /-
    C : Type u
    inst✝² : CategoryTheory.Category.{v, u} C
    X : CategoryTheory.SimplicialObject C
    n : Nat
    J : Type v
    inst✝¹ : CategoryTheory.SmallCategory J
    inst✝ : CategoryTheory.Limits.HasColimitsOfShape J C
    ⊢ CategoryTheory.Limits.HasColimitsOfShape J (CategoryTheory.Functor (Opposite …
  -/
  infer_instance
  /-
    🎉 no goals
  -/


instance {n} [HasColimits C] : HasColimits (SimplicialObject.Truncated C n) :=
  ⟨inferInstance⟩


/-- Functor composition induces a functor on truncated simplicial objects. -/
@[simps!]
def whiskering {n} (D : Type*) [Category D] : (C ⥤ D) ⥤ Truncated C n ⥤ Truncated D n :=
  whiskeringRight _ _ _


/-- The truncation functor from simplicial objects to truncated simplicial objects. -/
def truncation (n : ℕ) : SimplicialObject C ⥤ SimplicialObject.Truncated C n :=
  (whiskeringLeft _ _ _).obj (SimplexCategory.Truncated.inclusion n).op


/-- The n-skeleton as a functor `SimplicialObject.Truncated C n ⥤ SimplicialObject C`. -/
protected abbrev Truncated.sk (n : ℕ) [∀ (F : (SimplexCategory.Truncated n)ᵒᵖ ⥤ C),
    (SimplexCategory.Truncated.inclusion n).op.HasLeftKanExtension F] :
    SimplicialObject.Truncated C n ⥤ SimplicialObject C :=
  lan (SimplexCategory.Truncated.inclusion n).op


/-- The n-coskeleton as a functor `SimplicialObject.Truncated C n ⥤ SimplicialObject C`. -/
protected abbrev Truncated.cosk (n : ℕ) [∀ (F : (SimplexCategory.Truncated n)ᵒᵖ ⥤ C),
    (SimplexCategory.Truncated.inclusion n).op.HasRightKanExtension F] :
    SimplicialObject.Truncated C n ⥤ SimplicialObject C :=
  ran (SimplexCategory.Truncated.inclusion n).op


/-- The n-skeleton as an endofunctor on `SimplicialObject C`. -/
abbrev sk (n : ℕ) [∀ (F : (SimplexCategory.Truncated n)ᵒᵖ ⥤ C),
    (SimplexCategory.Truncated.inclusion n).op.HasLeftKanExtension F] :
    SimplicialObject C ⥤ SimplicialObject C := truncation n ⋙ Truncated.sk n


/-- The n-coskeleton as an endofunctor on `SimplicialObject C`. -/
abbrev cosk (n : ℕ) [∀ (F : (SimplexCategory.Truncated n)ᵒᵖ ⥤ C),
    (SimplexCategory.Truncated.inclusion n).op.HasRightKanExtension F] :
    SimplicialObject C ⥤ SimplicialObject C := truncation n ⋙ Truncated.cosk n


/-- The adjunction between the n-skeleton and n-truncation.-/
noncomputable def skAdj : Truncated.sk (C := C) n ⊣ truncation n :=
  lanAdjunction _ _


/-- The adjunction between n-truncation and the n-coskeleton.-/
noncomputable def coskAdj : truncation (C := C) n ⊣ Truncated.cosk n :=
  ranAdjunction _ _


instance : ((sk n).obj X).IsLeftKanExtension ((skAdj n).unit.app _) := by
  /-
    C : Type u
    inst✝² : CategoryTheory.Category.{v, u} C
    X : CategoryTheory.SimplicialObject C
    n : Nat
    inst✝¹ : ∀ (F : CategoryTheory.Functor (Opposite (SimplexCategory.Truncated n) …
    inst✝ : ∀ (F : CategoryTheory.Functor (Opposite (SimplexCategory.Truncated n)) …
    ⊢ CategoryTheory.Functor.IsLeftKanExtension ((CategoryTheory.SimplicialObject. …
  -/
  dsimp [sk, skAdj]
  /-
    C : Type u
    inst✝² : CategoryTheory.Category.{v, u} C
    X : CategoryTheory.SimplicialObject C
    n : Nat
    inst✝¹ : ∀ (F : CategoryTheory.Functor (Opposite (SimplexCategory.Truncated n) …
    inst✝ : ∀ (F : CategoryTheory.Functor (Opposite (SimplexCategory.Truncated n)) …
    ⊢ CategoryTheory.Functor.IsLeftKanExtension ((CategoryTheory.SimplicialObject. …
  -/
  rw [lanAdjunction_unit]
  /-
    C : Type u
    inst✝² : CategoryTheory.Category.{v, u} C
    X : CategoryTheory.SimplicialObject C
    n : Nat
    inst✝¹ : ∀ (F : CategoryTheory.Functor (Opposite (SimplexCategory.Truncated n) …
    inst✝ : ∀ (F : CategoryTheory.Functor (Opposite (SimplexCategory.Truncated n)) …
    ⊢ CategoryTheory.Functor.IsLeftKanExtension ((CategoryTheory.SimplicialObject. …
  -/
  infer_instance
  /-
    🎉 no goals
  -/


instance : ((cosk n).obj X).IsRightKanExtension ((coskAdj n).counit.app _) := by
  /-
    C : Type u
    inst✝² : CategoryTheory.Category.{v, u} C
    X : CategoryTheory.SimplicialObject C
    n : Nat
    inst✝¹ : ∀ (F : CategoryTheory.Functor (Opposite (SimplexCategory.Truncated n) …
    inst✝ : ∀ (F : CategoryTheory.Functor (Opposite (SimplexCategory.Truncated n)) …
    ⊢ CategoryTheory.Functor.IsRightKanExtension ((CategoryTheory.SimplicialObject …
  -/
  dsimp [cosk, coskAdj]
  /-
    C : Type u
    inst✝² : CategoryTheory.Category.{v, u} C
    X : CategoryTheory.SimplicialObject C
    n : Nat
    inst✝¹ : ∀ (F : CategoryTheory.Functor (Opposite (SimplexCategory.Truncated n) …
    inst✝ : ∀ (F : CategoryTheory.Functor (Opposite (SimplexCategory.Truncated n)) …
    ⊢ CategoryTheory.Functor.IsRightKanExtension ((CategoryTheory.SimplicialObject …
  -/
  rw [ranAdjunction_counit]
  /-
    C : Type u
    inst✝² : CategoryTheory.Category.{v, u} C
    X : CategoryTheory.SimplicialObject C
    n : Nat
    inst✝¹ : ∀ (F : CategoryTheory.Functor (Opposite (SimplexCategory.Truncated n) …
    inst✝ : ∀ (F : CategoryTheory.Functor (Opposite (SimplexCategory.Truncated n)) …
    ⊢ CategoryTheory.Functor.IsRightKanExtension ((CategoryTheory.SimplicialObject …
  -/
  infer_instance
  /-
    🎉 no goals
  -/


instance cosk_reflective : IsIso (coskAdj (C := C) n).counit :=
  reflective' (SimplexCategory.Truncated.inclusion n).op


instance sk_coreflective : IsIso (skAdj (C := C) n).unit :=
  coreflective' (SimplexCategory.Truncated.inclusion n).op


/-- Since `Truncated.inclusion` is fully faithful, so is right Kan extension along it.-/
noncomputable def cosk.fullyFaithful :
    (Truncated.cosk (C := C) n).FullyFaithful := by
  /-
    C : Type u
    inst✝⁴ : CategoryTheory.Category.{v, u} C
    X : CategoryTheory.SimplicialObject C
    n : Nat
    inst✝³ : ∀ (F : CategoryTheory.Functor (Opposite (SimplexCategory.Truncated n) …
    inst✝² : ∀ (F : CategoryTheory.Functor (Opposite (SimplexCategory.Truncated n) …
    inst✝¹ : ∀ (F : CategoryTheory.Functor (Opposite (SimplexCategory.Truncated n) …
    inst✝ : ∀ (F : CategoryTheory.Functor (Opposite (SimplexCategory.Truncated n)) …
    ⊢ (CategoryTheory.SimplicialObject.Truncated.cosk n).FullyFaithful
  -/
  apply Adjunction.fullyFaithfulROfIsIsoCounit (coskAdj n)
  /-
    🎉 no goals
  -/


instance cosk.full : (Truncated.cosk (C := C) n).Full := FullyFaithful.full (cosk.fullyFaithful _)


instance cosk.faithful : (Truncated.cosk (C := C) n).Faithful :=
  FullyFaithful.faithful (cosk.fullyFaithful _)


noncomputable instance coskAdj.reflective : Reflective (Truncated.cosk (C := C) n) :=
  Reflective.mk (truncation _) (coskAdj _)


/-- Since `Truncated.inclusion` is fully faithful, so is left Kan extension along it.-/
noncomputable def sk.fullyFaithful : (Truncated.sk (C := C) n).FullyFaithful :=
  Adjunction.fullyFaithfulLOfIsIsoUnit (skAdj n)


instance sk.full : (Truncated.sk (C := C) n).Full := FullyFaithful.full (sk.fullyFaithful _)


instance sk.faithful : (Truncated.sk (C := C) n).Faithful :=
  FullyFaithful.faithful (sk.fullyFaithful _)


noncomputable instance skAdj.coreflective : Coreflective (Truncated.sk (C := C) n) :=
  Coreflective.mk (truncation _) (skAdj _)


/-- The constant simplicial object is the constant functor. -/
abbrev const : C ⥤ SimplicialObject C :=
  CategoryTheory.Functor.const _

-- Porting note (https://github.com/leanprover-community/mathlib4/issues/5171): removed @[nolint has_nonempty_instance]

/-- The category of augmented simplicial objects, defined as a comma category. -/
def Augmented :=
  Comma (𝟭 (SimplicialObject C)) (const C)


@[simps!]
instance : Category (Augmented C) := by
  /-
    C : Type u
    inst✝ : CategoryTheory.Category.{v, u} C
    X : CategoryTheory.SimplicialObject C
    ⊢ CategoryTheory.Category.{?u.170654, max u v} (CategoryTheory.SimplicialObjec …
  -/
  dsimp only [Augmented]
  /-
    C : Type u
    inst✝ : CategoryTheory.Category.{v, u} C
    X : CategoryTheory.SimplicialObject C
    ⊢ CategoryTheory.Category.{?u.170654, max u v} (CategoryTheory.Comma (Category …
  -/
  infer_instance
  /-
    🎉 no goals
  -/


@[ext]
lemma hom_ext {X Y : Augmented C} (f g : X ⟶ Y) (h₁ : f.left = g.left) (h₂ : f.right = g.right) :
    f = g :=
  Comma.hom_ext _ _ h₁ h₂


/-- Drop the augmentation. -/
@[simps!]
def drop : Augmented C ⥤ SimplicialObject C :=
  Comma.fst _ _


/-- The point of the augmentation. -/
@[simps!]
def point : Augmented C ⥤ C :=
  Comma.snd _ _


/-- The functor from augmented objects to arrows. -/
@[simps]
def toArrow : Augmented C ⥤ Arrow C where
  obj X :=
    { left := drop.obj X _[0]
      right := point.obj X
      hom := X.hom.app _ }
  map η :=
    { left := (drop.map η).app _
      right := point.map η
      w := by
        /-
          C : Type u
          inst✝ : CategoryTheory.Category.{v, u} C
          X : CategoryTheory.SimplicialObject C
          X✝ Y✝ : CategoryTheory.SimplicialObject.Augmented C
          η : Quiver.Hom X✝ Y✝
          ⊢ Eq (CategoryTheory.CategoryStruct.comp ((CategoryTheory.Functor.id C).map (( …
        -/
        dsimp
        /-
          C : Type u
          inst✝ : CategoryTheory.Category.{v, u} C
          X : CategoryTheory.SimplicialObject C
          X✝ Y✝ : CategoryTheory.SimplicialObject.Augmented C
          η : Quiver.Hom X✝ Y✝
          ⊢ Eq (CategoryTheory.CategoryStruct.comp (η.left.app { unop := SimplexCategory …
        -/
        rw [← NatTrans.comp_app]
        /-
          C : Type u
          inst✝ : CategoryTheory.Category.{v, u} C
          X : CategoryTheory.SimplicialObject C
          X✝ Y✝ : CategoryTheory.SimplicialObject.Augmented C
          η : Quiver.Hom X✝ Y✝
          ⊢ Eq ((CategoryTheory.CategoryStruct.comp η.left Y✝.hom).app { unop := Simplex …
        -/
        erw [η.w]
        /-
          C : Type u
          inst✝ : CategoryTheory.Category.{v, u} C
          X : CategoryTheory.SimplicialObject C
          X✝ Y✝ : CategoryTheory.SimplicialObject.Augmented C
          η : Quiver.Hom X✝ Y✝
          ⊢ Eq ((CategoryTheory.CategoryStruct.comp X✝.hom ((CategoryTheory.SimplicialOb …
        -/
        rfl }
        /-
          🎉 no goals
        -/


/-- The compatibility of a morphism with the augmentation, on 0-simplices -/
@[reassoc]
theorem w₀ {X Y : Augmented C} (f : X ⟶ Y) :
    (Augmented.drop.map f).app (op (SimplexCategory.mk 0)) ≫ Y.hom.app (op (SimplexCategory.mk 0)) =
      X.hom.app (op (SimplexCategory.mk 0)) ≫ Augmented.point.map f := by
  /-
    C : Type u
    inst✝ : CategoryTheory.Category.{v, u} C
    X Y : CategoryTheory.SimplicialObject.Augmented C
    f : Quiver.Hom X Y
    ⊢ Eq (CategoryTheory.CategoryStruct.comp ((CategoryTheory.SimplicialObject.Aug …
  -/
  convert congr_app f.w (op (SimplexCategory.mk 0))
  /-
    🎉 no goals
  -/


/-- Functor composition induces a functor on augmented simplicial objects. -/
@[simp]
def whiskeringObj (D : Type*) [Category D] (F : C ⥤ D) : Augmented C ⥤ Augmented D where
  obj X :=
    { left := ((whiskering _ _).obj F).obj (drop.obj X)
      right := F.obj (point.obj X)
      hom := whiskerRight X.hom F ≫ (Functor.constComp _ _ _).hom }
  map η :=
    { left := whiskerRight η.left _
      right := F.map η.right
      w := by
        /-
          C : Type u
          inst✝¹ : CategoryTheory.Category.{v, u} C
          X : CategoryTheory.SimplicialObject C
          D : Type u_1
          inst✝ : CategoryTheory.Category.{?u.196634, u_1} D
          F : CategoryTheory.Functor C D
          X✝ Y✝ : CategoryTheory.SimplicialObject.Augmented C
          η : Quiver.Hom X✝ Y✝
          ⊢ Eq (CategoryTheory.CategoryStruct.comp ((CategoryTheory.Functor.id (Category …
        -/
        ext
        /-
          case h
          C : Type u
          inst✝¹ : CategoryTheory.Category.{v, u} C
          X : CategoryTheory.SimplicialObject C
          D : Type u_1
          inst✝ : CategoryTheory.Category.{?u.196634, u_1} D
          F : CategoryTheory.Functor C D
          X✝ Y✝ : CategoryTheory.SimplicialObject.Augmented C
          η : Quiver.Hom X✝ Y✝
          n✝ : Opposite SimplexCategory
          ⊢ Eq ((CategoryTheory.CategoryStruct.comp ((CategoryTheory.Functor.id (Categor …
        -/
        dsimp [whiskerRight]
        /-
          case h
          C : Type u
          inst✝¹ : CategoryTheory.Category.{v, u} C
          X : CategoryTheory.SimplicialObject C
          D : Type u_1
          inst✝ : CategoryTheory.Category.{?u.196634, u_1} D
          F : CategoryTheory.Functor C D
          X✝ Y✝ : CategoryTheory.SimplicialObject.Augmented C
          η : Quiver.Hom X✝ Y✝
          n✝ : Opposite SimplexCategory
          ⊢ Eq (CategoryTheory.CategoryStruct.comp (F.map (η.left.app n✝)) (CategoryTheo …
        -/
        simp only [Category.comp_id, ← F.map_comp, ← NatTrans.comp_app]
        /-
          case h
          C : Type u
          inst✝¹ : CategoryTheory.Category.{v, u} C
          X : CategoryTheory.SimplicialObject C
          D : Type u_1
          inst✝ : CategoryTheory.Category.{?u.196634, u_1} D
          F : CategoryTheory.Functor C D
          X✝ Y✝ : CategoryTheory.SimplicialObject.Augmented C
          η : Quiver.Hom X✝ Y✝
          n✝ : Opposite SimplexCategory
          ⊢ Eq (F.map ((CategoryTheory.CategoryStruct.comp η.left Y✝.hom).app n✝)) (F.ma …
        -/
        erw [η.w]
        /-
          case h
          C : Type u
          inst✝¹ : CategoryTheory.Category.{v, u} C
          X : CategoryTheory.SimplicialObject C
          D : Type u_1
          inst✝ : CategoryTheory.Category.{?u.196634, u_1} D
          F : CategoryTheory.Functor C D
          X✝ Y✝ : CategoryTheory.SimplicialObject.Augmented C
          η : Quiver.Hom X✝ Y✝
          n✝ : Opposite SimplexCategory
          ⊢ Eq (F.map ((CategoryTheory.CategoryStruct.comp X✝.hom ((CategoryTheory.Simpl …
        -/
        rfl }
        /-
          🎉 no goals
        -/


/-- Functor composition induces a functor on augmented simplicial objects. -/
@[simps]
def whiskering (D : Type u') [Category.{v'} D] : (C ⥤ D) ⥤ Augmented C ⥤ Augmented D where
  obj := whiskeringObj _ _
  map η :=
    { app := fun A =>
        { left := whiskerLeft _ η
          right := η.app _
          w := by
            /-
              C : Type u
              inst✝¹ : CategoryTheory.Category.{v, u} C
              X : CategoryTheory.SimplicialObject C
              D : Type u'
              inst✝ : CategoryTheory.Category.{v', u'} D
              X✝ Y✝ : CategoryTheory.Functor C D
              η : Quiver.Hom X✝ Y✝
              A : CategoryTheory.SimplicialObject.Augmented C
              ⊢ Eq (CategoryTheory.CategoryStruct.comp ((CategoryTheory.Functor.id (Category …
            -/
            ext n
            /-
              case h
              C : Type u
              inst✝¹ : CategoryTheory.Category.{v, u} C
              X : CategoryTheory.SimplicialObject C
              D : Type u'
              inst✝ : CategoryTheory.Category.{v', u'} D
              X✝ Y✝ : CategoryTheory.Functor C D
              η : Quiver.Hom X✝ Y✝
              A : CategoryTheory.SimplicialObject.Augmented C
              n : Opposite SimplexCategory
              ⊢ Eq ((CategoryTheory.CategoryStruct.comp ((CategoryTheory.Functor.id (Categor …
            -/
            dsimp
            /-
              case h
              C : Type u
              inst✝¹ : CategoryTheory.Category.{v, u} C
              X : CategoryTheory.SimplicialObject C
              D : Type u'
              inst✝ : CategoryTheory.Category.{v', u'} D
              X✝ Y✝ : CategoryTheory.Functor C D
              η : Quiver.Hom X✝ Y✝
              A : CategoryTheory.SimplicialObject.Augmented C
              n : Opposite SimplexCategory
              ⊢ Eq (CategoryTheory.CategoryStruct.comp (η.app (A.left.obj n)) (CategoryTheor …
            -/
            rw [Category.comp_id, Category.comp_id, η.naturality] } }
            /-
              🎉 no goals
            -/
                            /-
                              C : Type u
                              inst✝¹ : CategoryTheory.Category.{v, u} C
                              X : CategoryTheory.SimplicialObject C
                              D : Type u'
                              inst✝ : CategoryTheory.Category.{v', u'} D
                              X✝ Y✝ Z✝ : CategoryTheory.Functor C D
                              x✝¹ : Quiver.Hom X✝ Y✝
                              x✝ : Quiver.Hom Y✝ Z✝
                              ⊢ Eq ({ obj := CategoryTheory.SimplicialObject.Augmented.whiskeringObj C D, ma …
                            -/
                                    /-
                                      🎉 no goals
                                    -/
  map_comp := fun _ _ => by ext <;> rfl
                                    /-
                                      🎉 no goals
                                    -/


/-- Augment a simplicial object with an object. -/
@[simps]
def augment (X : SimplicialObject C) (X₀ : C) (f : X _[0] ⟶ X₀)
    (w : ∀ (i : SimplexCategory) (g₁ g₂ : ([0] : SimplexCategory) ⟶ i),
      X.map g₁.op ≫ f = X.map g₂.op ≫ f) :
    SimplicialObject.Augmented C where
  left := X
  right := X₀
  hom :=
    { app := fun _ => X.map (SimplexCategory.const _ _ 0).op ≫ f
      naturality := by
        /-
          C : Type u
          inst✝ : CategoryTheory.Category.{v, u} C
          X✝ X : CategoryTheory.SimplicialObject C
          X₀ : C
          f : Quiver.Hom (X.obj { unop := SimplexCategory.mk 0 }) X₀
          w : ∀ (i : SimplexCategory) (g₁ g₂ : Quiver.Hom (SimplexCategory.mk 0) i), Eq  …
          ⊢ ∀ ⦃X_1 Y : Opposite SimplexCategory⦄ (f_1 : Quiver.Hom X_1 Y), Eq (CategoryT …
        -/
        intro i j g
        /-
          C : Type u
          inst✝ : CategoryTheory.Category.{v, u} C
          X✝ X : CategoryTheory.SimplicialObject C
          X₀ : C
          f : Quiver.Hom (X.obj { unop := SimplexCategory.mk 0 }) X₀
          w : ∀ (i : SimplexCategory) (g₁ g₂ : Quiver.Hom (SimplexCategory.mk 0) i), Eq  …
          i j : Opposite SimplexCategory
          g : Quiver.Hom i j
          ⊢ Eq (CategoryTheory.CategoryStruct.comp (((CategoryTheory.Functor.id (Categor …
        -/
        dsimp
        /-
          C : Type u
          inst✝ : CategoryTheory.Category.{v, u} C
          X✝ X : CategoryTheory.SimplicialObject C
          X₀ : C
          f : Quiver.Hom (X.obj { unop := SimplexCategory.mk 0 }) X₀
          w : ∀ (i : SimplexCategory) (g₁ g₂ : Quiver.Hom (SimplexCategory.mk 0) i), Eq  …
          i j : Opposite SimplexCategory
          g : Quiver.Hom i j
          ⊢ Eq (CategoryTheory.CategoryStruct.comp (X.map g) (CategoryTheory.CategoryStr …
        -/
        rw [← g.op_unop]
        /-
          C : Type u
          inst✝ : CategoryTheory.Category.{v, u} C
          X✝ X : CategoryTheory.SimplicialObject C
          X₀ : C
          f : Quiver.Hom (X.obj { unop := SimplexCategory.mk 0 }) X₀
          w : ∀ (i : SimplexCategory) (g₁ g₂ : Quiver.Hom (SimplexCategory.mk 0) i), Eq  …
          i j : Opposite SimplexCategory
          g : Quiver.Hom i j
          ⊢ Eq (CategoryTheory.CategoryStruct.comp (X.map g.unop.op) (CategoryTheory.Cat …
        -/
        simpa only [← X.map_comp, ← Category.assoc, Category.comp_id, ← op_comp] using w _ _ _ }
        /-
          🎉 no goals
        -/

-- Porting note: removed @[simp] as the linter complains

theorem augment_hom_zero (X : SimplicialObject C) (X₀ : C) (f : X _[0] ⟶ X₀) (w) :
                                                  /-
                                                    C : Type u
                                                    inst✝ : CategoryTheory.Category.{v, u} C
                                                    X : CategoryTheory.SimplicialObject C
                                                    X₀ : C
                                                    f : Quiver.Hom (X.obj { unop := SimplexCategory.mk 0 }) X₀
                                                    w : ∀ (i : SimplexCategory) (g₁ g₂ : Quiver.Hom (SimplexCategory.mk 0) i), Eq  …
                                                    ⊢ Eq ((X.augment X₀ f w).hom.app { unop := SimplexCategory.mk 0 }) f
                                                  -/
    (X.augment X₀ f w).hom.app (op [0]) = f := by simp
                                                  /-
                                                    🎉 no goals
                                                  -/


/-- Cosimplicial objects. -/
def CosimplicialObject :=
  SimplexCategory ⥤ C


@[simps!]
instance : Category (CosimplicialObject C) := by
  /-
    C : Type u
    inst✝ : CategoryTheory.Category.{v, u} C
    ⊢ CategoryTheory.Category.{?u.251742, max u v} (CategoryTheory.CosimplicialObj …
  -/
  dsimp only [CosimplicialObject]
  /-
    C : Type u
    inst✝ : CategoryTheory.Category.{v, u} C
    ⊢ CategoryTheory.Category.{?u.251742, max u v} (CategoryTheory.Functor Simplex …
  -/
  infer_instance
  /-
    🎉 no goals
  -/


/-- `X _[n]` denotes the `n`th-term of the cosimplicial object X -/
scoped[Simplicial]
  notation3:1000 X " _[" n "]" =>
    (X : CategoryTheory.CosimplicialObject _).obj (SimplexCategory.mk n)


instance {J : Type v} [SmallCategory J] [HasLimitsOfShape J C] :
    HasLimitsOfShape J (CosimplicialObject C) := by
  /-
    C : Type u
    inst✝² : CategoryTheory.Category.{v, u} C
    J : Type v
    inst✝¹ : CategoryTheory.SmallCategory J
    inst✝ : CategoryTheory.Limits.HasLimitsOfShape J C
    ⊢ CategoryTheory.Limits.HasLimitsOfShape J (CategoryTheory.CosimplicialObject C)
  -/
  dsimp [CosimplicialObject]
  /-
    C : Type u
    inst✝² : CategoryTheory.Category.{v, u} C
    J : Type v
    inst✝¹ : CategoryTheory.SmallCategory J
    inst✝ : CategoryTheory.Limits.HasLimitsOfShape J C
    ⊢ CategoryTheory.Limits.HasLimitsOfShape J (CategoryTheory.Functor SimplexCate …
  -/
  infer_instance
  /-
    🎉 no goals
  -/


instance [HasLimits C] : HasLimits (CosimplicialObject C) :=
  ⟨inferInstance⟩


instance {J : Type v} [SmallCategory J] [HasColimitsOfShape J C] :
    HasColimitsOfShape J (CosimplicialObject C) := by
  /-
    C : Type u
    inst✝² : CategoryTheory.Category.{v, u} C
    J : Type v
    inst✝¹ : CategoryTheory.SmallCategory J
    inst✝ : CategoryTheory.Limits.HasColimitsOfShape J C
    ⊢ CategoryTheory.Limits.HasColimitsOfShape J (CategoryTheory.CosimplicialObjec …
  -/
  dsimp [CosimplicialObject]
  /-
    C : Type u
    inst✝² : CategoryTheory.Category.{v, u} C
    J : Type v
    inst✝¹ : CategoryTheory.SmallCategory J
    inst✝ : CategoryTheory.Limits.HasColimitsOfShape J C
    ⊢ CategoryTheory.Limits.HasColimitsOfShape J (CategoryTheory.Functor SimplexCa …
  -/
  infer_instance
  /-
    🎉 no goals
  -/


instance [HasColimits C] : HasColimits (CosimplicialObject C) :=
  ⟨inferInstance⟩


@[ext]
lemma hom_ext {X Y : CosimplicialObject C} (f g : X ⟶ Y)
    (h : ∀ (n : SimplexCategory), f.app n = g.app n) : f = g :=
                   /-
                     C : Type u
                     inst✝ : CategoryTheory.Category.{v, u} C
                     X Y : CategoryTheory.CosimplicialObject C
                     f g : Quiver.Hom X Y
                     h : ∀ (n : SimplexCategory), Eq (f.app n) (g.app n)
                     ⊢ Eq f.app g.app
                   -/
  NatTrans.ext (by ext; apply h)
                        /-
                          🎉 no goals
                        -/


/-- Coface maps for a cosimplicial object. -/
def δ {n} (i : Fin (n + 2)) : X _[n] ⟶ X _[n + 1] :=
  X.map (SimplexCategory.δ i)


/-- Codegeneracy maps for a cosimplicial object. -/
def σ {n} (i : Fin (n + 1)) : X _[n + 1] ⟶ X _[n] :=
  X.map (SimplexCategory.σ i)


/-- Isomorphisms from identities in ℕ. -/
def eqToIso {n m : ℕ} (h : n = m) : X _[n] ≅ X _[m] :=
                                       /-
                                         C : Type u
                                         inst✝ : CategoryTheory.Category.{v, u} C
                                         X : CategoryTheory.CosimplicialObject C
                                         n m : Nat
                                         h : Eq n m
                                         ⊢ Eq (SimplexCategory.mk n) (SimplexCategory.mk m)
                                       -/
  X.mapIso (CategoryTheory.eqToIso (by rw [h]))
                                       /-
                                         🎉 no goals
                                       -/


@[simp]
theorem eqToIso_refl {n : ℕ} (h : n = n) : X.eqToIso h = Iso.refl _ := by
  /-
    C : Type u
    inst✝ : CategoryTheory.Category.{v, u} C
    X : CategoryTheory.CosimplicialObject C
    n : Nat
    h : Eq n n
    ⊢ Eq (X.eqToIso h) (CategoryTheory.Iso.refl (X.obj (SimplexCategory.mk n)))
  -/
  ext
  /-
    case w
    C : Type u
    inst✝ : CategoryTheory.Category.{v, u} C
    X : CategoryTheory.CosimplicialObject C
    n : Nat
    h : Eq n n
    ⊢ Eq (X.eqToIso h).hom (CategoryTheory.Iso.refl (X.obj (SimplexCategory.mk n)) …
  -/
  simp [eqToIso]
  /-
    🎉 no goals
  -/


/-- The generic case of the first cosimplicial identity -/
@[reassoc]
theorem δ_comp_δ {n} {i j : Fin (n + 2)} (H : i ≤ j) :
    X.δ i ≫ X.δ j.succ = X.δ j ≫ X.δ (Fin.castSucc i) := by
  /-
    C : Type u
    inst✝ : CategoryTheory.Category.{v, u} C
    X : CategoryTheory.CosimplicialObject C
    n : Nat
    i j : Fin (HAdd.hAdd n 2)
    H : LE.le i j
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (X.δ i) (X.δ j.succ)) (CategoryTheory …
  -/
  dsimp [δ]
  /-
    C : Type u
    inst✝ : CategoryTheory.Category.{v, u} C
    X : CategoryTheory.CosimplicialObject C
    n : Nat
    i j : Fin (HAdd.hAdd n 2)
    H : LE.le i j
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (X.map (SimplexCategory.δ i)) (X.map  …
  -/
  simp only [← X.map_comp, SimplexCategory.δ_comp_δ H]
  /-
    🎉 no goals
  -/


@[reassoc]
theorem δ_comp_δ' {n} {i : Fin (n + 2)} {j : Fin (n + 3)} (H : Fin.castSucc i < j) :
    X.δ i ≫ X.δ j =
                                         /-
                                           C : Type u
                                           inst✝ : CategoryTheory.Category.{v, u} C
                                           X : CategoryTheory.CosimplicialObject C
                                           n : Nat
                                           i : Fin (HAdd.hAdd n 2)
                                           j : Fin (HAdd.hAdd n 3)
                                           H : LT.lt i.castSucc j
                                           hj : Eq j 0
                                           ⊢ False
                                         -/
      X.δ (j.pred fun (hj : j = 0) => by simp only [hj, Fin.not_lt_zero] at H) ≫
                                         /-
                                           🎉 no goals
                                         -/
        X.δ (Fin.castSucc i) := by
  /-
    C : Type u
    inst✝ : CategoryTheory.Category.{v, u} C
    X : CategoryTheory.CosimplicialObject C
    n : Nat
    i : Fin (HAdd.hAdd n 2)
    j : Fin (HAdd.hAdd n 3)
    H : LT.lt i.castSucc j
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (X.δ i) (X.δ j)) (CategoryTheory.Cate …
  -/
  dsimp [δ]
  /-
    C : Type u
    inst✝ : CategoryTheory.Category.{v, u} C
    X : CategoryTheory.CosimplicialObject C
    n : Nat
    i : Fin (HAdd.hAdd n 2)
    j : Fin (HAdd.hAdd n 3)
    H : LT.lt i.castSucc j
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (X.map (SimplexCategory.δ i)) (X.map  …
  -/
  simp only [← X.map_comp, ← op_comp, SimplexCategory.δ_comp_δ' H]
  /-
    🎉 no goals
  -/


@[reassoc]
theorem δ_comp_δ'' {n} {i : Fin (n + 3)} {j : Fin (n + 2)} (H : i ≤ Fin.castSucc j) :
    X.δ (i.castLT (Nat.lt_of_le_of_lt (Fin.le_iff_val_le_val.mp H) j.is_lt)) ≫ X.δ j.succ =
      X.δ j ≫ X.δ i := by
  /-
    C : Type u
    inst✝ : CategoryTheory.Category.{v, u} C
    X : CategoryTheory.CosimplicialObject C
    n : Nat
    i : Fin (HAdd.hAdd n 3)
    j : Fin (HAdd.hAdd n 2)
    H : LE.le i j.castSucc
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (X.δ (i.castLT ⋯)) (X.δ j.succ)) (Cat …
  -/
  dsimp [δ]
  /-
    C : Type u
    inst✝ : CategoryTheory.Category.{v, u} C
    X : CategoryTheory.CosimplicialObject C
    n : Nat
    i : Fin (HAdd.hAdd n 3)
    j : Fin (HAdd.hAdd n 2)
    H : LE.le i j.castSucc
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (X.map (SimplexCategory.δ (i.castLT ⋯ …
  -/
  simp only [← X.map_comp, ← op_comp, SimplexCategory.δ_comp_δ'' H]
  /-
    🎉 no goals
  -/


/-- The special case of the first cosimplicial identity -/
@[reassoc]
theorem δ_comp_δ_self {n} {i : Fin (n + 2)} :
    X.δ i ≫ X.δ (Fin.castSucc i) = X.δ i ≫ X.δ i.succ := by
  /-
    C : Type u
    inst✝ : CategoryTheory.Category.{v, u} C
    X : CategoryTheory.CosimplicialObject C
    n : Nat
    i : Fin (HAdd.hAdd n 2)
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (X.δ i) (X.δ i.castSucc)) (CategoryTh …
  -/
  dsimp [δ]
  /-
    C : Type u
    inst✝ : CategoryTheory.Category.{v, u} C
    X : CategoryTheory.CosimplicialObject C
    n : Nat
    i : Fin (HAdd.hAdd n 2)
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (X.map (SimplexCategory.δ i)) (X.map  …
  -/
  simp only [← X.map_comp, SimplexCategory.δ_comp_δ_self]
  /-
    🎉 no goals
  -/


@[reassoc]
theorem δ_comp_δ_self' {n} {i : Fin (n + 2)} {j : Fin (n + 3)} (H : j = Fin.castSucc i) :
    X.δ i ≫ X.δ j = X.δ i ≫ X.δ i.succ := by
  /-
    C : Type u
    inst✝ : CategoryTheory.Category.{v, u} C
    X : CategoryTheory.CosimplicialObject C
    n : Nat
    i : Fin (HAdd.hAdd n 2)
    j : Fin (HAdd.hAdd n 3)
    H : Eq j i.castSucc
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (X.δ i) (X.δ j)) (CategoryTheory.Cate …
  -/
  subst H
  /-
    C : Type u
    inst✝ : CategoryTheory.Category.{v, u} C
    X : CategoryTheory.CosimplicialObject C
    n : Nat
    i : Fin (HAdd.hAdd n 2)
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (X.δ i) (X.δ i.castSucc)) (CategoryTh …
  -/
  rw [δ_comp_δ_self]
  /-
    🎉 no goals
  -/


/-- The second cosimplicial identity -/
@[reassoc]
theorem δ_comp_σ_of_le {n} {i : Fin (n + 2)} {j : Fin (n + 1)} (H : i ≤ Fin.castSucc j) :
    X.δ (Fin.castSucc i) ≫ X.σ j.succ = X.σ j ≫ X.δ i := by
  /-
    C : Type u
    inst✝ : CategoryTheory.Category.{v, u} C
    X : CategoryTheory.CosimplicialObject C
    n : Nat
    i : Fin (HAdd.hAdd n 2)
    j : Fin (HAdd.hAdd n 1)
    H : LE.le i j.castSucc
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (X.δ i.castSucc) (X.σ j.succ)) (Categ …
  -/
  dsimp [δ, σ]
  /-
    C : Type u
    inst✝ : CategoryTheory.Category.{v, u} C
    X : CategoryTheory.CosimplicialObject C
    n : Nat
    i : Fin (HAdd.hAdd n 2)
    j : Fin (HAdd.hAdd n 1)
    H : LE.le i j.castSucc
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (X.map (SimplexCategory.δ i.castSucc) …
  -/
  simp only [← X.map_comp, SimplexCategory.δ_comp_σ_of_le H]
  /-
    🎉 no goals
  -/


/-- The first part of the third cosimplicial identity -/
@[reassoc]
theorem δ_comp_σ_self {n} {i : Fin (n + 1)} : X.δ (Fin.castSucc i) ≫ X.σ i = 𝟙 _ := by
  /-
    C : Type u
    inst✝ : CategoryTheory.Category.{v, u} C
    X : CategoryTheory.CosimplicialObject C
    n : Nat
    i : Fin (HAdd.hAdd n 1)
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (X.δ i.castSucc) (X.σ i)) (CategoryTh …
  -/
  dsimp [δ, σ]
  /-
    C : Type u
    inst✝ : CategoryTheory.Category.{v, u} C
    X : CategoryTheory.CosimplicialObject C
    n : Nat
    i : Fin (HAdd.hAdd n 1)
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (X.map (SimplexCategory.δ i.castSucc) …
  -/
  simp only [← X.map_comp, SimplexCategory.δ_comp_σ_self, X.map_id]
  /-
    🎉 no goals
  -/


@[reassoc]
theorem δ_comp_σ_self' {n} {j : Fin (n + 2)} {i : Fin (n + 1)} (H : j = Fin.castSucc i) :
    X.δ j ≫ X.σ i = 𝟙 _ := by
  /-
    C : Type u
    inst✝ : CategoryTheory.Category.{v, u} C
    X : CategoryTheory.CosimplicialObject C
    n : Nat
    j : Fin (HAdd.hAdd n 2)
    i : Fin (HAdd.hAdd n 1)
    H : Eq j i.castSucc
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (X.δ j) (X.σ i)) (CategoryTheory.Cate …
  -/
  subst H
  /-
    C : Type u
    inst✝ : CategoryTheory.Category.{v, u} C
    X : CategoryTheory.CosimplicialObject C
    n : Nat
    i : Fin (HAdd.hAdd n 1)
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (X.δ i.castSucc) (X.σ i)) (CategoryTh …
  -/
  rw [δ_comp_σ_self]
  /-
    🎉 no goals
  -/


/-- The second part of the third cosimplicial identity -/
@[reassoc]
theorem δ_comp_σ_succ {n} {i : Fin (n + 1)} : X.δ i.succ ≫ X.σ i = 𝟙 _ := by
  /-
    C : Type u
    inst✝ : CategoryTheory.Category.{v, u} C
    X : CategoryTheory.CosimplicialObject C
    n : Nat
    i : Fin (HAdd.hAdd n 1)
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (X.δ i.succ) (X.σ i)) (CategoryTheory …
  -/
  dsimp [δ, σ]
  /-
    C : Type u
    inst✝ : CategoryTheory.Category.{v, u} C
    X : CategoryTheory.CosimplicialObject C
    n : Nat
    i : Fin (HAdd.hAdd n 1)
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (X.map (SimplexCategory.δ i.succ)) (X …
  -/
  simp only [← X.map_comp, SimplexCategory.δ_comp_σ_succ, X.map_id]
  /-
    🎉 no goals
  -/


@[reassoc]
theorem δ_comp_σ_succ' {n} {j : Fin (n + 2)} {i : Fin (n + 1)} (H : j = i.succ) :
    X.δ j ≫ X.σ i = 𝟙 _ := by
  /-
    C : Type u
    inst✝ : CategoryTheory.Category.{v, u} C
    X : CategoryTheory.CosimplicialObject C
    n : Nat
    j : Fin (HAdd.hAdd n 2)
    i : Fin (HAdd.hAdd n 1)
    H : Eq j i.succ
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (X.δ j) (X.σ i)) (CategoryTheory.Cate …
  -/
  subst H
  /-
    C : Type u
    inst✝ : CategoryTheory.Category.{v, u} C
    X : CategoryTheory.CosimplicialObject C
    n : Nat
    i : Fin (HAdd.hAdd n 1)
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (X.δ i.succ) (X.σ i)) (CategoryTheory …
  -/
  rw [δ_comp_σ_succ]
  /-
    🎉 no goals
  -/


/-- The fourth cosimplicial identity -/
@[reassoc]
theorem δ_comp_σ_of_gt {n} {i : Fin (n + 2)} {j : Fin (n + 1)} (H : Fin.castSucc j < i) :
    X.δ i.succ ≫ X.σ (Fin.castSucc j) = X.σ j ≫ X.δ i := by
  /-
    C : Type u
    inst✝ : CategoryTheory.Category.{v, u} C
    X : CategoryTheory.CosimplicialObject C
    n : Nat
    i : Fin (HAdd.hAdd n 2)
    j : Fin (HAdd.hAdd n 1)
    H : LT.lt j.castSucc i
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (X.δ i.succ) (X.σ j.castSucc)) (Categ …
  -/
  dsimp [δ, σ]
  /-
    C : Type u
    inst✝ : CategoryTheory.Category.{v, u} C
    X : CategoryTheory.CosimplicialObject C
    n : Nat
    i : Fin (HAdd.hAdd n 2)
    j : Fin (HAdd.hAdd n 1)
    H : LT.lt j.castSucc i
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (X.map (SimplexCategory.δ i.succ)) (X …
  -/
  simp only [← X.map_comp, SimplexCategory.δ_comp_σ_of_gt H]
  /-
    🎉 no goals
  -/


@[reassoc]
theorem δ_comp_σ_of_gt' {n} {i : Fin (n + 3)} {j : Fin (n + 2)} (H : j.succ < i) :
    X.δ i ≫ X.σ j =
      X.σ (j.castLT ((add_lt_add_iff_right 1).mp (lt_of_lt_of_le H i.is_le))) ≫
        X.δ (i.pred <|
                                 /-
                                   C : Type u
                                   inst✝ : CategoryTheory.Category.{v, u} C
                                   X : CategoryTheory.CosimplicialObject C
                                   n : Nat
                                   i : Fin (HAdd.hAdd n 3)
                                   j : Fin (HAdd.hAdd n 2)
                                   H : LT.lt j.succ i
                                   hi : Eq i 0
                                   ⊢ False
                                 -/
          fun (hi : i = 0) => by simp only [Fin.not_lt_zero, hi] at H) := by
                                 /-
                                   🎉 no goals
                                 -/
  /-
    C : Type u
    inst✝ : CategoryTheory.Category.{v, u} C
    X : CategoryTheory.CosimplicialObject C
    n : Nat
    i : Fin (HAdd.hAdd n 3)
    j : Fin (HAdd.hAdd n 2)
    H : LT.lt j.succ i
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (X.δ i) (X.σ j)) (CategoryTheory.Cate …
  -/
  dsimp [δ, σ]
  /-
    C : Type u
    inst✝ : CategoryTheory.Category.{v, u} C
    X : CategoryTheory.CosimplicialObject C
    n : Nat
    i : Fin (HAdd.hAdd n 3)
    j : Fin (HAdd.hAdd n 2)
    H : LT.lt j.succ i
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (X.map (SimplexCategory.δ i)) (X.map  …
  -/
  simp only [← X.map_comp, ← op_comp, SimplexCategory.δ_comp_σ_of_gt' H]
  /-
    🎉 no goals
  -/


/-- The fifth cosimplicial identity -/
@[reassoc]
theorem σ_comp_σ {n} {i j : Fin (n + 1)} (H : i ≤ j) :
    X.σ (Fin.castSucc i) ≫ X.σ j = X.σ j.succ ≫ X.σ i := by
  /-
    C : Type u
    inst✝ : CategoryTheory.Category.{v, u} C
    X : CategoryTheory.CosimplicialObject C
    n : Nat
    i j : Fin (HAdd.hAdd n 1)
    H : LE.le i j
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (X.σ i.castSucc) (X.σ j)) (CategoryTh …
  -/
  dsimp [δ, σ]
  /-
    C : Type u
    inst✝ : CategoryTheory.Category.{v, u} C
    X : CategoryTheory.CosimplicialObject C
    n : Nat
    i j : Fin (HAdd.hAdd n 1)
    H : LE.le i j
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (X.map (SimplexCategory.σ i.castSucc) …
  -/
  simp only [← X.map_comp, SimplexCategory.σ_comp_σ H]
  /-
    🎉 no goals
  -/


@[reassoc (attr := simp)]
theorem δ_naturality {X' X : CosimplicialObject C} (f : X ⟶ X') {n : ℕ} (i : Fin (n + 2)) :
    X.δ i ≫ f.app (SimplexCategory.mk (n + 1)) = f.app (SimplexCategory.mk n) ≫ X'.δ i :=
  f.naturality _


@[reassoc (attr := simp)]
theorem σ_naturality {X' X : CosimplicialObject C} (f : X ⟶ X') {n : ℕ} (i : Fin (n + 1)) :
    X.σ i ≫ f.app (SimplexCategory.mk n) = f.app (SimplexCategory.mk (n + 1)) ≫ X'.σ i :=
  f.naturality _


/-- Functor composition induces a functor on cosimplicial objects. -/
@[simps!]
def whiskering (D : Type*) [Category D] : (C ⥤ D) ⥤ CosimplicialObject C ⥤ CosimplicialObject D :=
  whiskeringRight _ _ _

-- Porting note (https://github.com/leanprover-community/mathlib4/issues/5171): removed @[nolint has_nonempty_instance]

/-- Truncated cosimplicial objects. -/
def Truncated (n : ℕ) :=
  SimplexCategory.Truncated n ⥤ C


instance {n : ℕ} : Category (Truncated C n) := by
  /-
    C : Type u
    inst✝ : CategoryTheory.Category.{v, u} C
    X : CategoryTheory.CosimplicialObject C
    n : Nat
    ⊢ CategoryTheory.Category.{?u.331101, max u v} (CategoryTheory.CosimplicialObj …
  -/
  dsimp [Truncated]
  /-
    C : Type u
    inst✝ : CategoryTheory.Category.{v, u} C
    X : CategoryTheory.CosimplicialObject C
    n : Nat
    ⊢ CategoryTheory.Category.{?u.331101, max u v} (CategoryTheory.Functor (Simple …
  -/
  infer_instance
  /-
    🎉 no goals
  -/


instance {n} {J : Type v} [SmallCategory J] [HasLimitsOfShape J C] :
    HasLimitsOfShape J (CosimplicialObject.Truncated C n) := by
  /-
    C : Type u
    inst✝² : CategoryTheory.Category.{v, u} C
    X : CategoryTheory.CosimplicialObject C
    n : Nat
    J : Type v
    inst✝¹ : CategoryTheory.SmallCategory J
    inst✝ : CategoryTheory.Limits.HasLimitsOfShape J C
    ⊢ CategoryTheory.Limits.HasLimitsOfShape J (CategoryTheory.CosimplicialObject. …
  -/
  dsimp [Truncated]
  /-
    C : Type u
    inst✝² : CategoryTheory.Category.{v, u} C
    X : CategoryTheory.CosimplicialObject C
    n : Nat
    J : Type v
    inst✝¹ : CategoryTheory.SmallCategory J
    inst✝ : CategoryTheory.Limits.HasLimitsOfShape J C
    ⊢ CategoryTheory.Limits.HasLimitsOfShape J (CategoryTheory.Functor (SimplexCat …
  -/
  infer_instance
  /-
    🎉 no goals
  -/


instance {n} [HasLimits C] : HasLimits (CosimplicialObject.Truncated C n) :=
  ⟨inferInstance⟩


instance {n} {J : Type v} [SmallCategory J] [HasColimitsOfShape J C] :
    HasColimitsOfShape J (CosimplicialObject.Truncated C n) := by
  /-
    C : Type u
    inst✝² : CategoryTheory.Category.{v, u} C
    X : CategoryTheory.CosimplicialObject C
    n : Nat
    J : Type v
    inst✝¹ : CategoryTheory.SmallCategory J
    inst✝ : CategoryTheory.Limits.HasColimitsOfShape J C
    ⊢ CategoryTheory.Limits.HasColimitsOfShape J (CategoryTheory.CosimplicialObjec …
  -/
  dsimp [Truncated]
  /-
    C : Type u
    inst✝² : CategoryTheory.Category.{v, u} C
    X : CategoryTheory.CosimplicialObject C
    n : Nat
    J : Type v
    inst✝¹ : CategoryTheory.SmallCategory J
    inst✝ : CategoryTheory.Limits.HasColimitsOfShape J C
    ⊢ CategoryTheory.Limits.HasColimitsOfShape J (CategoryTheory.Functor (SimplexC …
  -/
  infer_instance
  /-
    🎉 no goals
  -/


instance {n} [HasColimits C] : HasColimits (CosimplicialObject.Truncated C n) :=
  ⟨inferInstance⟩


/-- Functor composition induces a functor on truncated cosimplicial objects. -/
@[simps!]
def whiskering {n} (D : Type*) [Category D] : (C ⥤ D) ⥤ Truncated C n ⥤ Truncated D n :=
  whiskeringRight _ _ _


/-- The truncation functor from cosimplicial objects to truncated cosimplicial objects. -/
def truncation (n : ℕ) : CosimplicialObject C ⥤ CosimplicialObject.Truncated C n :=
  (whiskeringLeft _ _ _).obj (SimplexCategory.Truncated.inclusion n)


/-- The constant cosimplicial object. -/
abbrev const : C ⥤ CosimplicialObject C :=
  CategoryTheory.Functor.const _

-- Porting note (https://github.com/leanprover-community/mathlib4/issues/5171): removed @[nolint has_nonempty_instance]

/-- Augmented cosimplicial objects. -/
def Augmented :=
  Comma (const C) (𝟭 (CosimplicialObject C))


@[simps!]
instance : Category (Augmented C) := by
  /-
    C : Type u
    inst✝ : CategoryTheory.Category.{v, u} C
    X : CategoryTheory.CosimplicialObject C
    ⊢ CategoryTheory.Category.{?u.382615, max u v} (CategoryTheory.CosimplicialObj …
  -/
  dsimp only [Augmented]
  /-
    C : Type u
    inst✝ : CategoryTheory.Category.{v, u} C
    X : CategoryTheory.CosimplicialObject C
    ⊢ CategoryTheory.Category.{?u.382615, max u v} (CategoryTheory.Comma (Category …
  -/
  infer_instance
  /-
    🎉 no goals
  -/


/-- Drop the augmentation. -/
@[simps!]
def drop : Augmented C ⥤ CosimplicialObject C :=
  Comma.snd _ _


/-- The point of the augmentation. -/
@[simps!]
def point : Augmented C ⥤ C :=
  Comma.fst _ _


/-- The functor from augmented objects to arrows. -/
@[simps!]
def toArrow : Augmented C ⥤ Arrow C where
  obj X :=
    { left := point.obj X
      right := drop.obj X _[0]
      hom := X.hom.app _ }
  map η :=
    { left := point.map η
      right := (drop.map η).app _
      w := by
        /-
          C : Type u
          inst✝ : CategoryTheory.Category.{v, u} C
          X : CategoryTheory.CosimplicialObject C
          X✝ Y✝ : CategoryTheory.CosimplicialObject.Augmented C
          η : Quiver.Hom X✝ Y✝
          ⊢ Eq (CategoryTheory.CategoryStruct.comp ((CategoryTheory.Functor.id C).map (C …
        -/
        dsimp
        /-
          C : Type u
          inst✝ : CategoryTheory.Category.{v, u} C
          X : CategoryTheory.CosimplicialObject C
          X✝ Y✝ : CategoryTheory.CosimplicialObject.Augmented C
          η : Quiver.Hom X✝ Y✝
          ⊢ Eq (CategoryTheory.CategoryStruct.comp η.left (Y✝.hom.app (SimplexCategory.m …
        -/
        rw [← NatTrans.comp_app]
        /-
          C : Type u
          inst✝ : CategoryTheory.Category.{v, u} C
          X : CategoryTheory.CosimplicialObject C
          X✝ Y✝ : CategoryTheory.CosimplicialObject.Augmented C
          η : Quiver.Hom X✝ Y✝
          ⊢ Eq (CategoryTheory.CategoryStruct.comp η.left (Y✝.hom.app (SimplexCategory.m …
        -/
        erw [← η.w]
        /-
          C : Type u
          inst✝ : CategoryTheory.Category.{v, u} C
          X : CategoryTheory.CosimplicialObject C
          X✝ Y✝ : CategoryTheory.CosimplicialObject.Augmented C
          η : Quiver.Hom X✝ Y✝
          ⊢ Eq (CategoryTheory.CategoryStruct.comp η.left (Y✝.hom.app (SimplexCategory.m …
        -/
        rfl }
        /-
          🎉 no goals
        -/


/-- Functor composition induces a functor on augmented cosimplicial objects. -/
@[simp]
def whiskeringObj (D : Type*) [Category D] (F : C ⥤ D) : Augmented C ⥤ Augmented D where
  obj X :=
    { left := F.obj (point.obj X)
      right := ((whiskering _ _).obj F).obj (drop.obj X)
      hom := (Functor.constComp _ _ _).inv ≫ whiskerRight X.hom F }
  map η :=
    { left := F.map η.left
      right := whiskerRight η.right _
      w := by
        /-
          C : Type u
          inst✝¹ : CategoryTheory.Category.{v, u} C
          X : CategoryTheory.CosimplicialObject C
          D : Type u_1
          inst✝ : CategoryTheory.Category.{?u.401741, u_1} D
          F : CategoryTheory.Functor C D
          X✝ Y✝ : CategoryTheory.CosimplicialObject.Augmented C
          η : Quiver.Hom X✝ Y✝
          ⊢ Eq (CategoryTheory.CategoryStruct.comp ((CategoryTheory.CosimplicialObject.c …
        -/
        ext
        /-
          case h
          C : Type u
          inst✝¹ : CategoryTheory.Category.{v, u} C
          X : CategoryTheory.CosimplicialObject C
          D : Type u_1
          inst✝ : CategoryTheory.Category.{?u.401741, u_1} D
          F : CategoryTheory.Functor C D
          X✝ Y✝ : CategoryTheory.CosimplicialObject.Augmented C
          η : Quiver.Hom X✝ Y✝
          n✝ : SimplexCategory
          ⊢ Eq ((CategoryTheory.CategoryStruct.comp ((CategoryTheory.CosimplicialObject. …
        -/
        dsimp
        /-
          case h
          C : Type u
          inst✝¹ : CategoryTheory.Category.{v, u} C
          X : CategoryTheory.CosimplicialObject C
          D : Type u_1
          inst✝ : CategoryTheory.Category.{?u.401741, u_1} D
          F : CategoryTheory.Functor C D
          X✝ Y✝ : CategoryTheory.CosimplicialObject.Augmented C
          η : Quiver.Hom X✝ Y✝
          n✝ : SimplexCategory
          ⊢ Eq (CategoryTheory.CategoryStruct.comp (F.map η.left) (CategoryTheory.Catego …
        -/
        rw [Category.id_comp, Category.id_comp, ← F.map_comp, ← F.map_comp, ← NatTrans.comp_app]
        /-
          case h
          C : Type u
          inst✝¹ : CategoryTheory.Category.{v, u} C
          X : CategoryTheory.CosimplicialObject C
          D : Type u_1
          inst✝ : CategoryTheory.Category.{?u.401741, u_1} D
          F : CategoryTheory.Functor C D
          X✝ Y✝ : CategoryTheory.CosimplicialObject.Augmented C
          η : Quiver.Hom X✝ Y✝
          n✝ : SimplexCategory
          ⊢ Eq (F.map (CategoryTheory.CategoryStruct.comp η.left (Y✝.hom.app n✝))) (F.ma …
        -/
        erw [← η.w]
        /-
          case h
          C : Type u
          inst✝¹ : CategoryTheory.Category.{v, u} C
          X : CategoryTheory.CosimplicialObject C
          D : Type u_1
          inst✝ : CategoryTheory.Category.{?u.401741, u_1} D
          F : CategoryTheory.Functor C D
          X✝ Y✝ : CategoryTheory.CosimplicialObject.Augmented C
          η : Quiver.Hom X✝ Y✝
          n✝ : SimplexCategory
          ⊢ Eq (F.map (CategoryTheory.CategoryStruct.comp η.left (Y✝.hom.app n✝))) (F.ma …
        -/
        rfl }
        /-
          🎉 no goals
        -/


/-- Functor composition induces a functor on augmented cosimplicial objects. -/
@[simps]
def whiskering (D : Type u') [Category.{v'} D] : (C ⥤ D) ⥤ Augmented C ⥤ Augmented D where
  obj := whiskeringObj _ _
  map η :=
    { app := fun A =>
        { left := η.app _
          right := whiskerLeft _ η
          w := by
            /-
              C : Type u
              inst✝¹ : CategoryTheory.Category.{v, u} C
              X : CategoryTheory.CosimplicialObject C
              D : Type u'
              inst✝ : CategoryTheory.Category.{v', u'} D
              X✝ Y✝ : CategoryTheory.Functor C D
              η : Quiver.Hom X✝ Y✝
              A : CategoryTheory.CosimplicialObject.Augmented C
              ⊢ Eq (CategoryTheory.CategoryStruct.comp ((CategoryTheory.CosimplicialObject.c …
            -/
            ext n
            /-
              case h
              C : Type u
              inst✝¹ : CategoryTheory.Category.{v, u} C
              X : CategoryTheory.CosimplicialObject C
              D : Type u'
              inst✝ : CategoryTheory.Category.{v', u'} D
              X✝ Y✝ : CategoryTheory.Functor C D
              η : Quiver.Hom X✝ Y✝
              A : CategoryTheory.CosimplicialObject.Augmented C
              n : SimplexCategory
              ⊢ Eq ((CategoryTheory.CategoryStruct.comp ((CategoryTheory.CosimplicialObject. …
            -/
            dsimp
            /-
              case h
              C : Type u
              inst✝¹ : CategoryTheory.Category.{v, u} C
              X : CategoryTheory.CosimplicialObject C
              D : Type u'
              inst✝ : CategoryTheory.Category.{v', u'} D
              X✝ Y✝ : CategoryTheory.Functor C D
              η : Quiver.Hom X✝ Y✝
              A : CategoryTheory.CosimplicialObject.Augmented C
              n : SimplexCategory
              ⊢ Eq (CategoryTheory.CategoryStruct.comp (η.app A.left) (CategoryTheory.Catego …
            -/
            rw [Category.id_comp, Category.id_comp, η.naturality] }
            /-
              🎉 no goals
            -/
                                    /-
                                      C : Type u
                                      inst✝¹ : CategoryTheory.Category.{v, u} C
                                      X : CategoryTheory.CosimplicialObject C
                                      D : Type u'
                                      inst✝ : CategoryTheory.Category.{v', u'} D
                                      X✝ Y✝ : CategoryTheory.Functor C D
                                      η : Quiver.Hom X✝ Y✝
                                      x✝¹ x✝ : CategoryTheory.CosimplicialObject.Augmented C
                                      f : Quiver.Hom x✝¹ x✝
                                      ⊢ Eq (CategoryTheory.CategoryStruct.comp ((CategoryTheory.CosimplicialObject.A …
                                    -/
                                                      /-
                                                        🎉 no goals
                                                      -/
      naturality := fun _ _ f => by ext <;> dsimp <;> simp }
                                                      /-
                                                        🎉 no goals
                                                      -/


/-- Augment a cosimplicial object with an object. -/
@[simps]
def augment (X : CosimplicialObject C) (X₀ : C) (f : X₀ ⟶ X.obj [0])
    (w : ∀ (i : SimplexCategory) (g₁ g₂ : ([0] : SimplexCategory) ⟶ i),
      f ≫ X.map g₁ = f ≫ X.map g₂) : CosimplicialObject.Augmented C where
  left := X₀
  right := X
  hom :=
    { app := fun _ => f ≫ X.map (SimplexCategory.const _ _ 0)
      naturality := by
        /-
          C : Type u
          inst✝ : CategoryTheory.Category.{v, u} C
          X✝ X : CategoryTheory.CosimplicialObject C
          X₀ : C
          f : Quiver.Hom X₀ (X.obj (SimplexCategory.mk 0))
          w : ∀ (i : SimplexCategory) (g₁ g₂ : Quiver.Hom (SimplexCategory.mk 0) i), Eq  …
          ⊢ ∀ ⦃X_1 Y : SimplexCategory⦄ (f_1 : Quiver.Hom X_1 Y), Eq (CategoryTheory.Cat …
        -/
        intro i j g
        /-
          C : Type u
          inst✝ : CategoryTheory.Category.{v, u} C
          X✝ X : CategoryTheory.CosimplicialObject C
          X₀ : C
          f : Quiver.Hom X₀ (X.obj (SimplexCategory.mk 0))
          w : ∀ (i : SimplexCategory) (g₁ g₂ : Quiver.Hom (SimplexCategory.mk 0) i), Eq  …
          i j : SimplexCategory
          g : Quiver.Hom i j
          ⊢ Eq (CategoryTheory.CategoryStruct.comp (((CategoryTheory.CosimplicialObject. …
        -/
        dsimp
        /-
          C : Type u
          inst✝ : CategoryTheory.Category.{v, u} C
          X✝ X : CategoryTheory.CosimplicialObject C
          X₀ : C
          f : Quiver.Hom X₀ (X.obj (SimplexCategory.mk 0))
          w : ∀ (i : SimplexCategory) (g₁ g₂ : Quiver.Hom (SimplexCategory.mk 0) i), Eq  …
          i j : SimplexCategory
          g : Quiver.Hom i j
          ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.id X₀) …
        -/
        rw [Category.id_comp, Category.assoc, ← X.map_comp, w] }
        /-
          🎉 no goals
        -/

-- Porting note: removed @[simp] as the linter complains

theorem augment_hom_zero (X : CosimplicialObject C) (X₀ : C) (f : X₀ ⟶ X.obj [0]) (w) :
                                             /-
                                               C : Type u
                                               inst✝ : CategoryTheory.Category.{v, u} C
                                               X : CategoryTheory.CosimplicialObject C
                                               X₀ : C
                                               f : Quiver.Hom X₀ (X.obj (SimplexCategory.mk 0))
                                               w : ∀ (i : SimplexCategory) (g₁ g₂ : Quiver.Hom (SimplexCategory.mk 0) i), Eq  …
                                               ⊢ Eq ((X.augment X₀ f w).hom.app (SimplexCategory.mk 0)) f
                                             -/
    (X.augment X₀ f w).hom.app [0] = f := by simp
                                             /-
                                               🎉 no goals
                                             -/


/-- The anti-equivalence between simplicial objects and cosimplicial objects. -/
@[simps!]
def simplicialCosimplicialEquiv : (SimplicialObject C)ᵒᵖ ≌ CosimplicialObject Cᵒᵖ :=
  Functor.leftOpRightOpEquiv _ _


/-- The anti-equivalence between cosimplicial objects and simplicial objects. -/
@[simps!]
def cosimplicialSimplicialEquiv : (CosimplicialObject C)ᵒᵖ ≌ SimplicialObject Cᵒᵖ :=
  Functor.opUnopEquiv _ _


/-- Construct an augmented cosimplicial object in the opposite
category from an augmented simplicial object. -/
@[simps!]
def SimplicialObject.Augmented.rightOp (X : SimplicialObject.Augmented C) :
    CosimplicialObject.Augmented Cᵒᵖ where
  left := Opposite.op X.right
  right := X.left.rightOp
  hom := NatTrans.rightOp X.hom


/-- Construct an augmented simplicial object from an augmented cosimplicial
object in the opposite category. -/
@[simps!]
def CosimplicialObject.Augmented.leftOp (X : CosimplicialObject.Augmented Cᵒᵖ) :
    SimplicialObject.Augmented C where
  left := X.right.leftOp
  right := X.left.unop
  hom := NatTrans.leftOp X.hom


/-- Converting an augmented simplicial object to an augmented cosimplicial
object and back is isomorphic to the given object. -/
@[simps!]
def SimplicialObject.Augmented.rightOpLeftOpIso (X : SimplicialObject.Augmented C) :
    X.rightOp.leftOp ≅ X :=
                                                                    /-
                                                                      C : Type u
                                                                      inst✝ : CategoryTheory.Category.{v, u} C
                                                                      X : CategoryTheory.SimplicialObject.Augmented C
                                                                      ⊢ Eq X.rightOp.leftOp.right X.right
                                                                    -/
                                                                    /-
                                                                      🎉 no goals
                                                                    -/
  Comma.isoMk X.left.rightOpLeftOpIso (CategoryTheory.eqToIso <| by aesop_cat)
  /-
    🎉 no goals
  -/


/-- Converting an augmented cosimplicial object to an augmented simplicial
object and back is isomorphic to the given object. -/
@[simps!]
def CosimplicialObject.Augmented.leftOpRightOpIso (X : CosimplicialObject.Augmented Cᵒᵖ) :
    X.leftOp.rightOp ≅ X :=
                                            /-
                                              C : Type u
                                              inst✝ : CategoryTheory.Category.{v, u} C
                                              X : CategoryTheory.CosimplicialObject.Augmented (Opposite C)
                                              ⊢ Eq X.leftOp.rightOp.left X.left
                                            -/
                                            /-
                                              🎉 no goals
                                            -/
  Comma.isoMk (CategoryTheory.eqToIso <| by simp) X.right.leftOpRightOpIso
  /-
    🎉 no goals
  -/


/-- A functorial version of `SimplicialObject.Augmented.rightOp`. -/
@[simps]
def simplicialToCosimplicialAugmented :
    (SimplicialObject.Augmented C)ᵒᵖ ⥤ CosimplicialObject.Augmented Cᵒᵖ where
  obj X := X.unop.rightOp
  map f :=
    { left := f.unop.right.op
      right := NatTrans.rightOp f.unop.left
      w := by
        /-
          C : Type u
          inst✝ : CategoryTheory.Category.{v, u} C
          X✝ Y✝ : Opposite (CategoryTheory.SimplicialObject.Augmented C)
          f : Quiver.Hom X✝ Y✝
          ⊢ Eq (CategoryTheory.CategoryStruct.comp ((CategoryTheory.CosimplicialObject.c …
        -/
        ext x
        /-
          case h
          C : Type u
          inst✝ : CategoryTheory.Category.{v, u} C
          X✝ Y✝ : Opposite (CategoryTheory.SimplicialObject.Augmented C)
          f : Quiver.Hom X✝ Y✝
          x : SimplexCategory
          ⊢ Eq ((CategoryTheory.CategoryStruct.comp ((CategoryTheory.CosimplicialObject. …
        -/
        dsimp
        /-
          case h
          C : Type u
          inst✝ : CategoryTheory.Category.{v, u} C
          X✝ Y✝ : Opposite (CategoryTheory.SimplicialObject.Augmented C)
          f : Quiver.Hom X✝ Y✝
          x : SimplexCategory
          ⊢ Eq (CategoryTheory.CategoryStruct.comp f.unop.right.op ((Opposite.unop Y✝).h …
        -/
        simp_rw [← op_comp]
        /-
          case h
          C : Type u
          inst✝ : CategoryTheory.Category.{v, u} C
          X✝ Y✝ : Opposite (CategoryTheory.SimplicialObject.Augmented C)
          f : Quiver.Hom X✝ Y✝
          x : SimplexCategory
          ⊢ Eq (CategoryTheory.CategoryStruct.comp ((Opposite.unop Y✝).hom.app { unop := …
        -/
        congr 1
        /-
          case h.e_f
          C : Type u
          inst✝ : CategoryTheory.Category.{v, u} C
          X✝ Y✝ : Opposite (CategoryTheory.SimplicialObject.Augmented C)
          f : Quiver.Hom X✝ Y✝
          x : SimplexCategory
          ⊢ Eq (CategoryTheory.CategoryStruct.comp ((Opposite.unop Y✝).hom.app { unop := …
        -/
        exact (congr_app f.unop.w (op x)).symm }
        /-
          🎉 no goals
        -/


/-- A functorial version of `Cosimplicial_object.Augmented.leftOp`. -/
@[simps]
def cosimplicialToSimplicialAugmented :
    CosimplicialObject.Augmented Cᵒᵖ ⥤ (SimplicialObject.Augmented C)ᵒᵖ where
  obj X := Opposite.op X.leftOp
  map f :=
    Quiver.Hom.op <|
      { left := NatTrans.leftOp f.right
        right := f.left.unop
        w := by
          /-
            C : Type u
            inst✝ : CategoryTheory.Category.{v, u} C
            X✝ Y✝ : CategoryTheory.CosimplicialObject.Augmented (Opposite C)
            f : Quiver.Hom X✝ Y✝
            ⊢ Eq (CategoryTheory.CategoryStruct.comp ((CategoryTheory.Functor.id (Category …
          -/
          ext x
          /-
            case h
            C : Type u
            inst✝ : CategoryTheory.Category.{v, u} C
            X✝ Y✝ : CategoryTheory.CosimplicialObject.Augmented (Opposite C)
            f : Quiver.Hom X✝ Y✝
            x : Opposite SimplexCategory
            ⊢ Eq ((CategoryTheory.CategoryStruct.comp ((CategoryTheory.Functor.id (Categor …
          -/
          dsimp
          /-
            case h
            C : Type u
            inst✝ : CategoryTheory.Category.{v, u} C
            X✝ Y✝ : CategoryTheory.CosimplicialObject.Augmented (Opposite C)
            f : Quiver.Hom X✝ Y✝
            x : Opposite SimplexCategory
            ⊢ Eq (CategoryTheory.CategoryStruct.comp (f.right.app (Opposite.unop x)).unop  …
          -/
          simp_rw [← unop_comp]
          /-
            case h
            C : Type u
            inst✝ : CategoryTheory.Category.{v, u} C
            X✝ Y✝ : CategoryTheory.CosimplicialObject.Augmented (Opposite C)
            f : Quiver.Hom X✝ Y✝
            x : Opposite SimplexCategory
            ⊢ Eq (CategoryTheory.CategoryStruct.comp (X✝.hom.app (Opposite.unop x)) (f.rig …
          -/
          congr 1
          /-
            case h.e_f
            C : Type u
            inst✝ : CategoryTheory.Category.{v, u} C
            X✝ Y✝ : CategoryTheory.CosimplicialObject.Augmented (Opposite C)
            f : Quiver.Hom X✝ Y✝
            x : Opposite SimplexCategory
            ⊢ Eq (CategoryTheory.CategoryStruct.comp (X✝.hom.app (Opposite.unop x)) (f.rig …
          -/
          exact (congr_app f.w (unop x)).symm }
          /-
            🎉 no goals
          -/


/-- The contravariant categorical equivalence between augmented simplicial
objects and augmented cosimplicial objects in the opposite category. -/
@[simps! functor inverse]
def simplicialCosimplicialAugmentedEquiv :
    (SimplicialObject.Augmented C)ᵒᵖ ≌ CosimplicialObject.Augmented Cᵒᵖ where
  functor := simplicialToCosimplicialAugmented _
  inverse := cosimplicialToSimplicialAugmented _
  unitIso := NatIso.ofComponents (fun X => X.unop.rightOpLeftOpIso.op) fun f => by
      /-
        C : Type u
        inst✝ : CategoryTheory.Category.{v, u} C
        X✝ Y✝ : Opposite (CategoryTheory.SimplicialObject.Augmented C)
        f : Quiver.Hom X✝ Y✝
        ⊢ Eq (CategoryTheory.CategoryStruct.comp ((CategoryTheory.Functor.id (Opposite …
      -/
      dsimp
      /-
        C : Type u
        inst✝ : CategoryTheory.Category.{v, u} C
        X✝ Y✝ : Opposite (CategoryTheory.SimplicialObject.Augmented C)
        f : Quiver.Hom X✝ Y✝
        ⊢ Eq (CategoryTheory.CategoryStruct.comp f (Opposite.unop Y✝).rightOpLeftOpIso …
      -/
      rw [← f.op_unop]
      /-
        C : Type u
        inst✝ : CategoryTheory.Category.{v, u} C
        X✝ Y✝ : Opposite (CategoryTheory.SimplicialObject.Augmented C)
        f : Quiver.Hom X✝ Y✝
        ⊢ Eq (CategoryTheory.CategoryStruct.comp f.unop.op (Opposite.unop Y✝).rightOpL …
      -/
      simp_rw [← op_comp]
      /-
        C : Type u
        inst✝ : CategoryTheory.Category.{v, u} C
        X✝ Y✝ : Opposite (CategoryTheory.SimplicialObject.Augmented C)
        f : Quiver.Hom X✝ Y✝
        ⊢ Eq (CategoryTheory.CategoryStruct.comp (Opposite.unop Y✝).rightOpLeftOpIso.h …
      -/
      congr 1
      /-
        case e_f
        C : Type u
        inst✝ : CategoryTheory.Category.{v, u} C
        X✝ Y✝ : Opposite (CategoryTheory.SimplicialObject.Augmented C)
        f : Quiver.Hom X✝ Y✝
        ⊢ Eq (CategoryTheory.CategoryStruct.comp (Opposite.unop Y✝).rightOpLeftOpIso.h …
      -/
      aesop_cat
      /-
        🎉 no goals
      -/
               /-
                 C : Type u
                 inst✝ : CategoryTheory.Category.{v, u} C
                 ⊢ ∀ {X Y : CategoryTheory.CosimplicialObject.Augmented (Opposite C)} (f : Quiv …
               -/
  counitIso := NatIso.ofComponents fun X => X.leftOpRightOpIso
               /-
                 🎉 no goals
               -/


