/-- The truncation of an `ℕ`-indexed chain complex,
deleting the object at `0` and shifting everything else down.
-/
@[simps]
def truncate [HasZeroMorphisms V] : ChainComplex V ℕ ⥤ ChainComplex V ℕ where
  obj C :=
    { X := fun i => C.X (i + 1)
      d := fun i j => C.d (i + 1) (j + 1)
                                              /-
                                                V : Type u
                                                inst✝¹ : CategoryTheory.Category.{v, u} V
                                                inst✝ : CategoryTheory.Limits.HasZeroMorphisms V
                                                C : ChainComplex V Nat
                                                i j : Nat
                                                w : Not ((ComplexShape.down Nat).Rel i j)
                                                ⊢ Not ((ComplexShape.down Nat).Rel (HAdd.hAdd i 1) (HAdd.hAdd j 1))
                                              -/
      shape := fun i j w => C.shape _ _ <| by simpa }
                                              /-
                                                🎉 no goals
                                              -/
  map f := { f := fun i => f.f (i + 1) }


/-- There is a canonical chain map from the truncation of a chain map `C` to
the "single object" chain complex consisting of the truncated object `C.X 0` in degree 0.
The components of this chain map are `C.d 1 0` in degree 0, and zero otherwise.
-/
def truncateTo [HasZeroObject V] [HasZeroMorphisms V] (C : ChainComplex V ℕ) :
    truncate.obj C ⟶ (single₀ V).obj (C.X 0) :=
                                                              /-
                                                                V : Type u
                                                                inst✝² : CategoryTheory.Category.{v, u} V
                                                                inst✝¹ : CategoryTheory.Limits.HasZeroObject V
                                                                inst✝ : CategoryTheory.Limits.HasZeroMorphisms V
                                                                C : ChainComplex V Nat
                                                                ⊢ Eq (CategoryTheory.CategoryStruct.comp ((ChainComplex.truncate.obj C).d 1 0) …
                                                              -/
  (toSingle₀Equiv (truncate.obj C) (C.X 0)).symm ⟨C.d 1 0, by aesop⟩
                                                              /-
                                                                🎉 no goals
                                                              -/

-- PROJECT when `V` is abelian (but not generally?)
-- `[∀ n, Exact (C.d (n+2) (n+1)) (C.d (n+1) n)] [Epi (C.d 1 0)]` iff `QuasiIso (C.truncate_to)`

/-- We can "augment" a chain complex by inserting an arbitrary object in degree zero
(shifting everything else up), along with a suitable differential.
-/
def augment (C : ChainComplex V ℕ) {X : V} (f : C.X 0 ⟶ X) (w : C.d 1 0 ≫ f = 0) :
    ChainComplex V ℕ where
  X | 0 => X
    | i + 1 => C.X i
  d | 1, 0 => f
    | i + 1, j + 1 => C.d i j
    | _, _ => 0
  shape
    | 1, 0, h => absurd rfl h
    | _ + 2, 0, _ => rfl
    | 0, _, _ => rfl
    | i + 1, j + 1, h => by
      /-
        V : Type u
        inst✝¹ : CategoryTheory.Category.{v, u} V
        inst✝ : CategoryTheory.Limits.HasZeroMorphisms V
        C : ChainComplex V Nat
        X : V
        f : Quiver.Hom (C.X 0) X
        w : Eq (CategoryTheory.CategoryStruct.comp (C.d 1 0) f) 0
        i j : Nat
        h : Not ((ComplexShape.down Nat).Rel (HAdd.hAdd i 1) (HAdd.hAdd j 1))
        ⊢ Eq ((fun x x_1 => ChainComplex.augment.match_2 (fun x x_2 => Quiver.Hom ((fu …
      -/
      simp only; exact C.shape i j (Nat.succ_ne_succ.1 h)
                 /-
                   🎉 no goals
                 -/
  d_comp_d'
    | _, _, 0, rfl, rfl => w
    | _, _, k + 1, rfl, rfl => C.d_comp_d _ _ _


@[simp]
theorem augment_X_zero (C : ChainComplex V ℕ) {X : V} (f : C.X 0 ⟶ X) (w : C.d 1 0 ≫ f = 0) :
    (augment C f w).X 0 = X :=
  rfl


@[simp]
theorem augment_X_succ (C : ChainComplex V ℕ) {X : V} (f : C.X 0 ⟶ X) (w : C.d 1 0 ≫ f = 0)
    (i : ℕ) : (augment C f w).X (i + 1) = C.X i :=
  rfl


@[simp]
theorem augment_d_one_zero (C : ChainComplex V ℕ) {X : V} (f : C.X 0 ⟶ X) (w : C.d 1 0 ≫ f = 0) :
    (augment C f w).d 1 0 = f :=
  rfl


@[simp]
theorem augment_d_succ_succ (C : ChainComplex V ℕ) {X : V} (f : C.X 0 ⟶ X) (w : C.d 1 0 ≫ f = 0)
    (i j : ℕ) : (augment C f w).d (i + 1) (j + 1) = C.d i j := by
  /-
    V : Type u
    inst✝¹ : CategoryTheory.Category.{v, u} V
    inst✝ : CategoryTheory.Limits.HasZeroMorphisms V
    C : ChainComplex V Nat
    X : V
    f : Quiver.Hom (C.X 0) X
    w : Eq (CategoryTheory.CategoryStruct.comp (C.d 1 0) f) 0
    i j : Nat
    ⊢ Eq ((C.augment f w).d (HAdd.hAdd i 1) (HAdd.hAdd j 1)) (C.d i j)
  -/
              /-
                🎉 no goals
              -/
  cases i <;> rfl
              /-
                🎉 no goals
              -/


/-- Truncating an augmented chain complex is isomorphic (with components the identity)
to the original complex.
-/
def truncateAugment (C : ChainComplex V ℕ) {X : V} (f : C.X 0 ⟶ X) (w : C.d 1 0 ≫ f = 0) :
    truncate.obj (augment C f w) ≅ C where
  hom := { f := fun _ => 𝟙 _ }
  inv :=
    { f := fun _ => 𝟙 _
      comm' := fun i j => by
        /-
          V : Type u
          inst✝¹ : CategoryTheory.Category.{v, u} V
          inst✝ : CategoryTheory.Limits.HasZeroMorphisms V
          C : ChainComplex V Nat
          X : V
          f : Quiver.Hom (C.X 0) X
          w : Eq (CategoryTheory.CategoryStruct.comp (C.d 1 0) f) 0
          i j : Nat
          ⊢ (ComplexShape.down Nat).Rel i j → Eq (CategoryTheory.CategoryStruct.comp ((f …
        -/
        cases j <;>
            /-
              case zero
              V : Type u
              inst✝¹ : CategoryTheory.Category.{v, u} V
              inst✝ : CategoryTheory.Limits.HasZeroMorphisms V
              C : ChainComplex V Nat
              X : V
              f : Quiver.Hom (C.X 0) X
              w : Eq (CategoryTheory.CategoryStruct.comp (C.d 1 0) f) 0
              i : Nat
              ⊢ (ComplexShape.down Nat).Rel i 0 → Eq (CategoryTheory.CategoryStruct.comp ((f …
            -/
            /-
              case zero
              V : Type u
              inst✝¹ : CategoryTheory.Category.{v, u} V
              inst✝ : CategoryTheory.Limits.HasZeroMorphisms V
              C : ChainComplex V Nat
              X : V
              f : Quiver.Hom (C.X 0) X
              w : Eq (CategoryTheory.CategoryStruct.comp (C.d 1 0) f) 0
              i : Nat
              ⊢ Eq 1 i → Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStru …
            -/
            /-
              🎉 no goals
            -/
            /-
              case succ
              V : Type u
              inst✝¹ : CategoryTheory.Category.{v, u} V
              inst✝ : CategoryTheory.Limits.HasZeroMorphisms V
              C : ChainComplex V Nat
              X : V
              f : Quiver.Hom (C.X 0) X
              w : Eq (CategoryTheory.CategoryStruct.comp (C.d 1 0) f) 0
              i n✝ : Nat
              ⊢ Eq (HAdd.hAdd (HAdd.hAdd n✝ 1) 1) i → Eq (CategoryTheory.CategoryStruct.comp …
            -/
            simp }
            /-
              🎉 no goals
            -/
  hom_inv_id := by
    /-
      V : Type u
      inst✝¹ : CategoryTheory.Category.{v, u} V
      inst✝ : CategoryTheory.Limits.HasZeroMorphisms V
      C : ChainComplex V Nat
      X : V
      f : Quiver.Hom (C.X 0) X
      w : Eq (CategoryTheory.CategoryStruct.comp (C.d 1 0) f) 0
      ⊢ Eq (CategoryTheory.CategoryStruct.comp { f := fun x => CategoryTheory.Catego …
    -/
    ext (_ | i) <;>
        /-
          case h.zero
          V : Type u
          inst✝¹ : CategoryTheory.Category.{v, u} V
          inst✝ : CategoryTheory.Limits.HasZeroMorphisms V
          C : ChainComplex V Nat
          X : V
          f : Quiver.Hom (C.X 0) X
          w : Eq (CategoryTheory.CategoryStruct.comp (C.d 1 0) f) 0
          ⊢ Eq ((CategoryTheory.CategoryStruct.comp { f := fun x => CategoryTheory.Categ …
        -/
        /-
          case h.zero
          V : Type u
          inst✝¹ : CategoryTheory.Category.{v, u} V
          inst✝ : CategoryTheory.Limits.HasZeroMorphisms V
          C : ChainComplex V Nat
          X : V
          f : Quiver.Hom (C.X 0) X
          w : Eq (CategoryTheory.CategoryStruct.comp (C.d 1 0) f) 0
          ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.id (C. …
        -/
        /-
          🎉 no goals
        -/
        /-
          case h.succ
          V : Type u
          inst✝¹ : CategoryTheory.Category.{v, u} V
          inst✝ : CategoryTheory.Limits.HasZeroMorphisms V
          C : ChainComplex V Nat
          X : V
          f : Quiver.Hom (C.X 0) X
          w : Eq (CategoryTheory.CategoryStruct.comp (C.d 1 0) f) 0
          i : Nat
          ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.id (C. …
        -/
        simp
        /-
          🎉 no goals
        -/
  inv_hom_id := by
    /-
      V : Type u
      inst✝¹ : CategoryTheory.Category.{v, u} V
      inst✝ : CategoryTheory.Limits.HasZeroMorphisms V
      C : ChainComplex V Nat
      X : V
      f : Quiver.Hom (C.X 0) X
      w : Eq (CategoryTheory.CategoryStruct.comp (C.d 1 0) f) 0
      ⊢ Eq (CategoryTheory.CategoryStruct.comp { f := fun x => CategoryTheory.Catego …
    -/
    ext (_ | i) <;>
        /-
          case h.zero
          V : Type u
          inst✝¹ : CategoryTheory.Category.{v, u} V
          inst✝ : CategoryTheory.Limits.HasZeroMorphisms V
          C : ChainComplex V Nat
          X : V
          f : Quiver.Hom (C.X 0) X
          w : Eq (CategoryTheory.CategoryStruct.comp (C.d 1 0) f) 0
          ⊢ Eq ((CategoryTheory.CategoryStruct.comp { f := fun x => CategoryTheory.Categ …
        -/
        /-
          case h.zero
          V : Type u
          inst✝¹ : CategoryTheory.Category.{v, u} V
          inst✝ : CategoryTheory.Limits.HasZeroMorphisms V
          C : ChainComplex V Nat
          X : V
          f : Quiver.Hom (C.X 0) X
          w : Eq (CategoryTheory.CategoryStruct.comp (C.d 1 0) f) 0
          ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.id (C. …
        -/
        /-
          🎉 no goals
        -/
        /-
          case h.succ
          V : Type u
          inst✝¹ : CategoryTheory.Category.{v, u} V
          inst✝ : CategoryTheory.Limits.HasZeroMorphisms V
          C : ChainComplex V Nat
          X : V
          f : Quiver.Hom (C.X 0) X
          w : Eq (CategoryTheory.CategoryStruct.comp (C.d 1 0) f) 0
          i : Nat
          ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.id (C. …
        -/
        simp
        /-
          🎉 no goals
        -/


@[simp]
theorem truncateAugment_hom_f (C : ChainComplex V ℕ) {X : V} (f : C.X 0 ⟶ X) (w : C.d 1 0 ≫ f = 0)
    (i : ℕ) : (truncateAugment C f w).hom.f i = 𝟙 (C.X i) :=
  rfl


@[simp]
theorem truncateAugment_inv_f (C : ChainComplex V ℕ) {X : V} (f : C.X 0 ⟶ X) (w : C.d 1 0 ≫ f = 0)
    (i : ℕ) : (truncateAugment C f w).inv.f i = 𝟙 ((truncate.obj (augment C f w)).X i) :=
  rfl


@[simp]
theorem chainComplex_d_succ_succ_zero (C : ChainComplex V ℕ) (i : ℕ) : C.d (i + 2) 0 = 0 := by
  /-
    V : Type u
    inst✝¹ : CategoryTheory.Category.{v, u} V
    inst✝ : CategoryTheory.Limits.HasZeroMorphisms V
    C : ChainComplex V Nat
    i : Nat
    ⊢ Eq (C.d (HAdd.hAdd i 2) 0) 0
  -/
  rw [C.shape]
  /-
    case a
    V : Type u
    inst✝¹ : CategoryTheory.Category.{v, u} V
    inst✝ : CategoryTheory.Limits.HasZeroMorphisms V
    C : ChainComplex V Nat
    i : Nat
    ⊢ Not ((ComplexShape.down Nat).Rel (HAdd.hAdd i 2) 0)
  -/
  exact i.succ_succ_ne_one.symm
  /-
    🎉 no goals
  -/


/-- Augmenting a truncated complex with the original object and morphism is isomorphic
(with components the identity) to the original complex.
-/
def augmentTruncate (C : ChainComplex V ℕ) :
    augment (truncate.obj C) (C.d 1 0) (C.d_comp_d _ _ _) ≅ C where
  hom :=
    { f := fun | 0 => 𝟙 _ | _+1 => 𝟙 _
      comm' := fun i j => by
        -- Porting note: was an rcases n with (_|_|n) but that was causing issues
        match i with
        | 0 | 1 | n+2 =>
          cases' j with j <;> dsimp [augment, truncate] <;> simp
    }
  inv :=
    { f := fun | 0 => 𝟙 _ | _+1 => 𝟙 _
      comm' := fun i j => by
        -- Porting note: was an rcases n with (_|_|n) but that was causing issues
        match i with
          | 0 | 1 | n+2 =>
          cases' j with j <;> dsimp [augment, truncate] <;> simp
    }
  hom_inv_id := by
    /-
      V : Type u
      inst✝¹ : CategoryTheory.Category.{v, u} V
      inst✝ : CategoryTheory.Limits.HasZeroMorphisms V
      C : ChainComplex V Nat
      ⊢ Eq (CategoryTheory.CategoryStruct.comp { f := fun x => ChainComplex.augment. …
    -/
    ext i
    /-
      case h
      V : Type u
      inst✝¹ : CategoryTheory.Category.{v, u} V
      inst✝ : CategoryTheory.Limits.HasZeroMorphisms V
      C : ChainComplex V Nat
      i : Nat
      ⊢ Eq ((CategoryTheory.CategoryStruct.comp { f := fun x => ChainComplex.augment …
    -/
    cases i <;>
        /-
          case h.zero
          V : Type u
          inst✝¹ : CategoryTheory.Category.{v, u} V
          inst✝ : CategoryTheory.Limits.HasZeroMorphisms V
          C : ChainComplex V Nat
          ⊢ Eq ((CategoryTheory.CategoryStruct.comp { f := fun x => ChainComplex.augment …
        -/
        /-
          case h.zero
          V : Type u
          inst✝¹ : CategoryTheory.Category.{v, u} V
          inst✝ : CategoryTheory.Limits.HasZeroMorphisms V
          C : ChainComplex V Nat
          ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.id (C. …
        -/
        /-
          🎉 no goals
        -/
        /-
          case h.succ
          V : Type u
          inst✝¹ : CategoryTheory.Category.{v, u} V
          inst✝ : CategoryTheory.Limits.HasZeroMorphisms V
          C : ChainComplex V Nat
          n✝ : Nat
          ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.id (C. …
        -/
        simp
        /-
          🎉 no goals
        -/
  inv_hom_id := by
    /-
      V : Type u
      inst✝¹ : CategoryTheory.Category.{v, u} V
      inst✝ : CategoryTheory.Limits.HasZeroMorphisms V
      C : ChainComplex V Nat
      ⊢ Eq (CategoryTheory.CategoryStruct.comp { f := fun x => ChainComplex.augment. …
    -/
    ext i
    /-
      case h
      V : Type u
      inst✝¹ : CategoryTheory.Category.{v, u} V
      inst✝ : CategoryTheory.Limits.HasZeroMorphisms V
      C : ChainComplex V Nat
      i : Nat
      ⊢ Eq ((CategoryTheory.CategoryStruct.comp { f := fun x => ChainComplex.augment …
    -/
    cases i <;>
        /-
          case h.zero
          V : Type u
          inst✝¹ : CategoryTheory.Category.{v, u} V
          inst✝ : CategoryTheory.Limits.HasZeroMorphisms V
          C : ChainComplex V Nat
          ⊢ Eq ((CategoryTheory.CategoryStruct.comp { f := fun x => ChainComplex.augment …
        -/
        /-
          case h.zero
          V : Type u
          inst✝¹ : CategoryTheory.Category.{v, u} V
          inst✝ : CategoryTheory.Limits.HasZeroMorphisms V
          C : ChainComplex V Nat
          ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.id (C. …
        -/
        /-
          🎉 no goals
        -/
        /-
          case h.succ
          V : Type u
          inst✝¹ : CategoryTheory.Category.{v, u} V
          inst✝ : CategoryTheory.Limits.HasZeroMorphisms V
          C : ChainComplex V Nat
          n✝ : Nat
          ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.id (C. …
        -/
        simp
        /-
          🎉 no goals
        -/


@[simp]
theorem augmentTruncate_hom_f_zero (C : ChainComplex V ℕ) :
    (augmentTruncate C).hom.f 0 = 𝟙 (C.X 0) :=
  rfl


@[simp]
theorem augmentTruncate_hom_f_succ (C : ChainComplex V ℕ) (i : ℕ) :
    (augmentTruncate C).hom.f (i + 1) = 𝟙 (C.X (i + 1)) :=
  rfl


@[simp]
theorem augmentTruncate_inv_f_zero (C : ChainComplex V ℕ) :
    (augmentTruncate C).inv.f 0 = 𝟙 (C.X 0) :=
  rfl


@[simp]
theorem augmentTruncate_inv_f_succ (C : ChainComplex V ℕ) (i : ℕ) :
    (augmentTruncate C).inv.f (i + 1) = 𝟙 (C.X (i + 1)) :=
  rfl


/-- A chain map from a chain complex to a single object chain complex in degree zero
can be reinterpreted as a chain complex.

This is the inverse construction of `truncateTo`.
-/
def toSingle₀AsComplex [HasZeroObject V] (C : ChainComplex V ℕ) (X : V)
    (f : C ⟶ (single₀ V).obj X) : ChainComplex V ℕ :=
  let ⟨f, w⟩ := toSingle₀Equiv C X f
  augment C f w


/-- The truncation of an `ℕ`-indexed cochain complex,
deleting the object at `0` and shifting everything else down.
-/
@[simps]
def truncate [HasZeroMorphisms V] : CochainComplex V ℕ ⥤ CochainComplex V ℕ where
  obj C :=
    { X := fun i => C.X (i + 1)
      d := fun i j => C.d (i + 1) (j + 1)
      shape := fun i j w => by
        /-
          V : Type u
          inst✝¹ : CategoryTheory.Category.{v, u} V
          inst✝ : CategoryTheory.Limits.HasZeroMorphisms V
          C : CochainComplex V Nat
          i j : Nat
          w : Not ((ComplexShape.up Nat).Rel i j)
          ⊢ Eq ((fun i j => C.d (HAdd.hAdd i 1) (HAdd.hAdd j 1)) i j) 0
        -/
        apply C.shape
        /-
          case a
          V : Type u
          inst✝¹ : CategoryTheory.Category.{v, u} V
          inst✝ : CategoryTheory.Limits.HasZeroMorphisms V
          C : CochainComplex V Nat
          i j : Nat
          w : Not ((ComplexShape.up Nat).Rel i j)
          ⊢ Not ((ComplexShape.up Nat).Rel (HAdd.hAdd i 1) (HAdd.hAdd j 1))
        -/
        simpa }
        /-
          🎉 no goals
        -/
  map f := { f := fun i => f.f (i + 1) }


/-- There is a canonical chain map from the truncation of a cochain complex `C` to
the "single object" cochain complex consisting of the truncated object `C.X 0` in degree 0.
The components of this chain map are `C.d 0 1` in degree 0, and zero otherwise.
-/
def toTruncate [HasZeroObject V] [HasZeroMorphisms V] (C : CochainComplex V ℕ) :
    (single₀ V).obj (C.X 0) ⟶ truncate.obj C :=
                                                                /-
                                                                  V : Type u
                                                                  inst✝² : CategoryTheory.Category.{v, u} V
                                                                  inst✝¹ : CategoryTheory.Limits.HasZeroObject V
                                                                  inst✝ : CategoryTheory.Limits.HasZeroMorphisms V
                                                                  C : CochainComplex V Nat
                                                                  ⊢ Eq (CategoryTheory.CategoryStruct.comp (C.d 0 1) ((CochainComplex.truncate.o …
                                                                -/
  (fromSingle₀Equiv (truncate.obj C) (C.X 0)).symm ⟨C.d 0 1, by aesop⟩
                                                                /-
                                                                  🎉 no goals
                                                                -/


/-- We can "augment" a cochain complex by inserting an arbitrary object in degree zero
(shifting everything else up), along with a suitable differential.
-/
def augment (C : CochainComplex V ℕ) {X : V} (f : X ⟶ C.X 0) (w : f ≫ C.d 0 1 = 0) :
    CochainComplex V ℕ where
  X | 0 => X
    | i + 1 => C.X i
  d | 0, 1 => f
    | i + 1, j + 1 => C.d i j
    | _, _ => 0
  shape i j s := by
    /-
      V : Type u
      inst✝¹ : CategoryTheory.Category.{v, u} V
      inst✝ : CategoryTheory.Limits.HasZeroMorphisms V
      C : CochainComplex V Nat
      X : V
      f : Quiver.Hom X (C.X 0)
      w : Eq (CategoryTheory.CategoryStruct.comp f (C.d 0 1)) 0
      i j : Nat
      s : Not ((ComplexShape.up Nat).Rel i j)
      ⊢ Eq ((fun x x_1 => CochainComplex.augment.match_1 (fun x x_2 => Quiver.Hom (( …
    -/
    simp? at s says simp only [ComplexShape.up_Rel] at s
    /-
      V : Type u
      inst✝¹ : CategoryTheory.Category.{v, u} V
      inst✝ : CategoryTheory.Limits.HasZeroMorphisms V
      C : CochainComplex V Nat
      X : V
      f : Quiver.Hom X (C.X 0)
      w : Eq (CategoryTheory.CategoryStruct.comp f (C.d 0 1)) 0
      i j : Nat
      s : Not (Eq (HAdd.hAdd i 1) j)
      ⊢ Eq ((fun x x_1 => CochainComplex.augment.match_1 (fun x x_2 => Quiver.Hom (( …
    -/
                                              /-
                                                🎉 no goals
                                              -/
                                              /-
                                                🎉 no goals
                                              -/
                                              /-
                                                🎉 no goals
                                              -/
                                              /-
                                                🎉 no goals
                                              -/
    rcases j with (_ | _ | j) <;> cases i <;> try simp
      /-
        case succ.zero.zero
        V : Type u
        inst✝¹ : CategoryTheory.Category.{v, u} V
        inst✝ : CategoryTheory.Limits.HasZeroMorphisms V
        C : CochainComplex V Nat
        X : V
        f : Quiver.Hom X (C.X 0)
        w : Eq (CategoryTheory.CategoryStruct.comp f (C.d 0 1)) 0
        s : Not (Eq (HAdd.hAdd 0 1) (HAdd.hAdd 0 1))
        ⊢ Eq f 0
      -/
    · contradiction
      /-
        🎉 no goals
      -/
      /-
        case succ.succ.succ
        V : Type u
        inst✝¹ : CategoryTheory.Category.{v, u} V
        inst✝ : CategoryTheory.Limits.HasZeroMorphisms V
        C : CochainComplex V Nat
        X : V
        f : Quiver.Hom X (C.X 0)
        w : Eq (CategoryTheory.CategoryStruct.comp f (C.d 0 1)) 0
        j n✝ : Nat
        s : Not (Eq (HAdd.hAdd (HAdd.hAdd n✝ 1) 1) (HAdd.hAdd (HAdd.hAdd j 1) 1))
        ⊢ Eq (C.d n✝ (HAdd.hAdd j 1)) 0
      -/
    · rw [C.shape]
      /-
        case succ.succ.succ.a
        V : Type u
        inst✝¹ : CategoryTheory.Category.{v, u} V
        inst✝ : CategoryTheory.Limits.HasZeroMorphisms V
        C : CochainComplex V Nat
        X : V
        f : Quiver.Hom X (C.X 0)
        w : Eq (CategoryTheory.CategoryStruct.comp f (C.d 0 1)) 0
        j n✝ : Nat
        s : Not (Eq (HAdd.hAdd (HAdd.hAdd n✝ 1) 1) (HAdd.hAdd (HAdd.hAdd j 1) 1))
        ⊢ Not ((ComplexShape.up Nat).Rel n✝ (HAdd.hAdd j 1))
      -/
      simp only [ComplexShape.up_Rel]
      /-
        case succ.succ.succ.a
        V : Type u
        inst✝¹ : CategoryTheory.Category.{v, u} V
        inst✝ : CategoryTheory.Limits.HasZeroMorphisms V
        C : CochainComplex V Nat
        X : V
        f : Quiver.Hom X (C.X 0)
        w : Eq (CategoryTheory.CategoryStruct.comp f (C.d 0 1)) 0
        j n✝ : Nat
        s : Not (Eq (HAdd.hAdd (HAdd.hAdd n✝ 1) 1) (HAdd.hAdd (HAdd.hAdd j 1) 1))
        ⊢ Not (Eq (HAdd.hAdd n✝ 1) (HAdd.hAdd j 1))
      -/
      contrapose! s
      /-
        case succ.succ.succ.a
        V : Type u
        inst✝¹ : CategoryTheory.Category.{v, u} V
        inst✝ : CategoryTheory.Limits.HasZeroMorphisms V
        C : CochainComplex V Nat
        X : V
        f : Quiver.Hom X (C.X 0)
        w : Eq (CategoryTheory.CategoryStruct.comp f (C.d 0 1)) 0
        j n✝ : Nat
        s : Eq (HAdd.hAdd n✝ 1) (HAdd.hAdd j 1)
        ⊢ Eq (HAdd.hAdd (HAdd.hAdd n✝ 1) 1) (HAdd.hAdd (HAdd.hAdd j 1) 1)
      -/
      rw [← s]
      /-
        🎉 no goals
      -/
  d_comp_d' i j k hij hjk := by
    /-
      V : Type u
      inst✝¹ : CategoryTheory.Category.{v, u} V
      inst✝ : CategoryTheory.Limits.HasZeroMorphisms V
      C : CochainComplex V Nat
      X : V
      f : Quiver.Hom X (C.X 0)
      w : Eq (CategoryTheory.CategoryStruct.comp f (C.d 0 1)) 0
      i j k : Nat
      hij : (ComplexShape.up Nat).Rel i j
      hjk : (ComplexShape.up Nat).Rel j k
      ⊢ Eq (CategoryTheory.CategoryStruct.comp ((fun x x_1 => CochainComplex.augment …
    -/
                                                                            /-
                                                                              🎉 no goals
                                                                            -/
                                                                            /-
                                                                              🎉 no goals
                                                                            -/
                                                                            /-
                                                                              🎉 no goals
                                                                            -/
                                                                            /-
                                                                              🎉 no goals
                                                                            -/
                                                                            /-
                                                                              🎉 no goals
                                                                            -/
                                                                            /-
                                                                              🎉 no goals
                                                                            -/
                                                                            /-
                                                                              🎉 no goals
                                                                            -/
                                                                            /-
                                                                              🎉 no goals
                                                                            -/
                                                                            /-
                                                                              🎉 no goals
                                                                            -/
                                                                            /-
                                                                              🎉 no goals
                                                                            -/
                                                                            /-
                                                                              🎉 no goals
                                                                            -/
                                                                            /-
                                                                              🎉 no goals
                                                                            -/
                                                                            /-
                                                                              🎉 no goals
                                                                            -/
                                                                            /-
                                                                              🎉 no goals
                                                                            -/
                                                                            /-
                                                                              🎉 no goals
                                                                            -/
                                                                            /-
                                                                              🎉 no goals
                                                                            -/
    rcases k with (_ | _ | k) <;> rcases j with (_ | _ | j) <;> cases i <;> try simp
                                                                            /-
                                                                              🎉 no goals
                                                                            -/
    /-
      case succ.succ.succ.zero.zero
      V : Type u
      inst✝¹ : CategoryTheory.Category.{v, u} V
      inst✝ : CategoryTheory.Limits.HasZeroMorphisms V
      C : CochainComplex V Nat
      X : V
      f : Quiver.Hom X (C.X 0)
      w : Eq (CategoryTheory.CategoryStruct.comp f (C.d 0 1)) 0
      k : Nat
      hjk : (ComplexShape.up Nat).Rel (HAdd.hAdd 0 1) (HAdd.hAdd (HAdd.hAdd k 1) 1)
      hij : (ComplexShape.up Nat).Rel 0 (HAdd.hAdd 0 1)
      ⊢ Eq (CategoryTheory.CategoryStruct.comp f (C.d 0 (HAdd.hAdd k 1))) 0
    -/
    cases k
      /-
        case succ.succ.succ.zero.zero.zero
        V : Type u
        inst✝¹ : CategoryTheory.Category.{v, u} V
        inst✝ : CategoryTheory.Limits.HasZeroMorphisms V
        C : CochainComplex V Nat
        X : V
        f : Quiver.Hom X (C.X 0)
        w : Eq (CategoryTheory.CategoryStruct.comp f (C.d 0 1)) 0
        hij : (ComplexShape.up Nat).Rel 0 (HAdd.hAdd 0 1)
        hjk : (ComplexShape.up Nat).Rel (HAdd.hAdd 0 1) (HAdd.hAdd (HAdd.hAdd 0 1) 1)
        ⊢ Eq (CategoryTheory.CategoryStruct.comp f (C.d 0 (HAdd.hAdd 0 1))) 0
      -/
    · exact w
      /-
        🎉 no goals
      -/
      /-
        case succ.succ.succ.zero.zero.succ
        V : Type u
        inst✝¹ : CategoryTheory.Category.{v, u} V
        inst✝ : CategoryTheory.Limits.HasZeroMorphisms V
        C : CochainComplex V Nat
        X : V
        f : Quiver.Hom X (C.X 0)
        w : Eq (CategoryTheory.CategoryStruct.comp f (C.d 0 1)) 0
        hij : (ComplexShape.up Nat).Rel 0 (HAdd.hAdd 0 1)
        n✝ : Nat
        hjk : (ComplexShape.up Nat).Rel (HAdd.hAdd 0 1) (HAdd.hAdd (HAdd.hAdd (HAdd.hA …
        ⊢ Eq (CategoryTheory.CategoryStruct.comp f (C.d 0 (HAdd.hAdd (HAdd.hAdd n✝ 1)  …
      -/
    · rw [C.shape, comp_zero]
      /-
        case succ.succ.succ.zero.zero.succ.a
        V : Type u
        inst✝¹ : CategoryTheory.Category.{v, u} V
        inst✝ : CategoryTheory.Limits.HasZeroMorphisms V
        C : CochainComplex V Nat
        X : V
        f : Quiver.Hom X (C.X 0)
        w : Eq (CategoryTheory.CategoryStruct.comp f (C.d 0 1)) 0
        hij : (ComplexShape.up Nat).Rel 0 (HAdd.hAdd 0 1)
        n✝ : Nat
        hjk : (ComplexShape.up Nat).Rel (HAdd.hAdd 0 1) (HAdd.hAdd (HAdd.hAdd (HAdd.hA …
        ⊢ Not ((ComplexShape.up Nat).Rel 0 (HAdd.hAdd (HAdd.hAdd n✝ 1) 1))
      -/
      simp only [ComplexShape.up_Rel, zero_add]
      /-
        case succ.succ.succ.zero.zero.succ.a
        V : Type u
        inst✝¹ : CategoryTheory.Category.{v, u} V
        inst✝ : CategoryTheory.Limits.HasZeroMorphisms V
        C : CochainComplex V Nat
        X : V
        f : Quiver.Hom X (C.X 0)
        w : Eq (CategoryTheory.CategoryStruct.comp f (C.d 0 1)) 0
        hij : (ComplexShape.up Nat).Rel 0 (HAdd.hAdd 0 1)
        n✝ : Nat
        hjk : (ComplexShape.up Nat).Rel (HAdd.hAdd 0 1) (HAdd.hAdd (HAdd.hAdd (HAdd.hA …
        ⊢ Not (Eq 1 (HAdd.hAdd (HAdd.hAdd n✝ 1) 1))
      -/
      exact (Nat.one_lt_succ_succ _).ne
      /-
        🎉 no goals
      -/


@[simp]
theorem augment_X_zero (C : CochainComplex V ℕ) {X : V} (f : X ⟶ C.X 0) (w : f ≫ C.d 0 1 = 0) :
    (augment C f w).X 0 = X :=
  rfl


@[simp]
theorem augment_X_succ (C : CochainComplex V ℕ) {X : V} (f : X ⟶ C.X 0) (w : f ≫ C.d 0 1 = 0)
    (i : ℕ) : (augment C f w).X (i + 1) = C.X i :=
  rfl


@[simp]
theorem augment_d_zero_one (C : CochainComplex V ℕ) {X : V} (f : X ⟶ C.X 0) (w : f ≫ C.d 0 1 = 0) :
    (augment C f w).d 0 1 = f :=
  rfl


@[simp]
theorem augment_d_succ_succ (C : CochainComplex V ℕ) {X : V} (f : X ⟶ C.X 0) (w : f ≫ C.d 0 1 = 0)
    (i j : ℕ) : (augment C f w).d (i + 1) (j + 1) = C.d i j :=
  rfl


/-- Truncating an augmented cochain complex is isomorphic (with components the identity)
to the original complex.
-/
def truncateAugment (C : CochainComplex V ℕ) {X : V} (f : X ⟶ C.X 0) (w : f ≫ C.d 0 1 = 0) :
    truncate.obj (augment C f w) ≅ C where
  hom := { f := fun _ => 𝟙 _ }
  inv :=
    { f := fun _ => 𝟙 _
      comm' := fun i j => by
        /-
          V : Type u
          inst✝¹ : CategoryTheory.Category.{v, u} V
          inst✝ : CategoryTheory.Limits.HasZeroMorphisms V
          C : CochainComplex V Nat
          X : V
          f : Quiver.Hom X (C.X 0)
          w : Eq (CategoryTheory.CategoryStruct.comp f (C.d 0 1)) 0
          i j : Nat
          ⊢ (ComplexShape.up Nat).Rel i j → Eq (CategoryTheory.CategoryStruct.comp ((fun …
        -/
        cases j <;>
            /-
              case zero
              V : Type u
              inst✝¹ : CategoryTheory.Category.{v, u} V
              inst✝ : CategoryTheory.Limits.HasZeroMorphisms V
              C : CochainComplex V Nat
              X : V
              f : Quiver.Hom X (C.X 0)
              w : Eq (CategoryTheory.CategoryStruct.comp f (C.d 0 1)) 0
              i : Nat
              ⊢ (ComplexShape.up Nat).Rel i 0 → Eq (CategoryTheory.CategoryStruct.comp ((fun …
            -/
            /-
              case zero
              V : Type u
              inst✝¹ : CategoryTheory.Category.{v, u} V
              inst✝ : CategoryTheory.Limits.HasZeroMorphisms V
              C : CochainComplex V Nat
              X : V
              f : Quiver.Hom X (C.X 0)
              w : Eq (CategoryTheory.CategoryStruct.comp f (C.d 0 1)) 0
              i : Nat
              ⊢ Eq (HAdd.hAdd i 1) 0 → Eq (CategoryTheory.CategoryStruct.comp (CategoryTheor …
            -/
            /-
              🎉 no goals
            -/
            /-
              case succ
              V : Type u
              inst✝¹ : CategoryTheory.Category.{v, u} V
              inst✝ : CategoryTheory.Limits.HasZeroMorphisms V
              C : CochainComplex V Nat
              X : V
              f : Quiver.Hom X (C.X 0)
              w : Eq (CategoryTheory.CategoryStruct.comp f (C.d 0 1)) 0
              i n✝ : Nat
              ⊢ Eq (HAdd.hAdd i 1) (HAdd.hAdd n✝ 1) → Eq (CategoryTheory.CategoryStruct.comp …
            -/
            simp }
            /-
              🎉 no goals
            -/
  hom_inv_id := by
    /-
      V : Type u
      inst✝¹ : CategoryTheory.Category.{v, u} V
      inst✝ : CategoryTheory.Limits.HasZeroMorphisms V
      C : CochainComplex V Nat
      X : V
      f : Quiver.Hom X (C.X 0)
      w : Eq (CategoryTheory.CategoryStruct.comp f (C.d 0 1)) 0
      ⊢ Eq (CategoryTheory.CategoryStruct.comp { f := fun x => CategoryTheory.Catego …
    -/
    ext i
    /-
      case h
      V : Type u
      inst✝¹ : CategoryTheory.Category.{v, u} V
      inst✝ : CategoryTheory.Limits.HasZeroMorphisms V
      C : CochainComplex V Nat
      X : V
      f : Quiver.Hom X (C.X 0)
      w : Eq (CategoryTheory.CategoryStruct.comp f (C.d 0 1)) 0
      i : Nat
      ⊢ Eq ((CategoryTheory.CategoryStruct.comp { f := fun x => CategoryTheory.Categ …
    -/
    cases i <;>
        /-
          case h.zero
          V : Type u
          inst✝¹ : CategoryTheory.Category.{v, u} V
          inst✝ : CategoryTheory.Limits.HasZeroMorphisms V
          C : CochainComplex V Nat
          X : V
          f : Quiver.Hom X (C.X 0)
          w : Eq (CategoryTheory.CategoryStruct.comp f (C.d 0 1)) 0
          ⊢ Eq ((CategoryTheory.CategoryStruct.comp { f := fun x => CategoryTheory.Categ …
        -/
        /-
          case h.zero
          V : Type u
          inst✝¹ : CategoryTheory.Category.{v, u} V
          inst✝ : CategoryTheory.Limits.HasZeroMorphisms V
          C : CochainComplex V Nat
          X : V
          f : Quiver.Hom X (C.X 0)
          w : Eq (CategoryTheory.CategoryStruct.comp f (C.d 0 1)) 0
          ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.id (C. …
        -/
        /-
          🎉 no goals
        -/
        /-
          case h.succ
          V : Type u
          inst✝¹ : CategoryTheory.Category.{v, u} V
          inst✝ : CategoryTheory.Limits.HasZeroMorphisms V
          C : CochainComplex V Nat
          X : V
          f : Quiver.Hom X (C.X 0)
          w : Eq (CategoryTheory.CategoryStruct.comp f (C.d 0 1)) 0
          n✝ : Nat
          ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.id (C. …
        -/
        simp
        /-
          🎉 no goals
        -/
  inv_hom_id := by
    /-
      V : Type u
      inst✝¹ : CategoryTheory.Category.{v, u} V
      inst✝ : CategoryTheory.Limits.HasZeroMorphisms V
      C : CochainComplex V Nat
      X : V
      f : Quiver.Hom X (C.X 0)
      w : Eq (CategoryTheory.CategoryStruct.comp f (C.d 0 1)) 0
      ⊢ Eq (CategoryTheory.CategoryStruct.comp { f := fun x => CategoryTheory.Catego …
    -/
    ext i
    /-
      case h
      V : Type u
      inst✝¹ : CategoryTheory.Category.{v, u} V
      inst✝ : CategoryTheory.Limits.HasZeroMorphisms V
      C : CochainComplex V Nat
      X : V
      f : Quiver.Hom X (C.X 0)
      w : Eq (CategoryTheory.CategoryStruct.comp f (C.d 0 1)) 0
      i : Nat
      ⊢ Eq ((CategoryTheory.CategoryStruct.comp { f := fun x => CategoryTheory.Categ …
    -/
    cases i <;>
        /-
          case h.zero
          V : Type u
          inst✝¹ : CategoryTheory.Category.{v, u} V
          inst✝ : CategoryTheory.Limits.HasZeroMorphisms V
          C : CochainComplex V Nat
          X : V
          f : Quiver.Hom X (C.X 0)
          w : Eq (CategoryTheory.CategoryStruct.comp f (C.d 0 1)) 0
          ⊢ Eq ((CategoryTheory.CategoryStruct.comp { f := fun x => CategoryTheory.Categ …
        -/
        /-
          case h.zero
          V : Type u
          inst✝¹ : CategoryTheory.Category.{v, u} V
          inst✝ : CategoryTheory.Limits.HasZeroMorphisms V
          C : CochainComplex V Nat
          X : V
          f : Quiver.Hom X (C.X 0)
          w : Eq (CategoryTheory.CategoryStruct.comp f (C.d 0 1)) 0
          ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.id (C. …
        -/
        /-
          🎉 no goals
        -/
        /-
          case h.succ
          V : Type u
          inst✝¹ : CategoryTheory.Category.{v, u} V
          inst✝ : CategoryTheory.Limits.HasZeroMorphisms V
          C : CochainComplex V Nat
          X : V
          f : Quiver.Hom X (C.X 0)
          w : Eq (CategoryTheory.CategoryStruct.comp f (C.d 0 1)) 0
          n✝ : Nat
          ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.id (C. …
        -/
        simp
        /-
          🎉 no goals
        -/


@[simp]
theorem truncateAugment_hom_f (C : CochainComplex V ℕ) {X : V} (f : X ⟶ C.X 0)
    (w : f ≫ C.d 0 1 = 0) (i : ℕ) : (truncateAugment C f w).hom.f i = 𝟙 (C.X i) :=
  rfl


@[simp]
theorem truncateAugment_inv_f (C : CochainComplex V ℕ) {X : V} (f : X ⟶ C.X 0)
    (w : f ≫ C.d 0 1 = 0) (i : ℕ) :
    (truncateAugment C f w).inv.f i = 𝟙 ((truncate.obj (augment C f w)).X i) :=
  rfl


@[simp]
theorem cochainComplex_d_succ_succ_zero (C : CochainComplex V ℕ) (i : ℕ) : C.d 0 (i + 2) = 0 := by
  /-
    V : Type u
    inst✝¹ : CategoryTheory.Category.{v, u} V
    inst✝ : CategoryTheory.Limits.HasZeroMorphisms V
    C : CochainComplex V Nat
    i : Nat
    ⊢ Eq (C.d 0 (HAdd.hAdd i 2)) 0
  -/
  rw [C.shape]
  /-
    case a
    V : Type u
    inst✝¹ : CategoryTheory.Category.{v, u} V
    inst✝ : CategoryTheory.Limits.HasZeroMorphisms V
    C : CochainComplex V Nat
    i : Nat
    ⊢ Not ((ComplexShape.up Nat).Rel 0 (HAdd.hAdd i 2))
  -/
  simp only [ComplexShape.up_Rel, zero_add]
  /-
    case a
    V : Type u
    inst✝¹ : CategoryTheory.Category.{v, u} V
    inst✝ : CategoryTheory.Limits.HasZeroMorphisms V
    C : CochainComplex V Nat
    i : Nat
    ⊢ Not (Eq 1 (HAdd.hAdd i 2))
  -/
  exact (Nat.one_lt_succ_succ _).ne
  /-
    🎉 no goals
  -/


/-- Augmenting a truncated complex with the original object and morphism is isomorphic
(with components the identity) to the original complex.
-/
def augmentTruncate (C : CochainComplex V ℕ) :
    augment (truncate.obj C) (C.d 0 1) (C.d_comp_d _ _ _) ≅ C where
  hom :=
    { f := fun | 0 => 𝟙 _ | _+1 => 𝟙 _
      comm' := fun i j => by
        /-
          V : Type u
          inst✝¹ : CategoryTheory.Category.{v, u} V
          inst✝ : CategoryTheory.Limits.HasZeroMorphisms V
          C : CochainComplex V Nat
          i j : Nat
          ⊢ (ComplexShape.up Nat).Rel i j → Eq (CategoryTheory.CategoryStruct.comp ((fun …
        -/
        rcases j with (_ | _ | j) <;> cases i <;>
            /-
              case zero.zero
              V : Type u
              inst✝¹ : CategoryTheory.Category.{v, u} V
              inst✝ : CategoryTheory.Limits.HasZeroMorphisms V
              C : CochainComplex V Nat
              ⊢ (ComplexShape.up Nat).Rel 0 0 → Eq (CategoryTheory.CategoryStruct.comp ((fun …
            -/
          · dsimp
            -- Porting note https://github.com/leanprover-community/mathlib4/issues/10959
            /-
              case zero.zero
              V : Type u
              inst✝¹ : CategoryTheory.Category.{v, u} V
              inst✝ : CategoryTheory.Limits.HasZeroMorphisms V
              C : CochainComplex V Nat
              ⊢ Eq 1 0 → Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStru …
            -/
            /-
              🎉 no goals
            -/
            /-
              🎉 no goals
            -/
            /-
              🎉 no goals
            -/
            /-
              🎉 no goals
            -/
            /-
              🎉 no goals
            -/
            /-
              case succ.succ.succ
              V : Type u
              inst✝¹ : CategoryTheory.Category.{v, u} V
              inst✝ : CategoryTheory.Limits.HasZeroMorphisms V
              C : CochainComplex V Nat
              j n✝ : Nat
              ⊢ Eq (HAdd.hAdd (HAdd.hAdd n✝ 1) 1) (HAdd.hAdd (HAdd.hAdd j 1) 1) → Eq (Catego …
            -/
            aesop }
            /-
              🎉 no goals
            -/
  inv :=
    { f := fun | 0 => 𝟙 _ | _+1 => 𝟙 _
      comm' := fun i j => by
        /-
          V : Type u
          inst✝¹ : CategoryTheory.Category.{v, u} V
          inst✝ : CategoryTheory.Limits.HasZeroMorphisms V
          C : CochainComplex V Nat
          i j : Nat
          ⊢ (ComplexShape.up Nat).Rel i j → Eq (CategoryTheory.CategoryStruct.comp ((fun …
        -/
        rcases j with (_ | _ | j) <;> cases' i with i <;>
            /-
              case zero.zero
              V : Type u
              inst✝¹ : CategoryTheory.Category.{v, u} V
              inst✝ : CategoryTheory.Limits.HasZeroMorphisms V
              C : CochainComplex V Nat
              ⊢ (ComplexShape.up Nat).Rel 0 0 → Eq (CategoryTheory.CategoryStruct.comp ((fun …
            -/
          · dsimp
            -- Porting note https://github.com/leanprover-community/mathlib4/issues/10959
            /-
              case zero.zero
              V : Type u
              inst✝¹ : CategoryTheory.Category.{v, u} V
              inst✝ : CategoryTheory.Limits.HasZeroMorphisms V
              C : CochainComplex V Nat
              ⊢ Eq 1 0 → Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStru …
            -/
            /-
              🎉 no goals
            -/
            /-
              🎉 no goals
            -/
            /-
              🎉 no goals
            -/
            /-
              🎉 no goals
            -/
            /-
              🎉 no goals
            -/
            /-
              case succ.succ.succ
              V : Type u
              inst✝¹ : CategoryTheory.Category.{v, u} V
              inst✝ : CategoryTheory.Limits.HasZeroMorphisms V
              C : CochainComplex V Nat
              j i : Nat
              ⊢ Eq (HAdd.hAdd (HAdd.hAdd i 1) 1) (HAdd.hAdd (HAdd.hAdd j 1) 1) → Eq (Categor …
            -/
            aesop }
            /-
              🎉 no goals
            -/
  hom_inv_id := by
    /-
      V : Type u
      inst✝¹ : CategoryTheory.Category.{v, u} V
      inst✝ : CategoryTheory.Limits.HasZeroMorphisms V
      C : CochainComplex V Nat
      ⊢ Eq (CategoryTheory.CategoryStruct.comp { f := fun x => ChainComplex.augment. …
    -/
    ext i
    /-
      case h
      V : Type u
      inst✝¹ : CategoryTheory.Category.{v, u} V
      inst✝ : CategoryTheory.Limits.HasZeroMorphisms V
      C : CochainComplex V Nat
      i : Nat
      ⊢ Eq ((CategoryTheory.CategoryStruct.comp { f := fun x => ChainComplex.augment …
    -/
    cases i <;>
        /-
          case h.zero
          V : Type u
          inst✝¹ : CategoryTheory.Category.{v, u} V
          inst✝ : CategoryTheory.Limits.HasZeroMorphisms V
          C : CochainComplex V Nat
          ⊢ Eq ((CategoryTheory.CategoryStruct.comp { f := fun x => ChainComplex.augment …
        -/
        /-
          case h.zero
          V : Type u
          inst✝¹ : CategoryTheory.Category.{v, u} V
          inst✝ : CategoryTheory.Limits.HasZeroMorphisms V
          C : CochainComplex V Nat
          ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.id (C. …
        -/
        /-
          🎉 no goals
        -/
        /-
          case h.succ
          V : Type u
          inst✝¹ : CategoryTheory.Category.{v, u} V
          inst✝ : CategoryTheory.Limits.HasZeroMorphisms V
          C : CochainComplex V Nat
          n✝ : Nat
          ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.id (C. …
        -/
        simp
        /-
          🎉 no goals
        -/
  inv_hom_id := by
    /-
      V : Type u
      inst✝¹ : CategoryTheory.Category.{v, u} V
      inst✝ : CategoryTheory.Limits.HasZeroMorphisms V
      C : CochainComplex V Nat
      ⊢ Eq (CategoryTheory.CategoryStruct.comp { f := fun x => ChainComplex.augment. …
    -/
    ext i
    /-
      case h
      V : Type u
      inst✝¹ : CategoryTheory.Category.{v, u} V
      inst✝ : CategoryTheory.Limits.HasZeroMorphisms V
      C : CochainComplex V Nat
      i : Nat
      ⊢ Eq ((CategoryTheory.CategoryStruct.comp { f := fun x => ChainComplex.augment …
    -/
    cases i <;>
        /-
          case h.zero
          V : Type u
          inst✝¹ : CategoryTheory.Category.{v, u} V
          inst✝ : CategoryTheory.Limits.HasZeroMorphisms V
          C : CochainComplex V Nat
          ⊢ Eq ((CategoryTheory.CategoryStruct.comp { f := fun x => ChainComplex.augment …
        -/
        /-
          case h.zero
          V : Type u
          inst✝¹ : CategoryTheory.Category.{v, u} V
          inst✝ : CategoryTheory.Limits.HasZeroMorphisms V
          C : CochainComplex V Nat
          ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.id (C. …
        -/
        /-
          🎉 no goals
        -/
        /-
          case h.succ
          V : Type u
          inst✝¹ : CategoryTheory.Category.{v, u} V
          inst✝ : CategoryTheory.Limits.HasZeroMorphisms V
          C : CochainComplex V Nat
          n✝ : Nat
          ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.id (C. …
        -/
        simp
        /-
          🎉 no goals
        -/


@[simp]
theorem augmentTruncate_hom_f_zero (C : CochainComplex V ℕ) :
    (augmentTruncate C).hom.f 0 = 𝟙 (C.X 0) :=
  rfl


@[simp]
theorem augmentTruncate_hom_f_succ (C : CochainComplex V ℕ) (i : ℕ) :
    (augmentTruncate C).hom.f (i + 1) = 𝟙 (C.X (i + 1)) :=
  rfl


@[simp]
theorem augmentTruncate_inv_f_zero (C : CochainComplex V ℕ) :
    (augmentTruncate C).inv.f 0 = 𝟙 (C.X 0) :=
  rfl


@[simp]
theorem augmentTruncate_inv_f_succ (C : CochainComplex V ℕ) (i : ℕ) :
    (augmentTruncate C).inv.f (i + 1) = 𝟙 (C.X (i + 1)) :=
  rfl


/-- A chain map from a single object cochain complex in degree zero to a cochain complex
can be reinterpreted as a cochain complex.

This is the inverse construction of `toTruncate`.
-/
def fromSingle₀AsComplex [HasZeroObject V] (C : CochainComplex V ℕ) (X : V)
    (f : (single₀ V).obj X ⟶ C) : CochainComplex V ℕ :=
  let ⟨f, w⟩ := fromSingle₀Equiv C X f
  augment C f w


