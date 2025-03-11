/-- The normalized Moore complex in degree `n`, as a subobject of `X n`.
-/
def objX : ∀ n : ℕ, Subobject (X.obj (op (SimplexCategory.mk n)))
  | 0 => ⊤
  | n + 1 => Finset.univ.inf fun k : Fin (n + 1) => kernelSubobject (X.δ k.succ)


@[simp] theorem objX_zero : objX X 0 = ⊤ :=
  rfl


@[simp] theorem objX_add_one (n) :
    objX X (n + 1) = Finset.univ.inf fun k : Fin (n + 1) => kernelSubobject (X.δ k.succ) :=
  rfl


/-- The differentials in the normalized Moore complex.
-/
@[simp]
def objD : ∀ n : ℕ, (objX X (n + 1) : C) ⟶ (objX X n : C)
  | 0 => Subobject.arrow _ ≫ X.δ (0 : Fin 2) ≫ inv (⊤ : Subobject _).arrow
  | n + 1 => by
    -- The differential is `Subobject.arrow _ ≫ X.δ (0 : Fin (n+3))`,
    -- factored through the intersection of the kernels.
    /-
      C : Type u_1
      inst✝¹ : CategoryTheory.Category.{?u.7272, u_1} C
      inst✝ : CategoryTheory.Abelian C
      X : CategoryTheory.SimplicialObject C
      n : Nat
      ⊢ Quiver.Hom (CategoryTheory.Subobject.underlying.obj (AlgebraicTopology.Norma …
    -/
    refine factorThru _ (arrow _ ≫ X.δ (0 : Fin (n + 3))) ?_
    -- We now need to show that it factors!
    -- A morphism factors through an intersection of subobjects if it factors through each.
    /-
      C : Type u_1
      inst✝¹ : CategoryTheory.Category.{?u.7272, u_1} C
      inst✝ : CategoryTheory.Abelian C
      X : CategoryTheory.SimplicialObject C
      n : Nat
      ⊢ (AlgebraicTopology.NormalizedMooreComplex.objX X (HAdd.hAdd n 1)).Factors (C …
    -/
    refine (finset_inf_factors _).mpr fun i _ => ?_
    -- A morphism `f` factors through the kernel of `g` exactly if `f ≫ g = 0`.
    /-
      C : Type u_1
      inst✝¹ : CategoryTheory.Category.{?u.7272, u_1} C
      inst✝ : CategoryTheory.Abelian C
      X : CategoryTheory.SimplicialObject C
      n : Nat
      i : Fin (HAdd.hAdd n 1)
      x✝ : Membership.mem Finset.univ i
      ⊢ (CategoryTheory.Limits.kernelSubobject (X.δ i.succ)).Factors (CategoryTheory …
    -/
    apply kernelSubobject_factors
    /-
      case w
      C : Type u_1
      inst✝¹ : CategoryTheory.Category.{?u.7272, u_1} C
      inst✝ : CategoryTheory.Abelian C
      X : CategoryTheory.SimplicialObject C
      n : Nat
      i : Fin (HAdd.hAdd n 1)
      x✝ : Membership.mem Finset.univ i
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.comp ( …
    -/
    dsimp [objX]
    -- Use a simplicial identity
    /-
      case w
      C : Type u_1
      inst✝¹ : CategoryTheory.Category.{?u.7272, u_1} C
      inst✝ : CategoryTheory.Abelian C
      X : CategoryTheory.SimplicialObject C
      n : Nat
      i : Fin (HAdd.hAdd n 1)
      x✝ : Membership.mem Finset.univ i
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.comp ( …
    -/
    erw [Category.assoc, ← X.δ_comp_δ (Fin.zero_le i.succ)]
    -- We can rewrite the arrow out of the intersection of all the kernels as a composition
    -- of a morphism we don't care about with the arrow out of the kernel of `X.δ i.succ.succ`.
    rw [← factorThru_arrow _ _ (finset_inf_arrow_factors Finset.univ _ i.succ (by simp)),
      Category.assoc, kernelSubobject_arrow_comp_assoc, zero_comp, comp_zero]


theorem d_squared (n : ℕ) : objD X (n + 1) ≫ objD X n = 0 := by
  -- It's a pity we need to do a case split here;
    -- after the first erw the proofs are almost identical
  /-
    C : Type u_1
    inst✝¹ : CategoryTheory.Category.{u_2, u_1} C
    inst✝ : CategoryTheory.Abelian C
    X : CategoryTheory.SimplicialObject C
    n : Nat
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (AlgebraicTopology.NormalizedMooreCom …
  -/
  rcases n with _ | n <;> dsimp [objD]
  · erw [Subobject.factorThru_arrow_assoc, Category.assoc,
      ← X.δ_comp_δ_assoc (Fin.zero_le (0 : Fin 2)),
      ← factorThru_arrow _ _ (finset_inf_arrow_factors Finset.univ _ (0 : Fin 2) (by simp)),
      Category.assoc, kernelSubobject_arrow_comp_assoc, zero_comp, comp_zero]
  · erw [factorThru_right, factorThru_eq_zero, factorThru_arrow_assoc, Category.assoc,
      ← X.δ_comp_δ (Fin.zero_le (0 : Fin (n + 3))),
      ← factorThru_arrow _ _ (finset_inf_arrow_factors Finset.univ _ (0 : Fin (n + 3)) (by simp)),
      Category.assoc, kernelSubobject_arrow_comp_assoc, zero_comp, comp_zero]


/-- The normalized Moore complex functor, on objects.
-/
@[simps!]
def obj (X : SimplicialObject C) : ChainComplex C ℕ :=
  ChainComplex.of (fun n => (objX X n : C))
    (-- the coercion here picks a representative of the subobject
      objD X) (d_squared X)


/-- The normalized Moore complex functor, on morphisms.
-/
@[simps!]
def map (f : X ⟶ Y) : obj X ⟶ obj Y :=
  ChainComplex.ofHom _ _ _ _ _ _
    (fun n => factorThru _ (arrow _ ≫ f.app (op (SimplexCategory.mk n))) (by
      /-
        C : Type u_1
        inst✝¹ : CategoryTheory.Category.{?u.33745, u_1} C
        inst✝ : CategoryTheory.Abelian C
        X Y : CategoryTheory.SimplicialObject C
        f✝ f : Quiver.Hom X Y
        n : Nat
        ⊢ (AlgebraicTopology.NormalizedMooreComplex.objX Y n).Factors (CategoryTheory. …
      -/
      cases n <;> dsimp
        /-
          case zero
          C : Type u_1
          inst✝¹ : CategoryTheory.Category.{?u.33745, u_1} C
          inst✝ : CategoryTheory.Abelian C
          X Y : CategoryTheory.SimplicialObject C
          f✝ f : Quiver.Hom X Y
          ⊢ Top.top.Factors (CategoryTheory.CategoryStruct.comp Top.top.arrow (f.app { u …
        -/
      · apply top_factors
        /-
          🎉 no goals
        -/
        /-
          case succ
          C : Type u_1
          inst✝¹ : CategoryTheory.Category.{?u.33745, u_1} C
          inst✝ : CategoryTheory.Abelian C
          X Y : CategoryTheory.SimplicialObject C
          f✝ f : Quiver.Hom X Y
          n✝ : Nat
          ⊢ (Finset.univ.inf fun k => CategoryTheory.Limits.kernelSubobject (Y.δ k.succ) …
        -/
      · refine (finset_inf_factors _).mpr fun i _ => kernelSubobject_factors _ _ ?_
        erw [Category.assoc, ← f.naturality,
          ← factorThru_arrow _ _ (finset_inf_arrow_factors Finset.univ _ i (by simp)),
          Category.assoc, kernelSubobject_arrow_comp_assoc, zero_comp, comp_zero]))
    fun n => by
    /-
      C : Type u_1
      inst✝¹ : CategoryTheory.Category.{?u.33745, u_1} C
      inst✝ : CategoryTheory.Abelian C
      X Y : CategoryTheory.SimplicialObject C
      f✝ f : Quiver.Hom X Y
      n : Nat
      ⊢ Eq (CategoryTheory.CategoryStruct.comp ((fun n => (AlgebraicTopology.Normali …
    -/
                                       /-
                                         🎉 no goals
                                       -/
    cases n <;> dsimp [objD, objX] <;> aesop_cat
                                       /-
                                         🎉 no goals
                                       -/


/-- The (normalized) Moore complex of a simplicial object `X` in an abelian category `C`.

The `n`-th object is intersection of
the kernels of `X.δ i : X.obj n ⟶ X.obj (n-1)`, for `i = 1, ..., n`.

The differentials are induced from `X.δ 0`,
which maps each of these intersections of kernels to the next.
-/
@[simps]
def normalizedMooreComplex : SimplicialObject C ⥤ ChainComplex C ℕ where
  obj := obj
  map f := map f
  -- Porting note: Why `aesop_cat` can't do `dsimp` steps?
                 /-
                   C : Type u_1
                   inst✝¹ : CategoryTheory.Category.{?u.57661, u_1} C
                   inst✝ : CategoryTheory.Abelian C
                   X : CategoryTheory.SimplicialObject C
                   ⊢ Eq ({ obj := AlgebraicTopology.NormalizedMooreComplex.obj, map := fun {X Y}  …
                 -/
                                           /-
                                             🎉 no goals
                                           -/
  map_id X := by ext (_ | _) <;> dsimp <;> aesop_cat
                                           /-
                                             🎉 no goals
                                           -/
                     /-
                       C : Type u_1
                       inst✝¹ : CategoryTheory.Category.{?u.57661, u_1} C
                       inst✝ : CategoryTheory.Abelian C
                       X✝ Y✝ Z✝ : CategoryTheory.SimplicialObject C
                       f : Quiver.Hom X✝ Y✝
                       g : Quiver.Hom Y✝ Z✝
                       ⊢ Eq ({ obj := AlgebraicTopology.NormalizedMooreComplex.obj, map := fun {X Y}  …
                     -/
                                                                                       /-
                                                                                         🎉 no goals
                                                                                       -/
  map_comp f g := by ext (_ | _) <;> apply Subobject.eq_of_comp_arrow_eq <;> dsimp <;> aesop_cat
                                                                                       /-
                                                                                         🎉 no goals
                                                                                       -/


theorem normalizedMooreComplex_objD (X : SimplicialObject C) (n : ℕ) :
    ((normalizedMooreComplex C).obj X).d (n + 1) n = NormalizedMooreComplex.objD X n :=
-- Porting note: in mathlib, `apply ChainComplex.of_d` was enough
  ChainComplex.of_d _ _ (d_squared X) n


