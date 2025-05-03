/-- The Kleisli category on the (type-)monad `m`. Note that the monad is not assumed to be lawful
yet. -/
@[nolint unusedArguments]
def KleisliCat (_ : Type u → Type v) :=
  Type u


/-- Construct an object of the Kleisli category from a type. -/
def KleisliCat.mk (m) (α : Type u) : KleisliCat m :=
  α


instance KleisliCat.categoryStruct {m} [Monad.{u, v} m] :
    CategoryStruct (KleisliCat m) where
  Hom α β := α → m β
  id _ x := pure x
  comp f g := f >=> g


instance KleisliCat.category {m} [Monad.{u, v} m] [LawfulMonad m] : Category (KleisliCat m) := by
  -- Porting note: was
  -- refine' { id_comp' := _, comp_id' := _, assoc' := _ } <;> intros <;> ext <;> unfold_projs <;>
  --  simp only [(· >=> ·), functor_norm]
  /-
    m : Type u → Type v
    inst✝¹ : Monad m
    inst✝ : LawfulMonad m
    ⊢ CategoryTheory.Category.{?u.264, u + 1} (CategoryTheory.KleisliCat m)
  -/
  refine { id_comp := ?_, comp_id := ?_, assoc := ?_ } <;> intros <;>
  /-
    case refine_1
    m : Type u → Type v
    inst✝¹ : Monad m
    inst✝ : LawfulMonad m
    X✝ Y✝ : CategoryTheory.KleisliCat m
    f✝ : Quiver.Hom X✝ Y✝
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.id X✝) …
  -/
  refine funext (fun x => ?_) <;>
  /-
    case refine_1
    m : Type u → Type v
    inst✝¹ : Monad m
    inst✝ : LawfulMonad m
    X✝ Y✝ : CategoryTheory.KleisliCat m
    f✝ : Quiver.Hom X✝ Y✝
    x : X✝
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.id X✝) …
  -/
  /-
    🎉 no goals
  -/
  /-
    🎉 no goals
  -/
  simp (config := { unfoldPartialApp := true }) [CategoryStruct.id, CategoryStruct.comp, (· >=> ·)]
  /-
    🎉 no goals
  -/


@[simp]
theorem KleisliCat.id_def {m} [Monad m] (α : KleisliCat m) : 𝟙 α = @pure m _ α :=
  rfl


theorem KleisliCat.comp_def {m} [Monad m] (α β γ : KleisliCat m) (xs : α ⟶ β) (ys : β ⟶ γ) (a : α) :
    (xs ≫ ys) a = xs a >>= ys :=
  rfl


instance : Inhabited (KleisliCat id) :=
  ⟨PUnit⟩


instance {α : Type u} [Inhabited α] : Inhabited (KleisliCat.mk id α) :=
  ⟨show α from default⟩


