/-- Type tag on `Unit` used to define single-object quivers. -/
-- Porting note: Removed `deriving Unique`.
@[nolint unusedArguments]
def SingleObj (_ : Type*) : Type :=
  Unit

-- Porting note: `deriving` from above has been moved to below.

instance {α : Type*} : Unique (SingleObj α) where
  default := ⟨⟩
  uniq := fun _ => rfl


instance : Quiver (SingleObj α) :=
  ⟨fun _ _ => α⟩


/-- The single object in `SingleObj α`. -/
def star : SingleObj α :=
  Unit.unit


instance : Inhabited (SingleObj α) :=
  ⟨star α⟩


lemma ext {x y : SingleObj α} : x = y := Unit.ext x y

-- See note [reducible non-instances]

/-- Equip `SingleObj α` with a reverse operation. -/
abbrev hasReverse (rev : α → α) : HasReverse (SingleObj α) := ⟨rev⟩

-- See note [reducible non-instances]

/-- Equip `SingleObj α` with an involutive reverse operation. -/
abbrev hasInvolutiveReverse (rev : α → α) (h : Function.Involutive rev) :
    HasInvolutiveReverse (SingleObj α) where
  toHasReverse := hasReverse rev
  inv' := h


/-- The type of arrows from `star α` to itself is equivalent to the original type `α`. -/
@[simps!]
def toHom : α ≃ (star α ⟶ star α) :=
  Equiv.refl _


/-- Prefunctors between two `SingleObj` quivers correspond to functions between the corresponding
arrows types.
-/
@[simps]
def toPrefunctor : (α → β) ≃ SingleObj α ⥤q SingleObj β where
  toFun f := ⟨id, f⟩
  invFun f a := f.map (toHom a)
  left_inv _ := rfl
  right_inv _ := rfl


theorem toPrefunctor_id : toPrefunctor id = 𝟭q (SingleObj α) :=
  rfl


@[simp]
theorem toPrefunctor_symm_id : toPrefunctor.symm (𝟭q (SingleObj α)) = id :=
  rfl


theorem toPrefunctor_comp (f : α → β) (g : β → γ) :
    toPrefunctor (g ∘ f) = toPrefunctor f ⋙q toPrefunctor g :=
  rfl


@[simp]
theorem toPrefunctor_symm_comp (f : SingleObj α ⥤q SingleObj β) (g : SingleObj β ⥤q SingleObj γ) :
    toPrefunctor.symm (f ⋙q g) = toPrefunctor.symm g ∘ toPrefunctor.symm f := by
  /-
    α : Type u_1
    β : Type u_2
    γ : Type u_3
    f : Prefunctor (Quiver.SingleObj α) (Quiver.SingleObj β)
    g : Prefunctor (Quiver.SingleObj β) (Quiver.SingleObj γ)
    ⊢ Eq (Quiver.SingleObj.toPrefunctor.symm (f.comp g)) (Function.comp (Quiver.Si …
  -/
  simp only [Equiv.symm_apply_eq, toPrefunctor_comp, Equiv.apply_symm_apply]
  /-
    🎉 no goals
  -/


/-- Auxiliary definition for `quiver.SingleObj.pathEquivList`.
Converts a path in the quiver `single_obj α` into a list of elements of type `a`.
-/
def pathToList : ∀ {x : SingleObj α}, Path (star α) x → List α
  | _, Path.nil => []
  | _, Path.cons p a => a :: pathToList p


/-- Auxiliary definition for `quiver.SingleObj.pathEquivList`.
Converts a list of elements of type `α` into a path in the quiver `SingleObj α`.
-/
@[simp]
def listToPath : List α → Path (star α) (star α)
  | [] => Path.nil
  | a :: l => (listToPath l).cons a


theorem listToPath_pathToList {x : SingleObj α} (p : Path (star α) x) :
    listToPath (pathToList p) = p.cast rfl ext := by
  induction p with
  | nil => rfl
  | cons _ _ ih => dsimp [pathToList] at *; rw [ih]


theorem pathToList_listToPath (l : List α) : pathToList (listToPath l) = l := by
  induction l with
  | nil => rfl
  | cons a l ih => change a :: pathToList (listToPath l) = a :: l; rw [ih]


/-- Paths in `SingleObj α` quiver correspond to lists of elements of type `α`. -/
def pathEquivList : Path (star α) (star α) ≃ List α :=
  ⟨pathToList, listToPath, fun p => listToPath_pathToList p, pathToList_listToPath⟩


@[simp]
theorem pathEquivList_nil : pathEquivList Path.nil = ([] : List α) :=
  rfl


@[simp]
theorem pathEquivList_cons (p : Path (star α) (star α)) (a : star α ⟶ star α) :
    pathEquivList (Path.cons p a) = a :: pathToList p :=
  rfl


@[simp]
theorem pathEquivList_symm_nil : pathEquivList.symm ([] : List α) = Path.nil :=
  rfl


@[simp]
theorem pathEquivList_symm_cons (l : List α) (a : α) :
    pathEquivList.symm (a :: l) = Path.cons (pathEquivList.symm l) a :=
  rfl


