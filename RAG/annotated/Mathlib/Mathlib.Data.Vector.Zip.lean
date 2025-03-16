/-- Apply the function `f : α → β → γ` to each corresponding pair of elements from two vectors. -/
                                                                                             /-
                                                                                               α : Type u_1
                                                                                               β : Type u_2
                                                                                               γ : Type u_3
                                                                                               n : Nat
                                                                                               f : α → β → γ
                                                                                               x : List.Vector α n
                                                                                               y : List.Vector β n
                                                                                               ⊢ Eq (List.zipWith f ↑x ↑y).length n
                                                                                             -/
def zipWith : Vector α n → Vector β n → Vector γ n := fun x y => ⟨List.zipWith f x.1 y.1, by simp⟩
                                                                                             /-
                                                                                               🎉 no goals
                                                                                             -/


@[simp]
theorem zipWith_toList (x : Vector α n) (y : Vector β n) :
    (Vector.zipWith f x y).toList = List.zipWith f x.toList y.toList :=
  rfl


@[simp]
theorem zipWith_get (x : Vector α n) (y : Vector β n) (i) :
    (Vector.zipWith f x y).get i = f (x.get i) (y.get i) := by
  /-
    α : Type u_1
    β : Type u_2
    γ : Type u_3
    n : Nat
    f : α → β → γ
    x : List.Vector α n
    y : List.Vector β n
    i : Fin n
    ⊢ Eq ((List.Vector.zipWith f x y).get i) (f (x.get i) (y.get i))
  -/
  dsimp only [Vector.zipWith, Vector.get]
  /-
    α : Type u_1
    β : Type u_2
    γ : Type u_3
    n : Nat
    f : α → β → γ
    x : List.Vector α n
    y : List.Vector β n
    i : Fin n
    ⊢ Eq ((List.zipWith f ↑x ↑y).get (Fin.cast ⋯ i)) (f ((↑x).get (Fin.cast ⋯ i))  …
  -/
  simp
  /-
    🎉 no goals
  -/


@[simp]
theorem zipWith_tail (x : Vector α n) (y : Vector β n) :
    (Vector.zipWith f x y).tail = Vector.zipWith f x.tail y.tail := by
  /-
    α : Type u_1
    β : Type u_2
    γ : Type u_3
    n : Nat
    f : α → β → γ
    x : List.Vector α n
    y : List.Vector β n
    ⊢ Eq (List.Vector.zipWith f x y).tail (List.Vector.zipWith f x.tail y.tail)
  -/
  ext
  /-
    case x
    α : Type u_1
    β : Type u_2
    γ : Type u_3
    n : Nat
    f : α → β → γ
    x : List.Vector α n
    y : List.Vector β n
    m✝ : Fin (HSub.hSub n 1)
    ⊢ Eq ((List.Vector.zipWith f x y).tail.get m✝) ((List.Vector.zipWith f x.tail  …
  -/
  simp [get_tail]
  /-
    🎉 no goals
  -/


@[to_additive]
theorem prod_mul_prod_eq_prod_zipWith [CommMonoid α] (x y : Vector α n) :
    x.toList.prod * y.toList.prod = (Vector.zipWith (· * ·) x y).toList.prod :=
  List.prod_mul_prod_eq_prod_zipWith_of_length_eq x.toList y.toList
    ((toList_length x).trans (toList_length y).symm)


