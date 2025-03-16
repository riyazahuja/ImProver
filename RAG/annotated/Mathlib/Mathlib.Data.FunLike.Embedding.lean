/-- The class `EmbeddingLike F α β` expresses that terms of type `F` have an
injective coercion to injective functions `α ↪ β`.
-/
class EmbeddingLike (F : Sort*) (α β : outParam (Sort*)) [FunLike F α β] : Prop where
  /-- The coercion to functions must produce injective functions. -/
  injective' : ∀ f : F, Function.Injective (DFunLike.coe f)


protected theorem injective (f : F) : Function.Injective f :=
  injective' f


@[simp]
theorem apply_eq_iff_eq (f : F) {x y : α} : f x = f y ↔ x = y :=
  (EmbeddingLike.injective f).eq_iff


@[simp]
theorem comp_injective {F : Sort*} [FunLike F β γ] [EmbeddingLike F β γ] (f : α → β) (e : F) :
    Function.Injective (e ∘ f) ↔ Function.Injective f :=
  (EmbeddingLike.injective e).of_comp_iff f


