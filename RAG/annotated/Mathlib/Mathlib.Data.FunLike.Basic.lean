/-- The class `DFunLike F α β` expresses that terms of type `F` have an
injective coercion to (dependent) functions from `α` to `β`.

For non-dependent functions you can also use the abbreviation `FunLike`.

This typeclass is used in the definition of the homomorphism typeclasses,
such as `ZeroHomClass`, `MulHomClass`, `MonoidHomClass`, ....
-/
@[notation_class * toFun Simps.findCoercionArgs]
class DFunLike (F : Sort*) (α : outParam (Sort*)) (β : outParam <| α → Sort*) where
  /-- The coercion from `F` to a function. -/
  coe : F → ∀ a : α, β a
  /-- The coercion to functions must be injective. -/
  coe_injective' : Function.Injective coe

-- https://github.com/leanprover/lean4/issues/2096

compile_def% DFunLike.coe


/-- The class `FunLike F α β` (`Fun`ction-`Like`) expresses that terms of type `F`
have an injective coercion to functions from `α` to `β`.
`FunLike` is the non-dependent version of `DFunLike`.
This typeclass is used in the definition of the homomorphism typeclasses,
such as `ZeroHomClass`, `MulHomClass`, `MonoidHomClass`, ....
-/
abbrev FunLike F α β := DFunLike F α fun _ => β


instance (priority := 100) hasCoeToFun : CoeFun F (fun _ ↦ ∀ a : α, β a) where
  coe := @DFunLike.coe _ _ β _ -- need to make explicit to beta reduce for non-dependent functions


theorem coe_eq_coe_fn : (DFunLike.coe (F := F)) = (fun f => ↑f) := rfl


theorem coe_injective : Function.Injective (fun f : F ↦ (f : ∀ a : α, β a)) :=
  DFunLike.coe_injective'


@[simp]
theorem coe_fn_eq {f g : F} : (f : ∀ a : α, β a) = (g : ∀ a : α, β a) ↔ f = g :=
                                                 /-
                                                   F : Sort u_1
                                                   α : Sort u_2
                                                   β : α → Sort u_3
                                                   i : DFunLike F α β
                                                   f g : F
                                                   h : Eq f g
                                                   ⊢ Eq ⇑f ⇑g
                                                 -/
  ⟨fun h ↦ DFunLike.coe_injective' h, fun h ↦ by cases h; rfl⟩
                                                          /-
                                                            🎉 no goals
                                                          -/


theorem ext' {f g : F} (h : (f : ∀ a : α, β a) = (g : ∀ a : α, β a)) : f = g :=
  DFunLike.coe_injective' h


theorem ext'_iff {f g : F} : f = g ↔ (f : ∀ a : α, β a) = (g : ∀ a : α, β a) :=
  coe_fn_eq.symm


theorem ext (f g : F) (h : ∀ x : α, f x = g x) : f = g :=
  DFunLike.coe_injective' (funext h)


theorem ext_iff {f g : F} : f = g ↔ ∀ x, f x = g x :=
  coe_fn_eq.symm.trans funext_iff


protected theorem congr_fun {f g : F} (h₁ : f = g) (x : α) : f x = g x :=
  congr_fun (congr_arg _ h₁) x


theorem ne_iff {f g : F} : f ≠ g ↔ ∃ a, f a ≠ g a :=
  ext_iff.not.trans not_forall


theorem exists_ne {f g : F} (h : f ≠ g) : ∃ x, f x ≠ g x :=
  ne_iff.mp h


/-- This is not an instance to avoid slowing down every single `Subsingleton` typeclass search. -/
lemma subsingleton_cod [∀ a, Subsingleton (β a)] : Subsingleton F :=
  ⟨fun _ _ ↦ coe_injective <| Subsingleton.elim _ _⟩


protected theorem congr {f g : F} {x y : α} (h₁ : f = g) (h₂ : x = y) : f x = g y :=
  congr (congr_arg _ h₁) h₂


protected theorem congr_arg (f : F) {x y : α} (h₂ : x = y) : f x = f y :=
  congr_arg _ h₂


