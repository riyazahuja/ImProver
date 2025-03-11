/-- An `EquivFunctor` is only functorial with respect to equivalences.

To construct an `EquivFunctor`, it suffices to supply just the function `f α → f β` from
an equivalence `α ≃ β`, and then prove the functor laws. It's then a consequence that
this function is part of an equivalence, provided by `EquivFunctor.mapEquiv`.
-/
class EquivFunctor (f : Type u₀ → Type u₁) where
  /-- The action of `f` on isomorphisms. -/
  map : ∀ {α β}, α ≃ β → f α → f β
  /-- `map` of `f` preserves the identity morphism. -/
  map_refl' : ∀ α, map (Equiv.refl α) = @id (f α) := by rfl
  /-- `map` is functorial on equivalences. -/
  map_trans' : ∀ {α β γ} (k : α ≃ β) (h : β ≃ γ), map (k.trans h) = map h ∘ map k := by rfl


/-- An `EquivFunctor` in fact takes every equiv to an equiv. -/
def mapEquiv : f α ≃ f β where
  toFun := EquivFunctor.map e
  invFun := EquivFunctor.map e.symm
  left_inv x := by
    /-
      f : Type u₀ → Type u₁
      inst✝ : EquivFunctor f
      α β : Type u₀
      e : Equiv α β
      x : f α
      ⊢ Eq (EquivFunctor.map e.symm (EquivFunctor.map e x)) x
    -/
    convert (congr_fun (EquivFunctor.map_trans' e e.symm) x).symm
    /-
      case h.e'_3
      f : Type u₀ → Type u₁
      inst✝ : EquivFunctor f
      α β : Type u₀
      e : Equiv α β
      x : f α
      ⊢ Eq x (EquivFunctor.map (e.trans e.symm) x)
    -/
    simp
    /-
      🎉 no goals
    -/
  right_inv y := by
    /-
      f : Type u₀ → Type u₁
      inst✝ : EquivFunctor f
      α β : Type u₀
      e : Equiv α β
      y : f β
      ⊢ Eq (EquivFunctor.map e (EquivFunctor.map e.symm y)) y
    -/
    convert (congr_fun (EquivFunctor.map_trans' e.symm e) y).symm
    /-
      case h.e'_3
      f : Type u₀ → Type u₁
      inst✝ : EquivFunctor f
      α β : Type u₀
      e : Equiv α β
      y : f β
      ⊢ Eq y (EquivFunctor.map (e.symm.trans e) y)
    -/
    simp
    /-
      🎉 no goals
    -/


@[simp]
theorem mapEquiv_apply (x : f α) : mapEquiv f e x = EquivFunctor.map e x :=
  rfl


theorem mapEquiv_symm_apply (y : f β) : (mapEquiv f e).symm y = EquivFunctor.map e.symm y :=
  rfl


@[simp]
theorem mapEquiv_refl (α) : mapEquiv f (Equiv.refl α) = Equiv.refl (f α) := by
 /-
   f : Type u₀ → Type u₁
   inst✝ : EquivFunctor f
   α : Type u₀
   ⊢ Eq (EquivFunctor.mapEquiv f (Equiv.refl α)) (Equiv.refl (f α))
 -/
 ext; simp [mapEquiv]
      /-
        🎉 no goals
      -/


@[simp]
theorem mapEquiv_symm : (mapEquiv f e).symm = mapEquiv f e.symm :=
  Equiv.ext <| mapEquiv_symm_apply f e


/-- The composition of `mapEquiv`s is carried over the `EquivFunctor`.
For plain `Functor`s, this lemma is named `map_map` when applied
or `map_comp_map` when not applied.
-/
@[simp]
theorem mapEquiv_trans {γ : Type u₀} (ab : α ≃ β) (bc : β ≃ γ) :
    (mapEquiv f ab).trans (mapEquiv f bc) = mapEquiv f (ab.trans bc) :=
                        /-
                          f : Type u₀ → Type u₁
                          inst✝ : EquivFunctor f
                          α β γ : Type u₀
                          ab : Equiv α β
                          bc : Equiv β γ
                          x : f α
                          ⊢ Eq (((EquivFunctor.mapEquiv f ab).trans (EquivFunctor.mapEquiv f bc)) x) ((E …
                        -/
  Equiv.ext fun x => by simp [mapEquiv, map_trans']
                        /-
                          🎉 no goals
                        -/


instance (priority := 100) ofLawfulFunctor (f : Type u₀ → Type u₁) [Functor f] [LawfulFunctor f] :
    EquivFunctor f where
  map {_ _} e := Functor.map e
  map_refl' α := by
    /-
      f : Type u₀ → Type u₁
      inst✝¹ : Functor f
      inst✝ : LawfulFunctor f
      α : Type u₀
      ⊢ Eq ((fun {x x_1} e => Functor.map ⇑e) (Equiv.refl α)) id
    -/
    ext
    /-
      case h
      f : Type u₀ → Type u₁
      inst✝¹ : Functor f
      inst✝ : LawfulFunctor f
      α : Type u₀
      x✝ : f α
      ⊢ Eq ((fun {x x_1} e => Functor.map ⇑e) (Equiv.refl α) x✝) (id x✝)
    -/
    apply LawfulFunctor.id_map
    /-
      🎉 no goals
    -/
  map_trans' {α β γ} k h := by
    /-
      f : Type u₀ → Type u₁
      inst✝¹ : Functor f
      inst✝ : LawfulFunctor f
      α β γ : Type u₀
      k : Equiv α β
      h : Equiv β γ
      ⊢ Eq ((fun {x x_1} e => Functor.map ⇑e) (k.trans h)) (Function.comp ((fun {x x …
    -/
    ext x
    /-
      case h
      f : Type u₀ → Type u₁
      inst✝¹ : Functor f
      inst✝ : LawfulFunctor f
      α β γ : Type u₀
      k : Equiv α β
      h : Equiv β γ
      x : f α
      ⊢ Eq ((fun {x x_1} e => Functor.map ⇑e) (k.trans h) x) (Function.comp ((fun {x …
    -/
    apply LawfulFunctor.comp_map k h x
    /-
      🎉 no goals
    -/


theorem mapEquiv.injective (f : Type u₀ → Type u₁)
    [Applicative f] [LawfulApplicative f] {α β : Type u₀}
    (h : ∀ γ, Function.Injective (pure : γ → f γ)) :
      Function.Injective (@EquivFunctor.mapEquiv f _ α β) :=
  fun e₁ e₂ H =>
                               /-
                                 f : Type u₀ → Type u₁
                                 inst✝¹ : Applicative f
                                 inst✝ : LawfulApplicative f
                                 α β : Type u₀
                                 h : ∀ (γ : Type u₀), Function.Injective Pure.pure
                                 e₁ e₂ : Equiv α β
                                 H : Eq (EquivFunctor.mapEquiv f e₁) (EquivFunctor.mapEquiv f e₂)
                                 x : α
                                 ⊢ Eq (Pure.pure (e₁ x)) (Pure.pure (e₂ x))
                               -/
    Equiv.ext fun x => h β (by simpa [EquivFunctor.map] using Equiv.congr_fun H (pure x))
                               /-
                                 🎉 no goals
                               -/


