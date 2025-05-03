instance {J} : IsConnected (WidePullbackShape J) := by
  /-
    J : Type u_1
    ⊢ CategoryTheory.IsConnected (CategoryTheory.Limits.WidePullbackShape J)
  -/
  apply IsConnected.of_constant_of_preserves_morphisms
  /-
    case h
    J : Type u_1
    ⊢ ∀ {α : Type u_1} (F : CategoryTheory.Limits.WidePullbackShape J → α), (∀ {j₁ …
  -/
  intros α F H
  /-
    case h
    J α : Type u_1
    F : CategoryTheory.Limits.WidePullbackShape J → α
    H : ∀ {j₁ j₂ : CategoryTheory.Limits.WidePullbackShape J}, Quiver.Hom j₁ j₂ →  …
    ⊢ ∀ (j j' : CategoryTheory.Limits.WidePullbackShape J), Eq (F j) (F j')
  -/
  suffices ∀ i, F i = F none from fun j j' ↦ (this j).trans (this j').symm
  /-
    case h
    J α : Type u_1
    F : CategoryTheory.Limits.WidePullbackShape J → α
    H : ∀ {j₁ j₂ : CategoryTheory.Limits.WidePullbackShape J}, Quiver.Hom j₁ j₂ →  …
    ⊢ ∀ (i : CategoryTheory.Limits.WidePullbackShape J), Eq (F i) (F Option.none)
  -/
  rintro ⟨⟩
  /-
    case h.none
    J α : Type u_1
    F : CategoryTheory.Limits.WidePullbackShape J → α
    H : ∀ {j₁ j₂ : CategoryTheory.Limits.WidePullbackShape J}, Quiver.Hom j₁ j₂ →  …
    ⊢ Eq (F Option.none) (F Option.none)
  -/
  exacts [rfl, H (.term _)]
  /-
    🎉 no goals
  -/


instance {J} : IsConnected (WidePushoutShape J) := by
  /-
    J : Type u_1
    ⊢ CategoryTheory.IsConnected (CategoryTheory.Limits.WidePushoutShape J)
  -/
  apply IsConnected.of_constant_of_preserves_morphisms
  /-
    case h
    J : Type u_1
    ⊢ ∀ {α : Type u_1} (F : CategoryTheory.Limits.WidePushoutShape J → α), (∀ {j₁  …
  -/
  intros α F H
  /-
    case h
    J α : Type u_1
    F : CategoryTheory.Limits.WidePushoutShape J → α
    H : ∀ {j₁ j₂ : CategoryTheory.Limits.WidePushoutShape J}, Quiver.Hom j₁ j₂ → E …
    ⊢ ∀ (j j' : CategoryTheory.Limits.WidePushoutShape J), Eq (F j) (F j')
  -/
  suffices ∀ i, F i = F none from fun j j' ↦ (this j).trans (this j').symm
  /-
    case h
    J α : Type u_1
    F : CategoryTheory.Limits.WidePushoutShape J → α
    H : ∀ {j₁ j₂ : CategoryTheory.Limits.WidePushoutShape J}, Quiver.Hom j₁ j₂ → E …
    ⊢ ∀ (i : CategoryTheory.Limits.WidePushoutShape J), Eq (F i) (F Option.none)
  -/
  rintro ⟨⟩
  /-
    case h.none
    J α : Type u_1
    F : CategoryTheory.Limits.WidePushoutShape J → α
    H : ∀ {j₁ j₂ : CategoryTheory.Limits.WidePushoutShape J}, Quiver.Hom j₁ j₂ → E …
    ⊢ Eq (F Option.none) (F Option.none)
  -/
  exacts [rfl, (H (.init _)).symm]
  /-
    🎉 no goals
  -/


