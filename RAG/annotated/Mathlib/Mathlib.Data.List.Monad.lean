instance instMonad : Monad List.{u} where
  pure := @List.singleton
  bind := @List.flatMap
  map := @List.map


@[simp] theorem pure_def (a : α) : pure a = [a] := rfl


                                                   /-
                                                     α : Type u
                                                     ⊢ ∀ {α β : Type u} (x : α) (y : List β), Eq (Functor.mapConst x y) (Functor.ma …
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
instance instLawfulMonad : LawfulMonad List.{u} := LawfulMonad.mk'
                                                   /-
                                                     🎉 no goals
                                                   -/
  (id_map := map_id)
  (pure_bind := fun _ _ => List.append_nil _)
  (bind_assoc := List.flatMap_assoc)
  (bind_pure_comp := fun _ _ => (map_eq_flatMap _ _).symm)


instance instAlternative : Alternative List.{u} where
  failure := @List.nil
  orElse l l' := List.append l (l' ())


