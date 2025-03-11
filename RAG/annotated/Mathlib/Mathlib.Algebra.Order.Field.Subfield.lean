/-- A subfield of a `LinearOrderedField` is a `LinearOrderedField`. -/
instance (priority := 75) toLinearOrderedField [LinearOrderedField K]
    [SubfieldClass S K] (s : S) : LinearOrderedField s :=
  Subtype.coe_injective.linearOrderedField Subtype.val rfl rfl (fun _ _ => rfl)
    (fun _ _ => rfl)
    (fun _ => rfl) (fun _ _ => rfl) (fun _ => rfl) (fun _ _ => rfl) (fun _ _ => rfl)
                                                                            /-
                                                                              K : Type u_1
                                                                              S : Type u_2
                                                                              inst✝² : SetLike S K
                                                                              inst✝¹ : LinearOrderedField K
                                                                              inst✝ : SubfieldClass S K
                                                                              s : S
                                                                              ⊢ ∀ (x : Subtype fun x => Membership.mem s x) (n : Int), Eq (↑(HPow.hPow x n)) …
                                                                            -/
    (fun _ _ => rfl) (fun _ _ => rfl) (fun _ _ => rfl) (fun _ _ => rfl) (by intros; rfl)
                                                                                    /-
                                                                                      🎉 no goals
                                                                                    -/
                                                     /-
                                                       K : Type u_1
                                                       S : Type u_2
                                                       inst✝² : SetLike S K
                                                       inst✝¹ : LinearOrderedField K
                                                       inst✝ : SubfieldClass S K
                                                       s : S
                                                       ⊢ ∀ (q : Rat), Eq ↑↑q ↑q
                                                     -/
    (fun _ => rfl) (fun _ => rfl) (fun _ => rfl) (by intros; rfl) (fun _ _ => rfl) fun _ _ => rfl
                                                             /-
                                                               🎉 no goals
                                                             -/


/-- A subfield of a `LinearOrderedField` is a `LinearOrderedField`. -/
instance toLinearOrderedField [LinearOrderedField K] (s : Subfield K) : LinearOrderedField s :=
  Subtype.coe_injective.linearOrderedField Subtype.val rfl rfl (fun _ _ => rfl) (fun _ _ => rfl)
    (fun _ => rfl) (fun _ _ => rfl) (fun _ => rfl) (fun _ _ => rfl) (fun _ _ => rfl)
                                                                            /-
                                                                              K : Type u_1
                                                                              inst✝ : LinearOrderedField K
                                                                              s : Subfield K
                                                                              ⊢ ∀ (x : Subtype fun x => Membership.mem s x) (n : Int), Eq (↑(HPow.hPow x n)) …
                                                                            -/
    (fun _ _ => rfl) (fun _ _ => rfl) (fun _ _ => rfl) (fun _ _ => rfl) (by intros; rfl)
                                                                                    /-
                                                                                      🎉 no goals
                                                                                    -/
                                                     /-
                                                       K : Type u_1
                                                       inst✝ : LinearOrderedField K
                                                       s : Subfield K
                                                       ⊢ ∀ (q : Rat), Eq ↑↑q ↑q
                                                     -/
    (fun _ => rfl) (fun _ => rfl) (fun _ => rfl) (by intros; rfl) (fun _ _ => rfl) fun _ _ => rfl
                                                             /-
                                                               🎉 no goals
                                                             -/


