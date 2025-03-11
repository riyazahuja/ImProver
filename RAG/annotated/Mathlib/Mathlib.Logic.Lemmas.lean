                                                                       /-
                                                                         a b c : Prop
                                                                         ⊢ Iff (Iff (Iff a b) c) (Iff a (Iff b c))
                                                                       -/
theorem iff_assoc {a b c : Prop} : ((a ↔ b) ↔ c) ↔ (a ↔ (b ↔ c)) := by tauto
                                                                       /-
                                                                         🎉 no goals
                                                                       -/

                                                                           /-
                                                                             a b c : Prop
                                                                             ⊢ Iff (Iff a (Iff b c)) (Iff b (Iff a c))
                                                                           -/
theorem iff_left_comm {a b c : Prop} : (a ↔ (b ↔ c)) ↔ (b ↔ (a ↔ c)) := by tauto
                                                                           /-
                                                                             🎉 no goals
                                                                           -/

                                                                            /-
                                                                              a b c : Prop
                                                                              ⊢ Iff (Iff (Iff a b) c) (Iff (Iff a c) b)
                                                                            -/
theorem iff_right_comm {a b c : Prop} : ((a ↔ b) ↔ c) ↔ ((a ↔ c) ↔ b) := by tauto
                                                                            /-
                                                                              🎉 no goals
                                                                            -/


protected alias ⟨HEq.eq, Eq.heq⟩ := heq_iff_eq


theorem dite_dite_distrib_left {a : p → α} {b : ¬p → q → α} {c : ¬p → ¬q → α} :
    (dite p a fun hp ↦ dite q (b hp) (c hp)) =
      dite q (fun hq ↦ (dite p a) fun hp ↦ b hp hq) fun hq ↦ (dite p a) fun hp ↦ c hp hq := by
  /-
    α : Sort u_1
    p q : Prop
    inst✝¹ : Decidable p
    inst✝ : Decidable q
    a : p → α
    b : Not p → q → α
    c : Not p → Not q → α
    ⊢ Eq (dite p a fun hp => dite q (b hp) (c hp)) (dite q (fun hq => dite p a fun …
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
  split_ifs <;> rfl
                /-
                  🎉 no goals
                -/


theorem dite_dite_distrib_right {a : p → q → α} {b : p → ¬q → α} {c : ¬p → α} :
    dite p (fun hp ↦ dite q (a hp) (b hp)) c =
      dite q (fun hq ↦ dite p (fun hp ↦ a hp hq) c) fun hq ↦ dite p (fun hp ↦ b hp hq) c := by
  /-
    α : Sort u_1
    p q : Prop
    inst✝¹ : Decidable p
    inst✝ : Decidable q
    a : p → q → α
    b : p → Not q → α
    c : Not p → α
    ⊢ Eq (dite p (fun hp => dite q (a hp) (b hp)) c) (dite q (fun hq => dite p (fu …
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
  split_ifs <;> rfl
                /-
                  🎉 no goals
                -/


theorem ite_dite_distrib_left {a : α} {b : q → α} {c : ¬q → α} :
    ite p a (dite q b c) = dite q (fun hq ↦ ite p a <| b hq) fun hq ↦ ite p a <| c hq :=
  dite_dite_distrib_left


theorem ite_dite_distrib_right {a : q → α} {b : ¬q → α} {c : α} :
    ite p (dite q a b) c = dite q (fun hq ↦ ite p (a hq) c) fun hq ↦ ite p (b hq) c :=
  dite_dite_distrib_right


theorem dite_ite_distrib_left {a : p → α} {b : ¬p → α} {c : ¬p → α} :
    (dite p a fun hp ↦ ite q (b hp) (c hp)) = ite q (dite p a b) (dite p a c) :=
  dite_dite_distrib_left


theorem dite_ite_distrib_right {a : p → α} {b : p → α} {c : ¬p → α} :
    dite p (fun hp ↦ ite q (a hp) (b hp)) c = ite q (dite p a c) (dite p b c) :=
  dite_dite_distrib_right


theorem ite_ite_distrib_left : ite p a (ite q b c) = ite q (ite p a b) (ite p a c) :=
  dite_dite_distrib_left


theorem ite_ite_distrib_right : ite p (ite q a b) c = ite q (ite p a c) (ite p b c) :=
  dite_dite_distrib_right


lemma Prop.forall {f : Prop → Prop} : (∀ p, f p) ↔ f True ∧ f False :=
                          /-
                            f : Prop → Prop
                            ⊢ And (f True) (f False) → ∀ (p : Prop), f p
                          -/
                                                                                    /-
                                                                                      🎉 no goals
                                                                                    -/
  ⟨fun h ↦ ⟨h _, h _⟩, by rintro ⟨h₁, h₀⟩ p; by_cases hp : p <;> simp only [hp] <;> assumption⟩
                                                                                    /-
                                                                                      🎉 no goals
                                                                                    -/


lemma Prop.exists {f : Prop → Prop} : (∃ p, f p) ↔ f True ∨ f False :=
                   /-
                     f : Prop → Prop
                     x✝ : Exists fun p => f p
                     p : Prop
                     h : f p
                     ⊢ Or (f True) (f False)
                   -/
                                                                         /-
                                                                           🎉 no goals
                                                                         -/
  ⟨fun ⟨p, h⟩ ↦ by refine (em p).imp ?_ ?_ <;> intro H <;> convert h <;> simp [H],
                                                                         /-
                                                                           🎉 no goals
                                                                         -/
       /-
         f : Prop → Prop
         ⊢ Or (f True) (f False) → Exists fun p => f p
       -/
                          /-
                            🎉 no goals
                          -/
    by rintro (h | h) <;> exact ⟨_, h⟩⟩
                          /-
                            🎉 no goals
                          -/

