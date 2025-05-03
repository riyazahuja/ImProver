/-- The sections of a multiset of multisets `s` consists of all those multisets
which can be put in bijection with `s`, so each element is a member of the corresponding multiset.
-/

def Sections (s : Multiset (Multiset α)) : Multiset (Multiset α) :=
  Multiset.recOn s {0} (fun s _ c => s.bind fun a => c.map (Multiset.cons a)) fun a₀ a₁ _ pi => by
    /-
      α : Type u_1
      s : Multiset (Multiset α)
      a₀ a₁ : Multiset α
      x✝ pi : Multiset (Multiset α)
      ⊢ HEq (a₀.bind fun a => Multiset.map (Multiset.cons a) (a₁.bind fun a => Multi …
    -/
    simp [map_bind, bind_bind a₀ a₁, cons_swap]
    /-
      🎉 no goals
    -/


@[simp]
theorem sections_zero : Sections (0 : Multiset (Multiset α)) = {0} :=
  rfl


@[simp]
theorem sections_cons (s : Multiset (Multiset α)) (m : Multiset α) :
    Sections (m ::ₘ s) = m.bind fun a => (Sections s).map (Multiset.cons a) :=
  recOn_cons m s


theorem coe_sections :
    ∀ l : List (List α),
      Sections (l.map fun l : List α => (l : Multiset α) : Multiset (Multiset α)) =
        (l.sections.map fun l : List α => (l : Multiset α) : Multiset (Multiset α))
  | [] => rfl
  | a :: l => by
    /-
      α : Type u_1
      a : List α
      l : List (List α)
      ⊢ Eq (↑(List.map (fun l => ↑l) (List.cons a l))).Sections ↑(List.map (fun l => …
    -/
    simp only [List.map_cons, List.sections]
    /-
      α : Type u_1
      a : List α
      l : List (List α)
      ⊢ Eq (↑(List.cons (↑a) (List.map (fun l => ↑l) l))).Sections ↑(List.map (fun l …
    -/
    rw [← cons_coe, sections_cons, bind_map_comm, coe_sections l]
    /-
      α : Type u_1
      a : List α
      l : List (List α)
      ⊢ Eq ((↑(List.map (fun l => ↑l) l.sections)).bind fun b => Multiset.map (fun a …
    -/
    simp [List.sections, Function.comp_def, List.flatMap]
    /-
      🎉 no goals
    -/


@[simp]
theorem sections_add (s t : Multiset (Multiset α)) :
    Sections (s + t) = (Sections s).bind fun m => (Sections t).map (m + ·) :=
                              /-
                                α : Type u_1
                                s t : Multiset (Multiset α)
                                ⊢ Eq (HAdd.hAdd 0 t).Sections ((Multiset.Sections 0).bind fun m => Multiset.ma …
                              -/
  Multiset.induction_on s (by simp) fun a s ih => by
                              /-
                                🎉 no goals
                              -/
    /-
      α : Type u_1
      s✝ t : Multiset (Multiset α)
      a : Multiset α
      s : Multiset (Multiset α)
      ih : Eq (HAdd.hAdd s t).Sections (s.Sections.bind fun m => Multiset.map (fun x …
      ⊢ Eq (HAdd.hAdd (Multiset.cons a s) t).Sections ((Multiset.cons a s).Sections. …
    -/
    simp [ih, bind_assoc, map_bind, bind_map]
    /-
      🎉 no goals
    -/


theorem mem_sections {s : Multiset (Multiset α)} :
    ∀ {a}, a ∈ Sections s ↔ s.Rel (fun s a => a ∈ s) a := by
  induction s using Multiset.induction_on with
  | empty => simp
  | cons _ _ ih => simp [ih, rel_cons_left, eq_comm]


theorem card_sections {s : Multiset (Multiset α)} : card (Sections s) = prod (s.map card) :=
                              /-
                                α : Type u_1
                                s : Multiset (Multiset α)
                                ⊢ Eq (Multiset.Sections 0).card (Multiset.map Multiset.card 0).prod
                              -/
                              /-
                                🎉 no goals
                              -/
  Multiset.induction_on s (by simp) (by simp +contextual)
                                        /-
                                          🎉 no goals
                                        -/


