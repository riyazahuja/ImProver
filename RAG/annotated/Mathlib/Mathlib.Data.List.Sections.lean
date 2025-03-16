theorem mem_sections {L : List (List α)} {f} : f ∈ sections L ↔ Forall₂ (· ∈ ·) f L := by
  /-
    α : Type u_1
    L : List (List α)
    f : List α
    ⊢ Iff (Membership.mem L.sections f) (List.Forall₂ (fun x1 x2 => Membership.mem …
  -/
  refine ⟨fun h => ?_, fun h => ?_⟩
    /-
      case refine_1
      α : Type u_1
      L : List (List α)
      f : List α
      h : Membership.mem L.sections f
      ⊢ List.Forall₂ (fun x1 x2 => Membership.mem x2 x1) f L
    -/
  · induction L generalizing f
      /-
        case refine_1.nil
        α : Type u_1
        f : List α
        h : Membership.mem List.nil.sections f
        ⊢ List.Forall₂ (fun x1 x2 => Membership.mem x2 x1) f List.nil
      -/
    · cases mem_singleton.1 h
      /-
        case refine_1.nil.refl
        α : Type u_1
        h : Membership.mem List.nil.sections List.nil
        ⊢ List.Forall₂ (fun x1 x2 => Membership.mem x2 x1) List.nil List.nil
      -/
      exact Forall₂.nil
      /-
        🎉 no goals
      -/
    /-
      case refine_1.cons
      α : Type u_1
      head✝ : List α
      tail✝ : List (List α)
      tail_ih✝ : ∀ {f : List α}, Membership.mem tail✝.sections f → List.Forall₂ (fun …
      f : List α
      h : Membership.mem (List.cons head✝ tail✝).sections f
      ⊢ List.Forall₂ (fun x1 x2 => Membership.mem x2 x1) f (List.cons head✝ tail✝)
    -/
    simp only [sections, bind_eq_flatMap, mem_flatMap, mem_map] at h
    /-
      case refine_1.cons
      α : Type u_1
      head✝ : List α
      tail✝ : List (List α)
      tail_ih✝ : ∀ {f : List α}, Membership.mem tail✝.sections f → List.Forall₂ (fun …
      f : List α
      h : Exists fun a => And (Membership.mem tail✝.sections a) (Exists fun a_1 => A …
      ⊢ List.Forall₂ (fun x1 x2 => Membership.mem x2 x1) f (List.cons head✝ tail✝)
    -/
    rcases h with ⟨_, _, _, _, rfl⟩
    /-
      case refine_1.cons.intro.intro.intro.intro
      α : Type u_1
      head✝ : List α
      tail✝ : List (List α)
      tail_ih✝ : ∀ {f : List α}, Membership.mem tail✝.sections f → List.Forall₂ (fun …
      w✝¹ : List α
      left✝¹ : Membership.mem tail✝.sections w✝¹
      w✝ : α
      left✝ : Membership.mem head✝ w✝
      ⊢ List.Forall₂ (fun x1 x2 => Membership.mem x2 x1) (List.cons w✝ w✝¹) (List.co …
    -/
    simp only [*, forall₂_cons, true_and]
    /-
      🎉 no goals
    -/
    /-
      case refine_2
      α : Type u_1
      L : List (List α)
      f : List α
      h : List.Forall₂ (fun x1 x2 => Membership.mem x2 x1) f L
      ⊢ Membership.mem L.sections f
    -/
  · induction' h with a l f L al fL fs
      /-
        case refine_2.nil
        α : Type u_1
        L : List (List α)
        f : List α
        ⊢ Membership.mem List.nil.sections List.nil
      -/
    · simp only [sections, mem_singleton]
      /-
        🎉 no goals
      -/
    /-
      case refine_2.cons
      α : Type u_1
      L✝ : List (List α)
      f✝ : List α
      a : α
      l f : List α
      L : List (List α)
      al : Membership.mem l a
      fL : List.Forall₂ (fun x1 x2 => Membership.mem x2 x1) f L
      fs : Membership.mem L.sections f
      ⊢ Membership.mem (List.cons l L).sections (List.cons a f)
    -/
    simp only [sections, bind_eq_flatMap, mem_flatMap, mem_map]
    /-
      case refine_2.cons
      α : Type u_1
      L✝ : List (List α)
      f✝ : List α
      a : α
      l f : List α
      L : List (List α)
      al : Membership.mem l a
      fL : List.Forall₂ (fun x1 x2 => Membership.mem x2 x1) f L
      fs : Membership.mem L.sections f
      ⊢ Exists fun a_1 => And (Membership.mem L.sections a_1) (Exists fun a_2 => And …
    -/
    exact ⟨f, fs, a, al, rfl⟩
    /-
      🎉 no goals
    -/


theorem mem_sections_length {L : List (List α)} {f} (h : f ∈ sections L) : length f = length L :=
  (mem_sections.1 h).length_eq


theorem rel_sections {r : α → β → Prop} :
    (Forall₂ (Forall₂ r) ⇒ Forall₂ (Forall₂ r)) sections sections
  | _, _, Forall₂.nil => Forall₂.cons Forall₂.nil Forall₂.nil
  | _, _, Forall₂.cons h₀ h₁ =>
    rel_flatMap (rel_sections h₁) fun _ _ hl => rel_map (fun _ _ ha => Forall₂.cons ha hl) h₀


