mk_iff_of_inductive_prop List.Chain List.chain_iff


theorem Chain.iff {S : α → α → Prop} (H : ∀ a b, R a b ↔ S a b) {a : α} {l : List α} :
    Chain R a l ↔ Chain S a l :=
  ⟨Chain.imp fun a b => (H a b).1, Chain.imp fun a b => (H a b).2⟩


theorem Chain.iff_mem {a : α} {l : List α} :
    Chain R a l ↔ Chain (fun x y => x ∈ a :: l ∧ y ∈ l ∧ R x y) a l :=
  ⟨fun p => by
    induction p with
    | nil => exact nil
    | @cons _ _ _ r _ IH =>
      constructor
      · exact ⟨mem_cons_self _ _, mem_cons_self _ _, r⟩
      · exact IH.imp fun a b ⟨am, bm, h⟩ => ⟨mem_cons_of_mem _ am, mem_cons_of_mem _ bm, h⟩,
    Chain.imp fun _ _ h => h.2.2⟩


theorem chain_singleton {a b : α} : Chain R a [b] ↔ R a b := by
  /-
    α : Type u
    R : α → α → Prop
    a b : α
    ⊢ Iff (List.Chain R a (List.cons b List.nil)) (R a b)
  -/
  simp only [chain_cons, Chain.nil, and_true]
  /-
    🎉 no goals
  -/


theorem chain_split {a b : α} {l₁ l₂ : List α} :
    Chain R a (l₁ ++ b :: l₂) ↔ Chain R a (l₁ ++ [b]) ∧ Chain R b l₂ := by
  /-
    α : Type u
    R : α → α → Prop
    a b : α
    l₁ l₂ : List α
    ⊢ Iff (List.Chain R a (HAppend.hAppend l₁ (List.cons b l₂))) (And (List.Chain  …
  -/
  induction' l₁ with x l₁ IH generalizing a <;>
    /-
      case nil
      α : Type u
      R : α → α → Prop
      b : α
      l₂ : List α
      a : α
      ⊢ Iff (List.Chain R a (HAppend.hAppend List.nil (List.cons b l₂))) (And (List. …
    -/
    /-
      🎉 no goals
    -/
    simp only [*, nil_append, cons_append, Chain.nil, chain_cons, and_true, and_assoc]
    /-
      🎉 no goals
    -/


@[simp]
theorem chain_append_cons_cons {a b c : α} {l₁ l₂ : List α} :
    Chain R a (l₁ ++ b :: c :: l₂) ↔ Chain R a (l₁ ++ [b]) ∧ R b c ∧ Chain R c l₂ := by
  /-
    α : Type u
    R : α → α → Prop
    a b c : α
    l₁ l₂ : List α
    ⊢ Iff (List.Chain R a (HAppend.hAppend l₁ (List.cons b (List.cons c l₂)))) (An …
  -/
  rw [chain_split, chain_cons]
  /-
    🎉 no goals
  -/


theorem chain_iff_forall₂ :
    ∀ {a : α} {l : List α}, Chain R a l ↔ l = [] ∨ Forall₂ R (a :: dropLast l) l
                /-
                  α : Type u
                  R : α → α → Prop
                  a : α
                  ⊢ Iff (List.Chain R a List.nil) (Or (Eq List.nil List.nil) (List.Forall₂ R (Li …
                -/
  | a, [] => by simp
                /-
                  🎉 no goals
                -/
  | a, b :: l => by
    /-
      α : Type u
      R : α → α → Prop
      a b : α
      l : List α
      ⊢ Iff (List.Chain R a (List.cons b l)) (Or (Eq (List.cons b l) List.nil) (List …
    -/
    by_cases h : l = [] <;>
    /-
      case pos
      α : Type u
      R : α → α → Prop
      a b : α
      l : List α
      h : Eq l List.nil
      ⊢ Iff (List.Chain R a (List.cons b l)) (Or (Eq (List.cons b l) List.nil) (List …
    -/
    /-
      🎉 no goals
    -/
    simp [@chain_iff_forall₂ b l, dropLast, *]
    /-
      🎉 no goals
    -/


theorem chain_append_singleton_iff_forall₂ :
                                                               /-
                                                                 α : Type u
                                                                 R : α → α → Prop
                                                                 l : List α
                                                                 a b : α
                                                                 ⊢ Iff (List.Chain R a (HAppend.hAppend l (List.cons b List.nil))) (List.Forall …
                                                               -/
    Chain R a (l ++ [b]) ↔ Forall₂ R (a :: l) (l ++ [b]) := by simp [chain_iff_forall₂]
                                                               /-
                                                                 🎉 no goals
                                                               -/


theorem chain_map (f : β → α) {b : β} {l : List β} :
    Chain R (f b) (map f l) ↔ Chain (fun a b : β => R (f a) (f b)) b l := by
  /-
    α : Type u
    β : Type v
    R : α → α → Prop
    f : β → α
    b : β
    l : List β
    ⊢ Iff (List.Chain R (f b) (List.map f l)) (List.Chain (fun a b => R (f a) (f b …
  -/
                                 /-
                                   🎉 no goals
                                 -/
  induction l generalizing b <;> simp only [map, Chain.nil, chain_cons, *]
                                 /-
                                   🎉 no goals
                                 -/


theorem chain_of_chain_map {S : β → β → Prop} (f : α → β) (H : ∀ a b : α, S (f a) (f b) → R a b)
    {a : α} {l : List α} (p : Chain S (f a) (map f l)) : Chain R a l :=
  ((chain_map f).1 p).imp H


theorem chain_map_of_chain {S : β → β → Prop} (f : α → β) (H : ∀ a b : α, R a b → S (f a) (f b))
    {a : α} {l : List α} (p : Chain R a l) : Chain S (f a) (map f l) :=
  (chain_map f).2 <| p.imp H


theorem chain_pmap_of_chain {S : β → β → Prop} {p : α → Prop} {f : ∀ a, p a → β}
    (H : ∀ a b ha hb, R a b → S (f a ha) (f b hb)) {a : α} {l : List α} (hl₁ : Chain R a l)
    (ha : p a) (hl₂ : ∀ a ∈ l, p a) : Chain S (f a ha) (List.pmap f l hl₂) := by
  /-
    α : Type u
    β : Type v
    R : α → α → Prop
    S : β → β → Prop
    p : α → Prop
    f : (a : α) → p a → β
    H : ∀ (a b : α) (ha : p a) (hb : p b), R a b → S (f a ha) (f b hb)
    a : α
    l : List α
    hl₁ : List.Chain R a l
    ha : p a
    hl₂ : ∀ (a : α), Membership.mem l a → p a
    ⊢ List.Chain S (f a ha) (List.pmap f l hl₂)
  -/
  induction' l with lh lt l_ih generalizing a
    /-
      case nil
      α : Type u
      β : Type v
      R : α → α → Prop
      S : β → β → Prop
      p : α → Prop
      f : (a : α) → p a → β
      H : ∀ (a b : α) (ha : p a) (hb : p b), R a b → S (f a ha) (f b hb)
      a : α
      hl₁ : List.Chain R a List.nil
      ha : p a
      hl₂ : ∀ (a : α), Membership.mem List.nil a → p a
      ⊢ List.Chain S (f a ha) (List.pmap f List.nil hl₂)
    -/
  · simp
    /-
      🎉 no goals
    -/
    /-
      case cons
      α : Type u
      β : Type v
      R : α → α → Prop
      S : β → β → Prop
      p : α → Prop
      f : (a : α) → p a → β
      H : ∀ (a b : α) (ha : p a) (hb : p b), R a b → S (f a ha) (f b hb)
      lh : α
      lt : List α
      l_ih : ∀ {a : α}, List.Chain R a lt → ∀ (ha : p a) (hl₂ : ∀ (a : α), Membershi …
      a : α
      hl₁ : List.Chain R a (List.cons lh lt)
      ha : p a
      hl₂ : ∀ (a : α), Membership.mem (List.cons lh lt) a → p a
      ⊢ List.Chain S (f a ha) (List.pmap f (List.cons lh lt) hl₂)
    -/
  · simp [H _ _ _ _ (rel_of_chain_cons hl₁), l_ih (chain_of_chain_cons hl₁)]
    /-
      🎉 no goals
    -/


theorem chain_of_chain_pmap {S : β → β → Prop} {p : α → Prop} (f : ∀ a, p a → β) {l : List α}
    (hl₁ : ∀ a ∈ l, p a) {a : α} (ha : p a) (hl₂ : Chain S (f a ha) (List.pmap f l hl₁))
    (H : ∀ a b ha hb, S (f a ha) (f b hb) → R a b) : Chain R a l := by
  /-
    α : Type u
    β : Type v
    R : α → α → Prop
    S : β → β → Prop
    p : α → Prop
    f : (a : α) → p a → β
    l : List α
    hl₁ : ∀ (a : α), Membership.mem l a → p a
    a : α
    ha : p a
    hl₂ : List.Chain S (f a ha) (List.pmap f l hl₁)
    H : ∀ (a b : α) (ha : p a) (hb : p b), S (f a ha) (f b hb) → R a b
    ⊢ List.Chain R a l
  -/
  induction' l with lh lt l_ih generalizing a
    /-
      case nil
      α : Type u
      β : Type v
      R : α → α → Prop
      S : β → β → Prop
      p : α → Prop
      f : (a : α) → p a → β
      H : ∀ (a b : α) (ha : p a) (hb : p b), S (f a ha) (f b hb) → R a b
      hl₁ : ∀ (a : α), Membership.mem List.nil a → p a
      a : α
      ha : p a
      hl₂ : List.Chain S (f a ha) (List.pmap f List.nil hl₁)
      ⊢ List.Chain R a List.nil
    -/
  · simp
    /-
      🎉 no goals
    -/
    /-
      case cons
      α : Type u
      β : Type v
      R : α → α → Prop
      S : β → β → Prop
      p : α → Prop
      f : (a : α) → p a → β
      H : ∀ (a b : α) (ha : p a) (hb : p b), S (f a ha) (f b hb) → R a b
      lh : α
      lt : List α
      l_ih : ∀ (hl₁ : ∀ (a : α), Membership.mem lt a → p a) {a : α} (ha : p a), List …
      hl₁ : ∀ (a : α), Membership.mem (List.cons lh lt) a → p a
      a : α
      ha : p a
      hl₂ : List.Chain S (f a ha) (List.pmap f (List.cons lh lt) hl₁)
      ⊢ List.Chain R a (List.cons lh lt)
    -/
  · simp [H _ _ _ _ (rel_of_chain_cons hl₂), l_ih _ _ (chain_of_chain_cons hl₂)]
    /-
      🎉 no goals
    -/


protected theorem Chain.pairwise [IsTrans α R] :
    ∀ {a : α} {l : List α}, Chain R a l → Pairwise R (a :: l)
  | _, [], Chain.nil => pairwise_singleton _ _
  | a, _, @Chain.cons _ _ _ b l h hb =>
    hb.pairwise.cons
      (by
        /-
          α : Type u
          R : α → α → Prop
          inst✝ : IsTrans α R
          a b : α
          l : List α
          h : R a b
          hb : List.Chain R b l
          ⊢ ∀ (a' : α), Membership.mem (List.cons b l) a' → R a a'
        -/
        simp only [mem_cons, forall_eq_or_imp, h, true_and]
        /-
          α : Type u
          R : α → α → Prop
          inst✝ : IsTrans α R
          a b : α
          l : List α
          h : R a b
          hb : List.Chain R b l
          ⊢ ∀ (a_1 : α), Membership.mem l a_1 → R a a_1
        -/
        exact fun c hc => _root_.trans h (rel_of_pairwise_cons hb.pairwise hc))
        /-
          🎉 no goals
        -/


theorem chain_iff_pairwise [IsTrans α R] {a : α} {l : List α} : Chain R a l ↔ Pairwise R (a :: l) :=
  ⟨Chain.pairwise, Pairwise.chain⟩


protected theorem Chain.sublist [IsTrans α R] (hl : l₂.Chain R a) (h : l₁ <+ l₂) :
    l₁.Chain R a := by
  /-
    α : Type u
    R : α → α → Prop
    l₁ l₂ : List α
    a : α
    inst✝ : IsTrans α R
    hl : List.Chain R a l₂
    h : l₁.Sublist l₂
    ⊢ List.Chain R a l₁
  -/
  rw [chain_iff_pairwise] at hl ⊢
  /-
    α : Type u
    R : α → α → Prop
    l₁ l₂ : List α
    a : α
    inst✝ : IsTrans α R
    hl : List.Pairwise R (List.cons a l₂)
    h : l₁.Sublist l₂
    ⊢ List.Pairwise R (List.cons a l₁)
  -/
  exact hl.sublist (h.cons_cons a)
  /-
    🎉 no goals
  -/


protected theorem Chain.rel [IsTrans α R] (hl : l.Chain R a) (hb : b ∈ l) : R a b := by
  /-
    α : Type u
    R : α → α → Prop
    l : List α
    a b : α
    inst✝ : IsTrans α R
    hl : List.Chain R a l
    hb : Membership.mem l b
    ⊢ R a b
  -/
  rw [chain_iff_pairwise] at hl
  /-
    α : Type u
    R : α → α → Prop
    l : List α
    a b : α
    inst✝ : IsTrans α R
    hl : List.Pairwise R (List.cons a l)
    hb : Membership.mem l b
    ⊢ R a b
  -/
  exact rel_of_pairwise_cons hl hb
  /-
    🎉 no goals
  -/


theorem chain_iff_get {R} : ∀ {a : α} {l : List α}, Chain R a l ↔
    (∀ h : 0 < length l, R a (get l ⟨0, h⟩)) ∧
      ∀ (i : ℕ) (h : i < l.length - 1),
                        /-
                          α : Type u
                          β : Type v
                          R✝ r : α → α → Prop
                          l✝ l₁ l₂ : List α
                          a✝ b : α
                          R : α → α → Prop
                          a : α
                          l : List α
                          i : Nat
                          h : LT.lt i (HSub.hSub l.length 1)
                          ⊢ LT.lt i l.length
                        -/
                        /-
                          🎉 no goals
                        -/
        R (get l ⟨i, by omega⟩) (get l ⟨i+1, by omega⟩)
                                                /-
                                                  🎉 no goals
                                                -/
                             /-
                               α : Type u
                               R : α → α → Prop
                               a : α
                               ⊢ List.Chain R a List.nil
                             -/
                             /-
                               🎉 no goals
                             -/
                                                /-
                                                  🎉 no goals
                                                -/
  | a, [] => iff_of_true (by simp) ⟨fun h => by simp at h, fun _ h => by simp at h⟩
                                                                         /-
                                                                           🎉 no goals
                                                                         -/
  | a, b :: t => by
    /-
      α : Type u
      R : α → α → Prop
      a b : α
      t : List α
      ⊢ Iff (List.Chain R a (List.cons b t)) (And (∀ (h : LT.lt 0 (List.cons b t).le …
    -/
    rw [chain_cons, @chain_iff_get _ _ t]
    /-
      α : Type u
      R : α → α → Prop
      a b : α
      t : List α
      ⊢ Iff (And (R a b) (And (∀ (h : LT.lt 0 t.length), R b (t.get ⟨0, h⟩)) (∀ (i : …
    -/
    constructor
      /-
        case mp
        α : Type u
        R : α → α → Prop
        a b : α
        t : List α
        ⊢ And (R a b) (And (∀ (h : LT.lt 0 t.length), R b (t.get ⟨0, h⟩)) (∀ (i : Nat) …
      -/
    · rintro ⟨R, ⟨h0, h⟩⟩
      /-
        case mp.intro.intro
        α : Type u
        R✝ : α → α → Prop
        a b : α
        t : List α
        R : R✝ a b
        h0 : ∀ (h : LT.lt 0 t.length), R✝ b (t.get ⟨0, h⟩)
        h : ∀ (i : Nat) (h : LT.lt i (HSub.hSub t.length 1)), R✝ (t.get ⟨i, ⋯⟩) (t.get …
        ⊢ And (∀ (h : LT.lt 0 (List.cons b t).length), R✝ a ((List.cons b t).get ⟨0, h …
      -/
      constructor
        /-
          case mp.intro.intro.left
          α : Type u
          R✝ : α → α → Prop
          a b : α
          t : List α
          R : R✝ a b
          h0 : ∀ (h : LT.lt 0 t.length), R✝ b (t.get ⟨0, h⟩)
          h : ∀ (i : Nat) (h : LT.lt i (HSub.hSub t.length 1)), R✝ (t.get ⟨i, ⋯⟩) (t.get …
          ⊢ ∀ (h : LT.lt 0 (List.cons b t).length), R✝ a ((List.cons b t).get ⟨0, h⟩)
        -/
      · intro _
        /-
          case mp.intro.intro.left
          α : Type u
          R✝ : α → α → Prop
          a b : α
          t : List α
          R : R✝ a b
          h0 : ∀ (h : LT.lt 0 t.length), R✝ b (t.get ⟨0, h⟩)
          h : ∀ (i : Nat) (h : LT.lt i (HSub.hSub t.length 1)), R✝ (t.get ⟨i, ⋯⟩) (t.get …
          h✝ : LT.lt 0 (List.cons b t).length
          ⊢ R✝ a ((List.cons b t).get ⟨0, h✝⟩)
        -/
        exact R
        /-
          🎉 no goals
        -/
      /-
        case mp.intro.intro.right
        α : Type u
        R✝ : α → α → Prop
        a b : α
        t : List α
        R : R✝ a b
        h0 : ∀ (h : LT.lt 0 t.length), R✝ b (t.get ⟨0, h⟩)
        h : ∀ (i : Nat) (h : LT.lt i (HSub.hSub t.length 1)), R✝ (t.get ⟨i, ⋯⟩) (t.get …
        ⊢ ∀ (i : Nat) (h : LT.lt i (HSub.hSub (List.cons b t).length 1)), R✝ ((List.co …
      -/
      intro i w
      /-
        case mp.intro.intro.right
        α : Type u
        R✝ : α → α → Prop
        a b : α
        t : List α
        R : R✝ a b
        h0 : ∀ (h : LT.lt 0 t.length), R✝ b (t.get ⟨0, h⟩)
        h : ∀ (i : Nat) (h : LT.lt i (HSub.hSub t.length 1)), R✝ (t.get ⟨i, ⋯⟩) (t.get …
        i : Nat
        w : LT.lt i (HSub.hSub (List.cons b t).length 1)
        ⊢ R✝ ((List.cons b t).get ⟨i, ⋯⟩) ((List.cons b t).get ⟨HAdd.hAdd i 1, ⋯⟩)
      -/
      cases' i with i
        /-
          case mp.intro.intro.right.zero
          α : Type u
          R✝ : α → α → Prop
          a b : α
          t : List α
          R : R✝ a b
          h0 : ∀ (h : LT.lt 0 t.length), R✝ b (t.get ⟨0, h⟩)
          h : ∀ (i : Nat) (h : LT.lt i (HSub.hSub t.length 1)), R✝ (t.get ⟨i, ⋯⟩) (t.get …
          w : LT.lt 0 (HSub.hSub (List.cons b t).length 1)
          ⊢ R✝ ((List.cons b t).get ⟨0, ⋯⟩) ((List.cons b t).get ⟨HAdd.hAdd 0 1, ⋯⟩)
        -/
      · apply h0
        /-
          🎉 no goals
        -/
        /-
          case mp.intro.intro.right.succ
          α : Type u
          R✝ : α → α → Prop
          a b : α
          t : List α
          R : R✝ a b
          h0 : ∀ (h : LT.lt 0 t.length), R✝ b (t.get ⟨0, h⟩)
          h : ∀ (i : Nat) (h : LT.lt i (HSub.hSub t.length 1)), R✝ (t.get ⟨i, ⋯⟩) (t.get …
          i : Nat
          w : LT.lt (HAdd.hAdd i 1) (HSub.hSub (List.cons b t).length 1)
          ⊢ R✝ ((List.cons b t).get ⟨HAdd.hAdd i 1, ⋯⟩) ((List.cons b t).get ⟨HAdd.hAdd  …
        -/
      · exact h i (by simp only [length_cons] at w; omega)
        /-
          🎉 no goals
        -/
    /-
      case mpr
      α : Type u
      R : α → α → Prop
      a b : α
      t : List α
      ⊢ And (∀ (h : LT.lt 0 (List.cons b t).length), R a ((List.cons b t).get ⟨0, h⟩ …
    -/
    rintro ⟨h0, h⟩; constructor
      /-
        case mpr.intro.left
        α : Type u
        R : α → α → Prop
        a b : α
        t : List α
        h0 : ∀ (h : LT.lt 0 (List.cons b t).length), R a ((List.cons b t).get ⟨0, h⟩)
        h : ∀ (i : Nat) (h : LT.lt i (HSub.hSub (List.cons b t).length 1)), R ((List.c …
        ⊢ R a b
      -/
    · apply h0
      /-
        case mpr.intro.left
        α : Type u
        R : α → α → Prop
        a b : α
        t : List α
        h0 : ∀ (h : LT.lt 0 (List.cons b t).length), R a ((List.cons b t).get ⟨0, h⟩)
        h : ∀ (i : Nat) (h : LT.lt i (HSub.hSub (List.cons b t).length 1)), R ((List.c …
        ⊢ LT.lt 0 (List.cons b t).length
      -/
      simp
      /-
        🎉 no goals
      -/
    /-
      case mpr.intro.right
      α : Type u
      R : α → α → Prop
      a b : α
      t : List α
      h0 : ∀ (h : LT.lt 0 (List.cons b t).length), R a ((List.cons b t).get ⟨0, h⟩)
      h : ∀ (i : Nat) (h : LT.lt i (HSub.hSub (List.cons b t).length 1)), R ((List.c …
      ⊢ And (∀ (h : LT.lt 0 t.length), R b (t.get ⟨0, h⟩)) (∀ (i : Nat) (h : LT.lt i …
    -/
    constructor
      /-
        case mpr.intro.right.left
        α : Type u
        R : α → α → Prop
        a b : α
        t : List α
        h0 : ∀ (h : LT.lt 0 (List.cons b t).length), R a ((List.cons b t).get ⟨0, h⟩)
        h : ∀ (i : Nat) (h : LT.lt i (HSub.hSub (List.cons b t).length 1)), R ((List.c …
        ⊢ ∀ (h : LT.lt 0 t.length), R b (t.get ⟨0, h⟩)
      -/
    · apply h 0
      /-
        🎉 no goals
      -/
    /-
      case mpr.intro.right.right
      α : Type u
      R : α → α → Prop
      a b : α
      t : List α
      h0 : ∀ (h : LT.lt 0 (List.cons b t).length), R a ((List.cons b t).get ⟨0, h⟩)
      h : ∀ (i : Nat) (h : LT.lt i (HSub.hSub (List.cons b t).length 1)), R ((List.c …
      ⊢ ∀ (i : Nat) (h : LT.lt i (HSub.hSub t.length 1)), R (t.get ⟨i, ⋯⟩) (t.get ⟨H …
    -/
    intro i w
    /-
      case mpr.intro.right.right
      α : Type u
      R : α → α → Prop
      a b : α
      t : List α
      h0 : ∀ (h : LT.lt 0 (List.cons b t).length), R a ((List.cons b t).get ⟨0, h⟩)
      h : ∀ (i : Nat) (h : LT.lt i (HSub.hSub (List.cons b t).length 1)), R ((List.c …
      i : Nat
      w : LT.lt i (HSub.hSub t.length 1)
      ⊢ R (t.get ⟨i, ⋯⟩) (t.get ⟨HAdd.hAdd i 1, ⋯⟩)
    -/
    exact h (i+1) (by simp only [length_cons]; omega)
    /-
      🎉 no goals
    -/


theorem chain_replicate_of_rel (n : ℕ) {a : α} (h : r a a) : Chain r a (replicate n a) :=
  match n with
  | 0 => Chain.nil
  | n + 1 => Chain.cons h (chain_replicate_of_rel n h)


theorem chain_eq_iff_eq_replicate {a : α} {l : List α} :
    Chain (· = ·) a l ↔ l = replicate l.length a :=
  match l with
             /-
               α : Type u
               a : α
               l : List α
               ⊢ Iff (List.Chain (fun x1 x2 => Eq x1 x2) a List.nil) (Eq List.nil (List.repli …
             -/
  | [] => by simp
             /-
               🎉 no goals
             -/
  | b :: l => by
    /-
      α : Type u
      a : α
      l✝ : List α
      b : α
      l : List α
      ⊢ Iff (List.Chain (fun x1 x2 => Eq x1 x2) a (List.cons b l)) (Eq (List.cons b  …
    -/
    rw [chain_cons]
    /-
      α : Type u
      a : α
      l✝ : List α
      b : α
      l : List α
      ⊢ Iff (And (Eq a b) (List.Chain (fun x1 x2 => Eq x1 x2) b l)) (Eq (List.cons b …
    -/
    simp (config := {contextual := true}) [eq_comm, replicate_succ, chain_eq_iff_eq_replicate]
    /-
      🎉 no goals
    -/


theorem Chain'.imp {S : α → α → Prop} (H : ∀ a b, R a b → S a b) {l : List α} (p : Chain' R l) :
                     /-
                       α : Type u
                       R S : α → α → Prop
                       H : ∀ (a b : α), R a b → S a b
                       l : List α
                       p : List.Chain' R l
                       ⊢ List.Chain' S l
                     -/
    Chain' S l := by cases l <;> [trivial; exact Chain.imp H p]
                     /-
                       🎉 no goals
                     -/


theorem Chain'.iff {S : α → α → Prop} (H : ∀ a b, R a b ↔ S a b) {l : List α} :
    Chain' R l ↔ Chain' S l :=
  ⟨Chain'.imp fun a b => (H a b).1, Chain'.imp fun a b => (H a b).2⟩


theorem Chain'.iff_mem : ∀ {l : List α}, Chain' R l ↔ Chain' (fun x y => x ∈ l ∧ y ∈ l ∧ R x y) l
  | [] => Iff.rfl
  | _ :: _ =>
    ⟨fun h => (Chain.iff_mem.1 h).imp fun _ _ ⟨h₁, h₂, h₃⟩ => ⟨h₁, mem_cons.2 (Or.inr h₂), h₃⟩,
      Chain'.imp fun _ _ h => h.2.2⟩


@[simp]
theorem chain'_nil : Chain' R [] :=
  trivial


@[simp]
theorem chain'_singleton (a : α) : Chain' R [a] :=
  Chain.nil


@[simp]
theorem chain'_cons {x y l} : Chain' R (x :: y :: l) ↔ R x y ∧ Chain' R (y :: l) :=
  chain_cons


theorem chain'_isInfix : ∀ l : List α, Chain' (fun x y => [x, y] <:+: l) l
  | [] => chain'_nil
  | [_] => chain'_singleton _
  | a :: b :: l =>
    chain'_cons.2
                  /-
                    α : Type u
                    a b : α
                    l : List α
                    ⊢ Eq (HAppend.hAppend (HAppend.hAppend List.nil (List.cons a (List.cons b List …
                  -/
                  /-
                    🎉 no goals
                  -/
      ⟨⟨[], l, by simp⟩, (chain'_isInfix (b :: l)).imp fun _ _ h => h.trans ⟨[a], [], by simp⟩⟩
                                                                                         /-
                                                                                           🎉 no goals
                                                                                         -/


theorem chain'_split {a : α} :
    ∀ {l₁ l₂ : List α}, Chain' R (l₁ ++ a :: l₂) ↔ Chain' R (l₁ ++ [a]) ∧ Chain' R (a :: l₂)
  | [], _ => (and_iff_right (chain'_singleton a)).symm
  | _ :: _, _ => chain_split


@[simp]
theorem chain'_append_cons_cons {b c : α} {l₁ l₂ : List α} :
    Chain' R (l₁ ++ b :: c :: l₂) ↔ Chain' R (l₁ ++ [b]) ∧ R b c ∧ Chain' R (c :: l₂) := by
  /-
    α : Type u
    R : α → α → Prop
    b c : α
    l₁ l₂ : List α
    ⊢ Iff (List.Chain' R (HAppend.hAppend l₁ (List.cons b (List.cons c l₂)))) (And …
  -/
  rw [chain'_split, chain'_cons]
  /-
    🎉 no goals
  -/


theorem chain'_map (f : β → α) {l : List β} :
    Chain' R (map f l) ↔ Chain' (fun a b : β => R (f a) (f b)) l := by
  /-
    α : Type u
    β : Type v
    R : α → α → Prop
    f : β → α
    l : List β
    ⊢ Iff (List.Chain' R (List.map f l)) (List.Chain' (fun a b => R (f a) (f b)) l)
  -/
  cases l <;> [rfl; exact chain_map _]
  /-
    🎉 no goals
  -/


theorem chain'_of_chain'_map {S : β → β → Prop} (f : α → β) (H : ∀ a b : α, S (f a) (f b) → R a b)
    {l : List α} (p : Chain' S (map f l)) : Chain' R l :=
  ((chain'_map f).1 p).imp H


theorem chain'_map_of_chain' {S : β → β → Prop} (f : α → β) (H : ∀ a b : α, R a b → S (f a) (f b))
    {l : List α} (p : Chain' R l) : Chain' S (map f l) :=
  (chain'_map f).2 <| p.imp H


theorem Pairwise.chain' : ∀ {l : List α}, Pairwise R l → Chain' R l
  | [], _ => trivial
  | _ :: _, h => Pairwise.chain h


theorem chain'_iff_pairwise [IsTrans α R] : ∀ {l : List α}, Chain' R l ↔ Pairwise R l
  | [] => (iff_true_intro Pairwise.nil).symm
  | _ :: _ => chain_iff_pairwise


protected theorem Chain'.sublist [IsTrans α R] (hl : l₂.Chain' R) (h : l₁ <+ l₂) : l₁.Chain' R := by
  /-
    α : Type u
    R : α → α → Prop
    l₁ l₂ : List α
    inst✝ : IsTrans α R
    hl : List.Chain' R l₂
    h : l₁.Sublist l₂
    ⊢ List.Chain' R l₁
  -/
  rw [chain'_iff_pairwise] at hl ⊢
  /-
    α : Type u
    R : α → α → Prop
    l₁ l₂ : List α
    inst✝ : IsTrans α R
    hl : List.Pairwise R l₂
    h : l₁.Sublist l₂
    ⊢ List.Pairwise R l₁
  -/
  exact hl.sublist h
  /-
    🎉 no goals
  -/


theorem Chain'.cons {x y l} (h₁ : R x y) (h₂ : Chain' R (y :: l)) : Chain' R (x :: y :: l) :=
  chain'_cons.2 ⟨h₁, h₂⟩


theorem Chain'.tail : ∀ {l}, Chain' R l → Chain' R l.tail
  | [], _ => trivial
  | [_], _ => trivial
  | _ :: _ :: _, h => (chain'_cons.mp h).right


theorem Chain'.rel_head {x y l} (h : Chain' R (x :: y :: l)) : R x y :=
  rel_of_chain_cons h


theorem Chain'.rel_head? {x l} (h : Chain' R (x :: l)) ⦃y⦄ (hy : y ∈ head? l) : R x y := by
  /-
    α : Type u
    R : α → α → Prop
    x : α
    l : List α
    h : List.Chain' R (List.cons x l)
    y : α
    hy : Membership.mem l.head? y
    ⊢ R x y
  -/
  rw [← cons_head?_tail hy] at h
  /-
    α : Type u
    R : α → α → Prop
    x : α
    l : List α
    y : α
    h : List.Chain' R (List.cons x (List.cons y l.tail))
    hy : Membership.mem l.head? y
    ⊢ R x y
  -/
  exact h.rel_head
  /-
    🎉 no goals
  -/


theorem Chain'.cons' {x} : ∀ {l : List α}, Chain' R l → (∀ y ∈ l.head?, R x y) → Chain' R (x :: l)
  | [], _, _ => chain'_singleton x
  | _ :: _, hl, H => hl.cons <| H _ rfl


theorem chain'_cons' {x l} : Chain' R (x :: l) ↔ (∀ y ∈ head? l, R x y) ∧ Chain' R l :=
  ⟨fun h => ⟨h.rel_head?, h.tail⟩, fun ⟨h₁, h₂⟩ => h₂.cons' h₁⟩


theorem chain'_append :
    ∀ {l₁ l₂ : List α},
      Chain' R (l₁ ++ l₂) ↔ Chain' R l₁ ∧ Chain' R l₂ ∧ ∀ x ∈ l₁.getLast?, ∀ y ∈ l₂.head?, R x y
                /-
                  α : Type u
                  R : α → α → Prop
                  l : List α
                  ⊢ Iff (List.Chain' R (HAppend.hAppend List.nil l)) (And (List.Chain' R List.ni …
                -/
  | [], l => by simp
                /-
                  🎉 no goals
                -/
                 /-
                   α : Type u
                   R : α → α → Prop
                   a : α
                   l : List α
                   ⊢ Iff (List.Chain' R (HAppend.hAppend (List.cons a List.nil) l)) (And (List.Ch …
                 -/
  | [a], l => by simp [chain'_cons', and_comm]
                 /-
                   🎉 no goals
                 -/
  | a :: b :: l₁, l₂ => by
    /-
      α : Type u
      R : α → α → Prop
      a b : α
      l₁ l₂ : List α
      ⊢ Iff (List.Chain' R (HAppend.hAppend (List.cons a (List.cons b l₁)) l₂)) (And …
    -/
    rw [cons_append, cons_append, chain'_cons, chain'_cons, ← cons_append, chain'_append, and_assoc]
    /-
      α : Type u
      R : α → α → Prop
      a b : α
      l₁ l₂ : List α
      ⊢ Iff (And (R a b) (And (List.Chain' R (List.cons b l₁)) (And (List.Chain' R l …
    -/
    simp
    /-
      🎉 no goals
    -/


theorem Chain'.append (h₁ : Chain' R l₁) (h₂ : Chain' R l₂)
    (h : ∀ x ∈ l₁.getLast?, ∀ y ∈ l₂.head?, R x y) : Chain' R (l₁ ++ l₂) :=
  chain'_append.2 ⟨h₁, h₂, h⟩


theorem Chain'.left_of_append (h : Chain' R (l₁ ++ l₂)) : Chain' R l₁ :=
  (chain'_append.1 h).1


theorem Chain'.right_of_append (h : Chain' R (l₁ ++ l₂)) : Chain' R l₂ :=
  (chain'_append.1 h).2.1


theorem Chain'.infix (h : Chain' R l) (h' : l₁ <:+: l) : Chain' R l₁ := by
  /-
    α : Type u
    R : α → α → Prop
    l l₁ : List α
    h : List.Chain' R l
    h' : l₁.IsInfix l
    ⊢ List.Chain' R l₁
  -/
  rcases h' with ⟨l₂, l₃, rfl⟩
  /-
    case intro.intro
    α : Type u
    R : α → α → Prop
    l₁ l₂ l₃ : List α
    h : List.Chain' R (HAppend.hAppend (HAppend.hAppend l₂ l₁) l₃)
    ⊢ List.Chain' R l₁
  -/
  exact h.left_of_append.right_of_append
  /-
    🎉 no goals
  -/


theorem Chain'.suffix (h : Chain' R l) (h' : l₁ <:+ l) : Chain' R l₁ :=
  h.infix h'.isInfix


theorem Chain'.prefix (h : Chain' R l) (h' : l₁ <+: l) : Chain' R l₁ :=
  h.infix h'.isInfix


theorem Chain'.drop (h : Chain' R l) (n : ℕ) : Chain' R (drop n l) :=
  h.suffix (drop_suffix _ _)


theorem Chain'.init (h : Chain' R l) : Chain' R l.dropLast :=
  h.prefix l.dropLast_prefix


theorem Chain'.take (h : Chain' R l) (n : ℕ) : Chain' R (take n l) :=
  h.prefix (take_prefix _ _)


theorem chain'_pair {x y} : Chain' R [x, y] ↔ R x y := by
  /-
    α : Type u
    R : α → α → Prop
    x y : α
    ⊢ Iff (List.Chain' R (List.cons x (List.cons y List.nil))) (R x y)
  -/
  simp only [chain'_singleton, chain'_cons, and_true]
  /-
    🎉 no goals
  -/


theorem Chain'.imp_head {x y} (h : ∀ {z}, R x z → R y z) {l} (hl : Chain' R (x :: l)) :
    Chain' R (y :: l) :=
  hl.tail.cons' fun _ hz => h <| hl.rel_head? hz


theorem chain'_reverse : ∀ {l}, Chain' R (reverse l) ↔ Chain' (flip R) l
  | [] => Iff.rfl
              /-
                α : Type u
                R : α → α → Prop
                a : α
                ⊢ Iff (List.Chain' R (List.cons a List.nil).reverse) (List.Chain' (flip R) (Li …
              -/
  | [a] => by simp only [chain'_singleton, reverse_singleton]
              /-
                🎉 no goals
              -/
  | a :: b :: l => by
    rw [chain'_cons, reverse_cons, reverse_cons, append_assoc, cons_append, nil_append,
      chain'_split, ← reverse_cons, @chain'_reverse (b :: l), and_comm, chain'_pair, flip]


theorem chain'_iff_get {R} : ∀ {l : List α}, Chain' R l ↔
    ∀ (i : ℕ) (h : i < length l - 1),
                      /-
                        α : Type u
                        β : Type v
                        R✝ r : α → α → Prop
                        l✝ l₁ l₂ : List α
                        a b : α
                        R : α → α → Prop
                        l : List α
                        i : Nat
                        h : LT.lt i (HSub.hSub l.length 1)
                        ⊢ LT.lt i l.length
                      -/
                      /-
                        🎉 no goals
                      -/
      R (get l ⟨i, by omega⟩) (get l ⟨i + 1, by omega⟩)
                                                /-
                                                  🎉 no goals
                                                -/
                          /-
                            α : Type u
                            R : α → α → Prop
                            ⊢ List.Chain' R List.nil
                          -/
                          /-
                            🎉 no goals
                          -/
  | [] => iff_of_true (by simp) (fun _ h => by simp at h)
                                               /-
                                                 🎉 no goals
                                               -/
                           /-
                             α : Type u
                             R : α → α → Prop
                             a : α
                             ⊢ List.Chain' R (List.cons a List.nil)
                           -/
                           /-
                             🎉 no goals
                           -/
  | [a] => iff_of_true (by simp) (fun _ h => by simp at h)
                                                /-
                                                  🎉 no goals
                                                -/
  | a :: b :: t => by
    /-
      α : Type u
      R : α → α → Prop
      a b : α
      t : List α
      ⊢ Iff (List.Chain' R (List.cons a (List.cons b t))) (∀ (i : Nat) (h : LT.lt i  …
    -/
    rw [← and_forall_add_one, chain'_cons, chain'_iff_get]
    /-
      α : Type u
      R : α → α → Prop
      a b : α
      t : List α
      ⊢ Iff (And (R a b) (∀ (i : Nat) (h : LT.lt i (HSub.hSub (List.cons b t).length …
    -/
    simp
    /-
      🎉 no goals
    -/


/-- If `l₁ l₂` and `l₃` are lists and `l₁ ++ l₂` and `l₂ ++ l₃` both satisfy
  `Chain' R`, then so does `l₁ ++ l₂ ++ l₃` provided `l₂ ≠ []` -/
theorem Chain'.append_overlap {l₁ l₂ l₃ : List α} (h₁ : Chain' R (l₁ ++ l₂))
    (h₂ : Chain' R (l₂ ++ l₃)) (hn : l₂ ≠ []) : Chain' R (l₁ ++ l₂ ++ l₃) :=
  h₁.append h₂.right_of_append <| by
    /-
      α : Type u
      R : α → α → Prop
      l₁ l₂ l₃ : List α
      h₁ : List.Chain' R (HAppend.hAppend l₁ l₂)
      h₂ : List.Chain' R (HAppend.hAppend l₂ l₃)
      hn : Ne l₂ List.nil
      ⊢ ∀ (x : α), Membership.mem (HAppend.hAppend l₁ l₂).getLast? x → ∀ (y : α), Me …
    -/
    simpa only [getLast?_append_of_ne_nil _ hn] using (chain'_append.1 h₂).2.2
    /-
      🎉 no goals
    -/


lemma chain'_flatten : ∀ {L : List (List α)}, [] ∉ L →
    (Chain' R L.flatten ↔ (∀ l ∈ L, Chain' R l) ∧
    L.Chain' (fun l₁ l₂ => ∀ᵉ (x ∈ l₁.getLast?) (y ∈ l₂.head?), R x y))
              /-
                α : Type u
                R : α → α → Prop
                x✝ : Not (Membership.mem List.nil List.nil)
                ⊢ Iff (List.Chain' R List.nil.flatten) (And (∀ (l : List α), Membership.mem Li …
              -/
| [], _ => by simp
              /-
                🎉 no goals
              -/
               /-
                 α : Type u
                 R : α → α → Prop
                 l : List α
                 x✝ : Not (Membership.mem (List.cons l List.nil) List.nil)
                 ⊢ Iff (List.Chain' R (List.cons l List.nil).flatten) (And (∀ (l_1 : List α), M …
               -/
| [l], _ => by simp [flatten]
               /-
                 🎉 no goals
               -/
| (l₁ :: l₂ :: L), hL => by
    /-
      α : Type u
      R : α → α → Prop
      l₁ l₂ : List α
      L : List (List α)
      hL : Not (Membership.mem (List.cons l₁ (List.cons l₂ L)) List.nil)
      ⊢ Iff (List.Chain' R (List.cons l₁ (List.cons l₂ L)).flatten) (And (∀ (l : Lis …
    -/
    rw [mem_cons, not_or, ← Ne] at hL
    /-
      α : Type u
      R : α → α → Prop
      l₁ l₂ : List α
      L : List (List α)
      hL : And (Ne List.nil l₁) (Not (Membership.mem (List.cons l₂ L) List.nil))
      ⊢ Iff (List.Chain' R (List.cons l₁ (List.cons l₂ L)).flatten) (And (∀ (l : Lis …
    -/
    rw [flatten, chain'_append, chain'_flatten hL.2, forall_mem_cons, chain'_cons]
    /-
      α : Type u
      R : α → α → Prop
      l₁ l₂ : List α
      L : List (List α)
      hL : And (Ne List.nil l₁) (Not (Membership.mem (List.cons l₂ L) List.nil))
      ⊢ Iff (And (List.Chain' R l₁) (And (And (And (List.Chain' R l₂) (∀ (x : List α …
    -/
    rw [mem_cons, not_or, ← Ne] at hL
    /-
      α : Type u
      R : α → α → Prop
      l₁ l₂ : List α
      L : List (List α)
      hL : And (Ne List.nil l₁) (And (Ne List.nil l₂) (Not (Membership.mem L List.ni …
      ⊢ Iff (And (List.Chain' R l₁) (And (And (And (List.Chain' R l₂) (∀ (x : List α …
    -/
    simp only [forall_mem_cons, and_assoc, flatten, head?_append_of_ne_nil _ hL.2.1.symm]
    /-
      α : Type u
      R : α → α → Prop
      l₁ l₂ : List α
      L : List (List α)
      hL : And (Ne List.nil l₁) (And (Ne List.nil l₂) (Not (Membership.mem L List.ni …
      ⊢ Iff (And (List.Chain' R l₁) (And (List.Chain' R l₂) (And (∀ (x : List α), Me …
    -/
    exact Iff.rfl.and (Iff.rfl.and <| Iff.rfl.and and_comm)
    /-
      🎉 no goals
    -/


@[deprecated (since := "2024-10-15")] alias chain'_join := chain'_flatten


theorem chain'_attachWith {l : List α} {p : α → Prop} (h : ∀ x ∈ l, p x)
    {r : {a // p a} → {a // p a} → Prop} :
    (l.attachWith p h).Chain' r ↔ l.Chain' fun a b ↦ ∃ ha hb, r ⟨a, ha⟩ ⟨b, hb⟩ := by
  induction l with
  | nil => rfl
  | cons a l IH =>
    rw [attachWith_cons, chain'_cons', chain'_cons', IH, and_congr_left]
    simp_rw [head?_attachWith]
    intros
    constructor <;>
    intro hc b (hb : _ = _)
    · simp_rw [hb, Option.pbind_some] at hc
      have hb' := h b (mem_cons_of_mem a (mem_of_mem_head? hb))
      exact ⟨h a (mem_cons_self a l), hb', hc ⟨b, hb'⟩ rfl⟩
    · cases l <;> aesop


theorem chain'_attach {l : List α} {r : {a // a ∈ l} → {a // a ∈ l} → Prop} :
    l.attach.Chain' r ↔ l.Chain' fun a b ↦ ∃ ha hb, r ⟨a, ha⟩ ⟨b, hb⟩ :=
  chain'_attachWith fun _ ↦ id


/-- If `a` and `b` are related by the reflexive transitive closure of `r`, then there is an
`r`-chain starting from `a` and ending on `b`.
The converse of `relationReflTransGen_of_exists_chain`.
-/
theorem exists_chain_of_relationReflTransGen (h : Relation.ReflTransGen r a b) :
    ∃ l, Chain r a l ∧ getLast (a :: l) (cons_ne_nil _ _) = b := by
  /-
    α : Type u
    r : α → α → Prop
    a b : α
    h : Relation.ReflTransGen r a b
    ⊢ Exists fun l => And (List.Chain r a l) (Eq ((List.cons a l).getLast ⋯) b)
  -/
  refine Relation.ReflTransGen.head_induction_on h ?_ ?_
    /-
      case refine_1
      α : Type u
      r : α → α → Prop
      a b : α
      h : Relation.ReflTransGen r a b
      ⊢ Exists fun l => And (List.Chain r b l) (Eq ((List.cons b l).getLast ⋯) b)
    -/
  · exact ⟨[], Chain.nil, rfl⟩
    /-
      🎉 no goals
    -/
    /-
      case refine_2
      α : Type u
      r : α → α → Prop
      a b : α
      h : Relation.ReflTransGen r a b
      ⊢ ∀ {a c : α}, r a c → Relation.ReflTransGen r c b → (Exists fun l => And (Lis …
    -/
  · intro c d e _ ih
    /-
      case refine_2
      α : Type u
      r : α → α → Prop
      a b : α
      h : Relation.ReflTransGen r a b
      c d : α
      e : r c d
      h✝ : Relation.ReflTransGen r d b
      ih : Exists fun l => And (List.Chain r d l) (Eq ((List.cons d l).getLast ⋯) b)
      ⊢ Exists fun l => And (List.Chain r c l) (Eq ((List.cons c l).getLast ⋯) b)
    -/
    obtain ⟨l, hl₁, hl₂⟩ := ih
    /-
      case refine_2.intro.intro
      α : Type u
      r : α → α → Prop
      a b : α
      h : Relation.ReflTransGen r a b
      c d : α
      e : r c d
      h✝ : Relation.ReflTransGen r d b
      l : List α
      hl₁ : List.Chain r d l
      hl₂ : Eq ((List.cons d l).getLast ⋯) b
      ⊢ Exists fun l => And (List.Chain r c l) (Eq ((List.cons c l).getLast ⋯) b)
    -/
    refine ⟨d :: l, Chain.cons e hl₁, ?_⟩
    /-
      case refine_2.intro.intro
      α : Type u
      r : α → α → Prop
      a b : α
      h : Relation.ReflTransGen r a b
      c d : α
      e : r c d
      h✝ : Relation.ReflTransGen r d b
      l : List α
      hl₁ : List.Chain r d l
      hl₂ : Eq ((List.cons d l).getLast ⋯) b
      ⊢ Eq ((List.cons c (List.cons d l)).getLast ⋯) b
    -/
    rwa [getLast_cons_cons]
    /-
      🎉 no goals
    -/


/-- Given a chain from `a` to `b`, and a predicate true at `a`, if `r x y → p x → p y` then
the predicate is true everywhere in the chain.
That is, we can propagate the predicate down the chain.
-/
theorem Chain.induction (p : α → Prop) (l : List α) (h : Chain r a l)
    (carries : ∀ ⦃x y : α⦄, r x y → p x → p y) (initial : p a) : ∀ i ∈ l, p i := by
  induction h with
  | nil => simp
  | @cons a b t hab _ h_ind =>
    simp only [mem_cons, forall_eq_or_imp]
    exact ⟨carries hab initial, h_ind (carries hab initial)⟩


/-- A version of `List.Chain.induction` for `List.Chain'`
-/
theorem Chain'.induction (p : α → Prop) (l : List α) (h : Chain' r l)
    (carries : ∀ ⦃x y : α⦄, r x y → p x → p y) (initial : (lne : l ≠ []) → p (l.head lne)) :
    ∀ i ∈ l, p i := by
  /-
    α : Type u
    r : α → α → Prop
    p : α → Prop
    l : List α
    h : List.Chain' r l
    carries : ∀ ⦃x y : α⦄, r x y → p x → p y
    initial : ∀ (lne : Ne l List.nil), p (l.head lne)
    ⊢ ∀ (i : α), Membership.mem l i → p i
  -/
  unfold Chain' at h
  /-
    α : Type u
    r : α → α → Prop
    p : α → Prop
    l : List α
    h : List.next?.match_1 (fun x => Prop) l (fun _ => True) fun a l => List.Chain …
    carries : ∀ ⦃x y : α⦄, r x y → p x → p y
    initial : ∀ (lne : Ne l List.nil), p (l.head lne)
    ⊢ ∀ (i : α), Membership.mem l i → p i
  -/
  split at h
    /-
      case h_1
      α : Type u
      r : α → α → Prop
      p : α → Prop
      carries : ∀ ⦃x y : α⦄, r x y → p x → p y
      x✝ : List α
      initial : ∀ (lne : Ne List.nil List.nil), p (List.nil.head lne)
      h : True
      ⊢ ∀ (i : α), Membership.mem List.nil i → p i
    -/
  · simp
    /-
      🎉 no goals
    -/
  · simp_all only [ne_eq, not_false_eq_true, head_cons, true_implies, mem_cons, forall_eq_or_imp,
      true_and, reduceCtorEq]
    /-
      case h_2
      α : Type u
      r : α → α → Prop
      p : α → Prop
      carries : ∀ ⦃x y : α⦄, r x y → p x → p y
      x✝ : List α
      a✝ : α
      l✝ : List α
      initial : p a✝
      h : List.Chain r a✝ l✝
      ⊢ ∀ (a : α), Membership.mem l✝ a → p a
    -/
    exact h.induction p _ carries initial
    /-
      🎉 no goals
    -/


/-- Given a chain from `a` to `b`, and a predicate true at `b`, if `r x y → p y → p x` then
the predicate is true everywhere in the chain and at `a`.
That is, we can propagate the predicate up the chain.
-/
theorem Chain.backwards_induction (p : α → Prop) (l : List α) (h : Chain r a l)
    (hb : getLast (a :: l) (cons_ne_nil _ _) = b) (carries : ∀ ⦃x y : α⦄, r x y → p y → p x)
    (final : p b) : ∀ i ∈ a :: l, p i := by
  /-
    α : Type u
    r : α → α → Prop
    a b : α
    p : α → Prop
    l : List α
    h : List.Chain r a l
    hb : Eq ((List.cons a l).getLast ⋯) b
    carries : ∀ ⦃x y : α⦄, r x y → p y → p x
    final : p b
    ⊢ ∀ (i : α), Membership.mem (List.cons a l) i → p i
  -/
  have : Chain' (flip (flip r)) (a :: l) := by simpa [Chain']
  /-
    α : Type u
    r : α → α → Prop
    a b : α
    p : α → Prop
    l : List α
    h : List.Chain r a l
    hb : Eq ((List.cons a l).getLast ⋯) b
    carries : ∀ ⦃x y : α⦄, r x y → p y → p x
    final : p b
    this : List.Chain' (flip (flip r)) (List.cons a l)
    ⊢ ∀ (i : α), Membership.mem (List.cons a l) i → p i
  -/
  replace this := chain'_reverse.mpr this
  /-
    α : Type u
    r : α → α → Prop
    a b : α
    p : α → Prop
    l : List α
    h : List.Chain r a l
    hb : Eq ((List.cons a l).getLast ⋯) b
    carries : ∀ ⦃x y : α⦄, r x y → p y → p x
    final : p b
    this : List.Chain' (flip r) (List.cons a l).reverse
    ⊢ ∀ (i : α), Membership.mem (List.cons a l) i → p i
  -/
  simp_rw (config := {singlePass := true}) [← List.mem_reverse]
  /-
    α : Type u
    r : α → α → Prop
    a b : α
    p : α → Prop
    l : List α
    h : List.Chain r a l
    hb : Eq ((List.cons a l).getLast ⋯) b
    carries : ∀ ⦃x y : α⦄, r x y → p y → p x
    final : p b
    this : List.Chain' (flip r) (List.cons a l).reverse
    ⊢ ∀ (i : α), Membership.mem (List.cons a l).reverse i → p i
  -/
  apply this.induction _ _ (fun _ _ h ↦ carries h)
  simpa only [ne_eq, reverse_eq_nil_iff, not_false_eq_true, head_reverse, forall_true_left, hb,
    reduceCtorEq]


/-- Given a chain from `a` to `b`, and a predicate true at `b`, if `r x y → p y → p x` then
the predicate is true at `a`.
That is, we can propagate the predicate all the way up the chain.
-/
@[elab_as_elim]
theorem Chain.backwards_induction_head (p : α → Prop) (l : List α) (h : Chain r a l)
    (hb : getLast (a :: l) (cons_ne_nil _ _) = b) (carries : ∀ ⦃x y : α⦄, r x y → p y → p x)
    (final : p b) : p a :=
  (Chain.backwards_induction p l h hb carries final) _ (mem_cons_self _ _)


/--
If there is an `r`-chain starting from `a` and ending at `b`, then `a` and `b` are related by the
reflexive transitive closure of `r`. The converse of `exists_chain_of_relationReflTransGen`.
-/
theorem relationReflTransGen_of_exists_chain (l : List α) (hl₁ : Chain r a l)
    (hl₂ : getLast (a :: l) (cons_ne_nil _ _) = b) : Relation.ReflTransGen r a b :=
  Chain.backwards_induction_head _ l hl₁ hl₂ (fun _ _ => Relation.ReflTransGen.head)
    Relation.ReflTransGen.refl


theorem Chain'.cons_of_le [LinearOrder α] {a : α} {as m : List α}
    (ha : List.Chain' (· > ·) (a :: as)) (hm : List.Chain' (· > ·) m) (hmas : m ≤ as) :
    List.Chain' (· > ·) (a :: m) := by
  cases m with
  | nil => simp only [List.chain'_singleton]
  | cons b bs =>
    apply hm.cons
    cases as with
    | nil =>
      simp only [le_iff_lt_or_eq, reduceCtorEq, or_false] at hmas
      exact (List.Lex.not_nil_right (·<·) _ hmas).elim
    | cons a' as =>
      rw [List.chain'_cons] at ha
      refine gt_of_gt_of_ge ha.1 ?_
      rw [le_iff_lt_or_eq] at hmas
      cases' hmas with hmas hmas
      · by_contra! hh
        rw [← not_le] at hmas
        apply hmas
        apply le_of_lt
        exact (List.lt_iff_lex_lt _ _).mp (List.lt.head _ _ hh)
      · simp_all only [List.cons.injEq, le_refl]


lemma Chain'.chain {α : Type*} {R : α → α → Prop} {l : List α} {v : α}
    (hl : l.Chain' R) (hv : (lne : l ≠ []) → R v (l.head lne)) : l.Chain R v := by
  /-
    α : Type u_1
    R : α → α → Prop
    l : List α
    v : α
    hl : List.Chain' R l
    hv : ∀ (lne : Ne l List.nil), R v (l.head lne)
    ⊢ List.Chain R v l
  -/
  rw [List.chain_iff_get]
  /-
    α : Type u_1
    R : α → α → Prop
    l : List α
    v : α
    hl : List.Chain' R l
    hv : ∀ (lne : Ne l List.nil), R v (l.head lne)
    ⊢ And (∀ (h : LT.lt 0 l.length), R v (l.get ⟨0, h⟩)) (∀ (i : Nat) (h : LT.lt i …
  -/
  constructor
    /-
      case left
      α : Type u_1
      R : α → α → Prop
      l : List α
      v : α
      hl : List.Chain' R l
      hv : ∀ (lne : Ne l List.nil), R v (l.head lne)
      ⊢ ∀ (h : LT.lt 0 l.length), R v (l.get ⟨0, h⟩)
    -/
  · intro h
    /-
      case left
      α : Type u_1
      R : α → α → Prop
      l : List α
      v : α
      hl : List.Chain' R l
      hv : ∀ (lne : Ne l List.nil), R v (l.head lne)
      h : LT.lt 0 l.length
      ⊢ R v (l.get ⟨0, h⟩)
    -/
    rw [List.get_mk_zero]
    /-
      case left
      α : Type u_1
      R : α → α → Prop
      l : List α
      v : α
      hl : List.Chain' R l
      hv : ∀ (lne : Ne l List.nil), R v (l.head lne)
      h : LT.lt 0 l.length
      ⊢ R v (l.head ⋯)
    -/
    apply hv
    /-
      🎉 no goals
    -/
    /-
      case right
      α : Type u_1
      R : α → α → Prop
      l : List α
      v : α
      hl : List.Chain' R l
      hv : ∀ (lne : Ne l List.nil), R v (l.head lne)
      ⊢ ∀ (i : Nat) (h : LT.lt i (HSub.hSub l.length 1)), R (l.get ⟨i, ⋯⟩) (l.get ⟨H …
    -/
  · exact List.chain'_iff_get.mp hl
    /-
      🎉 no goals
    -/


lemma Chain'.iterate_eq_of_apply_eq {α : Type*} {f : α → α} {l : List α}
    (hl : l.Chain' (fun x y ↦ f x = y)) (i : ℕ) (hi : i < l.length) :
    f^[i] l[0] = l[i] := by
  /-
    α : Type u_1
    f : α → α
    l : List α
    hl : List.Chain' (fun x y => Eq (f x) y) l
    i : Nat
    hi : LT.lt i l.length
    ⊢ Eq (Nat.iterate f i (GetElem.getElem l 0 ⋯)) (GetElem.getElem l i hi)
  -/
  induction' i with i h
    /-
      case zero
      α : Type u_1
      f : α → α
      l : List α
      hl : List.Chain' (fun x y => Eq (f x) y) l
      hi : LT.lt 0 l.length
      ⊢ Eq (Nat.iterate f 0 (GetElem.getElem l 0 ⋯)) (GetElem.getElem l 0 hi)
    -/
  · rfl
    /-
      🎉 no goals
    -/
    /-
      case succ
      α : Type u_1
      f : α → α
      l : List α
      hl : List.Chain' (fun x y => Eq (f x) y) l
      i : Nat
      h : ∀ (hi : LT.lt i l.length), Eq (Nat.iterate f i (GetElem.getElem l 0 ⋯)) (G …
      hi : LT.lt (HAdd.hAdd i 1) l.length
      ⊢ Eq (Nat.iterate f (HAdd.hAdd i 1) (GetElem.getElem l 0 ⋯)) (GetElem.getElem  …
    -/
  · rw [Function.iterate_succ', Function.comp_apply, h (by omega)]
    /-
      case succ
      α : Type u_1
      f : α → α
      l : List α
      hl : List.Chain' (fun x y => Eq (f x) y) l
      i : Nat
      h : ∀ (hi : LT.lt i l.length), Eq (Nat.iterate f i (GetElem.getElem l 0 ⋯)) (G …
      hi : LT.lt (HAdd.hAdd i 1) l.length
      ⊢ Eq (f (GetElem.getElem l i ⋯)) (GetElem.getElem l (HAdd.hAdd i 1) hi)
    -/
    rw [List.chain'_iff_get] at hl
    /-
      case succ
      α : Type u_1
      f : α → α
      l : List α
      hl : ∀ (i : Nat) (h : LT.lt i (HSub.hSub l.length 1)), Eq (f (l.get ⟨i, ⋯⟩)) ( …
      i : Nat
      h : ∀ (hi : LT.lt i l.length), Eq (Nat.iterate f i (GetElem.getElem l 0 ⋯)) (G …
      hi : LT.lt (HAdd.hAdd i 1) l.length
      ⊢ Eq (f (GetElem.getElem l i ⋯)) (GetElem.getElem l (HAdd.hAdd i 1) hi)
    -/
    apply hl
    /-
      case succ.h
      α : Type u_1
      f : α → α
      l : List α
      hl : ∀ (i : Nat) (h : LT.lt i (HSub.hSub l.length 1)), Eq (f (l.get ⟨i, ⋯⟩)) ( …
      i : Nat
      h : ∀ (hi : LT.lt i l.length), Eq (Nat.iterate f i (GetElem.getElem l 0 ⋯)) (G …
      hi : LT.lt (HAdd.hAdd i 1) l.length
      ⊢ LT.lt i (HSub.hSub l.length 1)
    -/
    omega
    /-
      🎉 no goals
    -/


theorem chain'_replicate_of_rel (n : ℕ) {a : α} (h : r a a) : Chain' r (replicate n a) :=
  match n with
  | 0 => chain'_nil
  | n + 1 => chain_replicate_of_rel n h


theorem chain'_eq_iff_eq_replicate {l : List α} :
    Chain' (· = ·) l ↔ ∀ a ∈ l.head?, l = replicate l.length a :=
  match l with
             /-
               α : Type u
               l : List α
               ⊢ Iff (List.Chain' (fun x1 x2 => Eq x1 x2) List.nil) (∀ (a : α), Membership.me …
             -/
  | [] => by simp
             /-
               🎉 no goals
             -/
                 /-
                   α : Type u
                   l✝ : List α
                   a : α
                   l : List α
                   ⊢ Iff (List.Chain' (fun x1 x2 => Eq x1 x2) (List.cons a l)) (∀ (a_1 : α), Memb …
                 -/
  | a :: l => by simp [Chain', chain_eq_iff_eq_replicate, replicate_succ]
                 /-
                   🎉 no goals
                 -/


/-- The type of `r`-decreasing chains -/
abbrev List.chains := { l : List α // l.Chain' (flip r) }


/-- The lexicographic order on the `r`-decreasing chains -/
abbrev List.lex_chains (l m : List.chains r) : Prop := List.Lex r l.val m.val


/-- If an `r`-decreasing chain `l` is empty or its head is accessible by `r`, then
  `l` is accessible by the lexicographic order `List.Lex r`. -/
theorem Acc.list_chain' {l : List.chains r} (acc : ∀ a ∈ l.val.head?, Acc r a) :
    Acc (List.lex_chains r) l := by
  /-
    α : Type u_1
    r : α → α → Prop
    l : List.chains r
    acc : ∀ (a : α), Membership.mem (↑l).head? a → Acc r a
    ⊢ Acc (List.lex_chains r) l
  -/
  obtain ⟨_ | ⟨a, l⟩, hl⟩ := l
    /-
      case mk.nil
      α : Type u_1
      r : α → α → Prop
      hl : List.Chain' (flip r) List.nil
      acc : ∀ (a : α), Membership.mem (↑⟨List.nil, hl⟩).head? a → Acc r a
      ⊢ Acc (List.lex_chains r) ⟨List.nil, hl⟩
    -/
  · apply Acc.intro; rintro ⟨_⟩ ⟨_⟩
                     /-
                       🎉 no goals
                     -/
  /-
    case mk.cons
    α : Type u_1
    r : α → α → Prop
    a : α
    l : List α
    hl : List.Chain' (flip r) (List.cons a l)
    acc : ∀ (a_1 : α), Membership.mem (↑⟨List.cons a l, hl⟩).head? a_1 → Acc r a_1
    ⊢ Acc (List.lex_chains r) ⟨List.cons a l, hl⟩
  -/
  specialize acc a _
    /-
      case mk.cons
      α : Type u_1
      r : α → α → Prop
      a : α
      l : List α
      hl : List.Chain' (flip r) (List.cons a l)
      acc : ∀ (a_1 : α), Membership.mem (↑⟨List.cons a l, hl⟩).head? a_1 → Acc r a_1
      ⊢ Membership.mem (↑⟨List.cons a l, hl⟩).head? a
    -/
  · rw [List.head?_cons, Option.mem_some_iff]
    /-
      🎉 no goals
    -/
  /- For an r-decreasing chain of the form a :: l, apply induction on a -/
  induction acc generalizing l with
  | intro a _ ih =>
    /- Bundle l with a proof that it is r-decreasing to form l' -/
    have hl' := (List.chain'_cons'.1 hl).2
    let l' : List.chains r := ⟨l, hl'⟩
    have : Acc (List.lex_chains r) l' := by
      cases' l with b l
      · apply Acc.intro; rintro ⟨_⟩ ⟨_⟩
      /- l' is accessible by induction hypothesis -/
      · apply ih b (List.chain'_cons.1 hl).1
    /- make l' a free variable and induct on l' -/
    revert hl
    rw [(by rfl : l = l'.1)]
    clear_value l'
    induction this with
    | intro l _ ihl =>
      intro hl
      apply Acc.intro
      rintro ⟨_ | ⟨b, m⟩, hm⟩ (_ | hr | hr)
      · apply Acc.intro; rintro ⟨_⟩ ⟨_⟩
      · apply ihl ⟨m, (List.chain'_cons'.1 hm).2⟩ hr
      · apply ih b hr


/-- If `r` is well-founded, the lexicographic order on `r`-decreasing chains is also. -/
theorem WellFounded.list_chain' (hwf : WellFounded r) :
    WellFounded (List.lex_chains r) :=
  ⟨fun _ ↦ Acc.list_chain' (fun _ _ => hwf.apply _)⟩


instance [hwf : IsWellFounded α r] :
    IsWellFounded (List.chains r) (List.lex_chains r) :=
  ⟨hwf.wf.list_chain'⟩

