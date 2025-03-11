instance : Trans (@List.Perm α) (@List.Perm α) List.Perm where
  trans := @List.Perm.trans α


lemma perm_rfl : l ~ l := Perm.refl _


theorem Perm.subset_congr_left {l₁ l₂ l₃ : List α} (h : l₁ ~ l₂) : l₁ ⊆ l₃ ↔ l₂ ⊆ l₃ :=
  ⟨h.symm.subset.trans, h.subset.trans⟩


theorem Perm.subset_congr_right {l₁ l₂ l₃ : List α} (h : l₁ ~ l₂) : l₃ ⊆ l₁ ↔ l₃ ⊆ l₂ :=
  ⟨fun h' => h'.trans h.subset, fun h' => h'.trans h.symm.subset⟩


local infixr:80 " ∘r " => Relation.Comp


theorem perm_comp_perm : (Perm ∘r Perm : List α → List α → Prop) = Perm := by
  /-
    α : Type u_1
    ⊢ Eq (Relation.Comp List.Perm List.Perm) List.Perm
  -/
  funext a c; apply propext
  /-
    case h.h.a
    α : Type u_1
    a c : List α
    ⊢ Iff (Relation.Comp List.Perm List.Perm a c) (a.Perm c)
  -/
  constructor
    /-
      case h.h.a.mp
      α : Type u_1
      a c : List α
      ⊢ Relation.Comp List.Perm List.Perm a c → a.Perm c
    -/
  · exact fun ⟨b, hab, hba⟩ => Perm.trans hab hba
    /-
      🎉 no goals
    -/
    /-
      case h.h.a.mpr
      α : Type u_1
      a c : List α
      ⊢ a.Perm c → Relation.Comp List.Perm List.Perm a c
    -/
  · exact fun h => ⟨a, Perm.refl a, h⟩
    /-
      🎉 no goals
    -/


theorem perm_comp_forall₂ {l u v} (hlu : Perm l u) (huv : Forall₂ r u v) :
    (Forall₂ r ∘r Perm) l v := by
  induction hlu generalizing v with
  | nil => cases huv; exact ⟨[], Forall₂.nil, Perm.nil⟩
  | cons u _hlu ih =>
    cases' huv with _ b _ v hab huv'
    rcases ih huv' with ⟨l₂, h₁₂, h₂₃⟩
    exact ⟨b :: l₂, Forall₂.cons hab h₁₂, h₂₃.cons _⟩
  | swap a₁ a₂ h₂₃ =>
    cases' huv with _ b₁ _ l₂ h₁ hr₂₃
    cases' hr₂₃ with _ b₂ _ l₂ h₂ h₁₂
    exact ⟨b₂ :: b₁ :: l₂, Forall₂.cons h₂ (Forall₂.cons h₁ h₁₂), Perm.swap _ _ _⟩
  | trans _ _ ih₁ ih₂ =>
    rcases ih₂ huv with ⟨lb₂, hab₂, h₂₃⟩
    rcases ih₁ hab₂ with ⟨lb₁, hab₁, h₁₂⟩
    exact ⟨lb₁, hab₁, Perm.trans h₁₂ h₂₃⟩


theorem forall₂_comp_perm_eq_perm_comp_forall₂ : Forall₂ r ∘r Perm = Perm ∘r Forall₂ r := by
  /-
    α : Type u_1
    β : Type u_2
    r : α → β → Prop
    ⊢ Eq (Relation.Comp (List.Forall₂ r) List.Perm) (Relation.Comp List.Perm (List …
  -/
  funext l₁ l₃; apply propext
  /-
    case h.h.a
    α : Type u_1
    β : Type u_2
    r : α → β → Prop
    l₁ : List α
    l₃ : List β
    ⊢ Iff (Relation.Comp (List.Forall₂ r) List.Perm l₁ l₃) (Relation.Comp List.Per …
  -/
  constructor
    /-
      case h.h.a.mp
      α : Type u_1
      β : Type u_2
      r : α → β → Prop
      l₁ : List α
      l₃ : List β
      ⊢ Relation.Comp (List.Forall₂ r) List.Perm l₁ l₃ → Relation.Comp List.Perm (Li …
    -/
  · intro h
    /-
      case h.h.a.mp
      α : Type u_1
      β : Type u_2
      r : α → β → Prop
      l₁ : List α
      l₃ : List β
      h : Relation.Comp (List.Forall₂ r) List.Perm l₁ l₃
      ⊢ Relation.Comp List.Perm (List.Forall₂ r) l₁ l₃
    -/
    rcases h with ⟨l₂, h₁₂, h₂₃⟩
    /-
      case h.h.a.mp.intro.intro
      α : Type u_1
      β : Type u_2
      r : α → β → Prop
      l₁ : List α
      l₃ l₂ : List β
      h₁₂ : List.Forall₂ r l₁ l₂
      h₂₃ : l₂.Perm l₃
      ⊢ Relation.Comp List.Perm (List.Forall₂ r) l₁ l₃
    -/
    have : Forall₂ (flip r) l₂ l₁ := h₁₂.flip
    /-
      case h.h.a.mp.intro.intro
      α : Type u_1
      β : Type u_2
      r : α → β → Prop
      l₁ : List α
      l₃ l₂ : List β
      h₁₂ : List.Forall₂ r l₁ l₂
      h₂₃ : l₂.Perm l₃
      this : List.Forall₂ (flip r) l₂ l₁
      ⊢ Relation.Comp List.Perm (List.Forall₂ r) l₁ l₃
    -/
    rcases perm_comp_forall₂ h₂₃.symm this with ⟨l', h₁, h₂⟩
    /-
      case h.h.a.mp.intro.intro.intro.intro
      α : Type u_1
      β : Type u_2
      r : α → β → Prop
      l₁ : List α
      l₃ l₂ : List β
      h₁₂ : List.Forall₂ r l₁ l₂
      h₂₃ : l₂.Perm l₃
      this : List.Forall₂ (flip r) l₂ l₁
      l' : List α
      h₁ : List.Forall₂ (flip r) l₃ l'
      h₂ : l'.Perm l₁
      ⊢ Relation.Comp List.Perm (List.Forall₂ r) l₁ l₃
    -/
    exact ⟨l', h₂.symm, h₁.flip⟩
    /-
      🎉 no goals
    -/
    /-
      case h.h.a.mpr
      α : Type u_1
      β : Type u_2
      r : α → β → Prop
      l₁ : List α
      l₃ : List β
      ⊢ Relation.Comp List.Perm (List.Forall₂ r) l₁ l₃ → Relation.Comp (List.Forall₂ …
    -/
  · exact fun ⟨l₂, h₁₂, h₂₃⟩ => perm_comp_forall₂ h₁₂ h₂₃
    /-
      🎉 no goals
    -/


theorem rel_perm_imp (hr : RightUnique r) : (Forall₂ r ⇒ Forall₂ r ⇒ (· → ·)) Perm Perm :=
  fun a b h₁ c d h₂ h =>
  have : (flip (Forall₂ r) ∘r Perm ∘r Forall₂ r) b d := ⟨a, h₁, c, h, h₂⟩
  have : ((flip (Forall₂ r) ∘r Forall₂ r) ∘r Perm) b d := by
    /-
      α : Type u_1
      β : Type u_2
      r : α → β → Prop
      hr : Relator.RightUnique r
      a : List α
      b : List β
      h₁ : List.Forall₂ r a b
      c : List α
      d : List β
      h₂ : List.Forall₂ r c d
      h : a.Perm c
      this : Relation.Comp (flip (List.Forall₂ r)) (Relation.Comp List.Perm (List.Fo …
      ⊢ Relation.Comp (Relation.Comp (flip (List.Forall₂ r)) (List.Forall₂ r)) List. …
    -/
    rwa [← forall₂_comp_perm_eq_perm_comp_forall₂, ← Relation.comp_assoc] at this
    /-
      🎉 no goals
    -/
  let ⟨b', ⟨_, hbc, hcb⟩, hbd⟩ := this
  have : b' = b := right_unique_forall₂' hr hcb hbc
  this ▸ hbd


theorem rel_perm (hr : BiUnique r) : (Forall₂ r ⇒ Forall₂ r ⇒ (· ↔ ·)) Perm Perm :=
  fun _a _b hab _c _d hcd =>
  Iff.intro (rel_perm_imp hr.2 hab hcd) (rel_perm_imp hr.left.flip hab.flip hcd.flip)


lemma count_eq_count_filter_add [DecidableEq α] (P : α → Prop) [DecidablePred P]
    (l : List α) (a : α) :
    count a l = count a (l.filter P) + count a (l.filter (¬ P ·)) := by
  /-
    α : Type u_1
    inst✝¹ : DecidableEq α
    P : α → Prop
    inst✝ : DecidablePred P
    l : List α
    a : α
    ⊢ Eq (List.count a l) (HAdd.hAdd (List.count a (List.filter (fun b => Decidabl …
  -/
  convert countP_eq_countP_filter_add l _ P
  /-
    case h.e'_3.h.e'_6.h.e'_1.h.e'_2.h
    α : Type u_1
    inst✝¹ : DecidableEq α
    P : α → Prop
    inst✝ : DecidablePred P
    l : List α
    a x✝ : α
    ⊢ Eq (Decidable.decide (Not (P x✝))) (Decidable.decide (P x✝)).not
  -/
  simp only [decide_not]
  /-
    🎉 no goals
  -/


theorem Perm.foldl_eq {f : β → α → β} {l₁ l₂ : List α} [rcomm : RightCommutative f] (p : l₁ ~ l₂) :
    ∀ b, foldl f b l₁ = foldl f b l₂ :=
  p.foldl_eq' fun x _hx y _hy z => rcomm.right_comm z x y


theorem Perm.foldr_eq {f : α → β → β} {l₁ l₂ : List α} [lcomm : LeftCommutative f] (p : l₁ ~ l₂) :
    ∀ b, foldr f b l₁ = foldr f b l₂ := by
  /-
    α : Type u_1
    β : Type u_2
    f : α → β → β
    l₁ l₂ : List α
    lcomm : LeftCommutative f
    p : l₁.Perm l₂
    ⊢ ∀ (b : β), Eq (List.foldr f b l₁) (List.foldr f b l₂)
  -/
  intro b
  induction p using Perm.recOnSwap' generalizing b with
  | nil => rfl
  | cons _ _ r  => simp [r b]
  | swap' _ _ _ r => simp only [foldr_cons]; rw [lcomm.left_comm, r b]
  | trans _ _ r₁ r₂ => exact Eq.trans (r₁ b) (r₂ b)


local notation a " * " b => op a b


local notation l " <*> " a => foldl op a l


theorem Perm.foldl_op_eq {l₁ l₂ : List α} {a : α} (h : l₁ ~ l₂) : (l₁ <*> a) = l₂ <*> a :=
  h.foldl_eq _


theorem Perm.foldr_op_eq {l₁ l₂ : List α} {a : α} (h : l₁ ~ l₂) : l₁.foldr op a = l₂.foldr op a :=
  h.foldr_eq _


@[deprecated (since := "2024-09-28")] alias Perm.fold_op_eq := Perm.foldl_op_eq


theorem perm_option_toList {o₁ o₂ : Option α} : o₁.toList ~ o₂.toList ↔ o₁ = o₂ := by
  /-
    α : Type u_1
    o₁ o₂ : Option α
    ⊢ Iff (o₁.toList.Perm o₂.toList) (Eq o₁ o₂)
  -/
  refine ⟨fun p => ?_, fun e => e ▸ Perm.refl _⟩
  /-
    α : Type u_1
    o₁ o₂ : Option α
    p : o₁.toList.Perm o₂.toList
    ⊢ Eq o₁ o₂
  -/
  cases' o₁ with a <;> cases' o₂ with b; · rfl
                                           /-
                                             🎉 no goals
                                           -/
    /-
      case none.some
      α : Type u_1
      b : α
      p : Option.none.toList.Perm (Option.some b).toList
      ⊢ Eq Option.none (Option.some b)
    -/
  · cases p.length_eq
    /-
      🎉 no goals
    -/
    /-
      case some.none
      α : Type u_1
      a : α
      p : (Option.some a).toList.Perm Option.none.toList
      ⊢ Eq (Option.some a) Option.none
    -/
  · cases p.length_eq
    /-
      🎉 no goals
    -/
    /-
      case some.some
      α : Type u_1
      a b : α
      p : (Option.some a).toList.Perm (Option.some b).toList
      ⊢ Eq (Option.some a) (Option.some b)
    -/
  · exact Option.mem_toList.1 (p.symm.subset <| by simp)
    /-
      🎉 no goals
    -/


@[deprecated (since := "2024-10-16")] alias perm_option_to_list := perm_option_toList


theorem perm_replicate_append_replicate
    [DecidableEq α] {l : List α} {a b : α} {m n : ℕ} (h : a ≠ b) :
    l ~ replicate m a ++ replicate n b ↔ count a l = m ∧ count b l = n ∧ l ⊆ [a, b] := by
  /-
    α : Type u_1
    inst✝ : DecidableEq α
    l : List α
    a b : α
    m n : Nat
    h : Ne a b
    ⊢ Iff (l.Perm (HAppend.hAppend (List.replicate m a) (List.replicate n b))) (An …
  -/
  rw [perm_iff_count, ← Decidable.and_forall_ne a, ← Decidable.and_forall_ne b]
  suffices l ⊆ [a, b] ↔ ∀ c, c ≠ b → c ≠ a → c ∉ l by
    simp +contextual [count_replicate, h, this, count_eq_zero, Ne.symm]
  /-
    α : Type u_1
    inst✝ : DecidableEq α
    l : List α
    a b : α
    m n : Nat
    h : Ne a b
    ⊢ Iff (HasSubset.Subset l (List.cons a (List.cons b List.nil))) (∀ (c : α), Ne …
  -/
  trans ∀ c, c ∈ l → c = b ∨ c = a
    /-
      α : Type u_1
      inst✝ : DecidableEq α
      l : List α
      a b : α
      m n : Nat
      h : Ne a b
      ⊢ Iff (HasSubset.Subset l (List.cons a (List.cons b List.nil))) (∀ (c : α), Me …
    -/
  · simp [subset_def, or_comm]
    /-
      🎉 no goals
    -/
    /-
      α : Type u_1
      inst✝ : DecidableEq α
      l : List α
      a b : α
      m n : Nat
      h : Ne a b
      ⊢ Iff (∀ (c : α), Membership.mem l c → Or (Eq c b) (Eq c a)) (∀ (c : α), Ne c  …
    -/
  · exact forall_congr' fun _ => by rw [← and_imp, ← not_or, not_imp_not]
    /-
      🎉 no goals
    -/


theorem Perm.flatMap_left (l : List α) {f g : α → List β} (h : ∀ a ∈ l, f a ~ g a) :
    l.flatMap f ~ l.flatMap g :=
  Perm.flatten_congr <| by
    /-
      α : Type u_1
      β : Type u_2
      l : List α
      f g : α → List β
      h : ∀ (a : α), Membership.mem l a → (f a).Perm (g a)
      ⊢ List.Forall₂ (fun x1 x2 => x1.Perm x2) (List.map f l) (List.map g l)
    -/
    rwa [List.forall₂_map_right_iff, List.forall₂_map_left_iff, List.forall₂_same]
    /-
      🎉 no goals
    -/


@[deprecated (since := "2024-10-16")] alias Perm.bind_left := Perm.flatMap_left


theorem flatMap_append_perm (l : List α) (f g : α → List β) :
    l.flatMap f ++ l.flatMap g ~ l.flatMap fun x => f x ++ g x := by
  /-
    α : Type u_1
    β : Type u_2
    l : List α
    f g : α → List β
    ⊢ (HAppend.hAppend (l.flatMap f) (l.flatMap g)).Perm (l.flatMap fun x => HAppe …
  -/
  induction' l with a l IH
    /-
      case nil
      α : Type u_1
      β : Type u_2
      f g : α → List β
      ⊢ (HAppend.hAppend (List.nil.flatMap f) (List.nil.flatMap g)).Perm (List.nil.f …
    -/
  · simp
    /-
      🎉 no goals
    -/
  /-
    case cons
    α : Type u_1
    β : Type u_2
    f g : α → List β
    a : α
    l : List α
    IH : (HAppend.hAppend (l.flatMap f) (l.flatMap g)).Perm (l.flatMap fun x => HA …
    ⊢ (HAppend.hAppend ((List.cons a l).flatMap f) ((List.cons a l).flatMap g)).Pe …
  -/
  simp only [flatMap_cons, append_assoc]
  /-
    case cons
    α : Type u_1
    β : Type u_2
    f g : α → List β
    a : α
    l : List α
    IH : (HAppend.hAppend (l.flatMap f) (l.flatMap g)).Perm (l.flatMap fun x => HA …
    ⊢ (HAppend.hAppend (f a) (HAppend.hAppend (l.flatMap f) (HAppend.hAppend (g a) …
  -/
  refine (Perm.trans ?_ (IH.append_left _)).append_left _
  /-
    case cons
    α : Type u_1
    β : Type u_2
    f g : α → List β
    a : α
    l : List α
    IH : (HAppend.hAppend (l.flatMap f) (l.flatMap g)).Perm (l.flatMap fun x => HA …
    ⊢ (HAppend.hAppend (l.flatMap f) (HAppend.hAppend (g a) (l.flatMap g))).Perm ( …
  -/
  rw [← append_assoc, ← append_assoc]
  /-
    case cons
    α : Type u_1
    β : Type u_2
    f g : α → List β
    a : α
    l : List α
    IH : (HAppend.hAppend (l.flatMap f) (l.flatMap g)).Perm (l.flatMap fun x => HA …
    ⊢ (HAppend.hAppend (HAppend.hAppend (l.flatMap f) (g a)) (l.flatMap g)).Perm ( …
  -/
  exact perm_append_comm.append_right _
  /-
    🎉 no goals
  -/


@[deprecated (since := "2024-10-16")] alias bind_append_perm := flatMap_append_perm


theorem map_append_flatMap_perm (l : List α) (f : α → β) (g : α → List β) :
    l.map f ++ l.flatMap g ~ l.flatMap fun x => f x :: g x := by
  /-
    α : Type u_1
    β : Type u_2
    l : List α
    f : α → β
    g : α → List β
    ⊢ (HAppend.hAppend (List.map f l) (l.flatMap g)).Perm (l.flatMap fun x => List …
  -/
  simpa [← map_eq_flatMap] using flatMap_append_perm l (fun x => [f x]) g
  /-
    🎉 no goals
  -/


@[deprecated (since := "2024-10-16")] alias map_append_bind_perm := map_append_flatMap_perm


theorem Perm.product_right {l₁ l₂ : List α} (t₁ : List β) (p : l₁ ~ l₂) :
    product l₁ t₁ ~ product l₂ t₁ :=
  p.flatMap_right _


theorem Perm.product_left (l : List α) {t₁ t₂ : List β} (p : t₁ ~ t₂) :
    product l t₁ ~ product l t₂ :=
  (Perm.flatMap_left _) fun _ _ => p.map _


theorem Perm.product {l₁ l₂ : List α} {t₁ t₂ : List β} (p₁ : l₁ ~ l₂) (p₂ : t₁ ~ t₂) :
    product l₁ t₁ ~ product l₂ t₂ :=
  (p₁.product_right t₁).trans (p₂.product_left l₂)


theorem perm_lookmap (f : α → Option α) {l₁ l₂ : List α}
    (H : Pairwise (fun a b => ∀ c ∈ f a, ∀ d ∈ f b, a = b ∧ c = d) l₁) (p : l₁ ~ l₂) :
    lookmap f l₁ ~ lookmap f l₂ := by
  /-
    α : Type u_1
    f : α → Option α
    l₁ l₂ : List α
    H : List.Pairwise (fun a b => ∀ (c : α), Membership.mem (f a) c → ∀ (d : α), M …
    p : l₁.Perm l₂
    ⊢ (List.lookmap f l₁).Perm (List.lookmap f l₂)
  -/
  induction' p with a l₁ l₂ p IH a b l l₁ l₂ l₃ p₁ _ IH₁ IH₂; · simp
                                                                /-
                                                                  🎉 no goals
                                                                -/
    /-
      case cons
      α : Type u_1
      f : α → Option α
      l₁✝ l₂✝ : List α
      a : α
      l₁ l₂ : List α
      p : l₁.Perm l₂
      IH : List.Pairwise (fun a b => ∀ (c : α), Membership.mem (f a) c → ∀ (d : α),  …
      H : List.Pairwise (fun a b => ∀ (c : α), Membership.mem (f a) c → ∀ (d : α), M …
      ⊢ (List.lookmap f (List.cons a l₁)).Perm (List.lookmap f (List.cons a l₂))
    -/
  · cases h : f a
      /-
        case cons.none
        α : Type u_1
        f : α → Option α
        l₁✝ l₂✝ : List α
        a : α
        l₁ l₂ : List α
        p : l₁.Perm l₂
        IH : List.Pairwise (fun a b => ∀ (c : α), Membership.mem (f a) c → ∀ (d : α),  …
        H : List.Pairwise (fun a b => ∀ (c : α), Membership.mem (f a) c → ∀ (d : α), M …
        h : Eq (f a) Option.none
        ⊢ (List.lookmap f (List.cons a l₁)).Perm (List.lookmap f (List.cons a l₂))
      -/
    · simpa [h] using IH (pairwise_cons.1 H).2
      /-
        🎉 no goals
      -/
      /-
        case cons.some
        α : Type u_1
        f : α → Option α
        l₁✝ l₂✝ : List α
        a : α
        l₁ l₂ : List α
        p : l₁.Perm l₂
        IH : List.Pairwise (fun a b => ∀ (c : α), Membership.mem (f a) c → ∀ (d : α),  …
        H : List.Pairwise (fun a b => ∀ (c : α), Membership.mem (f a) c → ∀ (d : α), M …
        val✝ : α
        h : Eq (f a) (Option.some val✝)
        ⊢ (List.lookmap f (List.cons a l₁)).Perm (List.lookmap f (List.cons a l₂))
      -/
    · simp [lookmap_cons_some _ _ h, p]
      /-
        🎉 no goals
      -/
    /-
      case swap
      α : Type u_1
      f : α → Option α
      l₁ l₂ : List α
      a b : α
      l : List α
      H : List.Pairwise (fun a b => ∀ (c : α), Membership.mem (f a) c → ∀ (d : α), M …
      ⊢ (List.lookmap f (List.cons b (List.cons a l))).Perm (List.lookmap f (List.co …
    -/
  · cases' h₁ : f a with c <;> cases' h₂ : f b with d
      /-
        case swap.none.none
        α : Type u_1
        f : α → Option α
        l₁ l₂ : List α
        a b : α
        l : List α
        H : List.Pairwise (fun a b => ∀ (c : α), Membership.mem (f a) c → ∀ (d : α), M …
        h₁ : Eq (f a) Option.none
        h₂ : Eq (f b) Option.none
        ⊢ (List.lookmap f (List.cons b (List.cons a l))).Perm (List.lookmap f (List.co …
      -/
    · simpa [h₁, h₂] using swap _ _ _
      /-
        🎉 no goals
      -/
      /-
        case swap.none.some
        α : Type u_1
        f : α → Option α
        l₁ l₂ : List α
        a b : α
        l : List α
        H : List.Pairwise (fun a b => ∀ (c : α), Membership.mem (f a) c → ∀ (d : α), M …
        h₁ : Eq (f a) Option.none
        d : α
        h₂ : Eq (f b) (Option.some d)
        ⊢ (List.lookmap f (List.cons b (List.cons a l))).Perm (List.lookmap f (List.co …
      -/
    · simpa [h₁, lookmap_cons_some _ _ h₂] using swap _ _ _
      /-
        🎉 no goals
      -/
      /-
        case swap.some.none
        α : Type u_1
        f : α → Option α
        l₁ l₂ : List α
        a b : α
        l : List α
        H : List.Pairwise (fun a b => ∀ (c : α), Membership.mem (f a) c → ∀ (d : α), M …
        c : α
        h₁ : Eq (f a) (Option.some c)
        h₂ : Eq (f b) Option.none
        ⊢ (List.lookmap f (List.cons b (List.cons a l))).Perm (List.lookmap f (List.co …
      -/
    · simpa [lookmap_cons_some _ _ h₁, h₂] using swap _ _ _
      /-
        🎉 no goals
      -/
      /-
        case swap.some.some
        α : Type u_1
        f : α → Option α
        l₁ l₂ : List α
        a b : α
        l : List α
        H : List.Pairwise (fun a b => ∀ (c : α), Membership.mem (f a) c → ∀ (d : α), M …
        c : α
        h₁ : Eq (f a) (Option.some c)
        d : α
        h₂ : Eq (f b) (Option.some d)
        ⊢ (List.lookmap f (List.cons b (List.cons a l))).Perm (List.lookmap f (List.co …
      -/
    · rcases (pairwise_cons.1 H).1 _ (mem_cons.2 (Or.inl rfl)) _ h₂ _ h₁ with ⟨rfl, rfl⟩
      /-
        case swap.some.some.intro
        α : Type u_1
        f : α → Option α
        l₁ l₂ : List α
        b : α
        l : List α
        d : α
        h₂ : Eq (f b) (Option.some d)
        H : List.Pairwise (fun a b => ∀ (c : α), Membership.mem (f a) c → ∀ (d : α), M …
        h₁ : Eq (f b) (Option.some d)
        ⊢ (List.lookmap f (List.cons b (List.cons b l))).Perm (List.lookmap f (List.co …
      -/
      exact Perm.refl _
      /-
        🎉 no goals
      -/
    /-
      case trans
      α : Type u_1
      f : α → Option α
      l₁✝ l₂✝ l₁ l₂ l₃ : List α
      p₁ : l₁.Perm l₂
      a✝ : l₂.Perm l₃
      IH₁ : List.Pairwise (fun a b => ∀ (c : α), Membership.mem (f a) c → ∀ (d : α), …
      IH₂ : List.Pairwise (fun a b => ∀ (c : α), Membership.mem (f a) c → ∀ (d : α), …
      H : List.Pairwise (fun a b => ∀ (c : α), Membership.mem (f a) c → ∀ (d : α), M …
      ⊢ (List.lookmap f l₁).Perm (List.lookmap f l₃)
    -/
  · refine (IH₁ H).trans (IH₂ ((p₁.pairwise_iff ?_).1 H))
    /-
      case trans
      α : Type u_1
      f : α → Option α
      l₁✝ l₂✝ l₁ l₂ l₃ : List α
      p₁ : l₁.Perm l₂
      a✝ : l₂.Perm l₃
      IH₁ : List.Pairwise (fun a b => ∀ (c : α), Membership.mem (f a) c → ∀ (d : α), …
      IH₂ : List.Pairwise (fun a b => ∀ (c : α), Membership.mem (f a) c → ∀ (d : α), …
      H : List.Pairwise (fun a b => ∀ (c : α), Membership.mem (f a) c → ∀ (d : α), M …
      ⊢ ∀ {x y : α}, (∀ (c : α), Membership.mem (f x) c → ∀ (d : α), Membership.mem  …
    -/
    intro x y h c hc d hd
    /-
      case trans
      α : Type u_1
      f : α → Option α
      l₁✝ l₂✝ l₁ l₂ l₃ : List α
      p₁ : l₁.Perm l₂
      a✝ : l₂.Perm l₃
      IH₁ : List.Pairwise (fun a b => ∀ (c : α), Membership.mem (f a) c → ∀ (d : α), …
      IH₂ : List.Pairwise (fun a b => ∀ (c : α), Membership.mem (f a) c → ∀ (d : α), …
      H : List.Pairwise (fun a b => ∀ (c : α), Membership.mem (f a) c → ∀ (d : α), M …
      x y : α
      h : ∀ (c : α), Membership.mem (f x) c → ∀ (d : α), Membership.mem (f y) d → An …
      c : α
      hc : Membership.mem (f y) c
      d : α
      hd : Membership.mem (f x) d
      ⊢ And (Eq y x) (Eq c d)
    -/
    rw [@eq_comm _ y, @eq_comm _ c]
    /-
      case trans
      α : Type u_1
      f : α → Option α
      l₁✝ l₂✝ l₁ l₂ l₃ : List α
      p₁ : l₁.Perm l₂
      a✝ : l₂.Perm l₃
      IH₁ : List.Pairwise (fun a b => ∀ (c : α), Membership.mem (f a) c → ∀ (d : α), …
      IH₂ : List.Pairwise (fun a b => ∀ (c : α), Membership.mem (f a) c → ∀ (d : α), …
      H : List.Pairwise (fun a b => ∀ (c : α), Membership.mem (f a) c → ∀ (d : α), M …
      x y : α
      h : ∀ (c : α), Membership.mem (f x) c → ∀ (d : α), Membership.mem (f y) d → An …
      c : α
      hc : Membership.mem (f y) c
      d : α
      hd : Membership.mem (f x) d
      ⊢ And (Eq x y) (Eq d c)
    -/
    apply h d hd c hc
    /-
      🎉 no goals
    -/


