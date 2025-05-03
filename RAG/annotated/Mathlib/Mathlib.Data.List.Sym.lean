/-- `xs.sym2` is a list of all unordered pairs of elements from `xs`.
If `xs` has no duplicates then neither does `xs.sym2`. -/
protected def sym2 : List α → List (Sym2 α)
  | [] => []
  | x :: xs => (x :: xs).map (fun y => s(x, y)) ++ xs.sym2


theorem sym2_map (f : α → β) (xs : List α) :
    (xs.map f).sym2 = xs.sym2.map (Sym2.map f) := by
  induction xs with
  | nil => simp [List.sym2]
  | cons x xs ih => simp [List.sym2, ih, Function.comp]


theorem mem_sym2_cons_iff {x : α} {xs : List α} {z : Sym2 α} :
    z ∈ (x :: xs).sym2 ↔ z = s(x, x) ∨ (∃ y, y ∈ xs ∧ z = s(x, y)) ∨ z ∈ xs.sym2 := by
  /-
    α : Type u_1
    x : α
    xs : List α
    z : Sym2 α
    ⊢ Iff (Membership.mem (List.cons x xs).sym2 z) (Or (Eq z (Sym2.mk { fst := x,  …
  -/
  simp only [List.sym2, map_cons, cons_append, mem_cons, mem_append, mem_map]
  /-
    α : Type u_1
    x : α
    xs : List α
    z : Sym2 α
    ⊢ Iff (Or (Eq z (Sym2.mk { fst := x, snd := x })) (Or (Exists fun a => And (Me …
  -/
  simp only [eq_comm]
  /-
    🎉 no goals
  -/


@[simp]
theorem sym2_eq_nil_iff {xs : List α} : xs.sym2 = [] ↔ xs = [] := by
  /-
    α : Type u_1
    xs : List α
    ⊢ Iff (Eq xs.sym2 List.nil) (Eq xs List.nil)
  -/
               /-
                 🎉 no goals
               -/
  cases xs <;> simp [List.sym2]
               /-
                 🎉 no goals
               -/


theorem left_mem_of_mk_mem_sym2 {xs : List α} {a b : α}
    (h : s(a, b) ∈ xs.sym2) : a ∈ xs := by
  induction xs with
  | nil => exact (not_mem_nil _ h).elim
  | cons x xs ih =>
    rw [mem_cons]
    rw [mem_sym2_cons_iff] at h
    obtain (h | ⟨c, hc, h⟩ | h) := h
    · rw [Sym2.eq_iff, ← and_or_left] at h
      exact .inl h.1
    · rw [Sym2.eq_iff] at h
      obtain (⟨rfl, rfl⟩ | ⟨rfl, rfl⟩) := h <;> simp [hc]
    · exact .inr <| ih h


theorem right_mem_of_mk_mem_sym2 {xs : List α} {a b : α}
    (h : s(a, b) ∈ xs.sym2) : b ∈ xs := by
  /-
    α : Type u_1
    xs : List α
    a b : α
    h : Membership.mem xs.sym2 (Sym2.mk { fst := a, snd := b })
    ⊢ Membership.mem xs b
  -/
  rw [Sym2.eq_swap] at h
  /-
    α : Type u_1
    xs : List α
    a b : α
    h : Membership.mem xs.sym2 (Sym2.mk { fst := b, snd := a })
    ⊢ Membership.mem xs b
  -/
  exact left_mem_of_mk_mem_sym2 h
  /-
    🎉 no goals
  -/


theorem mk_mem_sym2 {xs : List α} {a b : α} (ha : a ∈ xs) (hb : b ∈ xs) :
    s(a, b) ∈ xs.sym2 := by
  induction xs with
  | nil => simp at ha
  | cons x xs ih =>
    rw [mem_sym2_cons_iff]
    rw [mem_cons] at ha hb
    obtain (rfl | ha) := ha <;> obtain (rfl | hb) := hb
    · left; rfl
    · right; left; use b
    · right; left; rw [Sym2.eq_swap]; use a
    · right; right; exact ih ha hb


theorem mk_mem_sym2_iff {xs : List α} {a b : α} :
    s(a, b) ∈ xs.sym2 ↔ a ∈ xs ∧ b ∈ xs := by
  /-
    α : Type u_1
    xs : List α
    a b : α
    ⊢ Iff (Membership.mem xs.sym2 (Sym2.mk { fst := a, snd := b })) (And (Membersh …
  -/
  constructor
    /-
      case mp
      α : Type u_1
      xs : List α
      a b : α
      ⊢ Membership.mem xs.sym2 (Sym2.mk { fst := a, snd := b }) → And (Membership.me …
    -/
  · intro h
    /-
      case mp
      α : Type u_1
      xs : List α
      a b : α
      h : Membership.mem xs.sym2 (Sym2.mk { fst := a, snd := b })
      ⊢ And (Membership.mem xs a) (Membership.mem xs b)
    -/
    exact ⟨left_mem_of_mk_mem_sym2 h, right_mem_of_mk_mem_sym2 h⟩
    /-
      🎉 no goals
    -/
    /-
      case mpr
      α : Type u_1
      xs : List α
      a b : α
      ⊢ And (Membership.mem xs a) (Membership.mem xs b) → Membership.mem xs.sym2 (Sy …
    -/
  · rintro ⟨ha, hb⟩
    /-
      case mpr.intro
      α : Type u_1
      xs : List α
      a b : α
      ha : Membership.mem xs a
      hb : Membership.mem xs b
      ⊢ Membership.mem xs.sym2 (Sym2.mk { fst := a, snd := b })
    -/
    exact mk_mem_sym2 ha hb
    /-
      🎉 no goals
    -/


theorem mem_sym2_iff {xs : List α} {z : Sym2 α} :
    z ∈ xs.sym2 ↔ ∀ y ∈ z, y ∈ xs := by
  /-
    α : Type u_1
    xs : List α
    z : Sym2 α
    ⊢ Iff (Membership.mem xs.sym2 z) (∀ (y : α), Membership.mem z y → Membership.m …
  -/
  refine z.ind (fun a b => ?_)
  /-
    α : Type u_1
    xs : List α
    z : Sym2 α
    a b : α
    ⊢ Iff (Membership.mem xs.sym2 (Sym2.mk { fst := a, snd := b })) (∀ (y : α), Me …
  -/
  simp [mk_mem_sym2_iff]
  /-
    🎉 no goals
  -/


protected theorem Nodup.sym2 {xs : List α} (h : xs.Nodup) : xs.sym2.Nodup := by
  induction xs with
  | nil => simp only [List.sym2, nodup_nil]
  | cons x xs ih =>
    rw [List.sym2]
    specialize ih h.of_cons
    rw [nodup_cons] at h
    refine Nodup.append (Nodup.cons ?notmem (h.2.map ?inj)) ih ?disj
    case disj =>
      intro z hz hz'
      simp only [mem_cons, mem_map] at hz
      obtain ⟨_, (rfl | _), rfl⟩ := hz
        <;> simp [left_mem_of_mk_mem_sym2 hz'] at h
    case notmem =>
      intro h'
      simp only [h.1, mem_map, Sym2.eq_iff, true_and, or_self, exists_eq_right] at h'
    case inj =>
      intro a b
      simp only [Sym2.eq_iff, true_and]
      rintro (rfl | ⟨rfl, rfl⟩) <;> rfl


theorem map_mk_sublist_sym2 (x : α) (xs : List α) (h : x ∈ xs) :
    map (fun y ↦ s(x, y)) xs <+ xs.sym2 := by
  induction xs with
  | nil => simp
  | cons x' xs ih =>
    simp only [map_cons, List.sym2, cons_append]
    cases h with
    | head =>
      exact (sublist_append_left _ _).cons₂ _
    | tail _ h =>
      refine .cons _ ?_
      rw [← singleton_append]
      refine .append ?_ (ih h)
      rw [singleton_sublist, mem_map]
      exact ⟨_, h, Sym2.eq_swap⟩


theorem map_mk_disjoint_sym2 (x : α) (xs : List α) (h : x ∉ xs) :
    (map (fun y ↦ s(x, y)) xs).Disjoint xs.sym2 := by
  induction xs with
  | nil => simp
  | cons x' xs ih =>
    simp only [mem_cons, not_or] at h
    rw [List.sym2, map_cons, map_cons, disjoint_cons_left, disjoint_append_right,
      disjoint_cons_right]
    refine ⟨?_, ⟨?_, ?_⟩, ?_⟩
    · refine not_mem_cons_of_ne_of_not_mem ?_ (not_mem_append ?_ ?_)
      · simp [h.1]
      · simp_rw [mem_map, not_exists, not_and]
        intro x'' hx
        simp_rw [Sym2.mk_eq_mk_iff, Prod.swap_prod_mk, Prod.mk.injEq, true_and]
        rintro (⟨rfl, rfl⟩ | rfl)
        · exact h.2 hx
        · exact h.2 hx
      · simp [mk_mem_sym2_iff, h.2]
    · simp [h.1]
    · intro z hx hy
      rw [List.mem_map] at hx hy
      obtain ⟨a, hx, rfl⟩ := hx
      obtain ⟨b, hy, hx⟩ := hy
      simp [Sym2.mk_eq_mk_iff, Ne.symm h.1] at hx
      obtain ⟨rfl, rfl⟩ := hx
      exact h.2 hy
    · exact ih h.2


theorem dedup_sym2 [DecidableEq α] (xs : List α) : xs.sym2.dedup = xs.dedup.sym2 := by
  induction xs with
  | nil => simp only [List.sym2, dedup_nil]
  | cons x xs ih =>
    simp only [List.sym2, map_cons, cons_append]
    obtain hm | hm := Decidable.em (x ∈ xs)
    · rw [dedup_cons_of_mem hm, ← ih, dedup_cons_of_mem,
        List.Subset.dedup_append_right (map_mk_sublist_sym2 _ _ hm).subset]
      refine mem_append_left _ ?_
      rw [mem_map]
      exact ⟨_, hm, Sym2.eq_swap⟩
    · rw [dedup_cons_of_not_mem hm, List.sym2, map_cons, ← ih, dedup_cons_of_not_mem, cons_append,
        List.Disjoint.dedup_append, dedup_map_of_injective]
      · exact (Sym2.mkEmbedding _).injective
      · exact map_mk_disjoint_sym2 x xs hm
      · simp [hm, mem_sym2_iff]


protected theorem Perm.sym2 {xs ys : List α} (h : xs ~ ys) :
    xs.sym2 ~ ys.sym2 := by
  induction h with
  | nil => rfl
  | cons x h ih =>
    simp only [List.sym2, map_cons, cons_append, perm_cons]
    exact (h.map _).append ih
  | swap x y xs =>
    simp only [List.sym2, map_cons, cons_append]
    conv => enter [1,2,1]; rw [Sym2.eq_swap]
    -- Explicit permutation to speed up simps that follow.
    refine Perm.trans (Perm.swap ..) (Perm.trans (Perm.cons _ ?_) (Perm.swap ..))
    simp only [← Multiset.coe_eq_coe, ← Multiset.cons_coe,
      ← Multiset.coe_add, ← Multiset.singleton_add]
    simp only [add_assoc, add_left_comm]
  | trans _ _ ih1 ih2 => exact ih1.trans ih2


protected theorem Sublist.sym2 {xs ys : List α} (h : xs <+ ys) : xs.sym2 <+ ys.sym2 := by
  induction h with
  | slnil => apply slnil
  | cons a h ih =>
    simp only [List.sym2]
    exact Sublist.append (nil_sublist _) ih
  | cons₂ a h ih =>
    simp only [List.sym2, map_cons, cons_append]
    exact cons₂ _ (append (Sublist.map _ h) ih)


protected theorem Subperm.sym2 {xs ys : List α} (h : xs <+~ ys) : xs.sym2 <+~ ys.sym2 := by
  /-
    α : Type u_1
    xs ys : List α
    h : xs.Subperm ys
    ⊢ xs.sym2.Subperm ys.sym2
  -/
  obtain ⟨xs', hx, h⟩ := h
  /-
    case intro.intro
    α : Type u_1
    xs ys xs' : List α
    hx : xs'.Perm xs
    h : xs'.Sublist ys
    ⊢ xs.sym2.Subperm ys.sym2
  -/
  exact hx.sym2.symm.subperm.trans h.sym2.subperm
  /-
    🎉 no goals
  -/


theorem length_sym2 {xs : List α} : xs.sym2.length = Nat.choose (xs.length + 1) 2 := by
  induction xs with
  | nil => rfl
  | cons x xs ih =>
    rw [List.sym2, length_append, length_map, length_cons,
        Nat.choose_succ_succ, ← ih, Nat.choose_one_right]


/-- `xs.sym n` is all unordered `n`-tuples from the list `xs` in some order. -/
protected def sym : (n : ℕ) → List α → List (Sym α n)
  | 0, _ => [.nil]
  | _, [] => []
  | n + 1, x :: xs => ((x :: xs).sym n |>.map fun p => x ::ₛ p) ++ xs.sym (n + 1)


theorem sym_one_eq : xs.sym 1 = xs.map (· ::ₛ .nil) := by
  induction xs with
  | nil => simp only [List.sym, Nat.succ_eq_add_one, Nat.reduceAdd, map_nil]
  | cons x xs ih =>
    rw [map_cons, ← ih, List.sym, List.sym, map_singleton, singleton_append]


theorem sym2_eq_sym_two : xs.sym2.map (Sym2.equivSym α) = xs.sym 2 := by
  induction xs with
  | nil => simp only [List.sym, map_eq_nil_iff, sym2_eq_nil_iff]
  | cons x xs ih =>
    rw [List.sym, ← ih, sym_one_eq, map_map, List.sym2, map_append, map_map]
    rfl


theorem sym_map {β : Type*} (f : α → β) (n : ℕ) (xs : List α) :
    (xs.map f).sym n = (xs.sym n).map (Sym.map f) :=
  match n, xs with
               /-
                 α : Type u_1
                 β : Type u_3
                 f : α → β
                 n : Nat
                 xs x✝ : List α
                 ⊢ Eq (List.sym 0 (List.map f x✝)) (List.map (Sym.map f) (List.sym 0 x✝))
               -/
  | 0, _ => by simp only [List.sym]; rfl
                                     /-
                                       🎉 no goals
                                     -/
                    /-
                      α : Type u_1
                      β : Type u_3
                      f : α → β
                      n✝ : Nat
                      xs : List α
                      n : Nat
                      ⊢ Eq (List.sym (HAdd.hAdd n 1) (List.map f List.nil)) (List.map (Sym.map f) (L …
                    -/
  | n + 1, [] => by simp [List.sym]
                    /-
                      🎉 no goals
                    -/
  | n + 1, x :: xs => by
    /-
      α : Type u_1
      β : Type u_3
      f : α → β
      n✝ : Nat
      xs✝ : List α
      n : Nat
      x : α
      xs : List α
      ⊢ Eq (List.sym (HAdd.hAdd n 1) (List.map f (List.cons x xs))) (List.map (Sym.m …
    -/
    rw [map_cons, List.sym, ← map_cons, sym_map f n (x :: xs), sym_map f (n + 1) xs]
    /-
      α : Type u_1
      β : Type u_3
      f : α → β
      n✝ : Nat
      xs✝ : List α
      n : Nat
      x : α
      xs : List α
      ⊢ Eq (HAppend.hAppend (List.map (fun p => Sym.cons (f x) p) (List.map (Sym.map …
    -/
    simp only [map_map, List.sym, map_append, append_cancel_right_eq]
    /-
      α : Type u_1
      β : Type u_3
      f : α → β
      n✝ : Nat
      xs✝ : List α
      n : Nat
      x : α
      xs : List α
      ⊢ Eq (List.map (Function.comp (fun p => Sym.cons (f x) p) (Sym.map f)) (List.s …
    -/
    congr
    /-
      case e_f
      α : Type u_1
      β : Type u_3
      f : α → β
      n✝ : Nat
      xs✝ : List α
      n : Nat
      x : α
      xs : List α
      ⊢ Eq (Function.comp (fun p => Sym.cons (f x) p) (Sym.map f)) (Function.comp (S …
    -/
    ext s
    /-
      case e_f.h.h
      α : Type u_1
      β : Type u_3
      f : α → β
      n✝ : Nat
      xs✝ : List α
      n : Nat
      x : α
      xs : List α
      s : Sym α n
      ⊢ Eq ↑(Function.comp (fun p => Sym.cons (f x) p) (Sym.map f) s) ↑(Function.com …
    -/
    simp only [Function.comp_apply, Sym.map_cons]
    /-
      🎉 no goals
    -/


protected theorem Sublist.sym (n : ℕ) {xs ys : List α} (h : xs <+ ys) : xs.sym n <+ ys.sym n :=
  match n, h with
               /-
                 α : Type u_1
                 n : Nat
                 xs✝ ys✝ : List α
                 h : xs✝.Sublist ys✝
                 ys xs : List α
                 x✝ : xs.Sublist ys
                 ⊢ (List.sym 0 xs).Sublist (List.sym 0 ys)
               -/
  | 0, _ => by simp [List.sym]
               /-
                 🎉 no goals
               -/
                        /-
                          α : Type u_1
                          n✝ : Nat
                          xs ys : List α
                          h : xs.Sublist ys
                          n : Nat
                          ⊢ (List.sym (HAdd.hAdd n 1) List.nil).Sublist (List.sym (HAdd.hAdd n 1) List.n …
                        -/
  | n + 1, .slnil => by simp only [refl]
                        /-
                          🎉 no goals
                        -/
  | n + 1, .cons a h => by
    /-
      α : Type u_1
      n✝ : Nat
      xs✝ ys : List α
      h✝ : xs✝.Sublist ys
      xs : List α
      n : Nat
      l₂✝ : List α
      a : α
      h : xs.Sublist l₂✝
      ⊢ (List.sym (HAdd.hAdd n 1) xs).Sublist (List.sym (HAdd.hAdd n 1) (List.cons a …
    -/
    rw [List.sym, ← nil_append (List.sym (n + 1) xs)]
    /-
      α : Type u_1
      n✝ : Nat
      xs✝ ys : List α
      h✝ : xs✝.Sublist ys
      xs : List α
      n : Nat
      l₂✝ : List α
      a : α
      h : xs.Sublist l₂✝
      ⊢ (HAppend.hAppend List.nil (List.sym (HAdd.hAdd n 1) xs)).Sublist (HAppend.hA …
    -/
    apply Sublist.append (nil_sublist _)
    /-
      α : Type u_1
      n✝ : Nat
      xs✝ ys : List α
      h✝ : xs✝.Sublist ys
      xs : List α
      n : Nat
      l₂✝ : List α
      a : α
      h : xs.Sublist l₂✝
      ⊢ (List.sym (HAdd.hAdd n 1) xs).Sublist (List.sym (HAdd.hAdd n 1) l₂✝)
    -/
    exact h.sym (n + 1)
    /-
      🎉 no goals
    -/
  | n + 1, .cons₂ a h => by
    /-
      α : Type u_1
      n✝ : Nat
      xs ys : List α
      h✝ : xs.Sublist ys
      n : Nat
      l₁✝ l₂✝ : List α
      a : α
      h : l₁✝.Sublist l₂✝
      ⊢ (List.sym (HAdd.hAdd n 1) (List.cons a l₁✝)).Sublist (List.sym (HAdd.hAdd n  …
    -/
    rw [List.sym, List.sym]
    /-
      α : Type u_1
      n✝ : Nat
      xs ys : List α
      h✝ : xs.Sublist ys
      n : Nat
      l₁✝ l₂✝ : List α
      a : α
      h : l₁✝.Sublist l₂✝
      ⊢ (HAppend.hAppend (List.map (fun p => Sym.cons a p) (List.sym n (List.cons a  …
    -/
    apply Sublist.append
      /-
        case hl
        α : Type u_1
        n✝ : Nat
        xs ys : List α
        h✝ : xs.Sublist ys
        n : Nat
        l₁✝ l₂✝ : List α
        a : α
        h : l₁✝.Sublist l₂✝
        ⊢ (List.map (fun p => Sym.cons a p) (List.sym n (List.cons a l₁✝))).Sublist (L …
      -/
    · exact ((cons₂ a h).sym n).map _
      /-
        🎉 no goals
      -/
      /-
        case hr
        α : Type u_1
        n✝ : Nat
        xs ys : List α
        h✝ : xs.Sublist ys
        n : Nat
        l₁✝ l₂✝ : List α
        a : α
        h : l₁✝.Sublist l₂✝
        ⊢ (List.sym (HAdd.hAdd n 1) l₁✝).Sublist (List.sym (HAdd.hAdd n 1) l₂✝)
      -/
    · exact h.sym (n + 1)
      /-
        🎉 no goals
      -/


theorem sym_sublist_sym_cons {a : α} : xs.sym n <+ (a :: xs).sym n :=
  (sublist_cons_self a xs).sym n


theorem mem_of_mem_of_mem_sym {n : ℕ} {xs : List α} {a : α} {z : Sym α n}
    (ha : a ∈ z) (hz : z ∈ xs.sym n) : a ∈ xs :=
  match n, xs with
  | 0, xs => by
    /-
      α : Type u_1
      n : Nat
      xs✝ : List α
      a : α
      xs : List α
      z : Sym α 0
      ha : Membership.mem z a
      hz : Membership.mem (List.sym 0 xs) z
      ⊢ Membership.mem xs a
    -/
    cases Sym.eq_nil_of_card_zero z
    /-
      case refl
      α : Type u_1
      n : Nat
      xs✝ : List α
      a : α
      xs : List α
      ha : Membership.mem Sym.nil a
      hz : Membership.mem (List.sym 0 xs) Sym.nil
      ⊢ Membership.mem xs a
    -/
    simp at ha
    /-
      🎉 no goals
    -/
                    /-
                      α : Type u_1
                      n✝ : Nat
                      xs : List α
                      a : α
                      n : Nat
                      z : Sym α (HAdd.hAdd n 1)
                      ha : Membership.mem z a
                      hz : Membership.mem (List.sym (HAdd.hAdd n 1) List.nil) z
                      ⊢ Membership.mem List.nil a
                    -/
  | n + 1, [] => by simp [List.sym] at hz
                    /-
                      🎉 no goals
                    -/
  | n + 1, x :: xs => by
    /-
      α : Type u_1
      n✝ : Nat
      xs✝ : List α
      a : α
      n : Nat
      x : α
      xs : List α
      z : Sym α (HAdd.hAdd n 1)
      ha : Membership.mem z a
      hz : Membership.mem (List.sym (HAdd.hAdd n 1) (List.cons x xs)) z
      ⊢ Membership.mem (List.cons x xs) a
    -/
    rw [List.sym, mem_append, mem_map] at hz
    /-
      α : Type u_1
      n✝ : Nat
      xs✝ : List α
      a : α
      n : Nat
      x : α
      xs : List α
      z : Sym α (HAdd.hAdd n 1)
      ha : Membership.mem z a
      hz : Or (Exists fun a => And (Membership.mem (List.sym n (List.cons x xs)) a)  …
      ⊢ Membership.mem (List.cons x xs) a
    -/
    obtain ⟨z, hz, rfl⟩ | hz := hz
      /-
        case inl.intro.intro
        α : Type u_1
        n✝ : Nat
        xs✝ : List α
        a : α
        n : Nat
        x : α
        xs : List α
        z : Sym α n
        hz : Membership.mem (List.sym n (List.cons x xs)) z
        ha : Membership.mem (Sym.cons x z) a
        ⊢ Membership.mem (List.cons x xs) a
      -/
    · rw [Sym.mem_cons] at ha
      /-
        case inl.intro.intro
        α : Type u_1
        n✝ : Nat
        xs✝ : List α
        a : α
        n : Nat
        x : α
        xs : List α
        z : Sym α n
        hz : Membership.mem (List.sym n (List.cons x xs)) z
        ha : Or (Eq a x) (Membership.mem z a)
        ⊢ Membership.mem (List.cons x xs) a
      -/
      obtain rfl | ha := ha
        /-
          case inl.intro.intro.inl
          α : Type u_1
          n✝ : Nat
          xs✝ : List α
          a : α
          n : Nat
          xs : List α
          z : Sym α n
          hz : Membership.mem (List.sym n (List.cons a xs)) z
          ⊢ Membership.mem (List.cons a xs) a
        -/
      · simp
        /-
          🎉 no goals
        -/
        /-
          case inl.intro.intro.inr
          α : Type u_1
          n✝ : Nat
          xs✝ : List α
          a : α
          n : Nat
          x : α
          xs : List α
          z : Sym α n
          hz : Membership.mem (List.sym n (List.cons x xs)) z
          ha : Membership.mem z a
          ⊢ Membership.mem (List.cons x xs) a
        -/
      · exact mem_of_mem_of_mem_sym ha hz
        /-
          🎉 no goals
        -/
      /-
        case inr
        α : Type u_1
        n✝ : Nat
        xs✝ : List α
        a : α
        n : Nat
        x : α
        xs : List α
        z : Sym α (HAdd.hAdd n 1)
        ha : Membership.mem z a
        hz : Membership.mem (List.sym (HAdd.hAdd n 1) xs) z
        ⊢ Membership.mem (List.cons x xs) a
      -/
    · rw [mem_cons]
      /-
        case inr
        α : Type u_1
        n✝ : Nat
        xs✝ : List α
        a : α
        n : Nat
        x : α
        xs : List α
        z : Sym α (HAdd.hAdd n 1)
        ha : Membership.mem z a
        hz : Membership.mem (List.sym (HAdd.hAdd n 1) xs) z
        ⊢ Or (Eq a x) (Membership.mem xs a)
      -/
      right
      /-
        case inr.h
        α : Type u_1
        n✝ : Nat
        xs✝ : List α
        a : α
        n : Nat
        x : α
        xs : List α
        z : Sym α (HAdd.hAdd n 1)
        ha : Membership.mem z a
        hz : Membership.mem (List.sym (HAdd.hAdd n 1) xs) z
        ⊢ Membership.mem xs a
      -/
      exact mem_of_mem_of_mem_sym ha hz
      /-
        🎉 no goals
      -/


theorem first_mem_of_cons_mem_sym {xs : List α} {n : ℕ} {a : α} {z : Sym α n}
    (h : a ::ₛ z ∈ xs.sym (n + 1)) : a ∈ xs :=
  mem_of_mem_of_mem_sym (Sym.mem_cons_self a z) h


protected theorem Nodup.sym (n : ℕ) {xs : List α} (h : xs.Nodup) : (xs.sym n).Nodup :=
  match n, xs with
               /-
                 α : Type u_1
                 n : Nat
                 xs x✝ : List α
                 h : x✝.Nodup
                 ⊢ (List.sym 0 x✝).Nodup
               -/
  | 0, _ => by simp [List.sym]
               /-
                 🎉 no goals
               -/
                    /-
                      α : Type u_1
                      n✝ : Nat
                      xs : List α
                      n : Nat
                      h : List.nil.Nodup
                      ⊢ (List.sym (HAdd.hAdd n 1) List.nil).Nodup
                    -/
  | n + 1, [] => by simp [List.sym]
                    /-
                      🎉 no goals
                    -/
  | n + 1, x :: xs => by
    /-
      α : Type u_1
      n✝ : Nat
      xs✝ : List α
      n : Nat
      x : α
      xs : List α
      h : (List.cons x xs).Nodup
      ⊢ (List.sym (HAdd.hAdd n 1) (List.cons x xs)).Nodup
    -/
    rw [List.sym]
    /-
      α : Type u_1
      n✝ : Nat
      xs✝ : List α
      n : Nat
      x : α
      xs : List α
      h : (List.cons x xs).Nodup
      ⊢ (HAppend.hAppend (List.map (fun p => Sym.cons x p) (List.sym n (List.cons x  …
    -/
    refine Nodup.append (Nodup.map ?inj (Nodup.sym n h)) (Nodup.sym (n + 1) h.of_cons) ?disj
    case inj =>
      intro z z'
      simp
    case disj =>
      intro z hz hz'
      rw [mem_map] at hz
      obtain ⟨z, _hz, rfl⟩ := hz
      have := first_mem_of_cons_mem_sym hz'
      simp only [nodup_cons, this, not_true_eq_false, false_and] at h


theorem length_sym {n : ℕ} {xs : List α} :
    (xs.sym n).length = Nat.multichoose xs.length n :=
  match n, xs with
               /-
                 α : Type u_1
                 n : Nat
                 xs x✝ : List α
                 ⊢ Eq (List.sym 0 x✝).length (x✝.length.multichoose 0)
               -/
  | 0, _ => by rw [List.sym, Nat.multichoose]; rfl
                                               /-
                                                 🎉 no goals
                                               -/
                    /-
                      α : Type u_1
                      n✝ : Nat
                      xs : List α
                      n : Nat
                      ⊢ Eq (List.sym (HAdd.hAdd n 1) List.nil).length (List.nil.length.multichoose ( …
                    -/
  | n + 1, [] => by simp [List.sym]
                    /-
                      🎉 no goals
                    -/
  | n + 1, x :: xs => by
    /-
      α : Type u_1
      n✝ : Nat
      xs✝ : List α
      n : Nat
      x : α
      xs : List α
      ⊢ Eq (List.sym (HAdd.hAdd n 1) (List.cons x xs)).length ((List.cons x xs).leng …
    -/
    rw [List.sym, length_append, length_map, length_cons]
    /-
      α : Type u_1
      n✝ : Nat
      xs✝ : List α
      n : Nat
      x : α
      xs : List α
      ⊢ Eq (HAdd.hAdd (List.sym n (List.cons x xs)).length (List.sym (HAdd.hAdd n 1) …
    -/
    rw [@length_sym n (x :: xs), @length_sym (n + 1) xs]
    /-
      α : Type u_1
      n✝ : Nat
      xs✝ : List α
      n : Nat
      x : α
      xs : List α
      ⊢ Eq (HAdd.hAdd ((List.cons x xs).length.multichoose n) (xs.length.multichoose …
    -/
    rw [Nat.multichoose_succ_succ, length_cons, add_comm]
    /-
      🎉 no goals
    -/


