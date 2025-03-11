local notation a " * " b => op a b


/-- `fold op b f s` folds the commutative associative operation `op` over the
  `f`-image of `s`, i.e. `fold (+) b f {1,2,3} = f 1 + f 2 + f 3 + b`. -/
def fold (b : β) (f : α → β) (s : Finset α) : β :=
  (s.1.map f).fold op b


@[simp]
theorem fold_empty : (∅ : Finset α).fold op b f = b :=
  rfl


@[simp]
theorem fold_cons (h : a ∉ s) : (cons a s h).fold op b f = f a * s.fold op b f := by
  /-
    α : Type u_1
    β : Type u_2
    op : β → β → β
    hc : Std.Commutative op
    ha : Std.Associative op
    f : α → β
    b : β
    s : Finset α
    a : α
    h : Not (Membership.mem s a)
    ⊢ Eq (Finset.fold op b f (Finset.cons a s h)) (op (f a) (Finset.fold op b f s))
  -/
  dsimp only [fold]
  /-
    α : Type u_1
    β : Type u_2
    op : β → β → β
    hc : Std.Commutative op
    ha : Std.Associative op
    f : α → β
    b : β
    s : Finset α
    a : α
    h : Not (Membership.mem s a)
    ⊢ Eq (Multiset.fold op b (Multiset.map f (Finset.cons a s h).val)) (op (f a) ( …
  -/
  rw [cons_val, Multiset.map_cons, fold_cons_left]
  /-
    🎉 no goals
  -/


@[simp]
theorem fold_insert [DecidableEq α] (h : a ∉ s) :
    (insert a s).fold op b f = f a * s.fold op b f := by
  /-
    α : Type u_1
    β : Type u_2
    op : β → β → β
    hc : Std.Commutative op
    ha : Std.Associative op
    f : α → β
    b : β
    s : Finset α
    a : α
    inst✝ : DecidableEq α
    h : Not (Membership.mem s a)
    ⊢ Eq (Finset.fold op b f (Insert.insert a s)) (op (f a) (Finset.fold op b f s))
  -/
  unfold fold
  /-
    α : Type u_1
    β : Type u_2
    op : β → β → β
    hc : Std.Commutative op
    ha : Std.Associative op
    f : α → β
    b : β
    s : Finset α
    a : α
    inst✝ : DecidableEq α
    h : Not (Membership.mem s a)
    ⊢ Eq (Multiset.fold op b (Multiset.map f (Insert.insert a s).val)) (op (f a) ( …
  -/
  rw [insert_val, ndinsert_of_not_mem h, Multiset.map_cons, fold_cons_left]
  /-
    🎉 no goals
  -/


@[simp]
theorem fold_singleton : ({a} : Finset α).fold op b f = f a * b :=
  rfl


@[simp]
theorem fold_map {g : γ ↪ α} {s : Finset γ} : (s.map g).fold op b f = s.fold op b (f ∘ g) := by
  /-
    α : Type u_1
    β : Type u_2
    γ : Type u_3
    op : β → β → β
    hc : Std.Commutative op
    ha : Std.Associative op
    f : α → β
    b : β
    g : Function.Embedding γ α
    s : Finset γ
    ⊢ Eq (Finset.fold op b f (Finset.map g s)) (Finset.fold op b (Function.comp f  …
  -/
  simp only [fold, map, Multiset.map_map]
  /-
    🎉 no goals
  -/


@[simp]
theorem fold_image [DecidableEq α] {g : γ → α} {s : Finset γ}
    (H : ∀ x ∈ s, ∀ y ∈ s, g x = g y → x = y) : (s.image g).fold op b f = s.fold op b (f ∘ g) := by
  /-
    α : Type u_1
    β : Type u_2
    γ : Type u_3
    op : β → β → β
    hc : Std.Commutative op
    ha : Std.Associative op
    f : α → β
    b : β
    inst✝ : DecidableEq α
    g : γ → α
    s : Finset γ
    H : ∀ (x : γ), Membership.mem s x → ∀ (y : γ), Membership.mem s y → Eq (g x) ( …
    ⊢ Eq (Finset.fold op b f (Finset.image g s)) (Finset.fold op b (Function.comp  …
  -/
  simp only [fold, image_val_of_injOn H, Multiset.map_map]
  /-
    🎉 no goals
  -/


@[congr]
theorem fold_congr {g : α → β} (H : ∀ x ∈ s, f x = g x) : s.fold op b f = s.fold op b g := by
  /-
    α : Type u_1
    β : Type u_2
    op : β → β → β
    hc : Std.Commutative op
    ha : Std.Associative op
    f : α → β
    b : β
    s : Finset α
    g : α → β
    H : ∀ (x : α), Membership.mem s x → Eq (f x) (g x)
    ⊢ Eq (Finset.fold op b f s) (Finset.fold op b g s)
  -/
  rw [fold, fold, map_congr rfl H]
  /-
    🎉 no goals
  -/


theorem fold_op_distrib {f g : α → β} {b₁ b₂ : β} :
    (s.fold op (b₁ * b₂) fun x => f x * g x) = s.fold op b₁ f * s.fold op b₂ g := by
  /-
    α : Type u_1
    β : Type u_2
    op : β → β → β
    hc : Std.Commutative op
    ha : Std.Associative op
    s : Finset α
    f g : α → β
    b₁ b₂ : β
    ⊢ Eq (Finset.fold op (op b₁ b₂) (fun x => op (f x) (g x)) s) (op (Finset.fold  …
  -/
  simp only [fold, fold_distrib]
  /-
    🎉 no goals
  -/


theorem fold_const [hd : Decidable (s = ∅)] (c : β) (h : op c (op b c) = op b c) :
    Finset.fold op b (fun _ => c) s = if s = ∅ then b else op b c := by
  classical
    induction' s using Finset.induction_on with x s hx IH generalizing hd
    · simp
    · simp only [Finset.fold_insert hx, IH, if_false, Finset.insert_ne_empty]
      split_ifs
      · rw [hc.comm]
      · exact h


theorem fold_hom {op' : γ → γ → γ} [Std.Commutative op'] [Std.Associative op'] {m : β → γ}
    (hm : ∀ x y, m (op x y) = op' (m x) (m y)) :
    (s.fold op' (m b) fun x => m (f x)) = m (s.fold op b f) := by
  /-
    α : Type u_1
    β : Type u_2
    γ : Type u_3
    op : β → β → β
    hc : Std.Commutative op
    ha : Std.Associative op
    f : α → β
    b : β
    s : Finset α
    op' : γ → γ → γ
    inst✝¹ : Std.Commutative op'
    inst✝ : Std.Associative op'
    m : β → γ
    hm : ∀ (x y : β), Eq (m (op x y)) (op' (m x) (m y))
    ⊢ Eq (Finset.fold op' (m b) (fun x => m (f x)) s) (m (Finset.fold op b f s))
  -/
  rw [fold, fold, ← Multiset.fold_hom op hm, Multiset.map_map]
  /-
    α : Type u_1
    β : Type u_2
    γ : Type u_3
    op : β → β → β
    hc : Std.Commutative op
    ha : Std.Associative op
    f : α → β
    b : β
    s : Finset α
    op' : γ → γ → γ
    inst✝¹ : Std.Commutative op'
    inst✝ : Std.Associative op'
    m : β → γ
    hm : ∀ (x y : β), Eq (m (op x y)) (op' (m x) (m y))
    ⊢ Eq (Multiset.fold op' (m b) (Multiset.map (fun x => m (f x)) s.val)) (Multis …
  -/
  simp only [Function.comp_apply]
  /-
    🎉 no goals
  -/


theorem fold_disjUnion {s₁ s₂ : Finset α} {b₁ b₂ : β} (h) :
    (s₁.disjUnion s₂ h).fold op (b₁ * b₂) f = s₁.fold op b₁ f * s₂.fold op b₂ f :=
  (congr_arg _ <| Multiset.map_add _ _ _).trans (Multiset.fold_add _ _ _ _ _)


theorem fold_disjiUnion {ι : Type*} {s : Finset ι} {t : ι → Finset α} {b : ι → β} {b₀ : β} (h) :
    (s.disjiUnion t h).fold op (s.fold op b₀ b) f = s.fold op b₀ fun i => (t i).fold op (b i) f :=
  (congr_arg _ <| Multiset.map_bind _ _ _).trans (Multiset.fold_bind _ _ _ _ _)


theorem fold_union_inter [DecidableEq α] {s₁ s₂ : Finset α} {b₁ b₂ : β} :
    ((s₁ ∪ s₂).fold op b₁ f * (s₁ ∩ s₂).fold op b₂ f) = s₁.fold op b₂ f * s₂.fold op b₁ f := by
  /-
    α : Type u_1
    β : Type u_2
    op : β → β → β
    hc : Std.Commutative op
    ha : Std.Associative op
    f : α → β
    inst✝ : DecidableEq α
    s₁ s₂ : Finset α
    b₁ b₂ : β
    ⊢ Eq (op (Finset.fold op b₁ f (Union.union s₁ s₂)) (Finset.fold op b₂ f (Inter …
  -/
  unfold fold
  rw [← fold_add op, ← Multiset.map_add, union_val, inter_val, union_add_inter, Multiset.map_add,
    hc.comm, fold_add]


@[simp]
theorem fold_insert_idem [DecidableEq α] [hi : Std.IdempotentOp op] :
    (insert a s).fold op b f = f a * s.fold op b f := by
  /-
    α : Type u_1
    β : Type u_2
    op : β → β → β
    hc : Std.Commutative op
    ha : Std.Associative op
    f : α → β
    b : β
    s : Finset α
    a : α
    inst✝ : DecidableEq α
    hi : Std.IdempotentOp op
    ⊢ Eq (Finset.fold op b f (Insert.insert a s)) (op (f a) (Finset.fold op b f s))
  -/
  by_cases h : a ∈ s
    /-
      case pos
      α : Type u_1
      β : Type u_2
      op : β → β → β
      hc : Std.Commutative op
      ha : Std.Associative op
      f : α → β
      b : β
      s : Finset α
      a : α
      inst✝ : DecidableEq α
      hi : Std.IdempotentOp op
      h : Membership.mem s a
      ⊢ Eq (Finset.fold op b f (Insert.insert a s)) (op (f a) (Finset.fold op b f s))
    -/
  · rw [← insert_erase h]
    /-
      case pos
      α : Type u_1
      β : Type u_2
      op : β → β → β
      hc : Std.Commutative op
      ha : Std.Associative op
      f : α → β
      b : β
      s : Finset α
      a : α
      inst✝ : DecidableEq α
      hi : Std.IdempotentOp op
      h : Membership.mem s a
      ⊢ Eq (Finset.fold op b f (Insert.insert a (Insert.insert a (s.erase a)))) (op  …
    -/
    simp [← ha.assoc, hi.idempotent]
    /-
      🎉 no goals
    -/
    /-
      case neg
      α : Type u_1
      β : Type u_2
      op : β → β → β
      hc : Std.Commutative op
      ha : Std.Associative op
      f : α → β
      b : β
      s : Finset α
      a : α
      inst✝ : DecidableEq α
      hi : Std.IdempotentOp op
      h : Not (Membership.mem s a)
      ⊢ Eq (Finset.fold op b f (Insert.insert a s)) (op (f a) (Finset.fold op b f s))
    -/
  · apply fold_insert h
    /-
      🎉 no goals
    -/


theorem fold_image_idem [DecidableEq α] {g : γ → α} {s : Finset γ} [hi : Std.IdempotentOp op] :
    (image g s).fold op b f = s.fold op b (f ∘ g) := by
  /-
    α : Type u_1
    β : Type u_2
    γ : Type u_3
    op : β → β → β
    hc : Std.Commutative op
    ha : Std.Associative op
    f : α → β
    b : β
    inst✝ : DecidableEq α
    g : γ → α
    s : Finset γ
    hi : Std.IdempotentOp op
    ⊢ Eq (Finset.fold op b f (Finset.image g s)) (Finset.fold op b (Function.comp  …
  -/
  induction' s using Finset.cons_induction with x xs hx ih
    /-
      case empty
      α : Type u_1
      β : Type u_2
      γ : Type u_3
      op : β → β → β
      hc : Std.Commutative op
      ha : Std.Associative op
      f : α → β
      b : β
      inst✝ : DecidableEq α
      g : γ → α
      hi : Std.IdempotentOp op
      ⊢ Eq (Finset.fold op b f (Finset.image g EmptyCollection.emptyCollection)) (Fi …
    -/
  · rw [fold_empty, image_empty, fold_empty]
    /-
      🎉 no goals
    -/
    /-
      case cons
      α : Type u_1
      β : Type u_2
      γ : Type u_3
      op : β → β → β
      hc : Std.Commutative op
      ha : Std.Associative op
      f : α → β
      b : β
      inst✝ : DecidableEq α
      g : γ → α
      hi : Std.IdempotentOp op
      x : γ
      xs : Finset γ
      hx : Not (Membership.mem xs x)
      ih : Eq (Finset.fold op b f (Finset.image g xs)) (Finset.fold op b (Function.c …
      ⊢ Eq (Finset.fold op b f (Finset.image g (Finset.cons x xs hx))) (Finset.fold  …
    -/
  · haveI := Classical.decEq γ
    /-
      case cons
      α : Type u_1
      β : Type u_2
      γ : Type u_3
      op : β → β → β
      hc : Std.Commutative op
      ha : Std.Associative op
      f : α → β
      b : β
      inst✝ : DecidableEq α
      g : γ → α
      hi : Std.IdempotentOp op
      x : γ
      xs : Finset γ
      hx : Not (Membership.mem xs x)
      ih : Eq (Finset.fold op b f (Finset.image g xs)) (Finset.fold op b (Function.c …
      this : DecidableEq γ
      ⊢ Eq (Finset.fold op b f (Finset.image g (Finset.cons x xs hx))) (Finset.fold  …
    -/
    rw [fold_cons, cons_eq_insert, image_insert, fold_insert_idem, ih]
    /-
      case cons
      α : Type u_1
      β : Type u_2
      γ : Type u_3
      op : β → β → β
      hc : Std.Commutative op
      ha : Std.Associative op
      f : α → β
      b : β
      inst✝ : DecidableEq α
      g : γ → α
      hi : Std.IdempotentOp op
      x : γ
      xs : Finset γ
      hx : Not (Membership.mem xs x)
      ih : Eq (Finset.fold op b f (Finset.image g xs)) (Finset.fold op b (Function.c …
      this : DecidableEq γ
      ⊢ Eq (op (f (g x)) (Finset.fold op b (Function.comp f g) xs)) (op (Function.co …
    -/
    simp only [Function.comp_apply]
    /-
      🎉 no goals
    -/


/-- A stronger version of `Finset.fold_ite`, but relies on
an explicit proof of idempotency on the seed element, rather
than relying on typeclass idempotency over the whole type. -/
theorem fold_ite' {g : α → β} (hb : op b b = b) (p : α → Prop) [DecidablePred p] :
    Finset.fold op b (fun i => ite (p i) (f i) (g i)) s =
      op (Finset.fold op b f (s.filter p)) (Finset.fold op b g (s.filter fun i => ¬p i)) := by
  classical
    induction' s using Finset.induction_on with x s hx IH
    · simp [hb]
    · simp only [Finset.fold_insert hx]
      split_ifs with h
      · have : x ∉ Finset.filter p s := by simp [hx]
        simp [Finset.filter_insert, h, Finset.fold_insert this, ha.assoc, IH]
      · have : x ∉ Finset.filter (fun i => ¬ p i) s := by simp [hx]
        simp [Finset.filter_insert, h, Finset.fold_insert this, IH, ← ha.assoc, hc.comm]


/-- A weaker version of `Finset.fold_ite'`,
relying on typeclass idempotency over the whole type,
instead of solely on the seed element.
However, this is easier to use because it does not generate side goals. -/
theorem fold_ite [Std.IdempotentOp op] {g : α → β} (p : α → Prop) [DecidablePred p] :
    Finset.fold op b (fun i => ite (p i) (f i) (g i)) s =
      op (Finset.fold op b f (s.filter p)) (Finset.fold op b g (s.filter fun i => ¬p i)) :=
  fold_ite' (Std.IdempotentOp.idempotent _) _


theorem fold_op_rel_iff_and {r : β → β → Prop} (hr : ∀ {x y z}, r x (op y z) ↔ r x y ∧ r x z)
    {c : β} : r c (s.fold op b f) ↔ r c b ∧ ∀ x ∈ s, r c (f x) := by
  classical
    induction' s using Finset.induction_on with a s ha IH
    · simp
    rw [Finset.fold_insert ha, hr, IH, ← and_assoc, @and_comm (r c (f a)), and_assoc]
    apply and_congr Iff.rfl
    constructor
    · rintro ⟨h₁, h₂⟩
      intro b hb
      rw [Finset.mem_insert] at hb
      rcases hb with (rfl | hb) <;> solve_by_elim
    · intro h
      constructor
      · exact h a (Finset.mem_insert_self _ _)
      · exact fun b hb => h b <| Finset.mem_insert_of_mem hb


theorem fold_op_rel_iff_or {r : β → β → Prop} (hr : ∀ {x y z}, r x (op y z) ↔ r x y ∨ r x z)
    {c : β} : r c (s.fold op b f) ↔ r c b ∨ ∃ x ∈ s, r c (f x) := by
  classical
    induction' s using Finset.induction_on with a s ha IH
    · simp
    rw [Finset.fold_insert ha, hr, IH, ← or_assoc, @or_comm (r c (f a)), or_assoc]
    apply or_congr Iff.rfl
    constructor
    · rintro (h₁ | ⟨x, hx, h₂⟩)
      · use a
        simp [h₁]
      · refine ⟨x, by simp [hx], h₂⟩
    · rintro ⟨x, hx, h⟩
      exact (mem_insert.mp hx).imp (fun hx => by rwa [hx] at h) (fun hx => ⟨x, hx, h⟩)


@[simp]
theorem fold_union_empty_singleton [DecidableEq α] (s : Finset α) :
    Finset.fold (· ∪ ·) ∅ singleton s = s := by
  /-
    α : Type u_1
    inst✝ : DecidableEq α
    s : Finset α
    ⊢ Eq (Finset.fold (fun x1 x2 => Union.union x1 x2) EmptyCollection.emptyCollec …
  -/
  induction' s using Finset.induction_on with a s has ih
    /-
      case empty
      α : Type u_1
      inst✝ : DecidableEq α
      ⊢ Eq (Finset.fold (fun x1 x2 => Union.union x1 x2) EmptyCollection.emptyCollec …
    -/
  · simp only [fold_empty]
    /-
      🎉 no goals
    -/
    /-
      case insert
      α : Type u_1
      inst✝ : DecidableEq α
      a : α
      s : Finset α
      has : Not (Membership.mem s a)
      ih : Eq (Finset.fold (fun x1 x2 => Union.union x1 x2) EmptyCollection.emptyCol …
      ⊢ Eq (Finset.fold (fun x1 x2 => Union.union x1 x2) EmptyCollection.emptyCollec …
    -/
  · rw [fold_insert has, ih, insert_eq]
    /-
      🎉 no goals
    -/


theorem fold_sup_bot_singleton [DecidableEq α] (s : Finset α) :
    Finset.fold (· ⊔ ·) ⊥ singleton s = s :=
  fold_union_empty_singleton s


theorem le_fold_min : c ≤ s.fold min b f ↔ c ≤ b ∧ ∀ x ∈ s, c ≤ f x :=
  fold_op_rel_iff_and le_min_iff


theorem fold_min_le : s.fold min b f ≤ c ↔ b ≤ c ∨ ∃ x ∈ s, f x ≤ c := by
  /-
    α : Type u_1
    β : Type u_2
    f : α → β
    b : β
    s : Finset α
    inst✝ : LinearOrder β
    c : β
    ⊢ Iff (LE.le (Finset.fold Min.min b f s) c) (Or (LE.le b c) (Exists fun x => A …
  -/
  show _ ≥ _ ↔ _
  /-
    α : Type u_1
    β : Type u_2
    f : α → β
    b : β
    s : Finset α
    inst✝ : LinearOrder β
    c : β
    ⊢ Iff (GE.ge c (Finset.fold Min.min b f s)) (Or (LE.le b c) (Exists fun x => A …
  -/
  apply fold_op_rel_iff_or
  /-
    case hr
    α : Type u_1
    β : Type u_2
    f : α → β
    b : β
    s : Finset α
    inst✝ : LinearOrder β
    c : β
    ⊢ ∀ {x y z : β}, Iff (GE.ge x (Min.min y z)) (Or (GE.ge x y) (GE.ge x z))
  -/
  intro x y z
  /-
    case hr
    α : Type u_1
    β : Type u_2
    f : α → β
    b : β
    s : Finset α
    inst✝ : LinearOrder β
    c x y z : β
    ⊢ Iff (GE.ge x (Min.min y z)) (Or (GE.ge x y) (GE.ge x z))
  -/
  show _ ≤ _ ↔ _
  /-
    case hr
    α : Type u_1
    β : Type u_2
    f : α → β
    b : β
    s : Finset α
    inst✝ : LinearOrder β
    c x y z : β
    ⊢ Iff (LE.le (Min.min y z) x) (Or (GE.ge x y) (GE.ge x z))
  -/
  exact min_le_iff
  /-
    🎉 no goals
  -/


theorem lt_fold_min : c < s.fold min b f ↔ c < b ∧ ∀ x ∈ s, c < f x :=
  fold_op_rel_iff_and lt_min_iff


theorem fold_min_lt : s.fold min b f < c ↔ b < c ∨ ∃ x ∈ s, f x < c := by
  /-
    α : Type u_1
    β : Type u_2
    f : α → β
    b : β
    s : Finset α
    inst✝ : LinearOrder β
    c : β
    ⊢ Iff (LT.lt (Finset.fold Min.min b f s) c) (Or (LT.lt b c) (Exists fun x => A …
  -/
  show _ > _ ↔ _
  /-
    α : Type u_1
    β : Type u_2
    f : α → β
    b : β
    s : Finset α
    inst✝ : LinearOrder β
    c : β
    ⊢ Iff (GT.gt c (Finset.fold Min.min b f s)) (Or (LT.lt b c) (Exists fun x => A …
  -/
  apply fold_op_rel_iff_or
  /-
    case hr
    α : Type u_1
    β : Type u_2
    f : α → β
    b : β
    s : Finset α
    inst✝ : LinearOrder β
    c : β
    ⊢ ∀ {x y z : β}, Iff (GT.gt x (Min.min y z)) (Or (GT.gt x y) (GT.gt x z))
  -/
  intro x y z
  /-
    case hr
    α : Type u_1
    β : Type u_2
    f : α → β
    b : β
    s : Finset α
    inst✝ : LinearOrder β
    c x y z : β
    ⊢ Iff (GT.gt x (Min.min y z)) (Or (GT.gt x y) (GT.gt x z))
  -/
  show _ < _ ↔ _
  /-
    case hr
    α : Type u_1
    β : Type u_2
    f : α → β
    b : β
    s : Finset α
    inst✝ : LinearOrder β
    c x y z : β
    ⊢ Iff (LT.lt (Min.min y z) x) (Or (GT.gt x y) (GT.gt x z))
  -/
  exact min_lt_iff
  /-
    🎉 no goals
  -/


theorem fold_max_le : s.fold max b f ≤ c ↔ b ≤ c ∧ ∀ x ∈ s, f x ≤ c := by
  /-
    α : Type u_1
    β : Type u_2
    f : α → β
    b : β
    s : Finset α
    inst✝ : LinearOrder β
    c : β
    ⊢ Iff (LE.le (Finset.fold Max.max b f s) c) (And (LE.le b c) (∀ (x : α), Membe …
  -/
  show _ ≥ _ ↔ _
  /-
    α : Type u_1
    β : Type u_2
    f : α → β
    b : β
    s : Finset α
    inst✝ : LinearOrder β
    c : β
    ⊢ Iff (GE.ge c (Finset.fold Max.max b f s)) (And (LE.le b c) (∀ (x : α), Membe …
  -/
  apply fold_op_rel_iff_and
  /-
    case hr
    α : Type u_1
    β : Type u_2
    f : α → β
    b : β
    s : Finset α
    inst✝ : LinearOrder β
    c : β
    ⊢ ∀ {x y z : β}, Iff (GE.ge x (Max.max y z)) (And (GE.ge x y) (GE.ge x z))
  -/
  intro x y z
  /-
    case hr
    α : Type u_1
    β : Type u_2
    f : α → β
    b : β
    s : Finset α
    inst✝ : LinearOrder β
    c x y z : β
    ⊢ Iff (GE.ge x (Max.max y z)) (And (GE.ge x y) (GE.ge x z))
  -/
  show _ ≤ _ ↔ _
  /-
    case hr
    α : Type u_1
    β : Type u_2
    f : α → β
    b : β
    s : Finset α
    inst✝ : LinearOrder β
    c x y z : β
    ⊢ Iff (LE.le (Max.max y z) x) (And (GE.ge x y) (GE.ge x z))
  -/
  exact max_le_iff
  /-
    🎉 no goals
  -/


theorem le_fold_max : c ≤ s.fold max b f ↔ c ≤ b ∨ ∃ x ∈ s, c ≤ f x :=
  fold_op_rel_iff_or le_max_iff


theorem fold_max_lt : s.fold max b f < c ↔ b < c ∧ ∀ x ∈ s, f x < c := by
  /-
    α : Type u_1
    β : Type u_2
    f : α → β
    b : β
    s : Finset α
    inst✝ : LinearOrder β
    c : β
    ⊢ Iff (LT.lt (Finset.fold Max.max b f s) c) (And (LT.lt b c) (∀ (x : α), Membe …
  -/
  show _ > _ ↔ _
  /-
    α : Type u_1
    β : Type u_2
    f : α → β
    b : β
    s : Finset α
    inst✝ : LinearOrder β
    c : β
    ⊢ Iff (GT.gt c (Finset.fold Max.max b f s)) (And (LT.lt b c) (∀ (x : α), Membe …
  -/
  apply fold_op_rel_iff_and
  /-
    case hr
    α : Type u_1
    β : Type u_2
    f : α → β
    b : β
    s : Finset α
    inst✝ : LinearOrder β
    c : β
    ⊢ ∀ {x y z : β}, Iff (GT.gt x (Max.max y z)) (And (GT.gt x y) (GT.gt x z))
  -/
  intro x y z
  /-
    case hr
    α : Type u_1
    β : Type u_2
    f : α → β
    b : β
    s : Finset α
    inst✝ : LinearOrder β
    c x y z : β
    ⊢ Iff (GT.gt x (Max.max y z)) (And (GT.gt x y) (GT.gt x z))
  -/
  show _ < _ ↔ _
  /-
    case hr
    α : Type u_1
    β : Type u_2
    f : α → β
    b : β
    s : Finset α
    inst✝ : LinearOrder β
    c x y z : β
    ⊢ Iff (LT.lt (Max.max y z) x) (And (GT.gt x y) (GT.gt x z))
  -/
  exact max_lt_iff
  /-
    🎉 no goals
  -/


theorem lt_fold_max : c < s.fold max b f ↔ c < b ∨ ∃ x ∈ s, c < f x :=
  fold_op_rel_iff_or lt_max_iff


