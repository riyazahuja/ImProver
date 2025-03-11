/-- For `n ≤ m`, `(n, m)` is in the reflexive-transitive closure of `~` if `i ~ succ i`
  for all `i` between `n` and `m`. -/
theorem reflTransGen_of_succ_of_le (r : α → α → Prop) {n m : α} (h : ∀ i ∈ Ico n m, r i (succ i))
    (hnm : n ≤ m) : ReflTransGen r n m := by
  /-
    α : Type u_1
    inst✝² : PartialOrder α
    inst✝¹ : SuccOrder α
    inst✝ : IsSuccArchimedean α
    r : α → α → Prop
    n m : α
    h : ∀ (i : α), Membership.mem (Set.Ico n m) i → r i (Order.succ i)
    hnm : LE.le n m
    ⊢ Relation.ReflTransGen r n m
  -/
  revert h; refine Succ.rec ?_ ?_ hnm
    /-
      case refine_1
      α : Type u_1
      inst✝² : PartialOrder α
      inst✝¹ : SuccOrder α
      inst✝ : IsSuccArchimedean α
      r : α → α → Prop
      n m : α
      hnm : LE.le n m
      ⊢ (∀ (i : α), Membership.mem (Set.Ico n n) i → r i (Order.succ i)) → Relation. …
    -/
  · intro _
    /-
      case refine_1
      α : Type u_1
      inst✝² : PartialOrder α
      inst✝¹ : SuccOrder α
      inst✝ : IsSuccArchimedean α
      r : α → α → Prop
      n m : α
      hnm : LE.le n m
      h✝ : ∀ (i : α), Membership.mem (Set.Ico n n) i → r i (Order.succ i)
      ⊢ Relation.ReflTransGen r n n
    -/
    exact ReflTransGen.refl
    /-
      🎉 no goals
    -/
    /-
      case refine_2
      α : Type u_1
      inst✝² : PartialOrder α
      inst✝¹ : SuccOrder α
      inst✝ : IsSuccArchimedean α
      r : α → α → Prop
      n m : α
      hnm : LE.le n m
      ⊢ ∀ (n_1 : α), LE.le n n_1 → ((∀ (i : α), Membership.mem (Set.Ico n n_1) i → r …
    -/
  · intro m hnm ih h
    /-
      case refine_2
      α : Type u_1
      inst✝² : PartialOrder α
      inst✝¹ : SuccOrder α
      inst✝ : IsSuccArchimedean α
      r : α → α → Prop
      n m✝ : α
      hnm✝ : LE.le n m✝
      m : α
      hnm : LE.le n m
      ih : (∀ (i : α), Membership.mem (Set.Ico n m) i → r i (Order.succ i)) → Relati …
      h : ∀ (i : α), Membership.mem (Set.Ico n (Order.succ m)) i → r i (Order.succ i)
      ⊢ Relation.ReflTransGen r n (Order.succ m)
    -/
    have : ReflTransGen r n m := ih fun i hi => h i ⟨hi.1, hi.2.trans_le <| le_succ m⟩
    /-
      case refine_2
      α : Type u_1
      inst✝² : PartialOrder α
      inst✝¹ : SuccOrder α
      inst✝ : IsSuccArchimedean α
      r : α → α → Prop
      n m✝ : α
      hnm✝ : LE.le n m✝
      m : α
      hnm : LE.le n m
      ih : (∀ (i : α), Membership.mem (Set.Ico n m) i → r i (Order.succ i)) → Relati …
      h : ∀ (i : α), Membership.mem (Set.Ico n (Order.succ m)) i → r i (Order.succ i)
      this : Relation.ReflTransGen r n m
      ⊢ Relation.ReflTransGen r n (Order.succ m)
    -/
    rcases (le_succ m).eq_or_lt with hm | hm
      /-
        case refine_2.inl
        α : Type u_1
        inst✝² : PartialOrder α
        inst✝¹ : SuccOrder α
        inst✝ : IsSuccArchimedean α
        r : α → α → Prop
        n m✝ : α
        hnm✝ : LE.le n m✝
        m : α
        hnm : LE.le n m
        ih : (∀ (i : α), Membership.mem (Set.Ico n m) i → r i (Order.succ i)) → Relati …
        h : ∀ (i : α), Membership.mem (Set.Ico n (Order.succ m)) i → r i (Order.succ i)
        this : Relation.ReflTransGen r n m
        hm : Eq m (Order.succ m)
        ⊢ Relation.ReflTransGen r n (Order.succ m)
      -/
    · rwa [← hm]
      /-
        🎉 no goals
      -/
    /-
      case refine_2.inr
      α : Type u_1
      inst✝² : PartialOrder α
      inst✝¹ : SuccOrder α
      inst✝ : IsSuccArchimedean α
      r : α → α → Prop
      n m✝ : α
      hnm✝ : LE.le n m✝
      m : α
      hnm : LE.le n m
      ih : (∀ (i : α), Membership.mem (Set.Ico n m) i → r i (Order.succ i)) → Relati …
      h : ∀ (i : α), Membership.mem (Set.Ico n (Order.succ m)) i → r i (Order.succ i)
      this : Relation.ReflTransGen r n m
      hm : LT.lt m (Order.succ m)
      ⊢ Relation.ReflTransGen r n (Order.succ m)
    -/
    exact this.tail (h m ⟨hnm, hm⟩)
    /-
      🎉 no goals
    -/


/-- For `m ≤ n`, `(n, m)` is in the reflexive-transitive closure of `~` if `succ i ~ i`
  for all `i` between `n` and `m`. -/
theorem reflTransGen_of_succ_of_ge (r : α → α → Prop) {n m : α} (h : ∀ i ∈ Ico m n, r (succ i) i)
    (hmn : m ≤ n) : ReflTransGen r n m := by
  /-
    α : Type u_1
    inst✝² : PartialOrder α
    inst✝¹ : SuccOrder α
    inst✝ : IsSuccArchimedean α
    r : α → α → Prop
    n m : α
    h : ∀ (i : α), Membership.mem (Set.Ico m n) i → r (Order.succ i) i
    hmn : LE.le m n
    ⊢ Relation.ReflTransGen r n m
  -/
  rw [← reflTransGen_swap]
  /-
    α : Type u_1
    inst✝² : PartialOrder α
    inst✝¹ : SuccOrder α
    inst✝ : IsSuccArchimedean α
    r : α → α → Prop
    n m : α
    h : ∀ (i : α), Membership.mem (Set.Ico m n) i → r (Order.succ i) i
    hmn : LE.le m n
    ⊢ Relation.ReflTransGen (Function.swap r) m n
  -/
  exact reflTransGen_of_succ_of_le (swap r) h hmn
  /-
    🎉 no goals
  -/


/-- For `n < m`, `(n, m)` is in the transitive closure of a relation `~` if `i ~ succ i`
  for all `i` between `n` and `m`. -/
theorem transGen_of_succ_of_lt (r : α → α → Prop) {n m : α} (h : ∀ i ∈ Ico n m, r i (succ i))
    (hnm : n < m) : TransGen r n m :=
  (reflTransGen_iff_eq_or_transGen.mp <| reflTransGen_of_succ_of_le r h hnm.le).resolve_left
    hnm.ne'


/-- For `m < n`, `(n, m)` is in the transitive closure of a relation `~` if `succ i ~ i`
  for all `i` between `n` and `m`. -/
theorem transGen_of_succ_of_gt (r : α → α → Prop) {n m : α} (h : ∀ i ∈ Ico m n, r (succ i) i)
    (hmn : m < n) : TransGen r n m :=
  (reflTransGen_iff_eq_or_transGen.mp <| reflTransGen_of_succ_of_ge r h hmn.le).resolve_left
    hmn.ne


/-- `(n, m)` is in the reflexive-transitive closure of `~` if `i ~ succ i` and `succ i ~ i`
  for all `i` between `n` and `m`. -/
theorem reflTransGen_of_succ (r : α → α → Prop) {n m : α} (h1 : ∀ i ∈ Ico n m, r i (succ i))
    (h2 : ∀ i ∈ Ico m n, r (succ i) i) : ReflTransGen r n m :=
  (le_total n m).elim (reflTransGen_of_succ_of_le r h1) <| reflTransGen_of_succ_of_ge r h2


/-- For `n ≠ m`,`(n, m)` is in the transitive closure of a relation `~` if `i ~ succ i` and
  `succ i ~ i` for all `i` between `n` and `m`. -/
theorem transGen_of_succ_of_ne (r : α → α → Prop) {n m : α} (h1 : ∀ i ∈ Ico n m, r i (succ i))
    (h2 : ∀ i ∈ Ico m n, r (succ i) i) (hnm : n ≠ m) : TransGen r n m :=
  (reflTransGen_iff_eq_or_transGen.mp (reflTransGen_of_succ r h1 h2)).resolve_left hnm.symm


/-- `(n, m)` is in the transitive closure of a reflexive relation `~` if `i ~ succ i` and
  `succ i ~ i` for all `i` between `n` and `m`. -/
theorem transGen_of_succ_of_reflexive (r : α → α → Prop) {n m : α} (hr : Reflexive r)
    (h1 : ∀ i ∈ Ico n m, r i (succ i)) (h2 : ∀ i ∈ Ico m n, r (succ i) i) : TransGen r n m := by
  /-
    α : Type u_1
    inst✝² : LinearOrder α
    inst✝¹ : SuccOrder α
    inst✝ : IsSuccArchimedean α
    r : α → α → Prop
    n m : α
    hr : Reflexive r
    h1 : ∀ (i : α), Membership.mem (Set.Ico n m) i → r i (Order.succ i)
    h2 : ∀ (i : α), Membership.mem (Set.Ico m n) i → r (Order.succ i) i
    ⊢ Relation.TransGen r n m
  -/
  rcases eq_or_ne m n with (rfl | hmn); · exact TransGen.single (hr m)
                                          /-
                                            🎉 no goals
                                          -/
  /-
    case inr
    α : Type u_1
    inst✝² : LinearOrder α
    inst✝¹ : SuccOrder α
    inst✝ : IsSuccArchimedean α
    r : α → α → Prop
    n m : α
    hr : Reflexive r
    h1 : ∀ (i : α), Membership.mem (Set.Ico n m) i → r i (Order.succ i)
    h2 : ∀ (i : α), Membership.mem (Set.Ico m n) i → r (Order.succ i) i
    hmn : Ne m n
    ⊢ Relation.TransGen r n m
  -/
  exact transGen_of_succ_of_ne r h1 h2 hmn.symm
  /-
    🎉 no goals
  -/


/-- For `m ≤ n`, `(n, m)` is in the reflexive-transitive closure of `~` if `i ~ pred i`
  for all `i` between `n` and `m`. -/
theorem reflTransGen_of_pred_of_ge (r : α → α → Prop) {n m : α} (h : ∀ i ∈ Ioc m n, r i (pred i))
    (hnm : m ≤ n) : ReflTransGen r n m :=
  reflTransGen_of_succ_of_le (α := αᵒᵈ) r (fun x hx => h x ⟨hx.2, hx.1⟩) hnm


/-- For `n ≤ m`, `(n, m)` is in the reflexive-transitive closure of `~` if `pred i ~ i`
  for all `i` between `n` and `m`. -/
theorem reflTransGen_of_pred_of_le (r : α → α → Prop) {n m : α} (h : ∀ i ∈ Ioc n m, r (pred i) i)
    (hmn : n ≤ m) : ReflTransGen r n m :=
  reflTransGen_of_succ_of_ge (α := αᵒᵈ) r (fun x hx => h x ⟨hx.2, hx.1⟩) hmn


/-- For `m < n`, `(n, m)` is in the transitive closure of a relation `~` for `n ≠ m` if `i ~ pred i`
  for all `i` between `n` and `m`. -/
theorem transGen_of_pred_of_gt (r : α → α → Prop) {n m : α} (h : ∀ i ∈ Ioc m n, r i (pred i))
    (hnm : m < n) : TransGen r n m :=
  transGen_of_succ_of_lt (α := αᵒᵈ) r (fun x hx => h x ⟨hx.2, hx.1⟩) hnm


/-- For `n < m`, `(n, m)` is in the transitive closure of a relation `~` for `n ≠ m` if `pred i ~ i`
  for all `i` between `n` and `m`. -/
theorem transGen_of_pred_of_lt (r : α → α → Prop) {n m : α} (h : ∀ i ∈ Ioc n m, r (pred i) i)
    (hmn : n < m) : TransGen r n m :=
  transGen_of_succ_of_gt (α := αᵒᵈ) r (fun x hx => h x ⟨hx.2, hx.1⟩) hmn


/-- `(n, m)` is in the reflexive-transitive closure of `~` if `i ~ pred i` and `pred i ~ i`
  for all `i` between `n` and `m`. -/
theorem reflTransGen_of_pred (r : α → α → Prop) {n m : α} (h1 : ∀ i ∈ Ioc m n, r i (pred i))
    (h2 : ∀ i ∈ Ioc n m, r (pred i) i) : ReflTransGen r n m :=
  reflTransGen_of_succ (α := αᵒᵈ) r (fun x hx => h1 x ⟨hx.2, hx.1⟩) fun x hx =>
    h2 x ⟨hx.2, hx.1⟩


/-- For `n ≠ m`, `(n, m)` is in the transitive closure of a relation `~` if `i ~ pred i` and
  `pred i ~ i` for all `i` between `n` and `m`. -/
theorem transGen_of_pred_of_ne (r : α → α → Prop) {n m : α} (h1 : ∀ i ∈ Ioc m n, r i (pred i))
    (h2 : ∀ i ∈ Ioc n m, r (pred i) i) (hnm : n ≠ m) : TransGen r n m :=
  transGen_of_succ_of_ne (α := αᵒᵈ) r (fun x hx => h1 x ⟨hx.2, hx.1⟩)
    (fun x hx => h2 x ⟨hx.2, hx.1⟩) hnm


/-- `(n, m)` is in the transitive closure of a reflexive relation `~` if `i ~ pred i` and
  `pred i ~ i` for all `i` between `n` and `m`. -/
theorem transGen_of_pred_of_reflexive (r : α → α → Prop) {n m : α} (hr : Reflexive r)
    (h1 : ∀ i ∈ Ioc m n, r i (pred i)) (h2 : ∀ i ∈ Ioc n m, r (pred i) i) : TransGen r n m :=
  transGen_of_succ_of_reflexive (α := αᵒᵈ) r hr (fun x hx => h1 x ⟨hx.2, hx.1⟩) fun x hx =>
    h2 x ⟨hx.2, hx.1⟩


