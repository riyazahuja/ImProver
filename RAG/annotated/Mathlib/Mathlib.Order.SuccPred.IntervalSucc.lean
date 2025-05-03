/-- If `α` is a linear archimedean succ order and `β` is a linear order, then for any monotone
function `f` and `m n : α`, the union of intervals `Set.Ioc (f i) (f (Order.succ i))`, `m ≤ i < n`,
is equal to `Set.Ioc (f m) (f n)` -/
theorem biUnion_Ico_Ioc_map_succ [SuccOrder α] [IsSuccArchimedean α] [LinearOrder β] {f : α → β}
    (hf : Monotone f) (m n : α) : ⋃ i ∈ Ico m n, Ioc (f i) (f (succ i)) = Ioc (f m) (f n) := by
  /-
    α : Type u_1
    β : Type u_2
    inst✝³ : LinearOrder α
    inst✝² : SuccOrder α
    inst✝¹ : IsSuccArchimedean α
    inst✝ : LinearOrder β
    f : α → β
    hf : Monotone f
    m n : α
    ⊢ Eq (Set.iUnion fun i => Set.iUnion fun h => Set.Ioc (f i) (f (Order.succ i)) …
  -/
  rcases le_total n m with hnm | hmn
    /-
      case inl
      α : Type u_1
      β : Type u_2
      inst✝³ : LinearOrder α
      inst✝² : SuccOrder α
      inst✝¹ : IsSuccArchimedean α
      inst✝ : LinearOrder β
      f : α → β
      hf : Monotone f
      m n : α
      hnm : LE.le n m
      ⊢ Eq (Set.iUnion fun i => Set.iUnion fun h => Set.Ioc (f i) (f (Order.succ i)) …
    -/
  · rw [Ico_eq_empty_of_le hnm, Ioc_eq_empty_of_le (hf hnm), biUnion_empty]
    /-
      🎉 no goals
    -/
    /-
      case inr
      α : Type u_1
      β : Type u_2
      inst✝³ : LinearOrder α
      inst✝² : SuccOrder α
      inst✝¹ : IsSuccArchimedean α
      inst✝ : LinearOrder β
      f : α → β
      hf : Monotone f
      m n : α
      hmn : LE.le m n
      ⊢ Eq (Set.iUnion fun i => Set.iUnion fun h => Set.Ioc (f i) (f (Order.succ i)) …
    -/
  · refine Succ.rec ?_ ?_ hmn
      /-
        case inr.refine_1
        α : Type u_1
        β : Type u_2
        inst✝³ : LinearOrder α
        inst✝² : SuccOrder α
        inst✝¹ : IsSuccArchimedean α
        inst✝ : LinearOrder β
        f : α → β
        hf : Monotone f
        m n : α
        hmn : LE.le m n
        ⊢ Eq (Set.iUnion fun i => Set.iUnion fun h => Set.Ioc (f i) (f (Order.succ i)) …
      -/
    · simp only [Ioc_self, Ico_self, biUnion_empty]
      /-
        🎉 no goals
      -/
      /-
        case inr.refine_2
        α : Type u_1
        β : Type u_2
        inst✝³ : LinearOrder α
        inst✝² : SuccOrder α
        inst✝¹ : IsSuccArchimedean α
        inst✝ : LinearOrder β
        f : α → β
        hf : Monotone f
        m n : α
        hmn : LE.le m n
        ⊢ ∀ (n : α), LE.le m n → Eq (Set.iUnion fun i => Set.iUnion fun h => Set.Ioc ( …
      -/
    · intro k hmk ihk
      /-
        case inr.refine_2
        α : Type u_1
        β : Type u_2
        inst✝³ : LinearOrder α
        inst✝² : SuccOrder α
        inst✝¹ : IsSuccArchimedean α
        inst✝ : LinearOrder β
        f : α → β
        hf : Monotone f
        m n : α
        hmn : LE.le m n
        k : α
        hmk : LE.le m k
        ihk : Eq (Set.iUnion fun i => Set.iUnion fun h => Set.Ioc (f i) (f (Order.succ …
        ⊢ Eq (Set.iUnion fun i => Set.iUnion fun h => Set.Ioc (f i) (f (Order.succ i)) …
      -/
      rw [← Ioc_union_Ioc_eq_Ioc (hf hmk) (hf <| le_succ _), union_comm, ← ihk]
      /-
        case inr.refine_2
        α : Type u_1
        β : Type u_2
        inst✝³ : LinearOrder α
        inst✝² : SuccOrder α
        inst✝¹ : IsSuccArchimedean α
        inst✝ : LinearOrder β
        f : α → β
        hf : Monotone f
        m n : α
        hmn : LE.le m n
        k : α
        hmk : LE.le m k
        ihk : Eq (Set.iUnion fun i => Set.iUnion fun h => Set.Ioc (f i) (f (Order.succ …
        ⊢ Eq (Set.iUnion fun i => Set.iUnion fun h => Set.Ioc (f i) (f (Order.succ i)) …
      -/
      by_cases hk : IsMax k
        /-
          case pos
          α : Type u_1
          β : Type u_2
          inst✝³ : LinearOrder α
          inst✝² : SuccOrder α
          inst✝¹ : IsSuccArchimedean α
          inst✝ : LinearOrder β
          f : α → β
          hf : Monotone f
          m n : α
          hmn : LE.le m n
          k : α
          hmk : LE.le m k
          ihk : Eq (Set.iUnion fun i => Set.iUnion fun h => Set.Ioc (f i) (f (Order.succ …
          hk : IsMax k
          ⊢ Eq (Set.iUnion fun i => Set.iUnion fun h => Set.Ioc (f i) (f (Order.succ i)) …
        -/
      · rw [hk.succ_eq, Ioc_self, empty_union]
        /-
          🎉 no goals
        -/
        /-
          case neg
          α : Type u_1
          β : Type u_2
          inst✝³ : LinearOrder α
          inst✝² : SuccOrder α
          inst✝¹ : IsSuccArchimedean α
          inst✝ : LinearOrder β
          f : α → β
          hf : Monotone f
          m n : α
          hmn : LE.le m n
          k : α
          hmk : LE.le m k
          ihk : Eq (Set.iUnion fun i => Set.iUnion fun h => Set.Ioc (f i) (f (Order.succ …
          hk : Not (IsMax k)
          ⊢ Eq (Set.iUnion fun i => Set.iUnion fun h => Set.Ioc (f i) (f (Order.succ i)) …
        -/
      · rw [Ico_succ_right_eq_insert_of_not_isMax hmk hk, biUnion_insert]
        /-
          🎉 no goals
        -/


/-- If `α` is a linear succ order, `β` is a preorder, and `f : α → β` is a monotone function, then
the intervals `Set.Ioc (f n) (f (Order.succ n))` are pairwise disjoint. -/
theorem pairwise_disjoint_on_Ioc_succ [SuccOrder α] [Preorder β] {f : α → β} (hf : Monotone f) :
    Pairwise (Disjoint on fun n => Ioc (f n) (f (succ n))) :=
  (pairwise_disjoint_on _).2 fun _ _ hmn =>
    disjoint_iff_inf_le.mpr fun _ ⟨⟨_, h₁⟩, ⟨h₂, _⟩⟩ =>
      h₂.not_le <| h₁.trans <| hf <| succ_le_of_lt hmn


/-- If `α` is a linear succ order, `β` is a preorder, and `f : α → β` is a monotone function, then
the intervals `Set.Ico (f n) (f (Order.succ n))` are pairwise disjoint. -/
theorem pairwise_disjoint_on_Ico_succ [SuccOrder α] [Preorder β] {f : α → β} (hf : Monotone f) :
    Pairwise (Disjoint on fun n => Ico (f n) (f (succ n))) :=
  (pairwise_disjoint_on _).2 fun _ _ hmn =>
    disjoint_iff_inf_le.mpr fun _ ⟨⟨_, h₁⟩, ⟨h₂, _⟩⟩ =>
      h₁.not_le <| (hf <| succ_le_of_lt hmn).trans h₂


/-- If `α` is a linear succ order, `β` is a preorder, and `f : α → β` is a monotone function, then
the intervals `Set.Ioo (f n) (f (Order.succ n))` are pairwise disjoint. -/
theorem pairwise_disjoint_on_Ioo_succ [SuccOrder α] [Preorder β] {f : α → β} (hf : Monotone f) :
    Pairwise (Disjoint on fun n => Ioo (f n) (f (succ n))) :=
  hf.pairwise_disjoint_on_Ico_succ.mono fun _ _ h => h.mono Ioo_subset_Ico_self Ioo_subset_Ico_self


/-- If `α` is a linear pred order, `β` is a preorder, and `f : α → β` is a monotone function, then
the intervals `Set.Ioc (f Order.pred n) (f n)` are pairwise disjoint. -/
theorem pairwise_disjoint_on_Ioc_pred [PredOrder α] [Preorder β] {f : α → β} (hf : Monotone f) :
    Pairwise (Disjoint on fun n => Ioc (f (pred n)) (f n)) := by
  /-
    α : Type u_1
    β : Type u_2
    inst✝² : LinearOrder α
    inst✝¹ : PredOrder α
    inst✝ : Preorder β
    f : α → β
    hf : Monotone f
    ⊢ Pairwise (Function.onFun Disjoint fun n => Set.Ioc (f (Order.pred n)) (f n))
  -/
  simpa only [(· ∘ ·), dual_Ico] using hf.dual.pairwise_disjoint_on_Ico_succ
  /-
    🎉 no goals
  -/


/-- If `α` is a linear pred order, `β` is a preorder, and `f : α → β` is a monotone function, then
the intervals `Set.Ico (f Order.pred n) (f n)` are pairwise disjoint. -/
theorem pairwise_disjoint_on_Ico_pred [PredOrder α] [Preorder β] {f : α → β} (hf : Monotone f) :
    Pairwise (Disjoint on fun n => Ico (f (pred n)) (f n)) := by
  /-
    α : Type u_1
    β : Type u_2
    inst✝² : LinearOrder α
    inst✝¹ : PredOrder α
    inst✝ : Preorder β
    f : α → β
    hf : Monotone f
    ⊢ Pairwise (Function.onFun Disjoint fun n => Set.Ico (f (Order.pred n)) (f n))
  -/
  simpa only [(· ∘ ·), dual_Ioc] using hf.dual.pairwise_disjoint_on_Ioc_succ
  /-
    🎉 no goals
  -/


/-- If `α` is a linear pred order, `β` is a preorder, and `f : α → β` is a monotone function, then
the intervals `Set.Ioo (f Order.pred n) (f n)` are pairwise disjoint. -/
theorem pairwise_disjoint_on_Ioo_pred [PredOrder α] [Preorder β] {f : α → β} (hf : Monotone f) :
    Pairwise (Disjoint on fun n => Ioo (f (pred n)) (f n)) := by
  /-
    α : Type u_1
    β : Type u_2
    inst✝² : LinearOrder α
    inst✝¹ : PredOrder α
    inst✝ : Preorder β
    f : α → β
    hf : Monotone f
    ⊢ Pairwise (Function.onFun Disjoint fun n => Set.Ioo (f (Order.pred n)) (f n))
  -/
  simpa only [(· ∘ ·), dual_Ioo] using hf.dual.pairwise_disjoint_on_Ioo_succ
  /-
    🎉 no goals
  -/


/-- If `α` is a linear succ order, `β` is a preorder, and `f : α → β` is an antitone function, then
the intervals `Set.Ioc (f (Order.succ n)) (f n)` are pairwise disjoint. -/
theorem pairwise_disjoint_on_Ioc_succ [SuccOrder α] [Preorder β] {f : α → β} (hf : Antitone f) :
    Pairwise (Disjoint on fun n => Ioc (f (succ n)) (f n)) :=
  hf.dual_left.pairwise_disjoint_on_Ioc_pred


/-- If `α` is a linear succ order, `β` is a preorder, and `f : α → β` is an antitone function, then
the intervals `Set.Ico (f (Order.succ n)) (f n)` are pairwise disjoint. -/
theorem pairwise_disjoint_on_Ico_succ [SuccOrder α] [Preorder β] {f : α → β} (hf : Antitone f) :
    Pairwise (Disjoint on fun n => Ico (f (succ n)) (f n)) :=
  hf.dual_left.pairwise_disjoint_on_Ico_pred


/-- If `α` is a linear succ order, `β` is a preorder, and `f : α → β` is an antitone function, then
the intervals `Set.Ioo (f (Order.succ n)) (f n)` are pairwise disjoint. -/
theorem pairwise_disjoint_on_Ioo_succ [SuccOrder α] [Preorder β] {f : α → β} (hf : Antitone f) :
    Pairwise (Disjoint on fun n => Ioo (f (succ n)) (f n)) :=
  hf.dual_left.pairwise_disjoint_on_Ioo_pred


/-- If `α` is a linear pred order, `β` is a preorder, and `f : α → β` is an antitone function, then
the intervals `Set.Ioc (f n) (f (Order.pred n))` are pairwise disjoint. -/
theorem pairwise_disjoint_on_Ioc_pred [PredOrder α] [Preorder β] {f : α → β} (hf : Antitone f) :
    Pairwise (Disjoint on fun n => Ioc (f n) (f (pred n))) :=
  hf.dual_left.pairwise_disjoint_on_Ioc_succ


/-- If `α` is a linear pred order, `β` is a preorder, and `f : α → β` is an antitone function, then
the intervals `Set.Ico (f n) (f (Order.pred n))` are pairwise disjoint. -/
theorem pairwise_disjoint_on_Ico_pred [PredOrder α] [Preorder β] {f : α → β} (hf : Antitone f) :
    Pairwise (Disjoint on fun n => Ico (f n) (f (pred n))) :=
  hf.dual_left.pairwise_disjoint_on_Ico_succ


/-- If `α` is a linear pred order, `β` is a preorder, and `f : α → β` is an antitone function, then
the intervals `Set.Ioo (f n) (f (Order.pred n))` are pairwise disjoint. -/
theorem pairwise_disjoint_on_Ioo_pred [PredOrder α] [Preorder β] {f : α → β} (hf : Antitone f) :
    Pairwise (Disjoint on fun n => Ioo (f n) (f (pred n))) :=
  hf.dual_left.pairwise_disjoint_on_Ioo_succ


