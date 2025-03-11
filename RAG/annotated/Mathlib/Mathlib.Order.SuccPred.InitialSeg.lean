@[simp]
theorem apply_covBy_apply_iff (f : α ≤i β) : f a ⋖ f b ↔ a ⋖ b :=
  (isLowerSet_range f).ordConnected.apply_covBy_apply_iff f.toOrderEmbedding


@[simp]
theorem apply_wCovBy_apply_iff (f : α ≤i β) : f a ⩿ f b ↔ a ⩿ b := by
  /-
    α : Type u_1
    β : Type u_2
    a b : α
    inst✝¹ : PartialOrder α
    inst✝ : PartialOrder β
    f : InitialSeg (fun x1 x2 => LT.lt x1 x2) fun x1 x2 => LT.lt x1 x2
    ⊢ Iff (WCovBy (f a) (f b)) (WCovBy a b)
  -/
  simp [wcovBy_iff_eq_or_covBy]
  /-
    🎉 no goals
  -/


theorem map_succ [SuccOrder α] [NoMaxOrder α] [SuccOrder β] (f : α ≤i β) (a : α) :
    f (succ a) = succ (f a) :=
  (f.apply_covBy_apply_iff.2 (covBy_succ a)).succ_eq.symm


theorem map_pred [PredOrder α] [NoMinOrder α] [PredOrder β] (f : α ≤i β) (a : α) :
    f (pred a) = pred (f a) :=
  (f.apply_covBy_apply_iff.2 (pred_covBy a)).pred_eq.symm


@[simp]
theorem isSuccPrelimit_apply_iff (f : α ≤i β) : IsSuccPrelimit (f a) ↔ IsSuccPrelimit a := by
  /-
    α : Type u_1
    β : Type u_2
    a : α
    inst✝¹ : PartialOrder α
    inst✝ : PartialOrder β
    f : InitialSeg (fun x1 x2 => LT.lt x1 x2) fun x1 x2 => LT.lt x1 x2
    ⊢ Iff (Order.IsSuccPrelimit (f a)) (Order.IsSuccPrelimit a)
  -/
  constructor <;> intro h b hb
    /-
      case mp
      α : Type u_1
      β : Type u_2
      a : α
      inst✝¹ : PartialOrder α
      inst✝ : PartialOrder β
      f : InitialSeg (fun x1 x2 => LT.lt x1 x2) fun x1 x2 => LT.lt x1 x2
      h : Order.IsSuccPrelimit (f a)
      b : α
      hb : CovBy b a
      ⊢ False
    -/
  · rw [← f.apply_covBy_apply_iff] at hb
    /-
      case mp
      α : Type u_1
      β : Type u_2
      a : α
      inst✝¹ : PartialOrder α
      inst✝ : PartialOrder β
      f : InitialSeg (fun x1 x2 => LT.lt x1 x2) fun x1 x2 => LT.lt x1 x2
      h : Order.IsSuccPrelimit (f a)
      b : α
      hb : CovBy (f b) (f a)
      ⊢ False
    -/
    exact h _ hb
    /-
      🎉 no goals
    -/
    /-
      case mpr
      α : Type u_1
      β : Type u_2
      a : α
      inst✝¹ : PartialOrder α
      inst✝ : PartialOrder β
      f : InitialSeg (fun x1 x2 => LT.lt x1 x2) fun x1 x2 => LT.lt x1 x2
      h : Order.IsSuccPrelimit a
      b : β
      hb : CovBy b (f a)
      ⊢ False
    -/
  · obtain ⟨c, rfl⟩ := f.mem_range_of_rel hb.lt
    /-
      case mpr.intro
      α : Type u_1
      β : Type u_2
      a : α
      inst✝¹ : PartialOrder α
      inst✝ : PartialOrder β
      f : InitialSeg (fun x1 x2 => LT.lt x1 x2) fun x1 x2 => LT.lt x1 x2
      h : Order.IsSuccPrelimit a
      c : α
      hb : CovBy (f c) (f a)
      ⊢ False
    -/
    rw [f.apply_covBy_apply_iff] at hb
    /-
      case mpr.intro
      α : Type u_1
      β : Type u_2
      a : α
      inst✝¹ : PartialOrder α
      inst✝ : PartialOrder β
      f : InitialSeg (fun x1 x2 => LT.lt x1 x2) fun x1 x2 => LT.lt x1 x2
      h : Order.IsSuccPrelimit a
      c : α
      hb : CovBy c a
      ⊢ False
    -/
    exact h _ hb
    /-
      🎉 no goals
    -/


@[simp]
theorem isSuccLimit_apply_iff (f : α ≤i β) : IsSuccLimit (f a) ↔ IsSuccLimit a := by
  /-
    α : Type u_1
    β : Type u_2
    a : α
    inst✝¹ : PartialOrder α
    inst✝ : PartialOrder β
    f : InitialSeg (fun x1 x2 => LT.lt x1 x2) fun x1 x2 => LT.lt x1 x2
    ⊢ Iff (Order.IsSuccLimit (f a)) (Order.IsSuccLimit a)
  -/
  simp [IsSuccLimit]
  /-
    🎉 no goals
  -/


@[simp]
theorem apply_covBy_apply_iff (f : α <i β) : f a ⋖ f b ↔ a ⋖ b :=
  (f : α ≤i β).apply_covBy_apply_iff


@[simp]
theorem apply_wCovBy_apply_iff (f : α <i β) : f a ⩿ f b ↔ a ⩿ b :=
  (f : α ≤i β).apply_wCovBy_apply_iff


theorem map_succ [SuccOrder α] [NoMaxOrder α] [SuccOrder β] (f : α <i β) (a : α) :
    f (succ a) = succ (f a) :=
  (f : α ≤i β).map_succ a


theorem map_pred [PredOrder α] [NoMinOrder α] [PredOrder β] (f : α ≤i β) (a : α) :
    f (pred a) = pred (f a) :=
  (f : α ≤i β).map_pred a


@[simp]
theorem isSuccPrelimit_apply_iff (f : α <i β) : IsSuccPrelimit (f a) ↔ IsSuccPrelimit a :=
  (f : α ≤i β).isSuccPrelimit_apply_iff


@[simp]
theorem isSuccLimit_apply_iff (f : α <i β) : IsSuccLimit (f a) ↔ IsSuccLimit a :=
  (f : α ≤i β).isSuccLimit_apply_iff


