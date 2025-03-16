@[simp] theorem minimal_toDual : Minimal (fun x ↦ P (ofDual x)) (toDual x) ↔ Maximal P x :=
  Iff.rfl


alias ⟨Minimal.of_dual, Minimal.dual⟩ := minimal_toDual


@[simp] theorem maximal_toDual : Maximal (fun x ↦ P (ofDual x)) (toDual x) ↔ Minimal P x :=
  Iff.rfl


alias ⟨Maximal.of_dual, Maximal.dual⟩ := maximal_toDual


@[simp] theorem minimal_false : ¬ Minimal (fun _ ↦ False) x := by
  /-
    α : Type u_1
    x : α
    inst✝ : LE α
    ⊢ Not (Minimal (fun x => False) x)
  -/
  simp [Minimal]
  /-
    🎉 no goals
  -/


@[simp] theorem maximal_false : ¬ Maximal (fun _ ↦ False) x := by
  /-
    α : Type u_1
    x : α
    inst✝ : LE α
    ⊢ Not (Maximal (fun x => False) x)
  -/
  simp [Maximal]
  /-
    🎉 no goals
  -/


@[simp] theorem minimal_true : Minimal (fun _ ↦ True) x ↔ IsMin x := by
  /-
    α : Type u_1
    x : α
    inst✝ : LE α
    ⊢ Iff (Minimal (fun x => True) x) (IsMin x)
  -/
  simp [IsMin, Minimal]
  /-
    🎉 no goals
  -/


@[simp] theorem maximal_true : Maximal (fun _ ↦ True) x ↔ IsMax x :=
  minimal_true (α := αᵒᵈ)


@[simp] theorem minimal_subtype {x : Subtype Q} :
    Minimal (fun x ↦ P x.1) x ↔ Minimal (P ⊓ Q) x := by
  /-
    α : Type u_1
    P Q : α → Prop
    inst✝ : LE α
    x : Subtype Q
    ⊢ Iff (Minimal (fun x => P ↑x) x) (Minimal (Min.min P Q) ↑x)
  -/
  obtain ⟨x, hx⟩ := x
  /-
    case mk
    α : Type u_1
    P Q : α → Prop
    inst✝ : LE α
    x : α
    hx : Q x
    ⊢ Iff (Minimal (fun x => P ↑x) ⟨x, hx⟩) (Minimal (Min.min P Q) ↑⟨x, hx⟩)
  -/
  simp only [Minimal, Subtype.forall, Subtype.mk_le_mk, Pi.inf_apply, inf_Prop_eq]
  /-
    case mk
    α : Type u_1
    P Q : α → Prop
    inst✝ : LE α
    x : α
    hx : Q x
    ⊢ Iff (And (P x) (∀ (a : α), Q a → P a → LE.le a x → LE.le x a)) (And (And (P  …
  -/
  tauto
  /-
    🎉 no goals
  -/


@[simp] theorem maximal_subtype {x : Subtype Q} :
    Maximal (fun x ↦ P x.1) x ↔ Maximal (P ⊓ Q) x :=
  minimal_subtype (α := αᵒᵈ)


theorem maximal_true_subtype {x : Subtype P} : Maximal (fun _ ↦ True) x ↔ Maximal P x := by
  /-
    α : Type u_1
    P : α → Prop
    inst✝ : LE α
    x : Subtype P
    ⊢ Iff (Maximal (fun x => True) x) (Maximal P ↑x)
  -/
  obtain ⟨x, hx⟩ := x
  /-
    case mk
    α : Type u_1
    P : α → Prop
    inst✝ : LE α
    x : α
    hx : P x
    ⊢ Iff (Maximal (fun x => True) ⟨x, hx⟩) (Maximal P ↑⟨x, hx⟩)
  -/
  simp [Maximal, hx]
  /-
    🎉 no goals
  -/


theorem minimal_true_subtype {x : Subtype P} : Minimal (fun _ ↦ True) x ↔ Minimal P x := by
  /-
    α : Type u_1
    P : α → Prop
    inst✝ : LE α
    x : Subtype P
    ⊢ Iff (Minimal (fun x => True) x) (Minimal P ↑x)
  -/
  obtain ⟨x, hx⟩ := x
  /-
    case mk
    α : Type u_1
    P : α → Prop
    inst✝ : LE α
    x : α
    hx : P x
    ⊢ Iff (Minimal (fun x => True) ⟨x, hx⟩) (Minimal P ↑⟨x, hx⟩)
  -/
  simp [Minimal, hx]
  /-
    🎉 no goals
  -/


@[simp] theorem minimal_minimal : Minimal (Minimal P) x ↔ Minimal P x :=
  ⟨fun h ↦ h.prop, fun h ↦ ⟨h, fun _ hy hyx ↦ h.le_of_le hy.prop hyx⟩⟩


@[simp] theorem maximal_maximal : Maximal (Maximal P) x ↔ Maximal P x :=
  minimal_minimal (α := αᵒᵈ)


/-- If `P` is down-closed, then minimal elements satisfying `P` are exactly the globally minimal
elements satisfying `P`. -/
theorem minimal_iff_isMin (hP : ∀ ⦃x y⦄, P y → x ≤ y → P x) : Minimal P x ↔ P x ∧ IsMin x :=
  ⟨fun h ↦ ⟨h.prop, fun _ h' ↦ h.le_of_le (hP h.prop h') h'⟩, fun h ↦ ⟨h.1, fun _ _  h' ↦ h.2 h'⟩⟩


/-- If `P` is up-closed, then maximal elements satisfying `P` are exactly the globally maximal
elements satisfying `P`. -/
theorem maximal_iff_isMax (hP : ∀ ⦃x y⦄, P y → y ≤ x → P x) : Maximal P x ↔ P x ∧ IsMax x :=
  ⟨fun h ↦ ⟨h.prop, fun _ h' ↦ h.le_of_ge (hP h.prop h') h'⟩, fun h ↦ ⟨h.1, fun _ _  h' ↦ h.2 h'⟩⟩


theorem Minimal.mono (h : Minimal P x) (hle : Q ≤ P) (hQ : Q x) : Minimal Q x :=
  ⟨hQ, fun y hQy ↦ h.le_of_le (hle y hQy)⟩


theorem Maximal.mono (h : Maximal P x) (hle : Q ≤ P) (hQ : Q x) : Maximal Q x :=
  ⟨hQ, fun y hQy ↦ h.le_of_ge (hle y hQy)⟩


theorem Minimal.and_right (h : Minimal P x) (hQ : Q x) : Minimal (fun x ↦ P x ∧ Q x) x :=
  h.mono (fun _ ↦ And.left) ⟨h.prop, hQ⟩


theorem Minimal.and_left (h : Minimal P x) (hQ : Q x) : Minimal (fun x ↦ (Q x ∧ P x)) x :=
  h.mono (fun _ ↦ And.right) ⟨hQ, h.prop⟩


theorem Maximal.and_right (h : Maximal P x) (hQ : Q x) : Maximal (fun x ↦ (P x ∧ Q x)) x :=
  h.mono (fun _ ↦ And.left) ⟨h.prop, hQ⟩


theorem Maximal.and_left (h : Maximal P x) (hQ : Q x) : Maximal (fun x ↦ (Q x ∧ P x)) x :=
  h.mono (fun _ ↦ And.right) ⟨hQ, h.prop⟩


@[simp] theorem minimal_eq_iff : Minimal (· = y) x ↔ x = y := by
  /-
    α : Type u_1
    x y : α
    inst✝ : LE α
    ⊢ Iff (Minimal (fun x => Eq x y) x) (Eq x y)
  -/
  simp (config := {contextual := true}) [Minimal]
  /-
    🎉 no goals
  -/


@[simp] theorem maximal_eq_iff : Maximal (· = y) x ↔ x = y := by
  /-
    α : Type u_1
    x y : α
    inst✝ : LE α
    ⊢ Iff (Maximal (fun x => Eq x y) x) (Eq x y)
  -/
  simp (config := {contextual := true}) [Maximal]
  /-
    🎉 no goals
  -/


theorem not_minimal_iff (hx : P x) : ¬ Minimal P x ↔ ∃ y, P y ∧ y ≤ x ∧ ¬ (x ≤ y) := by
  /-
    α : Type u_1
    P : α → Prop
    x : α
    inst✝ : LE α
    hx : P x
    ⊢ Iff (Not (Minimal P x)) (Exists fun y => And (P y) (And (LE.le y x) (Not (LE …
  -/
  simp [Minimal, hx]
  /-
    🎉 no goals
  -/


theorem not_maximal_iff (hx : P x) : ¬ Maximal P x ↔ ∃ y, P y ∧ x ≤ y ∧ ¬ (y ≤ x) :=
  not_minimal_iff (α := αᵒᵈ) hx


theorem Minimal.or (h : Minimal (fun x ↦ P x ∨ Q x) x) : Minimal P x ∨ Minimal Q x := by
  /-
    α : Type u_1
    P Q : α → Prop
    x : α
    inst✝ : LE α
    h : Minimal (fun x => Or (P x) (Q x)) x
    ⊢ Or (Minimal P x) (Minimal Q x)
  -/
  obtain ⟨h | h, hmin⟩ := h
    /-
      case intro.inl
      α : Type u_1
      P Q : α → Prop
      x : α
      inst✝ : LE α
      hmin : ∀ ⦃y : α⦄, (fun x => Or (P x) (Q x)) y → LE.le y x → LE.le x y
      h : P x
      ⊢ Or (Minimal P x) (Minimal Q x)
    -/
  · exact .inl ⟨h, fun y hy hyx ↦ hmin (Or.inl hy) hyx⟩
    /-
      🎉 no goals
    -/
  /-
    case intro.inr
    α : Type u_1
    P Q : α → Prop
    x : α
    inst✝ : LE α
    hmin : ∀ ⦃y : α⦄, (fun x => Or (P x) (Q x)) y → LE.le y x → LE.le x y
    h : Q x
    ⊢ Or (Minimal P x) (Minimal Q x)
  -/
  exact .inr ⟨h, fun y hy hyx ↦ hmin (Or.inr hy) hyx⟩
  /-
    🎉 no goals
  -/


theorem Maximal.or (h : Maximal (fun x ↦ P x ∨ Q x) x) : Maximal P x ∨ Maximal Q x :=
  Minimal.or (α := αᵒᵈ) h


theorem minimal_and_iff_right_of_imp (hPQ : ∀ ⦃x⦄, P x → Q x) :
    Minimal (fun x ↦ P x ∧ Q x) x ↔ (Minimal P x) ∧ Q x := by
  /-
    α : Type u_1
    P Q : α → Prop
    x : α
    inst✝ : LE α
    hPQ : ∀ ⦃x : α⦄, P x → Q x
    ⊢ Iff (Minimal (fun x => And (P x) (Q x)) x) (And (Minimal P x) (Q x))
  -/
  simp_rw [and_iff_left_of_imp (fun x ↦ hPQ x), iff_self_and]
  /-
    α : Type u_1
    P Q : α → Prop
    x : α
    inst✝ : LE α
    hPQ : ∀ ⦃x : α⦄, P x → Q x
    ⊢ Minimal (fun x => P x) x → Q x
  -/
  exact fun h ↦ hPQ h.prop
  /-
    🎉 no goals
  -/


theorem minimal_and_iff_left_of_imp (hPQ : ∀ ⦃x⦄, P x → Q x) :
    Minimal (fun x ↦ Q x ∧ P x) x ↔ Q x ∧ (Minimal P x) := by
  /-
    α : Type u_1
    P Q : α → Prop
    x : α
    inst✝ : LE α
    hPQ : ∀ ⦃x : α⦄, P x → Q x
    ⊢ Iff (Minimal (fun x => And (Q x) (P x)) x) (And (Q x) (Minimal P x))
  -/
  simp_rw [iff_comm, and_comm, minimal_and_iff_right_of_imp hPQ, and_comm]
  /-
    🎉 no goals
  -/


theorem maximal_and_iff_right_of_imp (hPQ : ∀ ⦃x⦄, P x → Q x) :
    Maximal (fun x ↦ P x ∧ Q x) x ↔ (Maximal P x) ∧ Q x :=
  minimal_and_iff_right_of_imp (α := αᵒᵈ) hPQ


theorem maximal_and_iff_left_of_imp (hPQ : ∀ ⦃x⦄, P x → Q x) :
    Maximal (fun x ↦ Q x ∧ P x) x ↔ Q x ∧ (Maximal P x) :=
  minimal_and_iff_left_of_imp (α := αᵒᵈ) hPQ


theorem minimal_iff_forall_lt : Minimal P x ↔ P x ∧ ∀ ⦃y⦄, y < x → ¬ P y := by
  /-
    α : Type u_1
    P : α → Prop
    x : α
    inst✝ : Preorder α
    ⊢ Iff (Minimal P x) (And (P x) (∀ ⦃y : α⦄, LT.lt y x → Not (P y)))
  -/
  simp [Minimal, lt_iff_le_not_le, not_imp_not, imp.swap]
  /-
    🎉 no goals
  -/


theorem maximal_iff_forall_gt : Maximal P x ↔ P x ∧ ∀ ⦃y⦄, x < y → ¬ P y :=
  minimal_iff_forall_lt (α := αᵒᵈ)


theorem Minimal.not_prop_of_lt (h : Minimal P x) (hlt : y < x) : ¬ P y :=
  (minimal_iff_forall_lt.1 h).2 hlt


theorem Maximal.not_prop_of_gt (h : Maximal P x) (hlt : x < y) : ¬ P y :=
  (maximal_iff_forall_gt.1 h).2 hlt


theorem Minimal.not_lt (h : Minimal P x) (hy : P y) : ¬ (y < x) :=
  fun hlt ↦ h.not_prop_of_lt hlt hy


theorem Maximal.not_gt (h : Maximal P x) (hy : P y) : ¬ (x < y) :=
  fun hlt ↦ h.not_prop_of_gt hlt hy


@[simp] theorem minimal_le_iff : Minimal (· ≤ y) x ↔ x ≤ y ∧ IsMin x :=
  minimal_iff_isMin (fun _ _ h h' ↦ h'.trans h)


@[simp] theorem maximal_ge_iff : Maximal (y ≤ ·) x ↔ y ≤ x ∧ IsMax x :=
  minimal_le_iff (α := αᵒᵈ)


@[simp] theorem minimal_lt_iff : Minimal (· < y) x ↔ x < y ∧ IsMin x :=
  minimal_iff_isMin (fun _ _ h h' ↦ h'.trans_lt h)


@[simp] theorem maximal_gt_iff : Maximal (y < ·) x ↔ y < x ∧ IsMax x :=
  minimal_lt_iff (α := αᵒᵈ)


theorem not_minimal_iff_exists_lt (hx : P x) : ¬ Minimal P x ↔ ∃ y, y < x ∧ P y := by
  /-
    α : Type u_1
    P : α → Prop
    x : α
    inst✝ : Preorder α
    hx : P x
    ⊢ Iff (Not (Minimal P x)) (Exists fun y => And (LT.lt y x) (P y))
  -/
  simp_rw [not_minimal_iff hx, lt_iff_le_not_le, and_comm]
  /-
    🎉 no goals
  -/


alias ⟨exists_lt_of_not_minimal, _⟩ := not_minimal_iff_exists_lt


theorem not_maximal_iff_exists_gt (hx : P x) : ¬ Maximal P x ↔ ∃ y, x < y ∧ P y :=
  not_minimal_iff_exists_lt (α := αᵒᵈ) hx


alias ⟨exists_gt_of_not_maximal, _⟩ := not_maximal_iff_exists_gt


theorem Minimal.eq_of_ge (hx : Minimal P x) (hy : P y) (hge : y ≤ x) : x = y :=
  (hx.2 hy hge).antisymm hge


theorem Minimal.eq_of_le (hx : Minimal P x) (hy : P y) (hle : y ≤ x) : y = x :=
  (hx.eq_of_ge hy hle).symm


theorem Maximal.eq_of_le (hx : Maximal P x) (hy : P y) (hle : x ≤ y) : x = y :=
  hle.antisymm <| hx.2 hy hle


theorem Maximal.eq_of_ge (hx : Maximal P x) (hy : P y) (hge : x ≤ y) : y = x :=
  (hx.eq_of_le hy hge).symm


theorem minimal_iff : Minimal P x ↔ P x ∧ ∀ ⦃y⦄, P y → y ≤ x → x = y :=
  ⟨fun h ↦ ⟨h.1, fun _ ↦ h.eq_of_ge⟩, fun h ↦ ⟨h.1, fun _ hy hle ↦ (h.2 hy hle).le⟩⟩


theorem maximal_iff : Maximal P x ↔ P x ∧ ∀ ⦃y⦄, P y → x ≤ y → x = y :=
  minimal_iff (α := αᵒᵈ)


theorem minimal_mem_iff {s : Set α} : Minimal (· ∈ s) x ↔ x ∈ s ∧ ∀ ⦃y⦄, y ∈ s → y ≤ x → x = y :=
  minimal_iff


theorem maximal_mem_iff {s : Set α} : Maximal (· ∈ s) x ↔ x ∈ s ∧ ∀ ⦃y⦄, y ∈ s → x ≤ y → x = y :=
  maximal_iff


/-- If `P y` holds, and everything satisfying `P` is above `y`, then `y` is the unique minimal
element satisfying `P`. -/
theorem minimal_iff_eq (hy : P y) (hP : ∀ ⦃x⦄, P x → y ≤ x) : Minimal P x ↔ x = y :=
                                         /-
                                           α : Type u_1
                                           P : α → Prop
                                           x y : α
                                           inst✝ : PartialOrder α
                                           hy : P y
                                           hP : ∀ ⦃x : α⦄, P x → LE.le y x
                                           ⊢ Eq x y → Minimal P x
                                         -/
  ⟨fun h ↦ h.eq_of_ge hy (hP h.prop), by rintro rfl; exact ⟨hy, fun z hz _ ↦ hP hz⟩⟩
                                                     /-
                                                       🎉 no goals
                                                     -/


/-- If `P y` holds, and everything satisfying `P` is below `y`, then `y` is the unique maximal
element satisfying `P`. -/
theorem maximal_iff_eq (hy : P y) (hP : ∀ ⦃x⦄, P x → x ≤ y) : Maximal P x ↔ x = y :=
  minimal_iff_eq (α := αᵒᵈ) hy hP


@[simp] theorem minimal_ge_iff : Minimal (y ≤ ·) x ↔ x = y :=
  minimal_iff_eq rfl.le fun _ ↦ id


@[simp] theorem maximal_le_iff : Maximal (· ≤ y) x ↔ x = y :=
  maximal_iff_eq rfl.le fun _ ↦ id


theorem minimal_iff_minimal_of_imp_of_forall (hPQ : ∀ ⦃x⦄, Q x → P x)
    (h : ∀ ⦃x⦄, P x → ∃ y, y ≤ x ∧ Q y) : Minimal P x ↔ Minimal Q x := by
  refine ⟨fun h' ↦ ⟨?_, fun y hy hyx ↦ h'.le_of_le (hPQ hy) hyx⟩,
    fun h' ↦ ⟨hPQ h'.prop, fun y hy hyx ↦ ?_⟩⟩
    /-
      case refine_1
      α : Type u_1
      P Q : α → Prop
      x : α
      inst✝ : PartialOrder α
      hPQ : ∀ ⦃x : α⦄, Q x → P x
      h : ∀ ⦃x : α⦄, P x → Exists fun y => And (LE.le y x) (Q y)
      h' : Minimal P x
      ⊢ Q x
    -/
  · obtain ⟨y, hyx, hy⟩ := h h'.prop
    /-
      case refine_1.intro.intro
      α : Type u_1
      P Q : α → Prop
      x : α
      inst✝ : PartialOrder α
      hPQ : ∀ ⦃x : α⦄, Q x → P x
      h : ∀ ⦃x : α⦄, P x → Exists fun y => And (LE.le y x) (Q y)
      h' : Minimal P x
      y : α
      hyx : LE.le y x
      hy : Q y
      ⊢ Q x
    -/
    rwa [((h'.le_of_le (hPQ hy)) hyx).antisymm hyx]
    /-
      🎉 no goals
    -/
  /-
    case refine_2
    α : Type u_1
    P Q : α → Prop
    x : α
    inst✝ : PartialOrder α
    hPQ : ∀ ⦃x : α⦄, Q x → P x
    h : ∀ ⦃x : α⦄, P x → Exists fun y => And (LE.le y x) (Q y)
    h' : Minimal Q x
    y : α
    hy : P y
    hyx : LE.le y x
    ⊢ LE.le x y
  -/
  obtain ⟨z, hzy, hz⟩ := h hy
  /-
    case refine_2.intro.intro
    α : Type u_1
    P Q : α → Prop
    x : α
    inst✝ : PartialOrder α
    hPQ : ∀ ⦃x : α⦄, Q x → P x
    h : ∀ ⦃x : α⦄, P x → Exists fun y => And (LE.le y x) (Q y)
    h' : Minimal Q x
    y : α
    hy : P y
    hyx : LE.le y x
    z : α
    hzy : LE.le z y
    hz : Q z
    ⊢ LE.le x y
  -/
  exact (h'.le_of_le hz (hzy.trans hyx)).trans hzy
  /-
    🎉 no goals
  -/


theorem maximal_iff_maximal_of_imp_of_forall (hPQ : ∀ ⦃x⦄, Q x → P x)
    (h : ∀ ⦃x⦄, P x → ∃ y, x ≤ y ∧ Q y) : Maximal P x ↔ Maximal Q x :=
  minimal_iff_minimal_of_imp_of_forall (α := αᵒᵈ) hPQ h


theorem Minimal.eq_of_superset (h : Minimal P s) (ht : P t) (hts : t ⊆ s) : s = t :=
  h.eq_of_ge ht hts


theorem Maximal.eq_of_subset (h : Maximal P s) (ht : P t) (hst : s ⊆ t) : s = t :=
  h.eq_of_le ht hst


theorem Minimal.eq_of_subset (h : Minimal P s) (ht : P t) (hts : t ⊆ s) : t = s :=
  h.eq_of_le ht hts


theorem Maximal.eq_of_superset (h : Maximal P s) (ht : P t) (hst : s ⊆ t) : t = s :=
  h.eq_of_ge ht hst


theorem minimal_subset_iff : Minimal P s ↔ P s ∧ ∀ ⦃t⦄, P t → t ⊆ s → s = t :=
  _root_.minimal_iff


theorem maximal_subset_iff : Maximal P s ↔ P s ∧ ∀ ⦃t⦄, P t → s ⊆ t → s = t :=
  _root_.maximal_iff


theorem minimal_subset_iff' : Minimal P s ↔ P s ∧ ∀ ⦃t⦄, P t → t ⊆ s → s ⊆ t :=
  Iff.rfl


theorem maximal_subset_iff' : Maximal P s ↔ P s ∧ ∀ ⦃t⦄, P t → s ⊆ t → t ⊆ s :=
  Iff.rfl


theorem not_minimal_subset_iff (hs : P s) : ¬ Minimal P s ↔ ∃ t, t ⊂ s ∧ P t :=
  not_minimal_iff_exists_lt hs


theorem not_maximal_subset_iff (hs : P s) : ¬ Maximal P s ↔ ∃ t, s ⊂ t ∧ P t :=
  not_maximal_iff_exists_gt hs


theorem Set.minimal_iff_forall_ssubset : Minimal P s ↔ P s ∧ ∀ ⦃t⦄, t ⊂ s → ¬ P t :=
  minimal_iff_forall_lt


theorem Minimal.not_prop_of_ssubset (h : Minimal P s) (ht : t ⊂ s) : ¬ P t :=
  (minimal_iff_forall_lt.1 h).2 ht


theorem Minimal.not_ssubset (h : Minimal P s) (ht : P t) : ¬ t ⊂ s :=
  h.not_lt ht


theorem Maximal.mem_of_prop_insert (h : Maximal P s) (hx : P (insert x s)) : x ∈ s :=
  h.eq_of_subset hx (subset_insert _ _) ▸ mem_insert ..


theorem Minimal.not_mem_of_prop_diff_singleton (h : Minimal P s) (hx : P (s \ {x})) : x ∉ s :=
  fun hxs ↦ ((h.eq_of_superset hx diff_subset).subset hxs).2 rfl


theorem Set.minimal_iff_forall_diff_singleton (hP : ∀ ⦃s t⦄, P t → t ⊆ s → P s) :
    Minimal P s ↔ P s ∧ ∀ x ∈ s, ¬ P (s \ {x}) :=
  ⟨fun h ↦ ⟨h.1, fun _ hx hP ↦ h.not_mem_of_prop_diff_singleton hP hx⟩,
    fun h ↦ ⟨h.1, fun _ ht hts x hxs ↦ by_contra fun hxt ↦
      h.2 x hxs (hP ht <| subset_diff_singleton hts hxt)⟩⟩


theorem Set.exists_diff_singleton_of_not_minimal (hP : ∀ ⦃s t⦄, P t → t ⊆ s → P s) (hs : P s)
    (h : ¬ Minimal P s) : ∃ x ∈ s, P (s \ {x}) := by
  /-
    α : Type u_1
    P : Set α → Prop
    s : Set α
    hP : ∀ ⦃s t : Set α⦄, P t → HasSubset.Subset t s → P s
    hs : P s
    h : Not (Minimal P s)
    ⊢ Exists fun x => And (Membership.mem s x) (P (SDiff.sdiff s (Singleton.single …
  -/
  simpa [Set.minimal_iff_forall_diff_singleton hP, hs] using h
  /-
    🎉 no goals
  -/


theorem Set.maximal_iff_forall_ssuperset : Maximal P s ↔ P s ∧ ∀ ⦃t⦄, s ⊂ t → ¬ P t :=
  maximal_iff_forall_gt


theorem Maximal.not_prop_of_ssuperset (h : Maximal P s) (ht : s ⊂ t) : ¬ P t :=
  (maximal_iff_forall_gt.1 h).2 ht


theorem Maximal.not_ssuperset (h : Maximal P s) (ht : P t) : ¬ s ⊂ t :=
  h.not_gt ht


theorem Set.maximal_iff_forall_insert (hP : ∀ ⦃s t⦄, P t → s ⊆ t → P s) :
    Maximal P s ↔ P s ∧ ∀ x ∉ s, ¬ P (insert x s) := by
  /-
    α : Type u_1
    P : Set α → Prop
    s : Set α
    hP : ∀ ⦃s t : Set α⦄, P t → HasSubset.Subset s t → P s
    ⊢ Iff (Maximal P s) (And (P s) (∀ (x : α), Not (Membership.mem s x) → Not (P ( …
  -/
  simp only [not_imp_not]
  exact ⟨fun h ↦ ⟨h.1, fun x ↦ h.mem_of_prop_insert⟩,
    fun h ↦ ⟨h.1, fun t ht hst x hxt ↦ h.2 x (hP ht <| insert_subset hxt hst)⟩⟩


theorem Set.exists_insert_of_not_maximal (hP : ∀ ⦃s t⦄, P t → s ⊆ t → P s) (hs : P s)
    (h : ¬ Maximal P s) : ∃ x ∉ s, P (insert x s) := by
  /-
    α : Type u_1
    P : Set α → Prop
    s : Set α
    hP : ∀ ⦃s t : Set α⦄, P t → HasSubset.Subset s t → P s
    hs : P s
    h : Not (Maximal P s)
    ⊢ Exists fun x => And (Not (Membership.mem s x)) (P (Insert.insert x s))
  -/
  simpa [Set.maximal_iff_forall_insert hP, hs] using h
  /-
    🎉 no goals
  -/

/- TODO : generalize `minimal_iff_forall_diff_singleton` and `maximal_iff_forall_insert`
to `IsStronglyCoatomic`/`IsStronglyAtomic` orders. -/


theorem setOf_minimal_subset (s : Set α) : {x | Minimal (· ∈ s) x} ⊆ s :=
  sep_subset ..


theorem setOf_maximal_subset (s : Set α) : {x | Maximal (· ∈ s) x} ⊆ s :=
  sep_subset ..


theorem Set.Subsingleton.maximal_mem_iff (h : s.Subsingleton) : Maximal (· ∈ s) x ↔ x ∈ s := by
  /-
    α : Type u_1
    x : α
    s : Set α
    inst✝ : Preorder α
    h : s.Subsingleton
    ⊢ Iff (Maximal (fun x => Membership.mem s x) x) (Membership.mem s x)
  -/
                                                         /-
                                                           🎉 no goals
                                                         -/
  obtain (rfl | ⟨x, rfl⟩) := h.eq_empty_or_singleton <;> simp
                                                         /-
                                                           🎉 no goals
                                                         -/


theorem Set.Subsingleton.minimal_mem_iff (h : s.Subsingleton) : Minimal (· ∈ s) x ↔ x ∈ s := by
  /-
    α : Type u_1
    x : α
    s : Set α
    inst✝ : Preorder α
    h : s.Subsingleton
    ⊢ Iff (Minimal (fun x => Membership.mem s x) x) (Membership.mem s x)
  -/
                                                         /-
                                                           🎉 no goals
                                                         -/
  obtain (rfl | ⟨x, rfl⟩) := h.eq_empty_or_singleton <;> simp
                                                         /-
                                                           🎉 no goals
                                                         -/


theorem IsLeast.minimal (h : IsLeast s x) : Minimal (· ∈ s) x :=
  ⟨h.1, fun _b hb _ ↦ h.2 hb⟩


theorem IsGreatest.maximal (h : IsGreatest s x) : Maximal (· ∈ s) x :=
  ⟨h.1, fun _b hb _ ↦ h.2 hb⟩


theorem IsAntichain.minimal_mem_iff (hs : IsAntichain (· ≤ ·) s) : Minimal (· ∈ s) x ↔ x ∈ s :=
  ⟨fun h ↦ h.prop, fun h ↦ ⟨h, fun _ hys hyx ↦ (hs.eq hys h hyx).symm.le⟩⟩


theorem IsAntichain.maximal_mem_iff (hs : IsAntichain (· ≤ ·) s) : Maximal (· ∈ s) x ↔ x ∈ s :=
  hs.to_dual.minimal_mem_iff


/-- If `t` is an antichain shadowing and including the set of maximal elements of `s`,
then `t` *is* the set of maximal elements of `s`. -/
theorem IsAntichain.eq_setOf_maximal (ht : IsAntichain (· ≤ ·) t)
    (h : ∀ x, Maximal (· ∈ s) x → x ∈ t) (hs : ∀ a ∈ t, ∃ b, b ≤ a ∧ Maximal (· ∈ s) b) :
    {x | Maximal (· ∈ s) x} = t := by
  /-
    α : Type u_1
    s t : Set α
    inst✝ : Preorder α
    ht : IsAntichain (fun x1 x2 => LE.le x1 x2) t
    h : ∀ (x : α), Maximal (fun x => Membership.mem s x) x → Membership.mem t x
    hs : ∀ (a : α), Membership.mem t a → Exists fun b => And (LE.le b a) (Maximal  …
    ⊢ Eq (setOf fun x => Maximal (fun x => Membership.mem s x) x) t
  -/
  refine Set.ext fun x ↦ ⟨h _, fun hx ↦ ?_⟩
  /-
    α : Type u_1
    s t : Set α
    inst✝ : Preorder α
    ht : IsAntichain (fun x1 x2 => LE.le x1 x2) t
    h : ∀ (x : α), Maximal (fun x => Membership.mem s x) x → Membership.mem t x
    hs : ∀ (a : α), Membership.mem t a → Exists fun b => And (LE.le b a) (Maximal  …
    x : α
    hx : Membership.mem t x
    ⊢ Membership.mem (setOf fun x => Maximal (fun x => Membership.mem s x) x) x
  -/
  obtain ⟨y, hyx, hy⟩ := hs x hx
  /-
    case intro.intro
    α : Type u_1
    s t : Set α
    inst✝ : Preorder α
    ht : IsAntichain (fun x1 x2 => LE.le x1 x2) t
    h : ∀ (x : α), Maximal (fun x => Membership.mem s x) x → Membership.mem t x
    hs : ∀ (a : α), Membership.mem t a → Exists fun b => And (LE.le b a) (Maximal  …
    x : α
    hx : Membership.mem t x
    y : α
    hyx : LE.le y x
    hy : Maximal (fun x => Membership.mem s x) y
    ⊢ Membership.mem (setOf fun x => Maximal (fun x => Membership.mem s x) x) x
  -/
  rwa [← ht.eq (h y hy) hx hyx]
  /-
    🎉 no goals
  -/


/-- If `t` is an antichain shadowed by and including the set of minimal elements of `s`,
then `t` *is* the set of minimal elements of `s`. -/
theorem IsAntichain.eq_setOf_minimal (ht : IsAntichain (· ≤ ·) t)
    (h : ∀ x, Minimal (· ∈ s) x → x ∈ t) (hs : ∀ a ∈ t, ∃ b, a ≤ b ∧ Minimal (· ∈ s) b) :
    {x | Minimal (· ∈ s) x} = t :=
  ht.to_dual.eq_setOf_maximal h hs


theorem setOf_maximal_antichain (P : α → Prop) : IsAntichain (· ≤ ·) {x | Maximal P x} :=
  fun _ hx _ ⟨hy, _⟩ hne hle ↦ hne (hle.antisymm <| hx.2 hy hle)


theorem setOf_minimal_antichain (P : α → Prop) : IsAntichain (· ≤ ·) {x | Minimal P x} :=
  (setOf_maximal_antichain (α := αᵒᵈ) P).swap


theorem IsAntichain.minimal_mem_upperClosure_iff_mem (hs : IsAntichain (· ≤ ·) s) :
    Minimal (· ∈ upperClosure s) x ↔ x ∈ s := by
  /-
    α : Type u_1
    x : α
    s : Set α
    inst✝ : PartialOrder α
    hs : IsAntichain (fun x1 x2 => LE.le x1 x2) s
    ⊢ Iff (Minimal (fun x => Membership.mem (upperClosure s) x) x) (Membership.mem …
  -/
  simp only [upperClosure, UpperSet.mem_mk, mem_setOf_eq]
  /-
    α : Type u_1
    x : α
    s : Set α
    inst✝ : PartialOrder α
    hs : IsAntichain (fun x1 x2 => LE.le x1 x2) s
    ⊢ Iff (Minimal (fun x => Exists fun a => And (Membership.mem s a) (LE.le a x)) …
  -/
  refine ⟨fun h ↦ ?_, fun h ↦ ⟨⟨x, h, rfl.le⟩, fun b ⟨a, has, hab⟩ hbx ↦ ?_⟩⟩
    /-
      case refine_1
      α : Type u_1
      x : α
      s : Set α
      inst✝ : PartialOrder α
      hs : IsAntichain (fun x1 x2 => LE.le x1 x2) s
      h : Minimal (fun x => Exists fun a => And (Membership.mem s a) (LE.le a x)) x
      ⊢ Membership.mem s x
    -/
  · obtain ⟨a, has, hax⟩ := h.prop
    /-
      case refine_1.intro.intro
      α : Type u_1
      x : α
      s : Set α
      inst✝ : PartialOrder α
      hs : IsAntichain (fun x1 x2 => LE.le x1 x2) s
      h : Minimal (fun x => Exists fun a => And (Membership.mem s a) (LE.le a x)) x
      a : α
      has : Membership.mem s a
      hax : LE.le a x
      ⊢ Membership.mem s x
    -/
    rwa [h.eq_of_ge ⟨a, has, rfl.le⟩ hax]
    /-
      🎉 no goals
    -/
  /-
    case refine_2
    α : Type u_1
    x : α
    s : Set α
    inst✝ : PartialOrder α
    hs : IsAntichain (fun x1 x2 => LE.le x1 x2) s
    h : Membership.mem s x
    b : α
    x✝ : (fun x => Exists fun a => And (Membership.mem s a) (LE.le a x)) b
    hbx : LE.le b x
    a : α
    has : Membership.mem s a
    hab : LE.le a b
    ⊢ LE.le x b
  -/
  rwa [← hs.eq has h (hab.trans hbx)]
  /-
    🎉 no goals
  -/


theorem IsAntichain.maximal_mem_lowerClosure_iff_mem (hs : IsAntichain (· ≤ ·) s) :
    Maximal (· ∈ lowerClosure s) x ↔ x ∈ s :=
  hs.to_dual.minimal_mem_upperClosure_iff_mem


theorem IsLeast.minimal_iff (h : IsLeast s a) : Minimal (· ∈ s) x ↔ x = a :=
  ⟨fun h' ↦ h'.eq_of_ge h.1 (h.2 h'.prop), fun h' ↦ h' ▸ h.minimal⟩


theorem IsGreatest.maximal_iff (h : IsGreatest s a) : Maximal (· ∈ s) x ↔ x = a :=
  ⟨fun h' ↦ h'.eq_of_le h.1 (h.2 h'.prop), fun h' ↦ h' ▸ h.maximal⟩


theorem minimal_mem_image_monotone (hf : ∀ ⦃x y⦄, x ∈ s → y ∈ s → (f x ≤ f y ↔ x ≤ y))
    (hx : Minimal (· ∈ s) x) : Minimal (· ∈ f '' s) (f x) := by
  /-
    α : Type u_1
    x : α
    inst✝¹ : Preorder α
    β : Type u_2
    inst✝ : Preorder β
    s : Set α
    f : α → β
    hf : ∀ ⦃x y : α⦄, Membership.mem s x → Membership.mem s y → Iff (LE.le (f x) ( …
    hx : Minimal (fun x => Membership.mem s x) x
    ⊢ Minimal (fun x => Membership.mem (Set.image f s) x) (f x)
  -/
  refine ⟨mem_image_of_mem f hx.prop, ?_⟩
  /-
    α : Type u_1
    x : α
    inst✝¹ : Preorder α
    β : Type u_2
    inst✝ : Preorder β
    s : Set α
    f : α → β
    hf : ∀ ⦃x y : α⦄, Membership.mem s x → Membership.mem s y → Iff (LE.le (f x) ( …
    hx : Minimal (fun x => Membership.mem s x) x
    ⊢ ∀ ⦃y : β⦄, (fun x => Membership.mem (Set.image f s) x) y → LE.le y (f x) → L …
  -/
  rintro _ ⟨y, hy, rfl⟩
  /-
    case intro.intro
    α : Type u_1
    x : α
    inst✝¹ : Preorder α
    β : Type u_2
    inst✝ : Preorder β
    s : Set α
    f : α → β
    hf : ∀ ⦃x y : α⦄, Membership.mem s x → Membership.mem s y → Iff (LE.le (f x) ( …
    hx : Minimal (fun x => Membership.mem s x) x
    y : α
    hy : Membership.mem s y
    ⊢ LE.le (f y) (f x) → LE.le (f x) (f y)
  -/
  rw [hf hx.prop hy, hf hy hx.prop]
  /-
    case intro.intro
    α : Type u_1
    x : α
    inst✝¹ : Preorder α
    β : Type u_2
    inst✝ : Preorder β
    s : Set α
    f : α → β
    hf : ∀ ⦃x y : α⦄, Membership.mem s x → Membership.mem s y → Iff (LE.le (f x) ( …
    hx : Minimal (fun x => Membership.mem s x) x
    y : α
    hy : Membership.mem s y
    ⊢ LE.le y x → LE.le x y
  -/
  exact hx.le_of_le hy
  /-
    🎉 no goals
  -/


theorem maximal_mem_image_monotone (hf : ∀ ⦃x y⦄, x ∈ s → y ∈ s → (f x ≤ f y ↔ x ≤ y))
    (hx : Maximal (· ∈ s) x) : Maximal (· ∈ f '' s) (f x) :=
  minimal_mem_image_monotone (α := αᵒᵈ) (β := βᵒᵈ) (s := s) (fun _ _ hx hy ↦ hf hy hx) hx


theorem minimal_mem_image_monotone_iff (ha : a ∈ s)
    (hf : ∀ ⦃x y⦄, x ∈ s → y ∈ s → (f x ≤ f y ↔ x ≤ y)) :
    Minimal (· ∈ f '' s) (f a) ↔ Minimal (· ∈ s) a := by
  /-
    α : Type u_1
    a : α
    inst✝¹ : Preorder α
    β : Type u_2
    inst✝ : Preorder β
    s : Set α
    f : α → β
    ha : Membership.mem s a
    hf : ∀ ⦃x y : α⦄, Membership.mem s x → Membership.mem s y → Iff (LE.le (f x) ( …
    ⊢ Iff (Minimal (fun x => Membership.mem (Set.image f s) x) (f a)) (Minimal (fu …
  -/
  refine ⟨fun h ↦ ⟨ha, fun y hys ↦ ?_⟩, minimal_mem_image_monotone hf⟩
  /-
    α : Type u_1
    a : α
    inst✝¹ : Preorder α
    β : Type u_2
    inst✝ : Preorder β
    s : Set α
    f : α → β
    ha : Membership.mem s a
    hf : ∀ ⦃x y : α⦄, Membership.mem s x → Membership.mem s y → Iff (LE.le (f x) ( …
    h : Minimal (fun x => Membership.mem (Set.image f s) x) (f a)
    y : α
    hys : (fun x => Membership.mem s x) y
    ⊢ LE.le y a → LE.le a y
  -/
  rw [← hf ha hys, ← hf hys ha]
  /-
    α : Type u_1
    a : α
    inst✝¹ : Preorder α
    β : Type u_2
    inst✝ : Preorder β
    s : Set α
    f : α → β
    ha : Membership.mem s a
    hf : ∀ ⦃x y : α⦄, Membership.mem s x → Membership.mem s y → Iff (LE.le (f x) ( …
    h : Minimal (fun x => Membership.mem (Set.image f s) x) (f a)
    y : α
    hys : (fun x => Membership.mem s x) y
    ⊢ LE.le (f y) (f a) → LE.le (f a) (f y)
  -/
  exact h.le_of_le (mem_image_of_mem f hys)
  /-
    🎉 no goals
  -/


theorem maximal_mem_image_monotone_iff (ha : a ∈ s)
    (hf : ∀ ⦃x y⦄, x ∈ s → y ∈ s → (f x ≤ f y ↔ x ≤ y)) :
    Maximal (· ∈ f '' s) (f a) ↔ Maximal (· ∈ s) a :=
  minimal_mem_image_monotone_iff (α := αᵒᵈ) (β := βᵒᵈ) (s := s) ha fun _ _ hx hy ↦ hf hy hx


theorem minimal_mem_image_antitone (hf : ∀ ⦃x y⦄, x ∈ s → y ∈ s → (f x ≤ f y ↔ y ≤ x))
    (hx : Minimal (· ∈ s) x) : Maximal (· ∈ f '' s) (f x) :=
  minimal_mem_image_monotone (β := βᵒᵈ) (fun _ _ h h' ↦ hf h' h) hx


theorem maximal_mem_image_antitone (hf : ∀ ⦃x y⦄, x ∈ s → y ∈ s → (f x ≤ f y ↔ y ≤ x))
    (hx : Maximal (· ∈ s) x) : Minimal (· ∈ f '' s) (f x) :=
  maximal_mem_image_monotone (β := βᵒᵈ) (fun _ _ h h' ↦ hf h' h) hx


theorem minimal_mem_image_antitone_iff (ha : a ∈ s)
    (hf : ∀ ⦃x y⦄, x ∈ s → y ∈ s → (f x ≤ f y ↔ y ≤ x)) :
    Minimal (· ∈ f '' s) (f a) ↔ Maximal (· ∈ s) a :=
  maximal_mem_image_monotone_iff (β := βᵒᵈ) ha (fun _ _ h h' ↦ hf h' h)


theorem maximal_mem_image_antitone_iff (ha : a ∈ s)
    (hf : ∀ ⦃x y⦄, x ∈ s → y ∈ s → (f x ≤ f y ↔ y ≤ x)) :
    Maximal (· ∈ f '' s) (f a) ↔ Minimal (· ∈ s) a :=
  minimal_mem_image_monotone_iff (β := βᵒᵈ) ha (fun _ _ h h' ↦ hf h' h)


theorem image_monotone_setOf_minimal (hf : ∀ ⦃x y⦄, P x → P y → (f x ≤ f y ↔ x ≤ y)) :
    f '' {x | Minimal P x} = {x | Minimal (∃ x₀, P x₀ ∧ f x₀ = ·) x} := by
  /-
    α : Type u_1
    P : α → Prop
    inst✝¹ : Preorder α
    β : Type u_2
    inst✝ : Preorder β
    f : α → β
    hf : ∀ ⦃x y : α⦄, P x → P y → Iff (LE.le (f x) (f y)) (LE.le x y)
    ⊢ Eq (Set.image f (setOf fun x => Minimal P x)) (setOf fun x => Minimal (fun x …
  -/
  refine Set.ext fun x ↦ ⟨?_, fun h ↦ ?_⟩
    /-
      case refine_1
      α : Type u_1
      P : α → Prop
      inst✝¹ : Preorder α
      β : Type u_2
      inst✝ : Preorder β
      f : α → β
      hf : ∀ ⦃x y : α⦄, P x → P y → Iff (LE.le (f x) (f y)) (LE.le x y)
      x : β
      ⊢ Membership.mem (Set.image f (setOf fun x => Minimal P x)) x → Membership.mem …
    -/
  · rintro ⟨x, (hx : Minimal _ x), rfl⟩
    /-
      case refine_1.intro.intro
      α : Type u_1
      P : α → Prop
      inst✝¹ : Preorder α
      β : Type u_2
      inst✝ : Preorder β
      f : α → β
      hf : ∀ ⦃x y : α⦄, P x → P y → Iff (LE.le (f x) (f y)) (LE.le x y)
      x : α
      hx : Minimal P x
      ⊢ Membership.mem (setOf fun x => Minimal (fun x => Exists fun x₀ => And (P x₀) …
    -/
    exact (minimal_mem_image_monotone_iff hx.prop hf).2 hx
    /-
      🎉 no goals
    -/
  /-
    case refine_2
    α : Type u_1
    P : α → Prop
    inst✝¹ : Preorder α
    β : Type u_2
    inst✝ : Preorder β
    f : α → β
    hf : ∀ ⦃x y : α⦄, P x → P y → Iff (LE.le (f x) (f y)) (LE.le x y)
    x : β
    h : Membership.mem (setOf fun x => Minimal (fun x => Exists fun x₀ => And (P x …
    ⊢ Membership.mem (Set.image f (setOf fun x => Minimal P x)) x
  -/
  obtain ⟨y, hy, rfl⟩ := (mem_setOf_eq ▸ h).prop
  /-
    case refine_2.intro.intro
    α : Type u_1
    P : α → Prop
    inst✝¹ : Preorder α
    β : Type u_2
    inst✝ : Preorder β
    f : α → β
    hf : ∀ ⦃x y : α⦄, P x → P y → Iff (LE.le (f x) (f y)) (LE.le x y)
    y : α
    hy : P y
    h : Membership.mem (setOf fun x => Minimal (fun x => Exists fun x₀ => And (P x …
    ⊢ Membership.mem (Set.image f (setOf fun x => Minimal P x)) (f y)
  -/
  exact mem_image_of_mem _ <| (minimal_mem_image_monotone_iff (s := setOf P) hy hf).1 h
  /-
    🎉 no goals
  -/


theorem image_monotone_setOf_maximal (hf : ∀ ⦃x y⦄, P x → P y → (f x ≤ f y ↔ x ≤ y)) :
    f '' {x | Maximal P x} = {x | Maximal (∃ x₀, P x₀ ∧ f x₀ = ·) x} :=
  image_monotone_setOf_minimal (α := αᵒᵈ) (β := βᵒᵈ) (fun _ _ hx hy ↦ hf hy hx)


theorem image_antitone_setOf_minimal (hf : ∀ ⦃x y⦄, P x → P y → (f x ≤ f y ↔ y ≤ x)) :
    f '' {x | Minimal P x} = {x | Maximal (∃ x₀, P x₀ ∧ f x₀ = ·) x} :=
  image_monotone_setOf_minimal (β := βᵒᵈ) (fun _ _ hx hy ↦ hf hy hx)


theorem image_antitone_setOf_maximal (hf : ∀ ⦃x y⦄, P x → P y → (f x ≤ f y ↔ y ≤ x)) :
    f '' {x | Maximal P x} = {x | Minimal (∃ x₀, P x₀ ∧ f x₀ = ·) x} :=
  image_monotone_setOf_maximal (β := βᵒᵈ) (fun _ _ hx hy ↦ hf hy hx)


theorem image_monotone_setOf_minimal_mem (hf : ∀ ⦃x y⦄, x ∈ s → y ∈ s → (f x ≤ f y ↔ x ≤ y)) :
    f '' {x | Minimal (· ∈ s) x} = {x | Minimal (· ∈ f '' s) x} :=
  image_monotone_setOf_minimal hf


theorem image_monotone_setOf_maximal_mem (hf : ∀ ⦃x y⦄, x ∈ s → y ∈ s → (f x ≤ f y ↔ x ≤ y)) :
    f '' {x | Maximal (· ∈ s) x} = {x | Maximal (· ∈ f '' s) x} :=
  image_monotone_setOf_maximal hf


theorem image_antitone_setOf_minimal_mem (hf : ∀ ⦃x y⦄, x ∈ s → y ∈ s → (f x ≤ f y ↔ y ≤ x)) :
    f '' {x | Minimal (· ∈ s) x} = {x | Maximal (· ∈ f '' s) x} :=
  image_antitone_setOf_minimal hf


theorem image_antitone_setOf_maximal_mem (hf : ∀ ⦃x y⦄, x ∈ s → y ∈ s → (f x ≤ f y ↔ y ≤ x)) :
    f '' {x | Maximal (· ∈ s) x} = {x | Minimal (· ∈ f '' s) x} :=
  image_antitone_setOf_maximal hf


theorem minimal_mem_image (f : α ↪o β) (hx : Minimal (· ∈ s) x) : Minimal (· ∈ f '' s) (f x) :=
                                        /-
                                          α : Type u_1
                                          x : α
                                          inst✝¹ : Preorder α
                                          β : Type u_2
                                          inst✝ : Preorder β
                                          s : Set α
                                          f : OrderEmbedding α β
                                          hx : Minimal (fun x => Membership.mem s x) x
                                          ⊢ ∀ ⦃x y : α⦄, Membership.mem s x → Membership.mem s y → Iff (LE.le (f x) (f y …
                                        -/
  _root_.minimal_mem_image_monotone (by simp [f.le_iff_le]) hx
                                        /-
                                          🎉 no goals
                                        -/


theorem maximal_mem_image (f : α ↪o β) (hx : Maximal (· ∈ s) x) : Maximal (· ∈ f '' s) (f x) :=
                                        /-
                                          α : Type u_1
                                          x : α
                                          inst✝¹ : Preorder α
                                          β : Type u_2
                                          inst✝ : Preorder β
                                          s : Set α
                                          f : OrderEmbedding α β
                                          hx : Maximal (fun x => Membership.mem s x) x
                                          ⊢ ∀ ⦃x y : α⦄, Membership.mem s x → Membership.mem s y → Iff (LE.le (f x) (f y …
                                        -/
  _root_.maximal_mem_image_monotone (by simp [f.le_iff_le]) hx
                                        /-
                                          🎉 no goals
                                        -/


theorem minimal_mem_image_iff (ha : a ∈ s) : Minimal (· ∈ f '' s) (f a) ↔ Minimal (· ∈ s) a :=
                                               /-
                                                 α : Type u_1
                                                 a : α
                                                 inst✝¹ : Preorder α
                                                 β : Type u_2
                                                 inst✝ : Preorder β
                                                 s : Set α
                                                 f : OrderEmbedding α β
                                                 ha : Membership.mem s a
                                                 ⊢ ∀ ⦃x y : α⦄, Membership.mem s x → Membership.mem s y → Iff (LE.le (f x) (f y …
                                               -/
  _root_.minimal_mem_image_monotone_iff ha (by simp [f.le_iff_le])
                                               /-
                                                 🎉 no goals
                                               -/


theorem maximal_mem_image_iff (ha : a ∈ s) : Maximal (· ∈ f '' s) (f a) ↔ Maximal (· ∈ s) a :=
                                               /-
                                                 α : Type u_1
                                                 a : α
                                                 inst✝¹ : Preorder α
                                                 β : Type u_2
                                                 inst✝ : Preorder β
                                                 s : Set α
                                                 f : OrderEmbedding α β
                                                 ha : Membership.mem s a
                                                 ⊢ ∀ ⦃x y : α⦄, Membership.mem s x → Membership.mem s y → Iff (LE.le (f x) (f y …
                                               -/
  _root_.maximal_mem_image_monotone_iff ha (by simp [f.le_iff_le])
                                               /-
                                                 🎉 no goals
                                               -/


theorem minimal_apply_mem_inter_range_iff :
    Minimal (· ∈ t ∩ range f) (f x) ↔ Minimal (fun x ↦ f x ∈ t) x := by
  /-
    α : Type u_1
    x : α
    inst✝¹ : Preorder α
    β : Type u_2
    inst✝ : Preorder β
    f : OrderEmbedding α β
    t : Set β
    ⊢ Iff (Minimal (fun x => Membership.mem (Inter.inter t (Set.range ⇑f)) x) (f x …
  -/
  refine ⟨fun h ↦ ⟨h.prop.1, fun y hy ↦ ?_⟩, fun h ↦ ⟨⟨h.prop, by simp⟩, ?_⟩⟩
    /-
      case refine_1
      α : Type u_1
      x : α
      inst✝¹ : Preorder α
      β : Type u_2
      inst✝ : Preorder β
      f : OrderEmbedding α β
      t : Set β
      h : Minimal (fun x => Membership.mem (Inter.inter t (Set.range ⇑f)) x) (f x)
      y : α
      hy : (fun x => Membership.mem t (f x)) y
      ⊢ LE.le y x → LE.le x y
    -/
  · rw [← f.le_iff_le, ← f.le_iff_le]
    /-
      case refine_1
      α : Type u_1
      x : α
      inst✝¹ : Preorder α
      β : Type u_2
      inst✝ : Preorder β
      f : OrderEmbedding α β
      t : Set β
      h : Minimal (fun x => Membership.mem (Inter.inter t (Set.range ⇑f)) x) (f x)
      y : α
      hy : (fun x => Membership.mem t (f x)) y
      ⊢ LE.le (f y) (f x) → LE.le (f x) (f y)
    -/
    exact h.le_of_le ⟨hy, by simp⟩
    /-
      🎉 no goals
    -/
  /-
    case refine_2
    α : Type u_1
    x : α
    inst✝¹ : Preorder α
    β : Type u_2
    inst✝ : Preorder β
    f : OrderEmbedding α β
    t : Set β
    h : Minimal (fun x => Membership.mem t (f x)) x
    ⊢ ∀ ⦃y : β⦄, (fun x => Membership.mem (Inter.inter t (Set.range ⇑f)) x) y → LE …
  -/
  rintro _ ⟨hyt, ⟨y, rfl⟩⟩
  /-
    case refine_2.intro.intro
    α : Type u_1
    x : α
    inst✝¹ : Preorder α
    β : Type u_2
    inst✝ : Preorder β
    f : OrderEmbedding α β
    t : Set β
    h : Minimal (fun x => Membership.mem t (f x)) x
    y : α
    hyt : Membership.mem t (f y)
    ⊢ LE.le (f y) (f x) → LE.le (f x) (f y)
  -/
  simp_rw [f.le_iff_le]
  /-
    case refine_2.intro.intro
    α : Type u_1
    x : α
    inst✝¹ : Preorder α
    β : Type u_2
    inst✝ : Preorder β
    f : OrderEmbedding α β
    t : Set β
    h : Minimal (fun x => Membership.mem t (f x)) x
    y : α
    hyt : Membership.mem t (f y)
    ⊢ LE.le y x → LE.le x y
  -/
  exact h.le_of_le hyt
  /-
    🎉 no goals
  -/


theorem maximal_apply_mem_inter_range_iff :
    Maximal (· ∈ t ∩ range f) (f x) ↔ Maximal (fun x ↦ f x ∈ t) x :=
  f.dual.minimal_apply_mem_inter_range_iff


theorem minimal_apply_mem_iff (ht : t ⊆ Set.range f) :
    Minimal (· ∈ t) (f x) ↔ Minimal (fun x ↦ f x ∈ t) x := by
  /-
    α : Type u_1
    x : α
    inst✝¹ : Preorder α
    β : Type u_2
    inst✝ : Preorder β
    f : OrderEmbedding α β
    t : Set β
    ht : HasSubset.Subset t (Set.range ⇑f)
    ⊢ Iff (Minimal (fun x => Membership.mem t x) (f x)) (Minimal (fun x => Members …
  -/
  rw [← f.minimal_apply_mem_inter_range_iff, inter_eq_self_of_subset_left ht]
  /-
    🎉 no goals
  -/


theorem maximal_apply_iff (ht : t ⊆ Set.range f) :
    Maximal (· ∈ t) (f x) ↔ Maximal (fun x ↦ f x ∈ t) x :=
  f.dual.minimal_apply_mem_iff ht


@[simp] theorem image_setOf_minimal : f '' {x | Minimal (· ∈ s) x} = {x | Minimal (· ∈ f '' s) x} :=
                                          /-
                                            α : Type u_1
                                            inst✝¹ : Preorder α
                                            β : Type u_2
                                            inst✝ : Preorder β
                                            s : Set α
                                            f : OrderEmbedding α β
                                            ⊢ ∀ ⦃x y : α⦄, Membership.mem s x → Membership.mem s y → Iff (LE.le (f x) (f y …
                                          -/
  _root_.image_monotone_setOf_minimal (by simp [f.le_iff_le])
                                          /-
                                            🎉 no goals
                                          -/


@[simp] theorem image_setOf_maximal : f '' {x | Maximal (· ∈ s) x} = {x | Maximal (· ∈ f '' s) x} :=
                                          /-
                                            α : Type u_1
                                            inst✝¹ : Preorder α
                                            β : Type u_2
                                            inst✝ : Preorder β
                                            s : Set α
                                            f : OrderEmbedding α β
                                            ⊢ ∀ ⦃x y : α⦄, Membership.mem s x → Membership.mem s y → Iff (LE.le (f x) (f y …
                                          -/
  _root_.image_monotone_setOf_maximal (by simp [f.le_iff_le])
                                          /-
                                            🎉 no goals
                                          -/


theorem inter_preimage_setOf_minimal_eq_of_subset (hts : t ⊆ f '' s) :
    x ∈ s ∩ f ⁻¹' {y | Minimal (· ∈ t) y} ↔ Minimal (· ∈ s ∩ f ⁻¹' t) x := by
  simp_rw [mem_inter_iff, preimage_setOf_eq, mem_setOf_eq, mem_preimage,
    f.minimal_apply_mem_iff (hts.trans (image_subset_range _ _)),
    minimal_and_iff_left_of_imp (fun _ hx ↦ f.injective.mem_set_image.1 <| hts hx)]


theorem inter_preimage_setOf_maximal_eq_of_subset (hts : t ⊆ f '' s) :
    x ∈ s ∩ f ⁻¹' {y | Maximal (· ∈ t) y} ↔ Maximal (· ∈ s ∩ f ⁻¹' t) x :=
  f.dual.inter_preimage_setOf_minimal_eq_of_subset hts


theorem image_setOf_minimal (f : α ≃o β) (P : α → Prop) :
    f '' {x | Minimal P x} = {x | Minimal (fun x ↦ P (f.symm x)) x} := by
  /-
    α : Type u_1
    inst✝¹ : Preorder α
    β : Type u_2
    inst✝ : Preorder β
    f : OrderIso α β
    P : α → Prop
    ⊢ Eq (Set.image (⇑f) (setOf fun x => Minimal P x)) (setOf fun x => Minimal (fu …
  -/
  convert _root_.image_monotone_setOf_minimal (f := f) (by simp [f.le_iff_le])
  /-
    case h.e'_3.h.e'_2.h.h.e'_3.h.a
    α : Type u_1
    inst✝¹ : Preorder α
    β : Type u_2
    inst✝ : Preorder β
    f : OrderIso α β
    P : α → Prop
    x✝¹ x✝ : β
    ⊢ Iff (P (f.symm x✝)) (Exists fun x₀ => And (P x₀) (Eq (f x₀) x✝))
  -/
  aesop
  /-
    🎉 no goals
  -/


theorem image_setOf_maximal (f : α ≃o β) (P : α → Prop) :
    f '' {x | Maximal P x} = {x | Maximal (fun x ↦ P (f.symm x)) x} := by
  /-
    α : Type u_1
    inst✝¹ : Preorder α
    β : Type u_2
    inst✝ : Preorder β
    f : OrderIso α β
    P : α → Prop
    ⊢ Eq (Set.image (⇑f) (setOf fun x => Maximal P x)) (setOf fun x => Maximal (fu …
  -/
  convert _root_.image_monotone_setOf_maximal (f := f) (by simp [f.le_iff_le])
  /-
    case h.e'_3.h.e'_2.h.h.e'_3.h.a
    α : Type u_1
    inst✝¹ : Preorder α
    β : Type u_2
    inst✝ : Preorder β
    f : OrderIso α β
    P : α → Prop
    x✝¹ x✝ : β
    ⊢ Iff (P (f.symm x✝)) (Exists fun x₀ => And (P x₀) (Eq (f x₀) x✝))
  -/
  aesop
  /-
    🎉 no goals
  -/


theorem map_minimal_mem (f : s ≃o t) (hx : Minimal (· ∈ s) x) :
    Minimal (· ∈ t) (f ⟨x, hx.prop⟩) := by
  simpa only [show t = range (Subtype.val ∘ f) by simp, mem_univ, minimal_true_subtype, hx,
    true_imp_iff, image_univ] using OrderEmbedding.minimal_mem_image
    (f.toOrderEmbedding.trans (OrderEmbedding.subtype t)) (s := univ) (x := ⟨x, hx.prop⟩)


theorem map_maximal_mem (f : s ≃o t) (hx : Maximal (· ∈ s) x) :
    Maximal (· ∈ t) (f ⟨x, hx.prop⟩) := by
  simpa only [show t = range (Subtype.val ∘ f) by simp, mem_univ, maximal_true_subtype, hx,
    true_imp_iff, image_univ] using OrderEmbedding.maximal_mem_image
    (f.toOrderEmbedding.trans (OrderEmbedding.subtype t)) (s := univ) (x := ⟨x, hx.prop⟩)


/-- If two sets are order isomorphic, their minimals are also order isomorphic. -/
def mapSetOfMinimal (f : s ≃o t) : {x | Minimal (· ∈ s) x} ≃o {x | Minimal (· ∈ t) x} where
  toFun x := ⟨f ⟨x, x.2.1⟩, f.map_minimal_mem x.2⟩
  invFun x := ⟨f.symm ⟨x, x.2.1⟩, f.symm.map_minimal_mem x.2⟩
  left_inv x := Subtype.ext (congr_arg Subtype.val <| f.left_inv ⟨x, x.2.1⟩ :)
  right_inv x := Subtype.ext (congr_arg Subtype.val <| f.right_inv ⟨x, x.2.1⟩ :)
  map_rel_iff' := f.map_rel_iff


/-- If two sets are order isomorphic, their maximals are also order isomorphic. -/
def mapSetOfMaximal (f : s ≃o t) : {x | Maximal (· ∈ s) x} ≃o {x | Maximal (· ∈ t) x} where
  toFun x := ⟨f ⟨x, x.2.1⟩, f.map_maximal_mem x.2⟩
  invFun x := ⟨f.symm ⟨x, x.2.1⟩, f.symm.map_maximal_mem x.2⟩
  left_inv x := Subtype.ext (congr_arg Subtype.val <| f.left_inv ⟨x, x.2.1⟩ :)
  right_inv x := Subtype.ext (congr_arg Subtype.val <| f.right_inv ⟨x, x.2.1⟩ :)
  map_rel_iff' := f.map_rel_iff


/-- If two sets are antitonically order isomorphic, their minimals/maximals are too. -/
def setOfMinimalIsoSetOfMaximal (f : s ≃o tᵒᵈ) :
    {x | Minimal (· ∈ s) x} ≃o {x | Maximal (· ∈ t) (ofDual x)} where
      toFun x := ⟨(f ⟨x.1, x.2.1⟩).1, ((show s ≃o ofDual ⁻¹' t from f).mapSetOfMinimal x).2⟩
      invFun x := ⟨(f.symm ⟨x.1, x.2.1⟩).1,
        ((show ofDual ⁻¹' t ≃o s from f.symm).mapSetOfMinimal x).2⟩
      __ := (show s ≃o ofDual⁻¹' t from f).mapSetOfMinimal


/-- If two sets are antitonically order isomorphic, their maximals/minimals are too. -/
def setOfMaximalIsoSetOfMinimal (f : s ≃o tᵒᵈ) :
    {x | Maximal (· ∈ s) x} ≃o {x | Minimal (· ∈ t) (ofDual x)} where
  toFun x := ⟨(f ⟨x.1, x.2.1⟩).1, ((show s ≃o ofDual ⁻¹' t from f).mapSetOfMaximal x).2⟩
  invFun x := ⟨(f.symm ⟨x.1, x.2.1⟩).1,
        ((show ofDual ⁻¹' t ≃o s from f.symm).mapSetOfMaximal x).2⟩
  __ := (show s ≃o ofDual⁻¹' t from f).mapSetOfMaximal


theorem minimal_mem_Icc (hab : a ≤ b) : Minimal (· ∈ Icc a b) x ↔ x = a :=
  minimal_iff_eq ⟨rfl.le, hab⟩ (fun _ ↦ And.left)


theorem maximal_mem_Icc (hab : a ≤ b) : Maximal (· ∈ Icc a b) x ↔ x = b :=
  maximal_iff_eq ⟨hab, rfl.le⟩ (fun _ ↦ And.right)


theorem minimal_mem_Ico (hab : a < b) : Minimal (· ∈ Ico a b) x ↔ x = a :=
  minimal_iff_eq ⟨rfl.le, hab⟩ (fun _ ↦ And.left)


theorem maximal_mem_Ioc (hab : a < b) : Maximal (· ∈ Ioc a b) x ↔ x = b :=
  maximal_iff_eq ⟨hab, rfl.le⟩ (fun _ ↦ And.right)

/- Note : The one-sided interval versions of these lemmas are unnecessary,
since `simp` handles them with `maximal_le_iff` and `minimal_ge_iff`. -/


